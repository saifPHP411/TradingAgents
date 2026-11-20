"""Read the start of a public Bybit orderbook ZIP without downloading the full day.

This verifies *format and source reachability*, not trading results, archive
completeness, historical marks or exchange-confirmed fills.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
import zlib
from datetime import UTC, date, datetime
from pathlib import Path

MAX_PREFIX_BYTES = 512 * 1024
SYMBOL_PATTERN = re.compile(r"[A-Z0-9]{3,30}")


def inspect_zip_prefix(prefix: bytes, *, symbol: str, day: date) -> dict:
    """Parse the first member of a ZIP file from a bounded HTTP range response."""
    if len(prefix) < 30 or prefix[:4] != b"PK\x03\x04":
        raise ValueError("response is not a ZIP local-file header")
    (
        _, _, flags, method, _, _, _, _, _, filename_length, extra_length,
    ) = struct.unpack_from("<IHHHHHIIIHH", prefix)
    if flags & 1:
        raise ValueError("encrypted ZIP archive unsupported")
    offset = 30 + filename_length + extra_length
    if offset >= len(prefix):
        raise ValueError("ZIP prefix did not include compressed data")
    filename = prefix[30:30 + filename_length].decode("utf-8", errors="replace")
    if not filename.endswith(".data"):
        raise ValueError("ZIP member is not .data")
    if method == 8:
        decompressor = zlib.decompressobj(-15)
        decoded = decompressor.decompress(prefix[offset:], max_length=2_000_000)
    elif method == 0:
        decoded = prefix[offset:]
    else:
        raise ValueError(f"unsupported ZIP compression method {method}")
    # The last line may be truncated by the bounded HTTP read; discard it.
    complete_lines = decoded.split(b"\n")[:-1]
    if not complete_lines:
        raise ValueError("no complete JSONL record in the ZIP prefix")
    events = [json.loads(line) for line in complete_lines[:100] if line.strip()]
    if not events:
        raise ValueError("no valid JSONL events")
    snapshots = deltas = 0
    previous_ts = -1
    first_ts = None
    bid_levels: dict[float, float] = {}
    ask_levels: dict[float, float] = {}
    observed_spreads_bps: list[float] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("orderbook events must be objects")
        if event.get("type") not in {"snapshot", "delta"}:
            raise ValueError("unexpected orderbook event type")
        data = event.get("data")
        if not isinstance(data, dict) or data.get("s") != symbol:
            raise ValueError("symbol mismatch in orderbook events")
        if not isinstance(data.get("b"), list) or not isinstance(data.get("a"), list):
            raise ValueError("orderbook event lacks bid/ask arrays")
        ts = event.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < previous_ts:
            raise ValueError("invalid or out-of-order orderbook timestamp")
        previous_ts = ts
        first_ts = ts if first_ts is None else first_ts
        if event["type"] == "snapshot":
            snapshots += 1
            bid_levels.clear()
            ask_levels.clear()
        else:
            deltas += 1
        for side_key, levels in (("b", bid_levels), ("a", ask_levels)):
            for price_text, quantity_text in data[side_key]:
                price = float(price_text)
                quantity = float(quantity_text)
                if not (0 < price < float("inf") and 0 <= quantity < float("inf")):
                    raise ValueError("invalid price or quantity in sampled book")
                if quantity == 0:
                    levels.pop(price, None)
                else:
                    levels[price] = quantity
        if bid_levels and ask_levels:
            best_bid, best_ask = max(bid_levels), min(ask_levels)
            if best_bid >= best_ask:
                raise ValueError("crossed or locked sampled orderbook")
            midpoint = (best_bid + best_ask) / 2
            observed_spreads_bps.append((best_ask - best_bid) / midpoint * 10000)
    # Never conflate a validated prefix with a validated full-day archive.
    if first_ts is not None:
        observed_day = datetime.fromtimestamp(first_ts / 1000, UTC).date()
        if observed_day != day:
            raise ValueError("archive content date differs from requested date")
    return {
        "symbol": symbol,
        "requested_day": day.isoformat(),
        "member": filename,
        "sampled_complete_records": len(events),
        "sampled_snapshots": snapshots,
        "sampled_deltas": deltas,
        "sampled_spread_observations": len(observed_spreads_bps),
        "mean_quoted_spread_bps": (
            sum(observed_spreads_bps) / len(observed_spreads_bps)
            if observed_spreads_bps else None
        ),
        "min_quoted_spread_bps": min(observed_spreads_bps) if observed_spreads_bps else None,
        "max_quoted_spread_bps": max(observed_spreads_bps) if observed_spreads_bps else None,
        "first_ts_ms": first_ts,
        "last_sampled_ts_ms": previous_ts,
        "scope": "ZIP prefix only; full-day CRC and completeness NOT verified",
    }


def probe(symbol: str, day: date) -> dict:
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("invalid symbol")
    date_text = day.isoformat()
    url = (
        f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/"
        f"{date_text}_{symbol}_ob200.data.zip"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{MAX_PREFIX_BYTES - 1}",
            "User-Agent": "TradingAgents-readonly-archive-research/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=35) as response:
        status = response.status
        if status not in (200, 206):
            raise ValueError(f"unexpected archive HTTP status: {status}")
        prefix = response.read(MAX_PREFIX_BYTES)
    report = inspect_zip_prefix(prefix, symbol=symbol, day=day)
    report.update(
        url=url,
        http_status=status,
        compressed_prefix_bytes=len(prefix),
        analysis="archive reachability and prefix schema only; NOT a backtest",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("XRPUSDT", "LTCUSDT"), required=True)
    parser.add_argument("--date", default="2026-10-01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = probe(args.symbol, date.fromisoformat(args.date))
        report["status"] = "prefix_verified"
    except Exception as exc:
        report = {
            "status": "unavailable_or_invalid",
            "symbol": args.symbol,
            "date": args.date,
            "error": f"{type(exc).__name__}: {exc}",
            "scope": "No valid source coverage inferred",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "prefix_verified":
        sys.exit(2)


if __name__ == "__main__":
    main()
