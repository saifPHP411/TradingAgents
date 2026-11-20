"""Bounded REAL Bybit L2 prefix, precommitted quote-only IOC shadow comparison.

This is an execution-mechanics experiment, NOT the trading bot's historical
scalper backtest. No archived genuine scalper decision/Mark-price trace exists
for this sample, so we take the first valid quote in fixed 15-second UTC buckets
and alternate sides deterministically. The contemporaneous book *midpoint*
is used as an explicitly unverified Mark proxy. Same pseudo-signals are fed to
the existing Market IOC / 10-bp Limit IOC book-walk under 0/250/1000 ms latency.

Only the beginning of the archive is examined (a bounded HTTP range), not a
full day, so this cannot support a daily fill rate or net trading returns.
No authenticated exchange requests, financial orders, or credentials.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from datetime import UTC, date, datetime
from pathlib import Path

try:
    from .ioc_compare import Signal, _update_side, replay_archive, summarize
except ImportError:
    from ioc_compare import Signal, _update_side, replay_archive, summarize

PREFIX_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED = 96 * 1024 * 1024
MAX_EVENTS = 150_000
SAMPLE_INTERVAL_MS = 15_000
NOTIONAL_USDT = 10.0
LATENCIES_MS = (0, 250, 1000)
RESEARCH_TICKS = {"XRPUSDT": 0.0001, "LTCUSDT": 0.01}
SYMBOLS = tuple(RESEARCH_TICKS)


def zip_prefix_lines(prefix: bytes) -> tuple[list[bytes], bool]:
    """Parse only a bounded ZIP member prefix. Never claim ZIP CRC verified."""
    if len(prefix) < 30 or prefix[:4] != b"PK\x03\x04":
        raise ValueError("archive is not a ZIP local header")
    (_, _, flags, method, _, _, _, _, _, name_len, extra_len) = struct.unpack_from(
        "<IHHHHHIIIHH", prefix
    )
    if flags & 1:
        raise ValueError("encrypted archives are unsupported")
    name = prefix[30:30 + name_len].decode("utf-8", errors="strict")
    if not name.endswith(".data"):
        raise ValueError("expected Bybit .data JSONL member")
    begin = 30 + name_len + extra_len
    if begin >= len(prefix):
        raise ValueError("no compressed ZIP stream in requested range")
    if method == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        decoded = inflater.decompress(prefix[begin:], MAX_DECOMPRESSED + 1)
        is_complete = inflater.eof
    elif method == zipfile.ZIP_STORED:
        decoded = prefix[begin:begin + MAX_DECOMPRESSED + 1]
        is_complete = False
    else:
        raise ValueError("ZIP member compression not supported")
    if len(decoded) > MAX_DECOMPRESSED:
        raise ValueError("bounded ZIP range inflates beyond safety limit")
    if b"\n" not in decoded:
        raise ValueError("no complete JSONL records in archive prefix")
    raw_lines = decoded.split(b"\n")[:-1]
    if len(raw_lines) > MAX_EVENTS:
        raw_lines = raw_lines[:MAX_EVENTS]
        is_complete = False
    lines = [line for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError("no parseable JSONL record")
    return lines, is_complete


def build_quote_proxy_signals(
    lines: list[bytes], *, symbol: str, requested_day: date,
) -> list[Signal]:
    """Take only first available quote per fixed UTC bucket, not best returns."""
    if symbol not in RESEARCH_TICKS:
        raise ValueError("only supported predefined research symbols")
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    initialized = False
    prev_ts = -1
    buckets: set[int] = set()
    selected: list[Signal] = []
    for raw in lines:
        event = json.loads(raw)
        if not isinstance(event, dict) or not isinstance(event.get("data"), dict):
            raise ValueError("invalid L2 JSON record")
        data = event["data"]
        if data.get("s") != symbol:
            raise ValueError("archive symbol mismatch")
        ts = event.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < prev_ts:
            raise ValueError("out-of-order or invalid archive timestamp")
        prev_ts = ts
        if datetime.fromtimestamp(ts / 1000, UTC).date() != requested_day:
            raise ValueError("archive record date mismatch")
        kind = event.get("type")
        if kind == "snapshot":
            bids.clear()
            asks.clear()
            initialized = True
        elif kind != "delta":
            raise ValueError("unexpected L2 event type")
        elif not initialized:
            continue
        if not isinstance(data.get("b"), list) or not isinstance(data.get("a"), list):
            raise ValueError("archive record has no bid/ask arrays")
        _update_side(bids, data["b"])
        _update_side(asks, data["a"])
        if not bids or not asks or max(bids) >= min(asks):
            # A book resync or crossed book cannot be assumed executable.
            continue
        if ts % 60_000 > 60_000 - max(LATENCIES_MS) - 1:
            continue
        bucket = ts // SAMPLE_INTERVAL_MS
        if bucket in buckets:
            continue
        buckets.add(bucket)
        mark_proxy = (max(bids) + min(asks)) / 2
        if not math.isfinite(mark_proxy) or mark_proxy <= 0:
            continue
        side = "long" if bucket % 2 == 0 else "short"
        selected.append(Signal(
            signal_id=f"fixed-proxy:{symbol}:{ts}:{side}",
            symbol=symbol,
            signal_ts_ms=ts,
            side=side,
            mark=mark_proxy,
            quantity=NOTIONAL_USDT / mark_proxy,
            tick_size=RESEARCH_TICKS[symbol],
            expires_ms=(ts // 60_000 + 1) * 60_000,
            exit_price=None,
        ))
    return selected


def compare_prefix(
    prefix: bytes, *, symbol: str, day: date, source_url: str,
    latency_options: tuple[int, ...] = LATENCIES_MS,
) -> dict:
    lines, member_complete = zip_prefix_lines(prefix)
    signals = build_quote_proxy_signals(lines, symbol=symbol, requested_day=day)
    if len(signals) < 3:
        raise ValueError("too few fixed-time quote observations for comparison")
    first = json.loads(lines[0])["ts"]
    last = json.loads(lines[-1])["ts"]
    if last <= first:
        raise ValueError("archive range has no useful temporal coverage")
    # Synthetic container reproduces EXACT observed JSONL events from a bounded
    # prefix. The result is not a validated full-day archive.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{day.isoformat()}_{symbol}_ob200.data.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("replayed-prefix.data", b"\n".join(lines) + b"\n")
        modes = {}
        for latency_ms in latency_options:
            rows = replay_archive(
                path, signals, latency_ms=latency_ms, max_book_delay_ms=1000,
                limit_bps=10, taker_fee_rate=0.00055,
            )
            totals = summarize(rows)
            paired = [r for r in rows if r["status"] == "modeled"]
            both_filled = [
                r for r in paired
                if r["market_ioc"]["filled_quantity"] > 0
                and r["limit_ioc"]["filled_quantity"] > 0
            ]
            totals["paired_both_filled"] = len(both_filled)
            totals["net_strategy_profit_usdt"] = None
            totals["entry_price_mean_difference_bps_when_both_fill"] = (
                sum(
                    (1 if row["side"] == "long" else -1)
                    * (row["market_ioc"]["average_price"]
                       - row["limit_ioc"]["average_price"])
                    / row["limit_ioc"]["average_price"] * 10000
                    for row in both_filled
                ) / len(both_filled) if both_filled else None
            )
            totals["extra_market_filled_quantity"] = sum(
                r["market_ioc"]["filled_quantity"]
                - r["limit_ioc"]["filled_quantity"] for r in paired
            )
            modes[str(latency_ms)] = totals
    return {
        "status": "sampled_quote_only_not_strategy_backtest",
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "symbol": symbol,
        "date": day.isoformat(),
        "archive_url": source_url,
        "http_range_bytes": len(prefix),
        "parsed_complete_jsonl_records": len(lines),
        "first_book_ts_ms": first,
        "last_book_ts_ms": last,
        "sampled_duration_seconds": (last - first) / 1000,
        "sampled_quote_proxy_signals": len(signals),
        "full_day_archive_verified": False,
        "member_deflate_stream_complete": member_complete,
        "signal_provenance": (
            "Predeclared alternating sides at the first valid book event in "
            "each 15-second UTC bucket; quote midpoint used as a MARK PRICE PROXY, "
            "not an archived scalper decision and not a Bybit Mark price."
        ),
        "notional_per_pseudo_signal_usdt": NOTIONAL_USDT,
        "research_only_unverified_ticks": RESEARCH_TICKS[symbol],
        "venue_qty_steps_verified": False,
        "fee_per_side_estimate": 0.00055,
        "limitations": [
            "Only bounded ZIP beginning was observed, not random full-day timestamps.",
            "No genuine recorded historical live-scalper signals or Bybit Mark prices.",
            "No actual Bybit fills or hidden-liquidity/queue/adverse selection evidence.",
            "Market IOC baseline ignores Bybit's exchange-side protective IOC cap.",
            "No entry-to-exit P&L, funding, liquidation or portfolio returns can be inferred.",
            "Tick, quantity step and min-notional are unverified research assumptions.",
        ],
        "latency_sensitivity_ms": modes,
    }


def fetch_prefix(symbol: str, day: date, *, max_bytes: int = PREFIX_BYTES) -> tuple[bytes, str]:
    if symbol not in SYMBOLS or max_bytes <= 0 or max_bytes > PREFIX_BYTES:
        raise ValueError("unsupported source or range size")
    name = f"{day.isoformat()}_{symbol}_ob200.data.zip"
    url = f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{name}"
    req = urllib.request.Request(url, headers={
        "Range": f"bytes=0-{max_bytes - 1}",
        "User-Agent": "TradingAgents-bounded-readonly-quote-comparison/1.0",
    })
    with urllib.request.urlopen(req, timeout=50) as response:
        if response.status != 206:
            raise ValueError("server ignored bounded HTTP Range requirement")
        value = response.headers.get("Content-Range", "")
        if not value.startswith("bytes 0-"):
            raise ValueError("remote response was not a zero-offset prefix")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("server exceeded bounded archive prefix")
    return data, url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=SYMBOLS, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prefix, url = fetch_prefix(args.symbol, args.date)
        report = compare_prefix(prefix, symbol=args.symbol, day=args.date,
                                source_url=url)
    except Exception as exc:
        report = {
            "status": "source_or_sample_unavailable",
            "symbol": args.symbol,
            "date": args.date.isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "research_only": True,
            "orders_submitted": False,
            "demo_ready": False,
            "net_strategy_profit_usdt": None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "symbol": args.symbol,
        "date": args.date.isoformat(),
        "signals": report.get("sampled_quote_proxy_signals"),
        "latency_sensitivity_ms": report.get("latency_sensitivity_ms"),
        "error": report.get("error"),
    }))
    if report["status"] != "sampled_quote_only_not_strategy_backtest":
        sys.exit(2)


if __name__ == "__main__":
    main()
