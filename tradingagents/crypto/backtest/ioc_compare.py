"""Research-only, evidence-gated Market IOC vs signal-anchored Limit IOC replay.

Requires actual timestamped strategy signals (not generated from candle closes)
and Bybit linear L2 archive files. Does not send orders. An execution price
from the book is a model, not an exchange-confirmed fill.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    signal_ts_ms: int
    side: str
    mark: float
    quantity: float
    tick_size: float
    expires_ms: int
    exit_price: float | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Signal:
        ts = int(row["signal_ts_ms"])
        side = row["side"].lower()
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        mark = float(row["signal_mark"])
        quantity = float(row["quantity"])
        tick = float(row["tick_size"])
        expiry = int(row.get("signal_expires_ms") or (ts // 60000 + 1) * 60000)
        exit_price = float(row["exit_price"]) if row.get("exit_price") else None
        if (
            not all(math.isfinite(n) and n > 0 for n in (mark, quantity, tick))
            or expiry <= ts
            or (exit_price is not None and (not math.isfinite(exit_price) or exit_price <= 0))
        ):
            raise ValueError("invalid price, quantity or signal expiry")
        return cls(
            str(row["signal_id"]), str(row["symbol"]).upper(), ts, side,
            mark, quantity, tick, expiry, exit_price,
        )


def load_signals(path: Path) -> list[Signal]:
    with path.open(newline="", encoding="utf-8") as handle:
        signals = [Signal.from_row(row) for row in csv.DictReader(handle)]
    identifiers = [signal.signal_id for signal in signals]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("signal_id must be unique")
    return sorted(signals, key=lambda row: row.signal_ts_ms)


def cap_price(mark: float, tick: float, side: str, bps: float) -> float:
    if not math.isfinite(bps) or bps < 0:
        raise ValueError("bps must be finite and nonnegative")
    direction = 1 if side == "long" else -1
    raw = Decimal(str(mark)) * (1 + Decimal(direction) * Decimal(str(bps)) / 10000)
    rounding = ROUND_FLOOR if direction > 0 else ROUND_CEILING
    return float(
        (raw / Decimal(str(tick))).to_integral_value(rounding=rounding)
        * Decimal(str(tick))
    )


def walk_book(
    levels: list[tuple[float, float]],
    side: str,
    requested: float,
    max_price: float | None,
) -> dict:
    """Walk visible ask/bid depth; partial IOC remainder cancels immediately."""
    if side not in {"long", "short"} or requested <= 0:
        raise ValueError("invalid side or size")
    ordered = sorted(levels, key=lambda level: level[0], reverse=side == "short")
    filled = 0.0
    notional = 0.0
    for price, available in ordered:
        if price <= 0 or available <= 0:
            continue
        if max_price is not None and (
            (side == "long" and price > max_price)
            or (side == "short" and price < max_price)
        ):
            break
        take = min(available, requested - filled)
        filled += take
        notional += take * price
        if filled >= requested - 1e-12:
            break
    return {
        "requested_quantity": requested,
        "filled_quantity": min(filled, requested),
        "fill_fraction": min(filled / requested, 1.0),
        "average_price": notional / filled if filled > 0 else None,
    }


def _update_side(book: dict[float, float], levels: list[list[str]]) -> None:
    for raw_price, raw_qty in levels:
        price = float(raw_price)
        quantity = float(raw_qty)
        if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity) or quantity < 0:
            raise ValueError("invalid Bybit L2 level")
        if quantity == 0:
            book.pop(price, None)
        else:
            book[price] = quantity


def replay_archive(
    archive: Path,
    signals: list[Signal],
    *,
    latency_ms: int = 250,
    max_book_delay_ms: int = 1000,
    limit_bps: float = 10,
    market_cap_bps: float | None = None,
    taker_fee_rate: float = 0.00055,
) -> list[dict]:
    """Sample the first book event at/after modeled send; never use an earlier event."""
    if latency_ms < 0 or max_book_delay_ms < 0 or taker_fee_rate < 0:
        raise ValueError("invalid replay settings")
    if market_cap_bps is not None and market_cap_bps < 0:
        raise ValueError("invalid market cap")
    if not signals:
        return []
    symbol = signals[0].symbol
    if any(row.symbol != symbol for row in signals):
        raise ValueError("one symbol per archive")
    pending = sorted(signals, key=lambda row: row.signal_ts_ms)
    results: list[dict] = []
    index = 0
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    initialized = False
    last_ts = -1
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".data")]
        if len(names) != 1:
            raise ValueError("expected exactly one .data stream in ZIP")
        with bundle.open(names[0]) as stream:
            for raw in stream:
                event = json.loads(raw)
                if str(event.get("data", {}).get("s", "")) != symbol:
                    continue
                ts = int(event["ts"])
                if ts < last_ts:
                    raise ValueError("orderbook events must be time ordered")
                last_ts = ts
                event_type = event.get("type")
                data = event["data"]
                if event_type == "snapshot":
                    bids.clear()
                    asks.clear()
                    initialized = True
                elif event_type != "delta" or not initialized:
                    continue
                _update_side(bids, data.get("b") or [])
                _update_side(asks, data.get("a") or [])
                while index < len(pending) and ts >= pending[index].signal_ts_ms + latency_ms:
                    signal = pending[index]
                    index += 1
                    send_ms = signal.signal_ts_ms + latency_ms
                    if (
                        ts - send_ms > max_book_delay_ms
                        or ts >= signal.expires_ms
                        or not bids
                        or not asks
                        or max(bids) >= min(asks)
                    ):
                        results.append({
                            "signal_id": signal.signal_id,
                            "symbol": symbol,
                            "status": "unreplayable_book_or_expired",
                        })
                        continue
                    cap = cap_price(signal.mark, signal.tick_size, signal.side, limit_bps)
                    market_cap = (
                        cap_price(signal.mark, signal.tick_size, signal.side, market_cap_bps)
                        if market_cap_bps is not None else None
                    )
                    levels = list(asks.items()) if signal.side == "long" else list(bids.items())
                    both = {}
                    for name, price_limit in (("market_ioc", market_cap), ("limit_ioc", cap)):
                        entry = walk_book(levels, signal.side, signal.quantity, price_limit)
                        avg = entry["average_price"]
                        qty = entry["filled_quantity"]
                        entry["adverse_slippage_bps"] = (
                            (1 if signal.side == "long" else -1)
                            * (avg / signal.mark - 1) * 10000 if avg is not None else None
                        )
                        entry["entry_fees_usdt"] = (
                            qty * avg * taker_fee_rate if avg is not None else 0.0
                        )
                        # Same fixed exit price is conditional P&L, NOT a replay of TP/SL.
                        entry["conditional_net_pnl_usdt"] = (
                            (1 if signal.side == "long" else -1)
                            * (signal.exit_price - avg) * qty
                            - entry["entry_fees_usdt"]
                            - qty * signal.exit_price * taker_fee_rate
                            if avg is not None and signal.exit_price is not None else None
                        )
                        both[name] = entry
                    results.append({
                        "signal_id": signal.signal_id,
                        "symbol": symbol,
                        "status": "modeled",
                        "book_ts_ms": ts,
                        "signal_ts_ms": signal.signal_ts_ms,
                        "side": signal.side,
                        "entry_cap_price": cap,
                        **both,
                    })
    results.extend({
        "signal_id": signal.signal_id,
        "symbol": symbol,
        "status": "missing_orderbook",
    } for signal in pending[index:])
    return results


def summarize(results: list[dict]) -> dict:
    modeled = [item for item in results if item["status"] == "modeled"]
    output: dict = {
        "signals": len(results),
        "modeled_signals": len(modeled),
        "excluded": len(results) - len(modeled),
        "warning": (
            "Book-walk execution estimates only; no queue dynamics, venue market"
            " cap, actual signal feed or independent exits verified by this tool."
        ),
    }
    for mode in ("market_ioc", "limit_ioc"):
        fills = [item[mode] for item in modeled]
        with_fill = [item for item in fills if item["filled_quantity"] > 0]
        pnl = [item["conditional_net_pnl_usdt"] for item in fills
               if item["conditional_net_pnl_usdt"] is not None]
        output[mode] = {
            "any_fill_rate": len(with_fill) / len(fills) if fills else None,
            "full_fill_rate": (
                sum(item["fill_fraction"] >= 1 - 1e-9 for item in fills) / len(fills)
                if fills else None
            ),
            "quantity_fill_rate": (
                sum(item["filled_quantity"] for item in fills)
                / sum(item["requested_quantity"] for item in fills)
                if fills else None
            ),
            "mean_adverse_slippage_bps": (
                sum(item["adverse_slippage_bps"] for item in with_fill) / len(with_fill)
                if with_fill else None
            ),
            "entry_fees_usdt": sum(item["entry_fees_usdt"] for item in fills),
            "conditional_net_pnl_usdt": sum(pnl) if len(pnl) == len(fills) and pnl else None,
            "conditional_pnl_samples": len(pnl),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latency-ms", type=int, default=250)
    parser.add_argument("--max-book-delay-ms", type=int, default=1000)
    parser.add_argument("--market-cap-bps", type=float)
    parser.add_argument("--fee-per-fill", type=float, default=0.00055)
    args = parser.parse_args()
    signals = load_signals(args.signals)
    if not signals:
        raise SystemExit("No recorded signals; cannot invent historical fills")
    all_results: list[dict] = []
    for archive in args.archive:
        for symbol in {row.symbol for row in signals}:
            date_prefix = archive.name[:10]
            archive_symbol = archive.name.split("_")[1] if "_" in archive.name else ""
            if archive_symbol != symbol:
                continue
            group = [
                row for row in signals
                if row.symbol == symbol
                and datetime.fromtimestamp(row.signal_ts_ms / 1000, UTC).date().isoformat()
                == date_prefix
            ]
            all_results.extend(replay_archive(
                archive, group, latency_ms=args.latency_ms,
                max_book_delay_ms=args.max_book_delay_ms,
                market_cap_bps=args.market_cap_bps,
                taker_fee_rate=args.fee_per_fill,
            ))
    if not all_results:
        raise SystemExit("No signals matched the supplied archives")
    result = {
        "source": "Bybit linear L2 archive plus recorded signal CSV",
        "notional_reference_usdt": 50,
        "model": "timestamped visible-depth book walk; no actual exchange fills",
        "summary": summarize(all_results),
        "trades": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
