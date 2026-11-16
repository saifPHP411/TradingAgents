"""Short live Bybit Spot order-flow probe; research only, never submits orders.

Collects public recent trades, deduplicates trade IDs, builds minute-level
aggressive-buy volume imbalance, and evaluates a fixed next-minute/3-minute
paper return after explicit estimated costs. A short probe is exploratory data,
not evidence of a durable trading edge.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict
from datetime import UTC, datetime

import requests

BASE_URL = "https://api.bybit.com"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")


def _get(session: requests.Session, path: str, params: dict) -> dict:
    response = session.get(f"{BASE_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit public API error: {payload.get('retCode')} {payload.get('retMsg')}")
    return payload["result"]


def collect_trades(symbols: tuple[str, ...], duration_seconds: int, poll_seconds: float = 5.0) -> dict:
    if not symbols or duration_seconds < 60 or poll_seconds < 1:
        raise ValueError("symbols, duration >= 60s, and poll interval >= 1s are required")
    session = requests.Session()
    seen: dict[str, set[str]] = {symbol: set() for symbol in symbols}
    minute_flow: dict[str, dict[int, dict[str, float | int]]] = {
        symbol: defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "trades": 0}) for symbol in symbols
    }
    started_ms = int(time.time() * 1000)
    deadline = time.monotonic() + duration_seconds
    polls = 0
    while time.monotonic() < deadline:
        for symbol in symbols:
            result = _get(session, "/v5/market/recent-trade", {
                "category": "spot", "symbol": symbol, "limit": 1000,
            })
            for trade in result.get("list", []):
                trade_id = str(trade.get("execId") or trade.get("i") or "")
                if not trade_id or trade_id in seen[symbol]:
                    continue
                seen[symbol].add(trade_id)
                timestamp = int(trade["time"])
                if timestamp < started_ms:
                    continue
                bucket = timestamp // 60_000
                quantity = float(trade["size"])
                side = str(trade["side"]).lower()
                key = "buy" if side == "buy" else "sell"
                minute_flow[symbol][bucket][key] += quantity
                minute_flow[symbol][bucket]["trades"] += 1
        polls += 1
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    return {
        "started_at": datetime.fromtimestamp(started_ms / 1000, UTC).isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "polls": polls,
        "symbols": {
            symbol: {str(k): dict(v) for k, v in sorted(rows.items())}
            for symbol, rows in minute_flow.items()
        },
    }


def evaluate_probe(raw: dict, threshold: float = 0.35, min_trades: int = 20,
                   round_trip_cost_bps: float = 26.0,
                   candles_by_symbol: dict | None = None) -> dict:
    """Evaluate fixed +0.35 buy-imbalance signal at next 1m open, exit 3m close."""
    if not 0 < threshold < 1 or min_trades < 1 or round_trip_cost_bps < 0:
        raise ValueError("invalid signal or cost settings")
    session = requests.Session() if candles_by_symbol is None else None
    start_ms = int(datetime.fromisoformat(raw["started_at"]).timestamp() * 1000) - 60_000
    end_ms = int(datetime.fromisoformat(raw["ended_at"]).timestamp() * 1000) + 240_000
    all_trades = []
    per_symbol = {}
    for symbol, minute_rows in raw["symbols"].items():
        if candles_by_symbol is None:
            candles = _get(session, "/v5/market/kline", {
                "category": "spot", "symbol": symbol, "interval": "1",
                "start": start_ms, "end": end_ms, "limit": 1000,
            }).get("list", [])
            bars = {int(row[0]): {"open": float(row[1]), "close": float(row[4])} for row in candles}
        else:
            bars = candles_by_symbol.get(symbol, {})
        symbol_trades = []
        for minute_text, flow in minute_rows.items():
            minute = int(minute_text) * 60_000
            total = float(flow["buy"]) + float(flow["sell"])
            count = int(flow["trades"])
            if total <= 0 or count < min_trades:
                continue
            imbalance = (float(flow["buy"]) - float(flow["sell"])) / total
            entry = bars.get(minute + 60_000)
            exit_bar = bars.get(minute + 180_000)
            if imbalance < threshold or not entry or not exit_bar:
                continue
            gross_pct = (exit_bar["close"] / entry["open"] - 1) * 100
            net_pct = gross_pct - round_trip_cost_bps / 100
            row = {
                "symbol": symbol, "signal_minute_utc_ms": minute,
                "aggressive_buy_imbalance": round(imbalance, 5),
                "trade_count": count, "gross_return_pct": round(gross_pct, 5),
                "net_return_pct": round(net_pct, 5),
            }
            symbol_trades.append(row)
            all_trades.append(row)
        net = [row["net_return_pct"] for row in symbol_trades]
        positive = sum(value for value in net if value > 0)
        negative = -sum(value for value in net if value < 0)
        per_symbol[symbol] = {
            "signals": len(net), "wins": sum(value > 0 for value in net),
            "net_mean_pct": sum(net) / len(net) if net else 0.0,
            "net_profit_factor": positive / negative if negative else ("infinite" if positive else 0.0),
        }
    net = [row["net_return_pct"] for row in all_trades]
    gains = sum(value for value in net if value > 0)
    losses = -sum(value for value in net if value < 0)
    return {
        "study": "bybit_spot_aggressive_buy_imbalance_probe",
        "venue": "Bybit public Spot market data",
        "parameters": {"imbalance_threshold": threshold, "minimum_trades_per_minute": min_trades,
                       "entry": "next 1m candle open", "exit": "3m candle close after signal",
                       "round_trip_cost_bps": round_trip_cost_bps},
        "signals": len(net), "wins": sum(value > 0 for value in net),
        "mean_net_return_pct": sum(net) / len(net) if net else 0.0,
        "aggregate_net_profit_factor": gains / losses if losses else ("infinite" if gains else 0.0),
        "per_symbol": per_symbol, "trades": all_trades,
        "limitations": [
            "Short forward sample is exploratory, not a profitability claim.",
            "REST recent-trade polling can miss prints between polls or beyond the 1000-trade window.",
            "Estimated costs are not the user's measured Bybit fee tier or realized spread/impact.",
            "No order book queue, partial fills, adverse selection, or execution is simulated.",
            "Only positive imbalance is tested; no threshold tuning or selection on this sample.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a bounded Bybit Spot order-flow probe.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/bybit_orderflow_probe.json"))
    args = parser.parse_args()
    raw = collect_trades(tuple(args.symbols), args.duration_seconds)
    report = evaluate_probe(raw)
    report["collection"] = {key: raw[key] for key in ("started_at", "ended_at", "polls")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "signals", "wins", "mean_net_return_pct", "aggregate_net_profit_factor", "per_symbol"
    )}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
