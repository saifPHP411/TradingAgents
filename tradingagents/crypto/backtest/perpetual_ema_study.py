"""Run a read-only Bybit public-data research screen for the EMA perpetual strategy."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from tradingagents.crypto.backtest.perpetual_ema import PerpetualCosts, PerpetualEMAStrategy, run_backtest
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--equity", type=float, default=1_000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=5.5)
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("results/perpetual_ema_research.json"))
    return parser


def run_study(args: argparse.Namespace) -> dict:
    if isinstance(args.days, bool) or not isinstance(args.days, int) or not 35 <= args.days <= 3_650:
        raise ValueError("days must be an integer in [35, 3650]")
    if isinstance(args.equity, bool) or not math.isfinite(args.equity) or args.equity <= 0:
        raise ValueError("equity must be finite and positive")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - args.days * 86_400_000
    client = BybitLinearPublicClient()
    instrument = client.instrument_info(args.symbol)
    traded = client.klines(instrument.symbol, start_ms=start_ms, end_ms=now_ms)
    mark = client.mark_price_klines(instrument.symbol, start_ms=start_ms, end_ms=now_ms)
    funding = client.funding_rates(
        instrument.symbol, start_ms=start_ms, end_ms=now_ms,
        interval_minutes=instrument.funding_interval_minutes,
    )
    risk = PerpetualRiskSettings(leverage=args.leverage, max_leverage=2.0)
    cfg = PerpetualEMAStrategy(enabled=True, risk=risk)
    costs = PerpetualCosts(args.commission_bps, args.half_spread_bps, args.slippage_bps)
    result = run_backtest(traded, mark, funding, instrument, initial_equity=args.equity, strategy=cfg, costs=costs)
    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "credentials_used": False,
        "venue": "Bybit V5 public linear perpetual market data",
        "symbol": instrument.symbol,
        "start": traded.index[0].isoformat(),
        "end": traded.index[-1].isoformat(),
        "traded_candles": len(traded),
        "mark_candles": len(mark),
        "funding_events": len(funding),
        "instrument": asdict(instrument),
        "strategy": asdict(cfg),
        "costs": asdict(costs),
        "metrics": result.metrics(),
        "assumptions": result.assumptions,
        "trades": [trade.to_dict() for trade in result.trades],
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_study(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"symbol": report["symbol"], **report["metrics"], "orders_submitted": False}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
