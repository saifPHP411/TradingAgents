"""Chronological, cost-aware RAE research study over cached 1h candles.

The three chronological partitions are run independently with fresh capital.
No parameter fitting is performed on validation or holdout. This is a research
comparison, not a broker integration or a claim of future profitability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest.history import HistoryCache
from .ensemble_backtest import BacktestSettings, compare_modes
from .symbols import parse_pair


def _partition(frame, left: float, right: float):
    start = int(len(frame) * left)
    end = int(len(frame) * right)
    return frame.iloc[start:end]


def run_ensemble_study(
    *,
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "BNB/USDT"),
    cache_dir: str | None = None,
    initial_cash: float = 10_000.0,
) -> dict:
    cache = HistoryCache(cache_dir) if cache_dir else HistoryCache()
    partitions = {"development": (0.0, 0.60), "validation": (0.60, 0.80), "holdout": (0.80, 1.0)}
    report = {
        "schema_version": 1,
        "kind": "crypto_regime_adaptive_ensemble_study",
        "interval": "1h",
        "partitions": {},
        "cost_stress": "1.5x fee and slippage",
        "important": [
            "Completed 1h candles only.",
            "Signals are generated at bar close and executed at next bar open.",
            "Each symbol and partition is a separate single-symbol account; results are not added together.",
            "Each partition has an indicator warm-up period that is excluded from trading.",
            "No parameter optimization or holdout retuning is performed.",
        ],
    }
    btc_full = cache.load("BTC/USDT", "1h")
    if any(parse_pair(symbol).canonical != "BTC/USDT" for symbol in symbols) and btc_full is None:
        raise RuntimeError("BTC 1h history is required for the fail-closed altcoin risk overlay")
    for partition, (left, right) in partitions.items():
        report["partitions"][partition] = {}
        for raw in symbols:
            symbol = parse_pair(raw).canonical
            frame = cache.load(symbol, "1h")
            if frame is None or frame.empty:
                raise RuntimeError(f"missing cached 1h history for {symbol}")
            section = _partition(frame, left, right)
            if len(section) < 222:
                raise RuntimeError(
                    f"{symbol} {partition} has only {len(section)} bars; at least 222 required"
                )
            btc_section = None
            if symbol != "BTC/USDT" and btc_full is not None:
                btc_section = btc_full.loc[section.index[0] : section.index[-1]]
                if len(btc_section) < 222:
                    raise RuntimeError(f"insufficient aligned BTC filter history for {symbol}")
            base = BacktestSettings(initial_cash=initial_cash)
            stressed = BacktestSettings(
                initial_cash=initial_cash,
                risk_fraction=base.risk_fraction,
                max_notional_fraction=base.max_notional_fraction,
                fee_bps=base.fee_bps * 1.5,
                slippage_bps=base.slippage_bps * 1.5,
            )
            report["partitions"][partition][symbol] = {
                "coverage": {
                    "bars": len(section),
                    "first": section.index[0].isoformat(),
                    "last": section.index[-1].isoformat(),
                },
                "btc_filter": btc_section is not None,
                "base_costs": compare_modes(section, settings=base, btc_frame=btc_section),
                "stress_costs": compare_modes(section, settings=stressed, btc_frame=btc_section),
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline RAE research study.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    parser.add_argument("--cache", default=None)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--out", type=Path, default=Path("rae_study.json"))
    args = parser.parse_args(argv)
    report = run_ensemble_study(
        symbols=tuple(args.symbols), cache_dir=args.cache, initial_cash=args.initial_cash
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
