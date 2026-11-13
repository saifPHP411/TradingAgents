"""Run frozen RAE chronological OOS validation from the local public-data cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest.history import HistoryCache
from .ensemble_backtest import BacktestSettings
from .ensemble_portfolio import PortfolioSettings
from .ensemble_walkforward import WalkForwardSettings, run_walk_forward
from .symbols import parse_pair


def run_cached_walkforward(
    *,
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "BNB/USDT"),
    cache_dir: str | None = None,
    train_bars: int = 4_000,
    test_bars: int = 720,
    purge_bars: int = 24,
    embargo_bars: int = 24,
    simulations: int = 2_000,
    initial_cash: float = 10_000.0,
) -> dict:
    cache = HistoryCache(cache_dir) if cache_dir else HistoryCache()
    frames = {}
    for raw_symbol in symbols:
        symbol = parse_pair(raw_symbol).canonical
        frame = cache.load(symbol, "1h")
        if frame is None or frame.empty:
            raise RuntimeError(f"missing cached 1h history for {symbol}")
        frames[symbol] = frame
    backtest = BacktestSettings(initial_cash=initial_cash)
    walk_forward = WalkForwardSettings(
        train_bars=train_bars,
        test_bars=test_bars,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        monte_carlo_simulations=simulations,
    )
    modes = ("trend", "trend_pullback", "range", "breakout", "ensemble")
    reports = {}
    for mode in modes:
        base_report = run_walk_forward(
            frames,
            walk_forward_settings=walk_forward,
            backtest_settings=backtest,
            portfolio_settings=PortfolioSettings(),
            mode=mode,
        )
        stress_backtest = BacktestSettings(
            initial_cash=backtest.initial_cash,
            risk_fraction=backtest.risk_fraction,
            max_notional_fraction=backtest.max_notional_fraction,
            fee_bps=backtest.fee_bps * 1.5,
            slippage_bps=backtest.slippage_bps * 1.5,
            warmup_bars=backtest.warmup_bars,
            max_hold_bars=backtest.max_hold_bars,
            trailing_stop_atr_multiple=backtest.trailing_stop_atr_multiple,
        )
        stress_report = run_walk_forward(
            frames,
            walk_forward_settings=walk_forward,
            backtest_settings=stress_backtest,
            portfolio_settings=PortfolioSettings(),
            mode=mode,
        )
        reports[mode] = {"base_costs": base_report, "stress_costs_1_5x": stress_report}
    report = {
        "schema_version": 1,
        "kind": "rae_walkforward_mode_and_cost_comparison",
        "modes": reports,
        "cost_stress_multiplier": 1.5,
        "holdout_status": "exploratory_only; the fetched two-year history overlaps previously inspected RAE v1.1/v1.2 dates and is not an independent future holdout",
        "tjr_comparison": {
            "status": "not_matched",
            "reason": "TJR uses 1m/5m intraday triggers while this RAE runner uses 1h signals and next-hour-open fills; do not compare these as equivalent.",
        },
    }
    report["data"] = {
        symbol: {
            "bars": len(frame),
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
        }
        for symbol, frame in frames.items()
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen RAE walk-forward validation.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    parser.add_argument("--cache", default=None)
    parser.add_argument("--train-bars", type=int, default=4_000)
    parser.add_argument("--test-bars", type=int, default=720)
    parser.add_argument("--purge-bars", type=int, default=24)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rae_walkforward.json"))
    args = parser.parse_args(argv)
    report = run_cached_walkforward(
        symbols=tuple(args.symbols),
        cache_dir=args.cache,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
        simulations=args.simulations,
        initial_cash=args.initial_cash,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({mode: value["base_costs"]["summary"] for mode, value in report["modes"].items()}, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
