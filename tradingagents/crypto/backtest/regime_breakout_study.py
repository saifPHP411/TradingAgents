"""Chronological, multi-symbol OOS study for the isolated breakout candidate.

No parameter fitting is performed. The warm-up window supplies indicator history
only; every reported trade belongs to a later, disjoint test window. Results are
reported separately by symbol and execution-cost scenario.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .regime_breakout import (
    BacktestSettings,
    ExecutionCosts,
    RegimeBreakoutSettings,
    run_backtest,
)


COST_SCENARIOS = {
    "frictionless": ExecutionCosts(0, 0, 0),
    "realistic_estimate": ExecutionCosts(10, 1, 2),
    "conservative_estimate": ExecutionCosts(15, 3, 5),
}


def _buy_and_hold_return(
    frame: pd.DataFrame, start: int, end: int, initial_cash: float, costs: ExecutionCosts
) -> float:
    """Long-only buy-and-hold return with two-sided fees and execution impact."""
    impact = (costs.half_spread_bps + costs.slippage_bps) / 10_000
    fee_rate = costs.commission_bps / 10_000
    buy = float(frame.iloc[start]["open"]) * (1 + impact)
    sell = float(frame.iloc[end]["close"]) * (1 - impact)
    quantity = initial_cash / (buy * (1 + fee_rate))
    final_cash = quantity * sell * (1 - fee_rate)
    return (final_cash / initial_cash - 1) * 100


def run_walk_forward_study(
    frames: Mapping[str, pd.DataFrame],
    *,
    warmup_bars: int = 250,
    test_bars: int = 120,
    initial_cash: float = 10_000,
    strategy: RegimeBreakoutSettings = RegimeBreakoutSettings(),
    portfolio: BacktestSettings | None = None,
    scenarios: Mapping[str, ExecutionCosts] = COST_SCENARIOS,
) -> dict:
    """Run fixed-parameter, expanding-context OOS folds for every symbol.

    All symbols must have valid chronological OHLCV frames. Fold windows are
    disjoint and use identical positional boundaries; no training-period trades
    are included in OOS results. The first warm-up bars are context only.
    """
    if warmup_bars < 60 or test_bars < 20:
        raise ValueError("warmup_bars must be >= 60 and test_bars >= 20")
    if not frames:
        raise ValueError("at least one symbol is required")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    ordered = {symbol: frame for symbol, frame in sorted(frames.items())}
    lengths = {len(frame) for frame in ordered.values()}
    if len(lengths) != 1:
        raise ValueError("all symbols must have aligned bar counts")
    for frame in ordered.values():
        if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("all symbol frames must be non-empty and chronological")
    bar_count = next(iter(lengths))
    if bar_count < warmup_bars + test_bars:
        raise ValueError("not enough history for one complete OOS fold")

    portfolio = portfolio or BacktestSettings(initial_cash=initial_cash)
    folds = []
    for fold_number, start in enumerate(range(warmup_bars, bar_count - test_bars + 1, test_bars), 1):
        end = start + test_bars - 1
        fold = {
            "fold": fold_number,
            "context_start": ordered[next(iter(ordered))].index[0].isoformat(),
            "oos_start": ordered[next(iter(ordered))].index[start].isoformat(),
            "oos_end": ordered[next(iter(ordered))].index[end].isoformat(),
            "symbols": {},
            "scenarios": {},
        }
        for scenario_name, costs in scenarios.items():
            symbol_rows = {}
            for symbol, frame in ordered.items():
                result = run_backtest(
                    frame,
                    strategy=strategy,
                    execution=costs,
                    portfolio=portfolio,
                    start_index=start,
                    end_index=end,
                )
                metrics = result.metrics()
                metrics["buy_hold_return_pct"] = _buy_and_hold_return(
                    frame, start, end, initial_cash, costs
                )
                metrics["oos_start"] = frame.index[start].isoformat()
                metrics["oos_end"] = frame.index[end].isoformat()
                symbol_rows[symbol] = metrics
            returns = [row["net_return_pct"] for row in symbol_rows.values()]
            benchmarks = [row["buy_hold_return_pct"] for row in symbol_rows.values()]
            fold["scenarios"][scenario_name] = {
                "equal_weight_net_return_pct": sum(returns) / len(returns),
                "equal_weight_buy_hold_return_pct": sum(benchmarks) / len(benchmarks),
                "symbols": symbol_rows,
            }
        folds.append(fold)

    summary = {}
    for scenario_name in scenarios:
        returns = [fold["scenarios"][scenario_name]["equal_weight_net_return_pct"] for fold in folds]
        benchmarks = [fold["scenarios"][scenario_name]["equal_weight_buy_hold_return_pct"] for fold in folds]
        equity = initial_cash
        peak = equity
        max_drawdown = 0.0
        for value in returns:
            equity *= 1 + value / 100
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0)
        summary[scenario_name] = {
            "folds": len(folds),
            "compounded_oos_return_pct": (equity / initial_cash - 1) * 100,
            "max_drawdown_across_fold_returns_pct": max_drawdown * 100,
            "mean_fold_return_pct": sum(returns) / len(returns),
            "mean_buy_hold_return_pct": sum(benchmarks) / len(benchmarks),
            "positive_folds": sum(value > 0 for value in returns),
            "profitable_after_costs": all(value > 0 for value in returns),
        }
    return {
        "study": "regime_breakout_fixed_parameters_chronological_oos",
        "data_source": "Binance public Spot historical candles; execution target is Bybit Demo Spot",
        "parameters_fitted_on_oos": False,
        "warmup_bars": warmup_bars,
        "test_bars_per_fold": test_bars,
        "symbols": list(ordered),
        "bar_count_per_symbol": bar_count,
        "first_bar": next(iter(ordered.values())).index[0].isoformat(),
        "last_bar": next(iter(ordered.values())).index[-1].isoformat(),
        "strategy": asdict(strategy),
        "cost_scenarios_per_fill_bps": {
            name: {
                "commission": costs.commission_bps,
                "half_spread": costs.half_spread_bps,
                "slippage": costs.slippage_bps,
            }
            for name, costs in scenarios.items()
        },
        "summary": summary,
        "folds": folds,
        "selection_gate": (
            "Do not select for Demo unless realistic and conservative scenarios show "
            "positive compounded OOS returns, positive expectancy, acceptable drawdown, "
            "and no single-symbol dependency; compare against buy-and-hold."
        ),
    }


def load_hourly_frames(cache_dir: str, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load cached 1m bars and aggregate only completed UTC hourly candles."""
    from .history import HistoryCache

    cache = HistoryCache(cache_dir)
    now = pd.Timestamp.now(tz="UTC")
    frames = {}
    for symbol in symbols:
        minutes = cache.load(symbol, "1m")
        if minutes is None or minutes.empty:
            raise ValueError(f"missing cached 1m history for {symbol}")
        hourly = minutes.resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        hourly = hourly.loc[hourly.index + pd.Timedelta(hours=1) <= now]
        if hourly.empty:
            raise ValueError(f"no completed hourly candles for {symbol}")
        frames[symbol] = hourly
    return frames


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run fixed-parameter breakout OOS study.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    parser.add_argument("--cache", default="./crypto-cache")
    parser.add_argument("--warmup-bars", type=int, default=250)
    parser.add_argument("--test-bars", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("artifacts/regime_breakout_oos.json"))
    args = parser.parse_args(argv)
    frames = load_hourly_frames(args.cache, tuple(args.symbols))
    report = run_walk_forward_study(
        frames, warmup_bars=args.warmup_bars, test_bars=args.test_bars
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
