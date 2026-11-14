"""Cost-aware chronological OOS backtests for the adaptive research engine.

Signals are evaluated only on completed bars and entries occur at the next open.
This module is research-only and has no exchange/order dependencies.
"""
from __future__ import annotations

import collections.abc
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from .adaptive import AdaptiveConfig, analyze_market


@dataclass(frozen=True)
class AdaptiveExecutionCosts:
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(not pd.notna(value) or abs(value) == float("inf") or value < 0 for value in values):
            raise ValueError("execution costs must be finite and non-negative")


@dataclass(frozen=True)
class AdaptivePortfolio:
    initial_cash: float = 10_000.0
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10
    max_holding_bars: int = 48

    def __post_init__(self) -> None:
        if not pd.notna(self.initial_cash) or abs(self.initial_cash) == float("inf") or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive and finite")
        if not 0 < self.risk_fraction <= 0.02:
            raise ValueError("risk_fraction must be in (0, 0.02]")
        if not 0 < self.max_notional_fraction <= 1:
            raise ValueError("max_notional_fraction must be in (0, 1]")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be positive")


@dataclass
class AdaptiveTrade:
    entry_time: str
    exit_time: str
    strategy: str
    quantity: float
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: str
    fees: float
    gross_pnl: float
    net_pnl: float


@dataclass
class AdaptiveBacktestResult:
    initial_cash: float
    final_equity: float
    trades: list[AdaptiveTrade] = field(default_factory=list)
    total_fees: float = 0.0
    total_spread_slippage: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    regimes: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)

    def metrics(self) -> dict:
        winners = [trade.net_pnl for trade in self.trades if trade.net_pnl > 0]
        losers = [trade.net_pnl for trade in self.trades if trade.net_pnl < 0]
        net_profit_on_winners = sum(winners)
        net_loss_on_losers = abs(sum(losers))
        peak = self.initial_cash
        max_drawdown = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return {
            "initial_equity": self.initial_cash,
            "final_equity": self.final_equity,
            "net_pnl": self.final_equity - self.initial_cash,
            "net_return_pct": (self.final_equity / self.initial_cash - 1) * 100,
            "trades": len(self.trades),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate_pct": len(winners) / len(self.trades) * 100 if self.trades else 0.0,
            "profit_factor": (
                net_profit_on_winners / net_loss_on_losers
                if net_loss_on_losers else ("infinite" if net_profit_on_winners else 0.0)
            ),
            "net_profit_on_winners": net_profit_on_winners,
            "net_loss_on_losers": net_loss_on_losers,
            "expectancy_per_trade": sum(t.net_pnl for t in self.trades) / len(self.trades) if self.trades else 0.0,
            "gross_profit": sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0),
            "gross_loss": abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0)),
            "max_drawdown_pct": max_drawdown * 100,
            "fees": self.total_fees,
            "spread_and_slippage": self.total_spread_slippage,
            "regime_counts": dict(self.regimes),
            "rejection_counts": dict(self.rejections),
        }


def run_adaptive_backtest(
    frame: pd.DataFrame,
    *,
    config: AdaptiveConfig = AdaptiveConfig(),
    execution: AdaptiveExecutionCosts = AdaptiveExecutionCosts(),
    portfolio: AdaptivePortfolio = AdaptivePortfolio(),
    spread_bps: float = 2.0,
    start_index: int = 0,
    end_index: int | None = None,
    data_is_fresh: bool = True,
) -> AdaptiveBacktestResult:
    """Backtest one Spot symbol with closed-bar signals and next-open fills."""
    if not pd.notna(spread_bps) or abs(spread_bps) == float("inf") or spread_bps < 0:
        raise ValueError("spread_bps must be finite and non-negative")
    effective_spread_bps = max(spread_bps, execution.half_spread_bps)
    round_trip_cost_bps = 2 * (
        execution.commission_bps + execution.half_spread_bps + execution.slippage_bps
    )
    effective_config = replace(
        config,
        estimated_round_trip_cost_bps=max(config.estimated_round_trip_cost_bps, round_trip_cost_bps),
    )
    if effective_spread_bps > effective_config.max_spread_bps:
        raise ValueError("spread_bps must be within the configured liquidity limit")
    if frame.empty:
        return AdaptiveBacktestResult(portfolio.initial_cash, portfolio.initial_cash)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV index must be strictly chronological and unique")
    end_index = len(frame) - 1 if end_index is None else end_index
    if start_index < 0 or end_index >= len(frame) or start_index > end_index:
        raise ValueError("start_index/end_index must select a valid non-empty window")
    cash = float(portfolio.initial_cash)
    result = AdaptiveBacktestResult(cash, cash)
    position: dict | None = None
    impact = (execution.half_spread_bps + execution.slippage_bps) / 10_000
    fee_rate = execution.commission_bps / 10_000
    warmup = max(config.slow_ema, config.lookback, config.breakout_lookback, config.atr_period) + 5

    for i in range(start_index, end_index + 1):
        row = frame.iloc[i]
        stamp = frame.index[i]
        if position is not None:
            held = i - position["entry_index"]
            exit_raw = None
            exit_reason = None
            if float(row.low) <= position["stop"]:
                exit_raw = min(position["stop"], float(row.open))
                exit_reason = "stop"
            elif float(row.high) >= position["target"]:
                exit_raw = position["target"]
                exit_reason = "target"
            elif held >= portfolio.max_holding_bars:
                exit_raw = float(row.close)
                exit_reason = "time"
            elif i == end_index:
                exit_raw = float(row.close)
                exit_reason = "end_of_window"
            if exit_raw is not None:
                exit_fill = exit_raw * (1 - impact)
                exit_notional = position["quantity"] * exit_fill
                exit_fee = exit_notional * fee_rate
                cash += exit_notional - exit_fee
                gross = (exit_fill - position["entry_price"]) * position["quantity"]
                net = gross - position["entry_fee"] - exit_fee
                result.total_fees += exit_fee
                result.total_spread_slippage += abs(exit_raw - exit_fill) * position["quantity"]
                result.trades.append(AdaptiveTrade(
                    position["entry_time"], stamp.isoformat(), position["strategy"],
                    position["quantity"], position["entry_price"], exit_fill,
                    position["stop"], position["target"], exit_reason,
                    position["entry_fee"] + exit_fee, gross, net,
                ))
                position = None

        if position is None and i >= warmup and i + 1 <= end_index:
            history = frame.iloc[:i + 1]
            decision = analyze_market(
                history, config=effective_config, spread_bps=effective_spread_bps,
                data_is_fresh=data_is_fresh, closed_bars_only=True,
            )
            result.regimes[decision.regime] = result.regimes.get(decision.regime, 0) + 1
            if decision.blockers:
                for blocker in decision.blockers:
                    result.rejections[blocker] = result.rejections.get(blocker, 0) + 1
            if decision.action == "buy_candidate":
                next_open = float(frame.iloc[i + 1].open)
                entry_fill = next_open * (1 + impact)
                stop = float(decision.stop_price)
                target = float(decision.target_price)
                risk_per_unit = entry_fill - stop
                if risk_per_unit <= 0 or target <= entry_fill:
                    result.rejections["next-open gap invalidated setup"] = (
                        result.rejections.get("next-open gap invalidated setup", 0) + 1
                    )
                else:
                    risk_budget = cash * portfolio.risk_fraction
                    notional_cap = cash * portfolio.max_notional_fraction
                    quantity = min(risk_budget / risk_per_unit, notional_cap / entry_fill)
                    entry_notional = quantity * entry_fill
                    entry_fee = entry_notional * fee_rate
                    if quantity > 0 and entry_notional + entry_fee <= cash:
                        cash -= entry_notional + entry_fee
                        result.total_fees += entry_fee
                        result.total_spread_slippage += abs(entry_fill - next_open) * quantity
                        position = {
                            "entry_index": i + 1, "entry_time": frame.index[i + 1].isoformat(),
                            "strategy": decision.strategy, "quantity": quantity,
                            "entry_price": entry_fill, "entry_fee": entry_fee,
                            "stop": stop, "target": target,
                        }
                    else:
                        result.rejections["insufficient cash after entry fees"] = (
                            result.rejections.get("insufficient cash after entry fees", 0) + 1
                        )
        mark = cash
        if position is not None and i >= position["entry_index"]:
            mark += position["quantity"] * float(row.close)
        result.equity_curve.append(mark)

    result.final_equity = cash + (
        position["quantity"] * float(frame.iloc[end_index].close) if position is not None else 0.0
    )
    return result


DEFAULT_COST_SCENARIOS = {
    "frictionless": AdaptiveExecutionCosts(0, 0, 0),
    "realistic_estimate": AdaptiveExecutionCosts(10, 1, 2),
    "conservative_estimate": AdaptiveExecutionCosts(15, 3, 5),
}


def run_adaptive_walk_forward_study(
    frames: collections.abc.Mapping[str, pd.DataFrame],
    *,
    warmup_bars: int = 250,
    test_bars: int = 120,
    config: AdaptiveConfig = AdaptiveConfig(),
    portfolio: AdaptivePortfolio | None = None,
    scenarios: collections.abc.Mapping[str, AdaptiveExecutionCosts] = DEFAULT_COST_SCENARIOS,
    spread_bps: float = 2.0,
) -> dict:
    """Run disjoint chronological OOS folds, with identical timestamps per symbol."""
    if warmup_bars < 60 or test_bars < 20:
        raise ValueError("warmup_bars must be >= 60 and test_bars >= 20")
    if not frames:
        raise ValueError("at least one symbol is required")
    ordered = dict(sorted(frames.items()))
    lengths = {len(frame) for frame in ordered.values()}
    if len(lengths) != 1:
        raise ValueError("all symbols must have aligned bar counts")
    reference_index = next(iter(ordered.values())).index
    for frame in ordered.values():
        if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("all symbol frames must be non-empty and chronological")
        if not frame.index.equals(reference_index):
            raise ValueError("all symbols must have aligned timestamps")
    bar_count = next(iter(lengths))
    if bar_count < warmup_bars + test_bars:
        raise ValueError("not enough history for one complete OOS fold")
    base_portfolio = portfolio or AdaptivePortfolio()
    folds = []
    for fold_number, start in enumerate(range(warmup_bars, bar_count - test_bars + 1, test_bars), 1):
        end = start + test_bars - 1
        fold = {
            "fold": fold_number, "oos_start": reference_index[start].isoformat(),
            "oos_end": reference_index[end].isoformat(), "scenarios": {},
        }
        for scenario_name, costs in scenarios.items():
            symbol_rows = {}
            for symbol, frame in ordered.items():
                result = run_adaptive_backtest(
                    frame, config=config, execution=costs, portfolio=base_portfolio,
                    spread_bps=spread_bps, start_index=start, end_index=end,
                )
                symbol_rows[symbol] = result.metrics()
            returns = [row["net_return_pct"] for row in symbol_rows.values()]
            fold["scenarios"][scenario_name] = {
                "equal_weight_net_return_pct": sum(returns) / len(returns),
                "symbols": symbol_rows,
            }
        folds.append(fold)

    summary = {}
    for scenario_name in scenarios:
        returns = [fold["scenarios"][scenario_name]["equal_weight_net_return_pct"] for fold in folds]
        equity = base_portfolio.initial_cash
        peak, max_drawdown = equity, 0.0
        all_rows = [row for fold in folds for row in fold["scenarios"][scenario_name]["symbols"].values()]
        for value in returns:
            equity *= 1 + value / 100
            peak = max(peak, equity)
            if peak:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        winners = sum(row["net_profit_on_winners"] for row in all_rows)
        losers = sum(row["net_loss_on_losers"] for row in all_rows)
        trades = sum(row["trades"] for row in all_rows)
        pnl = sum(row["net_pnl"] for row in all_rows)
        per_symbol = {}
        for symbol in ordered:
            symbol_equity = base_portfolio.initial_cash
            for fold in folds:
                value = fold["scenarios"][scenario_name]["symbols"][symbol]["net_return_pct"]
                symbol_equity *= 1 + value / 100
            per_symbol[symbol] = (symbol_equity / base_portfolio.initial_cash - 1) * 100
        summary[scenario_name] = {
            "folds": len(folds), "trades": trades, "expectancy_per_trade": pnl / trades if trades else 0.0,
            "aggregate_profit_factor": winners / losers if losers else ("infinite" if winners else 0.0),
            "compounded_oos_return_pct": (equity / base_portfolio.initial_cash - 1) * 100,
            "max_drawdown_across_fold_returns_pct": max_drawdown * 100,
            "positive_folds": sum(value > 0 for value in returns),
            "per_symbol_compounded_oos_return_pct": per_symbol,
        }
    return {
        "study": "adaptive_multi_regime_chronological_oos",
        "parameters_fitted_on_oos": False,
        "data_source": "caller-provided historical OHLCV; venue and coverage must be recorded by caller",
        "symbols": list(ordered), "bar_count_per_symbol": bar_count,
        "warmup_bars": warmup_bars, "test_bars_per_fold": test_bars,
        "cost_scenarios_per_fill_bps": {
            name: asdict(costs) for name, costs in scenarios.items()
        },
        "spread_bps_assumption": spread_bps, "summary": summary, "folds": folds,
        "selection_gate": (
            "Research only: no Demo selection without positive after-cost OOS robustness, "
            "adequate trade count, independent market periods, and measured venue fills."
        ),
    }


def load_hourly_frames(cache_dir: str, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load cached 1m candles and aggregate only completed UTC hourly bars."""
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
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run adaptive multi-regime chronological OOS study.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    parser.add_argument("--cache", default="./crypto-cache")
    parser.add_argument("--warmup-bars", type=int, default=250)
    parser.add_argument("--test-bars", type=int, default=100)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/adaptive_crypto_oos.json"))
    args = parser.parse_args(argv)
    frames = load_hourly_frames(args.cache, tuple(args.symbols))
    report = run_adaptive_walk_forward_study(
        frames, warmup_bars=args.warmup_bars, test_bars=args.test_bars,
        spread_bps=args.spread_bps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
