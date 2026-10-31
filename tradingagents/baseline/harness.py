"""Run the official baseline: the split, the walk-forward, and the analyses.

This is glue. It builds the frozen feed, runs the reference strategy through the
existing ``PortfolioBacktester`` (no execution logic is reimplemented here),
measures with the existing ``performance`` engine, cross-checks the headline
numbers independently, and assembles the machine-readable artifact.

Everything downstream of the strategy is the production engine. If this module
and the engine ever disagree about accounting, the engine is right and this module
is wrong; the cross-check exists to catch exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.backtesting import BacktestConfig, PortfolioBacktester, run_walk_forward
from tradingagents.performance.report import metrics_from_result

from .crosscheck import all_agree, cross_check
from .protocol import BaselineProtocol
from .reference import align_benchmark, benchmark_equity


def _config(protocol: BaselineProtocol, *, costs=None) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=protocol.initial_cash,
        currency=protocol.currency,
        holding_period_days=protocol.holding_period_days,
        risk_limits=protocol.limits,
        costs=costs or protocol.costs,
        max_volume_participation=protocol.max_volume_participation,
        use_stops=protocol.use_stops,
        execute_on_next_open=protocol.execute_on_next_open,
    )


def run_window(feed, protocol: BaselineProtocol, strategy, start: str, end: str, *, costs=None):
    """Run the strategy over ``[start, end]`` and return the engine result."""
    engine = PortfolioBacktester(
        feed, strategy, config=_config(protocol, costs=costs), run_id=f"{strategy.name}_{start}_{end}"
    )
    return engine.run(start, end)


def _benchmark_series(feed, protocol: BaselineProtocol, sessions: list[str]) -> pd.Series:
    curve = benchmark_equity(
        feed, protocol.benchmark, initial_cash=protocol.initial_cash, sessions=sessions
    )
    return align_benchmark(curve, sessions)


@dataclass
class WindowResult:
    label: str
    start: str
    end: str
    result: object
    metrics: object
    benchmark: pd.Series
    checks: list

    @property
    def cross_check_passes(self) -> bool:
        return all_agree(self.checks)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "sessions": self.metrics.sessions,
            "metrics": self.metrics.to_dict(),
            "cross_check": [c.to_dict() for c in self.checks],
            "cross_check_passes": self.cross_check_passes,
            "trades": len(self.result.trades),
            "rejections": len(self.result.rejections),
            "unfilled": len(self.result.unfilled),
        }


def evaluate_window(feed, protocol: BaselineProtocol, strategy, label: str, start: str, end: str) -> WindowResult:
    """Run one window, measure it, and cross-check the headline numbers."""
    result = run_window(feed, protocol, strategy, start, end)
    sessions = result.sessions
    benchmark = _benchmark_series(feed, protocol, sessions)
    metrics = metrics_from_result(result, benchmark_prices=benchmark)
    checks = cross_check(result, metrics)
    return WindowResult(label, start, end, result, metrics, benchmark, checks)


def run_split(feed, protocol: BaselineProtocol, strategy) -> dict[str, WindowResult]:
    """The development / validation / test partition, each run over its own sessions."""
    out: dict[str, WindowResult] = {}
    for label, start, end in protocol.split_periods():
        out[label] = evaluate_window(feed, protocol, strategy, label, start, end)
    return out


def run_walkforward(feed, protocol: BaselineProtocol, strategy):
    """The rolling walk-forward over the whole calendar, one candidate.

    ``evaluate`` runs the candidate over a window's already-sliced feed and returns
    its total return. There is a single candidate, so selection is trivial and the
    value is in the *disjoint, contiguous test windows*: each is an honest
    out-of-sample sample, and together they cover most of the calendar.
    """

    def evaluate(candidate, window_feed, window) -> float:
        result = run_window(window_feed, protocol, candidate, window.start, window.end)
        if not result.equity_curve:
            return 0.0
        return (result.final_equity - result.initial_cash) / result.initial_cash

    return run_walk_forward(
        feed,
        candidates=[strategy],
        evaluate=evaluate,
        train_sessions=protocol.wf_train_sessions,
        validate_sessions=protocol.wf_validate_sessions,
        test_sessions=protocol.wf_test_sessions,
    )


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------


def regime_labels(feed, protocol: BaselineProtocol, sessions: list[str]) -> pd.DataFrame:
    """Classify each session by the benchmark's *trailing* trend and volatility.

    Both labels use only data up to and including the session, so a regime label
    is knowable at the session and no future information leaks in. Trend is the
    benchmark close against its own 200-session trailing mean; volatility is the
    60-session trailing realised volatility split at its own median. The labels
    are descriptive buckets, not predictions.
    """
    closes = feed.history(protocol.benchmark, sessions[-1])["close"]
    closes = closes.reindex(pd.to_datetime(sessions)).ffill()
    closes.index = pd.Index(sessions, name="session")

    trend_ma = closes.rolling(200, min_periods=60).mean()
    trend = pd.Series(
        ["bull" if c > m else "bear" for c, m in zip(closes, trend_ma, strict=True)],
        index=closes.index,
    )
    returns = closes.pct_change()
    vol = returns.rolling(60, min_periods=20).std()
    threshold = float(vol.median())
    vol_label = pd.Series(
        ["high_vol" if v > threshold else "low_vol" for v in vol.fillna(threshold)],
        index=closes.index,
    )
    return pd.DataFrame({"trend": trend, "volatility": vol_label}, index=closes.index)


def regime_breakdown(equity: pd.Series, labels: pd.DataFrame) -> dict:
    """Per-regime return contribution of the strategy's own equity curve."""
    returns = equity.pct_change().fillna(0.0)
    joined = labels.reindex(equity.index)
    out: dict[str, dict] = {}
    for column in ("trend", "volatility"):
        groups: dict[str, dict] = {}
        for name, idx in joined.groupby(column).groups.items():
            subset = returns.loc[idx]
            groups[str(name)] = {
                "sessions": int(len(subset)),
                "cumulative_return": float((1.0 + subset).prod() - 1.0),
                "mean_daily_return": float(subset.mean()),
            }
        out[column] = groups
    return out


# ---------------------------------------------------------------------------
# Concentration analysis
# ---------------------------------------------------------------------------


def concentration(trades: list, net_pnl: float) -> dict:
    """How much of the net P&L comes from the best trades and from one symbol."""
    pnls = sorted((float(t.net_pnl) for t in trades), reverse=True)
    total_positive = sum(p for p in pnls if p > 0)

    def share(n: int) -> float | None:
        if total_positive <= 0:
            return None
        return sum(pnls[:n]) / total_positive

    by_symbol: dict[str, float] = {}
    for trade in trades:
        by_symbol[trade.symbol] = by_symbol.get(trade.symbol, 0.0) + float(trade.net_pnl)
    best_symbol = max(by_symbol.items(), key=lambda kv: kv[1]) if by_symbol else None
    return {
        "net_pnl": net_pnl,
        "gross_profit": total_positive,
        "top_1_share_of_gross_profit": share(1),
        "top_3_share_of_gross_profit": share(3),
        "top_5_share_of_gross_profit": share(5),
        "pnl_by_symbol": by_symbol,
        "best_symbol": best_symbol[0] if best_symbol else None,
        "best_symbol_pnl": best_symbol[1] if best_symbol else None,
        "largest_single_trade": pnls[0] if pnls else None,
    }


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------


def cost_sensitivity(feed, protocol: BaselineProtocol, strategy, *, label="test") -> dict:
    """Rerun one window under baseline, doubled and quadrupled costs.

    A sensitivity test, not a tuning step: it shows how much of the result depends
    on the cost assumptions, and nothing here feeds back into the strategy.
    """
    window = {name: (s, e) for name, s, e in protocol.split_periods()}[label]
    start, end = window
    scenarios = {
        "baseline": protocol.costs,
        "double_cost": protocol.costs.model_copy(
            update={
                "commission_rate": protocol.commission_rate * 2,
                "half_spread_bps": protocol.half_spread_bps * 2,
                "slippage_bps": protocol.slippage_bps * 2,
            }
        ),
        "quadruple_cost": protocol.costs.model_copy(
            update={
                "commission_rate": protocol.commission_rate * 4,
                "half_spread_bps": protocol.half_spread_bps * 4,
                "slippage_bps": protocol.slippage_bps * 4,
            }
        ),
        "zero_cost": protocol.costs.model_copy(
            update={"commission_rate": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0}
        ),
    }
    out: dict[str, dict] = {}
    for name, costs in scenarios.items():
        result = run_window(feed, protocol, strategy, start, end, costs=costs)
        metrics = metrics_from_result(result)
        out[name] = {
            "total_return": metrics.total_return,
            "net_pnl": metrics.final_equity - metrics.initial_capital,
            "total_fees": metrics.total_fees,
            "num_trades": metrics.num_trades,
        }
    return out


def cost_share(metrics) -> dict:
    """Fees as a share of gross P&L and of traded notional."""
    gross = metrics.final_equity - metrics.initial_capital + metrics.total_fees
    return {
        "total_fees": metrics.total_fees,
        "cost_pct_of_gross_pnl": (metrics.total_fees / gross) if gross > 0 else None,
        "turnover": metrics.turnover,
    }


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


def summarize_window(window: WindowResult) -> dict:
    m = window.metrics
    return {
        "label": window.label,
        "start": window.start,
        "end": window.end,
        "sessions": m.sessions,
        "initial_capital": m.initial_capital,
        "final_equity": m.final_equity,
        "net_pnl": m.final_equity - m.initial_capital,
        "total_return": m.total_return,
        "annualized_return": m.annualized_return,
        "cagr": m.cagr,
        "max_drawdown": m.max_drawdown,
        "max_drawdown_start": m.max_drawdown_start,
        "max_drawdown_end": m.max_drawdown_end,
        "volatility": m.volatility,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "calmar_ratio": m.calmar_ratio,
        "num_trades": m.num_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "win_rate": m.win_rate,
        "average_win": m.average_win,
        "average_loss": m.average_loss,
        "profit_factor": m.profit_factor,
        "expectancy": m.expectancy,
        "exposure": m.exposure,
        "turnover": m.turnover,
        "total_fees": m.total_fees,
        "benchmark_return": m.benchmark_return,
        "excess_return": m.excess_return,
        "rejections": len(window.result.rejections),
        "unfilled": len(window.result.unfilled),
    }
