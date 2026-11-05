"""Turn a TJR run into the numbers the task asks for, honestly.

The performance engine already knows how to turn an equity curve and a list of
closed trades into metrics, so this module reuses it rather than re-deriving Sharpe
or drawdown. What it adds is the TJR-specific picture: why the strategy stood aside
(the reason-code tally), how each symbol did, and a Buy & Hold comparison.

Every figure is the ledger's own. Fees and slippage are the run's configured
schedule, printed alongside the result so a zero-cost figure is never mistaken for a
real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradingagents.performance.metrics import PerformanceMetrics, compute_metrics

from .engine import TjrResult
from .rules import (
    TJR_ENTRY_WINDOW_CLOSED,
    TJR_INVALID_GEOMETRY,
    TJR_NO_1M_ENTRY,
    TJR_NO_5M_CONFIRMATION,
    TJR_NO_CONTINUATION,
    TJR_NO_SETUP,
    TJR_NO_SWEEP,
    TJR_OUTSIDE_SESSION,
    TJR_READY,
    TJR_SHORT_SETUP_LONG_ONLY,
)

_REASON_LABELS: dict[str, str] = {
    TJR_NO_SETUP: "no level marked",
    TJR_NO_SWEEP: "no liquidity sweep",
    TJR_NO_5M_CONFIRMATION: "swept, no 5m BOS/IFVG",
    TJR_NO_CONTINUATION: "confirmed, no retrace into a zone / no target",
    TJR_NO_1M_ENTRY: "in the zone, no 1m confirmation",
    TJR_INVALID_GEOMETRY: "stop ended up on the wrong side of the entry",
    TJR_SHORT_SETUP_LONG_ONLY: "complete bearish setup, book is long-only",
    TJR_OUTSIDE_SESSION: "outside the regular session",
    TJR_ENTRY_WINDOW_CLOSED: "entry window closed",
    TJR_READY: "complete setup",
}


@dataclass
class TjrReport:
    """The full report of one TJR run."""

    metrics: PerformanceMetrics
    reason_counts: dict[str, int]
    per_symbol: dict[str, dict]
    exit_reasons: dict[str, int]
    average_rr: float | None
    """Mean planned reward-to-risk ``(target-entry)/(entry-stop)`` over the trades
    that carried both a stop and a target. This is the setup's design R:R, not the
    P&L it realised."""
    average_realized_r: float | None
    """Mean realised R-multiple ``net_pnl / (quantity x (entry-stop))`` over the
    trades with a positive per-unit risk. The realised counterpart to
    ``average_rr``: it says how much of the planned reward each trade actually
    captured."""
    entries_per_session: dict[str, int]
    """Entries placed per session, so the ``max_daily_orders`` cap is auditable."""
    max_entries_in_a_session: int
    fees: float
    slippage_note: str
    buy_and_hold_return: float | None
    buy_and_hold_method: str
    sessions: int
    symbols: tuple[str, ...]
    start: str | None
    end: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "reason_counts": self.reason_counts,
            "reason_labels": {code: _REASON_LABELS.get(code, code) for code in self.reason_counts},
            "per_symbol": self.per_symbol,
            "exit_reasons": self.exit_reasons,
            "average_rr": self.average_rr,
            "average_realized_r": self.average_realized_r,
            "entries_per_session": self.entries_per_session,
            "max_entries_in_a_session": self.max_entries_in_a_session,
            "fees": self.fees,
            "slippage_note": self.slippage_note,
            "buy_and_hold_return": self.buy_and_hold_return,
            "buy_and_hold_method": self.buy_and_hold_method,
            "sessions": self.sessions,
            "symbols": list(self.symbols),
            "start": self.start,
            "end": self.end,
            "notes": self.notes,
        }


def build_report(result: TjrResult, *, buy_and_hold: pd.Series | None = None) -> TjrReport:
    """Compute the report for a TJR run.

    ``buy_and_hold`` may be supplied as an explicit benchmark equity series. When
    it is not, one is derived from the strategy's own fills: an equal-weight book
    holding every symbol the strategy ever traded, from each symbol's first fill to
    the run's end. That is the honest comparison — the strategy chose which names
    to touch, so the benchmark should hold the same names — and it is reported
    with ``buy_and_hold_method`` naming exactly what it is.
    """
    equity = result.equity_series()
    exposure = result.exposure_series()
    trades = result.trades
    total_fees = result.ledger.total_fees if result.ledger is not None else 0.0
    if buy_and_hold is None:
        buy_and_hold = _equal_weight_benchmark(result)
    metrics = compute_metrics(
        equity,
        initial_capital=result.initial_cash,
        trades=trades,
        exposure=exposure,
        traded_notional=result.traded_notional,
        realized_pnl=result.ledger.realized_pnl if result.ledger is not None else 0.0,
        unrealized_pnl=result.ledger.unrealized_pnl if result.ledger is not None else 0.0,
        total_fees=total_fees,
        benchmark=buy_and_hold,
    )
    exit_reasons: dict[str, int] = {}
    planned_rr: list[float] = []
    realized_r: list[float] = []
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
        if trade.planned_reward_risk is not None:
            planned_rr.append(trade.planned_reward_risk)
        if trade.realized_r is not None:
            realized_r.append(trade.realized_r)
    average_rr = (sum(planned_rr) / len(planned_rr)) if planned_rr else None
    average_realized_r = (sum(realized_r) / len(realized_r)) if realized_r else None
    benchmark_return = metrics.benchmark_return
    notes: list[str] = []
    if result.config is not None and _is_frictionless(result.config.costs):
        notes.append(
            "costs are zero in this run; the figures are frictionless and are not a "
            "claim about net-of-fee performance"
        )
    return TjrReport(
        metrics=metrics,
        reason_counts=dict(sorted(result.reason_counts.items())),
        per_symbol=dict(sorted(result.per_symbol.items())),
        exit_reasons=dict(sorted(exit_reasons.items())),
        average_rr=average_rr,
        average_realized_r=average_realized_r,
        entries_per_session=dict(sorted(result.entries_per_session.items())),
        max_entries_in_a_session=max(result.entries_per_session.values(), default=0),
        fees=total_fees,
        slippage_note=_slippage_note(result),
        buy_and_hold_return=benchmark_return,
        buy_and_hold_method=_benchmark_method(result),
        sessions=len(result.sessions),
        symbols=result.symbols,
        start=metrics.start,
        end=metrics.end,
        notes=notes,
    )


def _equal_weight_benchmark(result: TjrResult) -> pd.Series | None:
    """Equal-weight Buy & Hold over the names the strategy traded, at 1m closes.

    Built on the same index as the equity curve, so the two series are comparable.
    Each traded symbol is bought at its first available close for an equal share of
    the starting cash; a symbol with no price yet is held as cash. Returns ``None``
    when nothing was traded, because there is no basket to hold.
    """
    traded = [symbol for symbol, stats in result.per_symbol.items() if stats.get("trades")]
    if not traded or not result.equity_curve:
        return None
    index = pd.DatetimeIndex(pd.to_datetime([stamp for stamp, _ in result.equity_curve]))
    weights = result.initial_cash / len(traded)
    total = pd.Series(0.0, index=index)
    for symbol in traded:
        series = result.one_minute.get(symbol)
        closes = None if series is None else series.frame["close"].reindex(index, method="ffill")
        entry = None if closes is None else closes.bfill().iloc[0]
        if closes is None or not entry or entry <= 0:
            total = total + weights
            continue
        units = weights / float(entry)
        total = total + closes.mul(units).where(closes.notna(), weights)
    return total


def _benchmark_method(result: TjrResult) -> str:
    if not any(stats.get("trades") for stats in result.per_symbol.values()):
        return "none: nothing was traded"
    return (
        "equal-weight Buy & Hold of the traded symbols at 1m closes, from each "
        "symbol's first available bar, against the run's own index"
    )


def _slippage_note(result: TjrResult) -> str:
    if result.config is None:
        return "no cost model recorded"
    costs = result.config.costs
    return (
        f"half_spread_bps={costs.half_spread_bps}, slippage_bps={costs.slippage_bps}, "
        f"commission_per_unit={costs.commission_per_unit}, commission_min={costs.commission_min}"
    )


def _is_frictionless(costs) -> bool:
    """Whether a cost model charges nothing at all, fees included."""
    return (
        costs.commission_flat == 0
        and costs.commission_per_unit == 0
        and costs.commission_rate == 0
        and costs.half_spread_bps == 0
        and costs.slippage_bps == 0
        and costs.taker_fee_bps == 0
        and costs.maker_fee_bps == 0
    )
