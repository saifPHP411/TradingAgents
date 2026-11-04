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
    fees: float
    slippage_note: str
    buy_and_hold_return: float | None
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
            "fees": self.fees,
            "slippage_note": self.slippage_note,
            "buy_and_hold_return": self.buy_and_hold_return,
            "sessions": self.sessions,
            "symbols": list(self.symbols),
            "start": self.start,
            "end": self.end,
            "notes": self.notes,
        }


def build_report(result: TjrResult, *, buy_and_hold: pd.Series | None = None) -> TjrReport:
    """Compute the report for a TJR run."""
    equity = result.equity_series()
    trades = result.trades
    total_fees = result.ledger.total_fees if result.ledger is not None else 0.0
    metrics = compute_metrics(
        equity,
        initial_capital=result.initial_cash,
        trades=trades,
        realized_pnl=result.ledger.realized_pnl if result.ledger is not None else 0.0,
        unrealized_pnl=result.ledger.unrealized_pnl if result.ledger is not None else 0.0,
        total_fees=total_fees,
        benchmark=buy_and_hold,
    )
    exit_reasons: dict[str, int] = {}
    rr_values: list[float] = []
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
        if trade.entry_price and trade.exit_price and trade.entry_price != trade.exit_price:
            rr_values.append(trade.net_pnl / (trade.entry_price * trade.quantity) if trade.quantity else 0.0)
    average_rr = (sum(rr_values) / len(rr_values)) if rr_values else None
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
        fees=total_fees,
        slippage_note=_slippage_note(result),
        buy_and_hold_return=benchmark_return,
        sessions=len(result.sessions),
        symbols=result.symbols,
        start=metrics.start,
        end=metrics.end,
        notes=notes,
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
