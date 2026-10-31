"""Independent recomputation of the headline metrics, from raw run data.

The performance engine already computes these numbers. This module computes them
again from the equity curve, the trade list and the ledger's event log, using
plain arithmetic that shares no code with ``performance/metrics.py``. The two are
then compared. A disagreement is a bug in one of them, and the baseline refuses to
publish a headline until they agree.

The point is not distrust of one function; it is that a metric computed one way is
a claim, and a metric computed two ways that agree is evidence. Recomputing by
hand is the cheapest way to catch an off-by-one, an annualisation applied twice,
or a drawdown measured from the wrong peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrossCheck:
    """The implementation's value, the independent value, and whether they agree."""

    metric: str
    reported: float | None
    independent: float | None
    agrees: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "reported": self.reported,
            "independent": self.independent,
            "agrees": self.agrees,
            "note": self.note,
        }


def _close(a: float | None, b: float | None, *, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def independent_total_return(equity: list[float], initial_cash: float) -> float:
    return (equity[-1] - initial_cash) / initial_cash if initial_cash else 0.0


def independent_net_pnl(equity: list[float], initial_cash: float) -> float:
    return equity[-1] - initial_cash


def independent_max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction of the running peak."""
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def independent_total_fees(ledger) -> float:
    """Sum the fee charged on every fill, from the event log, not the ledger total."""
    total = 0.0
    for event in ledger.events:
        if event.type.value == "FEE_CHARGED":
            total += float(event.detail.get("fee", 0.0))
    return total


def independent_trade_stats(trades: list) -> dict:
    """Win/loss counts, win rate and profit factor from the trade list alone."""
    pnls = [float(t.net_pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "num_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": (len(wins) / (len(wins) + len(losses))) if (wins or losses) else None,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "largest_win": max(wins) if wins else None,
        "largest_loss": min(losses) if losses else None,
        "average_trade": (sum(pnls) / len(pnls)) if pnls else None,
    }


def cross_check(result, metrics) -> list[CrossCheck]:
    """Recompute the headline metrics and compare them with the implementation's."""
    equity = [point.equity for point in result.equity_curve]
    ledger = result.ledger
    trades = result.trades
    stats = independent_trade_stats(trades)

    checks: list[CrossCheck] = []

    def add(metric, reported, independent, *, tol=1e-6, note=""):
        checks.append(
            CrossCheck(metric, reported, independent, _close(reported, independent, tol=tol), note)
        )

    add("total_return", metrics.total_return,
        independent_total_return(equity, result.initial_cash) if equity else None)
    add("net_pnl", metrics.final_equity - metrics.initial_capital,
        independent_net_pnl(equity, result.initial_cash) if equity else None)
    add("max_drawdown", metrics.max_drawdown,
        independent_max_drawdown(equity) if equity else None)
    add("num_trades", float(metrics.num_trades), float(stats["num_trades"]))
    add("winning_trades", float(metrics.winning_trades), float(stats["winning_trades"]))
    add("losing_trades", float(metrics.losing_trades), float(stats["losing_trades"]))
    add("win_rate", metrics.win_rate, stats["win_rate"])
    add("profit_factor", metrics.profit_factor, stats["profit_factor"])
    if ledger is not None:
        add("total_fees", metrics.total_fees, independent_total_fees(ledger))

    return checks


def all_agree(checks: list[CrossCheck]) -> bool:
    return all(check.agrees for check in checks)
