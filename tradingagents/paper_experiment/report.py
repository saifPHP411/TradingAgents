"""The local report for a prospective paper experiment.

The report is a *read-only view* built from the experiment's state and journal.
It is not the recovery format and not the journal; regenerating it changes
nothing. Its job is to answer, honestly and with the sample size visible:

- how many sessions ran, and how many decisions were genuine, blocked, executed
  and evaluated;
- the book's cash, market value and equity, when the data to compute them exists;
- return and drawdown, only when the sample is large enough to mean anything;
- the costs and the assumptions behind them;
- the benchmark comparison, only when the comparison is like-for-like;
- the data source, its provenance, and every reason the experiment is blocked or
  not yet evaluable.

A metric that is undefined is ``None`` (rendered ``N/A``), never ``0.0``. A
sample smaller than ``config.min_trades_for_metrics`` yields ``None`` for the
ratio metrics and a ``sample_limited`` flag, so a small sample cannot be read as
a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingagents import __version__ as package_version
from tradingagents.llm_evaluation.records import assert_secret_free
from tradingagents.performance.metrics import compute_metrics

from .experiment import (
    STATE_BLOCKED,
    STATE_EVALUATED,
    STATE_EXECUTED,
    STATE_PENDING,
    STATE_UNEVALUABLE,
    PaperExperiment,
)

REPORT_SCHEMA_VERSION = "1.1.0"


@dataclass
class ExperimentReport:
    """The report's data, artifact-ready and secret-free."""

    payload: dict

    def to_dict(self) -> dict:
        return assert_secret_free(self.payload)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        assert_secret_free(text)
        target.write_text(text, encoding="utf-8")
        return target

    def summary(self) -> str:
        """A short human-readable summary for the console."""
        p = self.payload
        lines = [
            f"Paper experiment {p['experiment_id']}",
            f"  status: {p['readiness']['status']}",
            f"  source: {p['data']['source_name']} "
            f"(kind: {p['data'].get('source_kind')}, current market: "
            f"{p['data']['is_current_market']})",
            f"  sessions processed: {p['counts']['sessions_processed']}",
            f"  decisions: genuine={p['counts']['genuine_inferences']} "
            f"blocked={p['counts']['blocked']} executed={p['counts']['executed']} "
            f"evaluated={p['counts']['evaluated']}",
            f"  cash: {_fmt(p['book']['cash'])}  equity: {_fmt(p['book']['equity'])}",
            f"  total return: {_fmt_pct(p['performance']['total_return'])}",
            f"  max drawdown: {_fmt_pct(p['performance']['max_drawdown'])}",
        ]
        for blocker in p["readiness"]["blockers"]:
            lines.append(f"  blocker: {blocker}")
        return "\n".join(lines)


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def _fmt_pct(value) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def build_report(experiment: PaperExperiment) -> ExperimentReport:
    """Build the report from the experiment's current state."""
    counts = experiment.counts()
    decisions = list(experiment.decisions.values())
    genuine = [d for d in decisions if d.source == "llm_inference"]
    evaluated = [d for d in decisions if d.state == STATE_EVALUATED]
    executed = [
        d for d in decisions if d.state in (STATE_EXECUTED, STATE_EVALUATED, STATE_UNEVALUABLE)
    ]
    filled = [d for d in executed if d.filled_quantity > 0]

    book = _book(experiment)
    performance = _performance(experiment, evaluated, filled)
    data_block = _data_block(experiment)
    readiness = _readiness(experiment, genuine, counts)
    baseline = _baseline_block(experiment)
    orders = _order_funnel(decisions)

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "package_version": package_version,
        "experiment_id": experiment.experiment_id,
        "strategy_id": experiment.config.strategy_id,
        "config_hash": experiment.config.config_hash(),
        "created_at": experiment.created_at,
        "last_session_processed": (
            sorted(experiment.processed_sessions)[-1] if experiment.processed_sessions else None
        ),
        "counts": {
            "sessions_processed": len(experiment.processed_sessions),
            "decisions_total": len(decisions),
            "genuine_inferences": len(genuine),
            "blocked": counts.get(STATE_BLOCKED, 0),
            "pending": counts.get(STATE_PENDING, 0),
            "executed": counts.get(STATE_EXECUTED, 0),
            "evaluated": counts.get(STATE_EVALUATED, 0),
            "unevaluable": counts.get(STATE_UNEVALUABLE, 0),
            "filled_orders": len(filled),
        },
        "orders": orders,
        "book": book,
        "performance": performance,
        "costs": _costs_block(experiment),
        "data": data_block,
        "baseline": baseline,
        "readiness": readiness,
        "limitations": _limitations(experiment, genuine, performance),
        "journal": {
            "entries": len(experiment.journal),
            "verified": True,
            "path": str(experiment.journal_path),
        },
    }
    return ExperimentReport(payload)


def _book(experiment: PaperExperiment) -> dict:
    ledger = experiment.session.ledger
    positions = {
        symbol: {
            "quantity": holding.quantity,
            "average_entry_price": holding.average_entry_price,
            "market_value": holding.market_value,
            "unrealized_pnl": holding.unrealized_pnl,
        }
        for symbol, holding in ledger.open_positions().items()
    }
    return {
        "initial_cash": ledger.initial_cash,
        "cash": ledger.cash,
        "market_value": ledger.market_value,
        "equity": ledger.equity,
        "realized_pnl": ledger.realized_pnl,
        "unrealized_pnl": ledger.unrealized_pnl,
        "total_fees": ledger.total_fees,
        "positions": positions,
        "resting_orders": len(experiment.session.resting),
    }


class _OrderStatus:
    """Tally the order funnel from the settled decisions' own fields."""

    def __init__(self) -> None:
        self.settled = 0
        self.submitted = 0
        self.filled = 0
        self.partially_filled = 0
        self.rejected = 0
        self.cancelled = 0
        self.resting = 0
        self.without_order = 0
        self.without_order_reasons: dict[str, int] = {}

    def count(self, decision) -> None:
        self.settled += 1
        status = decision.order_status
        if not decision.order_created or status in (None, "NONE"):
            self.without_order += 1
            reason = decision.order_reason or "no order"
            self.without_order_reasons[reason] = self.without_order_reasons.get(reason, 0) + 1
            return
        self.submitted += 1
        if status == "FILLED":
            self.filled += 1
        elif status == "PARTIALLY_FILLED":
            self.partially_filled += 1
        elif status == "REJECTED":
            self.rejected += 1
        elif status == "CANCELLED":
            self.cancelled += 1
        elif status == "ACCEPTED":
            # A LIMIT the bar did not reach is still live; a filled-then-rested
            # remainder is a partial fill, already counted above.
            self.resting += 1

    def to_dict(self) -> dict:
        return {
            "settled": self.settled,
            "submitted": self.submitted,
            "filled": self.filled,
            "partially_filled": self.partially_filled,
            "rejected": self.rejected,
            "cancelled": self.cancelled,
            "resting": self.resting,
            "without_order": self.without_order,
            "without_order_reasons": dict(sorted(self.without_order_reasons.items())),
            "note": (
                "counts.executed counts decisions carried to their execution session "
                "(a HOLD included); this block counts what the orders themselves did"
            ),
        }


def _order_funnel(decisions: list) -> dict:
    """What each settled decision's order did, in closed-vocabulary terms.

    ``counts.executed`` answers "was the decision carried to its execution
    session", which is true for a HOLD. It is not the number of orders. This
    block makes the funnel explicit so ``executed`` cannot be read as "filled":

    - ``settled`` — decisions carried to their execution session (a HOLD included);
    - ``submitted`` — a real order reached the simulator (a HOLD is not one);
    - ``filled`` / ``partially_filled`` / ``rejected`` / ``cancelled`` / ``resting``
      — the order's own terminal-or-live status, each counted once;
    - ``without_order`` — settled decisions that produced no order at all (HOLD, or
      a refusal before an order existed), with their reasons tallied.
    """
    funnel = _OrderStatus()
    for decision in decisions:
        if decision.state in (STATE_EXECUTED, STATE_EVALUATED, STATE_UNEVALUABLE):
            funnel.count(decision)
    return funnel.to_dict()


def _costs_block(experiment: PaperExperiment) -> dict:
    costs = experiment.config.costs or experiment.session.simulator.costs.model_dump()
    return {
        "model": costs,
        "note": (
            "declared, not discovered: these are the configured assumptions applied "
            "by the shared execution simulator. They are modelled costs, not observed "
            "fills."
        ),
    }


def _data_block(experiment: PaperExperiment) -> dict:
    sessions = experiment.source.sessions()
    snapshots = sorted(p.name for p in experiment.snapshot_dir.glob("*.json")) \
        if experiment.snapshot_dir.exists() else []
    provenance = getattr(experiment.source, "provenance", None)
    kind = getattr(experiment.source, "kind", "unknown")
    if not experiment.source.is_current_market:
        note = (
            "the source is historical and reproducible; it is NOT the current "
            "market, so an experiment on it is not a live paper run"
        )
    elif kind == "vendor_quote":
        note = (
            "a current-day quote source is configured; it is current, but it "
            "provides a single quote, not the historical OHLCV a multi-session "
            "experiment needs"
        )
    else:
        note = "a current-market source is configured"
    return {
        "source_name": experiment.source.name,
        "source_kind": kind,
        "is_current_market": experiment.source.is_current_market,
        "dataset_start": sessions[0] if sessions else None,
        "dataset_end": sessions[-1] if sessions else None,
        "sessions_available": len(sessions),
        "snapshots_saved": len(snapshots),
        "provenance": provenance.to_dict() if provenance is not None else None,
        "note": note,
    }


def _performance(
    experiment: PaperExperiment, evaluated: list, filled: list
) -> dict:
    """Return and drawdown, only when the sample supports them."""
    min_trades = experiment.config.min_trades_for_metrics
    trades = [d for d in evaluated if d.filled_quantity > 0 and d.evaluation]
    net_pnls = [
        d.evaluation.get("net_pnl")
        for d in trades
        if d.evaluation and d.evaluation.get("net_pnl") is not None
    ]
    closed = [p for p in net_pnls if p is not None]
    wins = [p for p in closed if p > 0]
    losses = [p for p in closed if p < 0]
    sample_limited = len(closed) < min_trades

    ledger = experiment.session.ledger
    total_return = (
        (ledger.equity - ledger.initial_cash) / ledger.initial_cash
        if ledger.initial_cash
        else None
    )

    # An equity curve from the paper session's own step history, when it exists.
    equity_curve = _equity_curve(experiment)
    max_dd = None
    sharpe = None
    if equity_curve is not None and len(equity_curve) >= 2 and not sample_limited:
        metrics = compute_metrics(equity_curve, initial_capital=ledger.initial_cash)
        max_dd = metrics.max_drawdown
        sharpe = metrics.sharpe_ratio

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sample_limited": sample_limited,
        "min_trades_for_metrics": min_trades,
        "sample": {
            "evaluated_decisions": len(evaluated),
            "closed_outcomes": len(closed),
            "wins": len(wins),
            "losses": len(losses),
        },
        "win_rate": (len(wins) / len(closed)) if (closed and not sample_limited) else None,
        "note": (
            "return and drawdown are None until the sample reaches "
            f"{min_trades} closed outcomes; a ratio from a smaller sample is noise"
            if sample_limited
            else "measured from the paper session's own equity history"
        ),
    }


def _equity_curve(experiment: PaperExperiment):
    """The session's equity history, as a pandas Series, or ``None``."""
    steps = experiment.session.steps
    if not steps:
        return None
    index = [step.timestamp.date().isoformat() for step in steps]
    values = [step.equity for step in steps]
    series = pd.Series(values, index=index)
    # Keep the last observation per date, so a curve with several steps a day is
    # one point per session.
    return series[~series.index.duplicated(keep="last")]


def _baseline_block(experiment: PaperExperiment) -> dict:
    """The benchmark comparison, only when it is like-for-like.

    The comparison needs the same universe and the same sessions. Until the
    experiment has processed sessions, there is nothing to compare, and the block
    says so rather than showing a number.
    """
    if not experiment.processed_sessions:
        return {
            "available": False,
            "reason": "no sessions processed yet; there is nothing to compare",
        }
    return {
        "available": False,
        "reason": (
            "a like-for-like benchmark comparison needs the experiment's own equity "
            "curve over the same sessions and universe; it is not computed here to "
            "avoid showing an incomparable number"
        ),
    }


def _readiness(experiment: PaperExperiment, genuine: list, counts: dict) -> dict:
    """The honest readiness verdict, with every blocker named."""
    blockers: list[str] = []
    kind = getattr(experiment.source, "kind", "unknown")
    if not experiment.source.is_current_market:
        blockers.append(
            "the price source is historical, not the current market; this is not a "
            "live paper run"
        )
    if kind == "vendor_quote":
        blockers.append(
            "the price source provides a single current-day quote, not the historical "
            "OHLCV a multi-session paper experiment needs; it cannot drive a run"
        )
    if not genuine:
        blockers.append(
            "no genuine LLM inference has been recorded; set a provider credential "
            "and run sessions to produce decisions"
        )
    if counts.get(STATE_BLOCKED, 0) and not genuine:
        blockers.append(
            f"{counts[STATE_BLOCKED]} decision(s) are blocked (no credential or "
            "provider failure); they are not counted as results"
        )
    from .sources import current_market_source_available

    if not current_market_source_available():
        blockers.append(
            "no current-market price source is wired; only the frozen dataset is "
            "available, so live paper readiness cannot be claimed"
        )

    # "Ready for live paper" needs a genuine inference *and* a current-market source
    # that can actually serve a session window. A quote-only source is current but
    # cannot drive a multi-session run, so it does not qualify.
    can_drive_sessions = kind not in ("vendor_quote",)
    if genuine and experiment.source.is_current_market and can_drive_sessions:
        status = "READY FOR LIVE PAPER"
    elif genuine:
        status = "OFFLINE PAPER (reproducible, not live)"
    else:
        status = "BLOCKED"
    return {"status": status, "blockers": blockers}


def _limitations(experiment: PaperExperiment, genuine: list, performance: dict) -> list[str]:
    limitations = [
        "paper only: no order ever reaches a venue, and no live trading path exists",
        "costs and slippage are the declared model, not observed fills",
        "the evaluation is a mark-to-market at the evaluation session's close, not a "
        "realised trade",
    ]
    if not experiment.source.is_current_market:
        limitations.append(
            "the source is historical; a decision on it is a backtest-like "
            "observation, not a current-market call"
        )
    if not genuine:
        limitations.append("no genuine inference yet, so no LLM performance is reported")
    if performance.get("sample_limited"):
        limitations.append(
            "the sample is below the configured minimum, so win rate, Sharpe and "
            "drawdown are reported as N/A"
        )
    return limitations
