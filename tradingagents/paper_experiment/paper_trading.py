"""Wiring a paper experiment's recorded decisions to the Alpaca paper worker.

This module is the *only* bridge between the analysis/experiment layer and the
broker-backed execution layer. It is deliberately thin and one-directional:

    the experiment's recorded decisions  ->  TradingDecision  ->  PaperTradingWorker
                                                                 ->  AlpacaPaperBroker

It reads decisions; it never writes the internal ledger, and it never lets a
broker fill become an internal fill. The experiment remains the simulation-only
analysis layer, and the worker is the separate broker-backed path.

The bridge is **opt-in**: importing this module starts nothing, and constructing
a source places no order. A worker is built and started only by an explicit
``start`` from an authenticated caller (the dashboard or the CLI), never by the
existing ``run``/``serve`` commands.

Sizing is deliberately conservative: a decision's ``position_size`` is a fraction
of the *protocol's* capital, which is not the paper account's capital. Using it
against a small account would over-size an order, so the bridge sizes every order
from the worker's configured ``order_notional`` instead, and lets the broker's own
risk checks cap it further.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tradingagents.llm_evaluation.records import DecisionRecord
from tradingagents.llm_evaluation.replay import signal_from_record
from tradingagents.signals import SignalAction

from .broker import TradingDecision
from .experiment import STATE_BLOCKED, STATE_PENDING, DecisionState, PaperExperiment

# A signal action that maps to a broker side. HOLD (and any actionless rating) is
# not an order, so it is not yielded.
_SIDE_BY_ACTION = {SignalAction.BUY: "buy", SignalAction.SELL: "sell"}


@dataclass
class ExperimentDecisionSource:
    """Yields one :class:`TradingDecision` per *due, tradeable* experiment decision.

    Deterministic ordering — ``(decision_session, symbol)`` — so a re-run yields
    the same decisions in the same sequence. A decision is yielded only when it is

    - **tradeable** — its journal record rebuilds to a BUY or a SELL through the
      production signal contract; a HOLD, or an actionless rating, yields nothing;
    - **not blocked** — a blocked decision (no credential, provider failure) has no
      record and is never an order;
    - **due** — its ``execution_session`` is covered by the price source, so a bar
      exists to trade against. A decision whose execution session has no data yet
      is not an order;
    - **not already consumed** — the source remembers the ids it has handed out,
      and consults the worker's own durable processed-decision set (via
      ``is_consumed``), so the same decision is never offered twice;
    - **not pre-existing history** — a decision already *settled* when the source
      was built is part of the experiment's past, not a trade to place now, so it
      is not offered. A decision still awaiting execution at build time, and any
      decision recorded after it, *is* offered. See below.

    Each yielded decision carries the experiment's own ``decision_id``, which is
    what makes the worker's ``client_order_id`` stable and idempotent across a
    re-run or a restart; the worker's own durable processed-decision set is the
    second guard.

    **Freshness — why this is not a one-shot loader.** The source re-reads the
    experiment's decisions on every :meth:`next_decision`. When ``reload`` is
    supplied (the dashboard worker supplies one that re-reads the experiment state
    from disk), the source sees decisions recorded by *another* writer — an
    autonomous ``auto-advance`` progressing the experiment while the worker runs —
    not just the snapshot taken when the worker was built. That is what lets a
    worker created before any decision still discover a BUY produced later.

    **The build-time baseline — why it does not replay the backlog.** When the
    source is built, it records the decisions that are already *settled*
    (``executed``/``evaluated``/``unevaluable``) as a baseline and never offers
    them. A settled decision is the experiment's own past, and re-ordering months
    of history the moment an operator presses START is not what "run the worker"
    should mean. A decision that is still ``pending`` at build time is *not* in the
    baseline — it is awaiting execution, which is exactly what the worker is for —
    and neither is any decision recorded after the build. So a decision the
    experiment settles (via ``auto-advance``) between the worker's build and its
    first poll is still offered: the reported failure mode, a decision lost to
    auto-advance racing the worker, cannot happen for a decision that was pending
    when the worker started, nor for one produced while it runs.
    """

    experiment: PaperExperiment
    order_notional: float
    reload: Callable[[], PaperExperiment] | None = None
    is_consumed: Callable[[str], bool] | None = None
    """Whether the worker has already ordered a decision id.

    The worker's durable processed-decision set is the authority; the controller
    wires this to it. Filtering here (rather than only in the worker) keeps a
    restarted worker from re-offering, and re-skipping, every historical decision
    before it reaches a new one.
    """
    _baseline: set[str] = field(default_factory=set, init=False)
    _yielded: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._baseline = {
            d.decision_id
            for d in self.experiment.decisions.values()
            if d.state != STATE_PENDING
        }

    def _current(self) -> PaperExperiment:
        """The freshest view of the experiment, falling back to the held one.

        A reload failure (a transient unreadable state) must not crash the cycle,
        and it must not re-offer a decision: falling back to the held experiment
        only ever yields decisions this source has not yielded yet, because the
        ``_yielded`` set spans both views.
        """
        if self.reload is None:
            return self.experiment
        try:
            return self.reload()
        except Exception:  # noqa: BLE001 - a stale view is better than a crash
            return self.experiment

    def _due(self, experiment: PaperExperiment, state: DecisionState) -> bool:
        """Whether a bar exists for the decision's execution session."""
        session = state.execution_session
        if not session:
            return False
        try:
            return bool(experiment.source.covers(session))
        except Exception:  # noqa: BLE001 - an unusable source is "not due"
            return False

    def _consumed(self, decision_id: str) -> bool:
        if decision_id in self._baseline or decision_id in self._yielded:
            return True
        if self.is_consumed is None:
            return False
        try:
            return bool(self.is_consumed(decision_id))
        except Exception:  # noqa: BLE001 - an unreadable guard must not re-offer
            return False

    def _candidates(self, experiment: PaperExperiment) -> list[DecisionState]:
        """Due, tradeable, not-already-ordered decisions, in deterministic order."""
        states = sorted(
            (
                d
                for d in experiment.decisions.values()
                if d.state != STATE_BLOCKED
                and not self._consumed(d.decision_id)
                and self._due(experiment, d)
            ),
            key=lambda d: (d.decision_session, d.symbol),
        )
        return [d for d in states if self._side_for(experiment, d) is not None]

    def _side_for(self, experiment: PaperExperiment, state: DecisionState) -> str | None:
        """The broker side for a decision, or ``None`` when it is not tradeable.

        The action comes from the decision's own journal record, rebuilt through
        the production signal contract, so the side is exactly what the strategy
        decided — not a re-derivation from prose.
        """
        record = self._record_for(experiment, state)
        if record is None:
            return None
        signal = signal_from_record(record)
        if signal is None:
            return None
        return _SIDE_BY_ACTION.get(signal.action)

    def _record_for(
        self, experiment: PaperExperiment, state: DecisionState
    ) -> DecisionRecord | None:
        entry = experiment.journal.get(
            f"{experiment.experiment_id}:decision:{state.decision_id}"
        )
        if entry is None:
            return None
        payload = entry.payload.get("record")
        if not isinstance(payload, dict):
            return None
        try:
            return DecisionRecord.from_dict(payload)
        except Exception:  # noqa: BLE001 - an unreadable record is not tradeable
            return None

    @staticmethod
    def _rationale_for(state: DecisionState) -> str:
        return state.order_reason or state.blocked_reason or ""

    def next_decision(self) -> TradingDecision | None:
        experiment = self._current()
        candidates = self._candidates(experiment)
        if not candidates:
            return None
        state = candidates[0]
        self._yielded.add(state.decision_id)
        side = self._side_for(experiment, state)
        if side is None:  # pragma: no cover - _candidates already filtered this
            return None
        return TradingDecision(
            decision_id=state.decision_id,
            symbol=state.symbol,
            side=side,
            notional=self.order_notional,
            rationale=self._rationale_for(state),
        )

    def reset(self) -> None:
        """Forget which decisions were yielded, so a fresh pass re-reads them.

        The worker's own durable processed-decision set still refuses a decision it
        has already ordered, so a reset re-offers only decisions that were never
        ordered.
        """
        self._yielded = set()
