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

from dataclasses import dataclass, field

from tradingagents.llm_evaluation.records import DecisionRecord
from tradingagents.llm_evaluation.replay import signal_from_record
from tradingagents.signals import SignalAction

from .broker import TradingDecision
from .experiment import STATE_PENDING, DecisionState, PaperExperiment

# A signal action that maps to a broker side. HOLD (and any actionless rating) is
# not an order, so it is not yielded.
_SIDE_BY_ACTION = {SignalAction.BUY: "buy", SignalAction.SELL: "sell"}


@dataclass
class ExperimentDecisionSource:
    """Yields one :class:`TradingDecision` per *pending* experiment decision.

    Deterministic ordering — ``(decision_session, symbol)`` — so a re-run yields
    the same decisions in the same sequence. Only decisions still awaiting
    execution (state ``pending``) whose signal maps to a BUY or a SELL are
    yielded; a HOLD, a blocked decision, or an already-settled one is not an
    order and is skipped. Each yielded decision carries the experiment's own
    ``decision_id``, which is what makes the worker's ``client_order_id`` stable
    and idempotent across a re-run or a restart.
    """

    experiment: PaperExperiment
    order_notional: float
    _queue: list[TradingDecision] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)

    def _load(self) -> None:
        if self._loaded:
            return
        pending: list[DecisionState] = sorted(
            (d for d in self.experiment.decisions.values() if d.state == STATE_PENDING),
            key=lambda d: (d.decision_session, d.symbol),
        )
        for state in pending:
            side = self._side_for(state)
            if side is None:
                continue
            self._queue.append(
                TradingDecision(
                    decision_id=state.decision_id,
                    symbol=state.symbol,
                    side=side,
                    notional=self.order_notional,
                    rationale=self._rationale_for(state),
                )
            )
        self._loaded = True

    def _side_for(self, state: DecisionState) -> str | None:
        """The broker side for a pending decision, or ``None`` when it is a HOLD.

        The action comes from the decision's own journal record, rebuilt through
        the production signal contract, so the side is exactly what the strategy
        decided — not a re-derivation from prose.
        """
        record = self._record_for(state)
        if record is None:
            return None
        signal = signal_from_record(record)
        if signal is None:
            return None
        return _SIDE_BY_ACTION.get(signal.action)

    def _record_for(self, state: DecisionState) -> DecisionRecord | None:
        entry = self.experiment.journal.get(
            f"{self.experiment.experiment_id}:decision:{state.decision_id}"
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
        self._load()
        if not self._queue:
            return None
        return self._queue.pop(0)

    def reset(self) -> None:
        """Re-read the experiment's decisions (e.g. after new sessions run)."""
        self._queue = []
        self._loaded = False
