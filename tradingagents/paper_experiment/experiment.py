"""The prospective paper experiment: run sessions, record decisions, evaluate later.

This is the orchestrator that ties the existing pieces together without
duplicating any of them. It reuses, rather than reimplements:

- :class:`~tradingagents.execution.paper.PaperSession` for the book, the risk
  check, the simulated fill and the ledger;
- :class:`~tradingagents.llm_evaluation.snapshot.PriceSnapshot` for the
  reproducible input window;
- :class:`~tradingagents.llm_evaluation.records.DecisionRecord` for the decision;
- :class:`~tradingagents.llm_evaluation.capture.build_record` for validation;
- the reference baseline for the benchmark column.

The one cycle, stated once so the timing cannot drift:

1. **Session T is decided.** The window is every bar up to and including T. The
   model (or a blocked marker) produces a decision dated T.
2. **Execution is at T+1's open.** A decision from T's close cannot fill at T's
   close; it fills at the next session's open. This is the look-ahead guard, and
   it is the same rule the backtester applies.
3. **Evaluation is at T+1+lag.** The outcome is measured once enough later bars
   exist, so no evaluation uses a price that did not exist at decision time.

Three timestamps are therefore kept apart on every decision: ``decision_session``
(T), ``execution_session`` (T+1) and ``evaluation_session`` (T+1+lag). Conflating
them is how a look-ahead result gets reported as real.

The inference seam is a callable, so the offline test suite drives the whole
cycle from a deterministic stub while the production path would use the real
provider. A stub decision is labelled ``synthetic_fixture`` and can never be
counted as a genuine inference, no matter what the caller passes.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tradingagents.execution.interface import ExecutionMode, resolve_mode
from tradingagents.execution.paper import PaperSession, PaperStateError, Quote, new_session
from tradingagents.llm_evaluation.capture import CaptureError, build_record
from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL
from tradingagents.llm_evaluation.records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_SYNTHETIC_FIXTURE,
    DecisionRecord,
    assert_secret_free,
)
from tradingagents.llm_evaluation.snapshot import PriceSnapshot, SnapshotError
from tradingagents.signals import action_for_rating

from .config import ExperimentConfig
from .journal import (
    KIND_DECISION_BLOCKED,
    KIND_DECISION_EVALUATED,
    KIND_DECISION_RECORDED,
    KIND_EXPERIMENT_CREATED,
    KIND_ORDER_EXECUTED,
    KIND_SESSION_PROCESSED,
    ExperimentJournal,
)
from .sources import PriceSource, PriceSourceError

PAPER_EXPERIMENT_SCHEMA_VERSION = 1
"""The version of the experiment's state file (distinct from the journal's)."""

# Decision states. The vocabulary is closed so a report cannot invent a state.
STATE_PENDING = "pending"
"""A decision was recorded but its execution session has not been processed yet."""
STATE_EXECUTED = "executed"
"""The decision was carried to its execution session and settled there.

*Settled* means the order was submitted to the simulator, or the decision was a
HOLD, or risk refused it before an order existed — see ``order_created`` and
``order_status``. It does **not** mean the order filled: a HOLD settles here with
no order at all. Read the order funnel in the report (``orders``) for that.
"""
STATE_EVALUATED = "evaluated"
"""Enough later bars exist and the outcome has been measured."""
STATE_BLOCKED = "blocked"
"""The decision could not be produced (no credential, provider failure)."""
STATE_UNEVALUABLE = "unevaluable"
"""The decision was executed but cannot be evaluated (missing later data)."""


class ExperimentError(RuntimeError):
    """The experiment was driven in a way it cannot support."""


class InferenceUnavailable(RuntimeError):
    """No genuine inference is possible: the provider is unconfigured or failed.

    Raised by an inference callable so the caller records a *blocked* decision
    rather than substituting a fabricated one. The message never carries a
    credential.
    """


class InferenceFn(Protocol):
    """Produces a decision's rating and rationale for a session, or raises.

    A real implementation calls the provider with the session's point-in-time
    price window. A stub returns a fixed decision and must declare
    ``genuine = False`` so the record is labelled a fixture.

    ``window`` is the source's point-in-time slice for the session: every bar up
    to and including the decision session, and never a later one. It is passed
    because a directional decision taken with no data is not a decision; the
    experiment already builds this window to anchor the record, so the model is
    shown the same input the record attests to.
    """

    genuine: bool

    def __call__(
        self, *, symbol: str, session: str, window=None
    ) -> tuple[str, str]:  # pragma: no cover
        ...


@dataclass
class SessionOutcome:
    """What processing one session produced, for the journal and the caller."""

    session: str
    processed: bool
    decision_id: str | None = None
    decision_state: str | None = None
    blocked_reason: str | None = None
    execution_session: str | None = None
    order_status: str | None = None
    filled_quantity: float = 0.0
    fill_price: float | None = None
    fees: float = 0.0
    snapshot_id: str | None = None
    evaluated: tuple[str, ...] = ()


@dataclass
class DecisionState:
    """The experiment's view of one decision, from record to evaluation."""

    decision_id: str
    symbol: str
    decision_session: str
    execution_session: str
    evaluation_session: str | None
    state: str
    source: str
    snapshot_id: str | None = None
    data_window_hash: str | None = None
    signal_id: str | None = None
    order_status: str | None = None
    order_created: bool = False
    order_reason: str | None = None
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    fees: float = 0.0
    evaluation: dict | None = None
    blocked_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "decision_session": self.decision_session,
            "execution_session": self.execution_session,
            "evaluation_session": self.evaluation_session,
            "state": self.state,
            "source": self.source,
            "snapshot_id": self.snapshot_id,
            "data_window_hash": self.data_window_hash,
            "signal_id": self.signal_id,
            "order_status": self.order_status,
            "order_created": self.order_created,
            "order_reason": self.order_reason,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "fees": self.fees,
            "evaluation": self.evaluation,
            "blocked_reason": self.blocked_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DecisionState:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class SessionQuoteSource:
    """A quote callable the experiment sets to the current execution session.

    ``PaperSession.on_signal`` reads its price from a callable, by design: the
    session owns no market data. The experiment is the caller that does, so it
    holds one of these, sets it to the exact bar it means to fill against, and
    hands the callable to the session. The session can therefore only ever be
    filled at a price the experiment explicitly placed there.
    """

    def __init__(self) -> None:
        self._quote: Quote | None = None

    def set(self, quote: Quote) -> None:
        self._quote = quote

    def clear(self) -> None:
        self._quote = None

    def __call__(self, symbol: str) -> Quote:
        if self._quote is None:
            raise ExperimentError(
                f"no execution quote was set for {symbol!r}; the experiment must set "
                "the session's open before driving a signal"
            )
        if self._quote.symbol != symbol:
            raise ExperimentError(
                f"the execution quote is for {self._quote.symbol!r}, not {symbol!r}"
            )
        return self._quote


class PaperExperiment:
    """A resumable, prospective paper experiment over one or more symbols.

    The experiment owns a :class:`PaperSession` (the book) and an
    :class:`ExperimentJournal` (the history). Both are persisted; the journal is
    append-only and hash-chained, the session state is the recovery file.
    """

    def __init__(
        self,
        *,
        source: PriceSource,
        config: ExperimentConfig,
        root: str | Path,
        inference: InferenceFn | None = None,
        session: PaperSession | None = None,
        journal: ExperimentJournal | None = None,
        decisions: dict[str, DecisionState] | None = None,
        processed_sessions: set[str] | None = None,
        experiment_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.root = Path(root)
        self.inference = inference
        self.mode = resolve_mode(ExecutionMode.PAPER)
        self.experiment_id = experiment_id or self._make_experiment_id()
        self.created_at = created_at or datetime.now(UTC).isoformat()

        self.journal_path = self.root / "journal.json"
        self.state_path = self.root / "session_state.json"
        self.snapshot_dir = self.root / "snapshots"

        self.journal = journal if journal is not None else ExperimentJournal(self.journal_path)
        self.decisions: dict[str, DecisionState] = decisions or {}
        self.processed_sessions: set[str] = processed_sessions or set()
        self._quote_source = SessionQuoteSource()

        if session is not None:
            self.session = session
            # A restored session must read quotes from this experiment's source,
            # not the throwaway callable it was reconstructed with.
            self.session.quotes = self._quote_source
        else:
            self.session = new_session(
                initial_cash=config.initial_cash,
                quotes=self._quote_source,
                limits=_limits_from(config),
                costs=_costs_from(config),
                max_volume_participation=config.max_volume_participation,
            )
            self.session.session_id = self.experiment_id

    # -- identity ---------------------------------------------------------

    def _make_experiment_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.config.strategy_id}-{self.config.config_hash()[:8]}-{stamp}"

    # -- persistence ------------------------------------------------------

    def _atomic_write(self, path: Path, text: str) -> None:
        assert_secret_free(text, path=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def state(self) -> dict:
        """The complete, JSON-safe experiment state, for restart recovery.

        Distinct from the report: this holds the paper session state, every
        decision's lifecycle and the processed-session set, so a restart resumes
        without reprocessing a session or duplicating a fill.
        """
        return {
            "schema_version": PAPER_EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash(),
            "source": {
                "name": self.source.name,
                "kind": getattr(self.source, "kind", "unknown"),
                "is_current_market": self.source.is_current_market,
                "provenance": _provenance_of(self.source),
                "restore": _source_restore_payload(self.source),
            },
            "processed_sessions": sorted(self.processed_sessions),
            "decisions": {k: v.to_dict() for k, v in sorted(self.decisions.items())},
            "session": self.session.state(),
        }

    def save_state(self) -> Path:
        """Persist the experiment state atomically."""
        self._atomic_write(self.state_path, json.dumps(self.state(), indent=2) + "\n")
        return self.state_path

    @classmethod
    def _restore_source(cls, state: dict) -> PriceSource:
        """Rebuild the price source from the bars persisted in the state.

        Raises :class:`ExperimentError` when the state carries no restorable
        bars, so a resume never silently continues on a different dataset than
        the run started on.
        """
        from .sources import FramePriceSource, PriceSourceError

        payload = (state.get("source") or {}).get("restore")
        if not payload:
            raise ExperimentError(
                "this experiment's state carries no restorable price bars; pass the "
                "source explicitly to resume (a quote-only source cannot be persisted)"
            )
        try:
            return FramePriceSource.from_dict(payload)
        except (PriceSourceError, KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(
                f"the persisted price source could not be restored: {exc}"
            ) from exc

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        source: PriceSource | None = None,
        inference: InferenceFn | None = None,
    ) -> PaperExperiment:
        """Restore an experiment from ``root``, verifying the journal.

        When ``source`` is omitted, the source is rebuilt from the bars persisted
        in the state — a vendor source then resumes on the *same* data without a
        second fetch (important on a quota-limited vendor). A caller that passes
        ``source`` explicitly takes responsibility for supplying equivalent data;
        the provenance recorded in the state is still the one the run started on.
        """
        root = Path(root)
        state_path = root / "session_state.json"
        if not state_path.exists():
            raise ExperimentError(f"no experiment state at {state_path}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentError(f"experiment state {state_path} is not usable: {exc}") from exc
        version = state.get("schema_version")
        if version != PAPER_EXPERIMENT_SCHEMA_VERSION:
            raise ExperimentError(
                f"unsupported experiment state schema_version {version!r}; this build "
                f"reads version {PAPER_EXPERIMENT_SCHEMA_VERSION}"
            )
        config = ExperimentConfig.from_dict(state["config"])
        if source is None:
            source = cls._restore_source(state)
        journal = ExperimentJournal.load(root / "journal.json")
        try:
            session = PaperSession.restore(state["session"], quotes=_refuse_quote)
        except PaperStateError as exc:
            raise ExperimentError(f"the paper session state could not be restored: {exc}") from exc
        decisions = {k: DecisionState.from_dict(v) for k, v in state.get("decisions", {}).items()}
        return cls(
            source=source,
            config=config,
            root=root,
            inference=inference,
            session=session,
            journal=journal,
            decisions=decisions,
            processed_sessions=set(state.get("processed_sessions", [])),
            experiment_id=state.get("experiment_id"),
            created_at=state.get("created_at"),
        )

    def ensure_created(self) -> None:
        """Record the experiment's creation in the journal exactly once."""
        self.journal.append(
            KIND_EXPERIMENT_CREATED,
            {
                "experiment_id": self.experiment_id,
                "created_at": self.created_at,
                "config": self.config.to_dict(),
                "config_hash": self.config.config_hash(),
                "source": {
                    "name": self.source.name,
                    "kind": getattr(self.source, "kind", "unknown"),
                    "is_current_market": self.source.is_current_market,
                    "dataset_start": getattr(self.source, "dataset_start", None),
                    "dataset_end": getattr(self.source, "dataset_end", None),
                    "provenance": _provenance_of(self.source),
                },
                "mode": self.mode.value,
                "schema_version": PAPER_EXPERIMENT_SCHEMA_VERSION,
            },
            event_id=f"{self.experiment_id}:created",
            timestamp=self.created_at,
        )

    # -- the cycle --------------------------------------------------------

    def process_session(self, session: str) -> SessionOutcome:
        """Process one session: settle what is due, then decide at T.

        Idempotent: a session already processed returns a no-op outcome without
        touching the book or the journal again.

        A decision is only *taken* on a cadence session (every
        ``decision_frequency_sessions`` sessions, and never beyond
        ``max_decisions``), but settlement — executing a due decision at its open
        and evaluating an outcome that is now measurable — runs on every session.
        Splitting the two is what lets a decision made at T be executed at T+1
        even when T+1 is not itself a decision session.
        """
        if session in self.processed_sessions:
            return SessionOutcome(session=session, processed=False)

        if not self.source.covers(session):
            sessions = self.source.sessions()
            raise ExperimentError(
                f"{session} is not covered by the price source {self.source.name!r} "
                f"({sessions[0]}..{sessions[-1]})"
            )

        # Execute and evaluate anything due at or before this session first, so a
        # previous decision's open exists before a new decision is taken.
        evaluated = self._settle_due(session)

        # Work any resting GTC limit against this session's bar, so a limit the
        # earlier session did not reach can fill here. This runs before the new
        # decision, and only when the symbol has a bar this session.
        self._work_resting(session)

        decision_ids: list[str] = []
        blocked_reasons: list[str] = []
        if self._is_decision_session(session):
            for symbol in self.config.symbols:
                outcome = self._decide(symbol, session)
                if outcome.decision_id:
                    decision_ids.append(outcome.decision_id)
                if outcome.blocked_reason:
                    blocked_reasons.append(outcome.blocked_reason)
            outcome = SessionOutcome(
                session=session,
                processed=True,
                decision_id=",".join(decision_ids) if decision_ids else None,
                decision_state=(
                    STATE_BLOCKED if blocked_reasons else STATE_PENDING
                ) if decision_ids else None,
                blocked_reason="; ".join(blocked_reasons) if blocked_reasons else None,
                execution_session=self._next_session(session),
            )
        else:
            outcome = SessionOutcome(session=session, processed=True)
        outcome.evaluated = evaluated

        self.processed_sessions.add(session)
        self.journal.append(
            KIND_SESSION_PROCESSED,
            {
                "session": session,
                "decision_id": outcome.decision_id,
                "decision_state": outcome.decision_state,
                "evaluated": list(evaluated),
            },
            event_id=f"{self.experiment_id}:session:{session}",
        )
        self.save_state()
        return outcome

    def _is_decision_session(self, session: str) -> bool:
        """Whether a decision is due on ``session``.

        The cadence is measured in sessions from the source's first session, so a
        resumed run computes the same cadence sessions as a fresh one — the check
        is a function of the session, not of how many have been processed.
        """
        # ``max_decisions`` counts decision *events* (one per symbol per cadence
        # session), so a multi-symbol experiment still respects the declared cap.
        decided_events = {
            d.decision_session
            for d in self.decisions.values()
            if d.source != "blocked"
        }
        if len(decided_events) * len(self.config.symbols) >= self.config.max_decisions:
            return False
        sessions = self.source.sessions()
        try:
            index = sessions.index(session)
        except ValueError:
            return False
        return index % self.config.decision_frequency_sessions == 0

    def _decide(self, symbol: str, session: str) -> SessionOutcome:
        execution_session = self._next_session(session)
        outcome = SessionOutcome(
            session=session, processed=True, execution_session=execution_session
        )

        snapshot = self._snapshot(symbol, session)
        outcome.snapshot_id = snapshot.snapshot_id if snapshot else None

        decision_id = f"{symbol}-{session}"
        if self.inference is None:
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                "no inference provider is configured; set a provider credential to "
                "produce a genuine decision",
            )
        if snapshot is None:
            # The record could not be anchored to a reproducible window, so the
            # decision would be refused below regardless. Refuse it before the
            # request: asking the model to decide with no data is the defect this
            # seam exists to avoid, and it must not spend a paid call either.
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                "no reproducible price window for this session, so no decision can "
                "be anchored; the model is not asked to decide blind",
            )
        try:
            rating, reasoning = self.inference(
                symbol=symbol,
                session=session,
                window=snapshot.to_frame(),
            )
        except InferenceUnavailable as exc:
            return self._block(outcome, decision_id, symbol, session, execution_session, str(exc))
        except Exception as exc:  # noqa: BLE001 - a provider failure is recorded, not hidden
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                f"provider failure: {type(exc).__name__}: {exc}",
            )

        action = action_for_rating(rating)
        if action is None:
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                f"rating {rating!r} has no tradeable action",
            )

        genuine = bool(getattr(self.inference, "genuine", False))
        source = SOURCE_LLM_INFERENCE if genuine else SOURCE_SYNTHETIC_FIXTURE
        frame = snapshot.to_frame() if snapshot is not None else None
        try:
            record, window_reproducible = build_record(
                ticker=symbol,
                session=session,
                rating=rating,
                action=action,
                frame=frame,
                protocol=DEFAULT_PROTOCOL,
                source=source,
                provider=self.config.provider,
                model_name=self.config.model,
                reasoning_excerpt=reasoning,
                snapshot_id=snapshot.snapshot_id if snapshot else None,
                captured_at=self._session_timestamp(session).isoformat(),
            )
        except CaptureError as exc:
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                f"the decision could not be captured: {exc}",
            )

        # Refuse a record that is not anchored to a reproducible window: such a
        # record cannot be evaluated honestly later.
        if not window_reproducible or snapshot is None or not snapshot.matches_record(record):
            return self._block(
                outcome,
                decision_id,
                symbol,
                session,
                execution_session,
                "the decision's price window is not reproducible against a named "
                "snapshot, so it cannot anchor an evaluation",
            )

        state = DecisionState(
            decision_id=decision_id,
            symbol=symbol,
            decision_session=session,
            execution_session=execution_session or session,
            evaluation_session=self._evaluation_session(execution_session or session),
            state=STATE_PENDING,
            source=source,
            snapshot_id=snapshot.snapshot_id,
            data_window_hash=record.data_window_hash,
        )
        self.decisions[decision_id] = state
        # The whole record is journalled, so the decision is durable and a later
        # execution rebuilds it exactly rather than reconstructing it from prose.
        self.journal.append(
            KIND_DECISION_RECORDED,
            {
                "decision_id": decision_id,
                "symbol": symbol,
                "decision_session": session,
                "execution_session": state.execution_session,
                "evaluation_session": state.evaluation_session,
                "source": source,
                "is_genuine": source == SOURCE_LLM_INFERENCE,
                "snapshot_id": snapshot.snapshot_id,
                "record": record.to_dict(),
            },
            event_id=f"{self.experiment_id}:decision:{decision_id}",
        )
        outcome.decision_id = decision_id
        outcome.decision_state = STATE_PENDING
        return outcome

    def _block(
        self,
        outcome: SessionOutcome,
        decision_id: str,
        symbol: str,
        session: str,
        execution_session: str | None,
        reason: str,
    ) -> SessionOutcome:
        state = DecisionState(
            decision_id=decision_id,
            symbol=symbol,
            decision_session=session,
            execution_session=execution_session or session,
            evaluation_session=None,
            state=STATE_BLOCKED,
            source="blocked",
            blocked_reason=reason,
        )
        self.decisions[decision_id] = state
        self.journal.append(
            KIND_DECISION_BLOCKED,
            {
                "decision_id": decision_id,
                "symbol": symbol,
                "decision_session": session,
                "reason": reason,
            },
            event_id=f"{self.experiment_id}:blocked:{decision_id}",
        )
        outcome.decision_id = decision_id
        outcome.decision_state = STATE_BLOCKED
        outcome.blocked_reason = reason
        return outcome

    def _settle_due(self, session: str) -> tuple[str, ...]:
        """Execute pending decisions due at or before ``session``; evaluate due ones.

        A decision always executes at its own recorded ``execution_session`` (T+1's
        open), never at whatever session happens to be current: using ``<=`` lets a
        gap in processing settle a due decision without moving its execution to a
        later bar, which would break the decision/execution timing rule.
        """
        evaluated: list[str] = []
        for state in list(self.decisions.values()):
            if state.state == STATE_PENDING and state.execution_session <= session:
                self._execute(state, state.execution_session)
        for state in list(self.decisions.values()):
            if (
                state.state == STATE_EXECUTED
                and state.evaluation_session is not None
                and state.evaluation_session <= session
            ):
                self._evaluate(state)
                evaluated.append(state.decision_id)
        return tuple(evaluated)

    def _work_resting(self, session: str) -> None:
        """Work resting GTC limits against ``session``'s bar, one symbol at a time.

        ``work_resting`` reads its prices from the session's quote callable, so the
        experiment sets the exact bar it means each symbol to be worked against.
        The session is marked once with every symbol's bar, so the day's clock
        rolls once, and the queue is worked once. A symbol with no bar this session
        is skipped rather than worked against a stale price.
        """
        if not self.session.resting:
            return
        resting_symbols = {order.symbol for order in self.session.resting}
        quotes: list[Quote] = []
        for symbol in resting_symbols:
            bar = self.source.bars(symbol, session)
            if bar is None:
                continue
            quotes.append(
                Quote(
                    symbol=symbol,
                    price=bar.open,
                    timestamp=self._session_timestamp(session),
                    is_open=True,
                    low=bar.low,
                    high=bar.high,
                    average_daily_volume=bar.volume,
                )
            )
        if not quotes:
            return
        self.session.mark(quotes)
        # A dict lookup keyed by symbol, so each order is worked against its own
        # bar rather than whichever quote happened to be set last.
        by_symbol = {quote.symbol: quote for quote in quotes}

        def quote_for(symbol: str) -> Quote:
            if symbol not in by_symbol:
                raise ExperimentError(
                    f"no bar for {symbol!r} at session {session}; a resting order for "
                    "it must not be worked against another symbol's price"
                )
            return by_symbol[symbol]

        self.session.quotes = quote_for
        try:
            steps = self.session.work_resting()
        finally:
            self.session.quotes = self._quote_source
        for step in steps:
            self.journal.append(
                KIND_ORDER_EXECUTED,
                {
                    "decision_id": None,
                    "execution_session": session,
                    "symbol": step.symbol,
                    "status": step.order_status,
                    "filled_quantity": float(step.quantity),
                    "fill_price": float(step.fill_price) if step.fill_price else None,
                    "fees": float(step.fees),
                    "resting_work": True,
                    "reason": step.reason,
                },
                event_id=(
                    f"{self.experiment_id}:resting:{session}:{step.symbol}:"
                    f"{step.order_status}:{round(float(step.quantity), 6)}"
                ),
            )

    def _execute(self, state: DecisionState, session: str) -> None:
        """Execute a decision at the session's open (the look-ahead guard)."""
        symbol = state.symbol
        bar = self.source.bars(symbol, session)
        if bar is None:
            state.state = STATE_UNEVALUABLE
            state.blocked_reason = f"no bar for {symbol!r} at execution session {session}"
            self.journal.append(
                KIND_ORDER_EXECUTED,
                {
                    "decision_id": state.decision_id,
                    "execution_session": session,
                    "status": "NO_BAR",
                    "reason": state.blocked_reason,
                },
                event_id=f"{self.experiment_id}:exec:{state.decision_id}",
            )
            return

        record = self._record_for(state)
        from tradingagents.llm_evaluation.replay import signal_from_record

        signal = signal_from_record(record)
        if signal is None:
            state.state = STATE_UNEVALUABLE
            state.blocked_reason = "the record maps to no tradeable signal"
            return

        # Re-date the signal to the execution session: the decision was made from
        # T's close, the order is placed for T+1. Nothing downstream can then read
        # the decision as having filled at T.
        signal = signal.model_copy(
            update={
                "as_of_date": session,
                "timestamp": self._session_timestamp(session),
            }
        )

        quote = Quote(
            symbol=symbol,
            price=bar.open,
            timestamp=self._session_timestamp(session),
            is_open=True,
            low=bar.low,
            high=bar.high,
            average_daily_volume=bar.volume,
        )
        self._quote_source.set(quote)
        try:
            step = self.session.on_signal(signal)
        finally:
            self._quote_source.clear()

        state.state = STATE_EXECUTED
        state.signal_id = signal.signal_id
        state.order_status = step.order_status
        # An order exists only when the simulator returned a real order status.
        # A HOLD or a pre-order refusal reports ``NONE``; recording that lets the
        # report separate "settled" from "an order actually existed".
        state.order_created = step.order_status != "NONE"
        state.order_reason = step.reason or None
        state.filled_quantity = float(step.quantity)
        state.average_fill_price = float(step.fill_price) if step.fill_price else None
        state.fees = float(step.fees)
        self.journal.append(
            KIND_ORDER_EXECUTED,
            {
                "decision_id": state.decision_id,
                "execution_session": session,
                "signal_id": signal.signal_id,
                "status": step.order_status,
                "order_created": state.order_created,
                "approved": step.approved,
                "filled_quantity": state.filled_quantity,
                "fill_price": state.average_fill_price,
                "fees": state.fees,
                "reason": step.reason,
            },
            event_id=f"{self.experiment_id}:exec:{state.decision_id}",
        )

    def _evaluate(self, state: DecisionState) -> None:
        """Measure a decision's outcome from data that existed after execution."""
        symbol = state.symbol
        evaluation_session = state.evaluation_session
        if evaluation_session is None or not self.source.covers(evaluation_session):
            state.state = STATE_UNEVALUABLE
            state.blocked_reason = (
                "not enough later bars to evaluate; the decision stays pending until "
                "its evaluation session is covered"
            )
            self._record_evaluation(state)
            return

        entry_price = state.average_fill_price
        if entry_price is None or state.filled_quantity <= 0:
            # Nothing was filled (rejected, or a HOLD). There is no P&L to measure.
            state.evaluation = {
                "evaluation_session": evaluation_session,
                "filled": False,
                "note": "no fill, so there is no trade outcome to measure",
            }
            state.state = STATE_EVALUATED
            self._record_evaluation(state)
            return

        bar = self.source.bars(symbol, evaluation_session)
        if bar is None:
            state.state = STATE_UNEVALUABLE
            state.blocked_reason = f"no bar at evaluation session {evaluation_session}"
            self._record_evaluation(state)
            return
        exit_price = bar.close
        quantity = state.filled_quantity
        gross = (exit_price - entry_price) * quantity
        net = gross - state.fees
        return_pct = (exit_price - entry_price) / entry_price if entry_price else None
        state.evaluation = {
            "evaluation_session": evaluation_session,
            "entry_session": state.execution_session,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_pnl": gross,
            "fees": state.fees,
            "net_pnl": net,
            "return_pct": return_pct,
            "sessions_held": _sessions_between(
                self.source.sessions(), state.execution_session, evaluation_session
            ),
            "note": "a paper mark-to-market outcome; not a realised trade, not a claim "
            "about live performance",
        }
        state.state = STATE_EVALUATED
        self._record_evaluation(state)

    def _record_evaluation(self, state: DecisionState) -> None:
        self.journal.append(
            KIND_DECISION_EVALUATED,
            {
                "decision_id": state.decision_id,
                "evaluation_session": state.evaluation_session,
                "state": state.state,
                "evaluation": state.evaluation,
                "blocked_reason": state.blocked_reason,
            },
            event_id=f"{self.experiment_id}:evaluated:{state.decision_id}",
        )

    # -- helpers ----------------------------------------------------------

    def _record_for(self, state: DecisionState) -> DecisionRecord:
        """Rebuild the decision's record from its journal entry."""
        entry = self.journal.get(f"{self.experiment_id}:decision:{state.decision_id}")
        if entry is None:
            raise ExperimentError(
                f"no recorded decision for {state.decision_id}; the journal and the "
                "state disagree"
            )
        return DecisionRecord.from_dict(entry.payload["record"])

    def _snapshot(self, symbol: str, session: str) -> PriceSnapshot | None:
        try:
            frame = self.source.window(symbol, session)
        except PriceSourceError:
            return None
        if frame is None or len(frame) == 0:
            return None
        try:
            snapshot = PriceSnapshot.from_frame(
                symbol, frame, source=f"{self.source.name}:{symbol}"
            )
        except SnapshotError:
            return None
        with contextlib.suppress(SnapshotError):
            snapshot.save(self.snapshot_dir / f"{snapshot.snapshot_id}.json")
        return snapshot

    def _next_session(self, session: str) -> str | None:
        sessions = self.source.sessions()
        try:
            index = sessions.index(session)
        except ValueError:
            return None
        return sessions[index + 1] if index + 1 < len(sessions) else None

    def _evaluation_session(self, execution_session: str) -> str | None:
        sessions = self.source.sessions()
        try:
            index = sessions.index(execution_session)
        except ValueError:
            return None
        target = index + self.config.evaluation_lag_sessions
        return sessions[target] if target < len(sessions) else None

    def _session_timestamp(self, session: str) -> datetime:
        """A fixed UTC timestamp for a session, at its close (21:00 UTC).

        Fixed per session so a re-run produces the same signal id and the
        processed-signal set is meaningful across a restart. A source may
        override the instant (a vendor's own timezone), but it must stay a pure
        function of the session, or the signal id would stop being stable.
        """
        override = getattr(self.source, "session_timestamp", None)
        if callable(override):
            stamp = override(session)
            if stamp is not None:
                return stamp
        return datetime.fromisoformat(f"{session}T21:00:00+00:00")

    # -- introspection ----------------------------------------------------

    def counts(self) -> dict:
        """How many decisions are in each state, for the report."""
        counts = {
            STATE_PENDING: 0,
            STATE_EXECUTED: 0,
            STATE_EVALUATED: 0,
            STATE_BLOCKED: 0,
            STATE_UNEVALUABLE: 0,
        }
        for state in self.decisions.values():
            counts[state.state] = counts.get(state.state, 0) + 1
        return counts


def _refuse_quote(symbol: str) -> Quote:
    raise ExperimentError(
        f"a restored experiment supplies quotes explicitly; {symbol!r} was requested "
        "without a session"
    )


def _provenance_of(source) -> dict | None:
    """The source's declared provenance, JSON-safe, or ``None`` if it has none."""
    provenance = getattr(source, "provenance", None)
    if provenance is None:
        return None
    return provenance.to_dict()


def _source_restore_payload(source) -> dict | None:
    """The source's restorable bars, when it can serialize them.

    A frame-backed source (a static fixture or a vendor daily source) persists
    its validated bars, so a restart rebuilds the *same* data without a second
    vendor call. A source that cannot serialize (the frozen dataset, which
    reloads locally, or a quote-only source) returns ``None``; the state then
    records the provenance but no bars, and the CLI resumes the frozen dataset
    from disk (checking its window) or refuses a source that cannot be persisted.
    """
    to_dict = getattr(source, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        return to_dict()
    except Exception:  # noqa: BLE001 - a non-serializable source is not fatal
        return None


def _limits_from(config: ExperimentConfig):
    from tradingagents.risk import RiskLimits

    return RiskLimits(**config.limits) if config.limits else RiskLimits()



def _costs_from(config: ExperimentConfig):
    from tradingagents.execution.costs import CostModel

    return CostModel(**config.costs) if config.costs else CostModel()


def _sessions_between(sessions: tuple[str, ...], start: str, end: str) -> int | None:
    try:
        return sessions.index(end) - sessions.index(start)
    except ValueError:
        return None
