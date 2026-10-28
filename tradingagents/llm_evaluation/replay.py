"""Replay recorded decisions through the frozen engine. No model is ever called.

A :class:`DecisionRecordSet` is turned into a callable the existing
``PortfolioBacktester`` already accepts (``Strategy = Callable[[StrategyContext],
TradingSignal | None]``). Nothing in the engine changes: this is an adapter that
sits where a strategy would, so a recorded LLM decision travels the same
decision -> risk -> order -> fill -> ledger path as any other strategy.

Three guarantees are enforced here rather than assumed:

1. **No model call.** The adapter holds only records. A guard replaces the LLM
   client factory with one that raises, so a replay that somehow reached for a
   model would fail loudly instead of making a paid call.
2. **No fabricated signal.** A session with no record yields ``None`` -- the
   engine's own "no decision this session". A record whose ticker is outside the
   feed is still emitted, so the engine records it as an ``UNKNOWN_SYMBOL``
   rejection; the adapter does not silently drop it.
3. **The window is the same window.** Every record carries the hash of the data
   window it saw. The adapter recomputes that hash from the session-bounded feed
   and refuses a mismatch, so a replay cannot quietly run on different data.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass

from tradingagents.signals import (
    OrderIntent,
    SignalAction,
    SignalTimeInForce,
    TradingSignal,
    action_for_rating,
    signal_from_decision,
)

from .records import DecisionRecord, DecisionRecordSet

# The rating vocabulary maps to actions exactly as the production signal path
# does. Kept as a re-export reference so a change to the contract is a change
# here too, not a silent divergence.
_ACTION_BY_NAME = {action.value: action for action in SignalAction}


class ReplayError(ValueError):
    """A replay could not proceed: a data mismatch or a malformed record."""


def window_hash(frame) -> str:
    """A stable SHA-256 of a price window, independent of pandas internals.

    Hashes the symbol, the session dates and each OHLC value at full precision,
    so two frames agree only if they carry the same bars. Used to prove a replay
    saw the same data the model did.
    """
    digest = hashlib.sha256()
    if frame is None or len(frame) == 0:
        return digest.hexdigest()
    for index, row in frame.iterrows():
        digest.update(str(index)[:10].encode())
        for column in ("open", "high", "low", "close"):
            digest.update(f"{column}={float(row[column]):.10f}".encode())
    return digest.hexdigest()


@contextlib.contextmanager
def no_llm_calls():
    """Fail loudly if any code path tries to build an LLM client.

    Patching the factory rather than trusting the adapter not to import it means
    a future edit that reaches for a model inside a replay is caught by a test,
    not by an unexpected invoice.
    """
    from tradingagents.llm_clients import factory

    def _refuse(*args, **kwargs):
        raise ReplayError(
            "replay must not create an LLM client; a model call was attempted"
        )

    original = factory.create_llm_client
    factory.create_llm_client = _refuse
    try:
        yield
    finally:
        factory.create_llm_client = original


def signal_from_record(record: DecisionRecord) -> TradingSignal | None:
    """Rebuild a validated signal from a record, through the production contract.

    ``signal_from_decision`` is the same function the live path uses, so a
    replayed signal is subject to every validation a live one is. A record whose
    rating has no action (``REVIEW``, unknown) yields ``None`` -- a decision
    nobody could read is not a trade, exactly as in production.
    """
    action = _ACTION_BY_NAME.get(record.action)
    if action is None:  # pragma: no cover - action is validated at construction
        raise ReplayError(f"record {record.decision_id!r} has no action")
    # The record states both a rating and the action it maps to. The constructor
    # already refuses a mismatch; this repeats the check because a record built
    # by an older schema could still slip through, and the signal path follows
    # the rating, not the stored action.
    derived = action_for_rating(record.rating)
    if derived is None:
        # A rating with no action (REVIEW, or unknown) is a decision nobody could
        # read. It is not a trade, exactly as in production: no signal.
        return None
    if derived is not action:
        raise ReplayError(
            f"record {record.decision_id!r} has action {record.action!r} but rating "
            f"{record.rating!r}, which maps to {derived.value!r}"
        )
    return signal_from_decision(
        symbol=record.ticker,
        rating=record.rating,
        as_of_date=record.as_of_date,
        timestamp=_parse_timestamp(record.signal_timestamp),
        entry_price=None,
        stop_loss=record.stop_loss,
        take_profit=record.take_profit,
        position_size=record.position_size,
        holding_period_days=record.holding_period_days,
        confidence=record.confidence,
        order_intent=OrderIntent(record.order_intent),
        limit_price=record.limit_price,
        time_in_force=SignalTimeInForce(record.time_in_force),
        source="llm_portfolio_manager",
        reasoning=record.reasoning_excerpt,
    )


def _parse_timestamp(value: str):
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReplayError(f"record has an unparseable signal_timestamp {value!r}") from exc


@dataclass
class ReplayStats:
    """What a replay did, for the artifact and for tests."""

    sessions_seen: int = 0
    sessions_with_record: int = 0
    signals_emitted: int = 0
    signals_skipped_review: int = 0
    window_hash_mismatches: int = 0
    config_hash_mismatches: int = 0
    records_used: int = 0


class ReplayStrategy:
    """A strategy callable that replays recorded decisions. Calls no model."""

    def __init__(
        self,
        record_set: DecisionRecordSet,
        *,
        strict_windows: bool = True,
        expected_config_hash: str | None = None,
    ) -> None:
        self.record_set = record_set
        self.strict_windows = strict_windows
        # When set, every record's config_hash must equal it: a record captured
        # under a different protocol must not be replayed as if it were this
        # one. None means the caller does not assert a configuration.
        self.expected_config_hash = expected_config_hash
        self.stats = ReplayStats()
        self._by_session: dict[str, list[DecisionRecord]] = {}
        for record in record_set.decisions:
            self._by_session.setdefault(record.as_of_date, []).append(record)
        # Deterministic order within a session, so two replays emit identically.
        for records in self._by_session.values():
            records.sort(key=lambda r: r.decision_id)

    @property
    def name(self) -> str:
        return f"llm_replay:{self.record_set.strategy_id}"

    def __call__(self, context) -> TradingSignal | None:
        self.stats.sessions_seen += 1
        records = self._by_session.get(context.session)
        if not records:
            return None
        self.stats.sessions_with_record += 1

        # The engine takes one signal per session. The production graph makes one
        # decision per ticker per date, so the honest mapping is one decision per
        # session; more than one is a protocol error, not something to pick from.
        if len(records) > 1:
            raise ReplayError(
                f"session {context.session} has {len(records)} decisions; the engine "
                "takes one signal per session and choosing among them would invent "
                "a priority the graph does not have"
            )
        record = records[0]
        # A record captured under a different configuration must not be replayed
        # as if it were this one. Checked before the window, so a config mismatch
        # is reported as itself rather than as a data mismatch.
        if (
            self.expected_config_hash is not None
            and record.config_hash != self.expected_config_hash
        ):
            self.stats.config_hash_mismatches += 1
            raise ReplayError(
                f"replay config mismatch for {record.decision_id}: the record was "
                f"captured under config {record.config_hash[:12]}…, the replay expects "
                f"{self.expected_config_hash[:12]}…"
            )
        # A ticker the feed does not hold cannot be window-verified, and the
        # engine already has a guard for it. Emit the signal and let that guard
        # reject it, so the rejection is recorded rather than turned into a
        # crash here. Verifying a symbol we cannot price would raise on the
        # feed, which is the wrong failure for an unknown ticker.
        if record.ticker in context.symbols:
            self._verify_window(context, record)

        signal = signal_from_record(record)
        if signal is None:
            self.stats.signals_skipped_review += 1
            return None
        self.stats.signals_emitted += 1
        self.stats.records_used += 1
        return signal

    def _verify_window(self, context, record: DecisionRecord) -> None:
        """Refuse a replay whose session data is not the data the model saw."""
        frame = context.history(record.ticker)
        if frame is None or len(frame) == 0:
            actual = hashlib.sha256().hexdigest()
            count = 0
        else:
            actual = window_hash(frame)
            count = len(frame)
        # The record hashed the window it saw; the replay recomputes it. A
        # mismatch means the replay is running on different data, which would
        # make the result a fiction.
        if self.strict_windows and (actual != record.data_window_hash or count != record.bar_count):
            self.stats.window_hash_mismatches += 1
            raise ReplayError(
                f"replay window mismatch for {record.decision_id}: the model saw "
                f"{record.bar_count} bars hashing {record.data_window_hash[:12]}…, "
                f"the replay has {count} bars hashing {actual[:12]}…"
            )
