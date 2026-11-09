"""A current-market decision source: strategy over live Alpaca data, not a session.

This is the independent, second :class:`~.broker.trading.DecisionSource`. It is
deliberately **not** an extension of :class:`~.paper_trading.ExperimentDecisionSource`:
that source replays a historical experiment's *recorded* decisions and is bounded
by the experiment's session set (so it reports ``HISTORY_EXHAUSTED`` once every
session is consumed). This source instead:

1. asks Alpaca whether the market is open (a closed market yields no decision);
2. fetches the *current* daily OHLCV window for the symbol from Alpaca;
3. runs the **same** strategy/inference seam the historical path uses
   (``ProviderInference`` / ``render_strategy_prompt``) over that window;
4. turns the rating into one BUY/SELL/HOLD, or an honest no-trade;
5. yields at most one :class:`TradingDecision`, at most once per symbol per
   trading session.

So a fully-consumed historical history no longer blocks trading: the two sources
are separate objects on separate data, and the worker is wired to one of them.

Cadence and idempotency
-----------------------

A decision is taken **once per symbol per trading session**, keyed on the market
clock's own session date. The ``decision_id`` is ``current-<symbol>-<session>``,
so it is deterministic: the same session yields the same id, a new session yields
a new one, and a restart within a session re-yields the *same* id rather than a
second decision. The worker's durable ``processed_decisions`` set and the broker's
``client_order_id`` idempotency are the second and third guards, so a restart
cannot place a duplicate order.

The evaluation outcome for a session is persisted in the source's **own** state
file (never the experiment's ``session_state.json``/``journal.json``), so a
restart replays the same decision without a second model call — and without a
chance of the model returning a different side for the same session.

No fabrication
--------------

The source never invents a size, a stop or a target: the notional is the worker's
own configured target (exactly as the historical bridge does), and ``stop_loss``/
``take_profit`` are ``None`` because the strategy proposed none. A rating with no
tradeable action, or a strategy that cannot be reached, yields *no* decision with
a named reason code — never a fake signal.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.signals import SignalAction, action_for_rating

from .broker.trading import (
    NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_NO_SIGNAL,
    NO_TRADE_STRATEGY_UNAVAILABLE,
    TradingDecision,
)
from .current_market_data import CurrentMarketData, CurrentMarketDataError
from .market_hours import MARKET_OPEN, MarketHoursError, MarketHoursGate

CURRENT_SOURCE_SCHEMA_VERSION = 1
"""The on-disk current-source state format."""

_SIDE_BY_ACTION = {SignalAction.BUY: "buy", SignalAction.SELL: "sell"}

_REASONING_EXCERPT = 400
"""How much of the model's own words to keep for a HOLD reason, bounded so the
statistics block stays small. The full reasoning is still persisted per decision;
this is only the summary excerpt the dashboard shows next to the HOLD count."""

_EMPTY_STATS: dict[str, Any] = {
    "evaluations": 0,
    "rating_counts": {},
    "side_counts": {"buy": 0, "sell": 0, "hold": 0},
    "last_rating": None,
    "last_side": None,
    "last_hold_reason": None,
    "last_symbol": None,
    "last_session": None,
    "first_evaluated_at": None,
    "last_evaluated_at": None,
}
"""The decision statistics the source keeps, one genuine evaluation at a time.

Persisted so a restart does not reset the tally, and additive: an older state file
without the block reads back as zeros rather than as a fabricated count. Only a
real model call bumps these — a replay, a skip or a market-closed poll does not,
so "evaluations" means "the strategy was asked", never "the worker polled".
"""


def _fresh_stats() -> dict[str, Any]:
    """A new tally with its nested maps copied, so instances never share one."""
    return {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in _EMPTY_STATS.items()
    }


class CurrentSourceError(RuntimeError):
    """The current-market source was driven in a way it cannot support."""


@dataclass
class _SymbolDecision:
    """The decision taken for one symbol in one session, persisted verbatim."""

    symbol: str
    session: str
    decision_id: str
    side: str | None
    rating: str
    reasoning: str
    window_hash: str | None
    bar_count: int | None
    evaluated_at: str
    stop_loss: float | None = None
    """The stop the strategy proposed, when it proposed one.

    Recorded verbatim, never derived here. ``None`` means the strategy proposed
    none, and the mandatory-protection gate then refuses the trade — which is the
    correct outcome, not a defect: a position whose exit is not defined cannot
    be managed."""
    take_profit: float | None = None
    """The target the strategy proposed, when it proposed one. Same rules."""
    targets: tuple[float, ...] = ()
    """Every target the strategy supports, nearest first. Empty when it named
    one, which is then ``take_profit``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session": self.session,
            "decision_id": self.decision_id,
            "side": self.side,
            "rating": self.rating,
            "reasoning": self.reasoning,
            "window_hash": self.window_hash,
            "bar_count": self.bar_count,
            "evaluated_at": self.evaluated_at,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "targets": list(self.targets),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _SymbolDecision:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class CurrentMarketDecisionSource:
    """Yields one current-market decision per symbol per trading session.

    Satisfies the worker's :class:`~.broker.trading.DecisionSource` protocol:
    :meth:`next_decision` returns a :class:`TradingDecision` or ``None``, and the
    source never places an order itself.
    """

    symbols: tuple[str, ...]
    inference: Any
    market_data: CurrentMarketData
    market_hours: MarketHoursGate
    order_notional: float
    strategy_name: str = "current-market"
    is_consumed: Callable[[str], bool] | None = None
    """Whether the worker has already acted on a decision id (its durable set)."""
    state_path: str | os.PathLike | None = None
    """Where the source's own per-session memory lives. Never the experiment state."""

    source_kind: str = field(default="current_market", init=False)
    skip_reason: str | None = field(default=None, init=False)
    skip_code: str | None = field(default=None, init=False)
    _decisions: dict[str, _SymbolDecision] = field(default_factory=dict, init=False)
    _yielded: set[str] = field(default_factory=set, init=False)
    """The decision ids offered *this process*. A worker that holds a decision and
    has not yet persisted it is not re-offered the same id within one run, so the
    first decision is not silently replaced by an identical second one."""
    _last_market_state: str | None = field(default=None, init=False)
    _last_market_session: str | None = field(default=None, init=False)
    _stats: dict[str, Any] = field(default_factory=lambda: _fresh_stats(), init=False)
    """The lifetime decision tally, loaded from and saved to the source's own state.
    Distinct from ``_decisions`` (the per-session decision memory): this survives a
    session roll and answers "how often has the strategy chosen BUY/SELL/HOLD"."""

    def __post_init__(self) -> None:
        if not self.symbols:
            raise CurrentSourceError("a current source needs at least one symbol")
        self.symbols = tuple(s.upper() for s in self.symbols)
        self._store = Path(self.state_path) if self.state_path else None
        if self._store is not None:
            payload = self._read_payload()
            self._decisions = self._decode_decisions(payload)
            self._stats = self._decode_stats(payload)

    # -- persistence ------------------------------------------------------

    def _read_payload(self) -> dict[str, Any]:
        assert self._store is not None
        if not self._store.exists():
            return {}
        try:
            payload = json.loads(self._store.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("schema_version") != CURRENT_SOURCE_SCHEMA_VERSION:
            return {}
        return payload

    def _decode_decisions(self, payload: dict[str, Any]) -> dict[str, _SymbolDecision]:
        raw = payload.get("decisions") or {}
        out: dict[str, _SymbolDecision] = {}
        for symbol, record in raw.items():
            if isinstance(record, dict):
                with contextlib.suppress(TypeError):
                    out[symbol] = _SymbolDecision.from_dict(record)
        return out

    def _decode_stats(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The persisted tally, or empty. A malformed block reads as empty, never
        as a partially-trusted count, so a dashboard never shows a corrupted total."""
        raw = payload.get("statistics")
        stats = _fresh_stats()
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key not in stats:
                    continue
                if isinstance(stats[key], dict):
                    if isinstance(value, dict):
                        stats[key].update(value)
                elif value is not None or stats[key] is None:
                    stats[key] = value
        return stats

    def statistics(self) -> dict[str, Any]:
        """A copy of the lifetime decision tally, for a status view."""
        return {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in self._stats.items()
        }

    def _record_evaluation(self, record: _SymbolDecision) -> None:
        """Bump the tally for one genuine evaluation, then persist.

        Called only where a real model call returned a rating, so the counts mean
        "the strategy was asked and answered", never "the worker polled". The
        HOLD reason is the model's own reasoning, bounded to an excerpt, so a
        dashboard can show *why* the strategy chose to do nothing.
        """
        stats = self._stats
        stats["evaluations"] = int(stats.get("evaluations") or 0) + 1
        ratings = stats.setdefault("rating_counts", {})
        ratings[record.rating] = int(ratings.get(record.rating) or 0) + 1
        sides = stats.setdefault("side_counts", {"buy": 0, "sell": 0, "hold": 0})
        side_key = record.side or "hold"
        sides[side_key] = int(sides.get(side_key) or 0) + 1
        stats["last_rating"] = record.rating
        stats["last_side"] = record.side
        stats["last_symbol"] = record.symbol
        stats["last_session"] = record.session
        stats["last_evaluated_at"] = record.evaluated_at
        if stats.get("first_evaluated_at") is None:
            stats["first_evaluated_at"] = record.evaluated_at
        if record.side is None:
            stats["last_hold_reason"] = (record.reasoning or "")[:_REASONING_EXCERPT] or None
        else:
            stats["last_hold_reason"] = None

    def _save(self) -> None:
        if self._store is None:
            return
        payload = {
            "schema_version": CURRENT_SOURCE_SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "decisions": {k: v.to_dict() for k, v in sorted(self._decisions.items())},
            "statistics": self._stats,
        }
        self._store.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._store.parent), prefix=".current-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._store)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- introspection ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """A secret-free view of the current source, for the dashboard/status."""
        view: dict[str, Any] = {
            "decision_source_kind": self.source_kind,
            "decision_source_symbols": list(self.symbols),
            "decision_source_strategy": self.strategy_name,
            "decision_source_last_market_state": self._last_market_state,
            "decision_source_last_market_session": self._last_market_session,
            "decision_source_statistics": self.statistics(),
        }
        describe = getattr(self.market_data, "describe", None)
        if callable(describe):
            with contextlib.suppress(Exception):
                view.update(describe())
        hours_describe = getattr(self.market_hours, "describe", None)
        if callable(hours_describe):
            with contextlib.suppress(Exception):
                view.update(hours_describe())
        return view

    def last_decision_for(self, symbol: str) -> dict[str, Any] | None:
        record = self._decisions.get(symbol.upper())
        return record.to_dict() if record is not None else None

    # -- the protocol -----------------------------------------------------

    def next_decision(self) -> TradingDecision | None:
        """The next due current-market decision, or ``None`` with a reason code."""
        hours = self._market_state()
        if hours is None:
            return self._skip(
                "the market clock could not be read, so the market state is unknown",
                NO_TRADE_MARKET_DATA_UNAVAILABLE,
            )
        state, session = hours
        self._last_market_state = state
        self._last_market_session = session
        if state != MARKET_OPEN:
            return self._skip(
                f"the market is {state}; no decision is taken outside market hours",
                NO_TRADE_MARKET_CLOSED,
            )
        if session is None:
            return self._skip(
                "the market clock reported no session date, so no decision can be dated",
                NO_TRADE_MARKET_DATA_UNAVAILABLE,
            )

        # Report the most informative reason when no symbol yields a decision: a
        # data/strategy failure outranks an honest HOLD, which outranks "already
        # consumed for this session". The gate is named, never a generic message.
        fallback: tuple[str, str] | None = None
        for symbol in sorted(self.symbols):
            outcome = self._next_for_symbol(symbol, session)
            if isinstance(outcome, TradingDecision):
                self.skip_reason = None
                self.skip_code = None
                return outcome
            if outcome is not None and fallback is None:
                fallback = outcome
        if fallback is not None:
            return self._skip(*fallback)
        return self._skip(
            "every symbol already has a decision for this session; new market data "
            "is required before the next decision",
            NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
        )

    # -- internals --------------------------------------------------------

    def _market_state(self) -> tuple[str, str | None] | None:
        try:
            status = self.market_hours.status()
        except MarketHoursError:
            return None
        except Exception:  # noqa: BLE001 - any clock failure is an unknown state
            return None
        return status.state, status.session

    def _next_for_symbol(
        self, symbol: str, session: str
    ) -> TradingDecision | tuple[str, str] | None:
        """A decision, or a ``(detail, code)`` skip to fall back to, or ``None``.

        ``None`` means "nothing to report for this symbol" (already consumed); the
        caller then tries the next symbol and, if none yields a decision, reports
        its own no-new-data code.
        """
        existing = self._decisions.get(symbol)
        if existing is not None and existing.session == session:
            # Same session: replay the decision taken (never re-call the model), so
            # a restart within a session is deterministic and cannot flip a side.
            if existing.side is None:
                return (
                    f"the strategy returned {existing.rating} for {symbol} this "
                    "session; a HOLD is not an order",
                    NO_TRADE_NO_SIGNAL,
                )
            if existing.decision_id in self._yielded:
                return None  # already offered this run; the worker holds it
            if self._consumed(existing.decision_id):
                return None  # the worker already ordered it; nothing to re-offer
            self._yielded.add(existing.decision_id)
            return self._to_decision(existing)

        # A new session: evaluate the strategy on freshly fetched current data.
        try:
            window = self.market_data.fetch(symbol, session=session)
        except CurrentMarketDataError as exc:
            return (f"current market data for {symbol} is unavailable: {exc}", exc.code)
        except Exception as exc:  # noqa: BLE001 - any fetch failure is unavailable
            return (
                f"current market data for {symbol} is unavailable: {type(exc).__name__}",
                NO_TRADE_MARKET_DATA_UNAVAILABLE,
            )
        try:
            rating, reasoning = self.inference(
                symbol=symbol, session=session, window=window.frame
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure yields no decision
            return (
                f"the strategy could not produce a decision for {symbol}: "
                f"{type(exc).__name__}",
                NO_TRADE_STRATEGY_UNAVAILABLE,
            )

        action = action_for_rating(rating)
        side = _SIDE_BY_ACTION.get(action) if action is not None else None
        # The levels the strategy proposed, if it proposed any. They come from the
        # inference seam and are recorded verbatim; this source never derives one.
        # A decision with no levels is carried honestly and is refused later, by
        # the mandatory-protection gate, rather than being padded with a level
        # this source made up.
        stop_loss = getattr(self.inference, "stop_loss", None)
        take_profit = getattr(self.inference, "take_profit", None)
        targets = tuple(getattr(self.inference, "targets", ()) or ())
        record = _SymbolDecision(
            symbol=symbol,
            session=session,
            decision_id=f"current-{symbol}-{session}",
            side=side,
            rating=rating,
            reasoning=reasoning,
            window_hash=window.content_hash,
            bar_count=window.bar_count,
            evaluated_at=datetime.now(UTC).isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            targets=targets,
        )
        self._decisions[symbol] = record
        self._record_evaluation(record)
        self._save()
        if side is None:
            return (
                f"the strategy returned {rating} for {symbol}; a HOLD is not an order",
                NO_TRADE_NO_SIGNAL,
            )
        if self._consumed(record.decision_id):
            return None  # the worker already ordered this id; do not re-offer
        self._yielded.add(record.decision_id)
        return self._to_decision(record)

    def _consumed(self, decision_id: str) -> bool:
        if self.is_consumed is None:
            return False
        try:
            return bool(self.is_consumed(decision_id))
        except Exception:  # noqa: BLE001 - an unreadable guard must not re-offer
            return False

    def _to_decision(self, record: _SymbolDecision) -> TradingDecision:
        assert record.side is not None  # only called for a tradeable side
        return TradingDecision(
            decision_id=record.decision_id,
            symbol=record.symbol,
            side=record.side,
            notional=self.order_notional,
            rationale=record.reasoning,
            # The levels the strategy proposed, carried verbatim. When it proposed
            # none, these stay ``None`` — never fabricated here. Sizing is the
            # worker's configured notional, exactly as the historical bridge.
            stop_loss=record.stop_loss,
            take_profit=record.take_profit,
            targets=tuple(record.targets),
            strategy_name=self.strategy_name,
        )

    def _skip(self, detail: str, code: str) -> None:
        self.skip_reason = detail
        self.skip_code = code
        return None


__all__ = [
    "CURRENT_SOURCE_SCHEMA_VERSION",
    "CurrentMarketDecisionSource",
    "CurrentSourceError",
]
