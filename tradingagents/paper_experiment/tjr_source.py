"""A TJR decision source: the deterministic rule chain, gated by the RiskEngine.

This is the third :class:`~.broker.trading.DecisionSource`, and it is the one that
wires the **TJR** strategy — not the LLM — into the Alpaca **paper** worker. The
existing sources are deliberately different things:

- ``ExperimentDecisionSource`` replays a *historical* experiment's recorded
  decisions (the LLM's ratings), bounded by the experiment's sessions;
- ``CurrentMarketDecisionSource`` decides *live* over current **daily** Alpaca
  bars, but again through the LLM inference seam (a five-tier rating);
- **this** source decides live over current **intraday** (1m/5m) Alpaca bars
  through the **TJR rule chain** (``tradingagents.tjr.rules``), which consults no
  model at all.

One decision per symbol per trading session
-------------------------------------------

TJR's own backtest attempts **at most one entry per symbol per session**
(``TjrBacktester._attempted``), so the live source matches it: the decision id is
``tjr-<symbol>-<session>``. Within a session the chain is re-evaluated on each
cycle until it completes; the moment it does, the id is fixed and the worker's
durable ``ordered_decisions`` set (and the broker's ``client_order_id``
idempotency) refuse a duplicate.

The order goes through the RiskEngine
-------------------------------------

A completed setup becomes a :class:`~tradingagents.signals.contract.TradingSignal`
(:mod:`tradingagents.tjr.strategy`), and that signal is put through the project's
own :class:`~tradingagents.risk.engine.RiskEngine` with
:func:`~tradingagents.risk.paper_parity.paper_parity_limits` — the same caps the
backtest and the paper account use. Only an **approved** signal becomes a
:class:`TradingDecision`, carrying the RiskEngine's own sized quantity and the
setup's stop and target. A rejection is a named no-trade, never a placed order.
The broker's own local limits are a second, independent layer.

The book the RiskEngine sizes against is a *mirror* of the broker's account and
positions (supplied by the worker), never a fabricated book: the source invents no
cash and no holding. With no book supplied the RiskEngine is handed an empty
book, so an order still cannot exceed the notional cap.

No fabrication
--------------

The source invents no size, no stop and no target. The size is the RiskEngine's;
the stop and target come from the rule chain's structure, and a setup that carries
none yields ``None`` (so the exit policy watches nothing rather than a made-up
level). A rating with no tradeable action, a closed market, or a data failure
yields *no* decision with a named reason code — never a fake signal.
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

import pandas as pd

from tradingagents.portfolio import Ledger
from tradingagents.risk import RiskEngine, RiskLimits, paper_parity_limits
from tradingagents.signals.contract import SignalAction
from tradingagents.tjr.frames import IntradaySeries
from tradingagents.tjr.rules import TjrSettings
from tradingagents.tjr.strategy import TjrStrategy

from .broker.trading import (
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_NO_SIGNAL,
    NO_TRADE_RISK_REJECTED,
    NO_TRADE_STRATEGY_UNAVAILABLE,
    NO_TRADE_TJR_NO_NEW_DATA,
    TradingDecision,
)
from .market_hours import MARKET_OPEN, MarketHoursError, MarketHoursGate
from .tjr_intraday_data import (
    TjrIntradayData,
    TjrIntradayDataError,
    resample_series,
)

TJR_SOURCE_SCHEMA_VERSION = 1
"""The on-disk TJR-source state format."""

TJR_NO_NEW_DATA = NO_TRADE_TJR_NO_NEW_DATA
"""Every symbol already produced its one decision for the session. A new session,
not a new poll, is what produces the next one. Aliased from the worker's closed
vocabulary so the two cannot drift."""

_REASONING_EXCERPT = 400


class TjrSourceError(RuntimeError):
    """The TJR source was driven in a way it cannot support."""


@dataclass(frozen=True)
class TjrBookSnapshot:
    """A read-only mirror of the broker book the RiskEngine sizes against.

    ``positions`` maps an upper-cased symbol to its long quantity. ``cash`` is the
    account's free cash and ``day_start_equity``/``equity_peak`` are the figures the
    daily-loss and drawdown checks need; ``None`` means "unknown", and the engine
    records that it could not check rather than assuming a value.
    """

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    daily_orders: int = 0
    day_start_equity: float | None = None
    equity_peak: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "daily_orders": self.daily_orders,
            "day_start_equity": self.day_start_equity,
            "equity_peak": self.equity_peak,
        }


def ledger_from_snapshot(snapshot: TjrBookSnapshot, *, prices: dict[str, float]) -> Ledger:
    """A :class:`Ledger` mirroring ``snapshot``, for the RiskEngine to read.

    The ledger is a *view*: it is populated from the broker's own cash and
    holdings, marked at the prices the caller supplies. It is never used to apply a
    fill, so it cannot diverge from the broker — the RiskEngine only reads
    ``cash``/``equity``/``open_positions`` from it.
    """
    cash = snapshot.cash if snapshot.cash > 0 else 1.0
    ledger = Ledger(initial_cash=cash)
    ledger.cash = snapshot.cash
    for symbol, quantity in snapshot.positions.items():
        if quantity <= 0:
            continue
        holding = ledger.holding(symbol.upper())
        holding.quantity = float(quantity)
        price = prices.get(symbol.upper())
        if price is not None and price > 0:
            holding.average_entry_price = float(price)
            holding.last_price = float(price)
    return ledger


@dataclass
class _SymbolRecord:
    """The decision taken for one symbol in one session, persisted verbatim.

    ``reason_code`` is a member of the worker's closed no-trade vocabulary
    (:data:`NO_TRADE_REASONS`), so the worker's persisted ``last_no_trade_code`` is
    always interpretable. ``chain_reason`` is the TJR rule chain's own reason code
    (``NO_1M_ENTRY`` and friends), kept for the audit trail and carried in
    ``detail``; it is deliberately *not* what the worker records as its code.
    """

    symbol: str
    session: str
    decision_id: str
    side: str | None
    reason_code: str
    chain_reason: str
    stage: str
    detail: str
    stop: float | None
    target: float | None
    quantity: float | None
    reference_price: float | None
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session": self.session,
            "decision_id": self.decision_id,
            "side": self.side,
            "reason_code": self.reason_code,
            "chain_reason": self.chain_reason,
            "stage": self.stage,
            "detail": self.detail,
            "stop": self.stop,
            "target": self.target,
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _SymbolRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class TjrDecisionSource:
    """Yields one TJR decision per symbol per trading session.

    Satisfies the worker's :class:`~.broker.trading.DecisionSource` protocol:
    :meth:`next_decision` returns a :class:`TradingDecision` or ``None``, and the
    source never places an order itself.
    """

    symbols: tuple[str, ...]
    data: TjrIntradayData
    market_hours: MarketHoursGate
    order_notional: float
    risk_limits: RiskLimits = field(default_factory=paper_parity_limits)
    settings: TjrSettings = field(default_factory=TjrSettings)
    strategy_name: str = "tjr"
    book_provider: Callable[[], TjrBookSnapshot] | None = None
    """The broker book the RiskEngine sizes against.

    ``None`` means *no book was supplied*, which the engine reads as an empty book
    with no cash and therefore refuses to size against — a fail-safe default: with
    no book there is no cash to size an order from, so no order is produced. The
    live path always supplies the worker's own broker account/positions through
    this seam, so an order is always sized against the real shared book."""
    price_provider: Callable[[str], float] | None = None
    """The *execution* price seam, so the RiskEngine sizes at the price the order
    will actually fill against. ``None`` falls back to the newest closed 1m close."""
    is_consumed: Callable[[str], bool] | None = None
    """Whether the worker has already ordered a decision id (its durable set)."""
    state_path: str | os.PathLike | None = None
    """Where the source's own per-session memory lives. Never the experiment state."""
    clock: Callable[[], datetime] | None = None

    source_kind: str = field(default="tjr", init=False)
    skip_reason: str | None = field(default=None, init=False)
    skip_code: str | None = field(default=None, init=False)
    _records: dict[str, _SymbolRecord] = field(default_factory=dict, init=False)
    _yielded: set[str] = field(default_factory=set, init=False)
    _last_market_state: str | None = field(default=None, init=False)
    _last_market_session: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise TjrSourceError("a TJR source needs at least one symbol")
        self.symbols = tuple(s.upper() for s in self.symbols)
        self._clock = self.clock or (lambda: datetime.now(UTC))
        self._store = Path(self.state_path) if self.state_path else None
        if self._store is not None:
            self._records = self._decode(self._read_payload())

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
        if payload.get("schema_version") != TJR_SOURCE_SCHEMA_VERSION:
            return {}
        return payload

    def _decode(self, payload: dict[str, Any]) -> dict[str, _SymbolRecord]:
        raw = payload.get("records") or {}
        out: dict[str, _SymbolRecord] = {}
        for symbol, record in raw.items():
            if isinstance(record, dict):
                with contextlib.suppress(TypeError):
                    out[symbol] = _SymbolRecord.from_dict(record)
        return out

    def _save(self) -> None:
        if self._store is None:
            return
        payload = {
            "schema_version": TJR_SOURCE_SCHEMA_VERSION,
            "updated_at": self._clock().isoformat(),
            "records": {k: v.to_dict() for k, v in sorted(self._records.items())},
        }
        self._store.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._store.parent), prefix=".tjr-", suffix=".tmp")
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
        """A secret-free view of the TJR source, for the dashboard/status."""
        view: dict[str, Any] = {
            "decision_source_kind": self.source_kind,
            "decision_source_symbols": list(self.symbols),
            "decision_source_strategy": self.strategy_name,
            "decision_source_last_market_state": self._last_market_state,
            "decision_source_last_market_session": self._last_market_session,
            "decision_source_records": {
                k: v.to_dict() for k, v in sorted(self._records.items())
            },
        }
        for seam in (self.data, self.market_hours):
            describe = getattr(seam, "describe", None)
            if callable(describe):
                with contextlib.suppress(Exception):
                    view.update(describe())
        return view

    def last_decision_for(self, symbol: str) -> dict[str, Any] | None:
        record = self._records.get(symbol.upper())
        return record.to_dict() if record is not None else None

    @property
    def no_decision_id(self) -> str | None:
        """The id of the decision the last evaluation produced but did not offer.

        The chain can complete and the RiskEngine can refuse it, or the chain can
        stop short of a signal; either way the evaluation had a decision id. The
        worker records the no-trade against it, so the persisted state names *this*
        evaluation's decision rather than whichever decision was last.
        """
        for record in self._records.values():
            if record.side is None and record.session == self._last_market_session:
                return record.decision_id
        return None

    # -- the protocol -----------------------------------------------------

    def next_decision(self) -> TradingDecision | None:
        """The next due TJR decision, or ``None`` with a reason code."""
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
            "every symbol already produced its one decision for this session; a new "
            "session is required before the next decision",
            TJR_NO_NEW_DATA,
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

        ``None`` means "nothing to report for this symbol"; the caller then tries
        the next symbol. An existing record for the session whose id the worker has
        already ordered is skipped so it is not re-offered.

        A record whose chain **did not complete** does *not* close the session. The
        backtest's engine
        (:meth:`tradingagents.tjr.engine.TjrEngine._process_entries`) re-evaluates
        the chain at every 5m bar close and only marks the session attempted when a
        setup *completed*; the live source must mirror that, or a session that
        opens with an unfinished setup would never be re-read and a later,
        complete setup would be invisible. So a no-signal record only short-circuits
        the symbol for the rest of the *same* evaluation instant, not the session.
        """
        record = self._records.get(symbol)
        if record is not None and record.session == session and self._closes_session(record):
            if record.side is None:
                # A completed setup the RiskEngine refused: the chain did complete,
                # so the session is over, but there is no side to trade. Report it
                # as the no-trade it is rather than offering a decision.
                return (record.detail, record.reason_code)
            if record.decision_id in self._yielded:
                return None
            if self._consumed(record.decision_id):
                return None
            self._yielded.add(record.decision_id)
            return self._to_decision(record)

        as_of = self._clock()
        try:
            five, one = self._series(symbol, session, as_of)
        except TjrIntradayDataError as exc:
            return (f"current intraday data for {symbol} is unavailable: {exc}", exc.code)
        except Exception as exc:  # noqa: BLE001 - any fetch failure is unavailable
            return (
                f"current intraday data for {symbol} is unavailable: {type(exc).__name__}",
                NO_TRADE_MARKET_DATA_UNAVAILABLE,
            )

        strategy = self._strategy(symbol, five, one)
        try:
            signal, evaluation = strategy.signal(symbol, session, pd.Timestamp(as_of))
        except Exception as exc:  # noqa: BLE001 - a rule failure yields no decision
            return (
                f"the TJR chain could not evaluate {symbol}: {type(exc).__name__}",
                NO_TRADE_STRATEGY_UNAVAILABLE,
            )

        detail = f"{evaluation.stage}:{evaluation.reason_code}:{evaluation.detail}"
        if signal.action is not SignalAction.BUY:
            # A chain that did not complete is an honest no-trade: the worker's
            # code is the generic NO_SIGNAL (a member of its closed vocabulary),
            # and the TJR chain's own reason code is carried in the detail.
            record = _SymbolRecord(
                symbol=symbol,
                session=session,
                decision_id=f"tjr-{symbol}-{session}",
                side=None,
                reason_code=NO_TRADE_NO_SIGNAL,
                chain_reason=evaluation.reason_code,
                stage=evaluation.stage,
                detail=detail,
                stop=None,
                target=None,
                quantity=None,
                reference_price=None,
                evaluated_at=as_of.isoformat(),
            )
            self._records[symbol] = record
            self._save()
            return (record.detail[:_REASONING_EXCERPT], record.reason_code)

        reference_price = self._reference_price(symbol, one)
        decision_id = f"tjr-{symbol}-{session}"
        # No book supplied means no cash: the engine refuses to size, so no order is
        # produced. The live path always supplies the worker's own book.
        snapshot = self.book_provider() if self.book_provider is not None else TjrBookSnapshot(0.0)
        ledger = ledger_from_snapshot(snapshot, prices={symbol: reference_price})
        risk = RiskEngine(self.risk_limits)
        verdict = risk.evaluate(
            signal,
            ledger,
            price=reference_price,
            equity_peak=snapshot.equity_peak,
            day_start_equity=snapshot.day_start_equity,
            daily_orders_placed=snapshot.daily_orders,
        )
        if not verdict.approved or verdict.quantity <= 0:
            record = _SymbolRecord(
                symbol=symbol,
                session=session,
                decision_id=decision_id,
                side=None,
                reason_code=NO_TRADE_RISK_REJECTED,
                chain_reason=evaluation.reason_code,
                stage=evaluation.stage,
                detail=f"the RiskEngine refused the entry: {verdict.explain()}",
                stop=evaluation.stop,
                target=evaluation.target,
                quantity=None,
                reference_price=reference_price,
                evaluated_at=as_of.isoformat(),
            )
            self._records[symbol] = record
            self._save()
            return (record.detail, NO_TRADE_RISK_REJECTED)

        record = _SymbolRecord(
            symbol=symbol,
            session=session,
            decision_id=decision_id,
            side="buy",
            reason_code=evaluation.reason_code,
            chain_reason=evaluation.reason_code,
            stage=evaluation.stage,
            detail=detail,
            stop=evaluation.stop,
            target=evaluation.target,
            quantity=float(verdict.quantity),
            reference_price=reference_price,
            evaluated_at=as_of.isoformat(),
        )
        self._records[symbol] = record
        self._save()
        if self._consumed(decision_id):
            return None
        self._yielded.add(decision_id)
        return self._to_decision(record)

    def _series(
        self, symbol: str, session: str, as_of: datetime
    ) -> tuple[IntradaySeries, IntradaySeries]:
        """The (5m, 1m) decision series for ``symbol`` at ``as_of``.

        Fetched fresh each call: the source is polled at the worker's cadence, and
        a fresh fetch is what guarantees a decision is never made on bars an earlier
        poll happened to cache. The seam itself drops every bar that had not closed
        at ``as_of``, so the closed-bar rule holds regardless of cadence.
        """
        return self.data.series(symbol, session=session, as_of=as_of)

    def _strategy(
        self, symbol: str, five: IntradaySeries, one: IntradaySeries
    ) -> TjrStrategy:
        """The TJR strategy over the fetched series, with the same hourly/4h levels
        the backtest builds (resampled from the 5m series, absent when a bucket
        cannot be filled)."""
        hourly = resample_series(five, "60m")
        four_hour = resample_series(five, "4h")
        return TjrStrategy(
            one_minute={symbol: one},
            five_minute={symbol: five},
            hourly=({symbol: hourly} if hourly is not None else None),
            four_hour=({symbol: four_hour} if four_hour is not None else None),
            settings=self.settings,
            allow_short=False,
        )

    def _reference_price(self, symbol: str, one: IntradaySeries) -> float:
        """The price the RiskEngine sizes against: the execution seam's, else the
        newest closed 1m close."""
        if self.price_provider is not None:
            price = self.price_provider(symbol)
            if price is not None and price > 0:
                return float(price)
        return float(one.frame["close"].iloc[-1])

    def _consumed(self, decision_id: str) -> bool:
        if self.is_consumed is None:
            return False
        try:
            return bool(self.is_consumed(decision_id))
        except Exception:  # noqa: BLE001 - an unreadable guard must not re-offer
            return False

    @staticmethod
    def _closes_session(record: _SymbolRecord) -> bool:
        """Whether a record ends the symbol's session, as the backtest defines it.

        The backtest marks ``(symbol, session)`` attempted only when the chain
        produced a BUY (:meth:`tradingagents.tjr.engine.TjrEngine._process_entries`),
        so a completed setup — whether it was sized, or refused by the RiskEngine —
        ends the session, while an unfinished chain does not. A risk refusal is a
        completed setup: the chain *did* complete and the engine *did* judge it, so
        re-reading the same bars would only re-run the same refusal.
        """
        if record.side is not None:
            return True
        return record.reason_code == NO_TRADE_RISK_REJECTED

    def _to_decision(self, record: _SymbolRecord) -> TradingDecision:
        assert record.side is not None  # only called for a tradeable side
        return TradingDecision(
            decision_id=record.decision_id,
            symbol=record.symbol,
            side=record.side,
            quantity=record.quantity,
            rationale=record.detail,
            stop_loss=record.stop,
            take_profit=record.target,
            strategy_name=self.strategy_name,
        )

    def _skip(self, detail: str, code: str) -> None:
        self.skip_reason = detail
        self.skip_code = code
        return None


__all__ = [
    "TJR_NO_NEW_DATA",
    "TJR_SOURCE_SCHEMA_VERSION",
    "TjrBookSnapshot",
    "TjrDecisionSource",
    "TjrSourceError",
    "ledger_from_snapshot",
]
