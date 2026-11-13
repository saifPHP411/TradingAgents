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
(``TjrBacktester._attempted``), and the live source matches it: within a session
the chain is re-evaluated at every 5m bar close and a symbol is attempted only
when a setup *completes*.

A "session" is not a closed **day**, though. The backtest judges the chain again
at every new bar close, so a sweep that appears later the same day is a *new*
setup and is judged on its own merits. The live source does the same: the symbol
is re-evaluated whenever newer **completed** bars have arrived, and a setup is
identified by **the sweep that produced it**. So a new sweep in the same session
gets its own decision id and may trade, while a re-read of the same sweep is the
same signal and is refused by the worker's durable ``ordered_decisions`` set (and
by the broker's ``client_order_id`` idempotency).

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
import hashlib
import json
import math
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
from .tjr_diagnostics import classify_evaluation_failure, format_evaluation_failure
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

    **A holding is matched by its base asset, not by its full symbol.** On the
    crypto path the broker holds the position under the *execution* pair
    (``XRP/USD``) while the strategy's price is quoted in the *market-data* pair
    (``XRP/USDT``). Matching the two spellings literally left every position
    marked at no price, so the RiskEngine computed an equity of the cash alone —
    understating the account by the whole value of the open positions, and with it
    every exposure and drawdown check that is a fraction of equity. The base asset
    is the one part of the symbol both venues spell the same way, so it is the
    key that actually identifies the holding.
    """
    cash = snapshot.cash if snapshot.cash > 0 else 1.0
    ledger = Ledger(initial_cash=cash)
    ledger.cash = snapshot.cash
    for symbol, quantity in snapshot.positions.items():
        if quantity <= 0:
            continue
        holding = ledger.holding(symbol.upper())
        holding.quantity = float(quantity)
        price = _matching_price(symbol, prices)
        if price is not None and price > 0:
            holding.average_entry_price = float(price)
            holding.last_price = float(price)
    return ledger


def _matching_price(symbol: str, prices: dict[str, float]) -> float | None:
    """The price for ``symbol`` from ``prices``, matched by base asset if needed.

    An exact (case-insensitive) symbol match wins. Failing that, the price whose
    symbol names the same **base asset** is used — the crypto path's
    ``XRP/USD`` holding is marked at the ``XRP/USDT`` price, because the two name
    the same coin. A symbol whose base asset matches nothing is left unmarked, so
    its value is excluded rather than guessed at.
    """
    key = symbol.upper()
    price = prices.get(key)
    if price is not None:
        return price
    base = _base_asset(key)
    if base is None:
        return None
    for candidate, candidate_price in prices.items():
        if _base_asset(candidate.upper()) == base:
            return candidate_price
    return None


def _base_asset(symbol: str) -> str | None:
    """The base asset of a ``BASE/QUOTE`` symbol, or ``None`` if it has no quote."""
    text = symbol.strip().upper()
    if "/" not in text:
        return None
    base = text.partition("/")[0]
    return base or None


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
    notional: float | None
    """The cash amount the entry was sized for, in the account's own currency.

    This is what the worker turns into an order quantity, at the price of the pair
    that will actually trade. ``quantity`` is kept alongside it as the audit trail
    of what the RiskEngine computed at the market-data venue's price.
    """
    reference_price: float | None
    evaluated_at: str
    targets: tuple[float, ...] = ()
    """Every take-profit the rule chain derived, nearest first.

    ``target`` is the first of these. The ladder holds only the draws on
    liquidity the market actually supports, so it can hold one level or several
    and is never padded.
    """
    setup_key: str | None = None
    """The identity of the setup this record came from.

    A completed setup is identified by **the sweep that produced it** — the 5m bar
    that swept the level, the level's price and the sweep's direction. Two
    evaluations that agree on this key are the *same* setup seen twice, so the
    second is a repeat and must not produce a second order; a different key is a
    different setup, and may produce a new one.

    ``None`` for a record whose chain did not complete: there is no setup to
    identify.
    """

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
            "targets": list(self.targets),
            "quantity": self.quantity,
            "notional": self.notional,
            "reference_price": self.reference_price,
            "evaluated_at": self.evaluated_at,
            "setup_key": self.setup_key,
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
    execution_price_provider: Callable[[str], float] | None = None
    """The execution venue's price for the **execution** symbol, in the account's
    own currency.

    The crypto path analyses one market and trades another: the TJR chain reads
    Bybit Spot's ``BTC/USDT`` and the order is placed on Bybit Demo Spot's same pair. The
    account is funded in USD and every cap the risk engine enforces — the $25
    order notional, the exposure fraction of equity — is denominated in the
    account's currency. Sizing a USD order from a USDT price therefore sized it in
    the wrong currency, and the cash value of the order was never the value the
    risk engine approved.

    This seam supplies the execution symbol's quote-currency price, so the risk engine
    sizes the entry in the currency the order will actually cost. It is separate
    from :attr:`price_provider` (the market-data venue's price) because the two
    are different instruments on different venues.

    ``None`` falls back to :attr:`price_provider`, and then to the newest closed
    1m close — the same fallback the market-data path uses. On a single-venue
    path the two prices are the same price, so nothing changes."""
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
        the next symbol.

        **The symbol is re-evaluated whenever it has not been decided yet, or
        whenever new completed data has arrived.** A record that ends the symbol's
        session (a completed setup, whether sized or refused by the RiskEngine)
        used to short-circuit the symbol for the *whole* session, which made a
        sweep that appeared later the same day invisible: the chain was never run
        again, so the opportunity was never seen. The backtest's engine
        (:meth:`tradingagents.tjr.engine.TjrEngine._process_entries`) re-evaluates
        at every 5m bar close and only marks the session attempted when a setup
        *completed*; the live source now mirrors that.

        **A repeated setup does not produce a second order.** A setup is
        identified by the sweep that produced it, so re-reading the same bars
        yields the same setup key and the same decision id: the worker's ordered
        set refuses it, and the source does not re-offer it either. A *different*
        sweep is a different setup and gets its own decision id, so it may trade.

        A record whose chain **did not complete** does not close the session
        either, so a session that opens with an unfinished setup is re-read and a
        later, complete setup is found.
        """
        record = self._records.get(symbol)
        as_of = self._clock()
        # The bars are fetched on **every** cycle, before anything else. That is
        # what detects a new opportunity: the chain can only see a new sweep if
        # the new bars are read. It is also what keeps the data seam's own
        # "last fetch" evidence current, so the view cannot claim the data is
        # stale (or absent) while the worker is polling it.
        try:
            five, one = self._series(symbol, session, as_of)
        except TjrIntradayDataError as exc:
            return (f"current intraday data for {symbol} is unavailable: {exc}", exc.code)
        except Exception as exc:  # noqa: BLE001 - any fetch failure is unavailable
            return (
                f"current intraday data for {symbol} is unavailable: {type(exc).__name__}",
                NO_TRADE_MARKET_DATA_UNAVAILABLE,
            )

        if (
            record is not None
            and record.session == session
            and self._closes_session(record)
            and not self._has_new_completed_data(five, record)
        ):
            # The session is closed for this symbol, but only until *new* completed
            # data arrives. The chain is re-run when the newest closed bar closed
            # after the instant the record was judged. A re-run that finds the
            # *same* setup is a repeat and is not re-offered; one that finds a
            # *new* setup is a new opportunity.
            if record.side is None:
                # A completed setup the RiskEngine refused: the chain did
                # complete, so there is nothing to re-judge until new data
                # arrives. Report it as the no-trade it is.
                return (record.detail, record.reason_code)
            if record.decision_id in self._yielded:
                return None
            if self._consumed(record.decision_id):
                return None
            self._yielded.add(record.decision_id)
            return self._to_decision(record)

        strategy = self._strategy(symbol, five, one)
        try:
            signal, evaluation = strategy.signal(symbol, session, pd.Timestamp(as_of))
        except Exception as exc:  # noqa: BLE001 - a rule failure yields no decision
            # The failure is **not** a no-signal: the chain could not produce an
            # answer, so the source refuses to produce a decision. What changes
            # is that the refusal now says *where* it happened and *which
            # contract field* was refused, so the fault is diagnosable without
            # the operator reproducing it. The detail is sanitized: it names the
            # stage and the field, and carries no market data and no secret.
            diagnostics = classify_evaluation_failure(exc)
            return (
                format_evaluation_failure(symbol, diagnostics),
                NO_TRADE_STRATEGY_UNAVAILABLE,
            )

        setup_key = self._setup_key(evaluation)
        detail = f"{evaluation.stage}:{evaluation.reason_code}:{evaluation.detail}"

        # A re-read of the **same** setup is a repeat, not a new opportunity. The
        # chain is re-run on new bars, and a new bar can confirm the setup that
        # was already found without changing which setup it is. The identity of
        # the setup decides: the same sweep is the same signal and must not be
        # offered twice; a different sweep is a different signal.
        if (
            record is not None
            and record.setup_key is not None
            and setup_key is not None
            and record.setup_key == setup_key
            and self._closes_session(record)
        ):
            if record.side is None:
                return (record.detail, record.reason_code)
            if record.decision_id in self._yielded or self._consumed(record.decision_id):
                return None
            self._yielded.add(record.decision_id)
            return self._to_decision(record)

        if signal.action is not SignalAction.BUY:
            # A chain that did not complete is an honest no-trade: the worker's
            # code is the generic NO_SIGNAL (a member of its closed vocabulary),
            # and the TJR chain's own reason code is carried in the detail.
            #
            # An unfinished chain does **not** replace a completed one: the record
            # still holds the completed setup, so the symbol keeps its session
            # closed and the next poll compares against that setup, not against
            # this unfinished one.
            if record is not None and self._closes_session(record):
                return None
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
                notional=None,
                reference_price=None,
                evaluated_at=as_of.isoformat(),
            )
            self._records[symbol] = record
            self._save()
            return (record.detail[:_REASONING_EXCERPT], record.reason_code)

        reference_price = self._reference_price(symbol, one)
        decision_id = self._decision_id(symbol, session, setup_key)
        # No book supplied means no cash: the engine refuses to size, so no order is
        # produced. The live path always supplies the worker's own book.
        snapshot = self.book_provider() if self.book_provider is not None else TjrBookSnapshot(0.0)
        # The price the risk engine sizes at is the **execution** venue's price for
        # the pair the order will trade, in the account's own currency. The
        # market-data venue's price (the chain's own ``reference_price``) is a
        # different instrument on a different venue, quoted in a different
        # currency, so sizing from it made the engine's $25 notional cap and its
        # equity fractions describe a USD order in USDT terms.
        execution_price = self._execution_price(symbol, reference_price)
        ledger = ledger_from_snapshot(snapshot, prices={symbol: execution_price})
        risk = RiskEngine(self.risk_limits)
        verdict = risk.evaluate(
            signal,
            ledger,
            price=execution_price,
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
                targets=tuple(evaluation.targets or ()),
                quantity=None,
                notional=None,
                reference_price=reference_price,
                evaluated_at=as_of.isoformat(),
                setup_key=setup_key,
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
            targets=tuple(evaluation.targets or ()),
            quantity=float(verdict.quantity),
            # The cash amount the RiskEngine sized the entry for. The worker turns
            # this into a quantity at the *execution* venue's price, so the order's
            # cash value is the one the risk engine approved. The engine's own
            # ``max_order_value`` cap is what bounds this figure, and it is
            # unchanged.
            notional=(
                float(verdict.notional)
                if verdict.notional and verdict.notional > 0
                else float(verdict.quantity) * execution_price
            ),
            # The price the entry was sized at — the execution venue's, in the
            # account's currency. It is the price the notional is denominated in,
            # so the two can never disagree.
            reference_price=execution_price,
            evaluated_at=as_of.isoformat(),
            setup_key=setup_key,
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

    def _execution_price(self, symbol: str, fallback: float) -> float:
        """The execution venue's price for ``symbol``, in the account's currency.

        Falls back to ``fallback`` (the market-data price) when no execution
        price is available, so a single-venue path — where the two prices are the
        same price — behaves exactly as it did before.
        """
        if self.execution_price_provider is not None:
            price = self.execution_price_provider(symbol)
            if price is not None and price > 0 and math.isfinite(float(price)):
                return float(price)
        return float(fallback)

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

        A closed session is not a closed **day**: it is closed *for the bars it was
        judged on*. New completed bars are a new judgement, and the backtest
        judges them too — it re-evaluates the chain at every 5m bar close and only
        refuses a symbol it has already *attempted*, which it does so at the
        instant the setup completes, not for the rest of the day.
        """
        if record.side is not None:
            return True
        return record.reason_code == NO_TRADE_RISK_REJECTED

    @staticmethod
    def _setup_key(evaluation: Any) -> str | None:
        """The identity of the setup an evaluation found.

        A setup is identified by **the sweep that produced it**: the 5m bar that
        swept the level, the level's price, and the sweep's direction. That is
        what distinguishes "the same setup, seen again" from "a new sweep, later
        the same day" — the distinction the session rule needs in order to allow a
        new opportunity without repeating an old signal.

        ``None`` when the chain did not complete: there is no setup to identify.
        """
        sweep = getattr(evaluation, "sweep", None)
        if sweep is None:
            return None
        bar = getattr(sweep, "bar", None)
        level = getattr(sweep, "level", None)
        price = getattr(level, "price", None) if level is not None else None
        direction = getattr(sweep, "direction", None)
        if bar is None or price is None or direction is None:
            return None
        return f"{pd.Timestamp(bar).isoformat()}|{float(price):.8f}|{direction}"

    @staticmethod
    def _decision_id(symbol: str, session: str, setup_key: str | None) -> str:
        """The decision id for a setup.

        The id must identify the **setup**, not merely the day, so that a new
        sweep in the same session is a new decision while a re-read of the same
        sweep is the same decision. The setup's identity is folded into the id as
        a short digest; without a setup key (an evaluation that did not complete)
        the id falls back to the session-scoped form, which is what an
        already-decided symbol must keep.
        """
        base = f"tjr-{symbol}-{session}"
        if not setup_key:
            return base
        digest = hashlib.sha256(setup_key.encode("utf-8")).hexdigest()[:12]
        return f"{base}-{digest}"

    @staticmethod
    def _has_new_completed_data(five: IntradaySeries, record: _SymbolRecord) -> bool:
        """Whether **completed** bars are available beyond the ones already judged.

        The record was judged at :attr:`_SymbolRecord.evaluated_at`, the instant
        of that evaluation. The chain is re-run when the data seam can serve a
        bar that had **closed after** that instant — the same rule the backtest
        uses when it judges the chain at every 5m bar close.

        The comparison is against the *evaluation instant*, not against the
        record's age, because the newest bar the seam serves is the newest bar
        that had closed by **now**: a bar that closed after the judgement is new
        information, and a bar that closed before it is not.

        A forming bar is never counted. The data seam drops every bar that had
        not closed at the evaluation instant, so a still-forming bar can never
        make the source believe it has new data.
        """
        frame = getattr(five, "frame", None)
        if frame is None or getattr(frame, "empty", True):
            return False
        evaluated_at = record.evaluated_at
        if not evaluated_at:
            return True
        try:
            judged_at = pd.Timestamp(evaluated_at)
        except Exception:  # noqa: BLE001 - an unusable stamp is not new data
            return False
        try:
            newest_close = pd.Timestamp(frame.index[-1]) + pd.Timedelta(
                minutes=getattr(five, "bar_minutes", 5)
            )
        except Exception:  # noqa: BLE001 - an unusable index is not new data
            return False
        return newest_close > judged_at

    def _to_decision(self, record: _SymbolRecord) -> TradingDecision:
        assert record.side is not None  # only called for a tradeable side
        return TradingDecision(
            decision_id=record.decision_id,
            symbol=record.symbol,
            side=record.side,
            # The **notional**, never the quantity. The RiskEngine sized the entry
            # at the market-data venue's price (Binance's ``XRP/USDT``), but the
            # order is placed on the execution venue's pair (``XRP/USD``). A
            # quantity sized from one venue's price means a different thing on the
            # other, so the worker would send an order whose cash value it never
            # computed. Carrying the cash amount instead lets the worker size the
            # order at the price the pair will actually trade at, which is the
            # only price the account's currency and the risk cap are denominated
            # in. The RiskEngine's own cap on this figure is unchanged.
            notional=record.notional,
            rationale=record.detail,
            stop_loss=record.stop,
            take_profit=record.target,
            targets=tuple(record.targets or ((record.target,) if record.target is not None else ())),
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
