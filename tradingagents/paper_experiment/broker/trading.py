"""A start/stop-controllable paper-trading worker that places Alpaca **paper** orders.

This is the one place in the project that turns a *strategy decision* into a
broker order. It is deliberately narrow, and it is the only module that passes
``allow_orders=True`` to the broker. Nothing here runs on its own: the worker is
``STOPPED`` until an authenticated caller explicitly starts it, and after a
restart it is ``STOPPED`` again, never ``RUNNING``.

The path, stated once:

    decision source -> risk checks -> AlpacaPaperBroker -> Alpaca Paper API
                    -> order status / fills -> durable worker state

``start()`` also launches a **background loop** so ``START`` means "run
continuously" rather than "set a flag". The loop waits ``poll_interval_seconds``
*first*, then calls :meth:`PaperTradingWorker.run_once` once per interval until
:meth:`PaperTradingWorker.stop` is called. Waiting before the first cycle is the
safe choice: it gives an operator a window to press STOP before any order can be
placed, and it makes "start places no order" true for a whole interval, not just
for the instant of the call.

Design rules, each of which a test pins:

- **Paper only.** The broker's own endpoint gate is the single chokepoint: only
  ``https://paper-api.alpaca.markets`` is accepted and every live host is refused
  by :func:`~.alpaca_paper.assert_paper_endpoint`. This module never constructs a
  broker with any other base URL, and there is no setting that turns paper into
  live.
- **No order on start.** ``start()`` runs preflight checks and sets the state to
  ``RUNNING``; it places no order. An order is placed only when the decision
  source yields a decision *and* the local risk checks pass.
- **One loop.** A second ``start()`` while the loop is live raises instead of
  spawning a second loop; ``stop()`` then ``start()`` yields exactly one new loop.
- **Idempotent.** Each decision maps to a deterministic ``client_order_id``
  derived from the decision id, symbol and side. A re-run, a timeout, or a
  restart reconciles by that id rather than submitting a second order.
- **Fail closed.** A missing or unconfirmed account, an unavailable market-data
  source, a buying-power shortfall, a breached cap, or a would-be short all refuse
  the order *before* it reaches the broker — the broker is never relied on to
  reject it.
- **Graceful stop.** ``stop()`` refuses further cycles and stops the loop; it does
  not close positions and does not cancel open orders. An order already in flight
  is reported with its status, not abandoned silently. ``stop()`` waits for the
  in-flight cycle to finish rather than killing the thread.
- **A cycle failure is contained.** An unexpected exception inside a cycle is
  recorded, the state goes to ``ERROR``, and the loop stops — the process does not
  crash, and the loop does not spin.

Secrets never appear here: the broker reads ``ALPACA_PAPER_API_KEY`` /
``ALPACA_PAPER_API_SECRET`` from the environment itself, and this module never
reads, logs, or persists a credential.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from tradingagents.crypto.symbols import try_parse_pair
from tradingagents.risk.paper_parity import WHOLE_SHARES

from ..symbol_mapping import (
    SymbolMapping,
    get_allowed_symbols_for_broker,
)
from .alpaca_paper import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
    BrokerOrder,
    BrokerRejectedError,
    InsufficientFundsError,
    PaperRiskLimits,
    PaperRiskRejected,
    assert_paper_endpoint,
)

WORKER_STATE_SCHEMA_VERSION = 1
"""The on-disk worker state format. Bumped when a reader must know the shape changed."""

# -- the closed state vocabulary -------------------------------------------------

STATUS_STOPPED = "STOPPED"
STATUS_STARTING = "STARTING"
STATUS_RUNNING = "RUNNING"
STATUS_STOPPING = "STOPPING"
STATUS_ERROR = "ERROR"

WORKER_STATUSES = (
    STATUS_STOPPED,
    STATUS_STARTING,
    STATUS_RUNNING,
    STATUS_STOPPING,
    STATUS_ERROR,
)

BANNER = "ALPACA PAPER — NOT LIVE"
"""Shown wherever the worker's state is surfaced, so no view can imply live trading."""

# -- market-data freshness -------------------------------------------------------

DEFAULT_MAX_SNAPSHOT_AGE_SECONDS = 300.0
"""How old a snapshot price may be before it is refused. Conservative: five
minutes. A *current* price that is older than this is not a current price, and
sizing an order from it would be sizing from a stale number."""

DEFAULT_SNAPSHOT_FUTURE_TOLERANCE_SECONDS = 5.0
"""How far ahead of the local clock a snapshot timestamp may be before it is
refused as future-dated. A small tolerance absorbs clock skew between this host
and the vendor; anything beyond it is a timestamp we cannot trust."""

DEFAULT_MAX_PRICE_DEVIATION = 0.05
"""The largest permitted gap between the decision price and the execution price,
as a fraction (5%).

The strategy decides on the market-data venue's price (``BTC/USDT``); the order
is priced on the execution venue's (``BTC/USD``). Those two are the same coin
quoted in different currencies on different venues, so they differ by the
quote-asset basis, the venue's own spread, and whatever the market did between
the decision and the order. A gap within this bound is ordinary market movement
and the trade is still the trade the strategy decided on; a gap beyond it means
the sizing would be materially different from the decision, so the order is
refused rather than placed at an unexpected notional.

Configurable per deployment (``max_price_deviation``); a value of zero would
require the two prices to be identical, which two different venues never are.
"""

MAX_PRICE_DEVIATION_ENV = "TRADINGAGENTS_PAPER_MAX_PRICE_DEVIATION"
"""The environment variable naming the permitted decision-to-execution gap.

A fraction (``0.05`` = 5%), not a percentage. The deployment's own value; the
default is :data:`DEFAULT_MAX_PRICE_DEVIATION`.
"""

MARKET_DATA_KIND_HISTORICAL = "historical"
MARKET_DATA_KIND_CURRENT = "current"
MARKET_DATA_KIND_TEST = "test"

MARKET_DATA_STATUS_OK = "ok"
MARKET_DATA_STATUS_NO_DATA = "no_data"
MARKET_DATA_STATUS_STALE = "stale"
MARKET_DATA_STATUS_FUTURE = "future"
MARKET_DATA_STATUS_MALFORMED = "malformed"
MARKET_DATA_STATUS_ERROR = "error"
MARKET_DATA_STATUS_NOT_STARTED = "not_started"

MARKET_DATA_STATUSES = (
    MARKET_DATA_STATUS_OK,
    MARKET_DATA_STATUS_NO_DATA,
    MARKET_DATA_STATUS_STALE,
    MARKET_DATA_STATUS_FUTURE,
    MARKET_DATA_STATUS_MALFORMED,
    MARKET_DATA_STATUS_ERROR,
    MARKET_DATA_STATUS_NOT_STARTED,
)

PROVIDER_BAR_OPEN = "bar_open"
PROVIDER_ALPACA_SNAPSHOT = "alpaca_snapshot"
PROVIDER_BINANCE_SNAPSHOT = "binance_snapshot"
"""A current *crypto* price from Binance's public market data (no key).

The crypto counterpart of ``alpaca_snapshot``: it answers "what is the price
now" from Binance's data-only host, validated for freshness before an order is
sized. It is a market-data seam only and never reaches an order endpoint.
"""
PROVIDER_ALPACA_CRYPTO = "alpaca_crypto"
"""A current price on the **execution** venue (Alpaca crypto), for the pair traded.

The cross-venue price seam: the strategy analyses Binance's ``BTC/USDT`` and the
order is placed on Alpaca's ``BTC/USD``, so the order must be sized from a price
of the *execution* pair, read from the execution venue itself.
"""
PROVIDER_STATIC = "static"
PROVIDER_UNKNOWN = "unknown"

# -- why no order was placed -----------------------------------------------------

# A closed vocabulary for "the worker ran but placed no order". A bare "no decision
# is due" cannot tell a cadence gap from a disarmed strategy, so every no-order
# cycle names the actual gate. Codes are stable (a UI or an alert can match on
# them); the human `detail` alongside is free text.
NO_TRADE_NONE = None
NO_TRADE_NO_DECISION_DUE = "NO_DECISION_DUE"
NO_TRADE_NO_SIGNAL = "NO_SIGNAL"
NO_TRADE_HISTORY_EXHAUSTED = "HISTORY_EXHAUSTED"
NO_TRADE_MAX_DECISIONS_REACHED = "MAX_DECISIONS_REACHED"
NO_TRADE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NO_TRADE_MARKET_DATA_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
NO_TRADE_SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
NO_TRADE_RISK_REJECTED = "RISK_REJECTED"
NO_TRADE_BROKER_REJECTED = "BROKER_REJECTED"
NO_TRADE_INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
"""The venue refused the order because the account had no buying power for it.

Distinct from :data:`NO_TRADE_BROKER_REJECTED` because the two call for
different actions: this one is the *account's* money, so it is fixed by funding
the account or freeing cash, while a general broker refusal is about the order
itself. The worker names it when the broker says so, never on its own judgement.
"""
NO_TRADE_ALREADY_PROCESSED = "ALREADY_PROCESSED"
NO_TRADE_ORDER_EXISTS = "ORDER_EXISTS"
NO_TRADE_EXIT_REFUSED = "EXIT_REFUSED"
NO_TRADE_MARKET_CLOSED = "MARKET_CLOSED"
"""The current-market source's market-hours gate: the market is closed."""
NO_TRADE_STRATEGY_UNAVAILABLE = "STRATEGY_UNAVAILABLE"
"""The current-market source could not reach a strategy decision (no inference)."""
NO_TRADE_CURRENT_MARKET_NO_NEW_DATA = "CURRENT_MARKET_NO_NEW_DATA"
"""Every symbol already has a decision for the current session; new data is needed.

Distinct from ``HISTORY_EXHAUSTED``: that names a fully-consumed historical
experiment, which does *not* block the current-market source.
"""
NO_TRADE_TJR_NO_NEW_DATA = "TJR_NO_NEW_DATA"
"""Every symbol already produced its one TJR decision for the session.

The TJR source attempts at most one entry per symbol per session, exactly as the
TJR backtest does, so once every symbol is decided a new *session* (not a new
poll) is what produces the next decision. Named apart from the current-market
code so a dashboard can tell the two session-scoped sources apart.
"""
NO_TRADE_NO_POSITION = "NO_POSITION"
"""A SELL with nothing held to close. Not a risk breach: closing a position that
does not exist is simply not a trade, so it is named apart from
``RISK_REJECTED`` and can never be recorded as a short. A long-only book has no
position to open short, and a SELL is only ever a close.
"""
NO_TRADE_POSITION_UNAVAILABLE = "POSITION_UNAVAILABLE"
"""The held quantity could not be read from the broker, so a SELL cannot be sized.

Distinct from ``NO_POSITION``: that means "the book holds none", which is a fact;
this means "the broker could not be asked", which is a failure. Kept apart so a
real broker error is never misreported as "nothing to close".
"""
NO_TRADE_EXECUTION_PRICE_UNAVAILABLE = "EXECUTION_PRICE_UNAVAILABLE"
"""No fresh price could be read on the **execution** venue for the order's pair.

The decision's own price came from the market-data venue (Binance), and it is not
the price the broker will trade at: the two venues quote different instruments
(``BTC/USDT`` vs ``BTC/USD``). Sizing an order from the market-data price would
commit a notional computed from the wrong instrument, so an execution-side price
is required. When it cannot be read, **no order is placed**.
"""
NO_TRADE_PRICE_DEVIATION = "PRICE_DEVIATION"
"""The execution-venue price differs from the decision price by more than allowed.

The decision was taken on the market-data venue's price; the order is priced on
the execution venue's. The two are allowed to differ by a bounded amount (the
venue's own spread, market movement, and the different quote asset), but a larger
gap means the trade would no longer be the trade the strategy decided on, so the
order is refused rather than placed at an unexpected notional.
"""

NO_TRADE_REASONS = (
    NO_TRADE_NO_DECISION_DUE,
    NO_TRADE_NO_SIGNAL,
    NO_TRADE_HISTORY_EXHAUSTED,
    NO_TRADE_MAX_DECISIONS_REACHED,
    NO_TRADE_INSUFFICIENT_DATA,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_SIZE_TOO_SMALL,
    NO_TRADE_RISK_REJECTED,
    NO_TRADE_BROKER_REJECTED,
    NO_TRADE_INSUFFICIENT_FUNDS,
    NO_TRADE_ALREADY_PROCESSED,
    NO_TRADE_ORDER_EXISTS,
    NO_TRADE_EXIT_REFUSED,
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_STRATEGY_UNAVAILABLE,
    NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
    NO_TRADE_TJR_NO_NEW_DATA,
    NO_TRADE_NO_POSITION,
    NO_TRADE_POSITION_UNAVAILABLE,
    NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
    NO_TRADE_PRICE_DEVIATION,
)
"""The closed vocabulary of reasons a RUNNING worker placed no order. A reason is
``None`` only when the cycle actually placed an order."""

DECISION_OUTCOME_ORDERED = "ORDERED"
"""The :attr:`WorkerState.decision_outcomes` value for a decision that produced an
order (submitted, or reconciled as already existing). Distinct from every
:data:`NO_TRADE_REASONS` value, which names why no order was produced. Kept out of
``NO_TRADE_REASONS`` because it is the *absence* of a no-trade reason, and a
consumer that iterates the no-trade vocabulary must not read it as one."""

MARKET_DATA_CHOICES = (PROVIDER_BAR_OPEN, PROVIDER_ALPACA_SNAPSHOT, PROVIDER_BINANCE_SNAPSHOT)
"""The price seams an operator may select. ``bar_open`` is the historical default;
``alpaca_snapshot`` is the equity current-price opt-in; ``binance_snapshot`` is the
crypto current-price opt-in (no key)."""

# -- which decision source the worker draws on -----------------------------------

DECISION_SOURCE_HISTORY = "history"
"""The default: the paper experiment's recorded, historical decisions.

This is the existing ``ExperimentDecisionSource``. It is bounded by the
experiment's session set, so it reports ``HISTORY_EXHAUSTED`` once every session
is consumed. Unchanged by the current-market work.
"""
DECISION_SOURCE_CURRENT_MARKET = "current_market"
"""Opt-in: a live decision over current Alpaca market data.

The independent ``CurrentMarketDecisionSource``. It is *not* the historical
source and is never blocked by a consumed historical history. It is opt-in so
the default behaviour is unchanged.
"""
DECISION_SOURCE_TJR = "tjr"
"""Opt-in: the deterministic TJR rule chain over current Alpaca *intraday* bars.

The independent ``TjrDecisionSource``. Like the current-market source it is
session-scoped (one decision per symbol per session) and is never blocked by a
consumed historical history, but its strategy is the TJR rule chain
(``tradingagents.tjr``), which consults no model at all. Opt-in, so the default
behaviour is unchanged.
"""
DECISION_SOURCE_CRYPTO = "crypto"
"""Opt-in: the deterministic TJR rule chain over current **crypto** intraday bars.

The same ``TjrDecisionSource`` class as ``tjr`` — the rule chain is identical, so
there is exactly one strategy — but wired with the crypto seams: Binance public
1m/5m bars (no key) and the 24/7 crypto clock. One decision per symbol per UTC
session, id ``tjr-<symbol>-<session>``. Opt-in and paper-only; the default is
unchanged.
"""

DECISION_SOURCE_CHOICES = (
    DECISION_SOURCE_HISTORY,
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_TJR,
    DECISION_SOURCE_CRYPTO,
)
"""The decision sources an operator may select. ``history`` is the default."""


@dataclass(frozen=True)
class SnapshotObservation:
    """One usable current price, with the timestamp of the *same* field.

    ``field`` names which part of the snapshot the price came from
    (``latestTrade``/``latestQuote``/``minuteBar``) and ``as_of`` is that field's
    own timestamp, so a freshness check can never compare a price from one field
    against a timestamp from another.
    """

    price: float
    as_of: str | None
    field: str


def _parse_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 stamp to an aware UTC datetime, or ``None``.

    A trailing ``Z`` is accepted (the vendor's format) and a naive value is read
    as UTC rather than as the host's local time, so the age of a snapshot cannot
    depend on the machine's timezone.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class WorkerError(RuntimeError):
    """The worker was driven in a way it cannot support."""


class PreflightError(WorkerError):
    """A start-time check failed, so the worker must not run."""

    def __init__(self, check: str, detail: str) -> None:
        self.check = check
        self.detail = detail
        super().__init__(f"preflight check {check!r} failed: {detail}")


class MarketDataError(WorkerError):
    """The market-data source could not supply a usable price."""


class NoPositionError(WorkerError):
    """A SELL asked to close a position the book does not hold.

    Distinct from :class:`WorkerError` (which the worker records as a sizing
    failure) because a SELL with nothing to close is not a defect: on a long-only
    book a SELL is only ever a close, so there is simply no trade to place. The
    worker reports it as ``NO_TRADE_NO_POSITION`` rather than a size error or a
    risk breach, and never turns it into a short.
    """


class PositionReadError(WorkerError):
    """The held quantity could not be read, so a SELL cannot be sized safely.

    A broker read failure is a *failure*, not the fact "nothing is held": the two
    are reported apart (``NO_TRADE_POSITION_UNAVAILABLE`` vs ``NO_TRADE_NO_POSITION``)
    so a genuine broker error is never misread as "there is no position".
    """


# ---------------------------------------------------------------------------
# Decision and market-data seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingDecision:
    """A strategy decision the worker may turn into one paper order.

    ``notional`` is the target cash amount to trade; the worker converts it to a
    share quantity from the *current* price, so sizing never depends on a stale
    price. ``quantity`` may be given instead, when the caller has already sized
    the order in units.

    When **both** are given, the notional is the quantity's cash value at the
    price the quantity was sized at, and the worker re-divides the notional by
    the price the order will actually trade at. That matters on the cross-venue
    crypto path, where a quantity sized at the market-data venue's price
    (``XRP/USDT``) would otherwise be sent as an order on the execution venue's
    pair (``XRP/USD``) — a quantity that means a different amount of cash on the
    other side, so the order's notional was never the one that was approved.
    """

    decision_id: str
    symbol: str
    side: str  # "buy" or "sell"
    notional: float | None = None
    quantity: float | None = None
    rationale: str = ""
    limit_price: float | None = None
    stop_loss: float | None = None
    """The strategy's proposed stop-loss, when it proposed one. Recorded with the
    position so the exit policy can act on it. ``None`` means the strategy proposed
    none — it is never fabricated, so a position without a recorded stop is simply
    not watched for a stop exit."""
    take_profit: float | None = None
    """The strategy's proposed take-profit, when it proposed one. Same rules as
    ``stop_loss``."""
    strategy_name: str | None = None
    """The name of the strategy that produced the decision, for the Decisions view.
    Never fabricated: ``None`` when the source did not name one."""

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise WorkerError("a decision must carry a decision_id")
        if not self.symbol:
            raise WorkerError("a decision must carry a symbol")
        if self.side.lower() not in ("buy", "sell"):
            raise WorkerError(f"side must be buy or sell, got {self.side!r}")
        if self.notional is None and self.quantity is None:
            raise WorkerError("a decision must carry a notional or a quantity")
        for value in (
            self.notional,
            self.quantity,
            self.limit_price,
            self.stop_loss,
            self.take_profit,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise WorkerError(
                    f"decision figures must be finite and positive, got {value!r}"
                )

    def client_order_id(self, *, strategy_id: str) -> str:
        """A deterministic, broker-safe id for this decision.

        Deterministic so a re-run, a timeout, or a restart produces the *same* id
        and the broker's own idempotency (and this worker's reconcile path) refuse
        a duplicate rather than placing a second order.
        """
        material = f"{strategy_id}:{self.decision_id}:{self.symbol.upper()}:{self.side.lower()}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
        return f"pt-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "limit_price": self.limit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
        }


class DecisionSource(Protocol):
    """Yields the next strategy decision, or ``None`` when there is nothing to do.
    A source must be *deterministic in its ordering* and must not itself place an
    order. The worker treats a decision it has already processed as done, so a
    source that yields the same decision twice is safe.
    """

    def next_decision(self) -> TradingDecision | None:  # pragma: no cover - protocol
        ...


class MarketData(Protocol):
    """Supplies a current price for a symbol, and reports whether it is usable.

    ``provider_name``/``kind``/``describe`` let the dashboard and the cycle result
    name the *actual* seam in use, so a historical bar open is never displayed as a
    live price. They are optional (a duck-typed test double need not implement
    them); :func:`market_data_identity` reads them defensively.
    """

    def price(self, symbol: str) -> float:  # pragma: no cover - protocol
        ...

    def available(self) -> bool:  # pragma: no cover - protocol
        ...


def market_data_identity(market_data: Any) -> dict[str, Any]:
    """The secret-free identity of a market-data seam, for a status view.

    Reads ``provider_name``/``kind``/``describe`` when a seam offers them and
    falls back to the class name, so a seam that predates this protocol is still
    labelled rather than mislabelled.
    """
    provider = getattr(market_data, "provider_name", None)
    kind = getattr(market_data, "kind", None)
    describe = getattr(market_data, "describe", None)
    identity = {
        "market_data_provider": provider if isinstance(provider, str) else PROVIDER_UNKNOWN,
        "market_data_kind": kind if isinstance(kind, str) else None,
    }
    if callable(describe):
        with contextlib.suppress(Exception):  # a status view must never raise
            identity.update(describe())
    return identity


def decision_source_identity(decision_source: Any) -> dict[str, Any]:
    """The secret-free identity of a decision source, for a status view.

    A source that describes itself (the current-market source does, with its
    last market state, session and last-fetched data) contributes those fields;
    a source that predates this protocol contributes nothing, so the view falls
    back to the *configured* ``decision_source`` label rather than mislabelling
    it. Never raises: a status view must not fail because a source is odd.
    """
    identity: dict[str, Any] = {}
    kind = getattr(decision_source, "source_kind", None)
    if isinstance(kind, str):
        identity["decision_source_kind"] = kind
    describe = getattr(decision_source, "describe", None)
    if callable(describe):
        with contextlib.suppress(Exception):
            described = describe()
            if isinstance(described, dict):
                identity.update(described)
    return identity


@dataclass
class StaticMarketData:
    """A fixed price map, for tests and offline demonstration."""

    prices: dict[str, float] = field(default_factory=dict)

    provider_name: str = PROVIDER_STATIC
    kind: str = MARKET_DATA_KIND_TEST

    def available(self) -> bool:
        return bool(self.prices)

    def price(self, symbol: str) -> float:
        value = self.prices.get(symbol.upper())
        if value is None:
            raise MarketDataError(f"no market data for {symbol!r}")
        if not math.isfinite(value) or value <= 0:
            raise MarketDataError(f"a usable price is required for {symbol!r}, got {value!r}")
        return float(value)

    def describe(self) -> dict[str, Any]:
        return {
            "market_data_detail": "a fixed price map (tests/demo), not a market price",
            # The snapshot seam records a status as it reads; this seam has no read
            # to record, so it derives one from availability. Without this the
            # engine view read ``market_data_status`` as absent and reported the
            # market data as unavailable even while the worker priced an order
            # from it.
            "market_data_status": (
                MARKET_DATA_STATUS_OK if self.available() else MARKET_DATA_STATUS_NO_DATA
            ),
        }


@dataclass
class SourceMarketData:
    """Prices from a price source's bar for a session — the "current" bar to trade.

    ``source`` is any object with ``bars(symbol, session)`` returning a bar with an
    ``open`` (the same :class:`~tradingagents.paper_experiment.sources.PriceSource`
    the experiment uses). This is the *latest available* bar for the session, which
    is the honest meaning of "current market data" for a daily-bar experiment. A
    live intraday quote provider can implement the same ``price``/``available``
    seam instead.

    This is **historical** data: the price is a past session's bar open, not a live
    quote. ``kind`` says so, so no view can present it as a current price.
    """

    source: Any
    session: str

    provider_name: str = PROVIDER_BAR_OPEN
    kind: str = MARKET_DATA_KIND_HISTORICAL

    def available(self) -> bool:
        try:
            return self.source.covers(self.session)
        except Exception:  # noqa: BLE001 - an unusable source is "not available"
            return False

    def price(self, symbol: str) -> float:
        try:
            bar = self.source.bars(symbol.upper(), self.session)
        except Exception as exc:  # noqa: BLE001 - a source failure is surfaced
            raise MarketDataError(f"the price source failed for {symbol!r}: {exc}") from None
        if bar is None:
            raise MarketDataError(f"no bar for {symbol!r} at session {self.session}")
        price = float(bar.open)
        if not math.isfinite(price) or price <= 0:
            raise MarketDataError(f"the bar for {symbol!r} carries no usable price: {price!r}")
        return price

    def describe(self) -> dict[str, Any]:
        return {
            "market_data_detail": (
                "the historical daily-bar open for the experiment's latest session; "
                "this is a past session price, not a live quote"
            ),
            "historical_session": self.session,
            "is_current_price": False,
            # Derived from availability, because this seam has no read to record a
            # status on. Absent, the engine view read the market data as
            # unavailable while the worker was pricing orders from it.
            "market_data_status": (
                MARKET_DATA_STATUS_OK if self.available() else MARKET_DATA_STATUS_NO_DATA
            ),
        }


class AlpacaSnapshotMarketData:
    """A *current* price from Alpaca's market-data snapshot endpoint.

    This is the modern-price seam the paper worker trades against, and it is
    deliberately separate from the experiment's historical/reproducible dataset:

    - the dataset answers "what was the price at session T", point-in-time, for a
      reproducible decision;
    - this answers "what is the price now", for sizing an order against the current
      market.

    A daily bar's **open** is not a current price, so this seam never reads one:
    the price is the **newest** of the latest trade, the latest quote midpoint and
    the minute-bar close, compared by their own timestamps, each of which is a
    genuinely current observation. A fresher quote therefore wins over an older
    trade, which matters on the IEX feed where a symbol can go minutes without
    printing a trade while its quote keeps updating. If no current price is
    available, :meth:`price` raises and the worker places no order rather than
    sizing from a stale or fabricated number.

    **Freshness.** A current price is only useful if it is current, so the
    timestamp of the *same* field the price came from is validated: the newest
    field being older than ``max_age_seconds`` (default
    :data:`DEFAULT_MAX_SNAPSHOT_AGE_SECONDS`, 300s), a missing or unparseable
    timestamp, or a timestamp materially in the future is refused with
    :class:`MarketDataError`. Selecting the newest field does not weaken this
    check: when every current field is stale the price is still refused. The
    worker then places **no** order. There is no fallback to a historical daily
    bar: this seam never reads one, so selecting it can never silently degrade to
    a stale price.

    Host and credentials are the market-data host and the paper credentials; this
    class never touches the trading endpoint and never places an order.
    """

    provider_name = PROVIDER_ALPACA_SNAPSHOT
    kind = MARKET_DATA_KIND_CURRENT

    def __init__(
        self,
        symbols,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        transport=None,
        timeout: float | None = None,
        max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
        future_tolerance_seconds: float = DEFAULT_SNAPSHOT_FUTURE_TOLERANCE_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from ..providers.alpaca_iex import (
            API_KEY_ENV,
            API_SECRET_ENV,
            DATA_BASE_URL,
            REQUEST_TIMEOUT,
            _auth_headers,
            _classify_status,
            assert_data_host,
        )

        self.symbols = tuple(s.upper() for s in symbols)
        self._base_url = assert_data_host(base_url or DATA_BASE_URL)
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        key, secret = key.strip(), secret.strip()
        if not key or not secret:
            raise MarketDataError(
                f"no Alpaca market-data credentials: set {API_KEY_ENV} and "
                f"{API_SECRET_ENV} so a current price can be read"
            )
        if max_age_seconds is not None and not (max_age_seconds > 0):
            raise MarketDataError(
                f"max_age_seconds must be positive or None, got {max_age_seconds!r}"
            )
        self.max_age_seconds = max_age_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
        self._key = key
        self._secret = secret
        self._timeout = timeout or REQUEST_TIMEOUT
        self._transport = transport
        self._headers = _auth_headers(key, secret)
        self._classify_status = _classify_status
        self._clock = clock or (lambda: datetime.now(UTC))
        # The most recent observation this seam has successfully read, so a status
        # view can report the last price, its timestamp and its age. Never carries a
        # credential.
        self.last_observation: SnapshotObservation | None = None
        self.last_observed_at: str | None = None
        self.last_status: str = MARKET_DATA_STATUS_NOT_STARTED
        self.last_status_detail: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """The secret-free identity and last-observation view for a dashboard."""
        observation = self.last_observation
        return {
            "market_data_detail": (
                "a current price from Alpaca market data (feed=iex); validated for "
                "freshness before any order is sized"
            ),
            "is_current_price": self._is_current_price(),
            "max_snapshot_age_seconds": self.max_age_seconds,
            "last_snapshot_at": observation.as_of if observation else None,
            "last_snapshot_price": observation.price if observation else None,
            "last_snapshot_field": observation.field if observation else None,
            "last_snapshot_age_seconds": self.last_snapshot_age_seconds(),
            "last_snapshot_observed_at": self.last_observed_at,
            "market_data_status": self.last_status,
            "market_data_status_detail": self.last_status_detail,
        }

    def _is_current_price(self) -> bool:
        """Whether the price this seam is currently reporting is a current one.

        The answer is derived from the **same** observation the age is measured
        from, so the two can never disagree: a price whose age has grown past
        ``max_age_seconds`` is reported as not current, whatever the last read
        said. A seam that has read nothing yet is not current either.
        """
        if self.last_observation is None:
            return False
        age = self.last_snapshot_age_seconds()
        if age is None:
            return False
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            return False
        return age >= -abs(self.future_tolerance_seconds)

    def last_snapshot_age_seconds(self) -> float | None:
        """The age, in seconds, of the last observed snapshot; ``None`` if unknown."""
        observation = self.last_observation
        if observation is None or observation.as_of is None:
            return None
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            return None
        return (self._clock() - parsed).total_seconds()

    def available(self) -> bool:
        """Whether a *fresh* current price can be read for every configured symbol."""
        if not self.symbols:
            return False
        try:
            snapshots = self._fetch()
        except Exception:  # noqa: BLE001 - an unreachable source is "not available"
            self._record_status(MARKET_DATA_STATUS_ERROR, "the snapshot request failed")
            return False
        if not isinstance(snapshots, dict):
            self._record_status(MARKET_DATA_STATUS_MALFORMED, "the response was not an object")
            return False
        for symbol in self.symbols:
            try:
                observation = self._observation_from(snapshots.get(symbol))
            except MarketDataError as exc:
                self._record_status(_status_for(exc), str(exc))
                return False
            if self.last_observation is None:
                # Stored from *this* read, so the reported age and the reported
                # status describe the same price rather than two different reads.
                self.last_observation = observation
                self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return True

    def price(self, symbol: str) -> float:
        """The current, fresh price for ``symbol``, or raise :class:`MarketDataError`."""
        symbol = symbol.upper()
        try:
            snapshots = self._fetch()
        except MarketDataError as exc:
            self._record_status(_status_for(exc), str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - a source failure is surfaced
            detail = f"the Alpaca snapshot request failed for {symbol!r}: {type(exc).__name__}"
            self._record_status(MARKET_DATA_STATUS_ERROR, detail)
            raise MarketDataError(detail) from None
        if not isinstance(snapshots, dict):
            detail = "Alpaca's snapshot response was not an object"
            self._record_status(MARKET_DATA_STATUS_MALFORMED, detail)
            raise MarketDataError(detail)
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            detail = (
                f"Alpaca returned no snapshot for {symbol!r}; no current price is "
                "available, so no order can be sized"
            )
            self._record_status(MARKET_DATA_STATUS_NO_DATA, detail)
            raise MarketDataError(detail)
        try:
            observation = self._observation_from(snapshot)
        except MarketDataError as exc:
            # A malformed, missing, stale or future-dated price is recorded so a
            # status view can name *why* no order was sized.
            self._record_status(_status_for(exc), str(exc))
            raise
        self.last_observation = observation
        self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return observation.price

    # -- internals --------------------------------------------------------

    def _record_status(self, status: str, detail: str | None) -> None:
        self.last_status = status
        self.last_status_detail = detail

    def _observation_from(self, snapshot: Any) -> SnapshotObservation:
        """A fresh, usable observation from a snapshot, or raise ``MarketDataError``."""
        observation = self._price_from(snapshot)
        if observation is None:
            raise MarketDataError(
                "Alpaca's snapshot carried no usable current price (no latest "
                "trade, quote, or minute bar)"
            )
        self._assert_fresh(observation)
        return observation

    def _assert_fresh(self, observation: SnapshotObservation) -> None:
        """Refuse a snapshot whose timestamp is missing, stale, or future-dated."""
        if observation.as_of is None:
            raise MarketDataError(
                f"the snapshot's {observation.field} carries no timestamp, so its "
                "age cannot be verified; refusing to size an order from an "
                "unverifiable price"
            )
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            raise MarketDataError(
                f"the snapshot's {observation.field} timestamp {observation.as_of!r} "
                "is malformed; refusing to size an order from an unverifiable price"
            )
        age = (self._clock() - parsed).total_seconds()
        if age < -abs(self.future_tolerance_seconds):
            raise MarketDataError(
                f"the snapshot's {observation.field} timestamp is {-age:.1f}s in the "
                "future, beyond the allowed clock skew; refusing a future-dated price"
            )
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            raise MarketDataError(
                f"the snapshot is stale: the {observation.field} is {age:.1f}s old, "
                f"beyond the {self.max_age_seconds:.0f}s maximum; refusing to size an "
                "order from a stale price"
            )

    def _fetch(self) -> dict:
        from ..providers.base import fetch_json

        return fetch_json(
            url=f"{self._base_url}/v2/stocks/snapshots",
            params={"symbols": ",".join(self.symbols), "feed": "iex"},
            timeout=self._timeout,
            secrets=(self._key, self._secret),
            headers=self._headers,
            transport=self._transport,
            classify_status=self._classify_status,
        )

    @staticmethod
    def _price_from(snapshot: Any) -> SnapshotObservation | None:
        """The newest usable current price *and its own timestamp*, or ``None``.

        The fields are compared by the timestamp each one carries, not by a fixed
        preference order: the price wanted here is the newest observation of the
        *current* market, so a fresher quote is chosen over an older trade. That
        matters because a symbol can go minutes without printing on the IEX feed
        while its quote keeps updating, and a trade-first order would then refuse
        a perfectly fresh market.

        ``dailyBar`` and ``prevDailyBar`` are never candidates: a daily bar is not
        a current price. Every candidate must be finite and positive, so a NaN
        cannot pass a comparison and poison a later sizing step.

        An exact tie on the timestamp is settled by
        :data:`_SNAPSHOT_FIELD_PREFERENCE` (trade, then quote, then minute bar),
        so the choice stays deterministic. A field whose timestamp is missing or
        unparseable cannot outrank a field with a usable timestamp; it is returned
        only when *no* candidate has a usable timestamp, so the freshness check
        can still name precisely why the price is unusable. Ranking is by
        timestamp alone — a stale or future-dated newest field is still selected
        and then refused by the freshness check, never quietly replaced by an
        older field.
        """
        return _select_snapshot_observation(_snapshot_candidates(snapshot))


class BinanceSnapshotMarketData:
    """A *current* crypto price from Binance's public market data. No key.

    The crypto counterpart of :class:`AlpacaSnapshotMarketData`, and the seam the
    crypto paper path sizes an order against. It is deliberately separate from the
    historical/reproducible dataset:

    - the frozen dataset answers "what was the price at session T", point-in-time;
    - this answers "what is the price now", from Binance's 24-hour ticker.

    Binance's public data hosts need no key and no registration, so this seam reads
    no credential at all — it can therefore never be pointed at an order endpoint,
    because it has no order endpoint and no key to reach one with.

    **Freshness.** The price is only useful if it is current, so the ticker's own
    ``observed_at`` (Binance's ``closeTime``) is validated: a ticker older than
    ``max_age_seconds`` (default :data:`DEFAULT_MAX_SNAPSHOT_AGE_SECONDS`), a
    missing or unparseable timestamp, or one materially in the future is refused
    with :class:`MarketDataError`, and the worker then places **no** order rather
    than sizing from a stale or fabricated number. The price chosen is the *last
    trade* when present, else the quote midpoint; both are current observations of
    the same ticker and share its timestamp, so the freshness check is coherent.

    The Binance client is injectable, so every branch is exercised offline.
    """

    provider_name = PROVIDER_BINANCE_SNAPSHOT
    kind = MARKET_DATA_KIND_CURRENT

    def __init__(
        self,
        symbols,
        *,
        client: Any | None = None,
        max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
        future_tolerance_seconds: float = DEFAULT_SNAPSHOT_FUTURE_TOLERANCE_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_age_seconds is not None and not (max_age_seconds > 0):
            raise MarketDataError(
                f"max_age_seconds must be positive or None, got {max_age_seconds!r}"
            )
        self.symbols = tuple(s.upper() for s in symbols)
        self.max_age_seconds = max_age_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))
        self.last_observation: SnapshotObservation | None = None
        self.last_observed_at: str | None = None
        self.last_status: str = MARKET_DATA_STATUS_NOT_STARTED
        self.last_status_detail: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        observation = self.last_observation
        return {
            "market_data_detail": (
                "a current crypto price from Binance public market data (no key); "
                "validated for freshness before any order is sized"
            ),
            "is_current_price": self._is_current_price(),
            "max_snapshot_age_seconds": self.max_age_seconds,
            "last_snapshot_at": observation.as_of if observation else None,
            "last_snapshot_price": observation.price if observation else None,
            "last_snapshot_field": observation.field if observation else None,
            "last_snapshot_age_seconds": self.last_snapshot_age_seconds(),
            "last_snapshot_observed_at": self.last_observed_at,
            "market_data_status": self.last_status,
            "market_data_status_detail": self.last_status_detail,
            "market_data_diagnostics": self._diagnostics(),
        }

    def _is_current_price(self) -> bool:
        """Whether the price this seam is currently reporting is a current one.

        Derived from the **same** observation the reported age is measured from,
        so the two can never disagree. A price whose age has grown past
        ``max_age_seconds`` is reported as not current, whatever the last read
        said at the time; a seam that has read nothing yet is not current either.
        """
        if self.last_observation is None:
            return False
        age = self.last_snapshot_age_seconds()
        if age is None:
            return False
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            return False
        return age >= -abs(self.future_tolerance_seconds)

    def _diagnostics(self) -> list[dict[str, Any]]:
        """The venue's recent request records, sanitized for display.

        Each record names the endpoint path, the HTTP status, the venue's own
        error code and message, the response's JSON type and whether the symbol
        was present in the response. It never carries a credential, a header or a
        query string, so it is safe to show in the dashboard and to log.
        """
        client = self._client
        records = getattr(client, "last_diagnostics", None)
        if not isinstance(records, list):
            return []
        return [dict(record) for record in records[-10:]]

    def last_snapshot_age_seconds(self) -> float | None:
        observation = self.last_observation
        if observation is None or observation.as_of is None:
            return None
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            return None
        return (self._clock() - parsed).total_seconds()

    # -- reads ------------------------------------------------------------

    def _client_or_build(self):
        if self._client is None:
            from tradingagents.crypto.binance import BinancePublicClient

            self._client = BinancePublicClient()
        return self._client

    def available(self) -> bool:
        """Whether a *fresh* current price can be read for every configured symbol."""
        if not self.symbols:
            return False
        try:
            tickers = self._client_or_build().ticker_24h(self.symbols)
        except Exception as exc:  # noqa: BLE001 - an unreachable source is "not available"
            detail = f"the Binance ticker request failed: {type(exc).__name__}"
            self._record_status(MARKET_DATA_STATUS_ERROR, detail)
            return False
        if not isinstance(tickers, dict):
            self._record_status(MARKET_DATA_STATUS_MALFORMED, "the response was not an object")
            return False
        observations: list[SnapshotObservation] = []
        for symbol in self.symbols:
            try:
                observations.append(self._observation_from(_find_ticker(tickers, symbol)))
            except MarketDataError as exc:
                self._record_status(_status_for(exc), str(exc))
                return False
        # The observation is stored from *this* read, so the reported age and the
        # reported status describe the same price. Validating without storing left
        # the view showing the age of an older read beside the status of a newer
        # one, which is how a 273s-old price was reported as "ok".
        if observations:
            self.last_observation = observations[0]
            self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return True

    def price(self, symbol: str) -> float:
        """The current, fresh crypto price for ``symbol``, or raise.

        ``symbol`` may be written either way round — the canonical ``DOGE/USDT``
        or the venue's own ``DOGEUSDT``. Both name the same instrument, so both
        must find the same ticker; a spelling difference is not a reason to
        report that no price exists.
        """
        symbol = symbol.upper()
        try:
            tickers = self._client_or_build().ticker_24h([symbol])
        except Exception as exc:  # noqa: BLE001 - a source failure is surfaced
            detail = f"the Binance ticker request failed for {symbol!r}: {type(exc).__name__}"
            self._record_status(MARKET_DATA_STATUS_ERROR, detail)
            raise MarketDataError(detail) from None
        if not isinstance(tickers, dict):
            detail = "the Binance ticker response was not an object"
            self._record_status(MARKET_DATA_STATUS_MALFORMED, detail)
            raise MarketDataError(detail)
        ticker = _find_ticker(tickers, symbol)
        if ticker is None:
            detail = (
                f"Binance returned no ticker for {symbol!r}; no current price is "
                "available, so no order can be sized"
            )
            self._record_status(MARKET_DATA_STATUS_NO_DATA, detail)
            raise MarketDataError(detail)
        try:
            observation = self._observation_from(ticker)
        except MarketDataError as exc:
            self._record_status(_status_for(exc), str(exc))
            raise
        self.last_observation = observation
        self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return observation.price

    # -- internals --------------------------------------------------------

    def _record_status(self, status: str, detail: str | None) -> None:
        self.last_status = status
        self.last_status_detail = detail

    def _observation_from(self, ticker: Any) -> SnapshotObservation:
        observation = _crypto_observation(ticker)
        if observation is None:
            raise MarketDataError(
                "Binance's ticker carried no usable current price (no last trade and "
                "no coherent bid/ask)"
            )
        self._assert_fresh(observation)
        return observation

    def _assert_fresh(self, observation: SnapshotObservation) -> None:
        if observation.as_of is None:
            raise MarketDataError(
                f"the ticker's {observation.field} carries no timestamp, so its age "
                "cannot be verified; refusing to size an order from an unverifiable price"
            )
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            raise MarketDataError(
                f"the ticker's {observation.field} timestamp {observation.as_of!r} is "
                "malformed; refusing to size an order from an unverifiable price"
            )
        age = (self._clock() - parsed).total_seconds()
        if age < -abs(self.future_tolerance_seconds):
            raise MarketDataError(
                f"the ticker's {observation.field} timestamp is {-age:.1f}s in the "
                "future, beyond the allowed clock skew; refusing a future-dated price"
            )
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            raise MarketDataError(
                f"the ticker is stale: the {observation.field} is {age:.1f}s old, beyond "
                f"the {self.max_age_seconds:.0f}s maximum; refusing to size an order from "
                "a stale price"
            )


class AlpacaCryptoExecutionPrice:
    """A current price on the **execution** venue, for the pair that will be traded.

    The crypto path analyses one market and trades another: the TJR chain reads
    Binance's ``BTC/USDT``, and the order is placed on Alpaca's ``BTC/USD``. A
    price from one is not the price of the other — different quote asset, different
    venue, and the two can drift apart. This seam is the *execution-side* price:
    it reads Alpaca's own crypto snapshot for the **execution** symbol and refuses
    anything that is not current.

    It therefore:

    - reads the execution venue's price for the execution pair, never the
      market-data pair;
    - validates freshness on the timestamp of the *same* field the price came
      from, exactly as the other snapshot seams do;
    - raises :class:`MarketDataError` when no usable, fresh execution price exists,
      so the worker places **no** order rather than sizing from the market-data
      venue's price.

    The client is injectable, so every branch is exercised offline.
    """

    provider_name = PROVIDER_ALPACA_CRYPTO
    kind = MARKET_DATA_KIND_CURRENT

    def __init__(
        self,
        symbols,
        *,
        client: Any | None = None,
        max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
        future_tolerance_seconds: float = DEFAULT_SNAPSHOT_FUTURE_TOLERANCE_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.symbols = tuple(s.upper() for s in symbols)
        self._client = client
        self.max_age_seconds = max_age_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self.last_observation: SnapshotObservation | None = None
        self.last_observed_at: str | None = None
        self.last_status: str = MARKET_DATA_STATUS_NOT_STARTED
        self.last_status_detail: str | None = None

    def _client_or_build(self):
        if self._client is None:
            from ..providers.alpaca_crypto import AlpacaCryptoClient

            self._client = AlpacaCryptoClient()
        return self._client

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        observation = self.last_observation
        return {
            "market_data_detail": (
                "a current price on the EXECUTION venue (Alpaca crypto), read for the "
                "pair the order will actually trade; validated for freshness"
            ),
            "is_current_price": self._is_current_price(),
            "execution_data_provider": self.provider_name,
            "execution_data_max_snapshot_age_seconds": self.max_age_seconds,
            "execution_data_last_snapshot_at": observation.as_of if observation else None,
            "execution_data_last_snapshot_price": observation.price if observation else None,
            "execution_data_last_snapshot_field": observation.field if observation else None,
            "execution_data_last_snapshot_age_seconds": self.last_snapshot_age_seconds(),
            "execution_data_last_observed_at": self.last_observed_at,
            "execution_data_status": self.last_status,
            "execution_data_status_detail": self.last_status_detail,
        }

    def _is_current_price(self) -> bool:
        """Whether the execution price being reported is a current one.

        Derived from the **same** observation the reported age is measured from,
        so the two can never disagree: a price whose age has grown past
        ``max_age_seconds`` is reported as not current, whatever the last read
        said at the time. A seam that has read nothing yet is not current either.
        """
        if self.last_observation is None:
            return False
        age = self.last_snapshot_age_seconds()
        if age is None:
            return False
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            return False
        return age >= -abs(self.future_tolerance_seconds)

    def last_snapshot_age_seconds(self) -> float | None:
        observation = self.last_observation
        if observation is None or observation.as_of is None:
            return None
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            return None
        return (self._clock() - parsed).total_seconds()

    def available(self) -> bool:
        """Whether a fresh execution price can be read for every configured symbol."""
        if not self.symbols:
            return False
        try:
            snapshots = self._client_or_build().latest_snapshot(self.symbols)
        except Exception:  # noqa: BLE001 - an unreachable source is "not available"
            self._record_status(MARKET_DATA_STATUS_ERROR, "the execution price request failed")
            return False
        observations: list[SnapshotObservation] = []
        for symbol in self.symbols:
            try:
                observations.append(self._observation_from(snapshots.get(symbol)))
            except MarketDataError as exc:
                self._record_status(_status_for(exc), str(exc))
                return False
        # Stored from *this* read, so the reported age and the reported status
        # describe the same price rather than two different reads.
        if observations:
            self.last_observation = observations[0]
            self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return True

    def price(self, symbol: str) -> float:
        """The current, fresh execution-venue price for ``symbol``.

        Raises :class:`MarketDataError` when the execution venue cannot serve a
        usable, fresh price. The worker then places **no** order: a market-data
        price is never used as a substitute for the execution price.
        """
        from ..providers.alpaca_crypto import snapshot_price_and_time

        symbol = symbol.upper()
        try:
            snapshots = self._client_or_build().latest_snapshot((symbol,))
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - a source failure is surfaced
            detail = (
                f"the Alpaca execution-price request failed for {symbol!r}: "
                f"{type(exc).__name__}"
            )
            self._record_status(MARKET_DATA_STATUS_ERROR, detail)
            raise MarketDataError(detail) from None
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            detail = (
                f"Alpaca returned no execution snapshot for {symbol!r}; no current "
                "price is available on the execution venue, so no order can be sized"
            )
            self._record_status(MARKET_DATA_STATUS_NO_DATA, detail)
            raise MarketDataError(detail)
        price, observed_at, field = snapshot_price_and_time(snapshot)
        if price is None:
            detail = (
                f"Alpaca's execution snapshot for {symbol!r} carried no usable price "
                "(no latest trade, quote, or minute bar)"
            )
            self._record_status(MARKET_DATA_STATUS_NO_DATA, detail)
            raise MarketDataError(detail)
        observation = SnapshotObservation(
            price=float(price),
            as_of=observed_at.isoformat() if observed_at is not None else None,
            field=field or "latestTrade",
        )
        self._assert_fresh(observation)
        self.last_observation = observation
        self.last_observed_at = self._clock().isoformat()
        self._record_status(MARKET_DATA_STATUS_OK, None)
        return observation.price

    # -- internals --------------------------------------------------------

    def _record_status(self, status: str, detail: str | None) -> None:
        self.last_status = status
        self.last_status_detail = detail

    def _observation_from(self, snapshot: Any) -> SnapshotObservation:
        from ..providers.alpaca_crypto import snapshot_price_and_time

        price, observed_at, field = snapshot_price_and_time(snapshot)
        if price is None:
            raise MarketDataError(
                "Alpaca's execution snapshot carried no usable current price"
            )
        observation = SnapshotObservation(
            price=float(price),
            as_of=observed_at.isoformat() if observed_at is not None else None,
            field=field or "latestTrade",
        )
        self._assert_fresh(observation)
        return observation

    def _assert_fresh(self, observation: SnapshotObservation) -> None:
        """Refuse an execution price whose timestamp is missing, stale or future-dated."""
        if observation.as_of is None:
            raise MarketDataError(
                f"the execution snapshot's {observation.field} carries no timestamp, so "
                "its age cannot be verified; refusing to size an order from an "
                "unverifiable execution price"
            )
        parsed = _parse_utc(observation.as_of)
        if parsed is None:
            raise MarketDataError(
                f"the execution snapshot's {observation.field} timestamp "
                f"{observation.as_of!r} is malformed; refusing to size an order from an "
                "unverifiable execution price"
            )
        age = (self._clock() - parsed).total_seconds()
        if age < -abs(self.future_tolerance_seconds):
            raise MarketDataError(
                f"the execution snapshot's {observation.field} timestamp is {-age:.1f}s "
                "in the future, beyond the allowed clock skew; refusing a future-dated "
                "execution price"
            )
        if self.max_age_seconds is not None and age > self.max_age_seconds:
            raise MarketDataError(
                f"the execution price is stale: the {observation.field} is {age:.1f}s "
                f"old, beyond the {self.max_age_seconds:.0f}s maximum; refusing to size "
                "an order from a stale execution price"
            )


def _find_ticker(tickers: dict[str, Any], symbol: str) -> Any:
    """The ticker for ``symbol``, whatever spelling either side used.

    The venue's ticker map is keyed by the canonical ``BASE/QUOTE`` pair, while a
    caller may hold the same instrument in the venue's own ``BASEQUOTE`` spelling
    (the paper worker's symbol mapping does exactly that). Both name the same
    instrument, so the lookup tries the canonical form first and then the venue's
    spelling, and only reports "absent" when neither is in the map.

    The comparison is on the *canonical* pair, so a symbol can never be matched
    to a different instrument by accident: ``BTC/USDT`` and ``BTCUSDT`` are the
    same pair, and nothing else is.
    """
    ticker = tickers.get(symbol)
    if ticker is not None:
        return ticker
    pair = try_parse_pair(symbol)
    if pair is None:
        return None
    for key in (pair.canonical, pair.binance_symbol, pair.alpaca_symbol):
        ticker = tickers.get(key)
        if ticker is not None:
            return ticker
    # A client that keyed its map by the venue's spelling only is still found.
    for key, ticker in tickers.items():
        other = try_parse_pair(str(key))
        if other is not None and other.canonical == pair.canonical:
            return ticker
    return None


def _crypto_observation(ticker: Any) -> SnapshotObservation | None:
    """The current crypto price *and its own timestamp*, or ``None``.

    The last trade is preferred when present; otherwise the quote midpoint is used
    when the quote is coherent (``ask >= bid > 0``). Both fields share the ticker's
    ``observed_at``, so the freshness check compares the price against the timestamp
    of that same ticker. A non-finite or non-positive price is refused, so a NaN
    cannot pass and later poison a sizing step.
    """
    observed_at = getattr(ticker, "observed_at", None)
    as_of = None if observed_at is None else observed_at.isoformat()
    last = getattr(ticker, "last", None)
    if last is not None and math.isfinite(last) and last > 0:
        return SnapshotObservation(price=float(last), as_of=as_of, field="lastPrice")
    mid = getattr(ticker, "mid", None)
    if mid is not None and math.isfinite(mid) and mid > 0:
        return SnapshotObservation(price=float(mid), as_of=as_of, field="quoteMid")
    return None


def _stamp(value: Any) -> str | None:
    """A timestamp as its original string, or ``None`` when absent/blank."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _status_for(exc: MarketDataError) -> str:
    """The closed-vocabulary status that best names a refusal reason."""
    text = str(exc)
    if "stale" in text:
        return MARKET_DATA_STATUS_STALE
    if "future" in text:
        return MARKET_DATA_STATUS_FUTURE
    if "timestamp" in text:
        return MARKET_DATA_STATUS_MALFORMED
    if "no usable current price" in text:
        return MARKET_DATA_STATUS_NO_DATA
    if "no snapshot" in text or "no ticker" in text:
        return MARKET_DATA_STATUS_NO_DATA
    return MARKET_DATA_STATUS_ERROR


def _exit_reason(
    price: float, stop_loss: Any, take_profit: Any
) -> str | None:
    """Which recorded exit level ``price`` has crossed, or ``None``.

    A stop is checked before a target (the protective exit wins when a bar spans
    both), and both are compared only against a finite price and a finite,
    positive recorded level, so a malformed level is ignored rather than treated as
    a trigger. Returns ``"stop_loss"`` or ``"take_profit"``.
    """
    if not math.isfinite(price) or price <= 0:
        return None
    stop = _as_float(stop_loss)
    if stop is not None and stop > 0 and price <= stop:
        return "stop_loss"
    target = _as_float(take_profit)
    if target is not None and target > 0 and price >= target:
        return "take_profit"
    return None


def _has_usable_level(stop_loss: Any, take_profit: Any) -> bool:
    """Whether a recorded position carries at least one level that can trigger.

    A level is usable only when it is a finite, positive number — the same
    rule :func:`_exit_reason` applies when it compares a price against it. A
    missing, malformed or non-positive level is **not** a trigger and **not** a
    protection: a position whose levels are all unusable is an unguarded
    position, and the exit policy must say so rather than treat it as watched.
    """
    for level in (stop_loss, take_profit):
        number = _as_float(level)
        if number is not None and number > 0:
            return True
    return False


def _exit_order_id(symbol: str, price: float, reason: str, episode: int = 0) -> str:
    """A deterministic, broker-safe id for one exit **episode**.

    Derived from the symbol, the trigger reason and the episode number, so
    every attempt to close the same position for the same reason carries the
    **same** id. That is what makes the exit idempotent: a price that ticks
    between two cycles, or a worker that restarts mid-exit, reconciles the
    order it already placed instead of placing a second one for the same
    position.

    The price is deliberately **not** part of the id. It used to be, which
    meant every price change minted a fresh id and therefore a fresh order — so
    a stop that stayed triggered across several cycles produced a stream of
    exit orders for one position. The broker's long-only guard bounds the
    damage, but a stream of exits is still the wrong behaviour: it spends the
    daily-order budget, and it would also close a position that a *new* entry
    had legitimately re-opened.

    ``episode`` distinguishes one *holding* of a symbol from a later one: a
    position that was closed and then re-opened is a new episode and gets its
    own id, so the broker does not reconcile the old episode's order instead
    of placing a new exit.

    ``price`` is still accepted so a caller does not have to change, but it no
    longer affects the id.
    """
    material = f"exit:{symbol.upper()}:{reason}:{episode}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"ptx-{digest}"


def _positive_finite(value: Any) -> float | None:
    """``value`` as a finite, positive float, or ``None`` for anything else."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


_SNAPSHOT_FIELD_PREFERENCE: tuple[str, ...] = ("latestTrade", "latestQuote", "minuteBar")
"""How an exact timestamp tie between snapshot fields is broken.

The newest observation wins; this order only settles an exact tie, so the choice
stays deterministic. It is *not* the selection rule — a fresher quote beats an
older trade. ``dailyBar``/``prevDailyBar`` are deliberately absent: a daily bar is
not a current price and is never a candidate.
"""


def _snapshot_candidates(snapshot: Any) -> list[SnapshotObservation]:
    """Every usable current observation in a snapshot, in preference order.

    A candidate exists only when its price is finite and positive: for a quote that
    means both the bid and the ask are valid, so the midpoint is well-defined. Each
    candidate carries the timestamp of its *own* field, never another field's.
    ``dailyBar``/``prevDailyBar`` are never read.
    """
    if not isinstance(snapshot, dict):
        return []
    candidates: list[SnapshotObservation] = []

    trade = snapshot.get("latestTrade") or {}
    if isinstance(trade, dict):
        price = _positive_finite(trade.get("p"))
        if price is not None:
            candidates.append(SnapshotObservation(price, _stamp(trade.get("t")), "latestTrade"))

    quote = snapshot.get("latestQuote") or {}
    if isinstance(quote, dict):
        bid = _positive_finite(quote.get("bp"))
        ask = _positive_finite(quote.get("ap"))
        if bid is not None and ask is not None:
            candidates.append(
                SnapshotObservation((bid + ask) / 2.0, _stamp(quote.get("t")), "latestQuote")
            )

    minute = snapshot.get("minuteBar") or {}
    if isinstance(minute, dict):
        price = _positive_finite(minute.get("c"))
        if price is not None:
            candidates.append(SnapshotObservation(price, _stamp(minute.get("t")), "minuteBar"))

    return candidates


def _select_snapshot_observation(
    candidates: list[SnapshotObservation],
) -> SnapshotObservation | None:
    """The newest verifiable observation, or the best unverifiable one, or ``None``.

    A candidate with a usable timestamp always outranks one without, whatever the
    preference order: an unverifiable field cannot be shown to be current, so it
    must not displace a field that can. Among usable timestamps the newest wins,
    and an exact tie is broken by :data:`_SNAPSHOT_FIELD_PREFERENCE`. When no
    candidate has a usable timestamp, the first in preference order is returned so
    the freshness check can name the exact field and reason it refused.
    """
    if not candidates:
        return None
    rank = {field: index for index, field in enumerate(_SNAPSHOT_FIELD_PREFERENCE)}

    def sort_key(observation: SnapshotObservation) -> tuple[int, float, int]:
        parsed = _parse_utc(observation.as_of) if observation.as_of is not None else None
        if parsed is None:
            # Unverifiable: ranked below every verifiable candidate.
            return (0, 0.0, -rank.get(observation.field, len(rank)))
        return (1, parsed.timestamp(), -rank.get(observation.field, len(rank)))

    return max(candidates, key=sort_key)


@dataclass
class ListDecisionSource:
    """Yields a fixed, ordered list of decisions, one per call, then ``None``.

    The order is the list's order, so a re-run yields the same decisions in the
    same sequence — which is what makes the worker's per-decision idempotency
    meaningful. The source holds a cursor; :meth:`reset` rewinds it.
    """

    decisions: list[TradingDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = 0

    def next_decision(self) -> TradingDecision | None:
        if self._index >= len(self.decisions):
            return None
        decision = self.decisions[self._index]
        self._index += 1
        return decision

    def reset(self) -> None:
        self._index = 0


@dataclass
class PaperTradingConfig:
    """The worker's configuration. Safe by default; every limit is explicit.

    ``order_notional`` is the target cash size of one order and is capped by
    ``limits.max_order_notional``; the two are checked together, so a large
    ``order_notional`` cannot silently exceed the notional cap.
    """

    strategy_id: str = "alpaca-paper"
    order_notional: float = 25.0
    """Target cash per order. Deliberately small, so a 50 USD paper account is not
    drained by one order."""
    whole_shares: bool = WHOLE_SHARES
    """Whether to size to whole shares, derived from the one parity constant.

    ``False`` (the default) allows a fractional quantity, which is what lets a
    $25 order on a $200+ name buy a positive quantity instead of flooring to zero.
    This is not a second policy: it reads :data:`tradingagents.risk.paper_parity.WHOLE_SHARES`,
    the same source the TJR backtest's ``fractional_shares`` comes from, so the
    worker and the backtest size identically. A broker that does not support
    fractional shares will reject the order rather than receive a fabricated zero
    quantity (see :meth:`PaperTradingWorker._size`). The $25 notional cap and the
    long-only rule are unchanged by this setting.
    """
    base_url: str = PAPER_BASE_URL
    limits: PaperRiskLimits = field(default_factory=PaperRiskLimits)
    state_path: str | os.PathLike | None = None
    poll_interval_seconds: float = 60.0
    market_data_provider: str = PROVIDER_BAR_OPEN
    """Which price seam this worker is configured to use — ``bar_open`` (historical,
    the default) or ``alpaca_snapshot`` (a current price).

    Recorded on the config so a read-only status view can name the *configured*
    provider without building a worker, and so the cycle result and the dashboard
    can never disagree about which seam is in use. It is a label of the wiring, not
    a switch: the seam itself is passed in as ``market_data``.
    """
    market_data_max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS
    """The maximum age of a snapshot price, in seconds, when the snapshot seam is
    used. ``None`` disables the age check (not recommended). Ignored by
    ``bar_open``, which is historical by construction."""
    decision_source: str = DECISION_SOURCE_HISTORY
    """Which decision source this worker draws on — ``history`` (the default) or
    ``current_market``.

    Recorded on the config so a read-only status view can name the *configured*
    source without building a worker. It is a label of the wiring, not a switch:
    the source itself is passed in as ``decision_source`` to the worker.
    ``history`` is the default, so existing behaviour is unchanged unless an
    operator opts into the current-market path explicitly.
    """
    symbol_mappings: tuple[SymbolMapping, ...] = field(default_factory=tuple)
    """Validated symbol mappings between market-data and execution venues.

    When configured, these mappings are used to translate decision symbols to
    execution symbols, and the execution symbols are enforced via
    ``limits.allowed_symbols``. An empty tuple means no mapping is configured
    and no additional symbol validation is performed beyond the existing
    ``allowed_symbols`` check.
    """
    max_price_deviation: float = DEFAULT_MAX_PRICE_DEVIATION
    """The largest permitted gap between the decision price and the execution
    price, as a fraction.

    The decision price comes from the market-data venue and the order is priced
    on the execution venue, so the two are different instruments' prices. A gap
    within this bound is ordinary market movement; a gap beyond it means the
    order would no longer be the trade the strategy decided on, and it is refused
    (``NO_TRADE_PRICE_DEVIATION``) rather than placed.
    """
    execution_price_seam: Any | None = None
    """The **execution-side** price seam, for the cross-venue crypto path.

    ``None`` means "no separate execution price": the worker's own market-data
    seam prices the order, which is correct for the single-venue paths (the
    equity ``alpaca_snapshot`` seam already reads the venue that will execute).
    When it is set, an order is sized and priced from this seam, in the
    *execution* symbol, and the market-data price is used only as the reference
    the deviation is measured against.

    A configured seam that cannot produce a fresh price refuses the order
    (``NO_TRADE_EXECUTION_PRICE_UNAVAILABLE``); the market-data price is never
    used as a substitute.
    """
    crypto: bool = False
    """Whether this worker trades **crypto** on Alpaca.

    Alpaca's crypto orders accept only the time-in-force values ``gtc`` and
    ``ioc``; the equity vocabulary's ``day`` is refused with HTTP 422
    ``invalid crypto time_in_force``. This flag selects the crypto order rules
    for the order the worker submits, so the equity path keeps the time-in-force
    it has always sent.

    It is ``False`` by default, so every existing wiring is unchanged. It does
    not select a strategy, a source, a seam or a risk limit — only the order
    rules Alpaca applies to the asset class being traded.
    """

    def __post_init__(self) -> None:
        if not (self.order_notional > 0):
            raise WorkerError("order_notional must be positive")
        if self.order_notional > self.limits.max_order_notional:
            raise WorkerError(
                f"order_notional {self.order_notional} exceeds the notional cap "
                f"{self.limits.max_order_notional}; lower one of them"
            )
        if self.poll_interval_seconds <= 0:
            raise WorkerError("poll_interval_seconds must be positive")
        if self.market_data_provider not in MARKET_DATA_CHOICES:
            raise WorkerError(
                f"unknown market_data_provider {self.market_data_provider!r}; expected "
                f"one of {', '.join(MARKET_DATA_CHOICES)}"
            )
        if self.decision_source not in DECISION_SOURCE_CHOICES:
            raise WorkerError(
                f"unknown decision_source {self.decision_source!r}; expected one of "
                f"{', '.join(DECISION_SOURCE_CHOICES)}"
            )
        if self.market_data_max_age_seconds is not None and not (
            self.market_data_max_age_seconds > 0
        ):
            raise WorkerError("market_data_max_age_seconds must be positive or None")
        if not math.isfinite(self.max_price_deviation) or self.max_price_deviation < 0:
            raise WorkerError(
                "max_price_deviation must be a finite, non-negative fraction, got "
                f"{self.max_price_deviation!r}"
            )
        # The endpoint is validated at configuration time, so a live host is
        # refused before a broker is ever constructed.
        assert_paper_endpoint(self.base_url)

        # If symbol mappings are provided, populate allowed_symbols in limits
        if self.symbol_mappings:
            allowed = get_allowed_symbols_for_broker(list(self.symbol_mappings))
            if allowed:
                # Create new limits with allowed_symbols populated
                self.limits = PaperRiskLimits(
                    max_order_notional=self.limits.max_order_notional,
                    max_open_positions=self.limits.max_open_positions,
                    max_daily_orders=self.limits.max_daily_orders,
                    allowed_symbols=allowed,
                    allow_short=self.limits.allow_short,
                    buying_power_fraction=self.limits.buying_power_fraction,
                )


# ---------------------------------------------------------------------------
# Durable worker state
# ---------------------------------------------------------------------------


class WorkerStateError(RuntimeError):
    """The persisted worker state is malformed, truncated, or a newer version."""


def _as_int(value: Any, default: int) -> int:
    """Coerce a persisted counter, tolerating a missing or malformed value.

    Additive fields must not make an otherwise readable state file unusable, so a
    bad counter falls back to the default rather than raising.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a persisted number, tolerating a missing or malformed value.

    Non-finite values (NaN/Inf) are refused rather than propagated: a NaN baseline
    would poison every daily-P&L subtraction silently, so it is read back as
    ``default`` (unavailable) instead.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".worker-state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass
class WorkerState:
    """Everything the worker needs to resume without duplicating an order.

    ``status`` is persisted but is always read back as ``STOPPED`` by
    :meth:`PaperTradingWorker.load`: a restart must never resume trading on its
    own. What *is* preserved is the order memory — every ``client_order_id`` the
    worker has seen, its last known broker status, the applied-fill set, the
    positions last read, and the last processed decision id — so a resumed worker
    reconciles instead of re-submitting.
    """

    schema_version: int = WORKER_STATE_SCHEMA_VERSION
    strategy_id: str = "alpaca-paper"
    status: str = STATUS_STOPPED
    last_processed_decision_id: str | None = None
    last_error: str | None = None
    last_cycle_at: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    orders: dict[str, dict] = field(default_factory=dict)
    applied_fills: set[str] = field(default_factory=set)
    positions: dict[str, dict] = field(default_factory=dict)
    positions_at: str | None = None
    positions_checked_at: str | None = None
    """When the broker's positions were last reconciled into this state.

    Distinct from ``positions_at`` (when the snapshot was *written*) only in
    emphasis: this is the operational "last successful position synchronization"
    the dashboard reports. Additive and optional."""
    account: dict[str, Any] = field(default_factory=dict)
    initial_equity: float | None = None
    """The account equity first observed for the current UTC day — the baseline a
    daily P&L is measured from. Set once per day, so "daily P&L" is the change
    since the day opened, not since the process started. Additive and optional: an
    older state file reads back as ``None`` and the dashboard reports the daily P&L
    as unavailable rather than fabricating a zero."""
    day: str | None = None
    """The UTC day ``initial_equity`` belongs to (``YYYY-MM-DD``). Additive."""
    last_account_at: str | None = None
    """When the account (cash, equity, buying power) was last read from the broker.
    Additive and optional."""
    last_strategy_evaluation_at: str | None = None
    """When the strategy was last *evaluated* — a cycle reached the decision source
    and asked it for a decision. This is distinct from a decision being *produced*:
    a cycle that finds no decision still evaluates. Additive and optional."""
    last_no_decision_reason: str | None = None
    """Why the most recent evaluation produced no order — the decision source's own
    skip reason, or ``None`` when the last evaluation yielded a decision. Recorded
    so a RUNNING-but-idle worker is explainable rather than silent. Additive."""
    last_no_trade_code: str | None = None
    """The stable code (from :data:`NO_TRADE_REASONS`) behind the most recent
    no-order cycle, paired with ``last_no_decision_reason`` as its human detail.
    ``None`` when the last cycle placed an order. Additive and optional: an older
    state file reads back as ``None``, which a view reports as unknown rather than
    as "no problem"."""
    last_no_trade_decision_id: str | None = None
    """The decision id whose most recent processing produced no order.

    Recorded so a later cycle that finds the same decision already offered — and
    so reports the current-market source's generic "no new data" — can still name
    *which* decision was left un-traded and *why*, instead of letting the generic
    code erase the honest reason. Additive and optional."""
    last_no_trade_decision_code: str | None = None
    """The stable no-trade code that decision produced (for example
    ``NO_POSITION``). Paired with :attr:`last_no_trade_decision_id`. Additive."""
    exits: dict[str, dict] = field(default_factory=dict)
    """Positions the exit policy has already acted on, keyed by symbol, with the
    reason and time. The durable "do not exit this position twice" memory. Additive
    and optional."""
    exit_attempts: dict[str, dict] = field(default_factory=dict)
    """Exit orders the worker has already placed, keyed by symbol.

    Distinct from :attr:`exits`: an *attempt* is an exit order that reached the
    broker, while an *exit* is a position that is actually closed. A partially
    filled exit is an attempt that is not yet an exit, and the two must not be
    confused — otherwise a remainder still held would look closed and its
    protection would go silent.

    This is the durable half of the duplicate-exit guard: while an attempt is
    outstanding for a symbol, the exit policy reconciles that attempt instead of
    placing another order, so a price that keeps moving (or a worker that
    restarts) cannot turn one stop trigger into a stream of orders. Additive
    and optional: an older state file without it reads back as "no attempt
    known"."""
    exit_attempt_counter: int = 0
    """How many exit episodes this worker has run.

    A position that is closed and later **re-opened** is a new holding, and it
    needs its own exit id — otherwise the new position would inherit the old
    episode's id and the broker would reconcile the old order instead of
    placing a new exit. The counter is part of the exit id, so each episode of
    a position gets a distinct, deterministic id."""
    stops: dict[str, dict] = field(default_factory=dict)
    """The stop-loss / take-profit recorded for each open position the worker
    opened, keyed by symbol. Written when an order fills from a decision that
    carried them, and read by the exit policy. A position with no entry here has no
    recorded exit level, so no exit is fabricated for it. Additive and optional."""
    cycles_completed: int = 0
    """Cycles the loop has run since the last start. Additive and optional: an
    older state file without the field reads back as ``0``, and a reader of an
    older file ignores it, so the schema version does not need to change."""
    processed_decisions: set[str] = field(default_factory=set)
    """Every decision id the worker has already acted on — the durable "do not
    order this decision again" set.

    ``last_processed_decision_id`` records only the most recent decision, which is
    enough to skip an immediate repeat but not to tell, after a restart, whether an
    *earlier* decision was already submitted. This set is that memory: the worker
    adds a decision when it submits an order for it, reconciles an existing one, or
    the broker definitively rejects it, and a resumed worker skips every id in the
    set. It is additive and optional: an older state file without the field reads
    back as "nothing processed", so the schema version does not change."""
    decision_outcomes: dict[str, str] = field(default_factory=dict)
    """The settled outcome of each decision the worker has acted on, by id.

    ``"ORDERED"`` for a decision that produced an order (submitted or reconciled),
    otherwise the :data:`NO_TRADE_REASONS` code that says why it produced none
    (``NO_POSITION``, ``RISK_REJECTED``, ``MARKET_CLOSED``, …). The durable record
    behind the dashboard's BUY/SELL/HOLD-and-no-trade statistics, so a restart does
    not reset the counts. Distinct from :attr:`processed_decisions`, which is only
    a membership set with no reason. Additive and optional: an older state file
    reads back as "no outcomes known", and the schema version does not change."""
    ordered_decisions: set[str] = field(default_factory=set)
    """The subset of :attr:`processed_decisions` that actually produced an order.

    ``processed_decisions`` means "do not act on this id again", which is correct
    for a decision that was ordered *and* for one the broker definitively refused.
    It is the wrong signal for "an order exists", though: a refusal is recorded
    there but places nothing, so a source that filters on it cannot tell a settled
    order from a no-trade and may report a fully-consumed session for a decision
    that never traded. This set is the order-only memory, added at the two sites
    that genuinely reach the broker (a submitted order, and a reconciled existing
    one), so a caller can ask whether a decision actually produced an order without
    inferring it from the refusal-inclusive set. Additive and optional: an older
    state file without the field reads back as "no order known", and the schema
    version does not change."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "last_processed_decision_id": self.last_processed_decision_id,
            "last_error": self.last_error,
            "last_cycle_at": self.last_cycle_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "orders": {k: dict(v) for k, v in sorted(self.orders.items())},
            "applied_fills": sorted(self.applied_fills),
            "positions": {k: dict(v) for k, v in sorted(self.positions.items())},
            "positions_at": self.positions_at,
            "positions_checked_at": self.positions_checked_at,
            "account": dict(self.account),
            "initial_equity": self.initial_equity,
            "day": self.day,
            "last_account_at": self.last_account_at,
            "last_strategy_evaluation_at": self.last_strategy_evaluation_at,
            "last_no_decision_reason": self.last_no_decision_reason,
            "last_no_trade_code": self.last_no_trade_code,
            "last_no_trade_decision_id": self.last_no_trade_decision_id,
            "last_no_trade_decision_code": self.last_no_trade_decision_code,
            "exits": {k: dict(v) for k, v in sorted(self.exits.items())},
            "exit_attempts": {k: dict(v) for k, v in sorted(self.exit_attempts.items())},
            "exit_attempt_counter": self.exit_attempt_counter,
            "stops": {k: dict(v) for k, v in sorted(self.stops.items())},
            "cycles_completed": self.cycles_completed,
            "processed_decisions": sorted(self.processed_decisions),
            "decision_outcomes": dict(sorted(self.decision_outcomes.items())),
            "ordered_decisions": sorted(self.ordered_decisions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> WorkerState:
        if not isinstance(payload, dict):
            raise WorkerStateError("the worker state must be a JSON object")
        version = payload.get("schema_version")
        if version != WORKER_STATE_SCHEMA_VERSION:
            raise WorkerStateError(
                f"unsupported worker state schema_version {version!r}; this build reads "
                f"version {WORKER_STATE_SCHEMA_VERSION}"
            )
        status = payload.get("status", STATUS_STOPPED)
        if status not in WORKER_STATUSES:
            raise WorkerStateError(f"unknown worker status {status!r}")
        orders = payload.get("orders") or {}
        if not isinstance(orders, dict):
            raise WorkerStateError("worker state 'orders' must be a mapping")
        return cls(
            schema_version=version,
            strategy_id=payload.get("strategy_id", "alpaca-paper"),
            status=status,
            last_processed_decision_id=payload.get("last_processed_decision_id"),
            last_error=payload.get("last_error"),
            last_cycle_at=payload.get("last_cycle_at"),
            started_at=payload.get("started_at"),
            stopped_at=payload.get("stopped_at"),
            orders={k: dict(v) for k, v in orders.items()},
            applied_fills=set(payload.get("applied_fills") or []),
            positions={k: dict(v) for k, v in (payload.get("positions") or {}).items()},
            positions_at=payload.get("positions_at"),
            positions_checked_at=payload.get("positions_checked_at"),
            account=dict(payload.get("account") or {}),
            initial_equity=_as_float(payload.get("initial_equity")),
            day=payload.get("day"),
            last_account_at=payload.get("last_account_at"),
            last_strategy_evaluation_at=payload.get("last_strategy_evaluation_at"),
            last_no_decision_reason=payload.get("last_no_decision_reason"),
            last_no_trade_code=payload.get("last_no_trade_code"),
            last_no_trade_decision_id=payload.get("last_no_trade_decision_id"),
            last_no_trade_decision_code=payload.get("last_no_trade_decision_code"),
            exits={k: dict(v) for k, v in (payload.get("exits") or {}).items()},
            exit_attempts={
                k: dict(v) for k, v in (payload.get("exit_attempts") or {}).items()
            },
            exit_attempt_counter=int(payload.get("exit_attempt_counter") or 0),
            stops={k: dict(v) for k, v in (payload.get("stops") or {}).items()},
            cycles_completed=_as_int(payload.get("cycles_completed"), 0),
            processed_decisions=set(payload.get("processed_decisions") or []),
            decision_outcomes={
                str(k): str(v)
                for k, v in (payload.get("decision_outcomes") or {}).items()
                if isinstance(v, str)
            },
            ordered_decisions=set(payload.get("ordered_decisions") or []),
        )


class WorkerStateStore:
    """Loads and saves :class:`WorkerState`, atomically and version-checked."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def load(self) -> WorkerState:
        if not self.path.exists():
            return WorkerState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkerStateError(f"worker state {self.path} is not usable: {exc}") from exc
        return WorkerState.from_dict(payload)

    def save(self, state: WorkerState) -> Path:
        _atomic_write(self.path, json.dumps(state.to_dict(), indent=2) + "\n")
        return self.path


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


@dataclass
class CycleResult:
    """What one :meth:`PaperTradingWorker.run_once` produced.

    ``reference_price`` is the exact price the order was sized from, and
    ``market_data_*`` names the seam that produced it, so the dashboard, the CLI
    and any journal entry can report the actual price used rather than a guess.
    None of these fields carries a credential.
    """

    ran: bool
    decision_id: str | None = None
    client_order_id: str | None = None
    order: BrokerOrder | None = None
    skipped_reason: str | None = None
    error: str | None = None
    reference_price: float | None = None
    market_data_provider: str | None = None
    market_data_kind: str | None = None
    market_data_status: str | None = None
    reason_code: str | None = None
    """Why no order was placed, from :data:`NO_TRADE_REASONS`. ``None`` when the
    cycle placed an order (or had nothing to report), so a UI can key off a stable
    code rather than parsing ``skipped_reason`` prose."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "decision_id": self.decision_id,
            "client_order_id": self.client_order_id,
            "order": self.order.to_dict() if self.order is not None else None,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
            "reference_price": self.reference_price,
            "market_data_provider": self.market_data_provider,
            "market_data_kind": self.market_data_kind,
            "market_data_status": self.market_data_status,
            "reason_code": self.reason_code,
        }


def _decision_statistics(state: WorkerState) -> dict[str, Any]:
    """The worker's own settled-outcome tally, from :attr:`WorkerState.decision_outcomes`.

    Counts how many decisions produced an order and how many did not, split by the
    :data:`NO_TRADE_REASONS` code that says why. Derived from persisted state, so a
    restart does not reset it, and it is a *worker* tally: it counts what the worker
    did, not what the strategy rated. The source's own rating tally (BUY/SELL/HOLD)
    is reported separately, because only the source sees a rating the worker then
    refused to trade.
    """
    outcomes = state.decision_outcomes or {}
    ordered = sum(1 for value in outcomes.values() if value == DECISION_OUTCOME_ORDERED)
    by_code: dict[str, int] = {}
    for value in outcomes.values():
        if value == DECISION_OUTCOME_ORDERED:
            continue
        by_code[value] = by_code.get(value, 0) + 1
    orders_by_side: dict[str, int] = {}
    for order in (state.orders or {}).values():
        side = str(order.get("side") or "").lower()
        if side:
            orders_by_side[side] = orders_by_side.get(side, 0) + 1
    return {
        "decisions_settled": len(outcomes),
        "orders_produced": ordered,
        "no_trade_total": len(outcomes) - ordered,
        "no_trade_by_code": by_code,
        "orders_by_side": orders_by_side,
        "risk_rejections": by_code.get(NO_TRADE_RISK_REJECTED, 0),
        "broker_rejections": by_code.get(NO_TRADE_BROKER_REJECTED, 0),
    }


def _position_protection_rows(state: WorkerState) -> dict[str, dict[str, Any]]:
    """The protection state of every open position, keyed by symbol.

    An open position is either **managed** — the worker opened it and recorded
    the strategy's stop and target for it, so the exit policy is watching it —
    or it is **unprotected**, and the reason says which kind of unprotected it
    is. A position the worker never opened (it was held before this worker
    existed, or it came from another tool) has no strategy decision behind it,
    so no stop or target can be derived for it and none is invented. A position
    whose recorded levels are missing or unusable is unprotected for the same
    reason: a level that cannot trigger is not a protection.

    The point of this view is that an unguarded position is never silent. It is
    reported, it carries a reason, and it says what an operator would have to
    do about it.

    A pure function of the persisted state, so the status view can build it
    without a worker — and therefore without a broker request.
    """
    rows: dict[str, dict[str, Any]] = {}
    for symbol, position in sorted((state.positions or {}).items()):
        if not isinstance(position, dict):
            continue
        try:
            quantity = float(position.get("quantity") or 0.0)
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity <= 0:
            continue
        level = (state.stops or {}).get(symbol)
        level = level if isinstance(level, dict) else {}
        stop_loss = _as_float(level.get("stop_loss"))
        take_profit = _as_float(level.get("take_profit"))
        usable_stop = stop_loss if stop_loss is not None and stop_loss > 0 else None
        usable_target = (
            take_profit if take_profit is not None and take_profit > 0 else None
        )
        exited = symbol in (state.exits or {})
        if exited:
            protection_state = "EXITED"
            reason = "EXIT_RECORDED"
            protected = True
        elif usable_stop is not None or usable_target is not None:
            protection_state = "TJR_MANAGED"
            reason = "RECORDED_EXIT_LEVELS"
            protected = True
        elif level:
            # Levels were recorded but none of them can trigger.
            protection_state = "UNPROTECTED"
            reason = "UNUSABLE_EXIT_LEVELS"
            protected = False
        else:
            protection_state = "UNPROTECTED"
            reason = "NO_RECORDED_EXIT_LEVELS"
            protected = False
        rows[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "protected": protected,
            "protection_state": protection_state,
            "reason": reason,
            "stop_loss": usable_stop,
            "take_profit": usable_target,
            "decision_id": level.get("decision_id"),
            "canonical_symbol": level.get("canonical_symbol"),
            "exit_reason": (
                (state.exits or {}).get(symbol, {}).get("reason") if exited else None
            ),
        }
    return rows


def worker_status_view(
    *,
    state: WorkerState,
    config: PaperTradingConfig,
    loop_running: bool = False,
    next_cycle_at: str | None = None,
    last_cycle: CycleResult | None = None,
    market_data_identity: dict[str, Any] | None = None,
    decision_source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The secret-free status of a paper worker, from its state and config alone.

    Split out from :meth:`PaperTradingWorker.status` so a read-only caller can
    report the persisted state **without building a worker** (and therefore without
    constructing the decision source or touching the broker). Every field is
    derived from data, so a build cannot freeze a live view: the decision source is
    never involved, which is the whole point.

    ``market_data_identity`` is the effective seam's self-description when a worker
    exists; with no worker it is ``None`` and the *configured* provider from
    ``config`` is reported instead, so the view always names a provider and never
    implies a current price it cannot prove.

    ``decision_source_identity`` is the decision source's own self-description
    when a worker exists (the current-market source reports its last market state,
    session and last-fetched data); with no worker it is ``None`` and the
    *configured* ``decision_source`` label from ``config`` is reported instead.
    """
    identity = dict(market_data_identity or {})
    configured_provider = config.market_data_provider
    # Prefer the live seam's own identity; fall back to the configured provider, so
    # a stopped worker still reports which seam it *would* use.
    identity.setdefault("market_data_provider", configured_provider)
    identity.setdefault("market_data_kind", None)
    source_identity = dict(decision_source_identity or {})
    source_identity.setdefault("decision_source_kind", config.decision_source)
    return {
        "banner": BANNER,
        "mode": "ALPACA_PAPER",
        "live": False,
        "status": state.status,
        "strategy_id": state.strategy_id,
        "endpoint": config.base_url,
        "paper_endpoint_only": config.base_url == PAPER_BASE_URL,
        "market_data_provider_configured": configured_provider,
        "market_data_max_age_seconds": config.market_data_max_age_seconds,
        "decision_source_configured": config.decision_source,
        **identity,
        **source_identity,
        "last_processed_decision_id": state.last_processed_decision_id,
        "last_error": state.last_error,
        "last_cycle_at": state.last_cycle_at,
        "last_strategy_evaluation_at": state.last_strategy_evaluation_at,
        "last_no_decision_reason": state.last_no_decision_reason,
        "last_no_trade_code": state.last_no_trade_code,
        "last_no_trade_decision_id": state.last_no_trade_decision_id,
        "last_no_trade_decision_code": state.last_no_trade_decision_code,
        "started_at": state.started_at,
        "stopped_at": state.stopped_at,
        "loop_running": loop_running,
        "next_cycle_at": next_cycle_at,
        "cycles_completed": state.cycles_completed,
        "poll_interval_seconds": config.poll_interval_seconds,
        "orders": {k: dict(v) for k, v in sorted(state.orders.items())},
        "positions": {k: dict(v) for k, v in sorted(state.positions.items())},
        "positions_at": state.positions_at,
        "positions_checked_at": state.positions_checked_at,
        "account": dict(state.account),
        "account_at": state.last_account_at,
        "initial_equity": state.initial_equity,
        "day": state.day,
        "exits": {k: dict(v) for k, v in sorted(state.exits.items())},
        "stops": {k: dict(v) for k, v in sorted(state.stops.items())},
        "position_protection": {
            k: dict(v) for k, v in sorted(_position_protection_rows(state).items())
        },
        "last_cycle": last_cycle.to_dict() if last_cycle is not None else None,
        "decision_statistics": _decision_statistics(state),
        "risk_limits": {
            "max_order_notional": config.limits.max_order_notional,
            "max_open_positions": config.limits.max_open_positions,
            "max_daily_orders": config.limits.max_daily_orders,
            "allow_short": config.limits.allow_short,
            "order_notional": config.order_notional,
            "whole_shares": config.whole_shares,
        },
    }


class PaperTradingWorker:
    """A start/stop paper-trading worker over one Alpaca paper broker.

    The broker is created lazily by :meth:`preflight` (so construction performs no
    request and needs no credential), through a factory that defaults to the real
    :class:`AlpacaPaperBroker`. A test injects a factory that returns a
    broker with a mocked transport.
    """

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        decision_source: DecisionSource,
        market_data: MarketData,
        broker_factory: Callable[[], AlpacaPaperBroker] | None = None,
        root: str | os.PathLike | None = None,
    ) -> None:
        self.config = config
        self.decision_source = decision_source
        self.market_data = market_data
        self._broker_factory = broker_factory or self._default_broker_factory
        state_path = config.state_path or (
            (Path(root) / "worker_state.json") if root is not None else None
        )
        self._store = WorkerStateStore(state_path) if state_path else None
        self._broker: AlpacaPaperBroker | None = None
        # A restart reads the state back as STOPPED, never RUNNING.
        self.state = self._store.load() if self._store else WorkerState()
        self.state.status = STATUS_STOPPED
        self.state.strategy_id = config.strategy_id
        self.last_cycle: CycleResult | None = None
        # The decision id the source last handed this worker in the current run,
        # used only to keep the honest no-trade reason visible once the source
        # stops re-offering it. In-memory: a restart is a fresh observation.
        self._last_offered_decision_id: str | None = None
        # Build symbol mapping lookup for crypto path (canonical -> execution)
        self._symbol_map: dict[str, SymbolMapping] = {
            m.canonical: m for m in config.symbol_mappings
        }
        # The reverse index (execution symbol -> market-data symbol) is what the
        # exit policy needs: a position is held and exited under the *execution*
        # symbol (``BTC/USD``), while the stop/target is monitored against the
        # *market-data* symbol (``BTCUSDT``). Built once, from the same validated
        # mappings, so the two can never disagree.
        self._execution_to_market_data: dict[str, str] = {
            m.execution_symbol.upper(): m.market_data_symbol.upper()
            for m in config.symbol_mappings
            if m.execution_symbol and m.market_data_symbol
        }
        # The loop. `_lock` serialises control transitions (start/stop) and is the
        # same lock a cycle holds while it runs, so a stop cannot interleave a
        # start with a cycle mid-flight. `_stop_event` is the loop's stop signal.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._loop: threading.Thread | None = None
        self._next_cycle_at: str | None = None

    # -- construction helpers --------------------------------------------

    def _default_broker_factory(self) -> AlpacaPaperBroker:
        """Build the real paper broker from the environment's credentials.

        ``state_path`` is the same directory as the worker state, so the broker's
        own order memory and the worker's live side by side. Credentials are read
        by the broker from ``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET``;
        this factory never touches them.
        """
        state_path = self._store.path.with_name("broker_state.json") if self._store else None
        return AlpacaPaperBroker(
            base_url=self.config.base_url,
            limits=self.config.limits,
            state_path=state_path,
        )

    def _save(self) -> None:
        if self._store:
            self._store.save(self.state)

    def _is_processed(self, decision_id: str) -> bool:
        """Whether this decision has already been acted on (durably).

        The guard is the *durable processed set* for the historical source, whose
        decisions are one-shot and bounded: a refused historical decision is
        deliberately not retried every cycle. The current-market and TJR sources
        are the opposite — their decision stands for a whole session and is
        re-offered until it trades — so a decision they hand out is "done" only
        when it actually produced an order. Reading the refusal-inclusive processed
        set there would let a stale no-order entry mask a live decision (the
        reported ``CURRENT_MARKET_NO_NEW_DATA`` with no order). ``ordered_decisions``
        is the order-only memory, so those session-scoped paths use it.
        """
        if self.config.decision_source in (
            DECISION_SOURCE_CURRENT_MARKET,
            DECISION_SOURCE_TJR,
        ):
            return self.has_order_for(decision_id)
        return (
            decision_id in self.state.processed_decisions
            or decision_id == self.state.last_processed_decision_id
        )

    def _mark_processed(self, decision_id: str) -> None:
        """Remember a decision as done, so no cycle — and no restart — repeats it."""
        self.state.last_processed_decision_id = decision_id
        self.state.processed_decisions.add(decision_id)

    def _mark_ordered(self, decision_id: str) -> None:
        """Remember a decision as having *produced an order* (not merely processed).

        Distinct from :meth:`_mark_processed`, which also records a decision the
        broker refused: a refusal places nothing, so it is "do not retry", not "an
        order exists". A caller that needs the latter (the current-market source,
        which must not treat a no-trade as a consumed session) asks
        :meth:`has_order_for` instead of reading the processed set.
        """
        self.state.ordered_decisions.add(decision_id)
        self.state.decision_outcomes[decision_id] = DECISION_OUTCOME_ORDERED

    def has_order_for(self, decision_id: str) -> bool:
        """Whether this decision actually produced an order the worker knows of.

        The order-only counterpart to :meth:`_is_processed`: true only for a
        decision that reached the broker (submitted, or reconciled as already
        existing), never for one that was refused or produced no trade.
        """
        return decision_id in self.state.ordered_decisions

    def _remember_no_trade(self, decision_id: str, code: str) -> None:
        """Remember which decision was left un-traded and the code that says why.

        Paired with :meth:`_mark_ordered` as the honest counterpart: an ordered
        decision is forgotten here (it is no longer un-traded), and a no-trade
        decision is recorded so a later cycle that only sees "no new data" can
        still name the decision and its reason.
        """
        self.state.last_no_trade_decision_id = decision_id
        self.state.last_no_trade_decision_code = code
        self.state.decision_outcomes[decision_id] = code

    # -- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A secret-free view of the worker, for the dashboard and the CLI."""
        return worker_status_view(
            state=self.state,
            config=self.config,
            loop_running=self.loop_running,
            next_cycle_at=self._next_cycle_at,
            last_cycle=self.last_cycle,
            market_data_identity=market_data_identity(self.market_data),
            decision_source_identity=decision_source_identity(self.decision_source),
        )

    # -- lifecycle --------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        """Run every start-time check. Returns the checks; raises on the first failure.

        Checks, in order: the broker connects (paper endpoint + credentials), the
        account status is usable, buying power is positive, the market-data source
        is available, and the risk configuration is valid. None of these places an
        order.
        """
        checks: list[dict[str, Any]] = []

        broker = self._broker_factory()
        account = broker.connect()  # raises on a bad endpoint/credential/timeout
        checks.append({"check": "broker_connection", "ok": True, "detail": "connected"})

        if account.status.upper() != "ACTIVE":
            raise PreflightError("account_status", f"account status is {account.status!r}")
        if account.trading_blocked or account.account_blocked:
            raise PreflightError(
                "account_status", f"account restrictions: {', '.join(account.restrictions())}"
            )
        checks.append({"check": "account_status", "ok": True, "detail": account.status})

        if not (account.buying_power > 0):
            raise PreflightError(
                "buying_power", f"buying power is {account.buying_power!r}"
            )
        checks.append(
            {"check": "buying_power", "ok": True, "detail": f"{account.buying_power:.2f}"}
        )

        if not self.market_data.available():
            identity = market_data_identity(self.market_data)
            provider = identity.get("market_data_provider") or "the market-data source"
            detail = identity.get("market_data_status_detail")
            raise PreflightError(
                "market_data",
                f"{provider} reports no usable current price"
                + (f" ({detail})" if detail else ""),
            )
        checks.append(
            {
                "check": "market_data",
                "ok": True,
                "detail": market_data_identity(self.market_data).get("market_data_provider")
                or "available",
            }
        )

        # Constructing the limits already validated them; re-check the relationship
        # between the order size and the cap so a misconfiguration cannot start.
        if self.config.order_notional > self.config.limits.max_order_notional:
            raise PreflightError(
                "risk_config",
                f"order_notional {self.config.order_notional} exceeds the cap "
                f"{self.config.limits.max_order_notional}",
            )
        checks.append({"check": "risk_config", "ok": True, "detail": "valid"})

        # Read positions so a later sell is checked against a known holding.
        broker.positions(refresh=True)

        self._broker = broker
        equity = _as_float(account.equity)
        today = datetime.now(UTC).date().isoformat()
        if equity is not None and (self.state.day != today or self.state.initial_equity is None):
            self.state.day = today
            self.state.initial_equity = equity
        self.state.account = {
            "account_id": account.account_id,
            "status": account.status,
            "cash": _as_float(account.cash),
            "equity": equity,
            "buying_power": _as_float(account.buying_power),
            "currency": account.currency,
            "restrictions": account.restrictions(),
        }
        self.state.last_account_at = datetime.now(UTC).isoformat()
        self._save()
        return {"checks": checks, "account": self.state.account}

    def start(self) -> dict[str, Any]:
        """Run preflight, mark the worker ``RUNNING``, and launch the loop.

        Places no order: preflight only reads, and the loop waits one
        ``poll_interval_seconds`` before its first cycle, so a STOP right after
        START prevents even the first order.

        Refuses if a loop is already live (one loop per worker), and refuses if
        preflight fails — setting ``ERROR`` with the failing check recorded, never
        ``RUNNING``, and never starting the loop.
        """
        with self._lock:
            if self.loop_running:
                raise WorkerError("a worker is already running; stop it before starting another")
            self.state.status = STATUS_STARTING
            self.state.last_error = None
            self._save()
            try:
                result = self.preflight()
            except (BrokerError, PreflightError, WorkerError) as exc:
                self.state.status = STATUS_ERROR
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self._save()
                raise
            self.state.status = STATUS_RUNNING
            self.state.started_at = datetime.now(UTC).isoformat()
            self.state.stopped_at = None
            self.state.cycles_completed = 0
            self._save()
            self._start_loop()
        return {"status": self.state.status, **result}

    def stop(self) -> dict[str, Any]:
        """Stop gracefully: signal the loop, wait for the in-flight cycle, then stop.

        Closes no position and cancels no order. An order already submitted is left
        alone (it belongs to the broker now); its last known status stays in
        :meth:`status`. The in-flight cycle is allowed to finish under the lock
        rather than being killed, so the ledger and the state file stay consistent.
        """
        with self._lock:
            # Signal the loop first, so a cycle it is about to start is skipped.
            self._stop_event.set()
            thread = self._loop
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=self._join_timeout())
        with self._lock:
            self._loop = None
            self._next_cycle_at = None
            self.state.status = STATUS_STOPPED
            self.state.stopped_at = datetime.now(UTC).isoformat()
            self._save()
            return self.status()

    # -- the loop ---------------------------------------------------------

    @property
    def loop_running(self) -> bool:
        """Whether the background loop thread is alive."""
        thread = self._loop
        return bool(thread is not None and thread.is_alive())

    def _join_timeout(self) -> float:
        """How long ``stop()`` waits for the in-flight cycle to finish.

        One poll interval plus a small margin: a cycle does at most one broker
        round trip, so this is generous without blocking shutdown indefinitely.
        """
        return self.config.poll_interval_seconds + 5.0

    def _start_loop(self) -> None:
        """Spawn the single loop thread. Caller holds ``_lock``."""
        self._stop_event.clear()
        self._next_cycle_at = self._in_seconds(self.config.poll_interval_seconds)
        thread = threading.Thread(target=self._loop_body, name="paper-trading-loop", daemon=True)
        self._loop = thread
        thread.start()

    def _loop_body(self) -> None:
        """Wait one interval, run one cycle, repeat until stopped.

        The wait uses ``Event.wait`` so a stop interrupts it immediately instead of
        sleeping out the full interval, and so there is no busy loop. The interval
        is re-read from the config on every pass. The whole iteration — the cycle
        and its bookkeeping save — is wrapped, so nothing inside can kill the
        thread with an unhandled error.
        """
        while not self._stop_event.is_set():
            interval = self.config.poll_interval_seconds
            if self._stop_event.wait(interval):
                break
            try:
                self._run_cycle_locked()
                with self._lock:
                    if self._stop_event.is_set():
                        break
                    self.state.cycles_completed += 1
                    self._save()
                    self._next_cycle_at = self._in_seconds(self.config.poll_interval_seconds)
            except Exception as exc:  # noqa: BLE001 - a cycle failure must not crash the process
                self._fail_cycle(exc)
                break

    def _run_cycle_locked(self) -> None:
        """Run exactly one cycle, skipping it if a stop was requested first.

        Holds ``_lock`` for the duration, so ``stop()`` cannot observe the worker
        as stopped while a cycle is mid-flight, and a start/stop cannot interleave.
        """
        with self._lock:
            if self._stop_event.is_set():
                return
            self.run_once()

    def _fail_cycle(self, exc: BaseException) -> None:
        """Contain an unexpected cycle failure: record it, stop safely.

        The loop stops (no auto-retry of a possibly-submitted order), the state
        goes to ``ERROR``, and the process is left running. The persisted order
        memory is untouched, so a later manual start reconciles instead of
        re-submitting. A failure to write the error state is swallowed: the loop
        is stopping anyway, and the thread must not die with a traceback.
        """
        with self._lock:
            self.state.status = STATUS_ERROR
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self.state.stopped_at = datetime.now(UTC).isoformat()
            self._next_cycle_at = None
            with contextlib.suppress(Exception):
                self._save()

    def _in_seconds(self, seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()

    # -- one cycle --------------------------------------------------------

    def run_once(self) -> CycleResult:
        """Process at most one decision, placing at most one order.

        Refuses unless ``RUNNING``. A decision already processed is skipped
        (idempotent). A duplicate ``client_order_id`` is reconciled by the broker
        rather than re-submitted. A transient market-data failure is recorded and
        the worker stays ``RUNNING``; the cycle simply does nothing.
        """
        if self.state.status != STATUS_RUNNING:
            raise WorkerError(
                f"the worker is {self.state.status}; it must be RUNNING to process a "
                "decision (call start() first)"
            )
        broker = self._broker
        if broker is None:  # pragma: no cover - start() always sets it
            raise WorkerError("the worker has no broker; call start() first")

        self.state.last_cycle_at = datetime.now(UTC).isoformat()
        # Refresh the account and positions *before* deciding, so the view and the
        # risk checks work from the account's current figures rather than a
        # start-up snapshot. Best-effort: a read failure is recorded, never fatal.
        self._sync_observability()

        # The exit policy runs before a new entry: a stop or target that has
        # triggered is the most urgent action, and at most one order is placed per
        # cycle. It returns a result only when it actually acted (or was refused).
        exit_result = self._check_exits()
        if exit_result is not None:
            self.last_cycle = exit_result
            return exit_result

        decision = self.decision_source.next_decision()
        # The strategy was evaluated this cycle whether or not it produced a
        # decision: the source was asked, and answered. Recording the evaluation
        # separately from the decision is what lets the dashboard tell "the engine
        # is looking and finding nothing to do" from "the engine is not looking".
        self.state.last_strategy_evaluation_at = self.state.last_cycle_at
        if decision is None:
            reason, code = self._no_decision_outcome()
            self.state.last_no_decision_reason = reason
            self.state.last_no_trade_code = code
            # A cycle that produced no decision is still a settled outcome, so the
            # no-trade fields must name *this* cycle. Leaving them untouched would
            # let an earlier cycle's code (say BROKER_REJECTED) survive here and
            # read as the current cause — which is exactly what a stale "last
            # no-trade" looks like on the dashboard.
            decision_id = self._no_decision_id()
            if decision_id is not None:
                self._remember_no_trade(decision_id, code)
            else:
                # No decision was in hand, so there is no decision id to name: the
                # previous cycle's *decision* is no longer the current no-trade.
                self.state.last_no_trade_decision_id = None
                self.state.last_no_trade_decision_code = None
            # A cycle that evaluated cleanly and found nothing has no error of its
            # own, so an error left over from an *earlier* cycle is cleared here.
            # Leaving it would show a stale failure beside a fresh "no signal" and
            # read as if the current cycle had failed. The refresh below has
            # already re-recorded a read failure if one happened just now.
            self.state.last_error = None
            self._sync_observability()
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision_id,
                skipped_reason=self.state.last_no_decision_reason or "no decision is due",
                reason_code=self.state.last_no_trade_code,
            )
            self.last_cycle = result
            return result
        self.state.last_no_decision_reason = None
        self.state.last_no_trade_code = None
        self._last_offered_decision_id = decision.decision_id

        if self._is_processed(decision.decision_id):
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                skipped_reason="the decision has already been processed",
                reason_code=NO_TRADE_ALREADY_PROCESSED,
            )
            self.last_cycle = result
            return result

        client_order_id = decision.client_order_id(strategy_id=self.config.strategy_id)
        # A decision whose order the worker already knows is not re-submitted.
        if client_order_id in self.state.orders:
            order = self._reconcile(client_order_id)
            self._mark_processed(decision.decision_id)
            self._mark_ordered(decision.decision_id)
            self.state.last_no_trade_decision_id = None
            self.state.last_no_trade_decision_code = None
            self._record_order(order)
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                order=order,
                skipped_reason="the order for this decision already exists",
                reason_code=NO_TRADE_ORDER_EXISTS,
            )
            self.last_cycle = result
            return result

        try:
            # Translate decision symbol to execution symbol if symbol mappings are configured
            # The decision carries the canonical symbol (e.g., BTC/USDT); broker needs execution symbol (e.g., BTC/USD)
            execution_symbol = decision.symbol
            market_data_symbol = decision.symbol
            mapping: SymbolMapping | None = None

            if self._symbol_map and decision.symbol in self._symbol_map:
                mapping = self._symbol_map[decision.symbol]
                execution_symbol = mapping.execution_symbol
                market_data_symbol = mapping.market_data_symbol
            elif self._symbol_map:
                # The decision names a pair the validated mappings do not cover. It
                # is refused here — never translated to a different pair and never
                # sent to the broker — and the refusal is a *cycle outcome*, not an
                # exception: an unmapped symbol must not stop the worker.
                detail = (
                    f"decision symbol {decision.symbol!r} has no valid symbol mapping; "
                    f"available mappings: {sorted(self._symbol_map)}"
                )
                self.state.last_error = detail
                self.state.last_no_decision_reason = detail
                self.state.last_no_trade_code = NO_TRADE_MARKET_DATA_UNAVAILABLE
                self._remember_no_trade(
                    decision.decision_id, NO_TRADE_MARKET_DATA_UNAVAILABLE
                )
                self._save()
                result = CycleResult(
                    ran=False,
                    decision_id=decision.decision_id,
                    error=detail,
                    skipped_reason=(
                        "the decision's symbol is not tradable on the configured "
                        "venues, so no order was placed"
                    ),
                    reason_code=NO_TRADE_MARKET_DATA_UNAVAILABLE,
                    **self._market_data_fields(),
                )
                self.last_cycle = result
                return result

            reference_price = self.market_data.price(market_data_symbol)
        except MarketDataError as exc:
            self.state.last_error = f"MarketDataError: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="market data was unavailable",
                reason_code=NO_TRADE_MARKET_DATA_UNAVAILABLE,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        # The execution price. With a separate execution seam configured, the order
        # is priced on the venue that will trade it, in the execution pair; the
        # market-data price above is only the reference the deviation is measured
        # against. Without a seam the market-data price *is* the execution price
        # (the single-venue paths read the venue that will execute).
        execution_price = reference_price
        if self.config.execution_price_seam is not None:
            try:
                execution_price = float(
                    self.config.execution_price_seam.price(execution_symbol)
                )
            except MarketDataError as exc:
                # Fail closed: the market-data price is never used as a substitute
                # for the execution price, because they are different instruments.
                self.state.last_error = f"MarketDataError: {exc}"
                self.state.last_no_decision_reason = str(exc)
                self.state.last_no_trade_code = NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                self._remember_no_trade(
                    decision.decision_id, NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                )
                self._save()
                result = CycleResult(
                    ran=False,
                    decision_id=decision.decision_id,
                    error=str(exc),
                    skipped_reason=(
                        "no fresh price was available on the execution venue, so no "
                        "order was sized"
                    ),
                    reason_code=NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
                    reference_price=reference_price,
                    **self._market_data_fields(),
                )
                self.last_cycle = result
                return result
            if not math.isfinite(execution_price) or execution_price <= 0:
                detail = (
                    f"the execution venue returned an unusable price {execution_price!r} "
                    f"for {execution_symbol}; no order was sized"
                )
                self.state.last_error = detail
                self.state.last_no_decision_reason = detail
                self.state.last_no_trade_code = NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                self._remember_no_trade(
                    decision.decision_id, NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                )
                self._save()
                result = CycleResult(
                    ran=False,
                    decision_id=decision.decision_id,
                    error=detail,
                    skipped_reason=detail,
                    reason_code=NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
                    reference_price=reference_price,
                    **self._market_data_fields(),
                )
                self.last_cycle = result
                return result
            deviation = self._price_deviation(reference_price, execution_price)
            if deviation > self.config.max_price_deviation:
                detail = (
                    f"the execution price {execution_price:.8g} for {execution_symbol} "
                    f"differs from the decision price {reference_price:.8g} by "
                    f"{deviation:.2%}, beyond the permitted "
                    f"{self.config.max_price_deviation:.2%}; no order was placed"
                )
                self.state.last_error = detail
                self.state.last_no_decision_reason = detail
                self.state.last_no_trade_code = NO_TRADE_PRICE_DEVIATION
                self._remember_no_trade(decision.decision_id, NO_TRADE_PRICE_DEVIATION)
                self._save()
                result = CycleResult(
                    ran=False,
                    decision_id=decision.decision_id,
                    error=detail,
                    skipped_reason=detail,
                    reason_code=NO_TRADE_PRICE_DEVIATION,
                    reference_price=execution_price,
                    **self._market_data_fields(),
                )
                self.last_cycle = result
                return result

        try:
            quantity = self._size(
                decision, execution_price, execution_symbol=execution_symbol
            )
        except NoPositionError as exc:
            # A SELL with nothing to close is not a defect and not a risk breach:
            # on this long-only book it is simply not a trade. Recorded with its
            # own code, never as a sizing failure or a short.
            self.state.last_error = None
            self.state.last_no_decision_reason = str(exc)
            self.state.last_no_trade_code = NO_TRADE_NO_POSITION
            self._remember_no_trade(decision.decision_id, NO_TRADE_NO_POSITION)
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                skipped_reason=str(exc),
                reason_code=NO_TRADE_NO_POSITION,
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result
        except PositionReadError as exc:
            # The broker could not be asked for the holding, so a SELL cannot be
            # sized. Recorded as a market-data-style unavailability, never as
            # "there is no position", so a real broker error is not swallowed.
            self.state.last_error = f"PositionReadError: {exc}"
            self.state.last_no_decision_reason = str(exc)
            self.state.last_no_trade_code = NO_TRADE_POSITION_UNAVAILABLE
            self._remember_no_trade(decision.decision_id, NO_TRADE_POSITION_UNAVAILABLE)
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="the held quantity could not be read",
                reason_code=NO_TRADE_POSITION_UNAVAILABLE,
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result
        except WorkerError as exc:
            self.state.last_error = f"WorkerError: {exc}"
            self.state.last_no_decision_reason = str(exc)
            self.state.last_no_trade_code = NO_TRADE_SIZE_TOO_SMALL
            self._remember_no_trade(decision.decision_id, NO_TRADE_SIZE_TOO_SMALL)
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="the order could not be sized",
                reason_code=NO_TRADE_SIZE_TOO_SMALL,
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        try:
            order = broker.submit_order(
                symbol=execution_symbol,
                quantity=quantity,
                side=decision.side,
                reference_price=execution_price,
                client_order_id=client_order_id,
                order_type="limit" if decision.limit_price else "market",
                allow_orders=True,
                refresh=True,
                crypto=self.config.crypto,
            )
        except (PaperRiskRejected, BrokerRejectedError, BrokerError) as exc:
            # A local refusal or a broker rejection is recorded, not hidden, and the
            # worker stays RUNNING so the next cycle can act.
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            # A refusal is definitive for this decision: the worker stays RUNNING
            # for the next decision, but this one is not re-attempted on every
            # subsequent cycle, exactly as a one-shot source would not re-yield it.
            self._mark_processed(decision.decision_id)
            # A local risk refusal and a broker rejection are different causes and
            # are named apart, so a dashboard can tell "our own limits stopped it"
            # from "the broker refused it".
            code = (
                NO_TRADE_RISK_REJECTED
                if isinstance(exc, PaperRiskRejected)
                else (
                    NO_TRADE_INSUFFICIENT_FUNDS
                    if isinstance(exc, InsufficientFundsError)
                    else NO_TRADE_BROKER_REJECTED
                )
            )
            self.state.last_no_decision_reason = str(exc)
            self.state.last_no_trade_code = code
            self._remember_no_trade(decision.decision_id, code)
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                error=str(exc),
                skipped_reason="the order was refused",
                reason_code=code,
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        self._mark_processed(decision.decision_id)
        self._mark_ordered(decision.decision_id)
        self.state.last_error = None
        self.state.last_no_decision_reason = None
        self.state.last_no_trade_code = None
        self.state.last_no_trade_decision_id = None
        self.state.last_no_trade_decision_code = None
        self._record_order(order)
        self._record_exit_levels(decision, order)
        result = CycleResult(
            ran=True,
            decision_id=decision.decision_id,
            client_order_id=client_order_id,
            order=order,
            reference_price=reference_price,
            **self._market_data_fields(),
        )
        self.last_cycle = result
        return result

    def _record_exit_levels(self, decision: TradingDecision, order: BrokerOrder) -> None:
        """Record the strategy's stop/target for a *filled* long entry.

        Only a long entry with a filled quantity is watched: a stop or target on a
        position the worker does not hold, or on an unfilled order, has nothing to
        exit. The levels are stored verbatim from the decision; when the strategy
        proposed none, nothing is stored, so the exit policy has no level to act on
        and fabricates none.
        """
        if order.side.lower() != "buy" or order.filled_quantity <= 0:
            return
        if decision.stop_loss is None and decision.take_profit is None:
            return
        symbol = order.symbol.upper()
        # A **new** entry starts a new exit episode for this symbol. The old
        # episode's guard is cleared so the new position can be exited on its
        # own levels, and the counter is advanced so the new episode's order
        # id cannot collide with the old one at the broker.
        self.state.exit_attempts.pop(symbol, None)
        self.state.exits.pop(symbol, None)
        self.state.exit_attempt_counter += 1
        self.state.stops[symbol] = {
            "symbol": symbol,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "decision_id": decision.decision_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            # The canonical decision symbol, so a reader can tell *which* market
            # the levels came from (``BTC/USDT``) from the execution symbol the
            # position is held under (``BTC/USD``). Additive and optional.
            "canonical_symbol": decision.symbol,
            # Which exit episode these levels belong to, so an exit order for a
            # later episode cannot be confused with one for these levels.
            "episode": self.state.exit_attempt_counter,
        }
        self._save()

    def _check_exits(self) -> CycleResult | None:
        """Close a worker-held long whose recorded stop or target has triggered.

        Returns ``None`` when there is nothing to exit (the common case, and the
        only case when no stop/target was ever recorded, so default behaviour is
        unchanged). Returns a :class:`CycleResult` only when it placed an exit order
        or was refused placing one — at most one exit per cycle.

        Bounds, each deliberate:

        - only a **long** position the worker holds is considered (long-only, the
          same rule the broker enforces);
        - only a position with a **recorded** stop/target is watched, and only one
          whose recorded level the current price has crossed;
        - only a position the worker itself opened is exited (its symbol is in
          ``state.stops``), so a manually-held position is never closed;
        - a symbol already exited is never exited again (``state.exits``);
        - the price comes from the worker's own seam, so a stale or invalid price
          refuses the exit rather than fabricating one.

        A refusal (for example a daily-order cap) is recorded and the position
        stays open; the worker stays ``RUNNING`` for the next cycle.
        """
        broker = self._broker
        if broker is None or not self.state.stops:
            return None
        for symbol in sorted(self.state.stops):
            if symbol in self.state.exits:
                continue
            level = self.state.stops[symbol]
            stop_loss = level.get("stop_loss")
            take_profit = level.get("take_profit")
            if not _has_usable_level(stop_loss, take_profit):
                continue

            # The **broker** is the authority for what is still held. The
            # worker's last synced view is a snapshot: an exit that filled
            # partially, or a position reduced elsewhere, leaves that snapshot
            # stale, and sizing the close from it would either oversell (refused
            # as a short) or close the wrong amount. A read failure is a refusal
            # to exit, never an assumption that the position is flat.
            try:
                held = self._held_quantity(symbol)
            except PositionReadError as exc:
                self.state.last_error = f"PositionReadError: {exc}"
                self.state.last_no_decision_reason = str(exc)
                self.state.last_no_trade_code = NO_TRADE_POSITION_UNAVAILABLE
                self._save()
                return CycleResult(
                    ran=False,
                    error=str(exc),
                    skipped_reason=f"the held quantity for {symbol} could not be read",
                    reason_code=NO_TRADE_POSITION_UNAVAILABLE,
                    **self._market_data_fields(),
                )

            attempt = self.state.exit_attempts.get(symbol)
            if attempt is not None:
                # An exit order for this position has already been placed. It is
                # reconciled, never replaced: a stop that stays triggered while
                # the price moves must not turn into a stream of orders, and a
                # restart must not re-place one. A *new* episode — the position
                # was closed and re-opened — carries its own attempt record and
                # is not caught here.
                exit_id = attempt.get("client_order_id")
                if not exit_id:
                    self.state.exit_attempts.pop(symbol, None)
                else:
                    order = self._reconcile(exit_id)
                    self._record_order(order)
                    # Reconciling an exit does not make it *done*. A partially
                    # filled exit still leaves a remainder, so the position stays
                    # watched; only a fully filled exit closes it.
                    if self._exit_completed(order, held):
                        self.state.exits[symbol] = {
                            "symbol": symbol,
                            "reason": attempt.get("reason") or "exit",
                            "price": attempt.get("price"),
                            "quantity": held,
                            "filled_quantity": order.filled_quantity,
                            "exited_at": datetime.now(UTC).isoformat(),
                        }
                        self.state.exit_attempts.pop(symbol, None)
                    self._save()
                    return CycleResult(
                        ran=False,
                        client_order_id=exit_id,
                        order=order,
                        skipped_reason=f"an exit order for {symbol} already exists",
                        reference_price=attempt.get("price"),
                        **self._market_data_fields(),
                    )
            if held <= 0:
                # The broker says the position is flat. There is nothing to
                # close, and no level to watch. This is the ordinary end of a
                # position that was closed outside the worker.
                continue

            # The stop/target is monitored against the **market-data** price (the
            # market the strategy analysed), while the exit *order* is placed under
            # the **execution** symbol the position is held under. The two come
            # from the same validated mapping, so they cannot drift apart.
            market_data_symbol = self._execution_to_market_data.get(symbol, symbol)

            try:
                price = self.market_data.price(market_data_symbol)
            except MarketDataError as exc:
                self.state.last_error = f"MarketDataError: {exc}"
                self._save()
                return CycleResult(
                    ran=False,
                    error=str(exc),
                    skipped_reason=f"an exit for {symbol} could not be priced",
                    reason_code=NO_TRADE_MARKET_DATA_UNAVAILABLE,
                    **self._market_data_fields(),
                )
            reason = _exit_reason(price, stop_loss, take_profit)
            if reason is None:
                continue
            exit_id = _exit_order_id(
                symbol, price, reason, int(level.get("episode") or 0)
            )
            if exit_id in self.state.orders:
                # Already attempted for this trigger: reconcile, do not resubmit.
                order = self._reconcile(exit_id)
                self._record_order(order)
                # Reconciling an exit does not make it *done*. A partially
                # filled exit still leaves a remainder, so the position stays
                # watched; only a fully filled exit closes it.
                if self._exit_completed(order, held):
                    self.state.exits[symbol] = {
                        "symbol": symbol,
                        "reason": reason,
                        "price": price,
                        "quantity": held,
                        "filled_quantity": order.filled_quantity,
                        "exited_at": datetime.now(UTC).isoformat(),
                    }
                    self.state.exit_attempts.pop(symbol, None)
                self._save()
                return CycleResult(
                    ran=False,
                    client_order_id=exit_id,
                    order=order,
                    skipped_reason=f"an exit order for {symbol} already exists",
                    reference_price=price,
                    **self._market_data_fields(),
                )
            # The exit order is priced on the **execution** venue, in the execution
            # symbol the position is held under — the same pair the entry used.
            # The market-data price above only *decided* that the level was crossed.
            exit_price = price
            if self.config.execution_price_seam is not None:
                try:
                    exit_price = float(
                        self.config.execution_price_seam.price(symbol)
                    )
                except MarketDataError as exc:
                    detail = (
                        f"no fresh execution price was available for the {symbol} exit: "
                        f"{exc}"
                    )
                    self.state.last_error = detail
                    self.state.last_no_decision_reason = detail
                    self.state.last_no_trade_code = NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                    self._save()
                    return CycleResult(
                        ran=False,
                        client_order_id=exit_id,
                        error=detail,
                        skipped_reason=detail,
                        reason_code=NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
                        reference_price=price,
                        **self._market_data_fields(),
                    )
                if not math.isfinite(exit_price) or exit_price <= 0:
                    detail = (
                        f"the execution venue returned an unusable price "
                        f"{exit_price!r} for the {symbol} exit; no exit was placed"
                    )
                    self.state.last_error = detail
                    self.state.last_no_decision_reason = detail
                    self.state.last_no_trade_code = NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
                    self._save()
                    return CycleResult(
                        ran=False,
                        client_order_id=exit_id,
                        error=detail,
                        skipped_reason=detail,
                        reason_code=NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
                        reference_price=price,
                        **self._market_data_fields(),
                    )
            try:
                order = broker.submit_order(
                    symbol=symbol,
                    quantity=held,
                    side="sell",
                    reference_price=exit_price,
                    client_order_id=exit_id,
                    order_type="market",
                    allow_orders=True,
                    refresh=True,
                    crypto=self.config.crypto,
                )
            except (PaperRiskRejected, BrokerRejectedError, BrokerError) as exc:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.last_no_decision_reason = str(exc)
                self.state.last_no_trade_code = NO_TRADE_EXIT_REFUSED
                self._save()
                return CycleResult(
                    ran=False,
                    client_order_id=exit_id,
                    error=str(exc),
                    skipped_reason=f"the exit for {symbol} was refused",
                    reason_code=NO_TRADE_EXIT_REFUSED,
                    reference_price=exit_price,
                    **self._market_data_fields(),
                )
            self.state.last_error = None
            self._record_order(order)
            # The exit order is recorded as an **attempt** the moment it
            # reaches the broker, whatever its fill status. That record is what
            # stops a second order: the next cycle reconciles this one instead
            # of placing another, so a stop that stays triggered across several
            # cycles cannot become a stream of orders, and a restart cannot
            # re-place one.
            self.state.exit_attempts[symbol] = {
                "symbol": symbol,
                "reason": reason,
                "price": price,
                "client_order_id": exit_id,
                "quantity": held,
                "attempted_at": datetime.now(UTC).isoformat(),
            }
            # The exit is recorded as *done* only when the position is actually
            # closed. A partially filled exit leaves a remainder still held, so
            # recording it as exited would silence the protection on the part
            # that is still open — the position would look closed while it is
            # not.
            if self._exit_completed(order, held):
                self.state.exits[symbol] = {
                    "symbol": symbol,
                    "reason": reason,
                    "price": price,
                    "quantity": held,
                    "filled_quantity": order.filled_quantity,
                    "exited_at": datetime.now(UTC).isoformat(),
                }
                self.state.exit_attempts.pop(symbol, None)
            self._save()
            return CycleResult(
                ran=True,
                client_order_id=exit_id,
                order=order,
                skipped_reason=None,
                reference_price=price,
                **self._market_data_fields(),
            )
        return None

    @staticmethod
    def _exit_completed(order: BrokerOrder, held: float) -> bool:
        """Whether an exit order has actually closed the position.

        The broker's **filled quantity** is the authority, not the order's
        status string: a market order the broker reports as ``accepted`` can
        already be fully filled, and a status of ``filled`` can still leave a
        remainder when the broker filled less than the order asked for. The
        position is closed when the filled quantity covers what was held.

        A partially filled exit, an unfilled order, or a canceled order all
        leave a remainder, so the position is still open and its recorded levels
        must keep being watched.
        """
        if order.status in ("canceled", "expired", "rejected"):
            return False
        return order.filled_quantity >= held - 1e-9

    def _market_data_fields(self) -> dict[str, Any]:
        """The seam's identity for a cycle result, read defensively.

        A cycle result must name the actual price provider even when the cycle
        failed, so an operator can tell a historical bar open from a current
        snapshot without reading the code. A seam that does not describe itself is
        labelled ``unknown`` rather than mislabelled.
        """
        identity = market_data_identity(self.market_data)
        return {
            "market_data_provider": identity.get("market_data_provider"),
            "market_data_kind": identity.get("market_data_kind"),
            "market_data_status": identity.get("market_data_status"),
        }

    def _price_deviation(self, decision_price: float, execution_price: float) -> float:
        """The fractional gap between the decision price and the execution price.

        Measured against the **decision** price — the price the strategy acted on —
        so the number is "how far the price moved since the decision", which is what
        the threshold is about. A non-finite or non-positive price on either side is
        an infinite deviation: an unusable price is never within tolerance.
        """
        if not math.isfinite(decision_price) or decision_price <= 0:
            return float("inf")
        if not math.isfinite(execution_price) or execution_price <= 0:
            return float("inf")
        return abs(execution_price - decision_price) / decision_price

    def _size(
        self,
        decision: TradingDecision,
        reference_price: float,
        *,
        execution_symbol: str | None = None,
    ) -> float:
        """Convert a decision into a share quantity.

        **BUY** is sized from the *notional* at the current price, exactly as
        before. With ``whole_shares`` (the default) the quantity is floored to an
        integer; a high-priced symbol under a small notional can then floor to zero
        — for example a 25 USD notional at a 580 USD price — which is refused with
        an explicit message naming the fix (enable fractional shares, or raise the
        notional), never sent to the broker as a zero-quantity order. With
        fractional shares enabled the quantity is floored to six decimals (never
        rounded up), so ``quantity * reference_price`` can never exceed the
        requested notional.

        **SELL** is *not* sized from the notional: on this long-only book a SELL is
        only ever a **close**, so its quantity is the position actually held,
        clamped to that holding. Sizing a close from a dollar target is the defect
        this split removes — it would sell a tiny fraction of a position (or floor
        to zero) instead of closing it. The held quantity comes from the broker's
        own positions, refreshed here, so a stale local view cannot size the
        close. A SELL with nothing held raises :class:`NoPositionError`, which the
        caller reports as ``NO_TRADE_NO_POSITION`` and never turns into a short.

        ``execution_symbol`` is the symbol the **broker** knows the position by
        (``BTC/USD`` on Alpaca), which is not the decision's canonical symbol
        (``BTC/USDT``) on the cross-venue crypto path. The holding is looked up by
        the execution symbol, so a close finds the position it is meant to close.
        It defaults to the decision's own symbol, which is correct on every
        single-venue path where the two are the same.
        """
        if decision.side.lower() == "sell":
            return self._size_sell(
                decision, execution_symbol=execution_symbol or decision.symbol
            )
        return self._size_buy(decision, reference_price)

    def _size_buy(self, decision: TradingDecision, reference_price: float) -> float:
        """The BUY quantity: the notional at the **execution** price.

        A decision that carries a *quantity* has one that was computed elsewhere,
        at a price that may not be the price this order will trade at. On the
        cross-venue crypto path that quantity was sized at the market-data
        venue's price (``XRP/USDT``) while the order is placed on the execution
        venue's pair (``XRP/USD``): a quantity sized from one venue's price means
        a different amount of cash on the other, so the order's notional was
        never the one the risk engine approved — the $25 cap included.

        So a quantity is honoured **only** when it is already denominated at the
        execution price. When the decision carries both, the notional is the
        quantity's cash value at the price it was sized at, and it is the notional
        that is re-divided by the execution price. A decision that carries only a
        quantity (no notional) cannot be re-based, so it is used as it stands —
        the quantity is the only statement of size there is.

        The notional, when present, is always what bounds the order: the broker
        re-checks ``quantity * execution_price`` against its own cap, so a
        re-sized order can never exceed the notional the risk engine approved.
        """
        if decision.quantity is not None and decision.notional is None:
            return float(decision.quantity)
        assert decision.notional is not None  # __post_init__ guarantees one
        raw = decision.notional / reference_price
        if self.config.whole_shares:
            quantity = math.floor(raw)
        else:
            quantity = math.floor(raw * 1_000_000) / 1_000_000
        if quantity <= 0:
            if self.config.whole_shares:
                raise WorkerError(
                    f"the notional {decision.notional} buys no whole shares at price "
                    f"{reference_price} (it would floor to 0); enable fractional "
                    "shares (whole_shares=False) for a high-priced symbol such as "
                    "SPY, or raise the order notional"
                )
            raise WorkerError(
                f"the notional {decision.notional} buys no shares at price "
                f"{reference_price}; the order would be empty"
            )
        return float(quantity)

    def _size_sell(self, decision: TradingDecision, *, execution_symbol: str) -> float:
        """The SELL quantity: the held long, clamped to it. Never the notional.

        The holding is read from the broker (refreshed), so the close is sized from
        the account's *current* position rather than a stale local copy. A decision
        that carries an explicit ``quantity`` may ask for a *partial* close; a
        quantity above the holding is clamped down to the holding, so a SELL can
        never exceed what is held (and so can never open a short). With no explicit
        quantity the whole position is closed.
        """
        held = self._held_quantity(execution_symbol)
        if held <= 0:
            raise NoPositionError(
                f"there is no position in {execution_symbol} to sell; a SELL closes a "
                "held long and this book will not open a short"
            )
        requested = decision.quantity
        if requested is None:
            return float(held)
        return float(min(float(requested), held))

    def _held_quantity(self, symbol: str) -> float:
        """The long quantity held for ``symbol``, from the broker when reachable.

        The broker is the authority: its positions are re-read here so a close is
        never sized from a stale local copy. When there is no broker (offline) the
        worker's own last-synced position view is used instead, which is the same
        figure the observability sync keeps current while running. Only a *long*
        is returned: a non-positive quantity means nothing to close.
        """
        symbol = symbol.upper()
        broker = self._broker
        if broker is not None:
            try:
                positions = broker.positions(refresh=True)
            except Exception as exc:  # noqa: BLE001 - a read failure is not "flat"
                raise PositionReadError(
                    f"the held quantity for {symbol} could not be read from the "
                    f"broker ({type(exc).__name__}), so a SELL cannot be sized; "
                    "refusing rather than assuming the position is flat"
                ) from None
            held = positions.get(symbol)
            quantity = float(held.quantity) if held is not None else 0.0
        else:
            position = self.state.positions.get(symbol)
            try:
                quantity = float(position.get("quantity") or 0.0) if position else 0.0
            except (TypeError, ValueError):
                quantity = 0.0
        return quantity if quantity > 0 else 0.0

    def _reconcile(self, client_order_id: str) -> BrokerOrder:
        broker = self._broker
        if broker is None:  # pragma: no cover
            raise WorkerError("the worker has no broker")
        return broker._reconcile(client_order_id)

    def _record_order(self, order: BrokerOrder) -> None:
        self.state.orders[order.client_order_id] = {
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "filled_avg_price": order.filled_avg_price,
            "status": order.status,
            "submitted_at": order.submitted_at,
        }
        self._save()

    def position_protection(self) -> dict[str, dict[str, Any]]:
        """The protection state of every open position, keyed by symbol.

        A thin wrapper over :func:`_position_protection_rows`, so the worker's
        own view and the status view can never disagree about which positions
        are guarded and which are not.
        """
        return _position_protection_rows(self.state)

    def sync_positions(self) -> dict[str, dict]:
        """Refresh the broker's positions into the worker state (read-only)."""
        broker = self._broker
        if broker is None:
            raise WorkerError("the worker has no broker; start it first")
        positions = broker.positions(refresh=True)
        self.state.positions = {s: p.to_dict() for s, p in positions.items()}
        now = datetime.now(UTC).isoformat()
        self.state.positions_at = now
        self.state.positions_checked_at = now
        self._save()
        return dict(self.state.positions)

    def sync_account(self) -> dict[str, Any]:
        """Refresh the account and roll the daily-P&L baseline (read-only).

        The account is read from the broker (never fabricated) and stored on the
        state so the dashboard reports real cash, equity and buying power. The
        first read of a new UTC day sets ``initial_equity`` — the baseline a daily
        P&L is measured from — so the daily figure is the change since the day
        opened rather than since the process started. A non-finite equity is
        refused rather than stored, so a bad read cannot poison the baseline.
        """
        broker = self._broker
        if broker is None:
            raise WorkerError("the worker has no broker; start it first")
        account = broker.refresh_account()
        equity = _as_float(account.equity)
        today = datetime.now(UTC).date().isoformat()
        if equity is not None and (self.state.day != today or self.state.initial_equity is None):
            self.state.day = today
            self.state.initial_equity = equity
        self.state.account = {
            "account_id": account.account_id,
            "status": account.status,
            "cash": _as_float(account.cash),
            "equity": equity,
            "buying_power": _as_float(account.buying_power),
            "currency": account.currency,
            "restrictions": account.restrictions(),
        }
        self.state.last_account_at = datetime.now(UTC).isoformat()
        self._save()
        return dict(self.state.account)

    def _sync_observability(self) -> None:
        """Best-effort account/position refresh, so the view is not stale.

        Read-only and *contained*: a broker read failure is recorded in
        ``last_error`` and never propagates, because an observability refresh must
        not be able to fail a trading cycle or stop the loop. This is what keeps
        the dashboard's cash, equity and positions current between orders instead
        of frozen at start-up.
        """
        if self._broker is None:
            return
        for step in (self.sync_account, self.sync_positions):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - a stale view is better than a crash
                self.state.last_error = f"{type(exc).__name__}: {exc}"

    def _no_decision_outcome(self) -> tuple[str, str]:
        """The ``(reason, code)`` for a cycle the source yielded no decision.

        Normally the source's own words. One case is worth not losing, though: the
        current-market source reports a generic ``CURRENT_MARKET_NO_NEW_DATA`` once
        every symbol already has a decision for the session — which includes the
        case where a decision *was* evaluated and produced no trade (a SELL with
        nothing to close, say). When this worker remembers that exact decision as
        its last no-trade, that honest reason is reported instead, so the dashboard
        keeps naming *why* the session was left un-traded rather than erasing it
        with the generic code. It stays a pure observation: nothing is re-evaluated
        and no order is affected.
        """
        code = self._no_trade_code()
        if (
            code == NO_TRADE_CURRENT_MARKET_NO_NEW_DATA
            and self._last_offered_decision_id is not None
            and self._last_offered_decision_id == self.state.last_no_trade_decision_id
            and self.state.last_no_trade_decision_code
        ):
            return (
                self.state.last_no_decision_reason
                or "the decision produced no order this session",
                self.state.last_no_trade_decision_code,
            )
        return self._no_decision_reason(), code

    def _no_decision_id(self) -> str | None:
        """The id of the decision this cycle evaluated but did not offer.

        A source that evaluated a decision and then withheld it — because the
        chain did not complete, or the RiskEngine refused it — can name that
        decision, so the no-trade is recorded against a decision id rather than
        against whichever decision happened to be last. A source that cannot name
        one returns ``None``, and the worker records the no-trade without an id
        instead of borrowing an earlier decision's.
        """
        for attribute in ("no_decision_id", "last_no_decision_id"):
            value = getattr(self.decision_source, attribute, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _no_decision_reason(self) -> str:
        """Why the last evaluation produced no order, in the source's own words.

        A source may expose ``skip_reason`` (the experiment source does); when it
        does, that reason is surfaced verbatim so a RUNNING-but-idle worker is
        explainable. Otherwise a generic reason is returned rather than a
        fabricated cause.
        """
        reason = getattr(self.decision_source, "skip_reason", None)
        if isinstance(reason, str) and reason:
            return reason
        return "no decision is due"

    def _no_trade_code(self) -> str:
        """The stable code behind :meth:`_no_decision_reason`.

        A source that names its own code (the experiment source does) is trusted;
        otherwise the generic :data:`NO_TRADE_NO_DECISION_DUE` is recorded, so the
        state always carries a code from the closed vocabulary and a dashboard never
        has to parse prose to decide what is blocking.
        """
        code = getattr(self.decision_source, "skip_code", None)
        if isinstance(code, str) and code:
            return code
        return NO_TRADE_NO_DECISION_DUE

    @classmethod
    def load(cls, path: str | os.PathLike) -> WorkerState:
        """Load the persisted worker state, always as ``STOPPED``.

        A restart must never resume trading by itself: the state is returned with
        its order memory intact but its status forced to ``STOPPED``.
        """
        state = WorkerStateStore(path).load()
        state.status = STATUS_STOPPED
        return state
