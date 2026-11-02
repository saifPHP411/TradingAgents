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

from .alpaca_paper import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
    BrokerOrder,
    BrokerRejectedError,
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
PROVIDER_STATIC = "static"
PROVIDER_UNKNOWN = "unknown"

MARKET_DATA_CHOICES = (PROVIDER_BAR_OPEN, PROVIDER_ALPACA_SNAPSHOT)
"""The price seams an operator may select. ``bar_open`` is the historical default;
``alpaca_snapshot`` is the current-price opt-in."""


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


# ---------------------------------------------------------------------------
# Decision and market-data seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingDecision:
    """A strategy decision the worker may turn into one paper order.

    ``notional`` is the target cash amount to trade; the worker converts it to a
    share quantity from the *current* price, so sizing never depends on a stale
    price. ``quantity`` may be given instead (then ``notional`` is ignored).
    """

    decision_id: str
    symbol: str
    side: str  # "buy" or "sell"
    notional: float | None = None
    quantity: float | None = None
    rationale: str = ""
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise WorkerError("a decision must carry a decision_id")
        if not self.symbol:
            raise WorkerError("a decision must carry a symbol")
        if self.side.lower() not in ("buy", "sell"):
            raise WorkerError(f"side must be buy or sell, got {self.side!r}")
        if self.notional is None and self.quantity is None:
            raise WorkerError("a decision must carry a notional or a quantity")
        for value in (self.notional, self.quantity, self.limit_price):
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
        return {"market_data_detail": "a fixed price map (tests/demo), not a market price"}


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
            "is_current_price": True,
            "max_snapshot_age_seconds": self.max_age_seconds,
            "last_snapshot_at": observation.as_of if observation else None,
            "last_snapshot_price": observation.price if observation else None,
            "last_snapshot_field": observation.field if observation else None,
            "last_snapshot_age_seconds": self.last_snapshot_age_seconds(),
            "last_snapshot_observed_at": self.last_observed_at,
            "market_data_status": self.last_status,
            "market_data_status_detail": self.last_status_detail,
        }

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
                self._observation_from(snapshots.get(symbol))
            except MarketDataError as exc:
                self._record_status(_status_for(exc), str(exc))
                return False
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
    if "no snapshot" in text:
        return MARKET_DATA_STATUS_NO_DATA
    return MARKET_DATA_STATUS_ERROR


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
    whole_shares: bool = True
    """When ``True`` (the default), size to whole shares.

    Alpaca supports fractional shares, but whole shares keep a tiny account's
    behaviour obvious. Set ``False`` to allow a fractional quantity, which is what
    lets a high-priced symbol such as SPY fit inside a small notional. This is a
    deliberate, per-worker opt-in: it is never enabled silently for every account,
    and a broker that does not support fractional shares will reject the order
    rather than receive a fabricated zero quantity (see :meth:`PaperTradingWorker._size`).
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
        if self.market_data_max_age_seconds is not None and not (
            self.market_data_max_age_seconds > 0
        ):
            raise WorkerError("market_data_max_age_seconds must be positive or None")
        # The endpoint is validated at configuration time, so a live host is
        # refused before a broker is ever constructed.
        assert_paper_endpoint(self.base_url)


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
    account: dict[str, Any] = field(default_factory=dict)
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
            "account": dict(self.account),
            "cycles_completed": self.cycles_completed,
            "processed_decisions": sorted(self.processed_decisions),
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
            account=dict(payload.get("account") or {}),
            cycles_completed=_as_int(payload.get("cycles_completed"), 0),
            processed_decisions=set(payload.get("processed_decisions") or []),
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
        }


def worker_status_view(
    *,
    state: WorkerState,
    config: PaperTradingConfig,
    loop_running: bool = False,
    next_cycle_at: str | None = None,
    last_cycle: CycleResult | None = None,
    market_data_identity: dict[str, Any] | None = None,
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
    """
    identity = dict(market_data_identity or {})
    configured_provider = config.market_data_provider
    # Prefer the live seam's own identity; fall back to the configured provider, so
    # a stopped worker still reports which seam it *would* use.
    identity.setdefault("market_data_provider", configured_provider)
    identity.setdefault("market_data_kind", None)
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
        **identity,
        "last_processed_decision_id": state.last_processed_decision_id,
        "last_error": state.last_error,
        "last_cycle_at": state.last_cycle_at,
        "started_at": state.started_at,
        "stopped_at": state.stopped_at,
        "loop_running": loop_running,
        "next_cycle_at": next_cycle_at,
        "cycles_completed": state.cycles_completed,
        "poll_interval_seconds": config.poll_interval_seconds,
        "orders": {k: dict(v) for k, v in sorted(state.orders.items())},
        "positions": {k: dict(v) for k, v in sorted(state.positions.items())},
        "positions_at": state.positions_at,
        "account": dict(state.account),
        "last_cycle": last_cycle.to_dict() if last_cycle is not None else None,
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
        """Whether this decision has already been acted on (durably)."""
        return (
            decision_id in self.state.processed_decisions
            or decision_id == self.state.last_processed_decision_id
        )

    def _mark_processed(self, decision_id: str) -> None:
        """Remember a decision as done, so no cycle — and no restart — repeats it."""
        self.state.last_processed_decision_id = decision_id
        self.state.processed_decisions.add(decision_id)

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
        self.state.account = {
            "account_id": account.account_id,
            "status": account.status,
            "cash": account.cash,
            "equity": account.equity,
            "buying_power": account.buying_power,
            "currency": account.currency,
            "restrictions": account.restrictions(),
        }
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
        decision = self.decision_source.next_decision()
        if decision is None:
            self._save()
            result = CycleResult(ran=False, skipped_reason="no decision is due")
            self.last_cycle = result
            return result

        if self._is_processed(decision.decision_id):
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                skipped_reason="the decision has already been processed",
            )
            self.last_cycle = result
            return result

        client_order_id = decision.client_order_id(strategy_id=self.config.strategy_id)
        # A decision whose order the worker already knows is not re-submitted.
        if client_order_id in self.state.orders:
            order = self._reconcile(client_order_id)
            self._mark_processed(decision.decision_id)
            self._record_order(order)
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                order=order,
                skipped_reason="the order for this decision already exists",
            )
            self.last_cycle = result
            return result

        try:
            reference_price = self.market_data.price(decision.symbol)
        except MarketDataError as exc:
            self.state.last_error = f"MarketDataError: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="market data was unavailable",
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        try:
            quantity = self._size(decision, reference_price)
        except WorkerError as exc:
            self.state.last_error = f"WorkerError: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="the order could not be sized",
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        try:
            order = broker.submit_order(
                symbol=decision.symbol,
                quantity=quantity,
                side=decision.side,
                reference_price=reference_price,
                client_order_id=client_order_id,
                order_type="limit" if decision.limit_price else "market",
                allow_orders=True,
                refresh=True,
            )
        except (PaperRiskRejected, BrokerRejectedError, BrokerError) as exc:
            # A local refusal or a broker rejection is recorded, not hidden, and the
            # worker stays RUNNING so the next cycle can act.
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            # A refusal is definitive for this decision: the worker stays RUNNING
            # for the next decision, but this one is not re-attempted on every
            # subsequent cycle, exactly as a one-shot source would not re-yield it.
            self._mark_processed(decision.decision_id)
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                error=str(exc),
                skipped_reason="the order was refused",
                reference_price=reference_price,
                **self._market_data_fields(),
            )
            self.last_cycle = result
            return result

        self._mark_processed(decision.decision_id)
        self.state.last_error = None
        self._record_order(order)
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

    # -- helpers ----------------------------------------------------------

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

    def _size(self, decision: TradingDecision, reference_price: float) -> float:
        """Convert a decision into a share quantity, sized from the current price.

        With ``whole_shares`` (the default) the quantity is floored to an integer.
        A high-priced symbol under a small notional can then floor to zero — for
        example a 25 USD notional at a 580 USD price. That is refused with an
        explicit message naming the fix (enable fractional shares, or raise the
        notional), never sent to the broker as a zero-quantity order.

        With fractional shares enabled the quantity is floored to six decimals (never
        rounded up), so ``quantity * reference_price`` can never exceed the requested
        notional — and therefore can never trip the ``max_order_notional`` cap by a
        rounding artefact. It is still refused if it floors to zero, so an order is
        never empty.
        """
        if decision.quantity is not None:
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

    def sync_positions(self) -> dict[str, dict]:
        """Refresh the broker's positions into the worker state (read-only)."""
        broker = self._broker
        if broker is None:
            raise WorkerError("the worker has no broker; start it first")
        positions = broker.positions(refresh=True)
        self.state.positions = {s: p.to_dict() for s, p in positions.items()}
        self.state.positions_at = datetime.now(UTC).isoformat()
        self._save()
        return dict(self.state.positions)

    @classmethod
    def load(cls, path: str | os.PathLike) -> WorkerState:
        """Load the persisted worker state, always as ``STOPPED``.

        A restart must never resume trading by itself: the state is returned with
        its order memory intact but its status forced to ``STOPPED``.
        """
        state = WorkerStateStore(path).load()
        state.status = STATUS_STOPPED
        return state
