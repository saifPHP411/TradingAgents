"""An Alpaca **paper** broker adapter, isolated from the internal simulator.

Paper only, by construction
---------------------------

The base URL is hard-validated: only Alpaca's paper host is accepted, and every
live host is refused by :func:`assert_paper_endpoint`. There is no setting that
turns paper into live, and no fallback that would reach the live API. Credentials
come from ``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET`` only; the live
variable names are refused rather than accepted as a fallback.

This adapter never touches the internal ledger
----------------------------------------------

The internal paper experiment has its own portfolio and ledger; Alpaca Paper has
its own simulated account. This module holds the *broker* account state and the
*broker* order records, and nothing else. It does not import the ledger, the
simulator, or the experiment, so a broker fill cannot be double-counted as an
internal fill. Reconciliation between the two, if ever wanted, is a separate,
explicit step — not something this module does silently.

Fail-closed behaviour
---------------------

- missing credentials -> :class:`BrokerNotConfiguredError`;
- a non-paper endpoint -> :class:`LiveEndpointRefusedError`;
- an unconfirmed account -> orders are refused, not placed blind;
- an ambiguous submission (a timeout) -> reconcile by ``client_order_id`` before
  deciding, never blind-retry.

The transport is injectable, so every behaviour above is exercised in tests with
a mocked HTTP response and no socket. The default transport is ``requests``.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Endpoint safety
# ---------------------------------------------------------------------------

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
"""The only endpoint this adapter will talk to. Alpaca's paper trading host."""

_LIVE_HOSTS = frozenset({
    "api.alpaca.markets",
    "alpaca.markets",
    "broker-api.alpaca.markets",
})
"""Alpaca live-trading hosts, refused outright. Never a fallback."""

API_KEY_ENV = "ALPACA_PAPER_API_KEY"
API_SECRET_ENV = "ALPACA_PAPER_API_SECRET"
_LIVE_KEY_ENVS = ("APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_SECRET", "APCA_API_SECRET_KEY")
"""Live/ambiguous credential variable names that are refused, not used."""

REQUEST_TIMEOUT = 30.0


class BrokerError(RuntimeError):
    """A broker-backed paper path could not proceed."""


class BrokerNotConfiguredError(BrokerError):
    """The broker was selected but its paper credentials are missing."""


class LiveEndpointRefusedError(BrokerError):
    """A non-paper (live) endpoint was requested. This is a hard refusal."""


class BrokerRejectedError(BrokerError):
    """The broker rejected the request (auth, risk, or a bad order)."""


class PaperRiskRejected(BrokerError):
    """A locally-enforced paper risk limit refused the order before submission."""


def _host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _today() -> str:
    """The current UTC calendar day, ``YYYY-MM-DD``. The daily-order window."""
    return datetime.now(UTC).date().isoformat()


def assert_paper_endpoint(base_url: str) -> str:
    """Return ``base_url`` if it is the approved paper endpoint, else raise.

    Accepts the paper host exactly. Any live host is refused with
    :class:`LiveEndpointRefusedError`; any other host is refused as unapproved.
    This is the single gate every request passes through, so a live call cannot
    be reached by a configuration mistake.
    """
    host = _host_of(base_url)
    if host in _LIVE_HOSTS:
        raise LiveEndpointRefusedError(
            f"{base_url!r} is an Alpaca LIVE endpoint; this adapter is paper-only and "
            f"refuses every live endpoint. Use {PAPER_BASE_URL}."
        )
    if host != _host_of(PAPER_BASE_URL):
        raise LiveEndpointRefusedError(
            f"{base_url!r} is not the approved Alpaca paper endpoint ({PAPER_BASE_URL}); "
            "refusing to send a request to an unapproved host"
        )
    return base_url


# ---------------------------------------------------------------------------
# Order status vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerOrder:
    """One order as the broker reports it, reduced to what this adapter needs.

    ``status`` is Alpaca's own vocabulary, mapped into the internal
    :class:`~tradingagents.execution.orders.OrderStatus` where the two overlap and
    kept as ``unknown`` where they do not, so an unrecognised broker status is
    surfaced rather than silently coerced to something it is not.
    """

    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    filled_avg_price: float | None
    status: str
    order_type: str = "market"
    time_in_force: str = "day"
    submitted_at: str | None = None
    raw_status: str | None = None

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("filled", "canceled", "expired", "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "filled_avg_price": self.filled_avg_price,
            "status": self.status,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "submitted_at": self.submitted_at,
        }


_ALPACA_STATUS_MAP = {
    "new": "accepted",
    "accepted": "accepted",
    "pending_new": "accepted",
    "partially_filled": "partially_filled",
    "filled": "filled",
    "done_for_day": "accepted",
    "canceled": "canceled",
    "cancelled": "canceled",
    "expired": "expired",
    "replaced": "accepted",
    "pending_cancel": "accepted",
    "pending_replace": "accepted",
    "rejected": "rejected",
    "suspended": "unknown",
    "stopped": "unknown",
    "calculated": "accepted",
}


def _map_status(raw: str | None) -> str:
    if not raw:
        return "unknown"
    return _ALPACA_STATUS_MAP.get(str(raw).lower(), "unknown")


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerAccount:
    """The broker account facts a paper run needs, from the read-only endpoint.

    Only fields Alpaca documents on ``GET /v2/account`` are read. The optional
    ones default to ``None`` so an older or trimmed response is still usable
    rather than mis-parsed; a missing field is reported as unknown, never as a
    fabricated zero.
    """

    account_id: str
    status: str
    cash: float
    equity: float
    buying_power: float
    currency: str
    trading_blocked: bool
    account_blocked: bool
    account_number: str | None = None
    portfolio_value: float | None = None
    last_equity: float | None = None
    long_market_value: float | None = None
    short_market_value: float | None = None
    multiplier: str | None = None
    shorting_enabled: bool | None = None
    transfers_blocked: bool | None = None
    trade_suspended_by_user: bool | None = None
    daytrade_count: int | None = None

    def restrictions(self) -> list[str]:
        """The account restrictions that are active, as a closed vocabulary.

        A blocked or suspended account is the thing a paper run must know before
        it acts, so this is a named list rather than prose. ``shorting_enabled``
        being false is reported too, since it constrains a short strategy.
        """
        active: list[str] = []
        if self.trading_blocked:
            active.append("TRADING_BLOCKED")
        if self.account_blocked:
            active.append("ACCOUNT_BLOCKED")
        if self.transfers_blocked:
            active.append("TRANSFERS_BLOCKED")
        if self.trade_suspended_by_user:
            active.append("TRADE_SUSPENDED_BY_USER")
        if self.shorting_enabled is False:
            active.append("SHORTING_DISABLED")
        return active

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "status": self.status,
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "currency": self.currency,
            "trading_blocked": self.trading_blocked,
            "account_blocked": self.account_blocked,
            "account_number": self.account_number,
            "portfolio_value": self.portfolio_value,
            "last_equity": self.last_equity,
            "long_market_value": self.long_market_value,
            "short_market_value": self.short_market_value,
            "multiplier": self.multiplier,
            "shorting_enabled": self.shorting_enabled,
            "transfers_blocked": self.transfers_blocked,
            "trade_suspended_by_user": self.trade_suspended_by_user,
            "daytrade_count": self.daytrade_count,
            "restrictions": self.restrictions(),
        }


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperRiskLimits:
    """The local limits enforced *before* any order reaches the broker.

    Every one is explicit and finite; there is no "unlimited" default for the
    notional cap, because an order with no cap is the thing a safe default is
    meant to prevent.

    ``max_order_notional`` is a *local* ceiling on one order's notional. It is
    enforced *in addition to* the account's own buying power: an order is refused
    when it exceeds either. ``allow_short`` defaults ``False`` because a cash
    paper account typically has shorting disabled, and a short must be a
    deliberate opt-in rather than an accident of a SELL with no holding.
    """

    max_order_notional: float = 25.0
    """A *local* ceiling on one order's notional, and the binding second layer.

    Sized to the smallest account this path is used with (a ~50 USD paper
    account), so it actually constrains an order instead of being nominal. It is
    enforced *in addition to* the account's own buying power: an order is refused
    when it exceeds either. ``order_notional`` (the target size) must not exceed
    it, checked at configuration time, so ``order_notional <= max_order_notional``
    always holds.
    """
    max_open_positions: int = 5
    max_daily_orders: int = 10
    allowed_symbols: tuple[str, ...] = ()
    """An empty tuple means "no allowlist"; a non-empty one is enforced."""
    allow_short: bool = False
    """When ``False`` (the default), a SELL beyond the held quantity is refused."""
    buying_power_fraction: float = 1.0
    """The most of the account's buying power one order may commit, as a fraction.

    ``1.0`` means "up to the whole buying power"; a smaller value leaves headroom.
    """

    def __post_init__(self) -> None:
        if not (self.max_order_notional > 0):
            raise PaperRiskRejected("max_order_notional must be positive")
        if self.max_open_positions < 1:
            raise PaperRiskRejected("max_open_positions must be at least 1")
        if self.max_daily_orders < 1:
            raise PaperRiskRejected("max_daily_orders must be at least 1")
        if not 0.0 < self.buying_power_fraction <= 1.0:
            raise PaperRiskRejected(
                "buying_power_fraction must be in (0, 1], got "
                f"{self.buying_power_fraction}"
            )


@dataclass(frozen=True)
class BrokerPosition:
    """One position as the broker reports it (``GET /v2/positions``).

    Only the fields a paper order decision needs. A missing/blank field is
    reported as ``0.0`` for a quantity (a position with no quantity is not a
    position) and ``None`` for a price (an unknown price is not zero).
    """

    symbol: str
    quantity: float
    market_value: float | None
    average_entry_price: float | None
    side: str | None = None

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "average_entry_price": self.average_entry_price,
            "side": self.side,
        }


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class BrokerHttpResponse:
    """The part of a broker HTTP response this adapter needs, and nothing else."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


BrokerTransport = Callable[..., BrokerHttpResponse]


def _default_transport(
    *, method: str, url: str, headers: Mapping, params: Mapping, json_body: Any, timeout: float
) -> BrokerHttpResponse:
    import requests

    response = requests.request(
        method,
        url,
        headers=dict(headers),
        params=dict(params or {}),
        json=json_body,
        timeout=timeout,
    )
    return BrokerHttpResponse(response.status_code, response.text)


class _TransportTimeout(BrokerError):
    """The request may or may not have reached the broker. Reconcile, do not retry."""


# ---------------------------------------------------------------------------
# The broker
# ---------------------------------------------------------------------------

_CLIENT_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,48}$")


@dataclass
class _OrderIntent:
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    notional: float
    submitted_at: str


class AlpacaPaperBroker:
    """A paper-only Alpaca broker client with explicit, fail-closed defaults.

    Constructing this class performs **no** network call. :meth:`connect` performs
    the read-only account check; :meth:`submit_order` places an order and requires
    ``allow_orders=True``, so an accidental construction cannot place an order.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = PAPER_BASE_URL,
        transport: BrokerTransport | None = None,
        timeout: float = REQUEST_TIMEOUT,
        limits: PaperRiskLimits | None = None,
        state_path: str | os.PathLike | None = None,
    ) -> None:
        # Endpoint first: refuse a live host before any credential is even read.
        self.base_url = assert_paper_endpoint(base_url)

        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        key, secret = key.strip(), secret.strip()
        if not key or not secret:
            # Do not fall back to a live credential variable, even if it is set.
            live_present = [name for name in _LIVE_KEY_ENVS if os.environ.get(name)]
            hint = (
                f" (found {', '.join(live_present)}, which are refused: this adapter uses "
                f"{API_KEY_ENV}/{API_SECRET_ENV} only)"
                if live_present
                else ""
            )
            raise BrokerNotConfiguredError(
                f"no Alpaca paper credentials: set {API_KEY_ENV} and {API_SECRET_ENV}{hint}"
            )
        self._key = key
        self._secret = secret
        self._transport = transport or _default_transport
        self.timeout = timeout
        self.limits = limits or PaperRiskLimits()
        self.state_path = state_path

        from .state import BrokerAccountState, BrokerStateStore

        self._store = BrokerStateStore(state_path) if state_path else None
        self._state: BrokerAccountState = (
            self._store.load() if self._store else BrokerAccountState()
        )
        self._orders_submitted: list[str] = []
        # The broker's own view of positions, refreshed on demand. ``None`` means
        # "not read this process", which is *not* the same as "flat": a sell
        # against an unknown position is refused rather than assumed to be flat.
        self._positions: dict[str, BrokerPosition] | None = None
        self._account: BrokerAccount | None = None

    # -- headers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """The auth headers. Never logged, never serialised."""
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }

    def _redact(self, text: str) -> str:
        out = text
        for secret in (self._key, self._secret):
            if secret:
                out = out.replace(secret, "***")
        return out

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping | None = None,
        json_body: Any = None,
        reconcile_ok: bool = False,
    ) -> Any:
        """One bounded request. A transport timeout is surfaced, not blind-retried."""
        url = f"{self.base_url}{path}"
        assert_paper_endpoint(url)  # belt and braces: the URL is re-checked
        try:
            response = self._transport(
                method=method,
                url=url,
                headers=self._headers(),
                params=params or {},
                json_body=json_body,
                timeout=self.timeout,
            )
        except Exception as exc:  # a transport failure is reported, never hidden
            if reconcile_ok:
                raise _TransportTimeout(
                    f"the request to {path} timed out or failed "
                    f"({type(exc).__name__}); the outcome is unknown, so reconcile by "
                    "client order id rather than re-submitting"
                ) from None
            raise BrokerError(
                f"the broker request to {path} failed: {type(exc).__name__}"
            ) from None

        if response.status_code >= 400:
            body = self._redact((response.text or "")[:200])
            if response.status_code in (401, 403):
                raise BrokerRejectedError(
                    f"the broker refused the credential (HTTP {response.status_code}): {body}"
                )
            raise BrokerRejectedError(
                f"the broker answered HTTP {response.status_code}: {body}"
            )
        if not response.text:
            return {}
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            raise BrokerError(
                f"the broker did not answer with JSON for {path}"
            ) from None

    # -- read-only account check ------------------------------------------

    def connect(self) -> BrokerAccount:
        """Read the account. Places no order. Fails closed on any problem.

        This is the only method a smoke test needs: it proves the credentials work
        against the paper endpoint and reports the account's figures and
        restrictions, without submitting anything.
        """
        return self.refresh_account()

    def refresh_account(self) -> BrokerAccount:
        """Re-read ``GET /v2/account`` and update the cached account state.

        Distinct from :meth:`connect` only in intent: a running worker calls this
        before each order so the buying-power and blocked-account checks are made
        against the account *now*, not against a figure read at start-up. It places
        no order and changes nothing at the broker.
        """
        payload = self._request("GET", "/v2/account")
        account = self._parse_account(payload)
        self._account = account
        self._state.endpoint = self.base_url
        self._state.account_id = account.account_id
        self._state.cash = account.cash
        self._state.equity = account.equity
        self._state.buying_power = account.buying_power
        self._state.currency = account.currency
        self._state.trading_blocked = account.trading_blocked
        self._state.account_blocked = account.account_blocked
        self._state.confirmed = True
        self._state.confirmed_at = datetime.now(UTC).isoformat()
        if self._store:
            self._store.save(self._state)
        return account

    @staticmethod
    def _parse_account(payload: Any) -> BrokerAccount:
        if not isinstance(payload, dict) or "id" not in payload:
            raise BrokerError("the broker account response was not the expected object")
        return BrokerAccount(
            account_id=str(payload.get("id", "")),
            status=str(payload.get("status", "")),
            cash=_as_float(payload.get("cash")),
            equity=_as_float(payload.get("equity")),
            buying_power=_as_float(payload.get("buying_power")),
            currency=str(payload.get("currency", "")),
            trading_blocked=bool(payload.get("trading_blocked", False)),
            account_blocked=bool(payload.get("account_blocked", False)),
            account_number=_as_optional_str(payload.get("account_number")),
            portfolio_value=_as_optional_float(payload.get("portfolio_value")),
            last_equity=_as_optional_float(payload.get("last_equity")),
            long_market_value=_as_optional_float(payload.get("long_market_value")),
            short_market_value=_as_optional_float(payload.get("short_market_value")),
            multiplier=_as_optional_str(payload.get("multiplier")),
            shorting_enabled=_as_optional_bool(payload.get("shorting_enabled")),
            transfers_blocked=_as_optional_bool(payload.get("transfers_blocked")),
            trade_suspended_by_user=_as_optional_bool(payload.get("trade_suspended_by_user")),
            daytrade_count=_as_optional_int(payload.get("daytrade_count")),
        )

    def positions(self, *, refresh: bool = True) -> dict[str, BrokerPosition]:
        """Read ``GET /v2/positions`` (the broker's own positions), keyed by symbol.

        Cached per process. ``refresh=False`` returns the cached view (or an empty
        mapping if never read) without a request — useful for a read-only view that
        must not call out. A position with a zero quantity is dropped, so a closed
        position does not read as open.
        """
        if refresh or self._positions is None:
            payload = self._request("GET", "/v2/positions")
            if not isinstance(payload, list):
                raise BrokerError("the broker positions response was not the expected list")
            found: dict[str, BrokerPosition] = {}
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("symbol", "")).upper()
                if not symbol:
                    continue
                quantity = _as_float(entry.get("qty"))
                if quantity == 0.0:
                    continue
                found[symbol] = BrokerPosition(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=_as_optional_float(entry.get("market_value")),
                    average_entry_price=_as_optional_float(entry.get("avg_entry_price")),
                    side=_as_optional_str(entry.get("side")),
                )
            self._positions = found
            self._state.positions = {s: p.to_dict() for s, p in sorted(found.items())}
            self._state.positions_at = datetime.now(UTC).isoformat()
            if self._store:
                self._store.save(self._state)
        return dict(self._positions)

    def open_position_count(self, *, refresh: bool = True) -> int:
        """How many positions the broker currently holds."""
        return len(self.positions(refresh=refresh))

    def daily_orders(self) -> int:
        """The number of orders accepted today, read from the persisted state.

        A count whose stored date is not today reads as zero, so a new UTC day
        starts fresh without an explicit reset. The count is durable: a restart on
        the same day sees the same number.
        """
        if self._state.daily_orders_date != _today():
            return 0
        return self._state.daily_orders_count

    def _record_daily_order(self) -> None:
        """Count one accepted order against today, rolling the day over first."""
        today = _today()
        if self._state.daily_orders_date != today:
            self._state.daily_orders_date = today
            self._state.daily_orders_count = 0
        self._state.daily_orders_count += 1
        if self._store:
            self._store.save(self._state)

    # -- order lifecycle ---------------------------------------------------

    def _check_risk(
        self, *, symbol: str, quantity: float, side: str, reference_price: float
    ) -> float:
        """Enforce the local limits and return the order notional. Raises to refuse.

        Checked *before* any request, so a limit breach is a local refusal rather
        than a hope that the broker will reject it. Order of checks: the cheap,
        purely-local ones first (symbol, daily count, price/quantity shape), then
        the account-state ones (buying power, open positions, long-only), so a
        malformed order is refused without needing a fresh account read.
        """
        if self.limits.allowed_symbols and symbol.upper() not in {
            s.upper() for s in self.limits.allowed_symbols
        }:
            raise PaperRiskRejected(f"{symbol} is not in the allowed symbol list")
        if self.daily_orders() >= self.limits.max_daily_orders:
            raise PaperRiskRejected(
                f"the daily order limit ({self.limits.max_daily_orders}) is reached"
            )
        if not math.isfinite(reference_price) or not math.isfinite(quantity):
            raise PaperRiskRejected(
                f"a finite reference price and quantity are required, got "
                f"price={reference_price}, quantity={quantity}"
            )
        if not (reference_price > 0) or quantity <= 0:
            raise PaperRiskRejected(
                f"a positive reference price and quantity are required, got "
                f"price={reference_price}, quantity={quantity}"
            )
        notional = quantity * reference_price
        if notional > self.limits.max_order_notional:
            raise PaperRiskRejected(
                f"the order notional {notional:.2f} exceeds the cap "
                f"{self.limits.max_order_notional:.2f}"
            )

        side = (side or "").lower()
        if side == "sell":
            self._check_sell_is_not_a_short(symbol=symbol, quantity=quantity)
        else:
            self._check_buy_against_account(symbol=symbol, notional=notional)
        return notional

    def _check_buy_against_account(self, *, symbol: str, notional: float) -> None:
        """A BUY must fit the account's buying power and the open-position cap."""
        account = self._account
        if account is None:
            raise PaperRiskRejected(
                "the account has not been read; refresh it before an order so buying "
                "power is known rather than assumed"
            )
        affordable = account.buying_power * self.limits.buying_power_fraction
        if notional > affordable:
            raise PaperRiskRejected(
                f"the order notional {notional:.2f} exceeds the affordable buying power "
                f"{affordable:.2f} (buying power {account.buying_power:.2f} x "
                f"fraction {self.limits.buying_power_fraction})"
            )
        # The open-position cap needs the broker's positions. When they have been
        # read, a buy that would open a *new* symbol past the cap is refused; when
        # they have not, the worker reads them before ordering, so this is only
        # skipped by a caller that deliberately did not ask the broker.
        if self._positions is None:
            return
        held = symbol.upper() in self._positions
        if not held and len(self._positions) >= self.limits.max_open_positions:
            raise PaperRiskRejected(
                f"opening {symbol} would exceed the open-position limit "
                f"({self.limits.max_open_positions}); {len(self._positions)} are open"
            )

    def _check_sell_is_not_a_short(self, *, symbol: str, quantity: float) -> None:
        """A SELL must not create a short unless shorting is explicitly allowed.

        The broker's own positions are required: a sell against an *unknown*
        holding is refused rather than assumed to be flat, because assuming flat
        would turn a long-only account's accidental short into a placed order.
        """
        if self._positions is None:
            raise PaperRiskRejected(
                "positions have not been read; refresh them before a sell so the held "
                "quantity is known rather than assumed to be zero"
            )
        held = self._positions.get(symbol.upper())
        held_quantity = held.quantity if held is not None else 0.0
        if quantity <= held_quantity:
            return
        account = self._account
        account_allows_short = account is not None and account.shorting_enabled is True
        if not self.limits.allow_short or not account_allows_short:
            raise PaperRiskRejected(
                f"selling {quantity} of {symbol} exceeds the held {held_quantity}; this "
                "would open a short, which is refused (shorting is not enabled for this "
                "account or allow_short is False)"
            )

    def submit_order(
        self,
        *,
        symbol: str,
        quantity: float,
        side: str,
        reference_price: float,
        client_order_id: str,
        order_type: str = "market",
        time_in_force: str = "day",
        allow_orders: bool = False,
        refresh: bool = False,
    ) -> BrokerOrder:
        """Submit a paper order, safely.

        Refuses unless ``allow_orders=True`` (an explicit, deliberate setting), the
        account has been confirmed by :meth:`connect`, the credentials and endpoint
        are valid, and the local risk limits pass. The order carries a
        ``client_order_id`` so a lost response can be reconciled rather than
        re-submitted. Intent is persisted *before* the request, so a crash between
        submission and confirmation is recoverable.

        ``refresh`` re-reads the account and the broker's positions immediately
        before the risk check, so buying power and held quantity are the account's
        *current* figures rather than a start-up snapshot. It defaults ``False`` to
        preserve the exact request sequence a caller already relies on; the trading
        worker passes ``True`` so every order is checked against live account state.
        """
        if not allow_orders:
            raise PaperRiskRejected(
                "order submission is disabled: pass allow_orders=True to place a paper "
                "order deliberately. Construction alone never places an order."
            )
        if not self._state.confirmed:
            raise BrokerError(
                "the account has not been confirmed; call connect() first so the broker "
                "state is known before an order is placed"
            )
        if self._state.account_blocked or self._state.trading_blocked:
            raise BrokerError("the broker account is blocked from trading")

        symbol = symbol.upper()
        if not _CLIENT_ORDER_ID_RE.match(client_order_id):
            raise PaperRiskRejected(
                f"client_order_id {client_order_id!r} is not a valid stable id"
            )

        # Idempotency: a client_order_id we already know is never re-submitted.
        if client_order_id in self._state.orders:
            return self._reconcile(client_order_id)

        if refresh:
            self.refresh_account()
            self.positions(refresh=True)
            if self._state.account_blocked or self._state.trading_blocked:
                raise BrokerError("the broker account is blocked from trading")

        notional = self._check_risk(
            symbol=symbol, quantity=quantity, side=side, reference_price=reference_price
        )

        intent = _OrderIntent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            notional=notional,
            submitted_at=datetime.now(UTC).isoformat(),
        )
        # Persist intent before the request, so a crash here is recoverable.
        self._state.orders[client_order_id] = {
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": intent.side,
            "quantity": quantity,
            "status": "submitting",
            "broker_order_id": None,
            "submitted_at": intent.submitted_at,
        }
        if self._store:
            self._store.save(self._state)

        body = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.lower(),
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
        }
        try:
            payload = self._request("POST", "/v2/orders", json_body=body, reconcile_ok=True)
        except _TransportTimeout:
            # The outcome is unknown. Reconcile by client order id; never re-submit.
            return self._reconcile(client_order_id)
        except BrokerRejectedError:
            self._state.orders[client_order_id]["status"] = "rejected"
            if self._store:
                self._store.save(self._state)
            raise

        order = self._parse_order(payload, fallback_client_id=client_order_id)
        self._record_order(order)
        self._record_daily_order()
        return order

    def _reconcile(self, client_order_id: str) -> BrokerOrder:
        """Ask the broker what became of a client order id, without re-submitting."""
        payload = self._request(
            "GET",
            "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
        )
        if isinstance(payload, dict) and "client_order_id" in payload:
            order = self._parse_order(payload)
            self._record_order(order)
            return order
        raise BrokerError(
            f"the broker has no record of client order id {client_order_id!r}; the "
            "submission did not reach it, and it is not safe to guess, so the order is "
            "left unresolved rather than re-submitted"
        )

    def get_order(self, broker_order_id: str) -> BrokerOrder:
        payload = self._request("GET", f"/v2/orders/{broker_order_id}")
        return self._parse_order(payload)

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        self._request("DELETE", f"/v2/orders/{broker_order_id}")
        return self.get_order(broker_order_id)

    def _record_order(self, order: BrokerOrder) -> None:
        self._state.orders[order.client_order_id] = {
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
        if self._store:
            self._store.save(self._state)

    def apply_fill(self, order: BrokerOrder) -> bool:
        """Mark an order's fill applied exactly once. Returns whether it was new.

        The broker's own account already reflects a fill; this guard exists so a
        caller that mirrors broker fills into any *other* book cannot apply the
        same fill twice across a restart. Keyed by client order id, so it survives
        a restart rather than living in memory.
        """
        if order.status not in ("filled", "partially_filled"):
            return False
        if order.client_order_id in self._state.applied_fills:
            return False
        self._state.applied_fills.add(order.client_order_id)
        if self._store:
            self._store.save(self._state)
        return True

    def _parse_order(self, payload: Any, fallback_client_id: str | None = None) -> BrokerOrder:
        if not isinstance(payload, dict):
            raise BrokerError("the broker order response was not the expected object")
        client_id = payload.get("client_order_id") or fallback_client_id
        if not client_id:
            raise BrokerError("the broker order response carried no client_order_id")
        raw_status = payload.get("status")
        return BrokerOrder(
            client_order_id=str(client_id),
            broker_order_id=(str(payload["id"]) if payload.get("id") else None),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")),
            quantity=_as_float(payload.get("qty")),
            filled_quantity=_as_float(payload.get("filled_qty")),
            filled_avg_price=(
                _as_float(payload["filled_avg_price"])
                if payload.get("filled_avg_price") not in (None, "")
                else None
            ),
            status=_map_status(raw_status),
            order_type=str(payload.get("type", "market")),
            time_in_force=str(payload.get("time_in_force", "day")),
            submitted_at=payload.get("submitted_at"),
            raw_status=str(raw_status) if raw_status is not None else None,
        )


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    """Parse a documented numeric field, or ``None`` when absent/unparseable.

    Distinct from :func:`_as_float`: a missing optional field is *unknown*, not
    zero, so it is reported as ``None`` rather than a fabricated figure.
    """
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _as_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


@dataclass
class BrokerStatus:
    """A secret-free report of the broker path's configuration."""

    paper_endpoint: str
    endpoint_ok: bool
    endpoint_detail: str
    key_env: str
    secret_env: str
    key_present: bool
    secret_present: bool
    live_credentials_present: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_endpoint": self.paper_endpoint,
            "endpoint_ok": self.endpoint_ok,
            "endpoint_detail": self.endpoint_detail,
            "key_env": self.key_env,
            "secret_env": self.secret_env,
            "key_present": self.key_present,
            "secret_present": self.secret_present,
            "live_credentials_present": list(self.live_credentials_present),
        }


def broker_status(*, base_url: str = PAPER_BASE_URL) -> BrokerStatus:
    """Report, without any secret value, whether the paper path is configured.

    Prints presence only: whether the paper key/secret env vars are set and
    non-blank, whether the endpoint is the approved paper host, and whether any
    live credential variable is *present* (which is a misconfiguration signal,
    never something this adapter uses). Never the value, any part of it, or its
    length.
    """
    endpoint_ok = True
    detail = "approved paper endpoint"
    try:
        assert_paper_endpoint(base_url)
    except LiveEndpointRefusedError as exc:
        endpoint_ok = False
        detail = str(exc)
    return BrokerStatus(
        paper_endpoint=PAPER_BASE_URL,
        endpoint_ok=endpoint_ok,
        endpoint_detail=detail,
        key_env=API_KEY_ENV,
        secret_env=API_SECRET_ENV,
        key_present=bool(os.environ.get(API_KEY_ENV, "").strip()),
        secret_present=bool(os.environ.get(API_SECRET_ENV, "").strip()),
        live_credentials_present=[n for n in _LIVE_KEY_ENVS if os.environ.get(n)],
    )
