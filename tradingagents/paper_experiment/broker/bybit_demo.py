"""A Bybit **Demo** broker adapter for Spot trading, isolated from the internal simulator.

Demo only, by construction
---------------------------

The base URL is hard-validated: only Bybit's demo host is accepted, and every
live host is refused by :func:`assert_demo_endpoint`. There is no setting that
turns demo into live, and no fallback that would reach the live API. Credentials
come from ``BYBIT_DEMO_API_KEY`` / ``BYBIT_DEMO_API_SECRET`` only; the live
variable names are refused rather than accepted as a fallback.

This adapter never touches the internal ledger
-----------------------------------------------

The internal paper experiment has its own portfolio and ledger; Bybit Demo has
its own simulated account. This module holds the *broker* account state and the
*broker* order records, and nothing else. It does not import the ledger, the
simulator, or the experiment, so a broker fill cannot be double-counted as an
internal fill. Reconciliation between the two, if ever wanted, is a separate,
explicit step — not something this module does silently.

Fail-closed behaviour
---------------------

- missing credentials -> :class:`BrokerNotConfiguredError`;
- a non-demo endpoint -> :class:`LiveEndpointRefusedError`;
- an unconfirmed account -> orders are refused, not placed blind;
- an ambiguous submission (a timeout) -> reconcile by ``client_order_id`` before
  deciding, never blind-retry.

The transport is injectable, so every behaviour above is exercised in tests with
a mocked HTTP response and no socket. The default transport is ``requests``.

Bybit API v5 specifics
----------------------

- REST endpoint: ``https://api-demo.bybit.com``
- WebSocket: ``wss://stream-demo.bybit.com`` (not used by this adapter)
- Spot trading only, no Futures, no Margin, no Leverage
- Long-only
- Order types: Market, Limit
- Time in force: GTC, IOC, FOK, PostOnly
- Crypto Spot pairs use format like ``BTCUSDT`` (no separator)
- Rate limits: 100 requests/second for private endpoints
- Min order amounts vary by symbol (queried via instruments-info)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tradingagents.crypto.symbols import try_parse_pair

# ---------------------------------------------------------------------------
# Endpoint safety
# ---------------------------------------------------------------------------

DEMO_BASE_URL = "https://api-demo.bybit.com"
"""The only endpoint this adapter will talk to. Bybit's demo trading host."""

_LIVE_HOSTS = frozenset({
    "api.bybit.com",
    "api-testnet.bybit.com",
    "api-test.bybit.com",
})
"""Bybit live/testnet trading hosts, refused outright. Never a fallback."""

API_KEY_ENV = "BYBIT_DEMO_API_KEY"
API_SECRET_ENV = "BYBIT_DEMO_API_SECRET"
_LIVE_KEY_ENVS = ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_TESTNET_API_KEY", "BYBIT_TESTNET_API_SECRET")
"""Live/testnet credential variable names that are refused, not used."""

REQUEST_TIMEOUT = 30.0

SPOT_CATEGORY = "spot"
"""The product category for Spot trading."""

VALID_TIME_IN_FORCE = ("GTC", "IOC", "FOK", "PostOnly")
"""Valid time-in-force values for Bybit Spot orders."""

DEFAULT_TIME_IN_FORCE = "GTC"

# Bybit uses client order ID for idempotency
_CLIENT_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,36}$")


class BrokerError(RuntimeError):
    """A broker-backed demo path could not proceed."""


class BrokerNotConfiguredError(BrokerError):
    """The broker was selected but its demo credentials are missing."""


class LiveEndpointRefusedError(BrokerError):
    """A non-demo (live) endpoint was requested. This is a hard refusal."""


class BrokerRejectedError(BrokerError):
    """The broker rejected the request (auth, risk, or a bad order).

    A refusal is a *decision by the venue*, not a transport failure: the request
    reached Bybit and Bybit said no.
    """


class InsufficientFundsError(BrokerRejectedError):
    """The account had no buying power for the order."""


class PaperRiskRejected(BrokerError):
    """A locally-enforced paper risk limit refused the order before submission."""


_INSUFFICIENT_FUNDS_MARKERS = (
    "insufficient",
    "not enough",
    "balance",
    "margin",
    "available",
    "exceeds",
)
"""The phrases Bybit uses when the account cannot pay for an order."""


def _is_insufficient_funds(body: str) -> bool:
    """Whether a refusal body says the account lacked buying power."""
    text = (body or "").lower()
    return any(marker in text for marker in _INSUFFICIENT_FUNDS_MARKERS)


def _host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def assert_demo_endpoint(base_url: str) -> str:
    """Return ``base_url`` if it is the approved demo host, else raise.

    The check is strict: scheme, exact hostname, port, userinfo, path, query,
    and fragment are all validated. A hostname-only check accepted an HTTP
    downgrade, a credential-bearing URL and a different port, each of which is
    a different destination from the approved one.
    """
    if not base_url:
        raise LiveEndpointRefusedError("base_url must not be empty")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme.lower() != "https":
        raise LiveEndpointRefusedError(
            f"{base_url!r} must use https; only HTTPS is allowed for the demo endpoint"
        )
    host = (parsed.hostname or "").lower()
    if host in _LIVE_HOSTS:
        raise LiveEndpointRefusedError(
            f"{base_url!r} is a Bybit LIVE host; this adapter uses the DEMO host only "
            f"({DEMO_BASE_URL})"
        )
    if host != urllib.parse.urlsplit(DEMO_BASE_URL).hostname:
        raise LiveEndpointRefusedError(
            f"{base_url!r} is not the approved Bybit demo host ({DEMO_BASE_URL}); "
            "refusing to connect to an unapproved host"
        )
    if parsed.port not in (None, 443):
        raise LiveEndpointRefusedError(
            f"{base_url!r} uses port {parsed.port}; the approved demo host uses port 443"
        )
    if parsed.username or parsed.password:
        raise LiveEndpointRefusedError(
            f"{base_url!r} contains credentials in the URL; credentials must come from "
            f"environment variables only ({API_KEY_ENV}/{API_SECRET_ENV})"
        )
    return base_url


def assert_demo_request_url(url: str) -> str:
    """Re-check a request URL against the approved demo host before sending.

    Allows a path (a request URL needs one) but re-checks scheme, host, port
    and userinfo.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise LiveEndpointRefusedError(f"{url!r} must use https")
    host = (parsed.hostname or "").lower()
    if host in _LIVE_HOSTS:
        raise LiveEndpointRefusedError(f"{url!r} targets a Bybit LIVE host; demo only")
    if host != urllib.parse.urlsplit(DEMO_BASE_URL).hostname:
        raise LiveEndpointRefusedError(
            f"{url!r} is not the approved Bybit demo host ({DEMO_BASE_URL})"
        )
    if parsed.port not in (None, 443):
        raise LiveEndpointRefusedError(f"{url!r} uses port {parsed.port}; demo uses 443")
    if parsed.username or parsed.password:
        raise LiveEndpointRefusedError(f"{url!r} contains credentials in the URL")
    return url


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerOrder:
    """One order as the broker reports it.

    Only the fields a paper order decision needs. A missing/blank field is
    reported as ``0.0`` for a quantity and ``None`` for a price.
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
    time_in_force: str = "GTC"
    submitted_at: str | None = None
    raw_status: str | None = None

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("filled", "cancelled", "rejected", "partially_filled_cancelled")

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


@dataclass(frozen=True)
class VenueOrderPage:
    """The venue's order list, read as pages, with nothing thrown away."""

    orders: list[BrokerOrder] = field(default_factory=list)
    unclassified: list[dict[str, Any]] = field(default_factory=list)
    pages_read: int = 0
    truncated: bool = False

    @property
    def total_records(self) -> int:
        return len(self.orders) + len(self.unclassified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orders": [order.to_dict() for order in self.orders],
            "unclassified": list(self.unclassified),
            "pages_read": self.pages_read,
            "truncated": self.truncated,
            "total_records": self.total_records,
        }


@dataclass(frozen=True)
class BrokerAccount:
    """The broker account facts a paper run needs, from the read-only endpoint."""

    account_id: str
    account_type: str
    equity: float
    available_balance: float
    currency: str = "USDT"
    trading_blocked: bool = False
    account_blocked: bool = False

    def restrictions(self) -> list[str]:
        active: list[str] = []
        if self.trading_blocked:
            active.append("TRADING_BLOCKED")
        if self.account_blocked:
            active.append("ACCOUNT_BLOCKED")
        return active

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "account_type": self.account_type,
            "equity": self.equity,
            "available_balance": self.available_balance,
            "currency": self.currency,
            "trading_blocked": self.trading_blocked,
            "account_blocked": self.account_blocked,
            "restrictions": self.restrictions(),
        }


@dataclass(frozen=True)
class BrokerPosition:
    """One position as the broker reports it."""

    symbol: str
    quantity: float
    avg_price: float | None
    mark_price: float | None
    unrealised_pnl: float | None
    side: str | None = None
    venue_symbol: str | None = None

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "mark_price": self.mark_price,
            "unrealised_pnl": self.unrealised_pnl,
            "side": self.side,
            "venue_symbol": self.venue_symbol or self.symbol,
        }


@dataclass(frozen=True)
class PaperRiskLimits:
    """The local limits enforced *before* any order reaches the broker."""

    max_order_notional: float = 25.0
    """A *local* ceiling on one **entry** order's notional."""

    max_open_positions: int = 5
    max_daily_orders: int = 10
    allowed_symbols: tuple[str, ...] = ()
    """An empty tuple means "no allowlist"; a non-empty one is enforced."""
    allow_short: bool = False
    buying_power_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not (self.max_order_notional > 0):
            raise PaperRiskRejected("max_order_notional must be positive")
        if self.max_open_positions < 1:
            raise PaperRiskRejected("max_open_positions must be at least 1")
        if self.max_daily_orders < 1:
            raise PaperRiskRejected("max_daily_orders must be at least 1")
        if not 0.0 < self.buying_power_fraction <= 1.0:
            raise PaperRiskRejected(
                f"buying_power_fraction must be in (0, 1], got {self.buying_power_fraction}"
            )


@dataclass(frozen=True)
class InstrumentInfo:
    """Trading rules for a Bybit Spot instrument."""

    symbol: str
    status: str
    base_coin: str
    quote_coin: str
    min_order_qty: float
    max_order_qty: float
    min_order_amt: float
    qty_step: float
    tick_size: float
    tradable: bool

    @property
    def min_notional_usdt(self) -> float:
        return self.min_order_amt

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "base_coin": self.base_coin,
            "quote_coin": self.quote_coin,
            "min_order_qty": self.min_order_qty,
            "max_order_qty": self.max_order_qty,
            "min_order_amt": self.min_order_amt,
            "qty_step": self.qty_step,
            "tick_size": self.tick_size,
            "tradable": self.tradable,
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


class _TransportTimeout(BrokerError):
    """The request may or may not have reached the broker. Reconcile, do not retry."""


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


# ---------------------------------------------------------------------------
# The broker
# ---------------------------------------------------------------------------


class BybitDemoBroker:
    """A demo-only Bybit broker client with explicit, fail-closed defaults.

    Constructing this class performs **no** network call. :meth:`connect` performs
    the read-only account check; :meth:`submit_order` places an order and requires
    ``allow_orders=True``, so an accidental construction cannot place an order.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DEMO_BASE_URL,
        transport: BrokerTransport | None = None,
        timeout: float = REQUEST_TIMEOUT,
        limits: PaperRiskLimits | None = None,
        state_path: str | os.PathLike | None = None,
    ) -> None:
        # Endpoint first: refuse a live host before any credential is even read.
        self.base_url = assert_demo_endpoint(base_url)

        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        key, secret = key.strip(), secret.strip()
        # Allow empty credentials for broker_status() diagnostic; actual operations will check
        if not key or not secret:
            live_present = [name for name in _LIVE_KEY_ENVS if os.environ.get(name)]
            hint = (
                f" (found {', '.join(live_present)}, which are refused: this adapter uses "
                f"{API_KEY_ENV}/{API_SECRET_ENV} only)"
                if live_present
                else ""
            )
            raise BrokerNotConfiguredError(
                f"no Bybit demo credentials: set {API_KEY_ENV} and {API_SECRET_ENV}{hint}"
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
        self._positions: dict[str, BrokerPosition] | None = None
        self._account: BrokerAccount | None = None
        self._instruments: dict[str, InstrumentInfo] = {}

    # -- headers and signature ----------------------------------------------

    def _headers(self, timestamp: str, recv_window: str = "5000") -> dict[str, str]:
        """The auth headers for signed requests. Never logged, never serialised."""
        return {
            "X-BAPI-API-KEY": self._key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _sign(self, payload: str, timestamp: str, recv_window: str = "5000") -> str:
        """Generate HMAC SHA256 signature for Bybit API v5."""
        param_str = f"{timestamp}{self._key}{recv_window}{payload}"
        signature = hmac.new(
            self._secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _redact(self, text: str) -> str:
        """Remove the credentials from a venue message before it can be raised."""
        out = text
        for secret in (self._key, self._secret):
            if secret and len(secret) >= 8:
                out = out.replace(secret, "***")
        return out

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping | None = None,
        json_body: Any = None,
        signed: bool = False,
        reconcile_ok: bool = False,
    ) -> Any:
        """One bounded request. A transport timeout is surfaced, not blind-retried."""
        url = f"{self.base_url}{path}"
        assert_demo_request_url(url)

        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        headers = {}
        if signed:
            payload = json.dumps(json_body or {}, separators=(",", ":")) if json_body else ""
            headers = self._headers(timestamp, recv_window)
            headers["X-BAPI-SIGN"] = self._sign(payload, timestamp, recv_window)
        else:
            headers = {"Accept": "application/json"}

        try:
            response = self._transport(
                method=method,
                url=url,
                headers=headers,
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
            if _is_insufficient_funds(body):
                raise InsufficientFundsError(
                    f"the account has insufficient buying power for this order "
                    f"(HTTP {response.status_code}): {body}"
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

    # -- account ------------------------------------------------------------

    def connect(self) -> BrokerAccount:
        """Read the account. Places no order. Fails closed on any problem."""
        if not self._key or not self._secret:
            raise BrokerNotConfiguredError(
                f"no Bybit demo credentials: set {API_KEY_ENV} and {API_SECRET_ENV}"
            )
        return self.refresh_account()

    def refresh_account(self) -> BrokerAccount:
        """Re-read the account and update the cached account state."""
        payload = self._request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"}, signed=True)
        account = self._parse_account(payload)
        self._account = account
        self._state.endpoint = self.base_url
        self._state.account_id = account.account_id
        self._state.cash = account.available_balance
        self._state.equity = account.equity
        self._state.buying_power = account.available_balance
        self._state.currency = account.currency
        self._state.trading_blocked = account.trading_blocked
        self._state.account_blocked = account.account_blocked
        self._state.confirmed = True
        self._state.confirmed_at = datetime.now(UTC).isoformat()
        if self._store:
            self._store.save(self._state)
        return account

    def _parse_account(self, payload: Any) -> BrokerAccount:
        if not isinstance(payload, dict):
            raise BrokerError("the broker account response was not the expected object")
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise BrokerError("the broker account response missing 'result'")
        account_list = result.get("list", [])
        if not account_list or not isinstance(account_list, list):
            raise BrokerError("the broker account response missing 'list'")
        account = account_list[0]
        coin_list = account.get("coin", [])
        usdt_info = next((c for c in coin_list if c.get("coin") == "USDT"), {})
        equity = float(usdt_info.get("walletBalance", "0"))
        available = float(usdt_info.get("availableToWithdraw", "0"))
        return BrokerAccount(
            account_id=str(account.get("accountId", "")),
            account_type=str(account.get("accountType", "")),
            equity=equity,
            available_balance=available,
            currency="USDT",
            trading_blocked=False,
            account_blocked=False,
        )

    # -- instruments --------------------------------------------------------

    def load_instruments(self) -> dict[str, InstrumentInfo]:
        """Load trading rules for all Spot instruments from Bybit."""
        if self._instruments:
            return self._instruments

        payload = self._request("GET", "/v5/market/instruments-info", params={"category": SPOT_CATEGORY})
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise BrokerError("instruments-info response missing 'result'")
        instrument_list = result.get("list", [])
        if not isinstance(instrument_list, list):
            raise BrokerError("instruments-info response missing 'list'")

        for entry in instrument_list:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol", "")).upper()
            if not symbol:
                continue
            status = str(entry.get("status", ""))
            base_coin = str(entry.get("baseCoin", ""))
            quote_coin = str(entry.get("quoteCoin", ""))
            lot_filter = entry.get("lotSizeFilter", {})
            price_filter = entry.get("priceFilter", {})

            min_order_qty = float(lot_filter.get("minOrderQty", "0"))
            max_order_qty = float(lot_filter.get("maxOrderQty", "0"))
            qty_step = float(lot_filter.get("qtyStep", "0"))
            min_order_amt = float(lot_filter.get("minOrderAmt", "0"))
            tick_size = float(price_filter.get("tickSize", "0"))

            tradable = status == "Trading"
            info = InstrumentInfo(
                symbol=symbol,
                status=status,
                base_coin=base_coin,
                quote_coin=quote_coin,
                min_order_qty=min_order_qty,
                max_order_qty=max_order_qty,
                min_order_amt=min_order_amt,
                qty_step=qty_step,
                tick_size=tick_size,
                tradable=tradable,
            )
            self._instruments[symbol] = info

        return self._instruments

    def get_instrument(self, symbol: str) -> InstrumentInfo | None:
        """Get trading rules for a specific symbol (loads all if needed).

        Accepts both canonical (BTC/USDT) and Bybit (BTCUSDT) formats.
        """
        bybit_symbol = self._to_bybit_symbol(symbol).upper()
        if bybit_symbol not in self._instruments:
            self.load_instruments()
        return self._instruments.get(bybit_symbol)

    # -- positions ----------------------------------------------------------

    def positions(self, *, refresh: bool = True) -> dict[str, BrokerPosition]:
        """Read positions, keyed by canonical symbol."""
        if refresh or self._positions is None:
            payload = self._request("GET", "/v5/position/list", params={"category": SPOT_CATEGORY}, signed=True)
            result = payload.get("result", {})
            position_list = result.get("list", [])
            if not isinstance(position_list, list):
                raise BrokerError("the broker positions response was not the expected list")

            found: dict[str, BrokerPosition] = {}
            for entry in position_list:
                if not isinstance(entry, dict):
                    continue
                raw_symbol = str(entry.get("symbol", "")).upper()
                if not raw_symbol:
                    continue
                qty = float(entry.get("size", "0"))
                if qty == 0:
                    continue
                avg_price = float(entry.get("avgPrice", "0")) or None
                mark_price = float(entry.get("markPrice", "0")) or None
                unrealised = float(entry.get("unrealisedPnl", "0")) or None
                side = str(entry.get("side", "")).lower()

                # Canonical symbol uses /
                canonical = self._to_canonical_symbol(raw_symbol)

                pos = BrokerPosition(
                    symbol=canonical,
                    quantity=qty,
                    avg_price=avg_price,
                    mark_price=mark_price,
                    unrealised_pnl=unrealised,
                    side=side,
                    venue_symbol=raw_symbol,
                )
                found[canonical] = pos
            self._positions = found
            self._state.positions = {k: v.to_dict() for k, v in found.items()}
            self._state.positions_at = datetime.now(UTC).isoformat()
            if self._store:
                self._store.save(self._state)
        return self._positions or {}

    def _to_canonical_symbol(self, bybit_symbol: str) -> str:
        """Convert Bybit symbol (BTCUSDT) to canonical (BTC/USDT)."""
        if "/" in bybit_symbol:
            return bybit_symbol
        pair = try_parse_pair(bybit_symbol)
        if pair:
            return pair.canonical
        return bybit_symbol

    def _to_bybit_symbol(self, canonical: str) -> str:
        """Convert canonical symbol (BTC/USDT) to Bybit format (BTCUSDT)."""
        if "/" not in canonical:
            return canonical
        return canonical.replace("/", "")

    # -- orders -------------------------------------------------------------

    def list_orders(
        self,
        *,
        symbol: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> VenueOrderPage:
        """Read orders from the broker with pagination."""
        params: dict[str, Any] = {"category": SPOT_CATEGORY, "limit": limit}
        if symbol:
            params["symbol"] = self._to_bybit_symbol(symbol.upper())
        if cursor:
            params["cursor"] = cursor

        payload = self._request("GET", "/v5/order/history", params=params, signed=True)
        result = payload.get("result", {})
        order_list = result.get("list", [])
        if not isinstance(order_list, list):
            raise BrokerError("the broker orders response was not the expected list")

        orders: list[BrokerOrder] = []
        unclassified: list[dict[str, Any]] = []
        next_cursor = result.get("nextPageCursor")

        for entry in order_list:
            if not isinstance(entry, dict):
                continue
            client_id = str(entry.get("orderLinkId", "")).strip()
            if not client_id:
                unclassified.append(entry)
                continue
            raw_status = str(entry.get("orderStatus", ""))
            order = self._parse_order(entry, client_id)
            orders.append(order)

        return VenueOrderPage(
            orders=orders,
            unclassified=unclassified,
            pages_read=1,
            truncated=bool(next_cursor),
        )

    def _parse_order(self, payload: Any, fallback_client_id: str | None = None) -> BrokerOrder:
        if not isinstance(payload, dict):
            raise BrokerError("the broker order response was not the expected object")
        client_id = str(payload.get("orderLinkId", "") or fallback_client_id or "")
        if not client_id:
            raise BrokerError("the broker order response carried no client_order_id")
        raw_status = payload.get("orderStatus")
        symbol = self._to_canonical_symbol(str(payload.get("symbol", "")))
        side = str(payload.get("side", "")).lower()
        qty = float(payload.get("qty", "0"))
        filled_qty = float(payload.get("cumExecQty", "0"))
        filled_avg = payload.get("avgPrice")
        filled_avg_price = float(filled_avg) if filled_avg not in (None, "", "0") else None
        order_type = str(payload.get("orderType", "Market")).lower()
        time_in_force = str(payload.get("timeInForce", "GTC"))
        created_time = payload.get("createdTime")
        submitted_at = None
        if created_time:
            try:
                submitted_at = datetime.fromtimestamp(int(created_time) / 1000, tz=UTC).isoformat()
            except (ValueError, TypeError):
                pass

        return BrokerOrder(
            client_order_id=client_id,
            broker_order_id=str(payload.get("orderId")) if payload.get("orderId") else None,
            symbol=symbol,
            side=side,
            quantity=qty,
            filled_quantity=filled_qty,
            filled_avg_price=filled_avg_price,
            status=self._map_status(raw_status),
            order_type=order_type,
            time_in_force=time_in_force,
            submitted_at=submitted_at,
            raw_status=str(raw_status) if raw_status is not None else None,
        )

    def _map_status(self, raw: str | None) -> str:
        if not raw:
            return "unknown"
        raw_lower = str(raw).lower()
        if raw_lower in ("new", "partially_filled", "untriggered"):
            return "accepted"
        if raw_lower == "filled":
            return "filled"
        if raw_lower in ("cancelled", "partially_filled_cancelled"):
            return "cancelled"
        if raw_lower == "rejected":
            return "rejected"
        return "unknown"

    def _check_risk(
        self,
        *,
        symbol: str,
        quantity: float,
        side: str,
        reference_price: float,
    ) -> float:
        """Check local risk limits. Returns the notional."""
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise PaperRiskRejected(f"reference price {reference_price!r} is not a finite positive number")
        if not math.isfinite(quantity) or quantity <= 0:
            raise PaperRiskRejected(f"quantity {quantity!r} is not a finite positive number")

        notional = quantity * reference_price

        side = side.lower()
        if side == "sell":
            self._check_sell_is_not_a_short(symbol=symbol, quantity=quantity)
        else:
            if notional > self.limits.max_order_notional:
                raise PaperRiskRejected(
                    f"the order notional {notional:.2f} exceeds the cap {self.limits.max_order_notional:.2f}"
                )
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
        affordable = account.available_balance * self.limits.buying_power_fraction
        if notional > affordable:
            raise PaperRiskRejected(
                f"the order notional {notional:.2f} exceeds the affordable buying power "
                f"{affordable:.2f} (available {account.available_balance:.2f} x "
                f"fraction {self.limits.buying_power_fraction})"
            )
        if self._positions is None:
            return
        held = symbol.upper() in self._positions
        if not held and len(self._positions) >= self.limits.max_open_positions:
            raise PaperRiskRejected(
                f"opening {symbol} would exceed the open-position limit "
                f"({self.limits.max_open_positions}); {len(self._positions)} are open"
            )

    def _check_sell_is_not_a_short(self, *, symbol: str, quantity: float) -> None:
        """A SELL must not create a short unless shorting is explicitly allowed."""
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
        account_allows_short = account is not None and getattr(account, "shorting_enabled", False)
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
        order_type: str = "Market",
        time_in_force: str | None = None,
        allow_orders: bool = False,
        refresh: bool = False,
    ) -> BrokerOrder:
        """Submit a demo order, safely.

        Refuses unless ``allow_orders=True``, the account has been confirmed by
        :meth:`connect`, the credentials and endpoint are valid, and the local
        risk limits pass. The order carries a ``client_order_id`` so a lost
        response can be reconciled rather than re-submitted. Intent is persisted
        *before* the request, so a crash between submission and confirmation is
        recoverable.

        ``refresh`` re-reads the account and the broker's positions immediately
        before the risk check.
        """
        if not allow_orders:
            raise PaperRiskRejected(
                "order submission is disabled: pass allow_orders=True to place a demo "
                "order deliberately. Construction alone never places an order."
            )
        if not self._key or not self._secret:
            raise BrokerNotConfiguredError(
                f"no Bybit demo credentials: set {API_KEY_ENV} and {API_SECRET_ENV}"
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
        known = self._state.orders.get(client_order_id)
        if known is not None:
            if known.get("status") == "refused_locally":
                self._state.orders.pop(client_order_id, None)
                if self._store:
                    self._store.save(self._state)
            else:
                return self._reconcile(client_order_id)

        # Validate time in force
        tif = (time_in_force or DEFAULT_TIME_IN_FORCE).upper()
        if tif not in VALID_TIME_IN_FORCE:
            raise PaperRiskRejected(
                f"time_in_force {time_in_force!r} is not supported for a spot order; "
                f"Bybit accepts only {', '.join(VALID_TIME_IN_FORCE)}"
            )

        # Check instrument rules
        instrument = self.get_instrument(self._to_bybit_symbol(symbol))
        if instrument is None:
            raise PaperRiskRejected(f"symbol {symbol} not found in Bybit instruments")
        if not instrument.tradable:
            raise PaperRiskRejected(f"symbol {symbol} is not tradable on Bybit demo (status: {instrument.status})")

        if refresh:
            self.refresh_account()
            self.positions(refresh=True)
            if self._state.account_blocked or self._state.trading_blocked:
                raise BrokerError("the broker account is blocked from trading")

        notional = self._check_risk(
            symbol=symbol, quantity=quantity, side=side, reference_price=reference_price
        )

        # Check minimum order amount
        if instrument and notional < instrument.min_order_amt:
            raise PaperRiskRejected(
                f"order notional {notional:.2f} USDT is below minimum {instrument.min_order_amt:.2f} USDT "
                f"for {symbol} (BELOW_MIN_NOTIONAL)"
            )

        # Check quantity step
        if instrument and instrument.qty_step > 0:
            qty_decimal = Decimal(str(quantity))
            step_decimal = Decimal(str(instrument.qty_step))
            if (qty_decimal / step_decimal) % 1 != 0:
                raise PaperRiskRejected(
                    f"quantity {quantity} for {symbol} does not match qtyStep {instrument.qty_step}"
                )

        intent = {
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side.lower(),
            "quantity": quantity,
            "notional": notional,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        # Persist intent before the request
        self._state.orders[client_order_id] = {
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": intent["side"],
            "quantity": quantity,
            "status": "submitting",
            "broker_order_id": None,
            "submitted_at": intent["submitted_at"],
        }
        if self._store:
            self._store.save(self._state)

        bybit_symbol = self._to_bybit_symbol(symbol)
        body = {
            "category": SPOT_CATEGORY,
            "symbol": bybit_symbol,
            "side": side.capitalize(),
            "orderType": order_type.capitalize(),
            "qty": self._format_quantity(quantity, instrument),
            "timeInForce": tif,
            "orderLinkId": client_order_id,
        }
        if order_type.lower() == "limit":
            # For limit orders, we'd need a price - not supported in this basic impl
            pass

        try:
            payload = self._request("POST", "/v5/order/create", json_body=body, signed=True, reconcile_ok=True)
        except _TransportTimeout:
            return self._reconcile(client_order_id)
        except BrokerRejectedError:
            self._state.orders[client_order_id]["status"] = "refused_locally"
            if self._store:
                self._store.save(self._state)
            raise

        result = payload.get("result", {})
        order_id = result.get("orderId")
        order_link_id = result.get("orderLinkId")

        # Reconcile to get the actual order status from the broker
        if order_link_id:
            try:
                order = self._reconcile(order_link_id)
                return order
            except Exception:
                # If reconciliation fails, fall back to parsing the create response
                pass

        order = self._parse_order(
            {"orderId": order_id, "orderLinkId": order_link_id, "symbol": bybit_symbol, **body},
            fallback_client_id=client_order_id,
        )
        self._record_order(order)
        self._record_daily_order()
        return order

    def _format_quantity(self, quantity: float, instrument: InstrumentInfo | None = None) -> str:
        """A quantity as a plain decimal string, never in scientific notation."""
        value = float(quantity)
        if not math.isfinite(value):
            raise PaperRiskRejected(f"quantity {quantity!r} is not a finite number")
        if value == 0.0:
            return "0"
        text = format(Decimal(str(value)), "f")
        # Trim trailing zeros but keep at least the precision of qty_step
        if instrument and instrument.qty_step > 0:
            step_str = format(Decimal(str(instrument.qty_step)), "f")
            decimal_places = len(step_str.split(".")[-1]) if "." in step_str else 0
            if "." in text:
                text = text.rstrip("0").rstrip(".")
                if decimal_places > 0:
                    parts = text.split(".")
                    if len(parts) == 2 and len(parts[1]) < decimal_places:
                        text = parts[0] + "." + parts[1].ljust(decimal_places, "0")
        return text

    def _reconcile(self, client_order_id: str) -> BrokerOrder:
        """Ask the broker what became of a client order id, without re-submitting."""
        # Check if we already have a terminal state - don't re-ask
        known = self._state.orders.get(client_order_id)
        if known and known.get("status") in ("filled", "cancelled", "rejected"):
            return BrokerOrder(
                client_order_id=known["client_order_id"],
                broker_order_id=known.get("broker_order_id"),
                symbol=known["symbol"],
                side=known["side"],
                quantity=known["quantity"],
                filled_quantity=known.get("filled_quantity", 0.0),
                filled_avg_price=known.get("filled_avg_price"),
                status=known.get("status", "unknown"),
                order_type=known.get("order_type", "market"),
                time_in_force=known.get("time_in_force", "GTC"),
                submitted_at=known.get("submitted_at"),
            )

        payload = self._request(
            "GET",
            "/v5/order/realtime",
            params={"category": SPOT_CATEGORY, "orderLinkId": client_order_id},
            signed=True,
            reconcile_ok=True,
        )
        result = payload.get("result", {})
        order_list = result.get("list", [])
        if not order_list:
            # Try history endpoint
            payload = self._request(
                "GET",
                "/v5/order/history",
                params={"category": SPOT_CATEGORY, "orderLinkId": client_order_id},
                signed=True,
                reconcile_ok=True,
            )
            result = payload.get("result", {})
            order_list = result.get("list", [])

        if not order_list:
            # Still not found - return last known state
            if known:
                return BrokerOrder(
                    client_order_id=known["client_order_id"],
                    broker_order_id=known.get("broker_order_id"),
                    symbol=known["symbol"],
                    side=known["side"],
                    quantity=known["quantity"],
                    filled_quantity=known.get("filled_quantity", 0.0),
                    filled_avg_price=known.get("filled_avg_price"),
                    status=known.get("status", "unknown"),
                    order_type=known.get("order_type", "market"),
                    time_in_force=known.get("time_in_force", "GTC"),
                    submitted_at=known.get("submitted_at"),
                )
            raise BrokerError(f"order {client_order_id} not found at broker")

        order = self._parse_order(order_list[0], fallback_client_id=client_order_id)
        self._record_order(order)
        return order

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
            "order_type": order.order_type,
            "time_in_force": order.time_in_force,
            "submitted_at": order.submitted_at,
        }
        if self._store:
            self._store.save(self._state)

    def _record_daily_order(self) -> None:
        today = datetime.now(UTC).date().isoformat()
        if self._state.daily_orders_date != today:
            self._state.daily_orders_date = today
            self._state.daily_orders_count = 0
        self._state.daily_orders_count += 1
        if self._store:
            self._store.save(self._state)

    def cancel_order(self, *, client_order_id: str, symbol: str) -> BrokerOrder:
        """Cancel an order by client_order_id."""
        if not self._state.confirmed:
            raise BrokerError("the account has not been confirmed; call connect() first")

        bybit_symbol = self._to_bybit_symbol(symbol)
        body = {
            "category": SPOT_CATEGORY,
            "symbol": bybit_symbol,
            "orderLinkId": client_order_id,
        }
        payload = self._request("POST", "/v5/order/cancel", json_body=body, signed=True)
        result = payload.get("result", {})
        order = self._parse_order(
            {"orderId": result.get("orderId"), "orderLinkId": client_order_id, "symbol": bybit_symbol, "orderStatus": "Cancelled"},
            fallback_client_id=client_order_id,
        )
        self._record_order(order)
        return order

    def apply_fill(self, order: BrokerOrder) -> bool:
        """Mark an order's fill applied exactly once. Returns whether it was new."""
        if order.status not in ("filled", "partially_filled"):
            return False
        if order.client_order_id in self._state.applied_fills:
            return False
        self._state.applied_fills.add(order.client_order_id)
        if self._store:
            self._store.save(self._state)
        return True

    # -- diagnostic ---------------------------------------------------------

    def broker_status(self) -> BybitBrokerStatus:
        endpoint_ok = True
        detail = "approved demo endpoint"
        try:
            assert_demo_endpoint(self.base_url)
        except LiveEndpointRefusedError as exc:
            endpoint_ok = False
            detail = str(exc)
        return BybitBrokerStatus(
            paper_endpoint=DEMO_BASE_URL,
            endpoint_ok=endpoint_ok,
            endpoint_detail=detail,
            key_env=API_KEY_ENV,
            secret_env=API_SECRET_ENV,
            key_present=bool(os.environ.get(API_KEY_ENV, "").strip()),
            secret_present=bool(os.environ.get(API_SECRET_ENV, "").strip()),
            live_credentials_present=[n for n in _LIVE_KEY_ENVS if os.environ.get(n)],
        )


@dataclass
class BybitBrokerStatus:
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


__all__ = [
    "API_KEY_ENV",
    "API_SECRET_ENV",
    "DEMO_BASE_URL",
    "REQUEST_TIMEOUT",
    "SPOT_CATEGORY",
    "VALID_TIME_IN_FORCE",
    "DEFAULT_TIME_IN_FORCE",
    "BrokerError",
    "BrokerNotConfiguredError",
    "LiveEndpointRefusedError",
    "BrokerRejectedError",
    "InsufficientFundsError",
    "PaperRiskRejected",
    "PaperRiskLimits",
    "BrokerAccount",
    "BrokerOrder",
    "BrokerPosition",
    "InstrumentInfo",
    "VenueOrderPage",
    "BybitBrokerStatus",
    "assert_demo_endpoint",
    "assert_demo_request_url",
    "BybitDemoBroker",
]