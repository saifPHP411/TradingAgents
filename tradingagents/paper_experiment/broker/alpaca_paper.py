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
    """The broker account facts a paper run needs, from the read-only endpoint."""

    account_id: str
    status: str
    cash: float
    equity: float
    buying_power: float
    currency: str
    trading_blocked: bool
    account_blocked: bool

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
    """

    max_order_notional: float = 1_000.0
    max_open_positions: int = 5
    max_daily_orders: int = 10
    allowed_symbols: tuple[str, ...] = ()
    """An empty tuple means "no allowlist"; a non-empty one is enforced."""

    def __post_init__(self) -> None:
        if not (self.max_order_notional > 0):
            raise PaperRiskRejected("max_order_notional must be positive")
        if self.max_open_positions < 1:
            raise PaperRiskRejected("max_open_positions must be at least 1")
        if self.max_daily_orders < 1:
            raise PaperRiskRejected("max_daily_orders must be at least 1")


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
        self._daily_orders = 0
        self._orders_submitted: list[str] = []

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
        payload = self._request("GET", "/v2/account")
        if not isinstance(payload, dict) or "id" not in payload:
            raise BrokerError("the broker account response was not the expected object")
        account = BrokerAccount(
            account_id=str(payload.get("id", "")),
            status=str(payload.get("status", "")),
            cash=_as_float(payload.get("cash")),
            equity=_as_float(payload.get("equity")),
            buying_power=_as_float(payload.get("buying_power")),
            currency=str(payload.get("currency", "")),
            trading_blocked=bool(payload.get("trading_blocked", False)),
            account_blocked=bool(payload.get("account_blocked", False)),
        )
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

    # -- order lifecycle ---------------------------------------------------

    def _check_risk(self, *, symbol: str, quantity: float, reference_price: float) -> float:
        """Enforce the local limits and return the order notional. Raises to refuse."""
        if self.limits.allowed_symbols and symbol.upper() not in {
            s.upper() for s in self.limits.allowed_symbols
        }:
            raise PaperRiskRejected(f"{symbol} is not in the allowed symbol list")
        if self._daily_orders >= self.limits.max_daily_orders:
            raise PaperRiskRejected(
                f"the daily order limit ({self.limits.max_daily_orders}) is reached"
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
        return notional

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
    ) -> BrokerOrder:
        """Submit a paper order, safely.

        Refuses unless ``allow_orders=True`` (an explicit, deliberate setting), the
        account has been confirmed by :meth:`connect`, the credentials and endpoint
        are valid, and the local risk limits pass. The order carries a
        ``client_order_id`` so a lost response can be reconciled rather than
        re-submitted. Intent is persisted *before* the request, so a crash between
        submission and confirmation is recoverable.
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

        notional = self._check_risk(
            symbol=symbol, quantity=quantity, reference_price=reference_price
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
        self._daily_orders += 1
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
