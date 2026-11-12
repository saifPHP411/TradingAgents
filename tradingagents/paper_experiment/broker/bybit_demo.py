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

import contextlib
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
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import quote

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

# -- order status vocabulary ------------------------------------------------
# Bybit V5 spells its statuses in CamelCase, while this project's own order
# statuses are lower snake_case. The mapping is explicit rather than a
# lower-cased comparison, because ``PartiallyFilled`` lower-cases to
# ``partiallyfilled`` and ``PartiallyFilledCanceled`` to
# ``partiallyfilledcanceled`` -- neither of which matches the snake_case
# spelling this project uses, so a naive ``.lower()`` sent a real Bybit
# partial fill to "unknown".
#
# Source: https://bybit-exchange.github.io/docs/v5/enum (orderStatus)
BYBIT_ORDER_STATUS_NEW = "New"
BYBIT_ORDER_STATUS_PARTIALLY_FILLED = "PartiallyFilled"
BYBIT_ORDER_STATUS_UNTRIGGERED = "Untriggered"
BYBIT_ORDER_STATUS_REJECTED = "Rejected"
BYBIT_ORDER_STATUS_PARTIALLY_FILLED_CANCELED = "PartiallyFilledCanceled"
BYBIT_ORDER_STATUS_FILLED = "Filled"
BYBIT_ORDER_STATUS_CANCELLED = "Cancelled"
BYBIT_ORDER_STATUS_TRIGGERED = "Triggered"
BYBIT_ORDER_STATUS_DEACTIVATED = "Deactivated"

_BYBIT_STATUS_TO_PROJECT: dict[str, str] = {
    "new": "accepted",
    "partiallyfilled": "partially_filled",
    "partiallyfilledcanceled": "partially_filled_cancelled",
    "untriggered": "accepted",
    "triggered": "accepted",
    "filled": "filled",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "rejected": "rejected",
    "deactivated": "cancelled",
}
"""Bybit's ``orderStatus`` mapped onto this project's order statuses.

A status the venue reports that is not in this table is **unknown**, not
guessed at. ``unknown`` is not terminal, so such an order keeps being read
back rather than being quietly retired.
"""

#: The statuses this project treats as "the order is still working".
_ACTIVE_PROJECT_STATUSES = frozenset({"accepted", "partially_filled"})
"""An order in one of these is live at the venue: it may still fill."""

#: The outcome of a counterpart cancellation, in three closed cases.
PROTECTION_CANCEL_CONFIRMED_TERMINAL = "confirmed_terminal"
"""The venue stated the counterpart is finished.

Only this outcome lets a replacement pair be created. The counterpart is
gone, so the position's only guard is the one that traded, and the
quantity it did not trade still needs a guard of its own.
"""

PROTECTION_CANCEL_STILL_ACTIVE = "still_active"
"""The venue still shows the counterpart working.

The cancel request was accepted, but the order has not finished. Creating
a replacement now would leave **two** live sells against the same
quantity, and the second one to trigger would sell more than the account
holds. The pair stays unresolved and the next cycle tries again.
"""

PROTECTION_CANCEL_UNKNOWN = "unknown"
"""The counterpart's fate could not be established.

The cancel request failed, the venue could not be read, or the venue
answered with a status this project does not recognise. None of those is
evidence that the order is gone, and none of them is evidence that it is
still live. The position is reported **unprotected** and the entry gate
closes, because the system cannot say what the account would do.
"""

#: The tolerance used when comparing a reported cumulative fill with the
#: quantity already applied. The venue reports quantities as decimal strings,
#: so the same fill can come back with a different binary representation.
_FILL_EPSILON = 1e-12

#: The statuses that **prove** an order is finished at the venue.
_TERMINAL_PROJECT_STATUSES = frozenset(
    {
        "filled",
        "cancelled",
        "partially_filled_cancelled",
        "rejected",
    }
)
"""A status in this set is the venue's own confirmation that the order is over.

This set is the **only** way a protection pair may be marked resolved. It is
deliberately an allow-list rather than "anything that is not active": a status
the venue did not report, or one this project does not recognise, is
**unknown**, and unknown is not evidence that anything finished.

The difference matters for an OCO pair. When one leg trades, its counterpart
must be cancelled, and the pair may only be called settled once the venue has
confirmed the counterpart is gone. Reading "the counterpart is not in the
active set" proves nothing: a missing order, a truncated history read and a
status this project has never seen all look exactly the same, and all three
mean the counterpart may still be working. Treating any of them as terminal
would retire a pair whose second leg is still live, and the position would be
left half-guarded with the system believing it is settled.
"""


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


class BrokerRateLimitedError(BrokerError):
    """The venue refused the request because a rate limit was exceeded.

    The request was **not** accepted, so the same intent may be tried again
    after a wait. This is different from a rejection: a rejection means the
    venue saw the order and said no, and this one means the venue never
    considered it.
    """


class DuplicateOrderError(BrokerRejectedError):
    """The venue reports this request as a duplicate of one it already saw.

    The order may well exist at the venue, so the answer is to **reconcile**
    by client order id, never to submit the order again.
    """


class PaperRiskRejected(BrokerError):
    """A locally-enforced paper risk limit refused the order before submission."""


_LOCAL_REFUSAL_STATUS = "refused_locally"
"""The marker for a refusal that never reached the venue.

A *local* refusal and the venue's own *rejected* are different facts. A
local refusal means nothing exists at the broker, so the same intent may be
tried again. The venue's ``rejected`` means the venue saw the order and
refused it: it exists and must be reconciled, never re-submitted. The
marker is this project's own so the two are never confused.
"""


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


def _encode_query_string(params: Mapping) -> str:
    """The query string exactly as it will be sent.

    The keys are sorted and the values are URL-encoded, which is what the
    HTTP client does when it builds the query from a mapping. The signature
    is computed over this string, so it has to be the same string.
    """
    return "&".join(
        f"{quote(str(key), safe='')}={quote(str(params[key]), safe='')}"
        for key in sorted(params, key=str)
    )


def _encode_json_body(body: Any) -> str:
    """The JSON body exactly as it will be sent.

    ``json.dumps`` with its default separators is what the HTTP client
    writes when it is handed a mapping for ``json=``, so the signature is
    computed over that form.
    """
    return json.dumps(body, allow_nan=False)


_OCO_LEG_LABELS = {"stop_loss": "SL", "take_profit": "TP"}
"""How the two legs of a protection pair are named in the audit trail."""


def _oco_action_message(counterpart_key: str, filled_key: str) -> str:
    """The audit line for a counterpart leg that was cancelled.

    The wording is fixed so the dashboard and the tests read the same fact:
    ``"TP cancelled after SL fill"``.
    """
    counterpart = _OCO_LEG_LABELS.get(counterpart_key, counterpart_key)
    filled = _OCO_LEG_LABELS.get(filled_key, filled_key)
    return f"{counterpart} cancelled after {filled} fill"


# The business error codes Bybit documents for V5. A V5 answer carries the
# outcome in ``retCode``; the HTTP status alone does not say whether the
# request was accepted. These are the codes that change what the adapter
# does next.
BYBIT_RATE_LIMIT_CODES = frozenset({10006, 10018})
"""``retCode`` values that mean the request was throttled.

Bybit's own error table: ``10006`` is "Exceeded the API Rate Limit" and
``10018`` is "Exceeded the IP Rate Limit". Both are retryable **after a
wait**, and neither means the venue saw the order.
"""

BYBIT_AUTH_ERROR_CODES = frozenset({10003, 10004, 10005, 10007, 10010, -2015})
"""``retCode`` values that mean the credential or the signature is wrong.

``10004`` is "Error sign, please check your signature generation algorithm",
which is the code a mismatched signature produces.
"""

BYBIT_INSUFFICIENT_FUNDS_CODES = frozenset({110007, 170131})
"""``retCode`` values that mean the account cannot cover the order.

``110007`` is "Available balance is insufficient" and ``170131`` is
"Balance insufficient".
"""

BYBIT_DUPLICATE_ORDER_CODES = frozenset({10014, 20006})
"""``retCode`` values that mean the request duplicated one already seen.

``10014`` is "Invalid duplicate request" and ``20006`` is "reqId is
duplicated". The order may exist at the venue, so the answer is to
**reconcile**, never to submit again.
"""


def _bybit_error_code(payload: Any) -> int | None:
    """The business error code in a V5 answer, if it carries one.

    A successful answer has ``retCode == 0``. An answer with no ``retCode``
    is not treated as an error: some endpoints answer with a plain body.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("retCode")
    if raw is None:
        return None
    try:
        code = int(raw)
    except (TypeError, ValueError):
        return None
    return code or None


def _as_float(value: Any) -> float:
    """Read a numeric field from a Bybit payload as a float.

    Bybit sends quantities as decimal strings. A missing or blank field is
    ``0.0``; a value that is not a finite number is ``0.0`` too, because a
    non-finite quantity that reached the ledger would poison cash and equity.
    """
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _slug(value: float) -> str:
    """Render a number as a short, readable token for an order id.

    A residual protection's client order id has to be **deterministic**: the
    same partial fill must produce the same id on every cycle, or the venue
    would be asked for a new pair each time. The quantity is the thing that
    distinguishes one residual from another, so it is what the token carries.

    The value is rendered in plain decimal notation, never in scientific
    notation: ``str(0.00001)`` is ``1e-05``, and an order id that says
    ``1e-05`` is both unreadable and a different token from the venue's own
    ``0.00001``.
    """
    return format(Decimal(str(value)), "f").replace(".", "p")


def _select_leg(legs: Any) -> dict | None:
    """Pick the one leg that best covers the position.

    A symbol can carry **several** orders of the same kind: after a partial
    fill the original pair is replaced by a residual pair, so both may be
    visible at the venue for a moment. The read keeps them all, and this
    helper picks the one that matters.

    The pick is the leg that is still **working** and covers the **most**
    quantity: that is the one that actually guards the position. A working
    leg is preferred over a finished one, because a filled leg protects
    nothing, and among working legs the larger one covers more of what is
    held. When no leg is working, the largest finished leg is returned so the
    caller can say *which* finished leg it found.

    The choice is made on what the venue reports, never on which row came
    last: the order of the venue's list is not a fact about the protection.
    """
    if not isinstance(legs, list):
        # A caller may hold the single-leg shape from an older read.
        return legs if isinstance(legs, dict) else None
    if not legs:
        return None
    working = [leg for leg in legs if isinstance(leg, dict) and leg.get("is_active")]
    pool = working or [leg for leg in legs if isinstance(leg, dict)]
    if not pool:
        return None
    return max(pool, key=lambda leg: _as_float(leg.get("quantity")))


def _find_leg_by_client_id(legs: Any, client_order_id: str) -> dict | None:
    """Find the one leg that carries this exact client order id.

    A specific order is confirmed by its **id**, never by its kind: two
    orders of the same kind can sit side by side (the original pair and the
    residual pair that replaced it), and only the id says which one the
    caller means. An empty id matches nothing, so a caller that does not
    know which order it created is told "not found" rather than being handed
    an arbitrary order of that kind.
    """
    if not client_order_id:
        return None
    if isinstance(legs, dict):
        legs = [legs]
    if not isinstance(legs, list):
        return None
    for leg in legs:
        if isinstance(leg, dict) and str(leg.get("order_link_id") or "") == client_order_id:
            return leg
    return None


def _working_quantity(legs: Any) -> float:
    """The total quantity the **working** legs of one kind cover.

    A finished leg protects nothing, so it contributes nothing to the
    coverage. The quantity of a working leg is what it still holds open for
    the position, which is its own quantity less what has already traded --
    a partially filled stop still guards only what is left of it.
    """
    if isinstance(legs, dict):
        legs = [legs]
    if not isinstance(legs, list):
        return 0.0
    total = 0.0
    for leg in legs:
        if not isinstance(leg, dict) or not leg.get("is_active"):
            continue
        quantity = _as_float(leg.get("quantity"))
        filled = _as_float(leg.get("filled_quantity"))
        total += max(0.0, quantity - filled)
    return total


def _as_optional_float(value: Any) -> float | None:
    """Read a numeric field that may be absent, as a float or ``None``.

    Bybit reports an average execution price as ``"0"`` until the order has
    traded, so a zero is "not known yet" rather than "the price was zero".
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == 0.0:
        return None
    return number


def _spot_qty_step(lot_filter: Mapping[str, Any]) -> float:
    """The quantity step for a Spot instrument, from the venue's own fields.

    A Spot ``lotSizeFilter`` has **no** ``qtyStep``. The venue expresses the
    step as ``basePrecision`` -- the precision of the base coin -- and the
    documented Spot example (BTCUSDT: ``basePrecision`` ``0.000001``) shows
    the quantity must be a multiple of it. A non-Spot shape may carry an
    explicit ``qtyStep``, which is used when it is present.

    Reading only ``qtyStep`` would yield a step of zero for every real Spot
    pair, and a zero step silently disables the step check: a quantity on a
    forbidden boundary would be sent and rejected by the venue instead of
    being refused here.
    """
    for field_name in ("qtyStep", "basePrecision"):
        raw = lot_filter.get(field_name)
        if raw is None or raw == "":
            continue
        try:
            step = float(raw)
        except (TypeError, ValueError):
            continue
        if step > 0 and math.isfinite(step):
            return step
    return 0.0


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

    @property
    def status(self) -> str:
        """Compatibility with the shared paper-worker account preflight.

        Bybit's V5 wallet-balance response does not expose an Alpaca-style
        account status. A successful authenticated wallet read is treated as
        ACTIVE unless the adapter has explicitly marked the account blocked.
        """
        return "BLOCKED" if self.trading_blocked or self.account_blocked else "ACTIVE"

    @property
    def cash(self) -> float:
        """USDT wallet balance, matching the Spot account's cash balance."""
        return self.equity

    @property
    def buying_power(self) -> float:
        """USDT available for new Spot orders after locked funds."""
        return self.available_balance

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
            "cash": self.cash,
            "buying_power": self.buying_power,
            "status": self.status,
            "currency": self.currency,
            "trading_blocked": self.trading_blocked,
            "account_blocked": self.account_blocked,
            "restrictions": self.restrictions(),
        }


@dataclass(frozen=True)
class BrokerPosition:
    """One position as the broker reports it.

    For Spot, a "position" is a holding of a non-quote asset.
    - quantity: total wallet balance of the base asset
    - available_quantity: wallet balance minus locked (for open orders)
    - avg_price: average entry price (if known)
    - mark_price: current market price
    - unrealised_pnl: unrealized P&L
    - side: "buy" or "sell" (for Spot, always "buy" for holdings)
    - venue_symbol: Bybit format (e.g., BTCUSDT)
    - protection_orders: dict with stop_loss and take_profit order info
    """

    symbol: str
    quantity: float
    available_quantity: float = 0.0  # wallet balance minus locked
    avg_price: float | None = None
    mark_price: float | None = None
    unrealised_pnl: float | None = None
    side: str | None = None
    venue_symbol: str | None = None
    protection_orders: dict | None = None  # {"stop_loss": {...}, "take_profit": {...}}

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_protected(self) -> bool:
        """Whether this position has verified protection orders from the broker.

        Both legs must be present **and** the broker must have reported them
        as still working. The ``is_active`` flag is the broker's own answer
        mapped into this project's vocabulary, so a leg the venue has not
        confirmed as working never counts as protection.

        A symbol can carry several orders of the same kind (a residual pair
        beside the original one), so the legs are looked up through the shared
        selection helper rather than assumed to be a single row.
        """
        if not self.protection_orders:
            return False
        sl = _select_leg(self.protection_orders.get("stop_loss"))
        tp = _select_leg(self.protection_orders.get("take_profit"))
        # Both must exist and be in active/pending state at the broker
        if not sl or not tp:
            return False
        return bool(sl.get("is_active")) and bool(tp.get("is_active"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "available_quantity": self.available_quantity,
            "avg_price": self.avg_price,
            "mark_price": self.mark_price,
            "unrealised_pnl": self.unrealised_pnl,
            "side": self.side,
            "venue_symbol": self.venue_symbol or self.symbol,
            "protection_orders": self.protection_orders,
            "is_protected": self.is_protected,
        }


@dataclass(frozen=True)
class ProtectionOrder:
    """A verified protection order from the broker."""
    order_id: str
    order_link_id: str
    symbol: str
    order_type: str  # "StopLoss" or "TakeProfit"
    trigger_price: float
    quantity: float
    status: str  # "active", "untriggered", "triggered", "filled", "cancelled", "rejected"
    side: str  # "Buy" or "Sell"
    time_in_force: str
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "order_link_id": self.order_link_id,
            "symbol": self.symbol,
            "order_type": self.order_type,
            "trigger_price": self.trigger_price,
            "quantity": self.quantity,
            "status": self.status,
            "side": self.side,
            "time_in_force": self.time_in_force,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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

    # The signed bytes and the sent bytes must be the same. The query string
    # and the JSON body are therefore built here with the same helpers the
    # signature used, and passed to the client as ready-made text rather than
    # as a mapping the client would re-encode.
    query = _encode_query_string(params) if params else ""
    if query:
        url = f"{url}?{query}"

    request_kwargs: dict[str, Any] = {
        "headers": dict(headers),
        "timeout": timeout,
    }
    if json_body:
        # Send the exact bytes that were signed.
        request_kwargs["data"] = _encode_json_body(json_body).encode("utf-8")
        request_kwargs["headers"] = {**dict(headers), "Content-Type": "application/json"}

    # The query string has already been appended to the URL above. Passing
    # params as well makes requests append it a second time, so Bybit
    # receives e.g. accountType=UNIFIED&accountType=UNIFIED even though the
    # signature covered only one copy. Keep the signed URL as the sole source
    # of query parameters.
    response = requests.request(method, url, **request_kwargs)
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
            # Bybit's rule: the signature covers ``timestamp + api_key +
            # recv_window + payload``, where the payload is the **query string**
            # for a GET and the **JSON body** for a POST.
            #
            # The bytes that are signed must be the bytes that go on the wire.
            # The query string is therefore built here in the same form the
            # transport will send it, and the body is serialised by the same
            # helper the transport uses -- otherwise the venue computes a
            # signature over a different string and refuses every request.
            if json_body:
                payload = _encode_json_body(json_body)
            elif params:
                payload = _encode_query_string(params)
            else:
                payload = ""
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
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            raise BrokerError(
                f"the broker did not answer with JSON for {path}"
            ) from None

        # A V5 answer carries the outcome in ``retCode``, and a business error
        # arrives with **HTTP 200**. Reading only the HTTP status would treat
        # "order does not exist" or "error sign" as a success, so the code is
        # checked here, before any caller can read the answer as an order.
        code = _bybit_error_code(payload)
        if code is not None:
            raise self._business_error(code, payload, path)
        return payload

    def _business_error(self, code: int, payload: dict, path: str) -> BrokerError:
        """Turn a V5 business error into the error the caller should act on.

        The code decides the *kind* of failure, because the kind decides
        whether the request may be retried, must be reconciled, or is simply
        a refusal.
        """
        message = self._redact(str(payload.get("retMsg") or "unknown error"))
        detail = f"Bybit refused {path} (retCode {code}): {message}"

        if code in BYBIT_RATE_LIMIT_CODES:
            return BrokerRateLimitedError(
                f"{detail}; the request was not accepted, so it may be retried after a wait"
            )
        if code in BYBIT_DUPLICATE_ORDER_CODES:
            return DuplicateOrderError(
                f"{detail}; the order may already exist at the venue, so it must be "
                "reconciled by client order id and never submitted again"
            )
        if code in BYBIT_INSUFFICIENT_FUNDS_CODES:
            return InsufficientFundsError(detail)
        if code in BYBIT_AUTH_ERROR_CODES:
            return BrokerRejectedError(f"{detail}; check the demo credential and its signature")
        return BrokerRejectedError(detail)

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
        equity = _as_float(usdt_info.get("walletBalance"))
        # Bybit deprecated availableToWithdraw for UNIFIED accounts and now
        # returns an empty string. For this Spot-only broker, spendable USDT is
        # the wallet balance less funds locked by open Spot orders.
        locked = _as_float(usdt_info.get("locked"))
        available_raw = usdt_info.get("availableToWithdraw")
        available = (
            _as_float(available_raw)
            if available_raw not in (None, "")
            else max(0.0, equity - locked)
        )
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

            # Exchange metadata may contain blank deprecated fields. Parse every
            # numeric field defensively; missing required rules must make the pair
            # non-tradable rather than crashing startup or silently disabling a guard.
            min_order_qty = _as_float(lot_filter.get("minOrderQty"))
            max_order_qty = _as_float(lot_filter.get("maxOrderQty"))
            qty_step = _spot_qty_step(lot_filter)
            min_order_amt = _as_float(lot_filter.get("minOrderAmt"))
            tick_size = _as_float(price_filter.get("tickSize"))
            # ``minOrderQty`` and ``maxOrderQty`` are **deprecated** for Spot:
            # the venue documents that it no longer checks them and checks
            # ``minOrderAmt`` instead. They are read and kept for the audit
            # record, but the bound that is enforced is ``minOrderAmt``.

            # A status of Trading is not enough if the venue omitted the
            # quantity increment or minimum notional: never send an order with
            # unknown sizing rules. Tick size is checked separately for limit orders.
            tradable = status == "Trading" and qty_step > 0 and min_order_amt > 0
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

    def _read_wallet_balances(self, account_type: str = "UNIFIED") -> dict[str, dict]:
        """Read wallet balances for the given account type (SPOT or UNIFIED).

        Returns a dict mapping coin to its balance info:
        - walletBalance: total balance
        - availableToWithdraw: available balance
        - locked: balance locked for open orders
        """
        payload = self._request("GET", "/v5/account/wallet-balance",
                               params={"accountType": account_type}, signed=True)
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise BrokerError("wallet-balance response missing 'result'")
        account_list = result.get("list", [])
        if not account_list or not isinstance(account_list, list):
            raise BrokerError("wallet-balance response missing 'list'")
        account = account_list[0]
        coin_list = account.get("coin", [])

        balances = {}
        for coin in coin_list:
            coin_name = str(coin.get("coin", "")).upper()
            if not coin_name:
                continue
            wallet_balance = _as_float(coin.get("walletBalance"))
            locked = _as_float(coin.get("locked"))
            available_raw = coin.get("availableToWithdraw")
            free_raw = coin.get("free")
            # Both availableToWithdraw (UNIFIED) and free are deprecated
            # and may be returned as "". Never float-convert those blanks.
            available = (
                _as_float(available_raw)
                if available_raw not in (None, "")
                else max(0.0, wallet_balance - locked)
            )
            free = (
                _as_float(free_raw)
                if free_raw not in (None, "")
                else available
            )
            balances[coin_name] = {
                "wallet_balance": wallet_balance,
                "available_to_withdraw": available,
                "locked": locked,
                "free": free,
            }
        return balances

    def _read_open_orders(self) -> dict[str, list[dict]]:
        """Read all open orders, grouped by symbol.

        Returns dict mapping Bybit symbol to list of open order dicts.
        """
        # Use realtime endpoint for open orders
        payload = self._request("GET", "/v5/order/realtime",
                               params={"category": SPOT_CATEGORY, "openOnly": 1}, signed=True)
        result = payload.get("result", {})
        order_list = result.get("list", [])
        if not isinstance(order_list, list):
            return {}

        open_orders: dict[str, list[dict]] = {}
        for entry in order_list:
            if not isinstance(entry, dict):
                continue
            raw_symbol = str(entry.get("symbol", "")).upper()
            if not raw_symbol:
                continue
            if raw_symbol not in open_orders:
                open_orders[raw_symbol] = []
            open_orders[raw_symbol].append(entry)
        return open_orders

    def _read_protection_orders(self) -> dict[str, dict]:
        """Read all conditional (protection) orders: StopLoss and TakeProfit.

        Reads from /v5/order/realtime with pagination support.
        Returns dict mapping Bybit symbol to dict with 'stop_loss' and 'take_profit' keys.
        Each contains the order details from the broker.

        The read is **complete or it is not trusted**: the venue's pagination is
        followed until it stops handing back a cursor, but the walk is bounded.
        If a cursor is still outstanding when the bound is reached, the result
        is marked ``truncated`` and every consumer must treat the protection as
        **unproven**. A partial read that happened not to include the stop
        order would otherwise look exactly like a position with no stop.

        Per Bybit V5 docs: https://bybit-exchange.github.io/docs/v5/order/open-order
        Response includes: orderId, orderLinkId, symbol, side, orderType, qty, timeInForce,
        orderStatus, orderFilter, stopOrderType, triggerPrice, createdTime, updatedTime
        """
        # Read from order/realtime with stop order types, handling pagination
        all_orders: list[dict] = []
        cursor: str | None = None
        max_pages = 10  # Safety limit to avoid infinite loops
        pages_read = 0
        truncated = False

        while pages_read < max_pages:
            params = {"category": SPOT_CATEGORY}
            if cursor:
                params["cursor"] = cursor

            payload = self._request("GET", "/v5/order/realtime", params=params, signed=True)
            result = payload.get("result", {})
            order_list = result.get("list", [])
            if not isinstance(order_list, list):
                truncated = True
                break

            all_orders.extend(order_list)
            pages_read += 1

            # Check for pagination cursor
            next_cursor = result.get("nextPageCursor")
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            # The bound was reached with a cursor still outstanding: the read
            # is incomplete, and an incomplete read cannot prove a protection.
            truncated = bool(cursor)

        protection_orders: dict[str, dict] = {}
        for entry in all_orders:
            if not isinstance(entry, dict):
                continue
            raw_symbol = str(entry.get("symbol", "")).upper()
            if not raw_symbol:
                continue

            # A Spot TP/SL order is reported by the venue with
            # ``orderFilter="tpslOrder"`` and a ``stopOrderType`` of
            # ``StopLoss`` or ``TakeProfit``. ``stopOrderType`` is a *response*
            # field: the create-order endpoint does not accept it.
            order_filter = str(entry.get("orderFilter", ""))
            stop_order_type = str(entry.get("stopOrderType", ""))
            order_status = str(entry.get("orderStatus", ""))
            mapped_status = self._map_status(order_status)

            stop_kind = re.sub(r"[^a-z]", "", stop_order_type.lower())
            is_tpsl = order_filter == "tpslOrder" or stop_kind in (
                "stoploss",
                "takeprofit",
                "partialstoploss",
                "partialtakeprofit",
            )
            if not is_tpsl:
                continue

            if raw_symbol not in protection_orders:
                protection_orders[raw_symbol] = {}

            trigger_price = _as_float(entry.get("triggerPrice"))
            # The venue's own classification is what distinguishes the stop
            # from the target. When it is absent, the order is **not** guessed
            # into a leg: an unclassifiable order proves nothing either way.
            if stop_kind in ("stoploss", "partialstoploss"):
                leg = "stop_loss"
                leg_type = "StopLoss"
            elif stop_kind in ("takeprofit", "partialtakeprofit"):
                leg = "take_profit"
                leg_type = "TakeProfit"
            else:
                continue

            protection_orders[raw_symbol].setdefault(leg, []).append(
                {
                    "order_id": entry.get("orderId"),
                    "order_link_id": entry.get("orderLinkId"),
                    "symbol": raw_symbol,
                    "order_type": leg_type,
                    "trigger_price": trigger_price,
                    "quantity": _as_float(entry.get("qty")),
                    "filled_quantity": _as_float(entry.get("cumExecQty")),
                    "leaves_quantity": _as_float(entry.get("leavesQty")),
                    # ``status`` is the venue's own spelling, lower-cased, so
                    # the record says exactly what Bybit said. ``mapped_status``
                    # is the same fact in this project's vocabulary. Both are
                    # kept: the first is the evidence, the second is what the
                    # project reasons about, and neither is derived from the
                    # other.
                    "status": order_status.lower(),
                    "venue_status": order_status,
                    "mapped_status": mapped_status,
                    "side": entry.get("side"),
                    "time_in_force": entry.get("timeInForce"),
                    "created_at": entry.get("createdTime"),
                    "updated_at": entry.get("updatedTime"),
                    # An order that is still working at the venue: it has not
                    # filled, has not been cancelled and has not been rejected.
                    "is_active": mapped_status in _ACTIVE_PROJECT_STATUSES,
                }
            )
        for symbol_rows in protection_orders.values():
            symbol_rows["truncated"] = truncated
        return protection_orders

    def _fetch_mark_prices(self, coins: list[str]) -> dict[str, float]:
        """Fetch current mark prices for given coins against USDT."""
        if not coins:
            return {}

        # Use tickers endpoint
        symbols = [f"{coin}USDT" for coin in coins]
        try:
            from tradingagents.crypto.bybit import BybitPublicClient
            client = BybitPublicClient()
            tickers = client.tickers(symbols)
            prices = {}
            for coin in coins:
                bybit_symbol = f"{coin}USDT"
                ticker = tickers.get(bybit_symbol, {})
                if isinstance(ticker, dict):
                    last_price = ticker.get("lastPrice") or ticker.get("last")
                    if last_price:
                        prices[coin] = float(last_price)
            return prices
        except Exception:
            return {}

    def positions(self, *, refresh: bool = True) -> dict[str, BrokerPosition]:
        """Read Spot positions from wallet balances, keyed by canonical symbol.

        For Spot, a "position" is any non-USDT asset holding with qty > 0.
        Available quantity = wallet_balance - locked (from open orders).
        Protection orders are verified via /v5/order/realtime.
        """
        if refresh or self._positions is None:
            # Read wallet balances
            wallet_balances = self._read_wallet_balances("UNIFIED")

            # Read open orders to calculate locked balances per coin
            self._read_open_orders()

            # Read protection orders
            protection_orders = self._read_protection_orders()

            # Get mark prices from market data
            mark_prices = self._fetch_mark_prices(list(wallet_balances.keys()))

            found: dict[str, BrokerPosition] = {}
            for coin, balance in wallet_balances.items():
                if coin == "USDT":
                    continue  # Skip quote currency

                wallet_balance = balance.get("wallet_balance", 0.0)
                if wallet_balance <= 0:
                    continue

                # Verify this coin has a valid Spot trading pair on Bybit
                bybit_symbol = f"{coin}USDT"
                instrument = self.get_instrument(bybit_symbol)
                if not instrument or not instrument.tradable:
                    # Skip coins without a valid Spot trading pair (dust, unsupported assets)
                    continue

                # Calculate available quantity (wallet - locked from open orders)
                # The locked field in balance should already reflect this, but we verify
                locked = balance.get("locked", 0.0)
                available = max(0.0, wallet_balance - locked)

                # Get mark price
                mark_price = mark_prices.get(coin)

                # Get protection orders for this symbol
                prot = protection_orders.get(bybit_symbol, {})

                canonical = self._to_canonical_symbol(bybit_symbol)

                pos = BrokerPosition(
                    symbol=canonical,
                    quantity=wallet_balance,
                    available_quantity=available,
                    avg_price=None,  # Would need trade history to calculate
                    mark_price=mark_price,
                    unrealised_pnl=None,  # Would need entry price
                    side="buy",  # Spot holdings are long
                    venue_symbol=bybit_symbol,
                    protection_orders=prot if prot else None,
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
        """Read the venue's orders: the ones still working and the finished ones.

        A V5 order lives in one of two places. ``GET /v5/order/realtime``
        carries the orders that are **still working**, and
        ``GET /v5/order/history`` carries the ones that have **finished**. A
        read of only one of them misses half of the book -- an open order
        that is not in the history, and a filled order that has left the
        realtime list -- so both are read and the two lists are merged.

        The merge is keyed by client order id: an order that appears in both
        (it was working and then finished) is one order, not two.
        """
        params: dict[str, Any] = {"category": SPOT_CATEGORY, "limit": limit}
        if symbol:
            params["symbol"] = self._to_bybit_symbol(symbol.upper())
        if cursor:
            params["cursor"] = cursor

        orders: list[BrokerOrder] = []
        unclassified: list[dict[str, Any]] = []
        seen: set[str] = set()
        pages_read = 0
        next_cursor: str | None = None

        for path in ("/v5/order/realtime", "/v5/order/history"):
            payload = self._request("GET", path, params=params, signed=True)
            result = payload.get("result", {})
            order_list = result.get("list", [])
            if not isinstance(order_list, list):
                raise BrokerError("the broker orders response was not the expected list")
            pages_read += 1

            for entry in order_list:
                if not isinstance(entry, dict):
                    continue
                client_id = str(entry.get("orderLinkId", "")).strip()
                if not client_id:
                    # An order nobody can identify is a finding, not noise:
                    # it is kept so the report can say so.
                    unclassified.append(entry)
                    continue
                if client_id in seen:
                    continue
                seen.add(client_id)
                orders.append(self._parse_order(entry, client_id))

            if path == "/v5/order/history":
                next_cursor = result.get("nextPageCursor")

        return VenueOrderPage(
            orders=orders,
            unclassified=unclassified,
            pages_read=pages_read,
            truncated=bool(next_cursor),
        )

    def _read_terminal_orders(
        self,
        *,
        symbol: str | None = None,
        max_pages: int = 10,
    ) -> tuple[dict[str, dict], bool]:
        """Read the **closed** orders of a symbol, keyed by client order id.

        ``GET /v5/order/realtime`` only carries the orders that are still
        working. An order that has already filled has left that list, so a
        reconciliation that reads only the live orders can never discover that
        a stop loss triggered -- the very event the reconciliation exists to
        notice. The venue's **order history** (``GET /v5/order/history``) is
        the source that carries those finished orders.

        The second element of the returned pair says whether the read was
        **complete**. A truncated read is not evidence of anything: the order
        that filled may simply be on a page that was not read, so a truncated
        read must not be used to claim a leg is still working.
        """
        all_orders: list[dict] = []
        cursor: str | None = None
        pages_read = 0
        truncated = False

        while pages_read < max_pages:
            params: dict[str, Any] = {"category": SPOT_CATEGORY, "limit": 50}
            if symbol:
                params["symbol"] = self._to_bybit_symbol(symbol.upper())
            if cursor:
                params["cursor"] = cursor

            payload = self._request("GET", "/v5/order/history", params=params, signed=True)
            result = payload.get("result", {})
            order_list = result.get("list", [])
            if not isinstance(order_list, list):
                truncated = True
                break

            all_orders.extend(order_list)
            pages_read += 1

            next_cursor = result.get("nextPageCursor")
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            truncated = bool(cursor)

        by_client_id: dict[str, dict] = {}
        for entry in all_orders:
            if not isinstance(entry, dict):
                continue
            client_id = str(entry.get("orderLinkId", "")).strip()
            if not client_id:
                continue
            by_client_id[client_id] = entry
        return by_client_id, truncated

    def _read_order_executions(self, *, symbol: str | None = None, max_pages: int = 10) -> tuple[dict[str, float], bool]:
        """Read the venue's **executions** for a symbol, keyed by client order id.

        ``GET /v5/execution/list`` is the per-trade record: one row per
        execution, with ``execQty`` and ``execPrice``. It is the source that
        says *how much* actually traded, which the order list alone does not
        distinguish from a status string.

        The path is the one the venue documents for Get Trade History; it is
        **not** ``/v5/order/execution``, which is not a route the demo service
        serves.

        The returned quantity is the sum of the executed quantities per order,
        so a partial fill that arrived in two executions is reported as the
        total that traded. The second element says whether the read was
        complete.
        """
        all_rows: list[dict] = []
        cursor: str | None = None
        pages_read = 0
        truncated = False

        while pages_read < max_pages:
            params: dict[str, Any] = {"category": SPOT_CATEGORY, "limit": 100}
            if symbol:
                params["symbol"] = self._to_bybit_symbol(symbol.upper())
            if cursor:
                params["cursor"] = cursor

            payload = self._request("GET", "/v5/execution/list", params=params, signed=True)
            result = payload.get("result", {})
            row_list = result.get("list", [])
            if not isinstance(row_list, list):
                truncated = True
                break

            all_rows.extend(row_list)
            pages_read += 1

            next_cursor = result.get("nextPageCursor")
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            truncated = bool(cursor)

        executed: dict[str, float] = {}
        for row in all_rows:
            if not isinstance(row, dict):
                continue
            client_id = str(row.get("orderLinkId", "")).strip()
            if not client_id:
                continue
            executed[client_id] = executed.get(client_id, 0.0) + _as_float(row.get("execQty"))
        return executed, truncated

    def _read_execution_rows(
        self,
        *,
        symbol: str | None = None,
        max_pages: int = 10,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Read the venue's per-trade execution **rows**, with their ids.

        :meth:`_read_order_executions` returns the summed quantity per order,
        which is what the residual calculation needs. This returns the raw
        rows, because the exactly-once accounting needs the **execution id**:
        the id names one trade, so the same trade is never counted twice no
        matter how many times the list is read.

        The second element says whether the read was complete. A truncated
        read is not evidence: a trade that was not read proves nothing about
        what traded.
        """
        all_rows: list[dict[str, Any]] = []
        cursor: str | None = None
        pages_read = 0
        truncated = False

        while pages_read < max_pages:
            params: dict[str, Any] = {"category": SPOT_CATEGORY, "limit": 100}
            if symbol:
                params["symbol"] = self._to_bybit_symbol(symbol.upper())
            if cursor:
                params["cursor"] = cursor

            payload = self._request("GET", "/v5/execution/list", params=params, signed=True)
            result = payload.get("result", {})
            row_list = result.get("list", [])
            if not isinstance(row_list, list):
                truncated = True
                break

            all_rows.extend(row_list)
            pages_read += 1

            next_cursor = result.get("nextPageCursor")
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            truncated = bool(cursor)

        return all_rows, truncated

    def _parse_order(self, payload: Any, fallback_client_id: str | None = None) -> BrokerOrder:
        if not isinstance(payload, dict):
            raise BrokerError("the broker order response was not the expected object")
        client_id = str(payload.get("orderLinkId", "") or fallback_client_id or "")
        if not client_id:
            raise BrokerError("the broker order response carried no client_order_id")
        raw_status = payload.get("orderStatus")
        symbol = self._to_canonical_symbol(str(payload.get("symbol", "")))
        side = str(payload.get("side", "")).lower()
        qty = _as_float(payload.get("qty"))
        filled_qty = _as_float(payload.get("cumExecQty"))
        filled_avg = payload.get("avgPrice")
        filled_avg_price = _as_optional_float(filled_avg)
        order_type = str(payload.get("orderType", "Market")).lower()
        time_in_force = str(payload.get("timeInForce", "GTC"))
        created_time = payload.get("createdTime")
        submitted_at = None
        if created_time:
            with contextlib.suppress(ValueError, TypeError):
                submitted_at = datetime.fromtimestamp(int(created_time) / 1000, tz=UTC).isoformat()

        status = self._map_status(raw_status)
        # A venue that says "filled" but reports no executed quantity has not
        # told us what traded. Reporting zero filled against a full status
        # would make the order look complete with nothing behind it, so the
        # status is downgraded to reflect what the numbers actually support.
        if status == "filled" and filled_qty <= 0:
            status = "accepted" if qty > 0 else "unknown"
        # A partial fill is not a full fill. If the venue reports a partial
        # status but the executed quantity covers the whole order, the
        # quantity is the stronger statement, and the order is complete.
        if status == "partially_filled" and qty > 0 and filled_qty >= qty:
            status = "filled"

        return BrokerOrder(
            client_order_id=client_id,
            broker_order_id=str(payload.get("orderId")) if payload.get("orderId") else None,
            symbol=symbol,
            side=side,
            quantity=qty,
            filled_quantity=filled_qty,
            filled_avg_price=filled_avg_price,
            status=status,
            order_type=order_type,
            time_in_force=time_in_force,
            submitted_at=submitted_at,
            raw_status=str(raw_status) if raw_status is not None else None,
        )

    def _map_status(self, raw: str | None) -> str:
        """Map Bybit's ``orderStatus`` onto this project's order statuses.

        Bybit spells its statuses in CamelCase (``PartiallyFilled``), while
        this project uses lower snake_case (``partially_filled``). The old
        comparison lower-cased the venue's value and compared it against
        snake_case literals, so ``PartiallyFilled`` -- a status Bybit really
        does return -- never matched and became ``unknown``. A partial fill is
        not a full fill and not a dead order either: it is a live order that
        has already traded part of its quantity, so it maps to
        ``partially_filled``.

        The map is keyed on the case-folded value with the separators removed,
        so ``PartiallyFilled``, ``partially_filled`` and ``PARTIALLYFILLED``
        all land on the same answer.
        """
        if not raw:
            return "unknown"
        key = re.sub(r"[^a-z]", "", str(raw).lower())
        return _BYBIT_STATUS_TO_PROJECT.get(key, "unknown")

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
        """A SELL must not create a short unless shorting is explicitly allowed.

        The quantity that decides this is the one the **venue** holds, not the
        one the worker's last read saw. The cached position set goes stale the
        moment anything trades: after a partial fill the cache still says
        0.1 while the venue holds 0.06, and a sell sized against the cache
        would sell more than the account has. So the coin's balance is read
        from the venue before the decision.

        An **unreadable** balance is not a zero balance and not a permission
        to sell: it is unknown, and an unknown balance cannot be sized
        against. The sell is refused.
        """
        held_quantity = self._held_quantity_for_sell(symbol)

        if quantity <= held_quantity + _FILL_EPSILON:
            return
        account = self._account
        account_allows_short = account is not None and getattr(account, "shorting_enabled", False)
        if not self.limits.allow_short or not account_allows_short:
            raise PaperRiskRejected(
                f"selling {quantity} of {symbol} exceeds the held {held_quantity}; this "
                "would open a short, which is refused (shorting is not enabled for this "
                "account or allow_short is False)"
            )

    def _held_quantity_for_sell(self, symbol: str) -> float:
        """The quantity the venue actually holds of a symbol's base asset.

        The venue's balance is the authority on what is held. The cached
        position set is only consulted when the balance cannot be read, and
        then only as a **lower** bound: a stale cache that says "more" would
        let a sell through that the venue would refuse, and a stale cache
        that says "less" would refuse a sell the venue would accept. The
        balance is preferred for both reasons.

        An unreadable balance raises, because an unknown balance cannot be
        sized against. The caller turns that into a refusal.
        """
        canonical = symbol.upper()
        base_asset = canonical.split("/")[0] if "/" in canonical else canonical
        if not base_asset:
            raise PaperRiskRejected(
                f"symbol {symbol!r} carries no base asset, so the held quantity is unknown"
            )
        try:
            balances = self._read_wallet_balances("SPOT")
        except Exception as exc:  # noqa: BLE001 - an unreadable balance is unknown
            raise PaperRiskRejected(
                f"the {base_asset} balance could not be read ({type(exc).__name__}: {exc}); "
                "the held quantity is unknown, so the sell is refused"
            ) from exc

        balance = balances.get(base_asset)
        if balance is None:
            # The venue carries no entry for the asset, so it holds none of it.
            return 0.0
        held = _as_float(balance.get("wallet_balance"))
        if not math.isfinite(held) or held < 0:
            raise PaperRiskRejected(
                f"the venue's {base_asset} balance is not a usable number; "
                "the held quantity is unknown, so the sell is refused"
            )
        return held

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
        crypto: bool = True,
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
        if not crypto:
            raise PaperRiskRejected(
                "Bybit Demo broker supports crypto spot orders only; non-crypto orders are refused"
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
            if known.get("status") == _LOCAL_REFUSAL_STATUS:
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
            raise PaperRiskRejected(
                f"symbol {symbol} is not tradable on Bybit demo or its required "
                f"quantity/minimum-notional rules are unavailable (status: {instrument.status})"
            )

        if refresh:
            self.refresh_account()
            self.positions(refresh=True)
            if self._state.account_blocked or self._state.trading_blocked:
                raise BrokerError("the broker account is blocked from trading")

        # Normalize to the venue's actual quantity step before risk checks and
        # before persisting/submitting the intent. The worker may calculate six
        # fractional decimals, while each Bybit symbol has its own qtyStep. Always
        # round DOWN (never up), then re-check every venue and project limit against
        # the exact quantity that will be sent.
        quantity_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(instrument.qty_step))
        if not quantity_decimal.is_finite() or quantity_decimal <= 0:
            raise PaperRiskRejected(f"quantity {quantity!r} for {symbol} must be a positive finite number")
        if step_decimal <= 0:
            raise PaperRiskRejected(f"Bybit returned an invalid qtyStep for {symbol}")
        quantity_units = (quantity_decimal / step_decimal).to_integral_value(rounding=ROUND_DOWN)
        normalized_quantity = quantity_units * step_decimal
        if normalized_quantity <= 0:
            raise PaperRiskRejected(
                f"quantity {quantity} for {symbol} is below the minimum usable qtyStep "
                f"{format(step_decimal, 'f')}"
            )
        # Bybit Spot documents minOrderQty/maxOrderQty as deprecated; minOrderAmt
        # is the venue-enforced lower bound, while project risk caps bound the upper.
        quantity = float(normalized_quantity)

        notional = self._check_risk(
            symbol=symbol, quantity=quantity, side=side, reference_price=reference_price
        )

        # Check minimum order amount
        if instrument and notional < instrument.min_order_amt:
            raise PaperRiskRejected(
                f"order notional {notional:.2f} USDT is below minimum {instrument.min_order_amt:.2f} USDT "
                f"for {symbol} (BELOW_MIN_NOTIONAL)"
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
        # For Spot Market orders, use marketUnit="baseCoin" because RiskEngine
        # always passes quantity in base currency (BTC, ETH, etc.), not quote currency.
        # This ensures the actual order value matches the notional approved by RiskEngine.
        if order_type.lower() == "market":
            body["marketUnit"] = "baseCoin"
            # The venue documents that a Market order always uses IOC, so the
            # request says IOC rather than a time in force the venue ignores.
            body["timeInForce"] = "IOC"
        if order_type.lower() == "limit":
            if instrument.tick_size <= 0:
                raise PaperRiskRejected(
                    f"Bybit did not provide a valid tickSize for limit order {symbol}"
                )
            # Limit orders require a price parameter
            # The caller must provide reference_price as the limit price
            if not reference_price or reference_price <= 0:
                raise PaperRiskRejected("limit orders require a valid reference_price as the limit price")
            # Validate tick size
            if instrument and instrument.tick_size > 0:
                price_decimal = Decimal(str(reference_price))
                tick_decimal = Decimal(str(instrument.tick_size))
                if (price_decimal / tick_decimal) % 1 != 0:
                    raise PaperRiskRejected(
                        f"reference_price {reference_price} for {symbol} does not match tickSize "
                        f"{format(Decimal(str(instrument.tick_size)), 'f')}"
                    )
            body["price"] = self._format_price(reference_price, instrument)

        try:
            payload = self._request("POST", "/v5/order/create", json_body=body, signed=True, reconcile_ok=True)
        except _TransportTimeout:
            return self._reconcile(client_order_id)
        except DuplicateOrderError:
            # The venue says this request duplicated one it already saw, so
            # the order may exist. It is **not** a local refusal: marking it
            # one would let the next attempt submit the order again. The
            # intent stays open, so the next attempt reconciles it.
            raise
        except BrokerRejectedError:
            # A refusal the venue made means nothing exists at the broker, so
            # the same intent may be tried again once the cause is gone. The
            # marker is this project's own, not the venue's "rejected".
            self._state.orders[client_order_id]["status"] = _LOCAL_REFUSAL_STATUS
            if self._store:
                self._store.save(self._state)
            raise

        result = payload.get("result", {})
        order_id = result.get("orderId")

        # The venue's own order id is recorded on the intent before the
        # reconcile, so a later read can name the order even if the client id
        # is all that is known.
        if order_id:
            self._state.orders[client_order_id]["broker_order_id"] = str(order_id)
            if self._store:
                self._store.save(self._state)

        # The order is reconciled under the client id **this adapter sent**.
        # The venue echoes its own ``orderLinkId`` back, and that value is not
        # guaranteed to be the one that was sent -- so it is never used as the
        # key. Using the venue's echo would record the order under a
        # different key from the persisted intent, and the intent would stay
        # stuck at "submitting" while the venue held a filled order: exactly
        # the shape of a duplicate on the next attempt.
        try:
            order = self._reconcile(client_order_id)
        except Exception:
            # If reconciliation fails, fall back to parsing the create response.
            order = self._parse_order(
                {"orderId": order_id, "orderLinkId": client_order_id, "symbol": bybit_symbol, **body},
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

    def _format_price(self, price: float, instrument: InstrumentInfo | None = None) -> str:
        """A price as a plain decimal string, never in scientific notation."""
        value = float(price)
        if not math.isfinite(value):
            raise PaperRiskRejected(f"price {price!r} is not a finite number")
        if value == 0.0:
            return "0"
        text = format(Decimal(str(value)), "f")
        # Trim trailing zeros but keep at least the precision of tick_size
        if instrument and instrument.tick_size > 0:
            step_str = format(Decimal(str(instrument.tick_size)), "f")
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
                # "submitting" is the marker for an intent whose outcome the
                # venue has not confirmed yet. It is not one of the project's
                # order statuses, so it is mapped to the project's own
                # "unknown" -- an order whose fate is not known is not
                # reported as accepted.
                status = (
                    "unknown"
                    if known.get("status") == "submitting"
                    else known.get("status", "unknown")
                )
                order = BrokerOrder(
                    client_order_id=known["client_order_id"],
                    broker_order_id=known.get("broker_order_id"),
                    symbol=known["symbol"],
                    side=known["side"],
                    quantity=known["quantity"],
                    filled_quantity=known.get("filled_quantity", 0.0),
                    filled_avg_price=known.get("filled_avg_price"),
                    status=status,
                    order_type=known.get("order_type", "market"),
                    time_in_force=known.get("time_in_force", "GTC"),
                    submitted_at=known.get("submitted_at"),
                )
                # The persisted intent is brought in line with what is now
                # known, so it does not stay stuck at "submitting" and read
                # as an order that is still on its way.
                self._record_order(order)
                return order
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
        """Request the cancellation of an order, by client order id.

        The venue's cancel endpoint is **asynchronous**: its acknowledgement
        says the *request* was accepted, not that the order is cancelled. So
        the order is not reported as ``cancelled`` here -- the order is read
        back from the venue, and the venue's own answer is what is reported.

        Reporting ``cancelled`` from the acknowledgement alone would let a
        cancelled-but-still-working order be counted as gone, which is how a
        protection leg ends up believed to be settled while it is still live.
        """
        if not self._state.confirmed:
            raise BrokerError("the account has not been confirmed; call connect() first")

        bybit_symbol = self._to_bybit_symbol(symbol)
        body = {
            "category": SPOT_CATEGORY,
            "symbol": bybit_symbol,
            "orderLinkId": client_order_id,
        }
        self._request("POST", "/v5/order/cancel", json_body=body, signed=True)
        # The acknowledgement carried no state. Read the venue's answer.
        return self._reconcile(client_order_id)

    def create_protection_orders(
        self,
        *,
        symbol: str,
        position_quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
        client_order_id_prefix: str,
        allow_orders: bool = False,
    ) -> dict[str, BrokerOrder]:
        """Create a StopLoss and a TakeProfit order for a Spot position.

        The request uses **only** parameters the V5 Spot create-order endpoint
        documents. ``POST /v5/order/create`` accepts ``orderFilter`` with the
        values ``Order``, ``tpslOrder`` and ``StopOrder``; it does **not**
        accept ``stopOrderType`` as a request parameter -- that field appears
        only in the *response* and in the enum, where the venue reports how it
        classified the order it created.

        So the two legs are created as standalone Spot TP/SL orders
        (``orderFilter="tpslOrder"``), distinguished by their trigger price,
        and the venue's own ``stopOrderType`` is what classifies them on
        read-back. The legs are therefore **verified against the venue** before
        they are accepted as protection: if the venue cannot confirm that both
        a stop and a target exist for the position, the protection is refused
        rather than assumed.

        Args:
            symbol: Canonical symbol (e.g., BTC/USDT)
            position_quantity: Quantity to protect (must match available position)
            stop_loss_price: Trigger price for StopLoss
            take_profit_price: Trigger price for TakeProfit
            client_order_id_prefix: Prefix for generating orderLinkIds
            allow_orders: Must be True to actually place orders

        Returns:
            Dict with 'stop_loss' and 'take_profit' BrokerOrder objects

        Raises:
            PaperRiskRejected: If validation fails or protection creation fails
        """
        if not allow_orders:
            raise PaperRiskRejected("order submission is disabled: pass allow_orders=True")
        if not self._key or not self._secret:
            raise BrokerNotConfiguredError(f"no Bybit demo credentials: set {API_KEY_ENV} and {API_SECRET_ENV}")
        if not self._state.confirmed:
            raise BrokerError("the account has not been confirmed; call connect() first")

        symbol = symbol.upper()
        bybit_symbol = self._to_bybit_symbol(symbol)

        # Verify position exists and has enough available quantity
        positions = self.positions(refresh=True)
        position = positions.get(symbol)
        if not position:
            raise PaperRiskRejected(f"no position found for {symbol} to protect")
        if position.available_quantity < position_quantity:
            raise PaperRiskRejected(
                f"insufficient available quantity for {symbol}: "
                f"available={position.available_quantity}, requested={position_quantity}"
            )

        instrument = self.get_instrument(bybit_symbol)
        if not instrument:
            raise PaperRiskRejected(f"symbol {symbol} not found in instruments")

        # Validate prices against tick size
        if instrument.tick_size > 0:
            for price, name in [(stop_loss_price, "stop_loss"), (take_profit_price, "take_profit")]:
                price_decimal = Decimal(str(price))
                tick_decimal = Decimal(str(instrument.tick_size))
                if (price_decimal / tick_decimal) % 1 != 0:
                    raise PaperRiskRejected(
                        f"{name}_price {price} for {symbol} does not match tickSize "
                        f"{format(Decimal(str(instrument.tick_size)), 'f')}"
                    )

        # Validate SL < TP for long positions
        if stop_loss_price >= take_profit_price:
            raise PaperRiskRejected(
                f"stop_loss_price {stop_loss_price} must be less than take_profit_price {take_profit_price}"
            )

        # Format prices
        sl_price_str = self._format_price(stop_loss_price, instrument)
        tp_price_str = self._format_price(take_profit_price, instrument)
        qty_str = self._format_quantity(position_quantity, instrument)

        created_orders = {}

        try:
            # Create the StopLoss leg.
            # Per https://bybit-exchange.github.io/docs/v5/order/create-order the
            # documented Spot TP/SL parameters are: category, symbol, side,
            # orderType, qty, timeInForce, orderLinkId, orderFilter=tpslOrder
            # and triggerPrice. No stopOrderType is sent -- the endpoint does
            # not accept one and the venue classifies the order itself.
            sl_client_id = f"{client_order_id_prefix}-sl"
            sl_body = {
                "category": SPOT_CATEGORY,
                "symbol": bybit_symbol,
                "side": "Sell",  # SL for a long position is a sell
                "orderType": "Market",
                "qty": qty_str,
                # The venue documents that a Market order always uses IOC, so
                # IOC is what is sent rather than a contradictory GTC.
                "timeInForce": "IOC",
                "orderLinkId": sl_client_id,
                "orderFilter": "tpslOrder",
                "triggerPrice": sl_price_str,
            }

            sl_payload = self._request("POST", "/v5/order/create", json_body=sl_body, signed=True)
            sl_result = sl_payload.get("result", {})
            sl_order = self._parse_order(
                {"orderId": sl_result.get("orderId"), "orderLinkId": sl_client_id, "symbol": bybit_symbol, **sl_body},
                fallback_client_id=sl_client_id,
            )
            self._record_order(sl_order)
            created_orders["stop_loss"] = sl_order

            # Create the TakeProfit leg.
            tp_client_id = f"{client_order_id_prefix}-tp"
            tp_body = {
                "category": SPOT_CATEGORY,
                "symbol": bybit_symbol,
                "side": "Sell",  # TP for a long position is a sell
                "orderType": "Market",
                "qty": qty_str,
                "timeInForce": "IOC",
                "orderLinkId": tp_client_id,
                "orderFilter": "tpslOrder",
                "triggerPrice": tp_price_str,
            }

            tp_payload = self._request("POST", "/v5/order/create", json_body=tp_body, signed=True)
            tp_result = tp_payload.get("result", {})
            tp_order = self._parse_order(
                {"orderId": tp_result.get("orderId"), "orderLinkId": tp_client_id, "symbol": bybit_symbol, **tp_body},
                fallback_client_id=tp_client_id,
            )
            self._record_order(tp_order)
            created_orders["take_profit"] = tp_order

            # Record the protection order pair for OCO handling
            # We store the link between SL and TP by using the same prefix
            self._state.protection_order_pairs[client_order_id_prefix] = {
                "sl_client_id": sl_client_id,
                "tp_client_id": tp_client_id,
                "symbol": symbol,
                "position_qty": position_quantity,
                "created_at": datetime.now(UTC).isoformat(),
            }
            # The levels the pair covers are kept so that a **partial** fill
            # can be re-protected with the very levels the strategy derived.
            # The broker never invents a level, and a residual may only be
            # guarded by the levels this pair was created from.
            self._state.protection_levels[symbol] = {
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "position_qty": position_quantity,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            if self._store:
                self._store.save(self._state)

        except Exception as e:
            # If either fails, cancel any created orders
            from contextlib import suppress
            for _key, order in created_orders.items():
                with suppress(Exception):
                    self.cancel_order(client_order_id=order.client_order_id, symbol=symbol)
            raise PaperRiskRejected(f"failed to create protection orders: {e}") from None

        return created_orders

    def verify_protection_orders(
        self,
        symbol: str,
        client_order_ids: dict[str, str] | None = None,
    ) -> dict[str, BrokerOrder | None]:
        """Verify protection orders exist and are active at the broker.

        Reads from /v5/order/realtime and checks for StopLoss/TakeProfit orders
        matching the given symbol.

        A **truncated** read verifies nothing: the leg that is missing may
        simply be on a page that was not read, so both legs are reported as
        ``None`` rather than one of them being confirmed off a partial read.

        ``client_order_ids`` names the **specific** legs to confirm, keyed by
        ``"stop_loss"`` and ``"take_profit"``. Without it the read picks the
        best leg of each kind for the symbol, which is right for "is this
        position guarded?" but wrong for "did *these* orders take?": a symbol
        can carry the original pair and a residual pair at once, and the
        original is the larger one. A caller that created a specific pair
        therefore asks about that pair by id, and a leg that is not there is
        reported as absent rather than silently replaced by another order.

        Returns dict with 'stop_loss' and 'take_profit' BrokerOrder objects,
        or None if not found/verified.
        """
        if not self._state.confirmed:
            raise BrokerError("the account has not been confirmed; call connect() first")

        symbol = symbol.upper()
        bybit_symbol = self._to_bybit_symbol(symbol)
        protection_orders = self._read_protection_orders()

        prot = protection_orders.get(bybit_symbol, {})
        if prot.get("truncated"):
            return {"stop_loss": None, "take_profit": None}

        result = {}
        for key in ("stop_loss", "take_profit"):
            if client_order_ids is not None:
                wanted = str(client_order_ids.get(key) or "")
                order_data = _find_leg_by_client_id(prot.get(key), wanted)
            else:
                order_data = _select_leg(prot.get(key))
            # Use is_active flag (based on orderStatus) for verification
            if order_data and order_data.get("is_active", False):
                result[key] = BrokerOrder(
                    client_order_id=order_data.get("order_link_id", ""),
                    broker_order_id=order_data.get("order_id"),
                    symbol=symbol,
                    side="sell",
                    quantity=order_data.get("quantity", 0.0),
                    filled_quantity=order_data.get("filled_quantity", 0.0),
                    filled_avg_price=None,
                    status=order_data.get("mapped_status", "unknown"),
                    order_type=order_data.get("order_type", "").lower(),
                    time_in_force=order_data.get("time_in_force", "GTC"),
                    submitted_at=order_data.get("created_at"),
                )
            else:
                result[key] = None

        return result

    def apply_fill(self, order: BrokerOrder) -> bool:
        """Account for an order's fill exactly once. Returns whether it was new.

        The guard is **not** keyed on the client order id alone. An order that
        traded 0.4 and then traded 0.2 more carries the **same** id, and the
        second report is a new quantity that has to be accounted for. Keying on
        the id alone dropped that second report on the floor: the order was
        marked "already applied" and the extra 0.2 never reached the
        accounting, so the system believed it had sold 0.4 when it had sold
        0.6.

        So the guard is the **cumulative** quantity:

        - the applied cumulative quantity for the order is read from the
          durable state;
        - a report whose cumulative quantity is **not greater** than what was
          applied is the same fill seen again, and is not applied twice;
        - a report whose cumulative quantity **is greater** is new: the
          difference is the quantity that traded since the last report, and
          only that difference is applied.

        The quantity, not the id, is what makes the accounting exactly-once,
        and it survives a restart because the applied cumulative quantity is
        persisted with the rest of the state.

        The venue remains the authority on what the position actually holds.
        The correction for a partially filled **position** is therefore made
        by reading the position back, not by replaying the fill -- but the
        **protection** of a partially filled protection order is a different
        question, and it is answered by
        :meth:`_handle_protection_order_fill`, which works out the residual
        from the venue's own execution record and balance.
        """
        if order.status not in ("filled", "partially_filled"):
            return False

        client_order_id = order.client_order_id
        applied = float(self._state.applied_fill_quantities.get(client_order_id, 0.0))
        reported = float(order.filled_quantity or 0.0)

        if not math.isfinite(reported) or reported < 0:
            # A quantity that is not a usable number cannot be accounted for.
            # Refusing it is safer than applying a NaN, which would poison
            # every later comparison.
            return False

        if reported <= applied + _FILL_EPSILON:
            # The same fill, seen again. Nothing new to account for.
            self._state.applied_fills.add(client_order_id)
            return False

        self._state.applied_fill_quantities[client_order_id] = reported
        self._state.applied_fills.add(client_order_id)
        if self._store:
            self._store.save(self._state)

        # If this was a protection order (SL or TP), check if we need to cancel the counterpart
        self._handle_protection_order_fill(order)
        return True

    def apply_executions(self, executions: list[dict[str, Any]]) -> dict[str, Any]:
        """Account for the venue's per-trade execution records, exactly once.

        The venue's execution list (``/v5/execution/list``) is the per-trade
        record: each row names one **execution id**, one symbol, one side and
        one quantity. It is finer than the order, so it is the unit of
        exactly-once accounting: the same execution id is never applied twice,
        no matter how many times the list is read.

        This is what makes a **growing** fill safe. An order that traded 0.4
        and then 0.2 more is two executions, and each is applied once. Reading
        the order's cumulative quantity instead would have to compare against
        a stored number, which is exactly the comparison that fails when the
        state is lost; the execution ids do not depend on any stored total.

        Returns a summary of what was applied, what was skipped as already
        applied, and what could not be read.
        """
        applied_ids = {
            execution_id
            for ids in self._state.applied_execution_ids.values()
            for execution_id in ids
        }
        newly_applied: list[str] = []
        already_applied: list[str] = []
        skipped: list[str] = []

        for row in executions:
            if not isinstance(row, dict):
                continue
            execution_id = str(row.get("execId") or row.get("executionId") or "").strip()
            if not execution_id:
                # A record with no usable execution id cannot be applied
                # exactly once, so it is skipped rather than guessed at.
                skipped.append(str(row.get("orderLinkId") or ""))
                continue
            if execution_id in applied_ids:
                already_applied.append(execution_id)
                continue
            applied_ids.add(execution_id)
            newly_applied.append(execution_id)
            link_id = str(row.get("orderLinkId") or "").strip()
            if link_id:
                known = self._state.applied_execution_ids.setdefault(link_id, [])
                known.append(execution_id)

        if newly_applied and self._store:
            self._store.save(self._state)

        return {
            "newly_applied": newly_applied,
            "already_applied": already_applied,
            "skipped": skipped,
        }

    def _handle_protection_order_fill(self, order: BrokerOrder) -> None:
        """Handle OCO logic when a protection order (SL/TP) fills.

        If SL or TP fills (fully or partially), cancel the counterpart order.
        This prevents selling more than the position quantity.

        A **partial** fill is not the end of the story. The leg that traded
        sold part of the position, and the part that did not trade is still
        held and still needs a guard. So the counterpart is cancelled (it
        would otherwise sell the whole position a second time) and the
        **residual** is protected afresh.

        The residual is only protected afresh once the counterpart is
        **confirmed finished**. A replacement pair placed while the original
        counterpart is still working would put two live sells against the
        same quantity, and whichever triggered second would sell more than
        the account holds. So the cancellation's outcome decides:

        - ``confirmed_terminal`` -- the venue stated the counterpart is
          finished, so the residual is protected afresh;
        - ``still_active`` -- the venue still shows it working, so no
          replacement is created and the position is reported unprotected
          until the next cycle reconciles it;
        - ``unknown`` -- the counterpart's fate could not be established, so
          no replacement is created and the position is reported unprotected.

        In the last two cases the entry gate closes, because the system
        cannot say what the account would do.
        """
        # Find the protection pair this order belongs to
        for prefix, pair in self._state.protection_order_pairs.items():
            sl_client_id = pair.get("sl_client_id")
            tp_client_id = pair.get("tp_client_id")

            if order.client_order_id == sl_client_id:
                # SL filled, cancel TP
                outcome = self._cancel_counterpart_protection(
                    prefix, tp_client_id, pair.get("symbol"), "TP cancelled due to SL fill"
                )
                if outcome == PROTECTION_CANCEL_CONFIRMED_TERMINAL:
                    self._protect_residual_after_partial_fill(prefix, pair, order)
                else:
                    self._mark_residual_unprotected(
                        prefix,
                        pair,
                        order,
                        reason=(
                            "the counterpart protection order was not confirmed finished "
                            f"({outcome}); no replacement pair was created, so the quantity "
                            "that is still held has no guard"
                        ),
                    )
                break
            elif order.client_order_id == tp_client_id:
                # TP filled, cancel SL
                outcome = self._cancel_counterpart_protection(
                    prefix, sl_client_id, pair.get("symbol"), "SL cancelled due to TP fill"
                )
                if outcome == PROTECTION_CANCEL_CONFIRMED_TERMINAL:
                    self._protect_residual_after_partial_fill(prefix, pair, order)
                else:
                    self._mark_residual_unprotected(
                        prefix,
                        pair,
                        order,
                        reason=(
                            "the counterpart protection order was not confirmed finished "
                            f"({outcome}); no replacement pair was created, so the quantity "
                            "that is still held has no guard"
                        ),
                    )
                break

    def _mark_residual_unprotected(
        self,
        prefix: str,
        pair: dict,
        order: BrokerOrder,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Record that the residual has no guard, and close the entry gate.

        This is the outcome when the protection could **not** be established:
        the counterpart could not be confirmed finished, the quantity could not
        be proven, the replacement could not be created, or the venue did not
        confirm the replacement. In every one of those cases the quantity that
        is still held has no guard, so the position is reported unprotected and
        the worker's entry gate closes.

        The verdict is written to the same durable record the successful case
        uses, so a restart knows why the position is unguarded rather than
        finding an open position with no protection and no explanation.
        """
        symbol = str(pair.get("symbol") or order.symbol or "").upper()
        outcome: dict[str, Any] = {
            "symbol": symbol,
            "protected": False,
            "residual_quantity": None,
            "executed_quantity": None,
            "reason": reason,
            "created": False,
            "verified": False,
        }
        self._state.protection_residuals[prefix] = outcome
        if self._store:
            self._store.save(self._state)
        return outcome

    def _protect_residual_after_partial_fill(
        self,
        prefix: str,
        pair: dict,
        order: BrokerOrder,
    ) -> dict[str, Any]:
        """Protect the quantity a **partial** protection fill left behind.

        A protection order that fills partly has sold part of the position.
        The counterpart is cancelled, so the quantity that is still held has
        **no guard at all**: cancelling the other leg removes a duplicate sell,
        it does not protect what is left. Leaving that silent is the defect
        this method exists to close -- the position would sit there with no
        stop and no target while the system reported the pair as settled.

        So the residual is worked out from two independent facts and then
        guarded:

        - the **executed** quantity comes from the venue's execution history
          (``/v5/execution/list``), which is the per-trade record and the only
          place that distinguishes "sold 0.4 of 1" from "sold all of it";
        - the **residual** comes from the position the venue actually holds
          now, read fresh from the wallet balance.

        The residual is the venue's holding, not ``position_qty - executed``:
        the venue is the authority on what is held, and the two can disagree
        if anything else traded in between.

        Two things are refused rather than guessed at:

        - a **truncated** read of the executions proves nothing about what
          traded, so no new order is placed and the position is left
          unprotected;
        - a coin whose balance could not be read is an **unknown** balance,
          and an unknown balance cannot be sized against. The coin's
          availability is also what decides whether the venue would accept a
          sell of the residual at all, so an uncertain balance is not
          something to place an order against.

        Either way the outcome is the same: the position is reported
        **unprotected**, the entry gate closes, and the next cycle tries
        again. The guard never claims a protection it did not prove at the
        venue.
        """
        symbol = str(pair.get("symbol") or order.symbol or "").upper()
        outcome: dict[str, Any] = {
            "symbol": symbol,
            "protected": False,
            "residual_quantity": None,
            "executed_quantity": None,
            "reason": None,
            "created": False,
            "verified": False,
        }
        if not symbol:
            outcome["reason"] = "the protection pair carries no symbol"
            self._state.protection_residuals[prefix] = outcome
            return outcome

        # -- the quantity that actually traded ----------------------------
        executed_qty, executions_truncated = self._read_order_executions(symbol=symbol)
        if executions_truncated:
            outcome["reason"] = (
                "the execution read was truncated; the traded quantity is unproven, "
                "so no new protection order was placed"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome
        executed = executed_qty.get(order.client_order_id)
        if executed is None:
            outcome["reason"] = (
                f"the venue's execution history carries no record for {order.client_order_id}; "
                "the traded quantity is unproven, so no new protection order was placed"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome
        outcome["executed_quantity"] = executed

        # -- the quantity the venue still holds ----------------------------
        try:
            balances = self._read_wallet_balances("SPOT")
        except Exception as exc:  # noqa: BLE001 - an unreadable balance is unknown
            outcome["reason"] = (
                f"the coin balance could not be read ({type(exc).__name__}: {exc}); "
                "the residual is unknown, so no new protection order was placed"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome

        base_asset = symbol.split("/")[0] if "/" in symbol else symbol
        balance = balances.get(base_asset)
        if balance is None:
            outcome["reason"] = (
                f"the venue's balance carries no {base_asset} entry; the residual is unknown, "
                "so no new protection order was placed"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome

        held = _as_float(balance.get("wallet_balance"))
        if not math.isfinite(held) or held < 0:
            outcome["reason"] = (
                f"the venue's {base_asset} balance is not a usable number; the residual is "
                "unknown, so no new protection order was placed"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome
        outcome["residual_quantity"] = held

        if held <= 0:
            # The position is gone: the leg filled the whole of it, or
            # something else closed it. There is no residual to protect, and
            # nothing is left unguarded.
            outcome["protected"] = True
            outcome["reason"] = (
                "the venue holds no residual position; there is nothing left to protect"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome

        # -- the levels for the residual -----------------------------------
        levels = self._state.protection_levels.get(symbol, {})
        stop_loss = levels.get("stop_loss")
        take_profit = levels.get("take_profit")
        if stop_loss is None or take_profit is None:
            outcome["reason"] = (
                "the residual has no recorded stop loss and take profit, so no "
                "protection order could be created for it"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome

        # -- create and verify a new pair for the residual -----------------
        residual_prefix = f"{prefix}-residual-{_slug(executed)}"
        try:
            created = self.create_protection_orders(
                symbol=symbol,
                position_quantity=held,
                stop_loss_price=float(stop_loss),
                take_profit_price=float(take_profit),
                client_order_id_prefix=residual_prefix,
                allow_orders=True,
            )
        except Exception as exc:  # noqa: BLE001 - a failed creation leaves it unprotected
            outcome["reason"] = (
                f"the protection for the residual could not be created ({type(exc).__name__}: {exc}); "
                "the position is unprotected"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome
        outcome["created"] = True

        # The new pair counts only once the **venue** confirms it. A created
        # order is a request, not a proof. The confirmation asks about these
        # **specific** legs: the symbol may still carry the original pair, and
        # confirming "some stop and some target" would prove nothing about
        # the replacement.
        sl_client_id = str(created["stop_loss"].client_order_id) if created.get("stop_loss") else ""
        tp_client_id = str(created["take_profit"].client_order_id) if created.get("take_profit") else ""
        verified = self.verify_protection_orders(
            symbol,
            client_order_ids={
                "stop_loss": sl_client_id,
                "take_profit": tp_client_id,
            },
        )
        sl_order = verified.get("stop_loss")
        tp_order = verified.get("take_profit")
        if (
            sl_order is None
            or tp_order is None
            or sl_order.client_order_id != sl_client_id
            or tp_order.client_order_id != tp_client_id
        ):
            outcome["reason"] = (
                "the venue did not confirm both legs of the residual protection; "
                "the position is unprotected"
            )
            self._state.protection_residuals[prefix] = outcome
            return outcome

        # The legs have to cover the residual, not merely exist.
        for label, order_obj in (("stop loss", sl_order), ("take profit", tp_order)):
            covered = _as_float(order_obj.quantity)
            if covered + 1e-12 < held:
                outcome["reason"] = (
                    f"the residual {label} covers {covered} of the {held} the venue still holds; "
                    "the position is unprotected"
                )
                self._state.protection_residuals[prefix] = outcome
                return outcome

        outcome["verified"] = True
        outcome["protected"] = True
        outcome["reason"] = f"the residual {held} was re-protected and confirmed at the venue"
        self._state.protection_residuals[prefix] = outcome
        if self._store:
            self._store.save(self._state)
        return outcome


    def _cancel_counterpart_protection(
        self,
        prefix: str,
        client_order_id: str,
        symbol: str,
        reason: str,
    ) -> str:
        """Cancel the counterpart protection order (SL or TP).

        Returns one of three outcomes, and the caller is expected to act on
        it rather than to ignore it:

        - ``confirmed_terminal`` -- the venue **stated** the counterpart is
          finished. Only then may a replacement pair be created;
        - ``still_active`` -- the venue still shows the counterpart working.
          The cancel request was accepted but has not taken effect, so a
          replacement would put two live sells on the same quantity;
        - ``unknown`` -- the counterpart's fate could not be established.

        A cancel request that was **accepted** is not the same fact as a
        cancelled order. The venue accepting the request says only that it
        will try; the confirmation has to come from the venue's own state.
        The counterpart's absence from the live list is not a confirmation
        either: the live list only carries the orders that are still working,
        so a missing order is equally consistent with "cancelled", "never
        read", and "a read that did not reach it". The order **history** is
        the source that carries the finished legs, so it is read too, and the
        pair stays unresolved unless one of the two carries a terminal status.
        """
        if not client_order_id or not symbol:
            # There is no counterpart to account for. That is not a
            # confirmation, so the outcome stays unknown.
            return PROTECTION_CANCEL_UNKNOWN

        # -- ask the venue to cancel it -------------------------------------
        cancel_error: str | None = None
        try:
            # Check if order is still active before cancelling
            order_state = self._state.orders.get(client_order_id)
            if order_state and order_state.get("status") in ("active", "untriggered", "new", "submitting", "accepted"):
                self.cancel_order(client_order_id=client_order_id, symbol=symbol)
        except Exception as exc:  # noqa: BLE001 - an unreadable venue is unknown
            # The cancellation did not happen. The pair stays unresolved, so
            # the reconciliation tries again on the next cycle and the
            # protection is not claimed to be settled.
            cancel_error = f"{type(exc).__name__}: {exc}"

        # -- ask the venue what state it is actually in ----------------------
        bybit_symbol = self._to_bybit_symbol(symbol.upper())
        try:
            live = self._read_protection_orders().get(bybit_symbol, {})
            terminal_orders, history_truncated = self._read_terminal_orders(symbol=symbol)
        except Exception:  # noqa: BLE001 - an unreadable venue is unknown
            return PROTECTION_CANCEL_UNKNOWN

        status: str | None = None
        for leg in ("stop_loss", "take_profit"):
            candidates = live.get(leg)
            if not isinstance(candidates, list):
                candidates = [candidates] if candidates else []
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("order_link_id") == client_order_id:
                    status = candidate.get("mapped_status")
                    break
            if status is not None:
                break
        if status is None and not history_truncated:
            terminal = terminal_orders.get(client_order_id)
            if terminal is not None:
                status = self._map_status(terminal.get("orderStatus"))

        if status is None:
            # Neither the live list nor a **complete** history carries the
            # counterpart. That is unknown, not terminal: the pair stays
            # unresolved and the position is reported unprotected.
            if cancel_error is not None:
                self._state.protection_cancel_outcomes[prefix] = {
                    "client_order_id": client_order_id,
                    "symbol": symbol.upper(),
                    "outcome": PROTECTION_CANCEL_UNKNOWN,
                    "detail": f"the cancellation failed and the counterpart's state could not be read: {cancel_error}",
                }
            return PROTECTION_CANCEL_UNKNOWN
        if status in _ACTIVE_PROJECT_STATUSES:
            # Still working at the venue: the cancel has not taken effect yet.
            self._state.protection_cancel_outcomes[prefix] = {
                "client_order_id": client_order_id,
                "symbol": symbol.upper(),
                "outcome": PROTECTION_CANCEL_STILL_ACTIVE,
                "detail": (
                    f"the venue still shows {client_order_id} as {status}; "
                    "the cancellation has not taken effect yet"
                ),
            }
            return PROTECTION_CANCEL_STILL_ACTIVE
        if status not in _TERMINAL_PROJECT_STATUSES:
            # A status this project does not recognise. Unknown is not
            # terminal, so the pair stays unresolved.
            self._state.protection_cancel_outcomes[prefix] = {
                "client_order_id": client_order_id,
                "symbol": symbol.upper(),
                "outcome": PROTECTION_CANCEL_UNKNOWN,
                "detail": (
                    f"the venue reported {client_order_id} as {status!r}, a status "
                    "this project does not recognise"
                ),
            }
            return PROTECTION_CANCEL_UNKNOWN

        # The venue stated the counterpart is finished.
        self._state.protection_cancel_outcomes[prefix] = {
            "client_order_id": client_order_id,
            "symbol": symbol.upper(),
            "outcome": PROTECTION_CANCEL_CONFIRMED_TERMINAL,
            "detail": f"the venue confirmed {client_order_id} is {status}",
        }
        self._mark_protection_pair_resolved(prefix, reason)
        return PROTECTION_CANCEL_CONFIRMED_TERMINAL

    def reconcile_protection_orders(self, symbol: str) -> dict[str, Any]:
        """Reconcile the protection orders of a symbol with the venue.

        The live order list (``/v5/order/realtime``) only carries the orders
        that are still working, so a stop loss that has already triggered has
        **left** that list. Reading only it can never notice the event the
        reconciliation exists to notice. So the venue's **order history**
        (``/v5/order/history``) is read as well, and it is the source that
        carries the finished legs.

        What the reconciliation does:

        - a leg that the venue reports as filled (or partially filled) has its
          **counterpart** cancelled, because both legs would otherwise sell
          the same position twice;
        - the cancellation is **verified** at the venue before the pair is
          marked resolved. A cancel that failed, or whose outcome could not be
          read, leaves the pair unresolved, so the next cycle tries again
          rather than the pair being quietly forgotten;
        - the executed quantity comes from the venue's **execution** records,
          so a partial fill is recognised as a partial fill and not as a full
          one;
        - a **truncated** read (the venue still had a page outstanding) is
          reported as incomplete, and an incomplete read never confirms that a
          leg is still working.

        Returns a dict describing the legs and the actions taken.
        """
        if not self._state.confirmed:
            return {"error": "not confirmed", "complete": False}

        symbol = symbol.upper()
        bybit_symbol = self._to_bybit_symbol(symbol)

        result: dict[str, Any] = {
            "symbol": symbol,
            "stop_loss": None,
            "take_profit": None,
            "actions_taken": [],
            "errors": [],
            "complete": True,
        }

        live_rows = self._read_protection_orders()
        live = live_rows.get(bybit_symbol, {})
        if live.get("truncated"):
            result["complete"] = False
            result["errors"].append(
                "the live protection order read was truncated; the protection is unproven"
            )

        terminal_orders, history_truncated = self._read_terminal_orders(symbol=symbol)
        if history_truncated:
            result["complete"] = False
            result["errors"].append(
                "the order history read was truncated; a finished leg may not have been seen"
            )

        # Account for the trades, exactly once, before anything is decided
        # from them. The execution id names one trade, so a trade that was
        # already counted is never counted again, however many times this
        # reconciliation runs. A leg that traded 0.4 and then 0.2 more is two
        # trades, and both are counted.
        try:
            execution_rows, executions_truncated = self._read_execution_rows(symbol=symbol)
            accounting = self.apply_executions(execution_rows)
        except Exception as exc:  # noqa: BLE001 - an unreadable record is unknown
            result["complete"] = False
            result["errors"].append(
                f"the execution records could not be accounted for ({type(exc).__name__}: {exc}); "
                "the traded quantity is unproven"
            )
            execution_rows, executions_truncated = [], True
            accounting = {"newly_applied": [], "already_applied": [], "skipped": []}
        result["executions"] = accounting

        executed_qty: dict[str, float] = {}
        for row in execution_rows:
            if not isinstance(row, dict):
                continue
            client_id = str(row.get("orderLinkId", "")).strip()
            if not client_id:
                continue
            executed_qty[client_id] = executed_qty.get(client_id, 0.0) + _as_float(
                row.get("execQty")
            )
        if executions_truncated:
            result["complete"] = False
            result["errors"].append(
                "the execution read was truncated; the traded quantity is unproven"
            )

        def leg_state(client_id: str | None, live_rows: dict[str, dict]) -> dict[str, Any]:
            """The venue's answer about one leg of the pair.

            The live list wins for an order that is still working; the history
            carries the legs that have finished. When neither carries the
            order, the leg is **unknown** -- never assumed to be working.

            The traded quantity comes from the venue's execution records when
            it has them, because they are the per-trade record and the order
            row only carries a running total.
            """
            if not client_id:
                return {"known": False, "status": None, "mapped_status": None, "executed": 0.0}
            live_row = None
            for leg in ("stop_loss", "take_profit"):
                candidates = live_rows.get(leg)
                if not isinstance(candidates, list):
                    candidates = [candidates] if candidates else []
                for candidate in candidates:
                    if isinstance(candidate, dict) and candidate.get("order_link_id") == client_id:
                        live_row = candidate
                        break
                if live_row is not None:
                    break
            if live_row is not None:
                return {
                    "known": True,
                    "status": live_row.get("status"),
                    "mapped_status": live_row.get("mapped_status"),
                    "executed": _as_float(live_row.get("filled_quantity")),
                    "row": live_row,
                }
            terminal = terminal_orders.get(client_id)
            if terminal is not None:
                return {
                    "known": True,
                    "status": terminal.get("orderStatus"),
                    "mapped_status": self._map_status(terminal.get("orderStatus")),
                    "executed": _as_float(terminal.get("cumExecQty")),
                    "row": terminal,
                }
            return {"known": False, "status": None, "mapped_status": None, "executed": 0.0}

        def executed_for(client_id: str, fallback: float) -> float:
            """The quantity the venue says actually traded for an order.

            The venue's **execution records** win over the order row's running
            total. The order row carries a cumulative quantity that the venue
            updates in place, so it cannot say *how much* traded since the
            last read; the execution records are per-trade, so their sum is
            the quantity that actually traded. The order row is the fallback
            for an order whose trades are not in the read.
            """
            if client_id in executed_qty:
                return executed_qty[client_id]
            return fallback

        pairs = [
            (prefix, pair)
            for prefix, pair in sorted(self._state.protection_order_pairs.items())
            if str(pair.get("symbol", "")).upper() == symbol
        ]
        if not pairs:
            # No pair is recorded for this symbol: there is nothing to
            # reconcile against, and the live view is still reported.
            for key, data in (
                ("stop_loss", _select_leg(live.get("stop_loss"))),
                ("take_profit", _select_leg(live.get("take_profit"))),
            ):
                if data:
                    result[key] = {
                        "order_id": data.get("order_id"),
                        "order_link_id": data.get("order_link_id"),
                        "status": data.get("status"),
                        "quantity": data.get("quantity"),
                        "trigger_price": data.get("trigger_price"),
                        "is_active": data.get("is_active", False),
                    }
            return result

        for prefix, pair in pairs:
            sl_client_id = str(pair.get("sl_client_id") or "")
            tp_client_id = str(pair.get("tp_client_id") or "")
            sl_state = leg_state(sl_client_id, live)
            tp_state = leg_state(tp_client_id, live)

            for key, state, client_id in (
                ("stop_loss", sl_state, sl_client_id),
                ("take_profit", tp_state, tp_client_id),
            ):
                if state["known"]:
                    result[key] = {
                        "order_link_id": client_id,
                        "status": state["status"],
                        "mapped_status": state["mapped_status"],
                        "executed_quantity": executed_for(client_id, state["executed"]),
                        "is_active": state["mapped_status"] in _ACTIVE_PROJECT_STATUSES,
                    }
                else:
                    result[key] = None

            if pair.get("resolved"):
                continue

            # A leg that traded (fully or partly) makes its counterpart
            # unwanted: both legs would sell the same position.
            for filled_key, counterpart_key, filled_state, counterpart_id in (
                ("stop_loss", "take_profit", sl_state, tp_client_id),
                ("take_profit", "stop_loss", tp_state, sl_client_id),
            ):
                if not filled_state["known"]:
                    continue
                if filled_state["mapped_status"] not in ("filled", "partially_filled"):
                    continue
                if not counterpart_id:
                    continue
                counterpart_state = leg_state(counterpart_id, live)
                if not counterpart_state["known"]:
                    # The venue does not carry the counterpart in either the
                    # live list or the history. That is **not** proof that it
                    # finished: the read may have been truncated, or the order
                    # may simply be one this project has never seen. The pair
                    # stays unresolved, so the protection is still unproven
                    # and the next cycle reads the venue again.
                    result["errors"].append(
                        f"the {counterpart_key} could not be read after the {filled_key} traded; "
                        "the protection stays unproven"
                    )
                    continue
                if counterpart_state["mapped_status"] in _TERMINAL_PROJECT_STATUSES:
                    # The venue has confirmed the counterpart is finished:
                    # nothing to cancel, and the pair is settled.
                    self._mark_protection_pair_resolved(prefix, f"{counterpart_key} was already {counterpart_state['mapped_status']}")
                    continue
                if counterpart_state["mapped_status"] not in _ACTIVE_PROJECT_STATUSES:
                    # The status is one this project does not recognise. It is
                    # **unknown**, and unknown is not terminal: the order may
                    # still be working, so the pair stays unresolved.
                    result["errors"].append(
                        f"the {counterpart_key} reads as {counterpart_state['mapped_status']!r}, "
                        "which is not a confirmed terminal status; the protection stays unproven"
                    )
                    continue
                # The counterpart is still working, and the position may only
                # be sold once: cancel it.
                try:
                    self.cancel_order(client_order_id=counterpart_id, symbol=symbol)
                except Exception as exc:  # noqa: BLE001 - a failed cancel must not raise
                    # The cancellation did not happen, so the pair is **not**
                    # resolved. The next cycle reads the venue again and tries
                    # again, and until then the protection stays unproven.
                    result["errors"].append(
                        f"failed to cancel the {counterpart_key} after the {filled_key} traded: {exc}"
                    )
                    result["actions_taken"].append(f"{counterpart_key} cancel failed after {filled_key} fill")
                    continue

                # The cancel was accepted. The pair is only resolved once the
                # venue confirms the counterpart is no longer working: an
                # accepted cancel request is not the same fact as a
                # cancelled order. The venue is read again for the answer.
                fresh_live = self._read_protection_orders().get(bybit_symbol, {})
                fresh_terminal, fresh_truncated = self._read_terminal_orders(symbol=symbol)
                if fresh_truncated:
                    result["complete"] = False
                    result["errors"].append(
                        "the order history read was truncated while verifying the cancellation"
                    )
                after_state = leg_state(counterpart_id, fresh_live)
                if not after_state["known"] and counterpart_id in fresh_terminal:
                    after_state = {
                        "known": True,
                        "status": fresh_terminal[counterpart_id].get("orderStatus"),
                        "mapped_status": self._map_status(fresh_terminal[counterpart_id].get("orderStatus")),
                        "executed": _as_float(fresh_terminal[counterpart_id].get("cumExecQty")),
                        "row": fresh_terminal[counterpart_id],
                    }
                if after_state["mapped_status"] in _ACTIVE_PROJECT_STATUSES:
                    result["errors"].append(
                        f"the {counterpart_key} still reads as working after the cancel request; "
                        "the protection stays unproven"
                    )
                    continue
                if not after_state["known"]:
                    result["errors"].append(
                        f"the {counterpart_key} could not be read after the cancel request; "
                        "the protection stays unproven"
                    )
                    continue
                if after_state["mapped_status"] not in _TERMINAL_PROJECT_STATUSES:
                    # The cancel was accepted and the order is no longer in the
                    # live list, but the venue has not stated a status this
                    # project recognises as terminal. It may be a status this
                    # project has never seen, or a read that did not reach the
                    # record. Unknown is not terminal, so the pair stays
                    # unresolved and the protection stays unproven.
                    result["errors"].append(
                        f"the {counterpart_key} reads as {after_state['mapped_status']!r} after the "
                        "cancel request, which is not a confirmed terminal status; "
                        "the protection stays unproven"
                    )
                    continue
                self._mark_protection_pair_resolved(prefix, f"{counterpart_key} cancelled after {filled_key} fill")
                result["actions_taken"].append(_oco_action_message(counterpart_key, filled_key))
                # The result carries the venue's **confirmed** answer for the
                # leg, not the one it had before the cancel was requested.
                result[counterpart_key] = {
                    "order_link_id": counterpart_id,
                    "status": after_state["status"],
                    "mapped_status": after_state["mapped_status"],
                    "executed_quantity": executed_for(counterpart_id, after_state["executed"]),
                    "is_active": after_state["mapped_status"] in _ACTIVE_PROJECT_STATUSES,
                }

        return result

    def _mark_protection_pair_resolved(self, prefix: str, reason: str) -> None:
        """Record that a protection pair no longer needs reconciling."""
        pair = self._state.protection_order_pairs.get(prefix)
        if pair is None:
            return
        pair["resolved"] = True
        pair["resolution_reason"] = reason
        pair["resolved_at"] = datetime.now(UTC).isoformat()
        if self._store:
            self._store.save(self._state)

    def is_position_protected(self, symbol: str) -> tuple[bool, str]:
        """Check if a position is properly protected with both SL and TP.

        Returns ``(is_protected, reason)``. If not protected, the reason
        explains why.

        The check is **fail-closed**: it only answers "protected" when the
        venue has been read completely and has confirmed both legs, live, for
        the quantity actually held. A truncated read, an unreadable position,
        a missing leg, an inactive leg, a quantity that does not cover the
        position, or a leg on the wrong side all answer "not protected".
        """
        if not self._state.confirmed:
            return False, "broker not confirmed"

        symbol = symbol.upper()
        bybit_symbol = self._to_bybit_symbol(symbol)

        # Check position exists
        position = self._positions.get(symbol) if self._positions else None
        if not position or position.quantity <= 0:
            return False, "no open position"

        # A residual whose protection could not be established is the **most
        # specific** finding about this position, so it is reported first.
        # The leg checks below would otherwise answer with a symptom of it --
        # "take_profit missing" -- which does not tell the operator that the
        # quantity left behind by a partial fill is the unguarded part.
        residual = self._unprotected_residual_for(symbol)
        if residual is not None:
            return False, residual

        protection_orders = self._read_protection_orders()
        prot = protection_orders.get(bybit_symbol, {})

        # A truncated read is not evidence: the leg that is missing may simply
        # be on a page that was not read, so nothing can be confirmed.
        if prot.get("truncated"):
            return False, "the protection order read was truncated; the protection is unproven"

        sl_legs = prot.get("stop_loss")
        tp_legs = prot.get("take_profit")

        position_qty = position.quantity

        # Both SL and TP must exist and be active
        if not sl_legs:
            return False, "stop_loss missing"
        if not tp_legs:
            return False, "take_profit missing"

        sl_data = _select_leg(sl_legs)
        tp_data = _select_leg(tp_legs)
        if sl_data is None:
            return False, "stop_loss missing"
        if tp_data is None:
            return False, "take_profit missing"

        if not sl_data.get("is_active", False):
            return False, f"stop_loss not active (status: {sl_data.get('status')})"
        if not tp_data.get("is_active", False):
            return False, f"take_profit not active (status: {tp_data.get('status')})"

        # The legs have to cover the position **as it is now**. After a
        # partial fill the venue holds less than the pair was created for, and
        # the residual pair covers that smaller amount; the coverage is
        # therefore the sum of every **working** leg of that kind, not the
        # quantity of whichever row happened to come last.
        sl_qty = _working_quantity(sl_legs)
        tp_qty = _working_quantity(tp_legs)

        # Both should cover the full position (Bybit TP/SL are full-position by default)
        # Allow 1% tolerance for rounding
        if abs(sl_qty - position_qty) > position_qty * 0.01:
            return False, f"stop_loss quantity mismatch: {sl_qty} vs position {position_qty}"
        if abs(tp_qty - position_qty) > position_qty * 0.01:
            return False, f"take_profit quantity mismatch: {tp_qty} vs position {position_qty}"

        # Verify sides are correct (both should be Sell for long position)
        if sl_data.get("side") != "Sell" or tp_data.get("side") != "Sell":
            return False, "protection order side mismatch (expected Sell for long)"

        return True, "protected"

    def _unprotected_residual_for(self, symbol: str) -> str | None:
        """The reason a symbol's residual is unprotected, if it is.

        A partial fill leaves the residual recorded in the broker's state. If
        that residual could not be re-protected, the position is **not**
        protected, even when the venue still shows the original pair. The
        original pair covered the quantity that was sold; the quantity that
        is still held is what the residual is about.

        Returns the reason, or ``None`` when every residual for the symbol
        was either re-protected or did not leave anything behind.
        """
        for _prefix, outcome in sorted(self._state.protection_residuals.items()):
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("symbol") or "").upper() != symbol.upper():
                continue
            if outcome.get("protected"):
                continue
            reason = outcome.get("reason") or "the residual after the partial fill is unprotected"
            return f"the residual after a partial fill is unprotected: {reason}"
        return None

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
    "ProtectionOrder",
    "InstrumentInfo",
    "VenueOrderPage",
    "BybitBrokerStatus",
    "assert_demo_endpoint",
    "assert_demo_request_url",
    "BybitDemoBroker",
]
