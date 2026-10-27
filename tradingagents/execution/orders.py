"""Orders and their lifecycle.

An order is a *request*, and the whole point of modelling it separately from a
fill is that most orders do not fill exactly as asked, and some do not fill at
all. A system that jumps straight from signal to position cannot express
"rejected for insufficient cash", "the limit was never reached", or "the market
was closed" — and those are the outcomes a backtest must be able to report
honestly.

The lifecycle is a small state machine with one terminal state per path:

    CREATED ──► REJECTED            (never accepted: bad size, no cash, closed market)
        │
        └──► ACCEPTED ──► FILLED    (fully executed)
                  │
                  ├──► PARTIALLY_FILLED ──► FILLED
                  └──► CANCELLED    (limit not reached by the end of its life)

``CREATED`` is transient: an order is built and immediately evaluated. It is in
the enum anyway, because the event log records the request before it is judged,
and a log that starts at ACCEPTED cannot show what was asked for.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderType(StrEnum):
    """How the order asks to be filled.

    Only two, and both are simulation-only. There is no stop, stop-limit or
    trailing variant because those need a price feed between bars to trigger
    honestly, which a daily-bar simulation cannot provide.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(StrEnum):
    """How long an order may rest before it is cancelled.

    ``DAY`` is the original behaviour: the order is worked once, on the first
    eligible bar, and cancelled if that bar does not fill it. ``GTC`` (good-till-
    cancelled) lets a resting order remain active across eligible bars until it
    fills or reaches an explicit expiration session.

    ``GTC`` here means "rests until filled or expired", not the venue-level
    guarantee of queue priority. A daily OHLC bar cannot prove the order sat at
    the front of a queue, so a fill on a bar whose range covers the limit is a
    modelled fill, not a claim about the real book.
    """

    DAY = "DAY"
    GTC = "GTC"


class OrderSide(StrEnum):
    """Which way the order moves the book. Long-only, so no short variants."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Where an order is in its life."""

    CREATED = "CREATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.REJECTED, OrderStatus.FILLED, OrderStatus.CANCELLED)


class RejectReason(StrEnum):
    """Why an order was refused. A closed vocabulary, so a rejection is analysable."""

    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_POSITION = "INSUFFICIENT_POSITION"
    MARKET_CLOSED = "MARKET_CLOSED"
    NO_PRICE = "NO_PRICE"
    RISK_LIMIT = "RISK_LIMIT"
    LIMIT_NOT_REACHED = "LIMIT_NOT_REACHED"
    EXPIRED = "EXPIRED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class Order(BaseModel):
    """A single order: what to trade, how much, and at what price limit.

    ``quantity`` is always positive — the direction lives in ``side``. That is
    not cosmetic: it removes the "negative quantity means sell" convention that
    made an accidental short a sign error away.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    order_id: str = Field(description="Stable id, derived from the signal and the order's index")
    symbol: str = Field(description="Instrument to trade")
    side: OrderSide = Field(description="BUY or SELL")
    quantity: float = Field(gt=0.0, description="Units to trade, always positive")
    order_type: OrderType = Field(default=OrderType.MARKET, description="MARKET or LIMIT")
    limit_price: float | None = Field(
        default=None, gt=0.0, description="Limit price; required for a LIMIT order, refused for a MARKET one"
    )
    created_at: datetime = Field(description="When the order was created (UTC-aware)")
    as_of_date: str = Field(description="The analysis date this order came from, YYYY-MM-DD")
    signal_id: str = Field(default="", description="The signal that produced it, for the audit trail")
    time_in_force: TimeInForce = Field(
        default=TimeInForce.DAY,
        description=(
            "How long the order may rest. DAY is worked once and cancelled if "
            "unfilled; GTC rests across eligible bars until filled or expired."
        ),
    )
    expires_session: str | None = Field(
        default=None,
        description=(
            "The last session (YYYY-MM-DD) the order may be worked on. None means "
            "no explicit expiry. Only meaningful for a GTC order; a DAY order is "
            "worked once regardless."
        ),
    )

    status: OrderStatus = Field(default=OrderStatus.CREATED, description="Current lifecycle state")
    filled_quantity: float = Field(default=0.0, ge=0.0, description="Units filled so far")
    average_fill_price: float | None = Field(default=None, description="Average price of the filled units")
    reject_reason: RejectReason | None = Field(default=None, description="Set when the order is rejected")
    reject_detail: str = Field(default="", description="Human-readable rejection explanation")
    created_recorded: bool = Field(
        default=False,
        description=(
            "Internal bookkeeping: whether an ORDER_CREATED event has already been "
            "recorded for this order. A caller that records creation itself (the "
            "backtester, which creates an order on the decision session and executes "
            "it on the next) sets this so the simulator does not record a second one."
        ),
    )
    accepted_recorded: bool = Field(
        default=False,
        description=(
            "Internal bookkeeping: whether an ORDER_ACCEPTED event has already been "
            "recorded. A resting order is worked on several bars, so this keeps the "
            "acceptance a once-only event instead of one per attempt."
        ),
    )

    @model_validator(mode="after")
    def _limit_price_matches_the_type(self) -> Order:
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("a LIMIT order needs a limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("a MARKET order must not carry a limit_price")
        if self.order_type is OrderType.MARKET and self.time_in_force is not TimeInForce.DAY:
            raise ValueError("a MARKET order must be DAY; it either fills on its bar or expires")
        if self.expires_session is not None and self.expires_session < self.as_of_date:
            raise ValueError("expires_session cannot be before as_of_date")
        if self.filled_quantity > self.quantity + 1e-9:
            raise ValueError("filled_quantity cannot exceed quantity")
        return self

    @property
    def remaining_quantity(self) -> float:
        """Units still to fill."""
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def notional(self) -> float:
        """Value of the filled units at their average fill price; 0 when unfilled."""
        if self.average_fill_price is None:
            return 0.0
        return self.filled_quantity * self.average_fill_price

    def is_open(self) -> bool:
        """Whether the order can still be worked."""
        return not self.status.is_terminal and self.status is not OrderStatus.CREATED
