"""The validated trading signal: the one hand-off from the AI brain to the engine.

Everything upstream of this module is reasoning. Everything downstream is
arithmetic. The signal is the boundary, and it is deliberately narrow: three
actions, a direction, optional levels, and a requested size that the risk engine
is free to ignore.

Why the boundary exists at all. The Portfolio Manager returns a five-tier rating
("Overweight") inside free text; the Trader returns a three-tier action plus a
prose sizing hint ("5% of portfolio"). Neither is a value an execution engine can
act on, and treating prose as an instruction is how an LLM ends up placing a
trade. So the rating is mapped to a fixed action here, deterministically, and
everything that carries a number is validated before it leaves this module.

Long-only by construction. There is no field, and no combination of fields, that
can express a short: the action set has no SHORT, ``position_size`` is a
non-negative fraction of equity, and quantities do not appear at all. A negative
quantity is not rejected downstream because it cannot be constructed here.

Sizing is not the model's decision. ``position_size`` is a *request* in the unit
the model can reason about (fraction of equity), never a quantity, and it is
optional. When it is absent the risk engine sizes the order from its own
deterministic rules. A prose sizing hint is never parsed into it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class SignalAction(StrEnum):
    """The only three actions an engine will ever be handed.

    A closed set, so an unknown action is a validation failure rather than a
    string that reaches the order engine and means something unintended.
    """

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class OrderIntent(StrEnum):
    """How the signal wants its order priced.

    A closed set, and it lives here rather than in the execution package so the
    signal layer does not import the engine it feeds. ``MARKET`` is the default
    and the original behaviour; ``LIMIT`` asks for a resting order at an explicit
    price. An unknown value is a validation failure, not a string the engine has
    to interpret.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class SignalTimeInForce(StrEnum):
    """How long a LIMIT signal's order may stay live.

    ``DAY`` is worked once and cancelled if unfilled; ``GTC`` rests across
    eligible bars until it fills or reaches its expiration. The engine maps these
    onto the execution layer's own vocabulary; keeping a separate enum here means
    the contract can be validated without importing the execution package, which
    would be a cycle (the paper session imports this module).
    """

    DAY = "DAY"
    GTC = "GTC"


class SignalError(ValueError):
    """A signal could not be built from the decision it was derived from."""


# The five-tier rating vocabulary, mapped to the engine's three actions. The
# mapping is many-to-one on purpose: Overweight and Underweight are the model's
# way of expressing conviction, which becomes confidence and size, not a
# separate action.
_RATING_TO_ACTION: dict[str, SignalAction] = {
    "Buy": SignalAction.BUY,
    "Overweight": SignalAction.BUY,
    "Hold": SignalAction.HOLD,
    "Underweight": SignalAction.SELL,
    "Sell": SignalAction.SELL,
}

# Deterministic confidence per rating tier, used when the caller does not supply
# one. Conviction tiers sit between the extremes so a strong call and a mild
# call of the same direction are distinguishable downstream, and the risk engine
# can size them differently without reading any model text.
_RATING_CONFIDENCE: dict[str, float] = {
    "Buy": 0.80,
    "Overweight": 0.65,
    "Hold": 0.50,
    "Underweight": 0.65,
    "Sell": 0.80,
}


def action_for_rating(rating: str | None) -> SignalAction | None:
    """Map a five-tier rating to an action, or ``None`` when there is no call.

    ``None`` (and the ``REVIEW`` sentinel, which never appears in the map) mean
    the decision could not be read as a position. Returning ``None`` rather than
    defaulting to HOLD keeps "nobody decided" distinct from "decided to do
    nothing", the same distinction ``agents.rating`` makes.
    """
    return _RATING_TO_ACTION.get((rating or "").strip())


def confidence_for_rating(rating: str | None) -> float:
    """Deterministic confidence for a rating tier; 0.0 when there is no call."""
    return _RATING_CONFIDENCE.get((rating or "").strip(), 0.0)


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------


class TradingSignal(BaseModel):
    """A validated, engine-facing trade proposal.

    Constructed only through :func:`signal_from_decision` or by passing already
    validated values; the validators below reject anything an engine could not
    act on safely.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: str = Field(description="Instrument the signal is for, e.g. AAPL or BTC-USD")
    action: SignalAction = Field(description="One of BUY / HOLD / SELL")
    confidence: float = Field(ge=0.0, le=1.0, description="Conviction in the action, 0..1")

    timestamp: datetime = Field(
        description="When the decision was taken (UTC-aware). The order timestamp follows."
    )
    as_of_date: str = Field(
        description=(
            "The analysis date (YYYY-MM-DD) the decision's data was clamped to. "
            "Distinct from ``timestamp``: this is the point-in-time cutoff, and it "
            "is what a backtest must schedule execution against."
        )
    )

    entry_price: float | None = Field(
        default=None, gt=0.0, description="Absolute reference price, or None to price at execution"
    )
    stop_loss: float | None = Field(default=None, gt=0.0, description="Absolute stop level, or None")
    take_profit: float | None = Field(default=None, gt=0.0, description="Absolute target, or None")

    position_size: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Requested fraction of portfolio equity to commit, 0 < size <= 1. A "
            "request, not an instruction: the risk engine clamps it. None lets the "
            "risk engine size from its own rules."
        ),
    )
    holding_period_days: int | None = Field(
        default=None, ge=1, description="Intended holding period in trading days, or None"
    )

    order_intent: OrderIntent = Field(
        default=OrderIntent.MARKET,
        description=(
            "How to price the order: MARKET fills at the bar price; LIMIT rests at "
            "``limit_price`` until it fills or expires. The risk engine rules on "
            "the trade either way; this only changes the price the order asks for."
        ),
    )
    limit_price: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Absolute limit price for a LIMIT intent. Required for LIMIT, refused "
            "for MARKET, and must sit on the correct side of the entry for the "
            "action, so a limit that could never fill is a construction error."
        ),
    )
    time_in_force: SignalTimeInForce = Field(
        default=SignalTimeInForce.DAY,
        description="DAY works the limit once; GTC lets it rest until filled or expired.",
    )
    expires_session: str | None = Field(
        default=None,
        description="The last session (YYYY-MM-DD) a GTC limit may be worked on, or None.",
    )

    rating: str | None = Field(
        default=None, description="The originating five-tier rating, kept for traceability"
    )
    source: str = Field(
        default="portfolio_manager",
        description="Which agent or process produced the underlying decision, for the audit trail",
    )
    reasoning: str = Field(
        default="",
        description="The decision's own text. Carried for the report only; never parsed for numbers.",
    )

    # --- Validators ---------------------------------------------------------

    @field_validator("symbol")
    @classmethod
    def _symbol_is_a_usable_token(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("symbol must not be empty")
        if len(cleaned) > 32:
            raise ValueError(f"symbol is longer than 32 characters: {value!r}")
        return cleaned

    @field_validator("as_of_date")
    @classmethod
    def _as_of_date_is_canonical(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(str(value), "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {value!r}") from exc
        if parsed.strftime("%Y-%m-%d") != str(value):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {value!r}")
        return str(value)

    @field_validator("expires_session")
    @classmethod
    def _expires_session_is_canonical(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.strptime(str(value), "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expires_session must be YYYY-MM-DD, got {value!r}") from exc
        if parsed.strftime("%Y-%m-%d") != str(value):
            raise ValueError(f"expires_session must be YYYY-MM-DD, got {value!r}")
        return str(value)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_utc_aware(cls, value: datetime) -> datetime:
        """A naive timestamp is ambiguous, and an ambiguous one orders wrongly."""
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware; pass a UTC datetime")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _levels_and_size_agree_with_the_action(self) -> TradingSignal:
        """Reject a signal whose levels or size contradict its action.

        A BUY whose stop sits above its entry, a HOLD carrying a stop, or a HOLD
        requesting size are all self-contradictory, and letting them through
        would put the contradiction in front of the risk engine, which would
        then have to guess which half of the signal to believe.
        """
        entry, stop, target = self.entry_price, self.stop_loss, self.take_profit

        if self.action is SignalAction.HOLD:
            if stop is not None or target is not None:
                raise ValueError("a HOLD must not carry stop_loss or take_profit levels")
            if self.position_size is not None:
                raise ValueError("a HOLD must not request a position_size")
            if self.order_intent is OrderIntent.LIMIT:
                raise ValueError("a HOLD must not request a LIMIT order")
            return self

        if self.action is SignalAction.BUY:
            if entry is not None and stop is not None and stop >= entry:
                raise ValueError(f"a BUY stop_loss ({stop}) must be below its entry_price ({entry})")
            if entry is not None and target is not None and target <= entry:
                raise ValueError(f"a BUY take_profit ({target}) must be above its entry_price ({entry})")
        else:  # SELL
            if entry is not None and stop is not None and stop <= entry:
                raise ValueError(f"a SELL stop_loss ({stop}) must be above its entry_price ({entry})")
            if entry is not None and target is not None and target >= entry:
                raise ValueError(f"a SELL take_profit ({target}) must be below its entry_price ({entry})")

        if self.order_intent is OrderIntent.LIMIT:
            if self.limit_price is None:
                raise ValueError("a LIMIT intent needs a limit_price")
            # A limit on the wrong side of the market can never fill, so it is a
            # construction error rather than a trade that silently never happens.
            # A BUY limit must be at or below the entry it is willing to pay; a
            # SELL limit at or above the price it will accept.
            if entry is not None:
                if self.action is SignalAction.BUY and self.limit_price > entry:
                    raise ValueError(
                        f"a BUY limit_price ({self.limit_price}) above the entry_price "
                        f"({entry}) can never fill"
                    )
                if self.action is SignalAction.SELL and self.limit_price < entry:
                    raise ValueError(
                        f"a SELL limit_price ({self.limit_price}) below the entry_price "
                        f"({entry}) can never fill"
                    )
        elif self.limit_price is not None:
            raise ValueError("a MARKET intent must not carry a limit_price")

        if self.time_in_force is SignalTimeInForce.GTC and self.order_intent is not OrderIntent.LIMIT:
            raise ValueError("only a LIMIT order can be GTC; a MARKET order is always DAY")
        return self

    # --- Derived ------------------------------------------------------------

    @property
    def is_actionable(self) -> bool:
        """Whether this signal asks the engine to trade at all."""
        return self.action is not SignalAction.HOLD

    @property
    def signal_id(self) -> str:
        """Stable digest of the signal's decision content.

        Deterministic, so replaying the same decision yields the same id and the
        event log can be deduplicated. ``timestamp`` participates at second
        resolution: two runs of the same date must produce the same id even
        though their wall clocks differ.
        """
        payload = {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 6),
            "timestamp": self.timestamp.replace(microsecond=0).isoformat(),
            "as_of_date": self.as_of_date,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "holding_period_days": self.holding_period_days,
            "order_intent": self.order_intent.value,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force.value,
            "expires_session": self.expires_session,
            "rating": self.rating,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:16]


# ---------------------------------------------------------------------------
# Building a signal from the AI brain's output
# ---------------------------------------------------------------------------


def signal_from_decision(
    *,
    symbol: str,
    rating: str | None,
    as_of_date: str,
    timestamp: datetime | None = None,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    position_size: float | None = None,
    holding_period_days: int | None = None,
    confidence: float | None = None,
    order_intent: OrderIntent = OrderIntent.MARKET,
    limit_price: float | None = None,
    time_in_force: SignalTimeInForce = SignalTimeInForce.DAY,
    expires_session: str | None = None,
    source: str = "portfolio_manager",
    reasoning: str = "",
) -> TradingSignal | None:
    """Build a validated signal from a run's rating, or ``None`` when there is no call.

    Returns ``None`` for an unreadable rating (``REVIEW``) or an unknown one,
    because a decision nobody can read is not a trade. Every other argument is a
    value the caller has already resolved: nothing is parsed out of ``reasoning``,
    and no size is inferred from prose.

    ``timestamp`` defaults to the current UTC instant; a backtest passes the
    analysis date's close so the signal's clock matches its data.
    """
    action = action_for_rating(rating)
    if action is None:
        return None
    return TradingSignal(
        symbol=symbol,
        action=action,
        confidence=confidence if confidence is not None else confidence_for_rating(rating),
        timestamp=timestamp or datetime.now(UTC),
        as_of_date=as_of_date,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        holding_period_days=holding_period_days,
        order_intent=order_intent,
        limit_price=limit_price,
        time_in_force=time_in_force,
        expires_session=expires_session,
        rating=rating,
        source=source,
        reasoning=reasoning,
    )
