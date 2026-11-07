"""Per-instrument trading rules, read from the venue and enforced before an order.

A crypto exchange does not accept an arbitrary quantity or price. Each instrument
carries its own *lot size* (a quantity step and a minimum), a *tick size* (a price
step) and a *minimum notional*; and it may be halted, delisted or restricted to
certain order types. A system that assumes "every coin is tradeable, at any size"
will have orders silently rejected — or, worse, sized wrong — so this module reads
the venue's own rules and answers the questions an order needs answered.

The rules are read from **Binance's** ``/api/v3/exchangeInfo`` because it is the
free, unauthenticated, machine-readable source that names every filter explicitly
(``LOT_SIZE``, ``PRICE_FILTER``, ``NOTIONAL``, ``orderTypes``). Nothing here is
hard-coded per coin: a new listing is a new entry in the venue's answer, and the
registry picks it up. The parsed object is *data*, so it can be cached, persisted
and reloaded without a second fetch.

Two things this module deliberately does **not** do:

- it does not invent a rule for an instrument the venue did not describe — an
  unknown symbol is ``None``, never a permissive default;
- it does not round silently. :func:`quantize_quantity` truncates to the step
  (rounding *up* would make an order larger than intended), and
  :func:`validate_order` returns the reasons an order would be refused, so a
  caller can refuse it first and say why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

from .symbols import CryptoPair, parse_pair

# Binance's own order-status vocabulary, mapped to a small set this code reasons
# about. ``TRADING`` is the only status an order may be placed in.
STATUS_TRADING = "TRADING"
STATUS_HALTED = "HALTED"
STATUS_BREAK = "BREAK"
STATUS_UNKNOWN = "UNKNOWN"

# Why an order would be refused, as a closed vocabulary. A caller matches on the
# code, never on prose.
REASON_NOT_TRADABLE = "NOT_TRADABLE"
REASON_BELOW_MIN_QTY = "BELOW_MIN_QTY"
REASON_ABOVE_MAX_QTY = "ABOVE_MAX_QTY"
REASON_QUANTITY_STEP = "QUANTITY_STEP"
REASON_BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
REASON_ABOVE_MAX_NOTIONAL = "ABOVE_MAX_NOTIONAL"
REASON_PRICE_STEP = "PRICE_STEP"
REASON_BELOW_MIN_PRICE = "BELOW_MIN_PRICE"
REASON_ABOVE_MAX_PRICE = "ABOVE_MAX_PRICE"
REASON_UNSUPPORTED_ORDER_TYPE = "UNSUPPORTED_ORDER_TYPE"
REASON_INVALID_QUANTITY = "INVALID_QUANTITY"
REASON_INVALID_PRICE = "INVALID_PRICE"

ORDER_REASONS: tuple[str, ...] = (
    REASON_NOT_TRADABLE,
    REASON_BELOW_MIN_QTY,
    REASON_ABOVE_MAX_QTY,
    REASON_QUANTITY_STEP,
    REASON_BELOW_MIN_NOTIONAL,
    REASON_ABOVE_MAX_NOTIONAL,
    REASON_PRICE_STEP,
    REASON_BELOW_MIN_PRICE,
    REASON_ABOVE_MAX_PRICE,
    REASON_UNSUPPORTED_ORDER_TYPE,
    REASON_INVALID_QUANTITY,
    REASON_INVALID_PRICE,
)


class InstrumentError(ValueError):
    """An instrument could not be described or an order could not be validated."""


def _decimals(step: str) -> int:
    """The number of decimal places a step string implies (``0.00001000`` -> 5)."""
    text = str(step).strip().lower()
    if "e" in text:
        # A scientific-notation step (unlikely from a venue, possible from a test).
        return max(-Decimal(text).as_tuple().exponent, 0)
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1].rstrip("0")) if text.split(".", 1)[1].strip("0") else 0


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class TradingRules:
    """One instrument's tradability and the size/price band an order must fit.

    Every bound is the venue's own number. ``None`` means the venue did not state
    a bound, which is different from a bound of zero: an unstated bound is not
    enforced, a stated one always is.
    """

    symbol: str
    """The canonical ``BASE/QUOTE`` symbol these rules describe."""
    base_asset: str
    quote_asset: str
    status: str = STATUS_UNKNOWN
    min_quantity: float | None = None
    max_quantity: float | None = None
    quantity_step: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    price_tick: float | None = None
    min_notional: float | None = None
    max_notional: float | None = None
    order_types: tuple[str, ...] = ()
    quote_precision: int = 0
    """Decimal places a price may carry (derived from the price tick)."""
    base_precision: int = 0
    """Decimal places a quantity may carry (derived from the quantity step)."""

    @property
    def tradable(self) -> bool:
        return self.status == STATUS_TRADING

    @property
    def pair(self) -> CryptoPair:
        return parse_pair(self.symbol)

    def quantize_quantity(self, quantity: float) -> float:
        """Truncate ``quantity`` to the venue's step, toward zero.

        Truncation, not rounding: rounding half-up could make the order *larger*
        than the caller asked for, and an order that is larger than intended is a
        worse error than one that is a hair smaller. A step of zero (or unstated)
        leaves the quantity as it is.
        """
        if self.quantity_step is None or self.quantity_step <= 0:
            return float(quantity)
        step = Decimal(str(self.quantity_step))
        steps = (Decimal(str(quantity)) / step).to_integral_value(rounding=ROUND_DOWN)
        return float(steps * step)

    def quantize_price(self, price: float) -> float:
        """Truncate ``price`` to the venue's tick, toward zero."""
        if self.price_tick is None or self.price_tick <= 0:
            return float(price)
        tick = Decimal(str(self.price_tick))
        ticks = (Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_DOWN)
        return float(ticks * tick)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "status": self.status,
            "tradable": self.tradable,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
            "quantity_step": self.quantity_step,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "price_tick": self.price_tick,
            "min_notional": self.min_notional,
            "max_notional": self.max_notional,
            "order_types": list(self.order_types),
            "quote_precision": self.quote_precision,
            "base_precision": self.base_precision,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TradingRules:
        known = set(cls.__dataclass_fields__)
        fields = {k: v for k, v in payload.items() if k in known}
        if "order_types" in fields and fields["order_types"] is not None:
            fields["order_types"] = tuple(fields["order_types"])
        return cls(**fields)


@dataclass(frozen=True)
class OrderCheck:
    """The verdict on one proposed order, with the reasons it would be refused."""

    symbol: str
    quantity: float
    price: float | None
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.reasons

    def explain(self) -> str:
        if self.ok:
            return f"{self.symbol}: order fits the venue's rules"
        return f"{self.symbol}: order refused ({', '.join(self.reasons)})"


def validate_order(
    rules: TradingRules,
    *,
    quantity: float,
    price: float | None = None,
    order_type: str | None = None,
) -> OrderCheck:
    """Check a proposed order against ``rules``, returning every violation.

    ``price`` is optional: a market order has no price of its own, so the
    price-tick and price-band checks are skipped when it is ``None`` — but the
    quantity and notional checks still run, because a market order still has a
    size and a notional. The notional is ``quantity * price`` when a price is
    known; when it is not, the notional checks cannot run and say so in the notes
    rather than passing silently.
    """
    reasons: list[str] = []
    notes: list[str] = []

    if not math.isfinite(quantity) or quantity <= 0:
        reasons.append(REASON_INVALID_QUANTITY)
        return OrderCheck(rules.symbol, quantity, price, tuple(reasons), tuple(notes))

    if not rules.tradable:
        reasons.append(REASON_NOT_TRADABLE)
        notes.append(f"the venue reports status {rules.status!r} for {rules.symbol}")

    if rules.min_quantity is not None and quantity < rules.min_quantity:
        reasons.append(REASON_BELOW_MIN_QTY)
        notes.append(f"quantity {quantity:g} is below min_quantity {rules.min_quantity:g}")
    if rules.max_quantity is not None and quantity > rules.max_quantity:
        reasons.append(REASON_ABOVE_MAX_QTY)
        notes.append(f"quantity {quantity:g} is above max_quantity {rules.max_quantity:g}")
    if rules.quantity_step:
        quantized = rules.quantize_quantity(quantity)
        if abs(quantized - quantity) > 1e-12 * max(1.0, abs(quantity)):
            reasons.append(REASON_QUANTITY_STEP)
            notes.append(
                f"quantity {quantity:g} is not a multiple of the {rules.quantity_step:g} step"
            )

    if (
        order_type is not None
        and rules.order_types
        and order_type.upper() not in {t.upper() for t in rules.order_types}
    ):
        reasons.append(REASON_UNSUPPORTED_ORDER_TYPE)
        notes.append(f"order type {order_type!r} is not in {', '.join(rules.order_types)}")

    if price is not None:
        if not math.isfinite(price) or price <= 0:
            reasons.append(REASON_INVALID_PRICE)
        else:
            if rules.price_tick:
                quantized = rules.quantize_price(price)
                if abs(quantized - price) > 1e-12 * max(1.0, abs(price)):
                    reasons.append(REASON_PRICE_STEP)
                    notes.append(
                        f"price {price:g} is not a multiple of the {rules.price_tick:g} tick"
                    )
            if rules.min_price is not None and price < rules.min_price:
                reasons.append(REASON_BELOW_MIN_PRICE)
            if rules.max_price is not None and price > rules.max_price:
                reasons.append(REASON_ABOVE_MAX_PRICE)

    notional: float | None = None
    if price is not None and math.isfinite(price) and price > 0:
        notional = quantity * price
        if rules.min_notional is not None and notional < rules.min_notional:
            reasons.append(REASON_BELOW_MIN_NOTIONAL)
            notes.append(
                f"notional {notional:,.2f} is below min_notional {rules.min_notional:,.2f}"
            )
        if rules.max_notional is not None and notional > rules.max_notional:
            reasons.append(REASON_ABOVE_MAX_NOTIONAL)
            notes.append(
                f"notional {notional:,.2f} is above max_notional {rules.max_notional:,.2f}"
            )
    else:
        notes.append("notional checks skipped: no price was supplied")

    return OrderCheck(rules.symbol, quantity, price, tuple(reasons), tuple(notes))


def _rules_from_binance(entry: dict) -> TradingRules | None:
    """Parse one Binance ``exchangeInfo`` symbol object into :class:`TradingRules`."""
    base = str(entry.get("baseAsset") or "").upper()
    quote = str(entry.get("quoteAsset") or "").upper()
    if not base or not quote:
        return None
    symbol = f"{base}/{quote}"
    filters = {f.get("filterType"): f for f in entry.get("filters") or [] if isinstance(f, dict)}
    lot = filters.get("LOT_SIZE", {})
    price = filters.get("PRICE_FILTER", {})
    notional = filters.get("NOTIONAL", {})
    step = str(lot.get("stepSize") or "")
    tick = str(price.get("tickSize") or "")
    order_types = tuple(str(t) for t in entry.get("orderTypes") or [])
    return TradingRules(
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        status=str(entry.get("status") or STATUS_UNKNOWN).upper(),
        min_quantity=_as_float(lot.get("minQty")),
        max_quantity=_as_float(lot.get("maxQty")),
        quantity_step=_as_float(step) if step else None,
        min_price=_as_float(price.get("minPrice")),
        max_price=_as_float(price.get("maxPrice")),
        price_tick=_as_float(tick) if tick else None,
        min_notional=_as_float(notional.get("minNotional")),
        max_notional=_as_float(notional.get("maxNotional")),
        order_types=order_types,
        quote_precision=_decimals(tick) if tick else 0,
        base_precision=_decimals(step) if step else 0,
    )


def parse_exchange_info(payload: dict) -> dict[str, TradingRules]:
    """Parse a Binance ``exchangeInfo`` body into canonical-symbol rules.

    Only spot instruments are described. An entry with no base or quote asset, or
    a malformed one, is skipped rather than guessed at; the caller sees exactly the
    instruments the venue named.
    """
    if not isinstance(payload, dict):
        raise InstrumentError("exchange info must be a JSON object")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise InstrumentError("exchange info carries no 'symbols' list")
    out: dict[str, TradingRules] = {}
    for entry in symbols:
        if not isinstance(entry, dict):
            continue
        rules = _rules_from_binance(entry)
        if rules is not None:
            out[rules.symbol] = rules
    if not out:
        raise InstrumentError("exchange info described no usable instruments")
    return out


@dataclass
class InstrumentRegistry:
    """The instruments a venue will trade, keyed by canonical symbol.

    Built from a venue's own answer (Binance's ``exchangeInfo``) so nothing is
    hard-coded. The registry is *data*: it can be written to disk and reloaded, so
    a restart does not refetch it, and it answers the two questions an order needs:
    *is this symbol tradeable?* and *what are its rules?*
    """

    rules: dict[str, TradingRules] = field(default_factory=dict)
    venue: str = "binance"
    fetched_at: str | None = None

    def __post_init__(self) -> None:
        self.rules = {parse_pair(s).canonical: r for s, r in self.rules.items()}

    def get(self, symbol: str) -> TradingRules | None:
        """The rules for ``symbol``, or ``None`` when the venue did not list it."""
        pair = parse_pair(symbol)
        return self.rules.get(pair.canonical)

    def is_supported(self, symbol: str) -> bool:
        """Whether the venue listed ``symbol`` at all (tradeable or not)."""
        return self.get(symbol) is not None

    def is_tradable(self, symbol: str) -> bool:
        """Whether the venue lists ``symbol`` *and* reports it as trading."""
        rules = self.get(symbol)
        return rules is not None and rules.tradable

    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.rules))

    def tradable_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(s for s, r in self.rules.items() if r.tradable))

    def filter_tradable(
        self, symbols: list[str] | tuple[str, ...]
    ) -> tuple[list[str], dict[str, str]]:
        """Split ``symbols`` into those that are tradeable and those that are not.

        Returns ``(tradeable, rejected)`` where ``rejected`` maps a symbol to why:
        ``"unknown"`` when the venue never listed it, else its status. The caller
        is expected to *not* send an order for anything in ``rejected``.
        """
        tradeable: list[str] = []
        rejected: dict[str, str] = {}
        for symbol in symbols:
            pair = parse_pair(symbol)
            rules = self.rules.get(pair.canonical)
            if rules is None:
                rejected[pair.canonical] = "unknown"
            elif not rules.tradable:
                rejected[pair.canonical] = rules.status
            else:
                tradeable.append(pair.canonical)
        return tradeable, rejected

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "venue": self.venue,
            "fetched_at": self.fetched_at,
            "rules": {s: r.to_dict() for s, r in sorted(self.rules.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> InstrumentRegistry:
        if not isinstance(payload, dict):
            raise InstrumentError("an instrument registry must be a mapping")
        raw = payload.get("rules") or {}
        rules = {s: TradingRules.from_dict(r) for s, r in raw.items() if isinstance(r, dict)}
        return cls(
            rules=rules,
            venue=str(payload.get("venue") or "binance"),
            fetched_at=payload.get("fetched_at"),
        )


__all__ = [
    "ORDER_REASONS",
    "REASON_ABOVE_MAX_NOTIONAL",
    "REASON_ABOVE_MAX_PRICE",
    "REASON_ABOVE_MAX_QTY",
    "REASON_BELOW_MIN_NOTIONAL",
    "REASON_BELOW_MIN_PRICE",
    "REASON_BELOW_MIN_QTY",
    "REASON_INVALID_PRICE",
    "REASON_INVALID_QUANTITY",
    "REASON_NOT_TRADABLE",
    "REASON_PRICE_STEP",
    "REASON_QUANTITY_STEP",
    "REASON_UNSUPPORTED_ORDER_TYPE",
    "STATUS_BREAK",
    "STATUS_HALTED",
    "STATUS_TRADING",
    "STATUS_UNKNOWN",
    "InstrumentError",
    "InstrumentRegistry",
    "OrderCheck",
    "TradingRules",
    "parse_exchange_info",
    "validate_order",
]
