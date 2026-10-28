"""The simulated ledger: cash, positions, average entry, P&L and fees.

This is the book that orders move. It is deliberately dumb: it knows how to
apply a fill and charge a fee, and it refuses anything that would make the book
inconsistent. It does not decide whether a trade is allowed (that is the risk
engine) and it does not decide what price a trade gets (that is the execution
simulator). Keeping those concerns out means the accounting can be tested by
hand, which is the only way to trust it.

Long-only, spot semantics. A position's quantity is never negative: a sell
larger than the holding is rejected rather than turning into a short, because
this book models an equity/spot account. There is no margin, so ``cash`` is
never negative either, and ``buying_power`` equals available cash.

Cash conservation is an invariant, not a convention. Every fill moves cash by
exactly ``-(signed quantity * price) - fee``, which gives

    cash + cost_basis + total_fees - realized_pnl_gross == initial_cash

while nothing is marked to market. That is the identity ``tests/test_ledger.py``
asserts. It only holds if ``realized_pnl`` is *gross* of fees, so fees are
tracked once, in ``total_fees``, and ``net_realized_pnl`` combines them for
reporting. A net-of-fee ``realized_pnl`` would double count the exit fee and
make the identity unstateable.

The ledger takes no dependency on prices. It holds ``last_price`` per holding
only when the caller marks it, so a ledger that is never marked reports
unrealised P&L of zero rather than a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tradingagents.portfolio.events import EventType, LedgerEvent


def _require_finite(value: float, what: str) -> float:
    """Reject NaN and infinity before they reach the book.

    A NaN is the worst input this ledger can be handed, because it is not a
    *wrong* number so much as a *contagious* one: ``cash + nan`` is NaN, and NaN
    compares false against every bound, so a single NaN price slips past the
    ``<= 0`` guards and poisons cash, equity and every later comparison at once.
    Infinity is quieter but no better: it passes the same guards and produces an
    infinite cash balance. Neither is a market outcome, so neither is accepted.
    """
    if not math.isfinite(value):
        raise LedgerError(f"{what} must be a finite number, got {value!r}")
    return float(value)


class LedgerError(Exception):
    """The book would become inconsistent if this operation were applied."""


class InsufficientFundsError(LedgerError):
    """A buy would take cash below zero."""


class InsufficientPositionError(LedgerError):
    """A sell exceeds the holding, which this long-only book does not allow."""


@dataclass
class Holding:
    """One open long position, with the figures needed to price and report it."""

    symbol: str
    quantity: float = 0.0
    average_entry_price: float = 0.0
    last_price: float | None = None
    realized_pnl: float = 0.0
    """Gross realised P&L: the price change on what was sold, before fees."""
    fees_paid: float = 0.0
    opened_at: datetime | None = None

    @property
    def cost_basis(self) -> float:
        """What the open quantity cost, at average entry."""
        return self.quantity * self.average_entry_price

    @property
    def market_value(self) -> float:
        """Current value, or the cost basis when the position has not been marked."""
        price = self.last_price if self.last_price is not None else self.average_entry_price
        return self.quantity * price

    @property
    def unrealized_pnl(self) -> float:
        """Gain or loss on the open quantity, zero until the position is marked."""
        if self.last_price is None:
            return 0.0
        return (self.last_price - self.average_entry_price) * self.quantity

    @property
    def is_open(self) -> bool:
        return self.quantity > 1e-12


@dataclass
class LedgerSnapshot:
    """A read-only view of the book at one instant, for reporting and metrics."""

    timestamp: datetime
    cash: float
    equity: float
    market_value: float
    total_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    positions: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    @property
    def available_cash(self) -> float:
        """Cash free to spend. Equals cash: this book has no margin."""
        return self.cash

    @property
    def buying_power(self) -> float:
        """How much can be committed to new positions. Equals available cash."""
        return self.cash


class Ledger:
    """A simulated cash-and-positions book with an append-only event log."""

    def __init__(self, initial_cash: float, currency: str = "USD") -> None:
        if initial_cash <= 0:
            raise LedgerError(f"initial_cash must be positive, got {initial_cash}")
        self.initial_cash = _require_finite(initial_cash, "initial_cash")
        self.currency = currency
        self.cash = self.initial_cash
        self.holdings: dict[str, Holding] = {}
        self.events: list[LedgerEvent] = []
        self._seq = 0
        self._unattributed_fees = 0.0

    # --- Event log ----------------------------------------------------------

    def _record(
        self,
        type_: EventType,
        *,
        timestamp: datetime,
        symbol: str | None = None,
        detail: dict | None = None,
        reason: str = "",
    ) -> LedgerEvent:
        self._seq += 1
        event = LedgerEvent(
            seq=self._seq,
            timestamp=_as_utc(timestamp),
            type=type_,
            symbol=symbol,
            detail=detail or {},
            reason=reason,
        )
        self.events.append(event)
        return event

    def record_event(
        self,
        type_: EventType,
        *,
        timestamp: datetime,
        symbol: str | None = None,
        detail: dict | None = None,
        reason: str = "",
    ) -> LedgerEvent:
        """Record an event the ledger did not itself cause.

        The execution engine records the order lifecycle here (created, rejected,
        accepted), so the log is one ordered history of everything that happened
        to the book rather than two logs that have to be interleaved later. The
        ledger remains the only writer, so ``seq`` stays dense.
        """
        return self._record(type_, timestamp=timestamp, symbol=symbol, detail=detail, reason=reason)

    # --- Positions ----------------------------------------------------------

    def holding(self, symbol: str) -> Holding:
        """The holding for ``symbol``, created empty on first access."""
        if symbol not in self.holdings:
            self.holdings[symbol] = Holding(symbol=symbol)
        return self.holdings[symbol]

    def open_positions(self) -> dict[str, Holding]:
        """Only the holdings with a non-zero quantity."""
        return {s: h for s, h in self.holdings.items() if h.is_open}

    def position_quantity(self, symbol: str) -> float:
        """Units held of ``symbol``; 0.0 when flat."""
        return self.holding(symbol).quantity

    def average_entry_price(self, symbol: str) -> float:
        """Average entry per unit of ``symbol``; 0.0 when flat."""
        return self.holding(symbol).average_entry_price

    # --- Marks --------------------------------------------------------------

    def mark(self, symbol: str, price: float) -> None:
        """Record the latest price for ``symbol``, so unrealised P&L is meaningful."""
        _require_finite(price, "mark price")
        if price <= 0:
            raise LedgerError(f"mark price must be positive, got {price}")
        self.holding(symbol).last_price = float(price)

    def mark_many(self, prices: dict[str, float]) -> None:
        """Mark several symbols at once; a symbol with no price keeps its last mark."""
        for symbol, price in prices.items():
            self.mark(symbol, price)

    # --- Mutations ----------------------------------------------------------

    def apply_fill(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
        fee: float,
        timestamp: datetime,
        reason: str = "",
        order_id: str | None = None,
    ) -> Holding:
        """Apply one fill: ``quantity`` > 0 buys, ``quantity`` < 0 sells.

        This is the ledger's only way to change cash or positions, so every
        accounting rule lives here. ``price`` is the fill price per unit and
        ``fee`` is the total commission for the fill; both are decided upstream
        by the execution simulator, which is why this method takes them as given.

        Raises :class:`LedgerError` for a zero quantity, a non-positive price, a
        negative fee, a buy the cash cannot cover, or a sell larger than the
        holding. Every rejection happens before any state changes.
        """
        if quantity == 0:
            raise LedgerError("a fill of zero units is not a fill")
        _require_finite(quantity, "fill quantity")
        _require_finite(price, "fill price")
        _require_finite(fee, "fill fee")
        if price <= 0:
            raise LedgerError(f"fill price must be positive, got {price}")
        if fee < 0:
            raise LedgerError(f"fee must not be negative, got {fee}")

        holding = self.holding(symbol)
        cash_delta = -(quantity * price) - fee

        if quantity > 0:
            if self.cash + cash_delta < -1e-9:
                raise InsufficientFundsError(
                    f"buying {quantity} {symbol} at {price} with fee {fee} needs "
                    f"{quantity * price + fee:.2f} but only {self.cash:.2f} is free"
                )
        else:
            if abs(quantity) > holding.quantity + 1e-9:
                raise InsufficientPositionError(
                    f"selling {abs(quantity)} {symbol} exceeds the {holding.quantity} held; "
                    "this book is long-only"
                )

        was_open = holding.is_open
        old_quantity = holding.quantity

        if quantity > 0:
            new_quantity = old_quantity + quantity
            # Weighted average, so adding to a position blends the entry rather
            # than replacing it with the latest price.
            holding.average_entry_price = (
                old_quantity * holding.average_entry_price + quantity * price
            ) / new_quantity
            holding.quantity = new_quantity
            if not was_open:
                holding.opened_at = _as_utc(timestamp)
        else:
            sold = abs(quantity)
            # Realised P&L is measured against average entry, gross of fees, so
            # the conservation identity in the module docstring holds. Fees are
            # accounted once, in fees_paid.
            holding.realized_pnl += (price - holding.average_entry_price) * sold
            holding.quantity = old_quantity - sold
            if not holding.is_open:
                holding.quantity = 0.0
                holding.average_entry_price = 0.0

        holding.fees_paid += fee
        holding.last_price = price
        self.cash += cash_delta

        # A float residue on a fully-closed position would read as a tiny holding.
        if abs(self.cash) < 1e-9:
            self.cash = 0.0

        fill_detail = {
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "cash_delta": cash_delta,
            "cash_after": self.cash,
        }
        # The order this fill belongs to, so a report can tie a fill back to the
        # order that produced it. Omitted for a direct ledger fill with no order
        # behind it, so every value in ``detail`` stays a figure or a flag.
        if order_id is not None:
            fill_detail["order_id"] = order_id
        self._record(
            EventType.ORDER_FILLED,
            timestamp=timestamp,
            symbol=symbol,
            detail=fill_detail,
            reason=reason,
        )
        self._record_position_event(symbol, holding, was_open, quantity, timestamp)
        if fee:
            self._record(
                EventType.FEE_CHARGED,
                timestamp=timestamp,
                symbol=symbol,
                detail={"fee": fee, "cumulative": holding.fees_paid},
            )
        return holding

    def _record_position_event(
        self, symbol: str, holding: Holding, was_open: bool, quantity: float, timestamp: datetime
    ) -> None:
        """Classify the fill's effect on the position, so the log reads as a history."""
        if quantity > 0:
            type_ = EventType.POSITION_INCREASED if was_open else EventType.POSITION_OPENED
        elif holding.is_open:
            type_ = EventType.POSITION_REDUCED
        else:
            type_ = EventType.POSITION_CLOSED
        self._record(
            type_,
            timestamp=timestamp,
            symbol=symbol,
            detail={
                "quantity": holding.quantity,
                "average_entry_price": holding.average_entry_price,
                "realized_pnl": holding.realized_pnl,
            },
        )

    def charge_fee(self, *, amount: float, timestamp: datetime, symbol: str | None = None,
                   reason: str = "") -> None:
        """Charge a cost with no market side (a custody or data fee)."""
        _require_finite(amount, "fee amount")
        if amount < 0:
            raise LedgerError(f"fee must not be negative, got {amount}")
        if amount > self.cash + 1e-9:
            raise InsufficientFundsError(
                f"a fee of {amount:.2f} exceeds the {self.cash:.2f} of cash available"
            )
        self.cash -= amount
        if symbol is not None:
            self.holding(symbol).fees_paid += amount
        else:
            # No instrument to attribute the cost to, but it was still charged:
            # total_fees must include it or the conservation identity breaks.
            self._unattributed_fees += amount
        self._record(
            EventType.FEE_CHARGED,
            timestamp=timestamp,
            symbol=symbol,
            detail={"fee": amount, "cash_after": self.cash},
            reason=reason,
        )

    # --- Aggregates ---------------------------------------------------------

    @property
    def market_value(self) -> float:
        """Value of the open positions at their last marks."""
        return sum(h.market_value for h in self.holdings.values())

    @property
    def equity(self) -> float:
        """Cash plus the marked value of the positions."""
        return self.cash + self.market_value

    @property
    def total_exposure(self) -> float:
        """Fraction of equity held in positions, 0 when equity is zero."""
        if self.equity <= 0:
            return 0.0
        return self.market_value / self.equity

    @property
    def realized_pnl(self) -> float:
        """Gross realised P&L across every holding, before fees."""
        return sum(h.realized_pnl for h in self.holdings.values())

    @property
    def net_realized_pnl(self) -> float:
        """Realised P&L after every fee paid, for reporting."""
        return self.realized_pnl - self.total_fees

    @property
    def unrealized_pnl(self) -> float:
        """Sum of unrealised P&L across every holding."""
        return sum(h.unrealized_pnl for h in self.holdings.values())

    @property
    def total_fees(self) -> float:
        """Every fee charged, realised or not, attributed or not."""
        return sum(h.fees_paid for h in self.holdings.values()) + self._unattributed_fees

    @property
    def total_pnl(self) -> float:
        """Equity less the starting cash: the book's whole result."""
        return self.equity - self.initial_cash

    def snapshot(self, timestamp: datetime | None = None) -> LedgerSnapshot:
        """A consistent read of the book for reporting and metric computation."""
        return LedgerSnapshot(
            timestamp=_as_utc(timestamp or datetime.now(UTC)),
            cash=self.cash,
            equity=self.equity,
            market_value=self.market_value,
            total_exposure=self.total_exposure,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            total_fees=self.total_fees,
            positions={s: h.quantity for s, h in self.open_positions().items()},
            prices={s: h.last_price for s, h in self.holdings.items() if h.last_price is not None},
        )

    # --- Persistence --------------------------------------------------------

    def to_state(self) -> dict:
        """The ledger's complete state, JSON-safe, so a run can be resumed.

        This is deliberately not :meth:`snapshot`. A snapshot is a reporting view
        of the headline figures, and it cannot rebuild a book: it drops the
        per-holding realised P&L, the fee bookkeeping, the marks of closed
        positions and the whole event log. Restoring from it would silently change
        the accounting. This captures every field the ledger holds — including the
        sequence counter, so the restored book's next ``seq`` is the one the
        original would have used — and :meth:`from_state` rebuilds it exactly.
        """
        return {
            "initial_cash": self.initial_cash,
            "currency": self.currency,
            "cash": self.cash,
            "unattributed_fees": self._unattributed_fees,
            "seq": self._seq,
            "holdings": {
                symbol: {
                    "symbol": holding.symbol,
                    "quantity": holding.quantity,
                    "average_entry_price": holding.average_entry_price,
                    "last_price": holding.last_price,
                    "realized_pnl": holding.realized_pnl,
                    "fees_paid": holding.fees_paid,
                    "opened_at": holding.opened_at.isoformat()
                    if holding.opened_at is not None else None,
                }
                for symbol, holding in self.holdings.items()
            },
            "events": [
                {
                    "seq": event.seq,
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.type.value,
                    "symbol": event.symbol,
                    "detail": event.detail,
                    "reason": event.reason,
                }
                for event in self.events
            ],
        }

    @classmethod
    def from_state(cls, state: dict) -> Ledger:
        """Rebuild a ledger from :meth:`to_state`, with identical accounting.

        The reconstruction is exact: cash, positions, average entries, realised
        P&L, fees, marks and the event log all come back, and ``_seq`` is restored
        to the saved value so a resumed run's event numbers continue rather than
        restart and collide with the ones already written.
        """
        ledger = cls(
            initial_cash=float(state["initial_cash"]),
            currency=state.get("currency", "USD"),
        )
        ledger.cash = float(state["cash"])
        ledger._unattributed_fees = float(state.get("unattributed_fees", 0.0))
        for symbol, holding_state in state.get("holdings", {}).items():
            opened_at = holding_state.get("opened_at")
            ledger.holdings[symbol] = Holding(
                symbol=holding_state.get("symbol", symbol),
                quantity=float(holding_state.get("quantity", 0.0)),
                average_entry_price=float(holding_state.get("average_entry_price", 0.0)),
                last_price=None if holding_state.get("last_price") is None
                else float(holding_state["last_price"]),
                realized_pnl=float(holding_state.get("realized_pnl", 0.0)),
                fees_paid=float(holding_state.get("fees_paid", 0.0)),
                opened_at=datetime.fromisoformat(opened_at) if opened_at else None,
            )
        ledger.events = [
            LedgerEvent(
                seq=int(event["seq"]),
                timestamp=datetime.fromisoformat(event["timestamp"]),
                type=EventType(event["type"]),
                symbol=event.get("symbol"),
                detail=event.get("detail", {}),
                reason=event.get("reason", ""),
            )
            for event in state.get("events", [])
        ]
        ledger._seq = int(state.get("seq", len(ledger.events)))
        return ledger


def _as_utc(value: datetime) -> datetime:
    """Normalize a timestamp to UTC, treating a naive one as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
