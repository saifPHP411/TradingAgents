"""The simulated execution engine: turns an order into a fill, or into a reason.

This is where a signal stops being an intention. The engine is handed an order
and a :class:`MarketContext` and answers with the order's outcome, applying the
cost model on the way. It never fetches a price, never touches a network, and
never decides whether a trade is *allowed* — the risk engine has already ruled on
that, and duplicating the check here would give two places to disagree.

The checks it does perform are the ones only it can make, because they depend on
the fill's own arithmetic:

- the market must be open and a price must exist for the requested timestamp;
- a LIMIT order fills only if the bar's price reaches its limit, and it fills at
  the limit or better (never worse — a limit that fills worse than its limit is
  not a limit);
- a MARKET order fills at the bar price moved adversely by spread and slippage;
- a buy must be covered by the cash it will need, fees included;
- a sell must be covered by the position actually held, because the book is
  long-only.

Partial fills are modelled for a market order whose size exceeds a configurable
share of the bar's volume: only that share fills, the order ends
``PARTIALLY_FILLED``, and the caller decides whether to re-submit. A limit order
does not partially fill, because a daily bar cannot say whether the limit was
touched before or after the volume was consumed.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from tradingagents.execution.costs import CostModel, Side
from tradingagents.execution.interface import (
    ExecutionMode,
    MarketContext,
    assert_simulation_only,
)
from tradingagents.execution.orders import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
)
from tradingagents.portfolio.events import EventType
from tradingagents.portfolio.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    Ledger,
)


class ExecutionSimulator:
    """Fills orders against supplied prices, charging configured costs.

    Constructed with the ledger it will move and the cost schedule it will
    charge, so both are explicit at the call site. ``mode`` defaults to
    ``BACKTEST``; a paper run passes ``PAPER``. Neither can be live.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        costs: CostModel | None = None,
        mode: ExecutionMode = ExecutionMode.BACKTEST,
        max_volume_participation: float = 0.0,
    ) -> None:
        assert_simulation_only(mode)
        if max_volume_participation < 0 or max_volume_participation > 1:
            raise ValueError(
                f"max_volume_participation must be between 0 and 1, got {max_volume_participation}"
            )
        self.ledger = ledger
        self.costs = costs or CostModel()
        self.mode = mode
        # 0 disables partial fills entirely, which is the default: a caller that
        # has not supplied volumes cannot have participation modelled for it.
        self.max_volume_participation = max_volume_participation

    # --- Public API ---------------------------------------------------------

    def submit(
        self, order: Order, *, market: MarketContext, idempotent: bool = False
    ) -> Order:
        """Attempt to execute ``order``, recording the outcome on the order.

        Never raises for a market condition: a rejected or unfilled order is a
        normal result that the caller must be able to record, not an exception
        that aborts a sweep. It raises only for a programming error — an order
        whose symbol disagrees with the market context it was given.

        Submitting an order that has already reached a terminal state is such a
        programming error. It is refused rather than filled again, because the
        ledger has no way to tell a re-submission from a second genuine order, and
        a double fill is a real position the caller never asked for. The first
        outcome is returned unchanged, so a caller that accidentally loops gets
        the same answer instead of an accumulating position.

        A ``PARTIALLY_FILLED`` order is the exception, because it is not terminal
        and a second submission would fill its remaining size — a real second
        fill, not a re-run of the first. A caller that re-drives an order it may
        already have submitted (a paper loop retrying, say) passes
        ``idempotent=True`` to get the recorded outcome back instead of filling
        again. The backtester, which submits each order exactly once, leaves it
        ``False`` so a genuine re-submission of a partially filled order still
        works.
        """
        if market.symbol != order.symbol:
            raise ValueError(
                f"order is for {order.symbol} but the market context is {market.symbol}"
            )
        if order.status.is_terminal:
            return order
        if idempotent and order.filled_quantity > 0:
            # This order was already worked. Return it unchanged rather than
            # filling its remaining size a second time.
            return order

        if not order.created_recorded:
            self.ledger.record_event(
                EventType.ORDER_CREATED, timestamp=order.created_at, symbol=order.symbol,
                detail={"order_id": order.order_id, "side": order.side.value,
                        "quantity": order.quantity, "type": order.order_type.value,
                        "limit_price": order.limit_price},
            )
            order.created_recorded = True

        accepted = self._pre_trade_rejection(order, market)
        if accepted is not None:
            return self._reject(order, *accepted)
        # The order passed every pre-trade check: it is accepted and will now be
        # worked. Recording acceptance here, after the checks and before the
        # fill, keeps the lifecycle in order (CREATED, then ACCEPTED, then
        # FILLED/CANCELLED) rather than logging acceptance before creation.
        self.ledger.record_event(
            EventType.ORDER_ACCEPTED, timestamp=order.created_at, symbol=order.symbol,
            detail={"order_id": order.order_id, "quantity": order.quantity},
        )

        fill_quantity, fill_price = self._resolve_fill(order, market)
        if fill_quantity <= 0:
            return self._cancel(
                order, RejectReason.LIMIT_NOT_REACHED,
                f"the {order.order_type.value.lower()} price {order.limit_price} was not reached "
                f"at a reference price of {market.price}",
            )
        # The fill's own arithmetic is the last place a bad number can enter the
        # book: a non-finite price or size here would be applied by the ledger and
        # poison cash and equity. Reject rather than attempt the fill.
        if not math.isfinite(fill_price) or fill_price <= 0:
            return self._reject(
                order, RejectReason.INVALID_PRICE,
                f"costs priced the fill at {fill_price!r}, which is not a usable price",
            )
        if not math.isfinite(fill_quantity) or fill_quantity <= 0:
            return self._reject(
                order, RejectReason.INVALID_QUANTITY,
                f"the fill size resolved to {fill_quantity!r}, which is not usable",
            )

        return self._apply(order, market, fill_quantity, fill_price)

    # --- Pre-trade checks ---------------------------------------------------

    def _pre_trade_rejection(self, order: Order, market: MarketContext) -> tuple[RejectReason, str] | None:
        """The reasons an order cannot be attempted at all."""
        if order.quantity <= 0:
            return RejectReason.INVALID_QUANTITY, f"quantity {order.quantity} is not positive"
        if not market.is_open:
            return RejectReason.MARKET_CLOSED, f"the market for {order.symbol} is closed at {market.timestamp}"
        if market.price is None:
            return RejectReason.NO_PRICE, f"no price for {order.symbol} at {market.timestamp}"
        if not math.isfinite(market.price):
            # Checked before the cost model sees it: a NaN or inf price is false
            # against ``<= 0``, so it would otherwise reach the costs and come
            # back as a non-finite fill price that the ledger would apply.
            return RejectReason.INVALID_PRICE, f"price {market.price!r} is not a finite number"
        if market.price <= 0:
            return RejectReason.INVALID_PRICE, f"price {market.price} is not positive"
        if order.order_type is OrderType.LIMIT and order.limit_price is None:
            return RejectReason.INVALID_PRICE, "a LIMIT order has no limit price"
        return None

    # --- Fill resolution ----------------------------------------------------

    def projected_fill_price(
        self, order: Order, *, market: MarketContext, quantity: float | None = None
    ) -> float | None:
        """The price a fill would get at ``quantity``, without moving the ledger.

        A caller that has to re-check a limit against the price the fill will
        actually get — a risk cap enforced at execution rather than at the
        decision — needs that price before committing. This resolves the fill the
        same way :meth:`submit` will, but applies nothing, so asking costs nothing
        and leaves no event behind.

        Returns ``None`` when the order would not fill at all: a limit the market
        has not reached, or a market that is closed or has no usable price. The
        caller then leaves the outcome to :meth:`submit` rather than guessing.
        """
        if market.symbol != order.symbol:
            raise ValueError(
                f"order is for {order.symbol} but the market context is {market.symbol}"
            )
        probe = order if quantity is None else order.model_copy(update={"quantity": quantity})
        if probe.quantity <= 0 or not math.isfinite(probe.quantity):
            return None
        if not market.is_open or market.price is None or not math.isfinite(market.price):
            return None
        if market.price <= 0:
            return None
        fill_quantity, fill_price = self._resolve_fill(probe, market)
        if fill_quantity <= 0 or not math.isfinite(fill_price) or fill_price <= 0:
            return None
        return fill_price

    def _resolve_fill(self, order: Order, market: MarketContext) -> tuple[float, float]:
        """The quantity and price this order fills at, or ``(0, 0)`` if it does not.

        For a LIMIT order the bar's price decides whether the limit is reachable,
        and the limit (or the bar price, if it is better) is the fill price. For
        a MARKET order the bar price is the reference and the costs move it.
        """
        assert market.price is not None
        reference = market.price

        if order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            remaining = order.remaining_quantity
            if order.side is OrderSide.BUY:
                # A buy limit fills only if the market came down to it.
                if reference > order.limit_price:
                    return 0.0, 0.0
                # Filling at the better of the two: the limit caps the price, it
                # does not set it.
                return remaining, min(reference, order.limit_price)
            if reference < order.limit_price:
                return 0.0, 0.0
            return remaining, max(reference, order.limit_price)

        fill_price, _, _ = self.costs.fill_price(
            side=Side.BUY if order.side is OrderSide.BUY else Side.SELL,
            reference_price=reference,
            quantity=order.quantity,
            average_daily_volume=market.average_daily_volume,
        )
        return self._fillable_quantity(order, market), fill_price

    def _fillable_quantity(self, order: Order, market: MarketContext) -> float:
        """Cap the fill at the configured share of the bar's volume, when known.

        Only market orders are capped: a limit order's fill is already
        conditional on price, and a daily bar cannot say how much volume traded
        at that price. The cap is applied to what remains to fill, so a
        partially filled order that is re-driven only ever fills its remainder.
        """
        remaining = order.remaining_quantity
        volume = market.average_daily_volume
        if not self.max_volume_participation or not volume or volume <= 0:
            return remaining
        cap = volume * self.max_volume_participation
        return min(remaining, cap)

    # --- Applying the fill --------------------------------------------------

    def _apply(self, order: Order, market: MarketContext, quantity: float, price: float) -> Order:
        """Move the ledger, then record the outcome on the order."""
        signed = quantity if order.side is OrderSide.BUY else -quantity
        # A limit order rests in the book (maker); a market order takes (taker).
        commission = self.costs.fee_for_fill(
            quantity=quantity, price=price, is_maker=order.order_type is OrderType.LIMIT
        )

        try:
            self.ledger.apply_fill(
                symbol=order.symbol,
                quantity=signed,
                price=price,
                fee=commission,
                timestamp=_as_utc(market.timestamp),
                reason=f"order {order.order_id}",
                order_id=order.order_id,
            )
        except InsufficientFundsError as exc:
            return self._reject(order, RejectReason.INSUFFICIENT_CASH, str(exc))
        except InsufficientPositionError as exc:
            return self._reject(order, RejectReason.INSUFFICIENT_POSITION, str(exc))

        # A quantity-weighted average across every fill, not the last fill price.
        # An order can fill in pieces at different prices (a market order capped
        # by volume participation on two sessions, a limit worked over two bars),
        # and the field is named ``average_fill_price``: reporting only the most
        # recent piece would understate or overstate what the order actually paid.
        # The ledger keeps its own VWAP for the position; this keeps the order's
        # record of its own fills consistent with it.
        applied = min(quantity, order.quantity - order.filled_quantity)
        prior_notional = (order.average_fill_price or 0.0) * order.filled_quantity
        order.filled_quantity += applied
        order.average_fill_price = (prior_notional + applied * price) / order.filled_quantity
        # A capped fill is partial only if something is left over; a market order
        # whose full size fits in the bar is simply filled. The status reflects the
        # total filled, so a partially filled order that is re-driven and fills
        # more still ends FILLED once its whole size is done.
        order.status = (
            OrderStatus.FILLED
            if order.filled_quantity >= order.quantity - 1e-9
            else OrderStatus.PARTIALLY_FILLED
        )
        return order

    def _reject(self, order: Order, reason: RejectReason, detail: str) -> Order:
        """Mark the order rejected and record why, without touching the book.

        A rejection means the order was never accepted — it was malformed, the
        market was closed, or the book could not fund it. It is distinct from a
        cancellation, where a valid order simply did not get its price.
        """
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        order.reject_detail = detail
        self.ledger.record_event(
            EventType.ORDER_REJECTED,
            timestamp=order.created_at,
            symbol=order.symbol,
            detail={"order_id": order.order_id, "reason": reason.value},
            reason=detail,
        )
        return order

    def cancel(self, order: Order, *, reason: RejectReason, detail: str) -> Order:
        """Cancel a live order that never got its price, recording the outcome.

        The honest terminal state for an order the market did not fill. A
        cancellation is recorded as ``ORDER_CANCELLED``, not ``ORDER_REJECTED``,
        so the event log distinguishes "the book refused this" from "the price
        never came". A caller that has already reached a terminal order gets it
        back unchanged, so cancelling twice is not a second event.
        """
        if order.status.is_terminal:
            return order
        return self._cancel(order, reason, detail)

    def _cancel(self, order: Order, reason: RejectReason, detail: str) -> Order:
        """Mark the order cancelled and record why, without touching the book."""
        order.status = OrderStatus.CANCELLED
        order.reject_reason = reason
        order.reject_detail = detail
        self.ledger.record_event(
            EventType.ORDER_CANCELLED,
            timestamp=order.created_at,
            symbol=order.symbol,
            detail={"order_id": order.order_id, "reason": reason.value},
            reason=detail,
        )
        return order


def _as_utc(value: datetime) -> datetime:
    """Normalize a fill timestamp to UTC, treating a naive one as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
