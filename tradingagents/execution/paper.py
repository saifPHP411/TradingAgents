"""Paper trading: the same pipeline as a backtest, driven by a live clock.

A paper session runs the identical chain the backtester runs — signal, risk
verdict, order, simulated fill, ledger — with two differences, both in the
caller rather than the engine:

- the clock is the wall clock instead of a historical calendar, and
- quotes come from a caller-supplied source instead of a bar feed.

That is the whole point of this module. The accounting cannot diverge between a
backtest and a paper run, because they share the ledger, the risk engine and the
execution simulator. If paper results drifted from backtest results, the paper
run would be telling a different story about the same strategy.

What this module is not, and cannot become by configuration:

There is no order path that reaches a venue. ``ExecutionMode.PAPER`` is resolved
through :func:`~tradingagents.execution.interface.resolve_mode`, the simulator's
constructor calls ``assert_simulation_only``, and the quote source is a plain
callable that returns a price. A live implementation would have to be a new class
written deliberately, not a setting.

The quote source is a callable rather than an interface for a reason: it keeps
the module free of any network dependency, so the tests can drive a whole session
from a dict of prices and the same code path is exercised as in production.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.execution.costs import CostModel
from tradingagents.execution.interface import (
    ExecutionMode,
    MarketContext,
    resolve_mode,
)
from tradingagents.execution.orders import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
    TimeInForce,
)
from tradingagents.execution.simulator import ExecutionSimulator
from tradingagents.portfolio.events import EventType
from tradingagents.portfolio.ledger import Ledger
from tradingagents.risk.engine import RiskEngine, RiskLimits, RiskVerdict
from tradingagents.signals.contract import (
    OrderIntent,
    SignalAction,
    SignalTimeInForce,
    TradingSignal,
)

PAPER_STATE_SCHEMA_VERSION = 1
"""The version of the persisted paper-session state.

Bumped when the on-disk shape changes in a way a reader must know about, so an
older reader refuses a newer file rather than misreading it. The reader accepts
the versions it understands and rejects the rest loudly.
"""


class PaperSessionError(RuntimeError):
    """A paper session was driven in a way it cannot support."""


class PaperStateError(PaperSessionError):
    """A persisted paper session could not be restored.

    Raised when the file is malformed, its schema version is not understood, or
    the state it holds is not one a session could have reached. A recovery that
    guessed at a corrupt state would resume with silently wrong accounting, so
    this fails instead.
    """


@dataclass(frozen=True)
class Quote:
    """A price observation for one symbol at one moment.

    ``timestamp`` is the observation's own time, not the time it was read, so a
    stale quote can be detected by the caller rather than silently filled.

    ``low`` and ``high`` are the session's intraday range when the caller has it.
    A LIMIT order needs them to decide whether the market traded to its limit: a
    single reference price can only say "the market is at X", not "the market
    passed through X". They are optional, so a caller with only one price is not
    forced to invent a range. ``average_daily_volume`` lets a large order be
    capped by participation, exactly as a bar's volume does in a backtest.
    """

    symbol: str
    price: float
    timestamp: datetime
    is_open: bool = True
    average_daily_volume: float | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise PaperSessionError(f"quote price must be finite and positive, got {self.price}")
        for name, value in (("low", self.low), ("high", self.high)):
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise PaperSessionError(
                    f"quote {name} must be finite and positive when supplied, got {value}"
                )
        if self.low is not None and self.high is not None and self.low > self.high:
            # A malformed range would make a LIMIT order's reachability test
            # meaningless — a buy limit "reached" by a low above the high is not a
            # market fact. Refuse it rather than fill against it.
            raise PaperSessionError(
                f"quote range is inconsistent: low {self.low} is above high {self.high}"
            )

    def as_market(self) -> MarketContext:
        return MarketContext(
            symbol=self.symbol,
            price=self.price,
            timestamp=self.timestamp,
            is_open=self.is_open,
            average_daily_volume=self.average_daily_volume,
            low=self.low,
            high=self.high,
        )


QuoteSource = Callable[[str], Quote]
"""Returns the current quote for a symbol. A dict lookup is a valid one."""


@dataclass
class SessionStep:
    """One decision's outcome, kept so a session can be inspected after the fact."""

    timestamp: datetime
    symbol: str
    action: str
    """The signal's action, or ``SKIPPED`` when no signal was produced."""
    approved: bool
    quantity: float
    order_status: str
    fill_price: float | None
    fees: float
    reason: str
    equity: float


@dataclass
class PaperSession:
    """Drives signals through risk, execution and the ledger, one at a time.

    The session owns no market data: it is handed a quote for each decision by
    its caller. That is what lets a test drive it from a dict and what keeps a
    network call out of the engine.
    """

    ledger: Ledger
    risk: RiskEngine
    simulator: ExecutionSimulator
    quotes: QuoteSource
    mode: ExecutionMode = ExecutionMode.PAPER
    session_id: str = ""
    """An optional caller-supplied id for the session, recorded in the state so a
    restored run can be shown to be the continuation of the same one."""
    equity_peak: float = field(default=0.0)
    day_start_equity: float | None = None
    day: str | None = None
    steps: list[SessionStep] = field(default_factory=list)
    resting: list[Order] = field(default_factory=list)
    """GTC LIMIT orders still live. Worked by :meth:`work_resting` against fresh
    quotes; a caller that never calls it leaves them resting, which is honest —
    an order nobody advanced is an order that did not fill."""
    processed_signal_ids: set[str] = field(default_factory=set)
    """Signal ids already driven through :meth:`on_signal`.

    A signal's id is a deterministic digest of its decision content, so replaying
    the same decision (a crash-recovery re-run, a duplicated message) yields the
    same id. The session refuses a second one rather than filling a second
    position nobody asked for. The refusal is explicit: the step records
    ``DUPLICATE_SIGNAL`` and no order is created. It is *not* silent, and it is
    *not* an exception, because a caller re-driving a known signal is a case the
    session must survive, not one that should abort it.
    """

    def __post_init__(self) -> None:
        # The mode is resolved through the same guard a backtest uses, so a
        # paper session cannot be constructed in a live mode either.
        self.mode = resolve_mode(self.mode)
        self.equity_peak = max(self.equity_peak, self.ledger.equity)
        if self.day_start_equity is None:
            self.day_start_equity = self.ledger.equity

    # --- Driving the session ------------------------------------------------

    def mark(self, quotes: Iterable[Quote]) -> None:
        """Mark the book to a set of quotes, so equity reflects current prices.

        Called before a decision, so the risk engine sizes against a current book
        rather than the last trade's prices.
        """
        for quote in quotes:
            self.ledger.mark(quote.symbol, quote.price)

    def on_signal(self, signal: TradingSignal) -> SessionStep:
        """Run one signal through risk, execution and the ledger.

        A HOLD, or a signal the risk engine refuses, produces a recorded step and
        no order. Nothing is sent anywhere in any case: the simulator fills
        against the supplied quote.

        A signal whose ``signal_id`` has already been driven is refused with a
        ``DUPLICATE_SIGNAL`` step and no order. The id is a deterministic digest
        of the decision, so this is what makes a re-driven decision — a crash
        recovery, a duplicated message — idempotent rather than a second position.
        """
        now = signal.timestamp
        if signal.signal_id in self.processed_signal_ids:
            duplicate = SessionStep(
                timestamp=now,
                symbol=signal.symbol,
                action=str(signal.action),
                approved=False,
                quantity=0.0,
                order_status="DUPLICATE_SIGNAL",
                fill_price=None,
                fees=0.0,
                reason=(
                    f"signal {signal.signal_id} was already processed; "
                    "refused to avoid a duplicate order"
                ),
                equity=self.ledger.equity,
            )
            self.steps.append(duplicate)
            return duplicate
        self._roll_day(now)
        quote = self.quotes(signal.symbol)
        if quote.symbol != signal.symbol:
            # A source that answered with the wrong symbol is a malformed feed, not
            # a price for this decision. Refuse rather than fill against it.
            raise PaperSessionError(
                f"quote source returned {quote.symbol!r} for a {signal.symbol!r} request"
            )
        # Marked only once the quote is in hand. A source that fails is a data
        # outage, not a processed decision, so the signal stays retryable rather
        # than being silently consumed by an error.
        self.processed_signal_ids.add(signal.signal_id)
        decision = self.risk.evaluate(
            signal,
            self.ledger,
            price=quote.price,
            equity_peak=self.equity_peak,
            day_start_equity=self.day_start_equity,
        )

        if signal.action is SignalAction.HOLD:
            return self._record(signal, decision, None, "no order: HOLD")

        if decision.verdict is not RiskVerdict.APPROVED:
            return self._record(signal, decision, None, decision.explain())

        # The simulator owns the ledger, so it applies the fill itself. The order
        # is built here rather than by the risk engine, so a decision stays a
        # decision and cannot place anything on its own.
        side = OrderSide.BUY if signal.action is SignalAction.BUY else OrderSide.SELL
        order_type = (
            OrderType.LIMIT if signal.order_intent is OrderIntent.LIMIT else OrderType.MARKET
        )
        time_in_force = (
            TimeInForce.GTC if signal.time_in_force is SignalTimeInForce.GTC
            else TimeInForce.DAY
        )
        order = Order(
            order_id=f"{signal.signal_id}-paper",
            symbol=signal.symbol,
            side=side,
            quantity=decision.quantity,
            order_type=order_type,
            limit_price=signal.limit_price,
            time_in_force=time_in_force,
            expires_session=signal.expires_session,
            created_at=now,
            as_of_date=signal.as_of_date,
            signal_id=signal.signal_id,
        )
        # Record creation before the fill-time re-check, so an order the re-check
        # refuses still has a CREATED event ahead of its ORDER_REJECTED. The
        # simulator skips its own creation record because of the flag.
        self.ledger.record_event(
            EventType.ORDER_CREATED,
            timestamp=now,
            symbol=order.symbol,
            detail={"order_id": order.order_id, "side": side.value,
                    "quantity": order.quantity, "type": order.order_type.value},
        )
        order.created_recorded = True
        # The caps were checked at the decision price. The fill gets a different
        # one — the quote has moved, or the costs moved it — so a buy is re-checked
        # at the price it will actually pay, exactly as the backtester does. A
        # sell reduces exposure and needs no re-check.
        if side is OrderSide.BUY:
            projected = self.simulator.projected_fill_price(order, market=quote.as_market())
            if projected is not None:
                review = self.risk.review_fill(
                    symbol=order.symbol,
                    action=SignalAction.BUY,
                    quantity=order.quantity,
                    ledger=self.ledger,
                    fill_price=projected,
                    max_notional=self.simulator.costs.affordable_notional(
                        cash=self.ledger.cash, price=projected
                    ),
                )
                if review.verdict is not RiskVerdict.APPROVED:
                    # The order was created before the re-check ran, so the refusal
                    # is recorded the same way the simulator records one: the order
                    # carries RISK_LIMIT and an ORDER_REJECTED event goes into the
                    # log. A refusal that left no trace would be invisible to the
                    # event log a report is built from.
                    detail = "refused at execution: " + (
                        review.explain() or "no size keeps the risk caps"
                    )
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = RejectReason.RISK_LIMIT
                    order.reject_detail = detail
                    self.ledger.record_event(
                        EventType.ORDER_REJECTED,
                        timestamp=order.created_at,
                        symbol=order.symbol,
                        detail={"order_id": order.order_id,
                                "reason": RejectReason.RISK_LIMIT.value},
                        reason=detail,
                    )
                    return self._record(signal, decision, None, detail)
                if review.quantity < order.quantity - 1e-12:
                    order = order.model_copy(update={"quantity": review.quantity})

        fees_before = self.ledger.total_fees
        rests = (
            order_type is OrderType.LIMIT and time_in_force is TimeInForce.GTC
        )
        filled = self.simulator.submit(
            order,
            market=quote.as_market(),
            rest_if_unfilled=rests,
            liquidity_budget=self._session_budget(quote, {}),
        )
        if filled.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED) and rests:
            if self._has_expired(filled, now):
                # The order's last eligible moment was before this signal, so it
                # may not be worked. It is cancelled with EXPIRED rather than
                # rested, and never filled on the session after it should have
                # died — the same bound the backtester enforces. This only applies
                # to a partial fill, since an unreached limit is returned ACCEPTED
                # before any fill; a partial fill that has already expired is
                # cancelled for its remainder.
                filled = self.simulator.cancel(
                    filled,
                    reason=RejectReason.EXPIRED,
                    detail=f"the resting order reached its expiry on {filled.expires_session}",
                    timestamp=now,
                )
            else:
                # A persistent limit the quote did not reach, or one that filled
                # only part of its size, stays live for a later :meth:`work_resting`
                # so its remainder is not silently dropped. Nothing terminal is
                # recorded, because nothing terminal happened.
                self.resting.append(filled)

        self.equity_peak = max(self.equity_peak, self.ledger.equity)
        reason = "" if filled.status.value == "FILLED" else (
            filled.reject_detail or filled.status.value
        )
        return self._record(signal, decision, filled, reason, fees_before=fees_before)

    def work_resting(self) -> list[SessionStep]:
        """Try every resting order against the current quotes.

        A caller that has just marked the book to fresh quotes calls this to
        advance the orders left from earlier signals. Each order is worked once
        against its symbol's current quote: it fills, partially fills, or stays
        resting. The returned steps describe what happened, in the order the
        orders were worked, so the outcome is deterministic.

        An order whose quote cannot be read is left resting, not cancelled and not
        dropped: no quote is not a reason to kill an order, and a source that
        raises is a data outage rather than a market fact. A quote that reports the
        market closed is treated the same way — a closed market is a reason to
        wait, not a terminal outcome for the order. The failure is isolated
        to that order — the remaining orders are still worked, and the session
        keeps its live state and its book — so a price-source exception cannot
        corrupt the book, duplicate an order, or leave a half-advanced queue. The
        error is re-raised after the loop, so it is never swallowed; the caller
        decides whether to retry. An order that has reached its ``expires_session``
        is cancelled with ``EXPIRED`` instead, exactly once.

        The day's opening equity is rolled from the quote's timestamp before the
        queue is worked, so a session whose only activity is resting fills still
        has a current daily-loss baseline.
        """
        if not self.resting:
            return []
        still_resting: list[Order] = []
        steps: list[SessionStep] = []
        failure: Exception | None = None
        # The session's shared units-of-volume allowance, per symbol, exactly as
        # the backtester computes it. It is consumed in queue order, so two orders
        # for one symbol cannot each fill the whole bar's worth: the first takes
        # what it needs and the second draws from what is left. Without this the
        # paper session would fill more than the backtest on the same data.
        budget: dict[str, float] = {}
        for order in self.resting:
            if order.status.is_terminal:
                continue
            try:
                quote = self.quotes(order.symbol)
            except (KeyError, LookupError, PaperSessionError):
                # The symbol has no quote this cycle. The order stays live; the
                # next work with a quote will advance it.
                still_resting.append(order)
                continue
            except Exception as exc:  # noqa: BLE001 - re-raised below, never swallowed
                # A price-source failure is not a market fact. Preserve the order
                # and the book, remember the failure, and keep working the rest so
                # one broken symbol cannot stall every other order.
                still_resting.append(order)
                failure = failure or exc
                continue

            self._roll_day(quote.timestamp)
            if not quote.is_open:
                # The market is closed: a reason to wait, not to kill the order. A
                # resting limit the venue is not currently pricing stays live, the
                # same as one whose quote is missing.
                still_resting.append(order)
                continue
            if self._has_expired(order, quote.timestamp):
                expired = self.simulator.cancel(
                    order,
                    reason=RejectReason.EXPIRED,
                    detail=f"the resting order reached its expiry on {order.expires_session}",
                    timestamp=quote.timestamp,
                )
                steps.append(self._resting_step(expired, quote, fees=0.0))
                continue

            if order.side is OrderSide.BUY:
                reviewed = self._review_resting_buy(order, quote)
                if reviewed is None:
                    # No positive size keeps the caps at the fill price: the
                    # remainder is rejected with RISK_LIMIT and recorded, exactly
                    # as an entry refused at the fill is. A gap must not breach a
                    # cap just because the order was resting.
                    detail = (
                        "refused at execution: no size keeps the risk caps at the resting fill price"
                    )
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = RejectReason.RISK_LIMIT
                    order.reject_detail = detail
                    self.ledger.record_event(
                        EventType.ORDER_REJECTED,
                        timestamp=quote.timestamp,
                        symbol=order.symbol,
                        detail={"order_id": order.order_id,
                                "reason": RejectReason.RISK_LIMIT.value},
                        reason=detail,
                    )
                    steps.append(self._resting_step(order, quote, fees=0.0))
                    continue
                order = reviewed

            fees_before = self.ledger.total_fees
            try:
                filled = self.simulator.submit(
                    order,
                    market=quote.as_market(),
                    rest_if_unfilled=True,
                    liquidity_budget=self._session_budget(quote, budget),
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below, never swallowed
                still_resting.append(order)
                failure = failure or exc
                continue
            if filled.filled_quantity > 0:
                budget[order.symbol] = max(
                    budget.get(order.symbol, 0.0) - filled.filled_quantity, 0.0
                )
            if filled.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                still_resting.append(filled)
            steps.append(
                self._resting_step(filled, quote, fees=self.ledger.total_fees - fees_before)
            )
        self.resting = still_resting
        self.steps.extend(steps)
        if failure is not None:
            # The book and the live orders are intact; the caller is told the
            # source failed so it can retry rather than believe the queue empty.
            raise failure
        return steps

    def _resting_step(self, order: Order, quote: Quote, *, fees: float) -> SessionStep:
        """The recorded outcome of working one resting order."""
        return SessionStep(
            timestamp=quote.timestamp,
            symbol=order.symbol,
            action="",
            approved=not order.status.is_terminal or order.status is OrderStatus.FILLED,
            quantity=float(order.filled_quantity),
            order_status=order.status.value,
            fill_price=float(order.average_fill_price)
            if order.average_fill_price else None,
            fees=max(fees, 0.0),
            reason=order.reject_detail or "",
            equity=self.ledger.equity,
        )

    def cancel_resting(self, order: Order, *, reason: RejectReason = RejectReason.EXPIRED,
                       detail: str = "cancelled by the caller") -> Order:
        """Cancel a resting order, recording the terminal event exactly once.

        The session-level counterpart of the simulator's ``cancel``: it removes the
        order from the live queue and, if the order is not already terminal,
        records one ``ORDER_CANCELLED`` event. Cancelling an order that is already
        terminal returns it unchanged, so a caller that cancels twice does not log
        two cancellations. A cancellation that removes the order from ``resting``
        is what makes "partial fill then cancel" leave no live remainder.
        """
        cancelled = self.simulator.cancel(order, reason=reason, detail=detail)
        self.resting = [o for o in self.resting if o is not order and o is not cancelled]
        return cancelled

    def _review_resting_buy(self, order: Order, quote: Quote) -> Order | None:
        """Re-check a resting BUY's size against the caps at the fill price.

        A resting order was sized when it was created, but the price that finally
        reaches its limit can have moved — a gap up to the limit, or costs. The
        caps are re-checked at the price the fill will actually get, exactly as an
        entry is at the moment of execution, so a resting order cannot leave a
        position past its cap. The size is reduced to the largest that holds; only
        when no positive size remains is the order refused (``None``). A SELL
        reduces exposure and is never re-checked.
        """
        market = quote.as_market()
        remaining = order.remaining_quantity
        if remaining <= 0:
            return order
        projected = self.simulator.projected_fill_price(
            order, market=market, quantity=remaining
        )
        if projected is None:
            # The order would not fill at all (limit not reached): leave it to
            # submit() to record the correct resting outcome.
            return order
        # The re-check is against the units this fill would *add*, so the headroom
        # is not double-counted by the units already filled on earlier sessions.
        review = self.risk.review_fill(
            symbol=order.symbol,
            action=SignalAction.BUY,
            quantity=remaining,
            ledger=self.ledger,
            fill_price=projected,
            max_notional=self.simulator.costs.affordable_notional(
                cash=self.ledger.cash, price=projected
            ),
        )
        if review.verdict is not RiskVerdict.APPROVED or review.quantity <= 0:
            return None
        if review.quantity < remaining - 1e-12:
            return order.model_copy(update={"quantity": order.filled_quantity + review.quantity})
        return order

    def _session_budget(self, quote: Quote, budget: dict[str, float]) -> float | None:
        """The units this symbol may still fill this session, shared across orders.

        Mirrors the backtester's per-session allowance: when participation is
        configured and the quote carries a volume, the symbol's allowance is its
        volume times the participation rate, computed once and drawn down as
        orders fill. ``None`` when participation is off or the volume is unknown,
        which is the default and means no liquidity limit. The running total in
        ``budget`` is what makes two orders for one symbol compete for one
        session's worth of volume rather than each claiming the whole bar.
        """
        participation = self.simulator.max_volume_participation
        volume = quote.average_daily_volume
        if not participation or not volume or volume <= 0:
            return None
        if quote.symbol not in budget:
            budget[quote.symbol] = volume * participation
        return budget[quote.symbol]

    def _has_expired(self, order: Order, moment: datetime) -> bool:
        """Whether a resting order has reached its expiry and must be cancelled.

        ``expires_session`` is the last session (``YYYY-MM-DD``) the order may be
        worked on, compared against the moment's UTC date, so an order is never
        filled on the session after it should have expired.
        """
        if order.expires_session is None:
            return False
        return moment.astimezone(UTC).date().isoformat() > order.expires_session

    def _record(self, signal, decision, order, reason: str, *, fees_before: float = 0.0) -> SessionStep:
        # The step reports what was actually filled when there was a fill, so a
        # report or a downstream reader cannot claim a larger position than the
        # ledger holds — which is exactly what a risk cap reduced at the fill
        # would otherwise cause. With no order, it reports the decision's size.
        if order is not None and order.filled_quantity > 0:
            quantity = float(order.filled_quantity)
        else:
            quantity = float(decision.quantity or 0.0)
        step = SessionStep(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            action=str(signal.action),
            approved=decision.verdict is RiskVerdict.APPROVED,
            quantity=quantity,
            order_status=order.status.value if order is not None else "NONE",
            fill_price=float(order.average_fill_price) if order is not None
            and order.average_fill_price else None,
            fees=max(self.ledger.total_fees - fees_before, 0.0),
            reason=reason,
            equity=self.ledger.equity,
        )
        self.steps.append(step)
        return step

    def _roll_day(self, now: datetime) -> None:
        """Reset the day's opening equity when the date advances.

        The clock only moves forward. A quote stamped before the current session
        is stale data, not a new day: rolling to it would move the daily-loss
        baseline backwards and let a late-arriving old quote reset the risk
        history. A quote on an earlier date than the session has seen is ignored
        for the clock, so the day and the baseline never go backwards.
        """
        today = now.astimezone(UTC).date().isoformat()
        if self.day is not None and today < self.day:
            return
        if self.day != today:
            self.day = today
            self.day_start_equity = self.ledger.equity

    # --- Reporting ----------------------------------------------------------

    def snapshot(self) -> dict:
        """The session's state and history, JSON-safe.

        This is a *reporting* view, not a recovery file. It is honest about what
        it shows — an order's status is the order's real status, so a resting
        order reads ``ACCEPTED`` and a partially filled one ``PARTIALLY_FILLED``,
        never ``FILLED`` — but it is deliberately a summary: it does not carry the
        ledger's event log, per-holding realised P&L, or the fee bookkeeping that
        a restart needs. :meth:`save_state` writes the durable, restorable state;
        this is for a human or a report to read.
        """
        ledger = self.ledger.snapshot()
        return {
            "schema_version": PAPER_STATE_SCHEMA_VERSION,
            "mode": self.mode.value,
            "session_id": self.session_id,
            "equity": self.ledger.equity,
            "cash": self.ledger.cash,
            "equity_peak": self.equity_peak,
            "realized_pnl": self.ledger.realized_pnl,
            "unrealized_pnl": self.ledger.unrealized_pnl,
            "total_fees": self.ledger.total_fees,
            "current_time": (
                self.steps[-1].timestamp.isoformat() if self.steps else None
            ),
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "processed_signal_count": len(self.processed_signal_ids),
            "resting_orders": [self._order_state(order) for order in self.resting],
            "positions": {
                symbol: {"quantity": holding.quantity,
                         "average_entry_price": holding.average_entry_price,
                         "market_value": holding.market_value}
                for symbol, holding in self.ledger.open_positions().items()
            },
            "steps": [
                {
                    "timestamp": step.timestamp.isoformat(),
                    "symbol": step.symbol,
                    "action": step.action,
                    "approved": step.approved,
                    "quantity": step.quantity,
                    "order_status": step.order_status,
                    "fill_price": step.fill_price,
                    "fees": step.fees,
                    "reason": step.reason,
                    "equity": step.equity,
                }
                for step in self.steps
            ],
            "ledger": {
                "timestamp": ledger.timestamp.isoformat(),
                "cash": ledger.cash,
                "equity": ledger.equity,
                "market_value": ledger.market_value,
                "total_exposure": ledger.total_exposure,
                "realized_pnl": ledger.realized_pnl,
                "unrealized_pnl": ledger.unrealized_pnl,
                "total_fees": ledger.total_fees,
                "positions": dict(ledger.positions),
                "prices": dict(ledger.prices),
            },
        }

    @staticmethod
    def _order_state(order: Order) -> dict:
        """One order as a JSON-safe dict, with its real lifecycle state.

        The full set is here because a reader must be able to tell ``FILLED`` from
        ``PARTIALLY_FILLED`` from ``ACCEPTED``: the filled and remaining
        quantities, the weighted average fill price, the time in force and the
        expiry are all reported, so the snapshot cannot overstate what happened.
        """
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "average_fill_price": order.average_fill_price,
            "limit_price": order.limit_price,
            "time_in_force": order.time_in_force.value,
            "expires_session": order.expires_session,
            "status": order.status.value,
            "reject_reason": order.reject_reason.value if order.reject_reason else None,
            "created_at": order.created_at.isoformat(),
            "signal_id": order.signal_id,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the session snapshot to ``path`` as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return target

    # --- Durable state and recovery -----------------------------------------

    def state(self) -> dict:
        """The complete, JSON-safe state needed to resume this session exactly.

        Distinct from :meth:`snapshot` on purpose. This is the recovery format: it
        holds everything a restart needs and nothing a report needs — the whole
        ledger (via ``Ledger.to_state``, including its event log and sequence
        counter), the risk limits, the cost schedule and simulator settings, the
        live resting orders with their remaining size and fill state, the
        processed-signal set that makes a re-driven decision idempotent, and the
        equity peak and day baseline. It carries a ``schema_version`` so a reader
        can refuse a file it does not understand rather than misread it.
        """
        return {
            "schema_version": PAPER_STATE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "mode": self.mode.value,
            "equity_peak": self.equity_peak,
            "day_start_equity": self.day_start_equity,
            "day": self.day,
            "risk_limits": self.risk.limits.model_dump(),
            "costs": self.simulator.costs.model_dump(),
            "simulator": {
                "mode": self.simulator.mode.value,
                "max_volume_participation": self.simulator.max_volume_participation,
            },
            "resting": [order.model_dump(mode="json") for order in self.resting],
            "processed_signal_ids": sorted(self.processed_signal_ids),
            "steps": [
                {
                    "timestamp": step.timestamp.isoformat(),
                    "symbol": step.symbol,
                    "action": step.action,
                    "approved": step.approved,
                    "quantity": step.quantity,
                    "order_status": step.order_status,
                    "fill_price": step.fill_price,
                    "fees": step.fees,
                    "reason": step.reason,
                    "equity": step.equity,
                }
                for step in self.steps
            ],
            "ledger": self.ledger.to_state(),
        }

    def save_state(self, path: str | Path) -> Path:
        """Write the durable session state to ``path`` as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def restore(
        cls,
        state: dict,
        *,
        quotes: QuoteSource,
        path: str | Path | None = None,
    ) -> PaperSession:
        """Rebuild a session from :meth:`state`, with identical accounting.

        ``quotes`` is supplied by the caller because a price source is a callable,
        not data: the state cannot carry a live callable, and inventing one would
        hide that the caller owns the prices. Everything else — the ledger, the
        risk limits, the cost schedule, the simulator settings, the resting orders,
        the processed signals and the day baseline — is restored, so the resumed
        session continues rather than restarts.

        ``path`` may be given instead of ``state`` to read the file written by
        :meth:`save_state`. A malformed file, an unknown ``schema_version``, or a
        state that is internally inconsistent raises :class:`PaperStateError`
        rather than restoring a book that would be silently wrong.
        """
        if path is not None:
            try:
                state = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PaperStateError(f"paper state file {path} is not usable: {exc}") from exc
        if not isinstance(state, dict):
            raise PaperStateError(f"paper state must be a mapping, got {type(state).__name__}")
        version = state.get("schema_version")
        if version != PAPER_STATE_SCHEMA_VERSION:
            raise PaperStateError(
                f"unsupported paper state schema_version {version!r}; "
                f"this build reads version {PAPER_STATE_SCHEMA_VERSION}"
            )
        try:
            ledger = Ledger.from_state(state["ledger"])
            limits = RiskLimits(**state.get("risk_limits", {}))
            costs = CostModel(**state.get("costs", {}))
            simulator_state = state.get("simulator", {})
            simulator = ExecutionSimulator(
                ledger,
                costs=costs,
                mode=simulator_state.get("mode", state.get("mode", ExecutionMode.PAPER.value)),
                max_volume_participation=simulator_state.get("max_volume_participation", 0.0),
            )
            session = cls(
                ledger=ledger,
                risk=RiskEngine(limits),
                simulator=simulator,
                quotes=quotes,
                mode=state.get("mode", ExecutionMode.PAPER.value),
                session_id=state.get("session_id", ""),
                equity_peak=float(state.get("equity_peak", ledger.equity)),
                day_start_equity=state.get("day_start_equity"),
                day=state.get("day"),
                resting=[Order.model_validate(o) for o in state.get("resting", [])],
                processed_signal_ids=set(state.get("processed_signal_ids", [])),
            )
        except PaperStateError:
            raise
        except Exception as exc:  # noqa: BLE001 - one error type for a bad state
            raise PaperStateError(f"paper state could not be restored: {exc}") from exc
        session.steps = [
            SessionStep(
                timestamp=datetime.fromisoformat(step["timestamp"]),
                symbol=step["symbol"],
                action=step["action"],
                approved=bool(step["approved"]),
                quantity=float(step["quantity"]),
                order_status=step["order_status"],
                fill_price=step["fill_price"],
                fees=float(step["fees"]),
                reason=step["reason"],
                equity=float(step["equity"]),
            )
            for step in state.get("steps", [])
        ]
        return session


def dict_quote_source(quotes: dict[str, float], *, timestamp: datetime | None = None,
                      is_open: bool = True) -> QuoteSource:
    """A quote source backed by a dict. The default for tests and notebooks."""
    moment = timestamp or datetime.now(UTC)

    def source(symbol: str) -> Quote:
        if symbol not in quotes:
            raise PaperSessionError(f"no quote is available for {symbol!r}")
        return Quote(symbol=symbol, price=float(quotes[symbol]), timestamp=moment, is_open=is_open)

    return source


def new_session(
    *,
    initial_cash: float,
    quotes: QuoteSource,
    limits=None,
    costs=None,
) -> PaperSession:
    """Build a paper session with the standard components.

    A convenience, not a second implementation: it wires the same ledger, risk
    engine and simulator a backtest uses.
    """
    ledger = Ledger(initial_cash=initial_cash)
    return PaperSession(
        ledger=ledger,
        risk=RiskEngine(limits),
        simulator=ExecutionSimulator(ledger, costs=costs or CostModel(), mode=ExecutionMode.PAPER),
        quotes=quotes,
    )
