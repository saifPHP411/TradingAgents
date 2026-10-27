"""The portfolio backtester: an event loop where the book, not a rating, is the result.

The old harness answered "was the rating right?" by measuring an asset's return
after N days. That cannot answer "what would this have made?", because it has no
cash, no position, no fee and no overlap: twenty decisions in one week are
twenty independent outcomes, not one book.

This engine runs the loop the task calls for:

    market event -> AI decision -> risk validation -> order -> execution
                 -> portfolio update -> PnL -> next event

with one rule that shapes everything: **an order executes on the session after
the decision, at that session's open.** The decision was made from data through
the decision date's close, so filling at that same close would trade on the
price the decision already saw. Filling at the next session's open is the first
price a trader could actually have got.

Scheduling, in full, because it is the part that is easy to get subtly wrong:

- **Decision** on session ``T``: the strategy sees bars up to and including ``T``.
- **Order** created at ``T``'s close.
- **Execution** at the **open** of the next session ``T+1``. If there is no
  ``T+1`` in the data, the order expires unfilled rather than executing at a
  price that does not exist.
- **Exit** for a position whose holding period has elapsed happens at the open of
  the session ``holding_period_days`` after entry. A stop or target is checked
  against each intervening session's high/low and, if touched, fills at the level
  itself — the conservative assumption, since the bar cannot say whether the
  level was reached before or after the extreme.
- **Valuation** marks the book at each session's close, carrying forward the last
  known close for a symbol that did not trade that session.

Overlap is modelled, not avoided: two decisions about the same symbol on
consecutive sessions both create orders, the second one adds to the position, and
the risk engine's exposure cap is what stops the book growing without bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from tradingagents.backtesting.data import MarketDataFeed, SessionClock
from tradingagents.execution import (
    CostModel,
    ExecutionMode,
    ExecutionSimulator,
    MarketContext,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from tradingagents.portfolio import Ledger
from tradingagents.portfolio.events import EventType
from tradingagents.risk import RiskEngine, RiskLimits, RiskReason, RiskVerdict
from tradingagents.signals.contract import SignalAction, TradingSignal


@dataclass
class BacktestConfig:
    """Everything that shapes a run, so the run is reproducible from it alone."""

    initial_cash: float = 100_000.0
    currency: str = "USD"
    holding_period_days: int = 5
    """Sessions between an entry and its scheduled exit."""
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    costs: CostModel = field(default_factory=CostModel)
    max_volume_participation: float = 0.0
    """Share of a bar's volume a market order may take; 0 disables partial fills."""
    use_stops: bool = True
    """Whether a signal's stop_loss and take_profit are enforced against each session."""
    execute_on_next_open: bool = True
    """The look-ahead guard. False would fill at the decision bar's close, which is the
    price the decision already saw; it exists only to let a test demonstrate the
    difference, and is documented as unsafe."""

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {self.initial_cash}")
        if self.holding_period_days < 1:
            raise ValueError(f"holding_period_days must be at least 1, got {self.holding_period_days}")


@dataclass
class PendingOrder:
    """An order waiting for the next session's open."""

    order: Order
    signal: TradingSignal
    created_session: str


@dataclass
class OpenPosition:
    """A position the engine is tracking for its exit."""

    symbol: str
    quantity: float
    entry_price: float
    entry_session: str
    sessions_held: int
    """How many sessions have elapsed since entry, counted in the feed's calendar."""
    holding_period_days: int
    """Sessions to hold before the scheduled exit."""
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_fees: float = 0.0

    @property
    def holding_period_elapsed(self) -> bool:
        """Whether enough sessions have passed for the scheduled exit.

        Counted in sessions rather than by comparing dates, so a position is not
        closed early merely because the run's data ends: a five-session hold with
        only three sessions left is still open when the data runs out.
        """
        return self.sessions_held >= self.holding_period_days


@dataclass
class Trade:
    """One completed round trip, for the trade-level statistics."""

    symbol: str
    entry_session: str
    exit_session: str
    quantity: float
    entry_price: float
    exit_price: float
    fees: float
    reason: str

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost else 0.0


@dataclass
class EquityPoint:
    """One mark of the book at a session's close."""

    session: str
    equity: float
    cash: float
    market_value: float
    exposure: float


@dataclass
class BacktestResult:
    """What a portfolio backtest produced, ready for the performance engine."""

    run_id: str
    sessions: list[str] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    signals: list[TradingSignal] = field(default_factory=list)
    rejections: list[tuple[str, str, str]] = field(default_factory=list)
    """(session, symbol, reason) for every signal the risk engine refused."""
    unfilled: list[tuple[str, str, str]] = field(default_factory=list)
    """(session, symbol, reason) for every order that did not fill."""
    ledger: Ledger | None = None
    config: BacktestConfig | None = None
    initial_cash: float = 0.0
    decisions: int = 0

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1].equity if self.equity_curve else self.initial_cash

    def equity_series(self) -> pd.Series:
        """The equity curve as a pandas Series indexed by session, for the metrics."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(
            [p.equity for p in self.equity_curve],
            index=pd.Index([p.session for p in self.equity_curve], name="session"),
            dtype=float,
        )


# A strategy turns the data visible as of a session into a signal for it. The
# feed passed in is already sliced, so a strategy cannot see the future even if
# it tries; it is given the bars, not the engine's internals.
Strategy = Callable[["StrategyContext"], TradingSignal | None]


@dataclass
class StrategyContext:
    """What a strategy is given for one session: data through that date, and the book.

    The history frames are cut at ``session`` before being handed over, so a
    strategy that reads ``context.history(symbol)`` receives only past bars. It is
    given the ledger read-only for context (how much is held, how much cash is
    free) but no way to move it: orders are the engine's business.
    """

    session: str
    feed: MarketDataFeed
    ledger: Ledger
    clock: SessionClock
    symbols: tuple[str, ...]

    def history(self, symbol: str) -> pd.DataFrame:
        """Bars for ``symbol`` through this session inclusive. Never a later bar."""
        return self.feed.history(symbol, self.session)

    def bar(self, symbol: str):
        """This session's bar for ``symbol``, or ``None`` if it did not trade."""
        return self.feed.bar(symbol, self.session)

    def close(self, symbol: str) -> float | None:
        """This session's close, or ``None``."""
        bar = self.bar(symbol)
        return bar.close if bar is not None else None

    def position(self, symbol: str) -> float:
        return self.ledger.position_quantity(symbol)


class PortfolioBacktester:
    """Runs a strategy over a feed, moving a ledger through the full event loop."""

    def __init__(
        self,
        feed: MarketDataFeed,
        strategy: Strategy,
        *,
        config: BacktestConfig | None = None,
        run_id: str = "backtest",
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.run_id = run_id
        self.ledger = Ledger(initial_cash=self.config.initial_cash, currency=self.config.currency)
        self.execution = ExecutionSimulator(
            self.ledger,
            costs=self.config.costs,
            mode=ExecutionMode.BACKTEST,
            max_volume_participation=self.config.max_volume_participation,
        )
        self.risk = RiskEngine(self.config.risk_limits)
        self.result = BacktestResult(
            run_id=run_id, ledger=self.ledger, config=self.config,
            initial_cash=self.config.initial_cash,
        )
        self._pending: list[PendingOrder] = []
        self._open: dict[str, OpenPosition] = {}
        self._equity_peak = self.config.initial_cash
        self._session_start_equity = self.config.initial_cash

    # --- The loop -----------------------------------------------------------

    def run(self, start: str | None = None, end: str | None = None) -> BacktestResult:
        """Run every session in ``[start, end]`` and return the result."""
        sessions = self.feed.calendar(start, end)
        if not sessions:
            raise ValueError(f"no sessions between {start} and {end}")
        for session in sessions:
            self._run_session(session)
        return self.result

    def _run_session(self, session: str) -> None:
        """One session: fill yesterday's orders, act on exits, decide, then mark."""
        self._session_start_equity = self.ledger.equity

        # 1. Orders created on the previous session execute at this session's open.
        self._execute_pending(session)
        # 2. Positions whose holding period elapsed, or whose stop/target was
        #    touched, exit at this session's open.
        self._process_exits(session)
        # 3. The strategy sees data through this session and may decide.
        self._decide(session)
        # 4. Mark the book at this session's close and record the equity point.
        self._mark(session)

    # --- Execution ----------------------------------------------------------

    def _execute_pending(self, session: str) -> None:
        """Fill orders scheduled for this session, then drop them."""
        pending, self._pending = self._pending, []
        for item in pending:
            bar = self.feed.bar(item.order.symbol, session)
            if bar is None:
                # The market for this symbol did not trade today. The order
                # expires rather than filling at a price that never existed.
                self.result.unfilled.append(
                    (session, item.order.symbol, "no bar on the execution session")
                )
                item.order.status = OrderStatus.CANCELLED
                continue
            market = MarketContext(
                symbol=item.order.symbol,
                price=bar.open,
                timestamp=_session_open_time(session),
                is_open=True,
                average_daily_volume=bar.volume,
            )
            filled = self.execution.submit(item.order, market=market)
            if filled.status is OrderStatus.FILLED:
                self._register_entry(item, filled, session)
            elif filled.status is OrderStatus.PARTIALLY_FILLED:
                self._register_entry(item, filled, session)
                self.result.unfilled.append(
                    (session, filled.symbol,
                     f"partial fill: {filled.filled_quantity} of {filled.quantity}")
                )
            else:
                self.result.unfilled.append(
                    (session, filled.symbol, filled.reject_detail or str(filled.reject_reason))
                )

    def _register_entry(self, item: PendingOrder, order: Order, session: str) -> None:
        """Record a filled entry so the position can be exited later."""
        existing = self._open.get(order.symbol)
        quantity = order.filled_quantity
        price = order.average_fill_price or 0.0
        fees = self.config.costs.commission(quantity=quantity, price=price)
        if existing is None:
            self._open[order.symbol] = OpenPosition(
                symbol=order.symbol,
                quantity=quantity,
                entry_price=price,
                entry_session=session,
                sessions_held=0,
                holding_period_days=self.config.holding_period_days,
                stop_loss=item.signal.stop_loss if self.config.use_stops else None,
                take_profit=item.signal.take_profit if self.config.use_stops else None,
                entry_fees=fees,
            )
        else:
            # Adding to a position blends the entry, matching the ledger's own
            # average, and keeps the earlier exit schedule.
            total = existing.quantity + quantity
            existing.entry_price = (
                existing.entry_price * existing.quantity + price * quantity
            ) / total
            existing.quantity = total
            existing.entry_fees += fees

    def _exit_session_after(self, session: str) -> str:
        """The session ``holding_period_days`` after ``session``, or the last available.

        Only used for reporting a position's intended exit; the loop itself counts
        sessions, so this cannot cause a premature exit when the data ends early.
        """
        cursor = session
        for _ in range(self.config.holding_period_days):
            nxt = self.feed.next_session(cursor)
            if nxt is None:
                return cursor
            cursor = nxt
        return cursor

    # --- Exits --------------------------------------------------------------

    def _process_exits(self, session: str) -> None:
        """Exit positions whose time is up, or whose stop or target was touched."""
        for symbol, position in list(self._open.items()):
            if position.entry_session == session:
                # Entered at this session's open, so no full session has elapsed
                # yet; its clock starts on the next session.
                continue
            bar = self.feed.bar(symbol, session)
            if bar is None:
                # The symbol did not trade; its clock does not advance either, so
                # a five-session hold is five of the symbol's own sessions.
                continue
            position.sessions_held += 1
            trigger = self._exit_trigger(position, bar, session)
            if trigger is None:
                continue
            price, reason = trigger
            self._close_position(position, session, price, reason)

    def _exit_trigger(
        self, position: OpenPosition, bar, session: str
    ) -> tuple[float, str] | None:
        """The price and reason to exit at, or ``None`` to keep holding.

        Stops and targets are checked before the scheduled exit, and both fill at
        the level rather than at the bar's open, because the level is the price
        the order was placed to get. A gap through the level fills at the open,
        which is the honest outcome: the stop could not have been honoured at its
        own price if the session opened beyond it.
        """
        if position.stop_loss is not None and bar.low <= position.stop_loss:
            # Filling at the open when it gapped below the stop: the level could
            # not have been honoured at its own price.
            price = min(bar.open, position.stop_loss)
            return price, "stop_loss"
        if position.take_profit is not None and bar.high >= position.take_profit:
            price = max(bar.open, position.take_profit)
            return price, "take_profit"
        if position.holding_period_elapsed:
            return bar.open, "holding_period"
        return None

    def _close_position(self, position: OpenPosition, session: str, price: float, reason: str) -> None:
        """Sell the position at ``price`` and record the round trip."""
        order = Order(
            order_id=f"exit-{position.symbol}-{session}",
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            created_at=_session_open_time(session),
            as_of_date=session,
        )
        market = MarketContext(
            symbol=position.symbol,
            price=price,
            timestamp=_session_open_time(session),
            is_open=True,
            average_daily_volume=None,
        )
        filled = self.execution.submit(order, market=market)
        if filled.status is not OrderStatus.FILLED:
            # The book refused the exit (a rounding artefact, say). Leaving the
            # position open is safer than forcing a state the ledger rejects.
            self.result.unfilled.append(
                (session, position.symbol, f"exit failed: {filled.reject_detail}")
            )
            return
        exit_price = filled.average_fill_price or price
        exit_fee = self.config.costs.commission(quantity=position.quantity, price=exit_price)
        self.result.trades.append(
            Trade(
                symbol=position.symbol,
                entry_session=position.entry_session,
                exit_session=session,
                quantity=position.quantity,
                entry_price=position.entry_price,
                exit_price=exit_price,
                fees=position.entry_fees + exit_fee,
                reason=reason,
            )
        )
        del self._open[position.symbol]

    # --- Decisions ----------------------------------------------------------

    def _decide(self, session: str) -> None:
        """Ask the strategy for a signal, validate it, and schedule the order."""
        context = StrategyContext(
            session=session,
            feed=self.feed,
            ledger=self.ledger,
            clock=SessionClock.for_session(session),
            symbols=self.feed.symbols,
        )
        signal = self.strategy(context)
        if signal is None:
            return
        self.result.decisions += 1
        self.result.signals.append(signal)

        price = self.feed.bar(signal.symbol, session)
        reference = price.close if price is not None else None
        decision = self.risk.evaluate(
            signal,
            self.ledger,
            price=reference,
            equity_peak=max(self._equity_peak, self.ledger.equity),
            day_start_equity=self._session_start_equity,
        )
        if decision.verdict is RiskVerdict.REJECTED:
            # A HOLD is not a rejection: it is a decision to do nothing, and
            # recording it as a refusal would make the run look like it was
            # blocked when it simply chose not to act.
            if RiskReason.NOT_ACTIONABLE not in decision.reasons:
                self.result.rejections.append(
                    (session, signal.symbol, ",".join(r.value for r in decision.reasons))
                )
            return
        if not decision.is_order:
            return

        self._schedule_order(signal, decision.quantity, session)

    def _schedule_order(self, signal: TradingSignal, quantity: float, session: str) -> None:
        """Create the order and queue it for the next session's open."""
        side = OrderSide.BUY if signal.action is SignalAction.BUY else OrderSide.SELL
        order = Order(
            order_id=f"{signal.signal_id}-{session}",
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            created_at=signal.timestamp,
            as_of_date=signal.as_of_date,
            signal_id=signal.signal_id,
        )
        self.ledger.record_event(
            EventType.ORDER_ACCEPTED, timestamp=signal.timestamp, symbol=signal.symbol,
            detail={"order_id": order.order_id, "quantity": quantity},
        )
        if not self.config.execute_on_next_open:
            # Documented as unsafe: fills at the price the decision already saw.
            bar = self.feed.bar(signal.symbol, session)
            if bar is not None:
                filled = self.execution.submit(
                    order,
                    market=MarketContext(symbol=signal.symbol, price=bar.close,
                                         timestamp=_session_close_time(session),
                                         is_open=True, average_daily_volume=bar.volume),
                )
                if filled.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self._register_entry(
                        PendingOrder(order=order, signal=signal, created_session=session),
                        filled, session,
                    )
                return
        self._pending.append(PendingOrder(order=order, signal=signal, created_session=session))

    # --- Valuation ----------------------------------------------------------

    def _mark(self, session: str) -> None:
        """Mark every holding at this session's close and record the equity point.

        A symbol that did not trade today keeps its last known close, which is the
        only defensible valuation: marking it to zero would invent a loss, and
        marking it to a later price would be look-ahead.
        """
        for symbol in list(self.ledger.holdings):
            bar = self.feed.bar(symbol, session) or self.feed.latest_bar_at_or_before(symbol, session)
            if bar is not None:
                self.ledger.mark(symbol, bar.close)
        snapshot = self.ledger.snapshot(_session_close_time(session))
        self._equity_peak = max(self._equity_peak, snapshot.equity)
        self.result.sessions.append(session)
        self.result.equity_curve.append(
            EquityPoint(
                session=session,
                equity=snapshot.equity,
                cash=snapshot.cash,
                market_value=snapshot.market_value,
                exposure=snapshot.total_exposure,
            )
        )

    # --- Introspection ------------------------------------------------------

    @property
    def open_positions(self) -> dict[str, OpenPosition]:
        """The positions still being tracked for an exit at the end of the run."""
        return dict(self._open)

    def force_close_all(self, session: str) -> None:
        """Exit every open position at ``session``'s close, for a run that must end flat.

        Off by default: a backtest that ends with positions open should report
        them, not pretend they were sold. A caller that wants the realised figure
        calls this explicitly.
        """
        for symbol, position in list(self._open.items()):
            bar = self.feed.bar(symbol, session) or self.feed.latest_bar_at_or_before(symbol, session)
            if bar is None:
                continue
            self._close_position(position, session, bar.close, "forced_close")


def _session_open_time(session: str) -> datetime:
    """14:30 UTC on ``session``: the US cash open, in a fixed offset."""
    parsed = datetime.strptime(session, "%Y-%m-%d").replace(hour=14, minute=30)
    return parsed.replace(tzinfo=UTC)


def _session_close_time(session: str) -> datetime:
    """21:00 UTC on ``session``: the US cash close, in a fixed offset."""
    parsed = datetime.strptime(session, "%Y-%m-%d").replace(hour=21, minute=0)
    return parsed.replace(tzinfo=UTC)


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EquityPoint",
    "OpenPosition",
    "PendingOrder",
    "PortfolioBacktester",
    "Strategy",
    "StrategyContext",
    "Trade",
    "RiskReason",
]
