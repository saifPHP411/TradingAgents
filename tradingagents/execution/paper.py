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
)
from tradingagents.execution.simulator import ExecutionSimulator
from tradingagents.portfolio.events import EventType
from tradingagents.portfolio.ledger import Ledger
from tradingagents.risk.engine import RiskEngine, RiskVerdict
from tradingagents.signals.contract import SignalAction, TradingSignal


class PaperSessionError(RuntimeError):
    """A paper session was driven in a way it cannot support."""


@dataclass(frozen=True)
class Quote:
    """A price observation for one symbol at one moment.

    ``timestamp`` is the observation's own time, not the time it was read, so a
    stale quote can be detected by the caller rather than silently filled.
    """

    symbol: str
    price: float
    timestamp: datetime
    is_open: bool = True
    average_daily_volume: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise PaperSessionError(f"quote price must be finite and positive, got {self.price}")

    def as_market(self) -> MarketContext:
        return MarketContext(
            symbol=self.symbol,
            price=self.price,
            timestamp=self.timestamp,
            is_open=self.is_open,
            average_daily_volume=self.average_daily_volume,
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
    equity_peak: float = field(default=0.0)
    day_start_equity: float | None = None
    day: str | None = None
    steps: list[SessionStep] = field(default_factory=list)

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
        """
        now = signal.timestamp
        self._roll_day(now)
        quote = self.quotes(signal.symbol)

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
        order = Order(
            order_id=f"{signal.signal_id}-paper",
            symbol=signal.symbol,
            side=side,
            quantity=decision.quantity,
            order_type=OrderType.MARKET,
            created_at=now,
            as_of_date=signal.as_of_date,
            signal_id=signal.signal_id,
        )
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
        filled = self.simulator.submit(order, market=quote.as_market())

        self.equity_peak = max(self.equity_peak, self.ledger.equity)
        reason = "" if filled.status.value == "FILLED" else (
            filled.reject_detail or filled.status.value
        )
        return self._record(signal, decision, filled, reason, fees_before=fees_before)

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
        """Reset the day's opening equity when the date advances."""
        today = now.astimezone(UTC).date().isoformat()
        if self.day != today:
            self.day = today
            self.day_start_equity = self.ledger.equity

    # --- Reporting ----------------------------------------------------------

    def snapshot(self) -> dict:
        """The session's state and history, JSON-safe."""
        ledger = self.ledger.snapshot()
        return {
            "mode": self.mode.value,
            "equity": self.ledger.equity,
            "cash": self.ledger.cash,
            "equity_peak": self.equity_peak,
            "realized_pnl": self.ledger.realized_pnl,
            "unrealized_pnl": self.ledger.unrealized_pnl,
            "total_fees": self.ledger.total_fees,
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

    def write_json(self, path: str | Path) -> Path:
        """Write the session snapshot to ``path`` as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return target


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
    from tradingagents.execution.costs import CostModel

    ledger = Ledger(initial_cash=initial_cash)
    return PaperSession(
        ledger=ledger,
        risk=RiskEngine(limits),
        simulator=ExecutionSimulator(ledger, costs=costs or CostModel(), mode=ExecutionMode.PAPER),
        quotes=quotes,
    )
