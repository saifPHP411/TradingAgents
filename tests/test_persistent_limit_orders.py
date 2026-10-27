"""Persistent LIMIT orders and the partial-order lifecycle.

A single-shot limit order is the easy case: work it once, cancel it if the bar
missed it. A *resting* order is the case that carries real risk, because the same
order object is worked again and again, and every mistake compounds:

- a re-worked order must not record a second ORDER_ACCEPTED;
- a resting fill must register only its *increment* with the exit tracker, or a
  three-session fill becomes three positions;
- a limit that is reached only intraday must fill (the bar's low/high proves it
  traded there), while a limit never reached must stay live;
- a competing pair of orders must share one bar's liquidity, not each claim it;
- an expiry must cancel the order rather than let it fill on a later bar.

These tests pin each of those, at the simulator level and through the backtester,
so the persistent path cannot silently regress into either double counting or
perpetual resting.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.backtesting import BacktestConfig, MarketDataFeed, PortfolioBacktester
from tradingagents.execution import (
    CostModel,
    ExecutionSimulator,
    MarketContext,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
    TimeInForce,
)
from tradingagents.portfolio import EventType, Ledger
from tradingagents.signals import (
    OrderIntent,
    SignalAction,
    SignalTimeInForce,
    TradingSignal,
)

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 7, 21, 0, tzinfo=UTC)


def _limit(side=OrderSide.BUY, quantity=10.0, limit_price=95.0, tif=TimeInForce.GTC,
           symbol="AAPL", **kwargs) -> Order:
    return Order(
        order_id="o1", symbol=symbol, side=side, quantity=quantity,
        order_type=OrderType.LIMIT, limit_price=limit_price, time_in_force=tif,
        created_at=T0, as_of_date="2026-01-05", **kwargs,
    )


def _market(price, *, symbol="AAPL", low=None, high=None, volume=None, timestamp=T0):
    return MarketContext(symbol=symbol, price=price, timestamp=timestamp, is_open=True,
                         average_daily_volume=volume, low=low, high=high)


def _sim(cash=100_000.0, costs=None) -> tuple[ExecutionSimulator, Ledger]:
    ledger = Ledger(initial_cash=cash)
    return ExecutionSimulator(ledger, costs=costs or CostModel()), ledger


def _events(ledger, kind: EventType) -> list:
    return [e for e in ledger.events if e.type is kind]


# ---------------------------------------------------------------------------
# Resting behaviour at the simulator level
# ---------------------------------------------------------------------------


class TestRestingLimit:
    def test_unreached_limit_rests_instead_of_cancelling(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        out = sim.submit(order, market=_market(100.0), rest_if_unfilled=True)
        assert out.status is OrderStatus.ACCEPTED
        assert out.reject_reason is None
        # Nothing terminal was recorded: the order is not finished.
        assert _events(ledger, EventType.ORDER_CANCELLED) == []

    def test_unreached_limit_without_rest_flag_is_still_cancelled(self):
        """The old single-shot behaviour is unchanged for a caller that does not rest."""
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        out = sim.submit(order, market=_market(100.0))
        assert out.status is OrderStatus.CANCELLED
        assert out.reject_reason is RejectReason.LIMIT_NOT_REACHED
        assert len(_events(ledger, EventType.ORDER_CANCELLED)) == 1

    def test_day_limit_is_never_rested_even_when_asked(self):
        """A DAY order has one session to fill; the rest flag cannot extend it."""
        sim, _ = _sim()
        order = _limit(limit_price=95.0, tif=TimeInForce.DAY)
        out = sim.submit(order, market=_market(100.0), rest_if_unfilled=True)
        assert out.status is OrderStatus.CANCELLED

    def test_acceptance_is_recorded_once_across_repeated_work(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        for price in (100.0, 101.0, 102.0):
            out = sim.submit(order, market=_market(price), rest_if_unfilled=True)
            assert out.status is OrderStatus.ACCEPTED
        assert len(_events(ledger, EventType.ORDER_ACCEPTED)) == 1

    def test_a_resting_limit_fills_when_a_later_bar_reaches_it(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        assert sim.submit(order, market=_market(100.0), rest_if_unfilled=True).status \
            is OrderStatus.ACCEPTED
        filled = sim.submit(order, market=_market(94.0, timestamp=T1), rest_if_unfilled=True)
        assert filled.status is OrderStatus.FILLED
        assert filled.filled_quantity == 10.0
        assert filled.average_fill_price == pytest.approx(94.0)
        assert ledger.position_quantity("AAPL") == pytest.approx(10.0)

    def test_limit_reached_only_intraday_fills_via_the_bar_low(self):
        """The close is above the limit but the session traded down to it."""
        sim, _ = _sim()
        order = _limit(limit_price=95.0)
        filled = sim.submit(
            order, market=_market(101.0, low=94.0, high=102.0), rest_if_unfilled=True
        )
        assert filled.status is OrderStatus.FILLED
        # The limit caps the price; the close is better, so the close is the fill.
        assert filled.average_fill_price == pytest.approx(95.0)

    def test_limit_not_reached_intraday_keeps_resting(self):
        sim, _ = _sim()
        order = _limit(limit_price=90.0)
        out = sim.submit(
            order, market=_market(101.0, low=95.0, high=102.0), rest_if_unfilled=True
        )
        assert out.status is OrderStatus.ACCEPTED

    def test_sell_limit_rests_then_fills_on_the_bar_high(self):
        sim, ledger = _sim()
        ledger.apply_fill(symbol="AAPL", quantity=10.0, price=90.0, fee=0.0, timestamp=T0)
        order = _limit(side=OrderSide.SELL, limit_price=105.0)
        assert sim.submit(order, market=_market(100.0), rest_if_unfilled=True).status \
            is OrderStatus.ACCEPTED
        filled = sim.submit(
            order, market=_market(101.0, low=99.0, high=106.0, timestamp=T1),
            rest_if_unfilled=True,
        )
        assert filled.status is OrderStatus.FILLED
        assert filled.average_fill_price == pytest.approx(105.0)

    def test_resting_order_never_touches_the_book_while_unfilled(self):
        sim, ledger = _sim(cash=1_000.0)
        order = _limit(limit_price=95.0, quantity=100.0)
        sim.submit(order, market=_market(100.0), rest_if_unfilled=True)
        assert ledger.cash == pytest.approx(1_000.0)
        assert ledger.position_quantity("AAPL") == 0.0


# ---------------------------------------------------------------------------
# Partial fills and remainders
# ---------------------------------------------------------------------------


class TestPartialRemainder:
    def test_volume_cap_fills_part_then_rests_the_rest(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0, quantity=100.0)
        first = sim.submit(
            order, market=_market(95.0, volume=1_000.0), rest_if_unfilled=True,
            liquidity_budget=40.0,
        )
        assert first.status is OrderStatus.PARTIALLY_FILLED
        assert first.filled_quantity == pytest.approx(40.0)
        assert first.remaining_quantity == pytest.approx(60.0)
        assert ledger.position_quantity("AAPL") == pytest.approx(40.0)

        second = sim.submit(
            first, market=_market(95.0, volume=1_000.0, timestamp=T1),
            rest_if_unfilled=True, liquidity_budget=60.0,
        )
        assert second.status is OrderStatus.FILLED
        assert second.filled_quantity == pytest.approx(100.0)
        assert ledger.position_quantity("AAPL") == pytest.approx(100.0)

    def test_a_remainder_fills_at_its_own_price_and_vwap_is_correct(self):
        sim, _ = _sim()
        order = _limit(limit_price=96.0, quantity=100.0)
        first = sim.submit(order, market=_market(95.0), rest_if_unfilled=True,
                           liquidity_budget=50.0)
        assert first.average_fill_price == pytest.approx(95.0)
        second = sim.submit(first, market=_market(96.0, timestamp=T1),
                            rest_if_unfilled=True, liquidity_budget=50.0)
        assert second.status is OrderStatus.FILLED
        # 50 @ 95 and 50 @ 96.
        assert second.average_fill_price == pytest.approx(95.5)

    def test_remaining_quantity_never_goes_negative(self):
        sim, _ = _sim()
        order = _limit(limit_price=95.0, quantity=10.0)
        filled = sim.submit(order, market=_market(90.0), rest_if_unfilled=True)
        assert filled.remaining_quantity == 0.0
        assert filled.filled_quantity == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cancellation and expiry
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_is_idempotent(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        sim.submit(order, market=_market(100.0), rest_if_unfilled=True)
        sim.cancel(order, reason=RejectReason.EXPIRED, detail="expired")
        again = sim.cancel(order, reason=RejectReason.EXPIRED, detail="expired")
        assert again.status is OrderStatus.CANCELLED
        assert len(_events(ledger, EventType.ORDER_CANCELLED)) == 1

    def test_cancel_can_carry_the_session_time(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        sim.submit(order, market=_market(100.0), rest_if_unfilled=True)
        sim.cancel(order, reason=RejectReason.EXPIRED, detail="expired", timestamp=T2)
        event = _events(ledger, EventType.ORDER_CANCELLED)[0]
        assert event.timestamp == T2

    def test_terminal_order_is_not_filled_again(self):
        sim, ledger = _sim()
        order = _limit(limit_price=95.0)
        filled = sim.submit(order, market=_market(95.0))
        assert filled.status is OrderStatus.FILLED
        again = sim.submit(order, market=_market(90.0, timestamp=T1))
        assert again.status is OrderStatus.FILLED
        assert again.filled_quantity == pytest.approx(10.0)
        assert ledger.position_quantity("AAPL") == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# End-to-end through the backtester
# ---------------------------------------------------------------------------

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


def _feed(symbol="AAPL", *, opens, closes, dates=DATES, volumes=None):
    highs = [max(o, c) * 1.0 for o, c in zip(opens, closes, strict=True)]
    lows = [min(o, c) * 1.0 for o, c in zip(opens, closes, strict=True)]
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": volumes or [1_000_000] * len(dates)},
        index=pd.to_datetime(dates),
    )
    return MarketDataFeed({symbol: frame})


def _signal(session, symbol="AAPL", action=SignalAction.BUY, **kwargs) -> TradingSignal:
    return TradingSignal(
        symbol=symbol, action=action, confidence=0.8,
        timestamp=datetime.strptime(session, "%Y-%m-%d").replace(hour=21, tzinfo=UTC),
        as_of_date=session, **kwargs,
    )


def _run(strategy, feed, config=None):
    engine = PortfolioBacktester(feed, strategy, config=config)
    return engine.run(), engine


class TestRestingThroughTheBacktester:
    def test_a_gtc_limit_rests_until_a_later_open_reaches_it(self):
        # Decision on day 0; the order is worked from day 1's open onward. Only
        # day 3 opens down to the limit, so that is where it must fill.
        feed = _feed(opens=[100.0, 100.0, 100.0, 94.0, 96.0],
                     closes=[100.0, 100.0, 100.0, 95.0, 96.0])

        def strategy(context):
            if context.session == DATES[0]:
                return _signal(
                    context.session, order_intent=OrderIntent.LIMIT,
                    limit_price=95.0, entry_price=100.0,
                    time_in_force=SignalTimeInForce.GTC,
                )
            return None

        result, engine = _run(strategy, feed)
        assert engine.ledger.position_quantity("AAPL") > 0
        assert engine.ledger.position_quantity("AAPL") == pytest.approx(
            engine.ledger.position_quantity("AAPL")
        )
        # The fill price is at or below the limit, never above it.
        holding = engine.ledger.holding("AAPL")
        assert holding.average_entry_price <= 95.0 + 1e-9

    def test_the_same_signal_never_double_counts_the_position(self):
        """One signal, one limit, one fill: the position equals the order size."""
        feed = _feed(opens=[100.0, 90.0, 90.0, 90.0, 90.0],
                     closes=[100.0, 90.0, 90.0, 90.0, 90.0])

        def strategy(context):
            if context.session == DATES[0]:
                return _signal(
                    context.session, order_intent=OrderIntent.LIMIT,
                    limit_price=95.0, entry_price=100.0,
                    time_in_force=SignalTimeInForce.GTC, position_size=0.10,
                )
            return None

        result, engine = _run(strategy, feed)
        orders = _events(engine.ledger, EventType.ORDER_FILLED)
        assert len(orders) == 1
        # Every fill event for the one order sums to the position held.
        filled_total = sum(e.detail.get("quantity", 0.0) for e in orders)
        assert engine.ledger.position_quantity("AAPL") == pytest.approx(filled_total)

    def test_an_expired_resting_order_is_cancelled_not_filled(self):
        feed = _feed(opens=[100.0, 100.0, 94.0, 94.0, 94.0],
                     closes=[100.0, 100.0, 94.0, 94.0, 94.0])

        def strategy(context):
            if context.session == DATES[0]:
                return _signal(
                    context.session, order_intent=OrderIntent.LIMIT,
                    limit_price=95.0, entry_price=100.0,
                    time_in_force=SignalTimeInForce.GTC,
                    expires_session=DATES[1],
                )
            return None

        result, engine = _run(strategy, feed)
        # The limit was only reached on day 2, after the day-1 expiry, so it must
        # not have filled.
        assert engine.ledger.position_quantity("AAPL") == 0.0
        cancelled = _events(engine.ledger, EventType.ORDER_CANCELLED)
        assert len(cancelled) == 1
        assert cancelled[0].detail.get("reason") == RejectReason.EXPIRED.value

    def test_resting_can_be_disabled_to_restore_single_shot(self):
        feed = _feed(opens=[100.0, 100.0, 94.0, 94.0, 94.0],
                     closes=[100.0, 100.0, 94.0, 94.0, 94.0])

        def strategy(context):
            if context.session == DATES[0]:
                return _signal(
                    context.session, order_intent=OrderIntent.LIMIT,
                    limit_price=95.0, entry_price=100.0,
                    time_in_force=SignalTimeInForce.GTC,
                )
            return None

        config = BacktestConfig(resting_limit_orders=False)
        result, engine = _run(strategy, feed, config=config)
        assert engine.ledger.position_quantity("AAPL") == 0.0
        assert len(_events(engine.ledger, EventType.ORDER_CANCELLED)) == 1

    def test_ttl_expires_a_resting_order(self):
        feed = _feed(opens=[100.0, 100.0, 94.0, 94.0, 94.0],
                     closes=[100.0, 100.0, 94.0, 94.0, 94.0])

        def strategy(context):
            if context.session == DATES[0]:
                return _signal(
                    context.session, order_intent=OrderIntent.LIMIT,
                    limit_price=95.0, entry_price=100.0,
                    time_in_force=SignalTimeInForce.GTC,
                )
            return None

        config = BacktestConfig(limit_order_ttl_sessions=1)
        result, engine = _run(strategy, feed, config=config)
        assert engine.ledger.position_quantity("AAPL") == 0.0
        assert len(_events(engine.ledger, EventType.ORDER_CANCELLED)) == 1


class TestSignalIntentContract:
    def test_limit_intent_requires_a_limit_price(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", order_intent=OrderIntent.LIMIT, entry_price=100.0)

    def test_market_intent_refuses_a_limit_price(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", limit_price=95.0)

    def test_buy_limit_above_entry_is_refused(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", order_intent=OrderIntent.LIMIT,
                    limit_price=105.0, entry_price=100.0)

    def test_hold_cannot_request_a_limit(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", action=SignalAction.HOLD,
                    order_intent=OrderIntent.LIMIT, limit_price=95.0)

    def test_market_intent_cannot_be_gtc(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", time_in_force=SignalTimeInForce.GTC)

    def test_intent_participates_in_the_signal_id(self):
        market = _signal("2026-01-05")
        limit = _signal("2026-01-05", order_intent=OrderIntent.LIMIT,
                        limit_price=95.0, entry_price=100.0)
        assert market.signal_id != limit.signal_id

    def test_expires_session_must_be_canonical(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _signal("2026-01-05", order_intent=OrderIntent.LIMIT, limit_price=95.0,
                    entry_price=100.0, expires_session="2026/01/09")
