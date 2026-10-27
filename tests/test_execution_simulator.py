"""The simulated execution engine: fills, rejections, partial fills, and the mode guard.

The engine is the last place a mistake becomes money, so these tests are mostly
about what it refuses. Two things matter most:

- the mode guard, because a live path must be unreachable, not merely unused;
- the rejection vocabulary, because "the order did not fill" is not an answer a
  backtest can report — "the limit was never reached" and "the cash ran out" are.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tradingagents.execution import (
    CostModel,
    ExecutionMode,
    ExecutionSimulator,
    LiveTradingNotSupportedError,
    MarketContext,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
    assert_simulation_only,
    resolve_mode,
)
from tradingagents.execution.orders import Order as _Order
from tradingagents.portfolio import EventType, Ledger

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)


def _order(side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, limit_price=None,
           symbol="AAPL") -> Order:
    return Order(
        order_id="o1", symbol=symbol, side=side, quantity=quantity, order_type=order_type,
        limit_price=limit_price, created_at=T0, as_of_date="2026-01-05",
    )


def _market(price=100.0, *, symbol="AAPL", is_open=True, volume=None, timestamp=T0) -> MarketContext:
    return MarketContext(symbol=symbol, price=price, timestamp=timestamp, is_open=is_open,
                         average_daily_volume=volume)


def _sim(cash=10_000.0, costs=None, **kwargs) -> tuple[ExecutionSimulator, Ledger]:
    ledger = Ledger(initial_cash=cash)
    return ExecutionSimulator(ledger, costs=costs, **kwargs), ledger


# ---------------------------------------------------------------------------
# The mode guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoLiveTrading:
    def test_only_two_modes_exist(self):
        assert {m.value for m in ExecutionMode} == {"BACKTEST", "PAPER"}

    def test_there_is_no_live_member(self):
        for member in ExecutionMode:
            assert "LIVE" not in member.value

    @pytest.mark.parametrize("asked", ["live", "LIVE", "real", "production", "prod",
                                       "broker", "exchange", "mainnet", " Live "])
    def test_a_live_mode_is_refused_loudly(self, asked):
        with pytest.raises(LiveTradingNotSupportedError):
            resolve_mode(asked)

    def test_a_simulation_mode_resolves(self):
        assert resolve_mode("backtest") is ExecutionMode.BACKTEST
        assert resolve_mode("PAPER") is ExecutionMode.PAPER
        assert resolve_mode(ExecutionMode.BACKTEST) is ExecutionMode.BACKTEST

    def test_an_unknown_mode_is_a_value_error(self):
        with pytest.raises(ValueError):
            resolve_mode("nonsense")

    def test_none_is_not_a_mode(self):
        """Defaulting an unspecified mode would hide which mode ran."""
        with pytest.raises(ValueError):
            resolve_mode(None)

    def test_the_assertion_helper_rejects_a_non_simulation(self):
        assert_simulation_only(ExecutionMode.BACKTEST)
        assert_simulation_only(ExecutionMode.PAPER)
        with pytest.raises(LiveTradingNotSupportedError):
            assert_simulation_only("LIVE")

    def test_the_simulator_refuses_to_be_constructed_live(self):
        with pytest.raises(LiveTradingNotSupportedError):
            ExecutionSimulator(Ledger(1_000.0), mode="live")

    def test_the_simulator_records_its_mode(self):
        sim, _ = _sim()
        assert sim.mode is ExecutionMode.BACKTEST
        sim, _ = _sim(mode=ExecutionMode.PAPER)
        assert sim.mode is ExecutionMode.PAPER


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarketOrders:
    def test_a_market_buy_fills_and_moves_the_book(self):
        sim, ledger = _sim()
        filled = sim.submit(_order(), market=_market(100.0))
        assert filled.status is OrderStatus.FILLED
        assert filled.filled_quantity == 10
        assert filled.average_fill_price == 100.0
        assert ledger.position_quantity("AAPL") == 10
        assert ledger.cash == pytest.approx(9_000.0)

    def test_a_market_sell_reduces_the_book(self):
        sim, ledger = _sim()
        sim.submit(_order(), market=_market(100.0))
        sold = sim.submit(_order(side=OrderSide.SELL, quantity=4), market=_market(110.0, timestamp=T1))
        assert sold.status is OrderStatus.FILLED
        assert ledger.position_quantity("AAPL") == 6
        assert ledger.realized_pnl == pytest.approx(40.0)

    def test_a_market_buy_with_costs_pays_more_than_the_mid(self):
        costs = CostModel(half_spread_bps=10.0, slippage_bps=10.0, commission_rate=0.001)
        sim, ledger = _sim(costs=costs)
        filled = sim.submit(_order(), market=_market(100.0))
        assert filled.average_fill_price > 100.0
        assert ledger.total_fees > 0.0

    def test_the_order_events_are_recorded_in_order(self):
        sim, ledger = _sim()
        sim.submit(_order(), market=_market(100.0))
        types = [e.type for e in ledger.events]
        assert types[0] is EventType.ORDER_CREATED
        assert EventType.ORDER_FILLED in types
        assert types.index(EventType.ORDER_CREATED) < types.index(EventType.ORDER_FILLED)


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLimitOrders:
    def test_a_buy_limit_above_the_market_fills_at_the_better_price(self):
        """A limit caps the price; it does not set it."""
        sim, ledger = _sim()
        filled = sim.submit(
            _order(order_type=OrderType.LIMIT, limit_price=105.0),
            market=_market(100.0),
        )
        assert filled.status is OrderStatus.FILLED
        assert filled.average_fill_price == pytest.approx(100.0)

    def test_a_buy_limit_below_the_market_does_not_fill(self):
        sim, ledger = _sim()
        filled = sim.submit(
            _order(order_type=OrderType.LIMIT, limit_price=95.0),
            market=_market(100.0),
        )
        assert filled.status is OrderStatus.REJECTED
        assert filled.reject_reason is RejectReason.LIMIT_NOT_REACHED
        assert ledger.position_quantity("AAPL") == 0.0
        assert ledger.cash == 10_000.0

    def test_a_buy_limit_at_the_market_fills_at_the_limit(self):
        sim, _ = _sim()
        filled = sim.submit(
            _order(order_type=OrderType.LIMIT, limit_price=100.0), market=_market(100.0)
        )
        assert filled.status is OrderStatus.FILLED
        assert filled.average_fill_price == pytest.approx(100.0)

    def test_a_sell_limit_below_the_market_fills_at_the_better_price(self):
        sim, ledger = _sim()
        sim.submit(_order(), market=_market(100.0))
        filled = sim.submit(
            _order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=95.0),
            market=_market(110.0, timestamp=T1),
        )
        assert filled.status is OrderStatus.FILLED
        assert filled.average_fill_price == pytest.approx(110.0)

    def test_a_sell_limit_above_the_market_does_not_fill(self):
        sim, ledger = _sim()
        sim.submit(_order(), market=_market(100.0))
        filled = sim.submit(
            _order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=120.0),
            market=_market(110.0, timestamp=T1),
        )
        assert filled.status is OrderStatus.REJECTED
        assert filled.reject_reason is RejectReason.LIMIT_NOT_REACHED
        assert ledger.position_quantity("AAPL") == 10

    def test_a_limit_order_without_a_limit_price_is_a_construction_error(self):
        with pytest.raises(ValidationError):
            _order(order_type=OrderType.LIMIT, limit_price=None)

    def test_a_market_order_with_a_limit_price_is_a_construction_error(self):
        with pytest.raises(ValidationError):
            _order(order_type=OrderType.MARKET, limit_price=100.0)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRejections:
    def test_a_closed_market_is_rejected(self):
        sim, ledger = _sim()
        filled = sim.submit(_order(), market=_market(100.0, is_open=False))
        assert filled.status is OrderStatus.REJECTED
        assert filled.reject_reason is RejectReason.MARKET_CLOSED
        assert ledger.cash == 10_000.0

    def test_a_missing_price_is_rejected_as_no_price_not_as_closed(self):
        sim, _ = _sim()
        filled = sim.submit(_order(), market=_market(None))
        assert filled.reject_reason is RejectReason.NO_PRICE

    def test_a_non_positive_price_is_rejected(self):
        sim, _ = _sim()
        filled = sim.submit(_order(), market=_market(0.0))
        assert filled.reject_reason is RejectReason.INVALID_PRICE

    def test_insufficient_cash_is_rejected_and_changes_nothing(self):
        sim, ledger = _sim(cash=500.0)
        filled = sim.submit(_order(quantity=10), market=_market(100.0))
        assert filled.status is OrderStatus.REJECTED
        assert filled.reject_reason is RejectReason.INSUFFICIENT_CASH
        assert ledger.cash == 500.0
        assert ledger.position_quantity("AAPL") == 0.0

    def test_selling_more_than_held_is_rejected(self):
        """Long-only: the engine must not let an oversized sell become a short."""
        sim, ledger = _sim()
        sim.submit(_order(quantity=5), market=_market(100.0))
        filled = sim.submit(
            _order(side=OrderSide.SELL, quantity=6), market=_market(100.0, timestamp=T1)
        )
        assert filled.reject_reason is RejectReason.INSUFFICIENT_POSITION
        assert ledger.position_quantity("AAPL") == 5

    def test_selling_a_symbol_never_held_is_rejected(self):
        sim, _ = _sim()
        filled = sim.submit(
            _order(side=OrderSide.SELL, symbol="TSLA"), market=_market(100.0, symbol="TSLA")
        )
        assert filled.reject_reason is RejectReason.INSUFFICIENT_POSITION

    def test_every_rejection_is_recorded_as_an_event(self):
        sim, ledger = _sim(cash=100.0)
        sim.submit(_order(quantity=10), market=_market(100.0))
        rejects = [e for e in ledger.events if e.type is EventType.ORDER_REJECTED]
        assert len(rejects) == 1
        assert rejects[0].detail["reason"] == RejectReason.INSUFFICIENT_CASH.value

    def test_a_rejection_leaves_no_fill_event(self):
        sim, ledger = _sim(cash=100.0)
        sim.submit(_order(quantity=10), market=_market(100.0))
        assert not [e for e in ledger.events if e.type is EventType.ORDER_FILLED]

    def test_a_rejection_names_the_symbol_and_a_reason(self):
        sim, _ = _sim(cash=100.0)
        filled = sim.submit(_order(quantity=10), market=_market(100.0))
        assert filled.symbol == "AAPL"
        assert filled.reject_detail


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialFills:
    def test_participation_is_disabled_by_default(self):
        sim, ledger = _sim()
        filled = sim.submit(_order(quantity=1_000), market=_market(10.0, volume=100))
        assert filled.status is OrderStatus.FILLED
        assert ledger.position_quantity("AAPL") == 1_000

    def test_a_large_order_fills_only_the_permitted_share(self):
        sim, ledger = _sim(max_volume_participation=0.1)
        filled = sim.submit(_order(quantity=1_000), market=_market(10.0, volume=1_000))
        assert filled.status is OrderStatus.PARTIALLY_FILLED
        assert filled.filled_quantity == pytest.approx(100.0)
        assert filled.remaining_quantity == pytest.approx(900.0)
        assert ledger.position_quantity("AAPL") == pytest.approx(100.0)

    def test_a_small_order_still_fills_completely(self):
        sim, _ = _sim(max_volume_participation=0.1)
        filled = sim.submit(_order(quantity=50), market=_market(10.0, volume=1_000))
        assert filled.status is OrderStatus.FILLED

    def test_no_volume_means_no_participation_cap(self):
        sim, _ = _sim(max_volume_participation=0.1)
        filled = sim.submit(_order(quantity=1_000), market=_market(10.0, volume=None))
        assert filled.status is OrderStatus.FILLED

    def test_a_limit_order_is_not_partially_filled(self):
        """A daily bar cannot say how much volume traded at the limit."""
        sim, _ = _sim(max_volume_participation=0.1)
        filled = sim.submit(
            _order(quantity=1_000, order_type=OrderType.LIMIT, limit_price=11.0),
            market=_market(10.0, volume=1_000),
        )
        assert filled.status is OrderStatus.FILLED
        assert filled.filled_quantity == 1_000

    def test_an_out_of_range_participation_is_refused(self):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError):
                ExecutionSimulator(Ledger(1_000.0), max_volume_participation=bad)


# ---------------------------------------------------------------------------
# Contract and accounting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContract:
    def test_a_symbol_mismatch_between_order_and_market_is_a_programming_error(self):
        sim, _ = _sim()
        with pytest.raises(ValueError, match="market context"):
            sim.submit(_order(symbol="AAPL"), market=_market(100.0, symbol="MSFT"))

    def test_quantity_is_always_positive_in_an_order(self):
        with pytest.raises(ValidationError):
            _order(quantity=-5)

    def test_filled_quantity_cannot_exceed_quantity(self):
        with pytest.raises(ValidationError):
            _Order(order_id="x", symbol="AAPL", side=OrderSide.BUY, quantity=10,
                   created_at=T0, as_of_date="2026-01-05", filled_quantity=11)

    def test_the_simulator_satisfies_the_execution_protocol(self):
        from tradingagents.execution import ExecutionEngine

        sim, _ = _sim()
        assert isinstance(sim, ExecutionEngine)


@pytest.mark.unit
class TestAccountingThroughExecution:
    def test_costs_are_in_the_ledger_not_only_in_a_report(self):
        costs = CostModel(commission_flat=5.0, half_spread_bps=20.0)
        sim, ledger = _sim(costs=costs)
        sim.submit(_order(quantity=10), market=_market(100.0))
        assert ledger.total_fees == pytest.approx(5.0)
        # cash = 10_000 - (10 * fill) - 5, and fill > 100 because of the spread
        assert ledger.cash < 10_000.0 - 1_000.0 - 5.0

    def test_a_round_trip_through_the_engine_loses_the_costs(self):
        costs = CostModel(commission_rate=0.001, half_spread_bps=5.0)
        sim, ledger = _sim(cash=10_000.0, costs=costs)
        sim.submit(_order(quantity=10), market=_market(100.0))
        sim.submit(_order(side=OrderSide.SELL, quantity=10), market=_market(100.0, timestamp=T1))
        assert ledger.position_quantity("AAPL") == 0.0
        # A flat book at an unchanged mid must be down by exactly what the round
        # trip cost: two commissions plus the half-spread paid on each leg. The
        # spread is the part that a fee-only model would miss.
        half_spread_leg = 10 * 100.0 * (5.0 / 10_000.0)
        expected_loss = ledger.total_fees + 2 * half_spread_leg
        assert ledger.equity == pytest.approx(10_000.0 - expected_loss)
        assert ledger.total_pnl == pytest.approx(-expected_loss)
        assert ledger.total_pnl < 0

    def test_the_cash_conservation_identity_holds_through_the_engine(self):
        costs = CostModel(commission_flat=1.0, half_spread_bps=10.0)
        sim, ledger = _sim(costs=costs)
        sim.submit(_order(quantity=10), market=_market(100.0))
        sim.submit(_order(quantity=5), market=_market(101.0, timestamp=T1))
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert (
            ledger.cash + basis + ledger.total_fees - ledger.realized_pnl
            == pytest.approx(10_000.0)
        )


# ---------------------------------------------------------------------------
# Non-finite reference prices and re-submission
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonFiniteReferencePrice:
    """A non-finite bar price must be rejected, not filled.

    Regression test for a real defect. ``_pre_trade_rejection`` guarded with
    ``market.price <= 0``, which is false for a NaN, so a NaN bar price reached
    the cost model, came back as a NaN fill price, and the ledger applied it —
    the order reported ``FILLED`` and cash and equity became NaN. Infinity took
    the same path and produced an infinite fill. Both must be a rejection.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_market_price_is_rejected(self, bad):
        sim, ledger = _sim()
        order = sim.submit(_order(quantity=10), market=_market(bad))
        assert order.status is OrderStatus.REJECTED
        assert order.reject_reason is RejectReason.INVALID_PRICE

    def test_a_non_finite_price_leaves_the_book_finite_and_untouched(self):
        sim, ledger = _sim()
        sim.submit(_order(quantity=10), market=_market(float("nan")))
        assert ledger.cash == pytest.approx(10_000.0)
        assert ledger.position_quantity("AAPL") == 0.0
        assert ledger.equity == pytest.approx(10_000.0)


@pytest.mark.unit
class TestAnOrderFillsAtMostOnce:
    """Re-submitting a terminal order must not fill it again.

    Regression test for a real defect: ``submit`` re-ran the whole lifecycle on
    an order that had already filled, so the same order object moved the ledger
    twice. Nothing in the ledger can tell a re-submission from a second genuine
    order, so the guard has to live here. The second call returns the first
    outcome unchanged.
    """

    def test_a_filled_order_is_not_filled_again(self):
        sim, ledger = _sim()
        order = _order(quantity=10)
        first = sim.submit(order, market=_market(100.0))
        assert first.status is OrderStatus.FILLED
        second = sim.submit(order, market=_market(100.0))
        assert second.status is OrderStatus.FILLED
        assert ledger.position_quantity("AAPL") == pytest.approx(10.0)
        assert ledger.cash == pytest.approx(9_000.0)

    def test_a_rejected_order_is_not_re_evaluated(self):
        sim, ledger = _sim(cash=100.0)
        order = _order(quantity=10)
        first = sim.submit(order, market=_market(100.0))
        assert first.status is OrderStatus.REJECTED
        # A later, more favourable attempt must still be refused: the order is
        # already terminal, and the ledger must not move.
        sim.submit(order, market=_market(1.0))
        assert ledger.position_quantity("AAPL") == 0.0
        assert ledger.cash == pytest.approx(100.0)
