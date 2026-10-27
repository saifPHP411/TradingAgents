"""Execution costs: commission, spread and slippage, and their determinism.

The point of charging costs in the ledger is that a result cannot be quoted gross
by accident. These tests check that each cost is applied, that each always works
against the trade, and that the same inputs always produce the same numbers —
a stochastic cost model would make the ledger untestable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingagents.execution.costs import NO_COSTS, CostModel, Side


@pytest.mark.unit
class TestCommission:
    def test_zero_by_default(self):
        assert NO_COSTS.commission(quantity=100, price=50.0) == 0.0

    def test_a_flat_fee(self):
        model = CostModel(commission_flat=1.0)
        assert model.commission(quantity=100, price=50.0) == pytest.approx(1.0)

    def test_a_per_unit_fee(self):
        model = CostModel(commission_per_unit=0.005)
        assert model.commission(quantity=100, price=50.0) == pytest.approx(0.5)

    def test_a_notional_rate(self):
        model = CostModel(commission_rate=0.001)
        # 100 * 50 = 5000 notional, 0.1% = 5
        assert model.commission(quantity=100, price=50.0) == pytest.approx(5.0)

    def test_the_rates_are_additive(self):
        model = CostModel(commission_flat=1.0, commission_per_unit=0.01, commission_rate=0.001)
        # 1.0 + 1.0 + 5.0
        assert model.commission(quantity=100, price=50.0) == pytest.approx(7.0)

    def test_a_minimum_applies(self):
        model = CostModel(commission_rate=0.0001, commission_min=2.0)
        # 0.05 would be the rate; the minimum wins
        assert model.commission(quantity=100, price=50.0) == pytest.approx(2.0)

    def test_a_minimum_above_the_rate_still_applies(self):
        model = CostModel(commission_flat=5.0, commission_min=1.0)
        assert model.commission(quantity=10, price=10.0) == pytest.approx(5.0)

    def test_commission_is_signed_independent(self):
        """A sell pays the same commission as a buy of the same size."""
        model = CostModel(commission_rate=0.001)
        assert model.commission(quantity=100, price=50.0) == model.commission(quantity=-100, price=50.0)

    def test_a_minimum_without_a_rate_is_rejected(self):
        """A minimum that no rate can reach is a configuration error."""
        with pytest.raises(ValidationError, match="commission_min"):
            CostModel(commission_min=1.0)


@pytest.mark.unit
class TestSpread:
    def test_zero_by_default(self):
        assert NO_COSTS.spread_cost(quantity=100, price=50.0) == 0.0

    def test_the_half_spread_is_a_cost(self):
        model = CostModel(half_spread_bps=10.0)  # 0.1%
        assert model.spread_cost(quantity=100, price=50.0) == pytest.approx(5.0)

    def test_spread_is_never_a_rebate(self):
        """A negative quantity is a sell; the spread still costs."""
        model = CostModel(half_spread_bps=10.0)
        assert model.spread_cost(quantity=-100, price=50.0) > 0


@pytest.mark.unit
class TestSlippage:
    def test_base_slippage_alone_when_no_volume_is_known(self):
        model = CostModel(slippage_bps=5.0)
        assert model.slippage_bps_for(quantity=100, price=50.0, average_daily_volume=None) == 5.0

    def test_participation_adds_slippage(self):
        model = CostModel(slippage_bps=5.0, slippage_volume_factor=1.0)
        # 100 of a 1,000-share day is 10% participation -> +10 bps
        assert model.slippage_bps_for(
            quantity=100, price=50.0, average_daily_volume=1_000
        ) == pytest.approx(15.0)

    def test_a_zero_volume_is_treated_as_unknown(self):
        model = CostModel(slippage_bps=5.0, slippage_volume_factor=1.0)
        assert model.slippage_bps_for(quantity=100, price=50.0, average_daily_volume=0) == 5.0

    def test_slippage_is_capped(self):
        model = CostModel(slippage_bps=10.0, slippage_volume_factor=1000.0, max_slippage_bps=50.0)
        assert model.slippage_bps_for(
            quantity=100, price=50.0, average_daily_volume=100
        ) == pytest.approx(50.0)


@pytest.mark.unit
class TestFillPrice:
    def test_a_buy_fills_at_or_above_the_mid(self):
        model = CostModel(half_spread_bps=10.0, slippage_bps=5.0)
        fill, _, _ = model.fill_price(side=Side.BUY, reference_price=100.0, quantity=10)
        assert fill > 100.0
        assert fill == pytest.approx(100.0 * (1 + 15 / 10_000))

    def test_a_sell_fills_at_or_below_the_mid(self):
        model = CostModel(half_spread_bps=10.0, slippage_bps=5.0)
        fill, _, _ = model.fill_price(side=Side.SELL, reference_price=100.0, quantity=10)
        assert fill < 100.0

    def test_no_costs_fill_at_the_reference(self):
        fill, spread, slippage = NO_COSTS.fill_price(
            side=Side.BUY, reference_price=100.0, quantity=10
        )
        assert fill == 100.0
        assert spread == 0.0 and slippage == 0.0

    def test_the_adverse_move_always_favours_the_market(self):
        model = CostModel(half_spread_bps=50.0, slippage_bps=50.0)
        for side in (Side.BUY, Side.SELL):
            fill, _, _ = model.fill_price(side=side, reference_price=100.0, quantity=1)
            assert (fill - 100.0) * (1 if side is Side.BUY else -1) > 0

    def test_a_non_positive_reference_price_is_refused(self):
        with pytest.raises(ValueError):
            NO_COSTS.fill_price(side=Side.BUY, reference_price=0.0, quantity=1)

    def test_costs_that_would_zero_a_sell_price_are_refused(self):
        model = CostModel(half_spread_bps=20_000.0)
        with pytest.raises(ValueError, match="reduce the configured"):
            model.fill_price(side=Side.SELL, reference_price=100.0, quantity=1)

    def test_the_breakdown_is_reported(self):
        model = CostModel(half_spread_bps=10.0, slippage_bps=10.0)
        _, spread, slippage = model.fill_price(side=Side.BUY, reference_price=100.0, quantity=10)
        assert spread == pytest.approx(1.0)
        assert slippage == pytest.approx(1.0)


@pytest.mark.unit
class TestDeterminism:
    def test_the_same_inputs_give_the_same_fill_price(self):
        model = CostModel(half_spread_bps=7.0, slippage_bps=3.0, slippage_volume_factor=0.5)
        args = {"side": Side.BUY, "reference_price": 123.45, "quantity": 37,
                "average_daily_volume": 5_000}
        first = model.fill_price(**args)
        for _ in range(20):
            assert model.fill_price(**args) == first

    def test_costs_are_independent_of_the_order_they_were_asked_about_before(self):
        model = CostModel(commission_rate=0.001, half_spread_bps=5.0)
        assert model.commission(quantity=10, price=100.0) == model.commission(quantity=10, price=100.0)


@pytest.mark.unit
class TestCostModelValidation:
    def test_extra_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            CostModel(taker_fee=0.1)

    def test_negative_rates_are_rejected(self):
        for field in ("commission_flat", "half_spread_bps", "slippage_bps"):
            with pytest.raises(ValidationError):
                CostModel(**{field: -1.0})

    def test_the_breakdown_total_sums_the_parts(self):
        from tradingagents.execution.costs import CostBreakdown

        breakdown = CostBreakdown(commission=1.0, spread=2.0, slippage=3.0)
        assert breakdown.total == pytest.approx(6.0)


@pytest.mark.unit
class TestMakerTakerFees:
    """A configured maker/taker rate must actually be charged.

    Regression test for a real defect: ``maker_fee_bps`` and ``taker_fee_bps``
    existed as fields and were documented, but no code path ever read them, so a
    schedule that set them collected nothing. The execution path now charges
    ``fee_for_fill``, which adds the rate on top of the commission.
    """

    def test_a_market_fill_pays_the_taker_rate(self):
        model = CostModel(taker_fee_bps=10.0, maker_fee_bps=2.0)
        assert model.fee_for_fill(quantity=100, price=100.0, is_maker=False) == pytest.approx(10.0)

    def test_a_limit_fill_pays_the_maker_rate(self):
        model = CostModel(taker_fee_bps=10.0, maker_fee_bps=2.0)
        assert model.fee_for_fill(quantity=100, price=100.0, is_maker=True) == pytest.approx(2.0)

    def test_the_rate_is_added_to_the_commission(self):
        model = CostModel(commission_flat=1.0, taker_fee_bps=10.0)
        assert model.fee_for_fill(quantity=100, price=100.0) == pytest.approx(11.0)

    def test_the_default_schedule_charges_nothing_extra(self):
        assert NO_COSTS.fee_for_fill(quantity=100, price=100.0) == 0.0


@pytest.mark.unit
class TestAffordableNotional:
    """``affordable_notional`` is the fee-inclusive cash a buy can spend.

    Regression test for a real defect: an order sized to exactly the available
    cash was rejected at the fill once the commission (or a gap) pushed the
    all-in cost past the cash, instead of being reduced to what the cash covered.
    """

    def test_no_fee_makes_the_whole_cash_affordable(self):
        assert NO_COSTS.affordable_notional(cash=1000.0, price=100.0) == pytest.approx(1000.0)

    def test_a_commission_rate_leaves_headroom_for_the_fee(self):
        model = CostModel(commission_rate=0.001)
        notional = model.affordable_notional(cash=100_000.0, price=100.0)
        assert notional + model.fee_for_fill(quantity=notional / 100.0, price=100.0) <= 100_000.0 + 1e-6

    def test_a_flat_fee_is_covered(self):
        model = CostModel(commission_flat=5.0)
        notional = model.affordable_notional(cash=1000.0, price=100.0)
        assert notional == pytest.approx(995.0)

    def test_a_commission_minimum_is_respected(self):
        model = CostModel(commission_per_unit=0.001, commission_min=50.0)
        notional = model.affordable_notional(cash=1000.0, price=100.0)
        # The minimum binds: only 1000 - 50 is spendable on notional.
        assert notional == pytest.approx(950.0)

    def test_a_taker_rate_is_covered(self):
        model = CostModel(taker_fee_bps=10.0)
        notional = model.affordable_notional(cash=1000.0, price=100.0)
        units = notional / 100.0
        assert notional + model.fee_for_fill(quantity=units, price=100.0) <= 1000.0 + 1e-6

    def test_zero_cash_affords_nothing(self):
        assert NO_COSTS.affordable_notional(cash=0.0, price=100.0) == 0.0

    def test_the_result_is_never_negative(self):
        model = CostModel(commission_flat=10.0)
        assert model.affordable_notional(cash=5.0, price=100.0) == 0.0
