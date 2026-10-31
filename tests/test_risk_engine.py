"""The deterministic risk engine: the limits it enforces and the shorts it refuses.

The engine is the safety boundary, so these tests are written the way an auditor
would read it: for each limit, a case that sits just inside it (approved) and a
case that breaches it (rejected), plus the long-only guarantees, which must hold
even when every other check would pass.

The long-only cases are the important ones. A prompt can be talked out of
"long-only"; arithmetic cannot.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tradingagents.portfolio import Ledger
from tradingagents.risk import (
    RiskDecision,
    RiskEngine,
    RiskLimits,
    RiskReason,
    RiskVerdict,
    drawdown_from_peak,
)
from tradingagents.signals import SignalAction, TradingSignal, signal_from_decision

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)


def _signal(action=SignalAction.BUY, *, symbol="AAPL", size=None, **kwargs) -> TradingSignal:
    return TradingSignal(
        symbol=symbol, action=action, confidence=0.8, timestamp=T0, as_of_date="2026-01-05",
        position_size=size, **kwargs,
    )


def _ledger(cash=100_000.0) -> Ledger:
    return Ledger(initial_cash=cash)


# ---------------------------------------------------------------------------
# Limits configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRiskLimits:
    def test_defaults_are_long_only(self):
        assert RiskLimits().long_only is True

    def test_disabling_long_only_is_a_configuration_error(self):
        """The engine only implements long-only; a flag cannot turn that off."""
        with pytest.raises(ValidationError, match="long-only"):
            RiskLimits(long_only=False)

    def test_a_default_size_above_the_position_cap_is_rejected(self):
        with pytest.raises(ValidationError, match="default_position_size"):
            RiskLimits(max_position_size=0.05, default_position_size=0.10)

    def test_an_inverted_order_value_band_is_rejected(self):
        with pytest.raises(ValidationError, match="min_order_value"):
            RiskLimits(min_order_value=1_000.0, max_order_value=100.0)

    def test_extra_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            RiskLimits(max_leverage=3)

    def test_limits_can_be_disabled_explicitly(self):
        limits = RiskLimits(
            max_position_size=None, max_portfolio_exposure=None, max_open_positions=None
        )
        assert limits.max_position_size is None


# ---------------------------------------------------------------------------
# Non-actionable and invalid input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputGuards:
    def test_a_hold_produces_no_order(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.HOLD), _ledger(), price=100.0)
        assert decision.verdict is RiskVerdict.REJECTED
        assert RiskReason.NOT_ACTIONABLE in decision.reasons
        assert not decision.is_order

    def test_a_missing_price_is_rejected(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(), _ledger(), price=None)
        assert RiskReason.INVALID_PRICE in decision.reasons

    def test_a_non_positive_price_is_rejected(self):
        engine = RiskEngine()
        assert RiskReason.INVALID_PRICE in engine.evaluate(_signal(), _ledger(), price=0.0).reasons
        assert RiskReason.INVALID_PRICE in engine.evaluate(_signal(), _ledger(), price=-5.0).reasons

    def test_a_book_with_no_equity_is_rejected(self):
        engine = RiskEngine()
        ledger = Ledger(initial_cash=1_000.0)
        ledger.cash = 0.0
        decision = engine.evaluate(_signal(), ledger, price=100.0)
        assert RiskReason.NO_EQUITY in decision.reasons

    def test_an_explicit_non_positive_quantity_is_rejected(self):
        """The guard for a hand-built order: a sign error cannot mean a short."""
        engine = RiskEngine()
        for bad in (0.0, -5.0):
            decision = engine.evaluate_quantity(
                symbol="AAPL", action=SignalAction.BUY, quantity=bad, ledger=_ledger(), price=100.0
            )
            assert RiskReason.NON_POSITIVE_QUANTITY in decision.reasons
            assert decision.quantity == 0.0

    def test_an_explicit_hold_quantity_is_rejected(self):
        engine = RiskEngine()
        decision = engine.evaluate_quantity(
            symbol="AAPL", action=SignalAction.HOLD, quantity=10, ledger=_ledger(), price=100.0
        )
        assert RiskReason.NOT_ACTIONABLE in decision.reasons


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSizing:
    def test_a_request_within_the_cap_is_honoured(self):
        engine = RiskEngine(RiskLimits(max_position_size=0.2, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=0.1), _ledger(100_000.0), price=100.0)
        assert decision.approved
        # 10% of 100,000 = 10,000 -> 100 units at 100
        assert decision.quantity == pytest.approx(100.0)
        assert decision.notional == pytest.approx(10_000.0)

    def test_a_request_above_the_cap_is_clamped_not_refused(self):
        engine = RiskEngine(RiskLimits(max_position_size=0.1, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=0.5), _ledger(100_000.0), price=100.0)
        assert decision.approved
        assert decision.quantity == pytest.approx(100.0)  # 10% of 100,000
        assert any("clamped" in note for note in decision.notes)

    def test_no_requested_size_uses_the_default(self):
        engine = RiskEngine(RiskLimits(default_position_size=0.02, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=None), _ledger(100_000.0), price=100.0)
        assert decision.quantity == pytest.approx(20.0)

    def test_a_size_is_capped_by_available_cash(self):
        """Cash, not equity, is the binding constraint once the book is invested."""
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="MSFT", quantity=90, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("MSFT", 100.0)  # cash 1,000; equity 10,000
        engine = RiskEngine(RiskLimits(max_position_size=1.0, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=1.0), ledger, price=100.0)
        assert decision.quantity == pytest.approx(10.0)
        assert any("cash" in note for note in decision.notes)

    def test_a_sell_sizes_to_the_holding(self):
        ledger = _ledger()
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.SELL), ledger, price=110.0)
        assert decision.approved
        assert decision.quantity == pytest.approx(10.0)

    def test_a_partial_sell_sell_request_is_a_fraction_of_the_holding(self):
        ledger = _ledger()
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.SELL, size=0.4), ledger, price=110.0)
        assert decision.quantity == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Position, exposure and count limits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBookLimits:
    def test_exposure_headroom_clamps_a_new_position(self):
        """With 90% already invested, a 5% position cannot fit under a 95% cap."""
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="MSFT", quantity=900, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("MSFT", 100.0)
        engine = RiskEngine(RiskLimits(max_portfolio_exposure=0.95, max_position_size=0.5))
        decision = engine.evaluate(_signal(size=0.5), ledger, price=100.0)
        # market value 90,000; equity 100,000; cap 95,000 -> 5,000 headroom -> 50 units
        assert decision.quantity == pytest.approx(50.0)

    def test_a_full_book_cannot_add_a_new_position(self):
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="MSFT", quantity=900, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("MSFT", 100.0)  # 90% invested, 10% cash
        engine = RiskEngine(RiskLimits(max_portfolio_exposure=0.90))
        decision = engine.evaluate(_signal(size=0.5), ledger, price=100.0)
        # No exposure headroom left, so nothing is ordered even though the
        # position cap alone would have allowed it.
        assert not decision.approved
        assert RiskReason.MAX_PORTFOLIO_EXPOSURE in decision.reasons
        assert decision.quantity == pytest.approx(0.0)
        assert not decision.is_order

    def test_exposure_headroom_clamps_a_large_request(self):
        """The exposure cap, not the position cap, decides here: 5% of headroom."""
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="MSFT", quantity=900, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("MSFT", 100.0)
        engine = RiskEngine(RiskLimits(max_portfolio_exposure=0.95, max_position_size=0.5))
        decision = engine.evaluate(_signal(size=0.5), ledger, price=100.0)
        assert decision.quantity == pytest.approx(50.0)

    def test_the_open_position_count_is_enforced(self):
        ledger = _ledger(1_000_000.0)
        for i in range(3):
            ledger.apply_fill(symbol=f"S{i}", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine(RiskLimits(max_open_positions=3, max_position_size=0.5,
                                       max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(symbol="NEW"), ledger, price=100.0)
        assert RiskReason.MAX_OPEN_POSITIONS in decision.reasons
        assert not decision.approved

    def test_adding_to_an_existing_position_does_not_consume_a_slot(self):
        ledger = _ledger(1_000_000.0)
        for i in range(3):
            ledger.apply_fill(symbol=f"S{i}", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine(RiskLimits(max_open_positions=3, max_position_size=0.5,
                                       max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(symbol="S0", size=0.01), ledger, price=100.0)
        assert decision.approved


# ---------------------------------------------------------------------------
# Order-value band
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrderValueBand:
    def test_an_order_below_the_minimum_is_rejected(self):
        engine = RiskEngine(RiskLimits(min_order_value=10_000.0, max_position_size=0.5,
                                       max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=0.01), _ledger(100_000.0), price=100.0)
        assert RiskReason.MIN_ORDER_VALUE in decision.reasons

    def test_an_order_above_the_maximum_is_clamped(self):
        engine = RiskEngine(RiskLimits(max_order_value=5_000.0, max_position_size=0.5,
                                       max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=0.5), _ledger(100_000.0), price=100.0)
        assert decision.approved
        assert decision.notional == pytest.approx(5_000.0)

    def test_an_order_inside_the_band_passes_untouched(self):
        engine = RiskEngine(RiskLimits(min_order_value=1_000.0, max_order_value=50_000.0,
                                       max_position_size=0.5, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(size=0.1), _ledger(100_000.0), price=100.0)
        assert decision.approved
        assert decision.notional == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Drawdown and daily loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrawdownAndDailyLoss:
    def test_a_breached_drawdown_limit_stops_new_positions(self):
        engine = RiskEngine(RiskLimits(max_drawdown=0.10, max_portfolio_exposure=None))
        ledger = _ledger(90_000.0)
        decision = engine.evaluate(_signal(), ledger, price=100.0, equity_peak=100_000.0)
        assert RiskReason.MAX_DRAWDOWN in decision.reasons
        assert not decision.approved

    def test_a_drawdown_inside_the_limit_passes(self):
        engine = RiskEngine(RiskLimits(max_drawdown=0.20, max_portfolio_exposure=None))
        ledger = _ledger(95_000.0)
        decision = engine.evaluate(_signal(), ledger, price=100.0, equity_peak=100_000.0)
        assert decision.approved

    def test_a_missing_peak_is_noted_not_silently_passed(self):
        engine = RiskEngine(RiskLimits(max_drawdown=0.10, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(), _ledger(), price=100.0, equity_peak=None)
        assert decision.approved
        assert any("max_drawdown not checked" in note for note in decision.notes)

    def test_a_breached_daily_loss_limit_stops_new_positions(self):
        engine = RiskEngine(RiskLimits(max_daily_loss=0.05, max_portfolio_exposure=None))
        ledger = _ledger(94_000.0)
        decision = engine.evaluate(_signal(), ledger, price=100.0, day_start_equity=100_000.0)
        assert RiskReason.MAX_DAILY_LOSS in decision.reasons

    def test_a_missing_day_start_is_noted_not_silently_passed(self):
        engine = RiskEngine(RiskLimits(max_daily_loss=0.05, max_portfolio_exposure=None))
        decision = engine.evaluate(_signal(), _ledger(), price=100.0, day_start_equity=None)
        assert decision.approved
        assert any("max_daily_loss not checked" in note for note in decision.notes)


# ---------------------------------------------------------------------------
# Long-only enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLongOnly:
    def test_a_sell_with_no_position_is_rejected(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.SELL), _ledger(), price=100.0)
        assert RiskReason.INSUFFICIENT_POSITION in decision.reasons
        assert "long-only" in " ".join(decision.notes)

    def test_a_sell_larger_than_the_holding_is_rejected(self):
        ledger = _ledger()
        ledger.apply_fill(symbol="AAPL", quantity=5, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine()
        decision = engine.evaluate_quantity(
            symbol="AAPL", action=SignalAction.SELL, quantity=6, ledger=ledger, price=100.0
        )
        assert RiskReason.SHORT_NOT_ALLOWED in decision.reasons

    def test_a_sell_equal_to_the_holding_is_approved(self):
        ledger = _ledger()
        ledger.apply_fill(symbol="AAPL", quantity=5, price=100.0, fee=0.0, timestamp=T0)
        engine = RiskEngine()
        decision = engine.evaluate_quantity(
            symbol="AAPL", action=SignalAction.SELL, quantity=5, ledger=ledger, price=100.0
        )
        assert decision.approved

    def test_no_signal_can_express_a_short(self):
        """A short is unrepresentable, so no input can request one."""
        for action in SignalAction:
            assert "SHORT" not in action.value

    def test_a_sell_of_an_unknown_symbol_is_refused_by_the_quantity_path(self):
        engine = RiskEngine()
        decision = engine.evaluate_quantity(
            symbol="TSLA", action=SignalAction.SELL, quantity=1, ledger=_ledger(), price=100.0
        )
        assert decision.reasons
        assert not decision.approved


# ---------------------------------------------------------------------------
# Verdict shape and determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerdict:
    def test_a_rejection_carries_zero_quantity_and_a_reason(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.HOLD), _ledger(), price=100.0)
        assert decision.quantity == 0.0
        assert decision.notional == 0.0
        assert decision.reasons

    def test_a_rejection_explains_itself(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(SignalAction.HOLD), _ledger(), price=100.0)
        assert "rejected" in decision.explain()
        assert "NOT_ACTIONABLE" in decision.explain()

    def test_an_approval_explains_itself(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(size=0.1), _ledger(), price=100.0)
        assert "approved" in decision.explain()

    def test_the_same_inputs_give_the_same_verdict(self):
        engine = RiskEngine()
        ledger = _ledger()
        first = engine.evaluate(_signal(size=0.1), ledger, price=100.0)
        for _ in range(20):
            again = engine.evaluate(_signal(size=0.1), ledger, price=100.0)
            assert (again.verdict, again.quantity, again.reasons) == (
                first.verdict, first.quantity, first.reasons
            )

    def test_a_decision_reports_the_symbol_and_action_it_concerned(self):
        engine = RiskEngine()
        decision = engine.evaluate(_signal(symbol="MSFT"), _ledger(), price=100.0)
        assert decision.symbol == "MSFT"
        assert decision.action is SignalAction.BUY


# ---------------------------------------------------------------------------
# End-to-end from a model rating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromDecisionToVerdict:
    def test_a_buy_rating_becomes_an_approved_order(self):
        signal = signal_from_decision(symbol="AAPL", rating="Buy", as_of_date="2026-01-05",
                                      timestamp=T0)
        assert signal is not None
        decision = RiskEngine().evaluate(signal, _ledger(100_000.0), price=100.0)
        assert decision.is_order

    def test_a_review_rating_produces_no_signal_and_so_no_order(self):
        signal = signal_from_decision(symbol="AAPL", rating="REVIEW", as_of_date="2026-01-05")
        assert signal is None

    def test_an_underweight_rating_becomes_a_sell_that_the_book_must_cover(self):
        signal = signal_from_decision(symbol="AAPL", rating="Underweight", as_of_date="2026-01-05",
                                      timestamp=T0)
        assert signal is not None and signal.action is SignalAction.SELL
        # An empty book cannot cover it, so the engine refuses rather than shorting.
        decision = RiskEngine().evaluate(signal, _ledger(), price=100.0)
        assert not decision.approved
        assert RiskReason.INSUFFICIENT_POSITION in decision.reasons

    def test_a_model_size_cannot_exceed_the_cap(self):
        signal = signal_from_decision(symbol="AAPL", rating="Buy", as_of_date="2026-01-05",
                                      timestamp=T0, position_size=0.9)
        assert signal is not None
        decision = RiskEngine(RiskLimits(max_position_size=0.1, max_portfolio_exposure=None)).evaluate(
            signal, _ledger(100_000.0), price=100.0
        )
        assert decision.quantity == pytest.approx(100.0)


@pytest.mark.unit
class TestHelpers:
    def test_drawdown_from_peak(self):
        assert drawdown_from_peak(90.0, 100.0) == pytest.approx(0.10)
        assert drawdown_from_peak(110.0, 100.0) == 0.0
        assert drawdown_from_peak(90.0, 0.0) == 0.0


@pytest.mark.unit
def test_risk_decision_defaults_are_safe():
    decision = RiskDecision(verdict=RiskVerdict.REJECTED, symbol="X", action=SignalAction.BUY)
    assert decision.quantity == 0.0
    assert not decision.is_order


# ---------------------------------------------------------------------------
# Fill-time re-check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillTimeRecheck:
    """The caps must hold for the position the fill creates, not the one planned.

    The engine is event-driven: a decision is sized against one price and filled
    at another. When the fill price gaps up, the position bought is worth more
    than the size allowed. ``review_fill`` re-checks the same caps at the fill
    price, which is the only point where that price is known and the book has not
    moved. These tests drive it directly, with the fill price as the input, so the
    cap arithmetic is pinned independently of the backtest loop.
    """

    def _engine(self, **limits):
        # default_position_size must stay <= max_position_size, so pin it low.
        limits.setdefault("default_position_size", 0.001)
        return RiskEngine(RiskLimits(**limits))

    def test_a_gap_up_is_capped_to_the_position_limit(self):
        # 20% of 100,000 is 20,000; at a fill price of 200 that is 100 units.
        # The decision asked for 200 units (sized at 100), which would be 40%.
        decision = self._engine(max_position_size=0.20).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=200,
            ledger=_ledger(100_000.0), fill_price=200.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED
        assert decision.quantity == pytest.approx(100.0)
        assert decision.notional == pytest.approx(20_000.0)
        assert RiskReason.MAX_POSITION_SIZE in decision.reasons

    def test_a_fill_within_the_limit_is_not_reduced(self):
        decision = self._engine(max_position_size=0.20).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=100,
            ledger=_ledger(100_000.0), fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED
        assert decision.quantity == pytest.approx(100.0)
        assert decision.reasons == []

    def test_a_gap_small_enough_not_to_breach_leaves_the_size_alone(self):
        # 100 units at 190 is 19,000, still under the 20,000 cap.
        decision = self._engine(max_position_size=0.20).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=100,
            ledger=_ledger(100_000.0), fill_price=190.0,
        )
        assert decision.quantity == pytest.approx(100.0)

    def test_the_reduction_accounts_for_an_existing_position(self):
        # Already holding 100 units marked at 200 = 20,000, exactly at the cap.
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=100, price=200.0, fee=0.0, timestamp=T0)
        ledger.mark("AAPL", 200.0)
        decision = self._engine(max_position_size=0.20).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=50,
            ledger=ledger, fill_price=200.0,
        )
        assert decision.verdict is RiskVerdict.REJECTED
        assert RiskReason.MAX_POSITION_SIZE in decision.reasons
        assert decision.quantity == 0.0

    def test_exposure_cap_binds_across_positions(self):
        # Two existing positions at 40% each = 80% exposure; a 95% cap leaves
        # 15% headroom. At a fill of 100 that is 15,000 / 100 = 150 units.
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="AAA", quantity=400, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="BBB", quantity=400, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("AAA", 100.0)
        ledger.mark("BBB", 100.0)
        decision = self._engine(
            max_position_size=1.0, max_portfolio_exposure=0.95
        ).review_fill(
            symbol="CCC", action=SignalAction.BUY, quantity=400,
            ledger=ledger, fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED
        assert decision.quantity == pytest.approx(150.0)

    def test_max_order_value_still_applies_at_the_fill_price(self):
        decision = self._engine(
            max_position_size=1.0, max_portfolio_exposure=1.0, max_order_value=5_000.0
        ).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=100,
            ledger=_ledger(100_000.0), fill_price=100.0,
        )
        assert decision.quantity == pytest.approx(50.0)

    def test_a_reduction_below_min_order_value_is_refused(self):
        # The cap leaves 10 units at 200 = 2,000, below a 5,000 minimum.
        decision = self._engine(
            max_position_size=0.02, min_order_value=5_000.0
        ).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=200,
            ledger=_ledger(100_000.0), fill_price=200.0,
        )
        assert decision.verdict is RiskVerdict.REJECTED
        assert RiskReason.MIN_ORDER_VALUE in decision.reasons
        assert decision.quantity == 0.0

    def test_no_positive_size_remains_when_the_book_is_at_the_cap(self):
        # The whole book is already one position at the exposure cap, so there is
        # no room for any additional exposure at all.
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="AAA", quantity=1_000, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("AAA", 100.0)
        decision = self._engine(
            max_position_size=1.0, max_portfolio_exposure=0.95
        ).review_fill(
            symbol="BBB", action=SignalAction.BUY, quantity=10,
            ledger=ledger, fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.REJECTED
        assert decision.quantity == 0.0

    def test_a_sell_is_not_capped_and_is_approved(self):
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=100, price=100.0, fee=0.0, timestamp=T0)
        decision = self._engine(max_position_size=0.01).review_fill(
            symbol="AAPL", action=SignalAction.SELL, quantity=100,
            ledger=ledger, fill_price=200.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED
        assert decision.quantity == pytest.approx(100.0)

    def test_a_sell_is_approved_even_when_it_exceeds_the_position(self):
        """The holding limit is the ledger's, enforced where the fill is applied.

        ``review_fill`` must not reject a sell for exceeding the holding: that
        check belongs to the fill, and duplicating it here with a different rule
        would let the two disagree.
        """
        ledger = _ledger(100_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=50, price=100.0, fee=0.0, timestamp=T0)
        decision = self._engine().review_fill(
            symbol="AAPL", action=SignalAction.SELL, quantity=100,
            ledger=ledger, fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_fill_price_is_refused(self, bad):
        decision = self._engine().review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=10,
            ledger=_ledger(100_000.0), fill_price=bad,
        )
        assert decision.verdict is RiskVerdict.REJECTED
        assert RiskReason.INVALID_PRICE in decision.reasons

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_a_non_positive_or_non_finite_quantity_is_refused(self, bad):
        decision = self._engine().review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=bad,
            ledger=_ledger(100_000.0), fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.REJECTED
        assert RiskReason.NON_POSITIVE_QUANTITY in decision.reasons

    def test_a_reduction_to_a_fractional_size_is_still_approved(self):
        """A cap that leaves less than one whole unit is not rounded down to zero.

        The ledger is quantity-based and carries fractional units throughout, so a
        reduced fill of 0.002 units is a real size, not an empty one. Rejecting it
        would silently turn a small book into a book that can never trade. The
        rejection case is the one where the cap leaves *no* headroom, which is
        covered by the at-cap tests.
        """
        decision = self._engine(max_position_size=0.20).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=10,
            ledger=_ledger(1.0), fill_price=100.0,
        )
        assert decision.verdict is RiskVerdict.APPROVED
        assert decision.quantity == pytest.approx(0.002)
        assert decision.notional == pytest.approx(0.2)

    def test_the_recheck_does_not_move_the_ledger(self):
        """It computes; it must not apply. A refused fill charges nothing."""
        ledger = _ledger(100_000.0)
        before = ledger.snapshot(T0)
        self._engine(max_position_size=0.02).review_fill(
            symbol="AAPL", action=SignalAction.BUY, quantity=200,
            ledger=ledger, fill_price=200.0,
        )
        after = ledger.snapshot(T0)
        assert after.cash == before.cash
        assert after.equity == before.equity
        assert after.total_fees == before.total_fees
        assert ledger.events == []
