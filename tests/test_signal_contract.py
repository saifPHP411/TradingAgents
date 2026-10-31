"""The signal contract: the AI brain's decision, validated before it can be acted on.

The point of these tests is not that the schema accepts good input, but that it
refuses everything an engine must not act on: unknown actions, a short, a
self-contradictory level, a HOLD that asks for size. A schema that only checks
types would let a model's prose become an order.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tradingagents.agents.rating import RATING_REVIEW, RATINGS_5_TIER
from tradingagents.signals import (
    SignalAction,
    TradingSignal,
    action_for_rating,
    confidence_for_rating,
    signal_from_decision,
)

NOW = datetime(2026, 1, 9, 21, 0, tzinfo=UTC)


def _signal(**overrides) -> TradingSignal:
    base = {
        "symbol": "AAPL",
        "action": SignalAction.BUY,
        "confidence": 0.8,
        "timestamp": NOW,
        "as_of_date": "2026-01-09",
    }
    base.update(overrides)
    return TradingSignal(**base)


# ---------------------------------------------------------------------------
# Action set
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActionSet:
    def test_only_three_actions_exist(self):
        assert {a.value for a in SignalAction} == {"BUY", "HOLD", "SELL"}

    def test_an_arbitrary_action_is_rejected(self):
        with pytest.raises(ValidationError):
            _signal(action="SHORT")

    def test_a_lowercase_action_is_rejected(self):
        """The engine compares enum members; "buy" is not "BUY"."""
        with pytest.raises(ValidationError):
            _signal(action="buy")

    def test_no_action_can_express_a_short(self):
        for member in SignalAction:
            assert "SHORT" not in member.value


@pytest.mark.unit
class TestRatingMapping:
    @pytest.mark.parametrize("rating", RATINGS_5_TIER)
    def test_every_rating_maps_to_a_known_action(self, rating):
        assert action_for_rating(rating) in set(SignalAction)

    @pytest.mark.parametrize("rating,expected", [
        ("Buy", SignalAction.BUY),
        ("Overweight", SignalAction.BUY),
        ("Hold", SignalAction.HOLD),
        ("Underweight", SignalAction.SELL),
        ("Sell", SignalAction.SELL),
    ])
    def test_rating_direction(self, rating, expected):
        assert action_for_rating(rating) is expected

    def test_review_maps_to_no_action(self):
        """A decision nobody can read is not a Hold and not a trade."""
        assert action_for_rating(RATING_REVIEW) is None

    def test_unknown_and_empty_ratings_map_to_no_action(self):
        for rating in (None, "", "   ", "Strong Buy", "banana"):
            assert action_for_rating(rating) is None

    def test_confidence_is_deterministic_per_tier(self):
        assert confidence_for_rating("Buy") == confidence_for_rating("Buy")
        assert confidence_for_rating("Buy") > confidence_for_rating("Overweight")
        assert confidence_for_rating("Overweight") > confidence_for_rating("Hold")

    def test_confidence_for_no_call_is_zero(self):
        assert confidence_for_rating(RATING_REVIEW) == 0.0
        assert confidence_for_rating(None) == 0.0


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFieldValidation:
    def test_confidence_is_bounded(self):
        for bad in (-0.01, 1.01, 2.0):
            with pytest.raises(ValidationError):
                _signal(confidence=bad)

    def test_confidence_bounds_are_inclusive(self):
        assert _signal(confidence=0.0).confidence == 0.0
        assert _signal(confidence=1.0).confidence == 1.0

    def test_a_naive_timestamp_is_rejected(self):
        """An ambiguous instant orders wrongly against a UTC-aware event log."""
        with pytest.raises(ValidationError):
            _signal(timestamp=datetime(2026, 1, 9, 21, 0))

    def test_a_non_utc_timestamp_is_converted(self):
        tokyo = timezone(timedelta(hours=9))
        signal = _signal(timestamp=datetime(2026, 1, 10, 6, 0, tzinfo=tokyo))
        assert signal.timestamp == NOW
        assert signal.timestamp.tzinfo is not None

    def test_a_non_canonical_as_of_date_is_rejected(self):
        for bad in ("2026-1-9", "09/01/2026", "20260109", ""):
            with pytest.raises(ValidationError):
                _signal(as_of_date=bad)

    def test_an_empty_symbol_is_rejected(self):
        for bad in ("", "   "):
            with pytest.raises(ValidationError):
                _signal(symbol=bad)

    def test_an_overlong_symbol_is_rejected(self):
        with pytest.raises(ValidationError):
            _signal(symbol="A" * 33)

    def test_non_positive_levels_are_rejected(self):
        for field in ("entry_price", "stop_loss", "take_profit"):
            for bad in (0.0, -1.0):
                with pytest.raises(ValidationError):
                    _signal(**{field: bad})

    def test_extra_fields_are_rejected(self):
        """A field nobody validated must not ride along into the engine."""
        with pytest.raises(ValidationError):
            _signal(leverage=10)

    def test_position_size_is_a_fraction_of_equity(self):
        assert _signal(position_size=0.05).position_size == 0.05
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValidationError):
                _signal(position_size=bad)

    def test_holding_period_must_be_at_least_one_day(self):
        assert _signal(holding_period_days=5).holding_period_days == 5
        with pytest.raises(ValidationError):
            _signal(holding_period_days=0)


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConsistency:
    def test_a_buy_stop_above_entry_is_rejected(self):
        with pytest.raises(ValidationError, match="must be below"):
            _signal(entry_price=100.0, stop_loss=105.0)

    def test_a_buy_target_below_entry_is_rejected(self):
        with pytest.raises(ValidationError, match="must be above"):
            _signal(entry_price=100.0, take_profit=95.0)

    def test_a_sell_stop_below_entry_is_rejected(self):
        with pytest.raises(ValidationError, match="must be above"):
            _signal(action=SignalAction.SELL, entry_price=100.0, stop_loss=95.0)

    def test_a_sell_target_above_entry_is_rejected(self):
        with pytest.raises(ValidationError, match="must be below"):
            _signal(action=SignalAction.SELL, entry_price=100.0, take_profit=105.0)

    def test_a_consistent_buy_is_accepted(self):
        signal = _signal(entry_price=100.0, stop_loss=95.0, take_profit=110.0, position_size=0.1)
        assert signal.is_actionable

    def test_a_hold_carrying_levels_is_rejected(self):
        for field in ("stop_loss", "take_profit"):
            with pytest.raises(ValidationError):
                _signal(action=SignalAction.HOLD, **{field: 95.0})

    def test_a_hold_requesting_size_is_rejected(self):
        with pytest.raises(ValidationError):
            _signal(action=SignalAction.HOLD, position_size=0.1)

    def test_a_bare_hold_is_accepted_and_is_not_actionable(self):
        signal = _signal(action=SignalAction.HOLD, confidence=0.5)
        assert not signal.is_actionable


# ---------------------------------------------------------------------------
# Building from a decision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSignalFromDecision:
    def test_a_buy_rating_builds_a_buy_signal(self):
        signal = signal_from_decision(symbol="AAPL", rating="Buy", as_of_date="2026-01-09",
                                      timestamp=NOW)
        assert signal is not None
        assert signal.action is SignalAction.BUY
        assert signal.confidence == confidence_for_rating("Buy")

    def test_an_unreadable_rating_builds_nothing(self):
        assert signal_from_decision(symbol="AAPL", rating=RATING_REVIEW,
                                    as_of_date="2026-01-09") is None

    def test_an_unknown_rating_builds_nothing(self):
        assert signal_from_decision(symbol="AAPL", rating="banana",
                                    as_of_date="2026-01-09") is None

    def test_prose_sizing_is_never_parsed(self):
        """The model's "5% of portfolio" must not become a position_size."""
        signal = signal_from_decision(
            symbol="AAPL", rating="Buy", as_of_date="2026-01-09", timestamp=NOW,
            reasoning="Enter with 5% of portfolio, sized at 120 shares.",
        )
        assert signal is not None
        assert signal.position_size is None
        assert signal.reasoning.startswith("Enter with 5%")

    def test_an_explicit_size_is_carried_as_a_request(self):
        signal = signal_from_decision(symbol="AAPL", rating="Buy", as_of_date="2026-01-09",
                                      timestamp=NOW, position_size=0.1)
        assert signal is not None and signal.position_size == 0.1

    def test_a_contradictory_level_from_the_caller_is_rejected(self):
        with pytest.raises(ValidationError):
            signal_from_decision(symbol="AAPL", rating="Buy", as_of_date="2026-01-09",
                                 timestamp=NOW, entry_price=100.0, stop_loss=120.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_the_same_decision_yields_the_same_signal_id(self):
        a = _signal(position_size=0.1)
        b = _signal(position_size=0.1)
        assert a.signal_id == b.signal_id

    def test_sub_second_clock_noise_does_not_change_the_id(self):
        a = _signal(timestamp=NOW)
        b = _signal(timestamp=NOW.replace(microsecond=123456))
        assert a.signal_id == b.signal_id

    def test_a_different_action_changes_the_id(self):
        assert _signal(action=SignalAction.BUY).signal_id != _signal(action=SignalAction.SELL).signal_id

    def test_a_different_size_changes_the_id(self):
        assert _signal(position_size=0.1).signal_id != _signal(position_size=0.2).signal_id

    def test_the_id_is_a_short_hex_digest(self):
        signal_id = _signal().signal_id
        assert len(signal_id) == 16
        int(signal_id, 16)  # raises if it is not hexadecimal
