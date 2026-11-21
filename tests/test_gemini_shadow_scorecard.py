"""Evidence gates use only closed, stressed shadow outcomes, not raw win rate."""
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.paper_experiment.perpetual_demo.scorecard import shadow_scorecard

START = datetime(2026, 11, 10, tzinfo=UTC)
AS_OF = int((START + timedelta(days=7)).timestamp() * 1000)


def rows(wins, losses, *, day_stride=1):
    values = wins + losses
    return [
        {
            "exit_ms": int((START + timedelta(days=i % 4 * day_stride, minutes=i)).timestamp() * 1000),
            "net_pct": value,
        }
        for i, value in enumerate(values)
    ]


def test_positive_multi_day_history_meets_risk_adjusted_shadow_gate():
    result = shadow_scorecard(rows([0.5] * 24, [-0.4] * 6), as_of_ms=AS_OF)
    assert result["settled_trades"] == 30
    assert result["win_rate"] == pytest.approx(0.8)
    assert result["win_rate_lower_95"] < result["win_rate"]
    assert result["profit_factor"] == pytest.approx(5.0)
    assert result["net_expectancy_pct_per_trade"] > 0
    assert result["eligible_for_demo"] is True
    assert result["failed_gates"] == []
    assert result["source"] == "settled_shadow_mark_model_not_venue_fills"


def test_high_win_rate_does_not_override_negative_after_cost_expectancy():
    result = shadow_scorecard(rows([0.1] * 25, [-1.0] * 5), as_of_ms=AS_OF)
    assert result["win_rate"] > 0.8
    assert result["eligible_for_demo"] is False
    assert "profit_factor_below_threshold" in result["failed_gates"]
    assert "nonpositive_net_expectancy" in result["failed_gates"]


def test_no_losses_is_not_an_infinite_profit_factor():
    result = shadow_scorecard(rows([0.5] * 30, []), as_of_ms=AS_OF)
    assert result["profit_factor"] is None
    assert result["eligible_for_demo"] is False
    assert "profit_factor_undefined_without_losses" in result["failed_gates"]


def test_one_day_cannot_qualify_even_with_many_profitable_closes():
    result = shadow_scorecard(rows([0.5] * 24, [-0.4] * 6, day_stride=0), as_of_ms=AS_OF)
    assert result["days_observed"] == 1
    assert "insufficient_independent_days" in result["failed_gates"]


def test_large_drawdown_blocks_and_is_not_account_drawdown():
    result = shadow_scorecard(rows([1.0] * 25, [-18.0] * 5), as_of_ms=AS_OF)
    assert result["modeled_max_drawdown_fraction"] > 0.15
    assert "modeled_drawdown_exceeds_threshold" in result["failed_gates"]


def test_future_nan_boolean_and_catastrophic_returns_fail_closed():
    for corrupted in [
        {"exit_ms": AS_OF, "net_pct": 0.5},
        {"exit_ms": AS_OF - 1, "net_pct": float("nan")},
        {"exit_ms": AS_OF - 1, "net_pct": True},
        {"exit_ms": AS_OF - 1, "net_pct": -100},
    ]:
        with pytest.raises(ValueError):
            shadow_scorecard([corrupted], as_of_ms=AS_OF)


def test_empty_history_is_explicitly_unqualified():
    result = shadow_scorecard([], as_of_ms=AS_OF)
    assert result["settled_trades"] == 0
    assert result["win_rate"] is None
    assert result["eligible_for_demo"] is False
