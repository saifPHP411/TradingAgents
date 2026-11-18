from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_ema import PerpetualEMAStrategy, _features
from tradingagents.crypto.backtest.perpetual_strategy_lab import (
    CANDIDATES,
    PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES,
    _rank_candidates,
    acceptance_gate,
)


def _development_summary(**overrides):
    result = {
        "trades": 100,
        "win_rate_pct": 80.0,
        "expectancy_per_trade": 0.25,
        "profit_factor": 1.35,
        "equal_weight_compounded_return_pct": 2.4,
        "positive_folds": 7,
        "fold_count": 10,
        "positive_symbols": 3,
        "symbol_count": 4,
        "worst_fold_drawdown_pct": 4.5,
        "liquidations": 0,
    }
    result.update(overrides)
    return result


@pytest.mark.unit
def test_development_gate_requires_win_rate_profit_factor_and_fold_breadth():
    summary = _development_summary(profit_factor=0.99)
    gate = acceptance_gate(summary, "realistic_estimate")

    assert gate["minimum_trade_count"] is True
    assert gate["target_win_rate"] is True
    assert gate["positive_expectancy"] is True
    assert gate["positive_compounded_return"] is True
    assert gate["positive_fold_ratio"] is True
    assert gate["positive_symbol_breadth"] is True
    assert gate["profit_factor"] is False
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("scenario", "profit_factor", "expected"),
    [
        ("realistic_estimate", 1.20, True),
        ("realistic_estimate", 1.19, False),
        ("conservative", 1.05, True),
        ("conservative", 1.04, False),
        ("stress", 1.00, True),
        ("stress", 0.99, False),
        ("stress", "infinite", True),
    ],
)
def test_candidate_gate_uses_distinct_predeclared_cost_thresholds(
    scenario, profit_factor, expected
):
    gate = acceptance_gate(_development_summary(profit_factor=profit_factor), scenario)
    assert gate["profit_factor"] is expected


@pytest.mark.unit
def test_development_gate_rejects_target_miss_low_expectancy_or_risk_failures():
    weak_folds = acceptance_gate(
        _development_summary(positive_folds=5), "realistic_estimate"
    )
    liquidation = acceptance_gate(
        _development_summary(liquidations=1), "realistic_estimate"
    )
    low_win_rate = acceptance_gate(
        _development_summary(win_rate_pct=78.99), "realistic_estimate"
    )
    negative_expectancy = acceptance_gate(
        _development_summary(expectancy_per_trade=-0.01), "realistic_estimate"
    )

    assert low_win_rate["target_win_rate"] is False
    assert low_win_rate["passed"] is False
    assert negative_expectancy["positive_expectancy"] is False
    assert negative_expectancy["passed"] is False
    assert weak_folds["positive_fold_ratio"] is False
    assert weak_folds["passed"] is False
    assert liquidation["no_liquidations"] is False
    assert liquidation["passed"] is False


@pytest.mark.unit
def test_tournament_candidates_are_fixed_and_include_trend_and_mean_reversion_families():
    assert len(CANDIDATES) == 6
    assert {"ema_cross", "donchian_breakout", "bollinger_reversion"} == {
        candidate["signal_mode"] for candidate in CANDIDATES.values()
    }
    assert all(candidate["trend"] == 200 for candidate in CANDIDATES.values())


@pytest.mark.unit
def test_donchian_breakout_uses_only_previous_completed_candles_for_channel():
    count = 224
    index = pd.date_range("2026-01-01", periods=count, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
        },
        index=index,
    )
    # The current candle spikes to 110 but closes below its own high. That spike
    # must not be included in the channel used to evaluate this same candle.
    frame.iloc[-3, frame.columns.get_loc("open")] = 100.0
    frame.iloc[-3, frame.columns.get_loc("high")] = 110.0
    frame.iloc[-3, frame.columns.get_loc("low")] = 99.0
    frame.iloc[-3, frame.columns.get_loc("close")] = 102.0
    # The next close remains below the channel high made by the prior candle.
    frame.iloc[-2, frame.columns.get_loc("open")] = 101.0
    frame.iloc[-2, frame.columns.get_loc("high")] = 103.0
    frame.iloc[-2, frame.columns.get_loc("low")] = 100.0
    frame.iloc[-2, frame.columns.get_loc("close")] = 102.0

    features = _features(
        frame,
        PerpetualEMAStrategy(
            enabled=True,
            trend=200,
            signal_mode="donchian_breakout",
            breakout_period=20,
            min_adx=None,
        ),
    )

    assert features["prior_channel_high"].iloc[-3] == pytest.approx(101.0)
    assert features["signal"].iloc[-3] == 1
    assert features["prior_channel_high"].iloc[-2] == pytest.approx(110.0)
    assert features["signal"].iloc[-2] == 0


@pytest.mark.unit
def test_invalid_signal_mode_is_rejected():
    with pytest.raises(ValueError, match="signal_mode"):
        PerpetualEMAStrategy(signal_mode="future_prediction")


@pytest.mark.unit
def test_previous_holdout_failure_overrides_a_positive_development_gate():
    rankings, eligible = _rank_candidates(
        [
            {
                "candidate": "ema_14_50_adx18",
                "leverage": 2.0,
                "development_gate_passed_all_costs": True,
                "minimum_scenario_profit_factor": 1.36,
                "conservative_return_pct": 1.92,
            },
            {
                "candidate": "donchian_20_adx18",
                "leverage": 2.0,
                "development_gate_passed_all_costs": False,
                "minimum_scenario_profit_factor": 0.97,
                "conservative_return_pct": 1.69,
            },
        ]
    )

    assert "ema_14_50_adx18" in PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES
    rejected = next(row for row in rankings if row["candidate"] == "ema_14_50_adx18")
    assert rejected["development_gate_passed_all_costs"] is True
    assert rejected["previous_holdout_rejected"] is True
    assert rejected["selection_eligible"] is False
    assert eligible == []


@pytest.mark.unit
def test_a_non_rejected_all_cost_candidate_can_still_be_shortlisted():
    rankings, eligible = _rank_candidates(
        [
            {
                "candidate": "ema_14_50_adx18",
                "leverage": 2.0,
                "development_gate_passed_all_costs": True,
                "minimum_scenario_profit_factor": 1.72,
                "conservative_return_pct": 1.92,
            },
            {
                "candidate": "donchian_20_adx18",
                "leverage": 1.0,
                "development_gate_passed_all_costs": True,
                "minimum_scenario_profit_factor": 1.12,
                "conservative_return_pct": 1.05,
            },
        ]
    )

    assert len(eligible) == 1
    assert eligible[0]["candidate"] == "donchian_20_adx18"
    assert not rankings[0]["selection_eligible"] or rankings[0]["candidate"] != "ema_14_50_adx18"
