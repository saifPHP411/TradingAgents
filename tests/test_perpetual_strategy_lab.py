from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_ema import PerpetualEMAStrategy, _features
from tradingagents.crypto.backtest.perpetual_ema_walkforward import (
    TARGET_WIN_RATE_PCT as WALKFORWARD_TARGET_WIN_RATE_PCT,
    _summarize_development,
)
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
def test_development_and_final_holdout_gates_share_the_70_percent_target():
    from tradingagents.crypto.backtest.perpetual_strategy_lab import (
        TARGET_WIN_RATE_PCT as DEVELOPMENT_TARGET_WIN_RATE_PCT,
    )

    assert DEVELOPMENT_TARGET_WIN_RATE_PCT == 70.0
    assert WALKFORWARD_TARGET_WIN_RATE_PCT == 70.0


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
def test_70_percent_win_rate_meets_the_revised_target():
    accepted = acceptance_gate(_development_summary(win_rate_pct=70.0), "realistic_estimate")
    rejected = acceptance_gate(_development_summary(win_rate_pct=69.99), "realistic_estimate")

    assert accepted["target_win_rate"] is True
    assert rejected["target_win_rate"] is False


def test_development_gate_rejects_target_miss_low_expectancy_or_risk_failures():
    weak_folds = acceptance_gate(
        _development_summary(positive_folds=5), "realistic_estimate"
    )
    liquidation = acceptance_gate(
        _development_summary(liquidations=1), "realistic_estimate"
    )
    low_win_rate = acceptance_gate(
        _development_summary(win_rate_pct=69.99), "realistic_estimate"
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
    assert len(CANDIDATES) == 31
    assert {
        "ema_14_50_adx18",
        "ema_9_30_adx22",
        "ema_20_60_adx16",
        "donchian_20_adx18",
        "donchian_55_adx18",
        "selective_bollinger_rsi_range",
        "bollinger_rsi_balanced",
        "bollinger_rsi_higher_frequency",
        "bollinger_rsi_wide_regime",
        "bollinger_rsi_exit_hysteresis",
        "bollinger_rsi_hysteresis_reward_filter",
        "bollinger_rsi_strong_recovery",
        "bollinger_rsi_adverse_di_exit",
        "bollinger_rsi_di_confirmed_recovery",
        "selective_bollinger_rsi_range_freq20",
        "bollinger_rsi_adverse_di_exit_freq20",
        "selective_bollinger_rsi_range_quality20",
        "bollinger_rsi_adverse_di_exit_quality20",
        "bollinger_rsi_adverse_di_exit_quality20_early_exit",
        "bollinger_rsi_quality20_tighter_atr18",
        "bollinger_rsi_quality20_fast_trend_exit",
        "bollinger_rsi_quality20_reward_risk05",
        "bollinger_rsi_quality20_atr16",
        "bollinger_rsi_quality20_atr17",
        "bollinger_rsi_quality20_atr19",
        "bollinger_rsi_quality20_adx16",
        "bollinger_rsi_quality20_atr16_recovery45",
        "bollinger_rsi_quality20_atr16_std18",
        "bollinger_rsi_quality20_atr16_adx20",
        "bollinger_rsi_quality20_atr16_di16",
        "bollinger_rsi_quality20_atr16_di20",
    } == set(CANDIDATES)
    assert {"ema_cross", "donchian_breakout", "bollinger_reversion"} == {
        candidate["signal_mode"] for candidate in CANDIDATES.values()
    }
    assert all(candidate["trend"] == 200 for candidate in CANDIDATES.values())
    assert all(
        CANDIDATES[name]["signal_mode"] == "bollinger_reversion"
        for name in (
            "selective_bollinger_rsi_range",
            "bollinger_rsi_balanced",
            "bollinger_rsi_higher_frequency",
            "bollinger_rsi_wide_regime",
            "bollinger_rsi_exit_hysteresis",
            "bollinger_rsi_hysteresis_reward_filter",
            "bollinger_rsi_strong_recovery",
            "bollinger_rsi_adverse_di_exit",
            "bollinger_rsi_di_confirmed_recovery",
            "selective_bollinger_rsi_range_freq20",
            "bollinger_rsi_adverse_di_exit_freq20",
            "selective_bollinger_rsi_range_quality20",
            "bollinger_rsi_adverse_di_exit_quality20",
            "bollinger_rsi_adverse_di_exit_quality20_early_exit",
            "bollinger_rsi_quality20_tighter_atr18",
            "bollinger_rsi_quality20_fast_trend_exit",
            "bollinger_rsi_quality20_reward_risk05",
            "bollinger_rsi_quality20_atr16",
            "bollinger_rsi_quality20_atr17",
            "bollinger_rsi_quality20_atr19",
            "bollinger_rsi_quality20_adx16",
            "bollinger_rsi_quality20_atr16_recovery45",
            "bollinger_rsi_quality20_atr16_std18",
            "bollinger_rsi_quality20_atr16_adx20",
            "bollinger_rsi_quality20_atr16_di16",
            "bollinger_rsi_quality20_atr16_di20",
        )
    )


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


@pytest.mark.unit
def test_frequency_variants_keep_original_benchmarks_and_use_moderate_thresholds():
    baseline = CANDIDATES["selective_bollinger_rsi_range"]
    frequent = CANDIDATES["selective_bollinger_rsi_range_freq20"]
    adverse = CANDIDATES["bollinger_rsi_adverse_di_exit_freq20"]

    assert baseline["mean_reversion_std"] == 2.0
    assert frequent["mean_reversion_std"] == 1.9
    assert frequent["mean_reversion_rsi_long"] == 35.0
    assert frequent["mean_reversion_rsi_short"] == 65.0
    assert frequent["mean_reversion_max_adx"] == 20.0
    assert adverse["mean_reversion_exit_requires_adverse_di"] is True
    assert adverse["mean_reversion_exit_adx"] == 24.0
    assert adverse["mean_reversion_exit_confirmation_bars"] == 2



@pytest.mark.unit
def test_quality_variants_use_observed_rsi_recovery_and_adx_filters():
    first = CANDIDATES["selective_bollinger_rsi_range_quality20"]
    second = CANDIDATES["bollinger_rsi_adverse_di_exit_quality20"]

    for candidate in (first, second):
        assert candidate["mean_reversion_std"] == 1.9
        assert candidate["mean_reversion_rsi_long"] == 35.0
        assert candidate["mean_reversion_rsi_short"] == 65.0
        assert candidate["mean_reversion_max_adx"] == 18.5
        assert candidate["mean_reversion_min_rsi_recovery"] == 5.0
    assert "mean_reversion_exit_requires_adverse_di" not in first
    assert second["mean_reversion_exit_requires_adverse_di"] is True
    assert second["mean_reversion_exit_adx"] == 24.0



@pytest.mark.unit
def test_early_exit_quality_candidate_is_a_small_controlled_follow_up():
    candidate = CANDIDATES["bollinger_rsi_adverse_di_exit_quality20_early_exit"]
    control = CANDIDATES["bollinger_rsi_adverse_di_exit_quality20"]

    assert candidate["mean_reversion_std"] == control["mean_reversion_std"] == 1.9
    assert candidate["mean_reversion_max_adx"] == control["mean_reversion_max_adx"] == 18.5
    assert candidate["mean_reversion_min_rsi_recovery"] == 4.5
    assert candidate["mean_reversion_exit_adx"] == 20.0
    assert candidate["mean_reversion_exit_confirmation_bars"] == 1
    assert candidate["mean_reversion_exit_requires_adverse_di"] is True
    assert control["mean_reversion_exit_adx"] == 24.0
    assert control["mean_reversion_exit_confirmation_bars"] == 2

@pytest.mark.unit
def test_quality20_exit_experiments_change_one_risk_control_at_a_time():
    baseline = CANDIDATES["bollinger_rsi_adverse_di_exit_quality20"]
    tighter = CANDIDATES["bollinger_rsi_quality20_tighter_atr18"]
    fast_exit = CANDIDATES["bollinger_rsi_quality20_fast_trend_exit"]
    reward_filter = CANDIDATES["bollinger_rsi_quality20_reward_risk05"]

    assert tighter["atr_multiple"] == 1.8
    assert fast_exit["mean_reversion_exit_adx"] == 20.0
    assert fast_exit["mean_reversion_exit_confirmation_bars"] == 1
    assert reward_filter["mean_reversion_min_reward_risk"] == 0.5
    for candidate in (tighter, fast_exit, reward_filter):
        for key in (
            "mean_reversion_std",
            "mean_reversion_rsi_long",
            "mean_reversion_rsi_short",
            "mean_reversion_max_adx",
            "mean_reversion_min_rsi_recovery",
        ):
            assert candidate[key] == baseline[key]

@pytest.mark.unit
def test_quality20_atr_sensitivity_and_adx_diagnostic_keep_entry_rules_fixed():
    baseline = CANDIDATES["bollinger_rsi_quality20_tighter_atr18"]
    expected_atr = {
        "bollinger_rsi_quality20_atr16": 1.6,
        "bollinger_rsi_quality20_atr17": 1.7,
        "bollinger_rsi_quality20_atr19": 1.9,
    }
    shared = (
        "mean_reversion_std",
        "mean_reversion_rsi_period",
        "mean_reversion_rsi_long",
        "mean_reversion_rsi_short",
        "mean_reversion_max_adx",
        "mean_reversion_min_rsi_recovery",
        "mean_reversion_exit_adx",
        "mean_reversion_exit_confirmation_bars",
        "mean_reversion_exit_requires_adverse_di",
    )
    for name, atr in expected_atr.items():
        candidate = CANDIDATES[name]
        assert candidate["atr_multiple"] == atr
        assert all(candidate[key] == baseline[key] for key in shared)

    adx16 = CANDIDATES["bollinger_rsi_quality20_adx16"]
    assert adx16["mean_reversion_max_adx"] == 16.0
    assert adx16["atr_multiple"] == 1.8
    assert all(adx16[key] == baseline[key] for key in shared if key != "mean_reversion_max_adx")


@pytest.mark.unit
def test_development_summary_exposes_small_account_rejections_by_symbol():
    from types import SimpleNamespace

    rows = [
        {
            "symbol": "BTCUSDT",
            "fold": 0,
            "result": SimpleNamespace(
                initial_equity=50.0,
                final_equity=50.0,
                trades=[],
                rejections={"below_minimum_notional": 3},
                metrics=lambda: {"max_drawdown_pct": 0.0},
            ),
        },
        {
            "symbol": "BNBUSDT",
            "fold": 0,
            "result": SimpleNamespace(
                initial_equity=50.0,
                final_equity=50.0,
                trades=[],
                rejections={"below_minimum_notional": 2, "liquidation_too_close_to_atr_stop": 1},
                metrics=lambda: {"max_drawdown_pct": 0.0},
            ),
        },
    ]
    summary = _summarize_development(rows, ["BTCUSDT", "BNBUSDT"])

    assert summary["rejections"] == {
        "below_minimum_notional": 5,
        "liquidation_too_close_to_atr_stop": 1,
    }
    assert summary["rejections_by_symbol"]["BTCUSDT"] == {"below_minimum_notional": 3}
    assert summary["rejections_by_symbol"]["BNBUSDT"] == {
        "below_minimum_notional": 2,
        "liquidation_too_close_to_atr_stop": 1,
    }

@pytest.mark.unit
def test_frequency_experiments_change_one_entry_threshold_at_a_time():
    baseline = CANDIDATES["bollinger_rsi_quality20_atr16"]
    recovery = CANDIDATES["bollinger_rsi_quality20_atr16_recovery45"]
    std = CANDIDATES["bollinger_rsi_quality20_atr16_std18"]
    adx = CANDIDATES["bollinger_rsi_quality20_atr16_adx20"]

    assert recovery["mean_reversion_min_rsi_recovery"] == 4.5
    assert std["mean_reversion_std"] == 1.8
    assert adx["mean_reversion_max_adx"] == 20.0
    for candidate in (recovery, std, adx):
        for key in (
            "mean_reversion_std",
            "mean_reversion_rsi_long",
            "mean_reversion_rsi_short",
            "mean_reversion_max_adx",
            "mean_reversion_min_rsi_recovery",
            "atr_multiple",
        ):
            if candidate is std and key == "mean_reversion_std":
                continue
            if candidate is adx and key == "mean_reversion_max_adx":
                continue
            if candidate is recovery and key == "mean_reversion_min_rsi_recovery":
                continue
            assert candidate[key] == baseline[key]


@pytest.mark.unit
def test_adverse_di_experiments_keep_quality16_entry_and_exit_baseline():
    baseline = CANDIDATES["bollinger_rsi_quality20_atr16"]
    for name, spread in (
        ("bollinger_rsi_quality20_atr16_di16", 16.0),
        ("bollinger_rsi_quality20_atr16_di20", 20.0),
    ):
        candidate = CANDIDATES[name]
        assert candidate["mean_reversion_max_adverse_di_spread"] == spread
        for key in (
            "mean_reversion_std",
            "mean_reversion_rsi_long",
            "mean_reversion_rsi_short",
            "mean_reversion_max_adx",
            "mean_reversion_min_rsi_recovery",
            "mean_reversion_exit_adx",
            "mean_reversion_exit_confirmation_bars",
            "atr_multiple",
        ):
            assert candidate[key] == baseline[key]
