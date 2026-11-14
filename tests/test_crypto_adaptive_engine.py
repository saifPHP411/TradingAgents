from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.adaptive import AdaptiveConfig, analyze_market

pytestmark = pytest.mark.unit

def frame(n=180):
    c=100+np.arange(n)*.1
    return pd.DataFrame({"open":c-.02,"high":c+.08,"low":c-.08,"close":c,"volume":np.full(n,1000.)},
        index=pd.date_range("2025-01-01",periods=n,freq="h",tz="UTC"))

def test_stale_forming_and_unknown_spread_fail_closed():
    x=frame()
    assert analyze_market(x,spread_bps=2,data_is_fresh=False).action=="no_trade"
    assert analyze_market(x,spread_bps=2,closed_bars_only=False).action=="no_trade"
    assert "spread is unknown" in analyze_market(x).blockers

def test_insufficient_history_fails_closed():
    assert analyze_market(frame(30),spread_bps=2).action=="no_trade"

def test_decision_is_deterministic():
    x=frame()
    assert analyze_market(x,spread_bps=2).to_dict()==analyze_market(x,spread_bps=2).to_dict()

def test_current_bar_cannot_define_its_own_range_or_volume_baseline():
    from tradingagents.crypto.adaptive import _features, _validate
    x=frame()
    a=_features(_validate(x),AdaptiveConfig()).iloc[-1]
    x.iloc[-1,x.columns.get_loc("high")]=999
    x.iloc[-1,x.columns.get_loc("volume")]=999999
    b=_features(_validate(x),AdaptiveConfig()).iloc[-1]
    assert a.prior_high==pytest.approx(b.prior_high)
    assert a.volume_base==pytest.approx(b.volume_base)

def test_excessive_spread_blocks_analysis():
    assert analyze_market(frame(),spread_bps=500).action=="no_trade"

def test_high_cost_assumption_can_veto_candidate():
    result=analyze_market(frame(),spread_bps=2,config=AdaptiveConfig(estimated_round_trip_cost_bps=1000))
    assert result.action=="no_trade"

def test_bad_prices_and_nonchronological_data_are_rejected():
    x=frame()
    x.iloc[-1,x.columns.get_loc("close")]=0
    with pytest.raises(ValueError, match="positive"):
        analyze_market(x, spread_bps=2)
    with pytest.raises(ValueError, match="chronological"):
        analyze_market(frame().iloc[::-1], spread_bps=2)

def test_long_only_action_vocabulary():
    result=analyze_market(frame(),spread_bps=2)
    assert result.action in {"buy_candidate","no_trade"}
    assert result.strategy in {"trend_pullback","range_reversion","breakout","none"}

def pullback_frame():
    x = frame()
    close = x.close.to_numpy().copy()
    close[-2] -= 1.2
    close[-1] += 0.2
    x["close"] = close
    x["open"] = close - 0.02
    x["high"] = close + 0.30
    x["low"] = close - 0.30
    return x


def test_bull_trend_pullback_expert_can_emit_explainable_candidate():
    result = analyze_market(pullback_frame(), spread_bps=2)
    assert result.regime == "bull_trend"
    assert result.strategy == "trend_pullback"
    assert result.action == "buy_candidate"
    assert result.stop_price < result.reference_price < result.target_price
    assert result.reasons


def test_same_candidate_is_vetoed_when_costs_are_excessive():
    result = analyze_market(
        pullback_frame(),
        spread_bps=2,
        config=AdaptiveConfig(estimated_round_trip_cost_bps=1000),
    )
    assert result.action == "no_trade"
    assert "estimated round-trip costs" in result.blockers[0]


def test_persistent_uptrend_has_defined_rsi_instead_of_missing_indicator():
    from tradingagents.crypto.adaptive import _features, _validate

    features = _features(_validate(frame()), AdaptiveConfig())
    assert features.rsi.iloc[-1] == pytest.approx(100.0)


def test_bear_trend_is_classified_but_short_entry_is_never_emitted():
    x = frame()
    close = 120 - np.arange(len(x)) * 0.1
    x["close"] = close
    x["open"] = close + 0.02
    x["high"] = close + 0.08
    x["low"] = close - 0.08
    result = analyze_market(x, spread_bps=2)
    assert result.regime == "bear_trend"
    assert result.action == "no_trade"
    assert result.strategy == "none"


def test_breakout_requires_relative_volume_confirmation():
    x = frame()
    x.iloc[-1, x.columns.get_loc("volume")] = 1500.0
    result = analyze_market(x, spread_bps=2)
    assert result.regime == "bull_trend"
    assert result.strategy == "breakout"
    assert result.action == "buy_candidate"
