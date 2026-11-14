from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.adaptive import AdaptiveConfig, AdaptiveDecision
from tradingagents.crypto.backtest.adaptive_study import (
    AdaptiveExecutionCosts,
    AdaptivePortfolio,
    run_adaptive_backtest,
    run_adaptive_walk_forward_study,
)

pytestmark = pytest.mark.unit


def _flat_frame(size: int = 420) -> pd.DataFrame:
    close = np.full(size, 100.0)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.full(size, 1000.0)},
        index=pd.date_range("2025-01-01", periods=size, freq="h", tz="UTC"),
    )


def test_backtest_uses_next_open_and_applies_two_sided_costs(monkeypatch):
    import tradingagents.crypto.backtest.adaptive_study as module

    frame = _flat_frame(90)
    calls = {"count": 0}

    def fake_analyze(data, features, index, **kwargs):
        calls["count"] += 1
        if index == 55:
            return AdaptiveDecision(
                "buy_candidate", "bull_trend", "trend_pullback", .8,
                100.0, 99.0, 103.0, ("test setup",), (), {},
            )
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None)

    monkeypatch.setattr(module, "_analyze_market_at", fake_analyze)
    result = run_adaptive_backtest(
        frame, execution=AdaptiveExecutionCosts(10, 1, 2),
        portfolio=AdaptivePortfolio(initial_cash=1000, risk_fraction=.01, max_notional_fraction=.5),
        start_index=55, end_index=60,
    )
    assert calls["count"] >= 1
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_time == frame.index[56].isoformat()
    assert trade.entry_price > frame.iloc[56].open
    assert trade.fees > 0
    assert result.total_spread_slippage > 0


def test_walk_forward_has_disjoint_oos_folds_and_cost_scenarios():
    frame = _flat_frame()
    report = run_adaptive_walk_forward_study(
        {"BTC/USDT": frame, "ETH/USDT": frame.copy()},
        warmup_bars=250, test_bars=60,
    )
    assert report["parameters_fitted_on_oos"] is False
    assert len(report["folds"]) == 2
    assert report["folds"][0]["oos_end"] < report["folds"][1]["oos_start"]
    assert set(report["summary"]) == {"frictionless", "realistic_estimate", "conservative_estimate"}
    assert report["summary"]["realistic_estimate"]["trades"] == 0
    assert "mean_buy_hold_return_pct" in report["summary"]["realistic_estimate"]
    assert "buy_hold_return_pct" in report["folds"][0]["scenarios"]["realistic_estimate"]["symbols"]["BTC/USDT"]


def test_walk_forward_rejects_misaligned_symbols():
    with pytest.raises(ValueError, match="aligned bar counts"):
        run_adaptive_walk_forward_study(
            {"BTC/USDT": _flat_frame(420), "ETH/USDT": _flat_frame(421)},
            warmup_bars=250, test_bars=60,
        )


def test_execution_costs_and_portfolio_reject_invalid_values():
    with pytest.raises(ValueError, match="finite and non-negative"):
        AdaptiveExecutionCosts(float("inf"), 0, 0)
    with pytest.raises(ValueError, match="risk_fraction"):
        AdaptivePortfolio(risk_fraction=.5)


def test_execution_cost_scenario_sets_matching_signal_cost_gate(monkeypatch):
    import tradingagents.crypto.backtest.adaptive_study as module

    frame = _flat_frame(70)
    observed = []

    def no_trade(data, features, index, **kwargs):
        observed.append(kwargs["config"].estimated_round_trip_cost_bps)
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None)

    monkeypatch.setattr(module, "_analyze_market_at", no_trade)
    run_adaptive_backtest(
        frame, execution=AdaptiveExecutionCosts(0, 0, 0),
        start_index=55, end_index=60,
    )
    run_adaptive_backtest(
        frame, execution=AdaptiveExecutionCosts(15, 3, 5),
        start_index=55, end_index=60,
    )
    assert observed[0] == 0
    assert observed[-1] == 46


def test_default_simulated_portfolio_inherits_adaptive_risk_caps(monkeypatch):
    import tradingagents.crypto.backtest.adaptive_study as module

    frame = _flat_frame(70)

    def candidate(data, features, index, **kwargs):
        return AdaptiveDecision(
            "buy_candidate", "bull_trend", "trend_pullback", .8,
            100.0, 99.0, 103.0, ("test setup",), (), {},
        )

    monkeypatch.setattr(module, "_analyze_market_at", candidate)
    result = run_adaptive_backtest(
        frame,
        config=AdaptiveConfig(risk_fraction=0.01, max_notional_fraction=0.2),
        execution=AdaptiveExecutionCosts(0, 0, 0),
        start_index=55, end_index=60,
    )
    assert result.trades
    assert result.trades[0].quantity * result.trades[0].entry_price == pytest.approx(2000.0)


def test_next_open_gap_that_erases_net_reward_risk_is_rejected(monkeypatch):
    import tradingagents.crypto.backtest.adaptive_study as module

    frame = _flat_frame(90)
    frame.iloc[56, frame.columns.get_loc("open")] = 102.8
    frame.iloc[56, frame.columns.get_loc("high")] = 103.0
    frame.iloc[56, frame.columns.get_loc("low")] = 102.7
    frame.iloc[56, frame.columns.get_loc("close")] = 102.85

    def fake_analyze(data, features, index, **kwargs):
        if index == 55:
            return AdaptiveDecision(
                "buy_candidate", "bull_trend", "trend_pullback", .8,
                100.0, 99.0, 103.0, ("test setup",), (), {},
            )
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None)

    monkeypatch.setattr(module, "_analyze_market_at", fake_analyze)
    result = run_adaptive_backtest(
        frame,
        execution=AdaptiveExecutionCosts(0, 0, 0),
        portfolio=AdaptivePortfolio(initial_cash=1000, risk_fraction=.01, max_notional_fraction=.5),
        start_index=55,
        end_index=60,
    )

    assert result.trades == []
    assert result.rejections["next-open gap/cost invalidated net reward-risk"] == 1


def test_risk_sizing_includes_estimated_round_trip_execution_costs(monkeypatch):
    import tradingagents.crypto.backtest.adaptive_study as module

    frame = _flat_frame(90)

    def fake_analyze(data, features, index, **kwargs):
        if index == 55:
            return AdaptiveDecision(
                "buy_candidate", "bull_trend", "trend_pullback", .8,
                100.0, 99.0, 103.0, ("test setup",), (), {},
            )
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None)

    monkeypatch.setattr(module, "_analyze_market_at", fake_analyze)
    result = run_adaptive_backtest(
        frame,
        execution=AdaptiveExecutionCosts(10, 1, 2),
        portfolio=AdaptivePortfolio(
            initial_cash=1000, risk_fraction=.01, max_notional_fraction=1.0
        ),
        start_index=55,
        end_index=60,
    )

    assert result.trades
    trade = result.trades[0]
    round_trip_cost_per_unit = trade.entry_price * 26 / 10_000
    modeled_risk = trade.quantity * (
        trade.entry_price - trade.stop_price + round_trip_cost_per_unit
    )
    assert modeled_risk <= 10.0
