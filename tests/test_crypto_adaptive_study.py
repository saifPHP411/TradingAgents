from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.adaptive_study import (
    AdaptiveExecutionCosts,
    AdaptivePortfolio,
    run_adaptive_backtest,
    run_adaptive_walk_forward_study,
)
from tradingagents.crypto.adaptive import AdaptiveDecision

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

    def fake_analyze(history, **kwargs):
        calls["count"] += 1
        if len(history) == 56:
            return AdaptiveDecision(
                "buy_candidate", "bull_trend", "trend_pullback", .8,
                100.0, 99.0, 103.0, ("test setup",), (), {},
            )
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None)

    monkeypatch.setattr(module, "analyze_market", fake_analyze)
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
