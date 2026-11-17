from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.turtle_trend import (
    TurtleExecutionCosts,
    TurtlePortfolioSettings,
    TurtleTrendSettings,
    resample_hourly_to_4h,
    run_backtest,
)

pytestmark = pytest.mark.unit


def _hourly_frame(size: int = 12, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=size, freq="h")
    close = np.linspace(100.0, 101.0, size)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(size, 1000.0),
        },
        index=index,
    )


def _features_with_signal(frame, benchmark_close, settings):
    features = frame.copy()
    features["atr"] = 1.0
    features["exit_low"] = 1.0
    features["long_entry"] = False
    features.iloc[1, features.columns.get_loc("long_entry")] = True
    return features


def test_hourly_to_4h_uses_only_complete_utc_buckets():
    hourly = _hourly_frame(9, "2026-01-01T01:00:00Z")
    four_hour = resample_hourly_to_4h(hourly)

    # 01:00-03:00 is incomplete; 04:00-07:00 is one complete bucket;
    # 08:00 is a partial final bucket and must be excluded.
    assert len(four_hour) == 1
    assert four_hour.index[0] == pd.Timestamp("2026-01-01T08:00:00Z")
    assert four_hour.iloc[0]["open"] == pytest.approx(hourly.iloc[3]["open"])
    assert four_hour.iloc[0]["close"] == pytest.approx(hourly.iloc[6]["close"])
    assert four_hour.iloc[0]["volume"] == pytest.approx(4000.0)


def test_hourly_to_4h_fails_closed_on_missing_hour():
    hourly = _hourly_frame(12).drop(pd.Timestamp("2026-01-01T05:00:00Z"))
    with pytest.raises(ValueError, match="missing or irregular 1h"):
        resample_hourly_to_4h(hourly)


def test_breakout_entry_executes_at_next_4h_open(monkeypatch):
    import tradingagents.crypto.backtest.turtle_trend as module

    frame = _hourly_frame(6, "2026-01-01T00:00:00Z")
    monkeypatch.setattr(module, "_features", _features_with_signal)
    result = run_backtest(
        frame,
        execution=TurtleExecutionCosts(10, 1, 2),
        portfolio=TurtlePortfolioSettings(
            initial_cash=1000, risk_fraction=0.01, max_notional_fraction=0.5
        ),
        start_index=0,
        end_index=5,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_time == frame.index[2].isoformat()
    assert trade.entry_price > frame.iloc[2].open
    assert trade.exit_reason == "end_of_window"
    assert trade.fees > 0


def test_atr_stop_gap_fills_at_adverse_open_not_stop(monkeypatch):
    import tradingagents.crypto.backtest.turtle_trend as module

    frame = _hourly_frame(6, "2026-01-01T00:00:00Z")
    frame.iloc[3, frame.columns.get_loc("open")] = 95.0
    frame.iloc[3, frame.columns.get_loc("high")] = 96.0
    frame.iloc[3, frame.columns.get_loc("low")] = 94.0
    frame.iloc[3, frame.columns.get_loc("close")] = 95.0
    monkeypatch.setattr(module, "_features", _features_with_signal)
    result = run_backtest(
        frame,
        execution=TurtleExecutionCosts(0, 1, 2),
        start_index=0,
        end_index=5,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "atr_stop"
    assert trade.exit_price < 95.0


def test_donchian_close_exit_is_filled_at_following_open(monkeypatch):
    import tradingagents.crypto.backtest.turtle_trend as module

    frame = _hourly_frame(6, "2026-01-01T00:00:00Z")
    frame.iloc[3, frame.columns.get_loc("close")] = 98.0
    frame.iloc[3, frame.columns.get_loc("high")] = 99.0
    frame.iloc[3, frame.columns.get_loc("low")] = 97.5
    frame.iloc[4, frame.columns.get_loc("open")] = 97.0
    frame.iloc[4, frame.columns.get_loc("high")] = 98.0
    frame.iloc[4, frame.columns.get_loc("low")] = 96.5
    frame.iloc[4, frame.columns.get_loc("close")] = 97.0
    monkeypatch.setattr(module, "_features", _features_with_signal)
    result = run_backtest(
        frame,
        execution=TurtleExecutionCosts(0, 1, 2),
        start_index=0,
        end_index=5,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "donchian_exit"
    assert trade.exit_time == frame.index[4].isoformat()
    assert trade.exit_price < 97.0


def test_cost_assumptions_are_validated():
    with pytest.raises(ValueError, match="finite and non-negative"):
        TurtleExecutionCosts(float("inf"), 0, 0)
    with pytest.raises(ValueError, match="risk_fraction"):
        TurtlePortfolioSettings(risk_fraction=0.5)
    with pytest.raises(ValueError, match="smaller than entry"):
        TurtleTrendSettings(entry_lookback=20, exit_lookback=20)
