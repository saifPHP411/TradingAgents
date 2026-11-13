from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.regime_breakout import (
    BacktestSettings,
    ExecutionCosts,
    RegimeBreakoutSettings,
    indicators,
    run_backtest,
)

pytestmark = pytest.mark.unit


def _trend_with_breakouts(size: int = 220) -> pd.DataFrame:
    close = 100.0 + np.arange(size) * 0.08
    for index in (75, 145):
        close[index:] += 2.5
    frame = pd.DataFrame(
        {
            "open": close - 0.03,
            "high": close + 0.05,
            "low": close - 0.10,
            "close": close,
            "volume": np.full(size, 1000.0),
        },
        index=pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC"),
    )
    for index in (75, 145):
        frame.iloc[index, frame.columns.get_loc("volume")] = 2500.0
    return frame


def test_signal_bar_cannot_change_its_own_breakout_or_volume_threshold():
    frame = _trend_with_breakouts()
    baseline = indicators(frame)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("high")] = 900.0
    changed.iloc[-1, changed.columns.get_loc("volume")] = 900_000.0
    revised = indicators(changed)
    assert revised["breakout_level"].iloc[-1] == pytest.approx(
        baseline["breakout_level"].iloc[-1]
    )
    assert revised["volume_baseline"].iloc[-1] == pytest.approx(
        baseline["volume_baseline"].iloc[-1]
    )


def test_backtest_fills_after_signal_bar_and_records_trades():
    frame = _trend_with_breakouts()
    result = run_backtest(
        frame,
        strategy=RegimeBreakoutSettings(max_holding_bars=12),
        execution=ExecutionCosts(commission_bps=0, half_spread_bps=0, slippage_bps=0),
        portfolio=BacktestSettings(initial_cash=10_000),
    )
    assert result.trades
    first = result.trades[0]
    assert pd.Timestamp(first.entry_time) > pd.Timestamp(frame.index[0])
    assert first.entry_price > 0
    assert first.stop < first.entry_price < first.target


def test_realistic_execution_costs_reduce_equity_and_are_reported():
    frame = _trend_with_breakouts()
    strategy = RegimeBreakoutSettings(max_holding_bars=12)
    portfolio = BacktestSettings(initial_cash=10_000)
    frictionless = run_backtest(
        frame, strategy=strategy,
        execution=ExecutionCosts(commission_bps=0, half_spread_bps=0, slippage_bps=0),
        portfolio=portfolio,
    )
    costed = run_backtest(
        frame, strategy=strategy,
        execution=ExecutionCosts(commission_bps=10, half_spread_bps=1, slippage_bps=2),
        portfolio=portfolio,
    )
    assert len(costed.trades) == len(frictionless.trades)
    assert costed.total_fees > 0
    assert costed.total_slippage_spread > 0
    assert costed.final_equity < frictionless.final_equity


def test_flat_market_does_not_create_breakout_entries():
    close = np.full(120, 100.0)
    frame = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1,
         "close": close, "volume": np.full(120, 1000.0)},
        index=pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC"),
    )
    assert not indicators(frame)["long_entry"].any()
    result = run_backtest(frame)
    assert result.trades == []
    assert result.final_equity == pytest.approx(10_000.0)
