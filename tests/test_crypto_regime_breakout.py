from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.regime_breakout import (
    RegimeBreakoutSettings,
    entry_levels,
    indicators,
)

pytestmark = pytest.mark.unit


def _bars(size: int = 100) -> pd.DataFrame:
    close = 100.0 + np.arange(size) * 0.1
    frame = pd.DataFrame(
        {
            "open": close - 0.03,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(size, 1000.0),
        },
        index=pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC"),
    )
    return frame


def test_breakout_reference_excludes_the_current_bar():
    frame = _bars()
    frame.iloc[-1, frame.columns.get_loc("high")] = 500.0
    out = indicators(frame)
    expected = frame["high"].iloc[-21:-1].max()
    assert out["breakout_level"].iloc[-1] == pytest.approx(expected)
    assert out["breakout_level"].iloc[-1] < 500.0


def test_volume_baseline_excludes_current_bar():
    frame = _bars()
    frame.iloc[-1, frame.columns.get_loc("volume")] = 100_000.0
    out = indicators(frame)
    assert out["volume_baseline"].iloc[-1] == pytest.approx(1000.0)


def test_trend_breakout_with_volume_confirms_long_signal():
    frame = _bars()
    frame.iloc[-1, frame.columns.get_loc("close")] += 4.0
    frame.iloc[-1, frame.columns.get_loc("high")] += 4.0
    frame.iloc[-1, frame.columns.get_loc("volume")] = 2000.0
    out = indicators(frame)
    assert bool(out["long_entry"].iloc[-1]) is True
    levels = entry_levels(out.iloc[-1])
    assert levels is not None
    entry, stop, target = levels
    assert stop < entry < target


def test_sideways_or_low_volume_breakout_is_not_accepted():
    frame = _bars()
    frame.iloc[-1, frame.columns.get_loc("close")] += 4.0
    frame.iloc[-1, frame.columns.get_loc("high")] += 4.0
    # The breakout is real, but volume does not confirm it.
    out = indicators(frame)
    assert bool(out["long_entry"].iloc[-1]) is False


def test_invalid_ohlcv_and_nonchronological_data_are_rejected():
    frame = _bars()
    frame.loc[frame.index[-1], "close"] = 0.0
    with pytest.raises(ValueError, match="prices must be positive"):
        indicators(frame)
    frame = _bars()
    frame = frame.iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        indicators(frame)


def test_settings_validate_periods_and_risk_geometry():
    with pytest.raises(ValueError, match="fast_ema"):
        RegimeBreakoutSettings(fast_ema=1)
    with pytest.raises(ValueError, match="smaller"):
        RegimeBreakoutSettings(fast_ema=50, slow_ema=20)
    with pytest.raises(ValueError, match="positive"):
        RegimeBreakoutSettings(reward_risk=0)
