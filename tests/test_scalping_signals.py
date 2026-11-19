from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.crypto.scalping import ScalpingConfig, evaluate_scalping_signal


def _frame(periods: int, frequency: str, slope: float, *, volume: bool = False) -> pd.DataFrame:
    x = np.arange(periods, dtype=float)
    close = 100.0 + slope * x + 0.12 * np.sin(x * 0.8)
    frame = pd.DataFrame(
        {
            "open": close - slope * 0.2,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close,
        },
        index=pd.date_range("2026-01-01", periods=periods, freq=frequency, tz="UTC"),
    )
    if volume:
        frame["volume"] = 1000.0
    return frame


def test_scalper_config_keeps_demo_leverage_and_short_hold_limits():
    config = ScalpingConfig()
    assert config.leverage == 2.0
    assert config.max_hold_minutes == 15
    assert config.cooldown_minutes == 3


def test_scalper_requires_aligned_5m_and_15m_trend():
    one = _frame(100, "1min", 0.03, volume=True)
    five = _frame(80, "5min", 0.03)
    fifteen = _frame(80, "15min", -0.03)

    result = evaluate_scalping_signal(one, five, fifteen, float(one.close.iloc[-1]))
    assert result["signal"] == 0
    assert result["reason"] == "no_aligned_pullback_reclaim"
