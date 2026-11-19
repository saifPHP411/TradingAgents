from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.crypto.scalping import evaluate_scalping_signal


def _frame(periods: int, frequency: str, slope: float) -> pd.DataFrame:
    x = np.arange(periods, dtype=float)
    close = 100.0 + slope * x + 0.12 * np.sin(x * 0.8)
    return pd.DataFrame(
        {
            "open": close - slope * 0.2,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close,
            "volume": 1000.0,
        },
        index=pd.date_range("2026-01-01", periods=periods, freq=frequency, tz="UTC"),
    )


def _indicators(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = frame.close
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = (100.0 * avg_gain / (avg_gain + avg_loss).replace(0, np.nan)).fillna(50.0)
    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    return rsi, ema9


def _long_setup() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    one = _frame(100, "1min", 0.03)
    closed = one.copy()
    closed.iloc[-2, closed.columns.get_loc("close")] -= 0.30
    closed.iloc[-2, closed.columns.get_loc("low")] -= 0.32
    _, ema = _indicators(closed)
    closed.iloc[-1, closed.columns.get_loc("close")] = float(ema.iloc[-1]) - 0.005
    closed.iloc[-1, closed.columns.get_loc("low")] = min(
        float(closed.low.iloc[-1]), float(closed.close.iloc[-1]) - 0.02
    )
    closed.iloc[-1, closed.columns.get_loc("high")] = max(
        float(closed.high.iloc[-1]), float(closed.close.iloc[-1]) + 0.15
    )
    forming = closed.iloc[[-1]].copy()
    forming.index = forming.index + pd.Timedelta(minutes=1)
    mark = float(closed.close.iloc[-1]) + 0.20
    forming.iloc[0, forming.columns.get_loc("open")] = float(closed.close.iloc[-1])
    forming.iloc[0, forming.columns.get_loc("high")] = mark + 0.10
    forming.iloc[0, forming.columns.get_loc("low")] = float(closed.close.iloc[-1]) - 0.10
    forming.iloc[0, forming.columns.get_loc("close")] = mark
    return (
        pd.concat([closed, forming]),
        _frame(80, "5min", 0.06),
        _frame(80, "15min", 0.06),
        mark,
    )


def _invert(frame: pd.DataFrame, pivot: float = 300.0) -> pd.DataFrame:
    result = frame.copy()
    old_high = frame.high.copy()
    old_low = frame.low.copy()
    result["open"] = pivot - frame.open
    result["high"] = pivot - old_low
    result["low"] = pivot - old_high
    result["close"] = pivot - frame.close
    return result


def test_scalper_emits_long_on_live_one_minute_reclaim_with_aligned_higher_timeframes():
    one, five, fifteen, mark = _long_setup()
    result = evaluate_scalping_signal(one, five, fifteen, mark)

    assert result["signal"] == 1
    assert result["reason"] == "long_1m_reclaim"
    assert result["stop_loss"] < mark < result["take_profit"]
    assert result["reward_risk"] == 2.0


def test_scalper_emits_short_on_live_one_minute_reclaim_with_aligned_higher_timeframes():
    one, five, fifteen, mark = _long_setup()
    result = evaluate_scalping_signal(
        _invert(one), _invert(five), _invert(fifteen), 300.0 - mark
    )

    assert result["signal"] == -1
    assert result["reason"] == "short_1m_reclaim"
    assert result["take_profit"] < 300.0 - mark < result["stop_loss"]
