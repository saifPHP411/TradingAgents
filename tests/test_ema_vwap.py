from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.crypto.backtest.ema_vwap import (
    EmaVwapCosts,
    _indicators,
    _simulate_symbol,
    resample_3m,
    run_ema_vwap_study,
)

pytestmark = pytest.mark.unit


def _minute_frames(prices: list[float], *, start: str = "2026-01-01T00:00:00Z"):
    start_at = pd.Timestamp(start)
    rows = []
    stamps = []
    for i, price in enumerate(prices):
        for minute in range(3):
            stamps.append(start_at + pd.Timedelta(minutes=3 * i + minute))
            rows.append({
                "open": price, "high": price * 1.001, "low": price * 0.999,
                "close": price, "volume": 100.0,
            })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))


def _trend_reversal_prices() -> list[float]:
    return [100.0] * 30 + [100.5 + i * 0.6 for i in range(12)] + [104.0] * 4 + [98.0] * 14


def test_resample_3m_keeps_only_complete_consecutive_groups():
    frame = _minute_frames([100.0, 101.0, 102.0, 103.0])
    frame = frame.drop(frame.index[4])
    result = resample_3m(frame, "BTC/USDT")
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert result.index[1] == pd.Timestamp("2026-01-01T00:06:00Z")
    assert result.iloc[0].volume == 300.0


def test_vwap_resets_at_utc_midnight():
    bars = pd.DataFrame({
        "open": [10.0, 20.0], "high": [10.0, 20.0],
        "low": [10.0, 20.0], "close": [10.0, 20.0],
        "volume": [5.0, 5.0],
    }, index=pd.to_datetime(["2026-01-01T23:57:00Z", "2026-01-02T00:00:00Z"]))
    result = _indicators(bars)
    assert result.vwap.iloc[0] == pytest.approx(10.0)
    assert result.vwap.iloc[1] == pytest.approx(20.0)


def test_entry_and_exit_are_filled_on_the_next_bar_not_signal_close():
    bars = resample_3m(_minute_frames(_trend_reversal_prices()), "BTC/USDT")
    trades, _ = _simulate_symbol("BTC/USDT", bars, EmaVwapCosts(10, 1, 2))
    assert trades
    trade = trades[0]
    assert pd.Timestamp(trade["entry_time"]) > pd.Timestamp(trade["signal_time"])
    if trade["exit_reason"] != "end_of_data":
        assert pd.Timestamp(trade["exit_time"]) > pd.Timestamp(trade["exit_signal_time"])
        assert trade["exit_reason"] == "closed_below_ema9_next_open"
    assert trade["round_trip_cost_bps"] == 26


def test_conservative_costs_cannot_improve_same_trade_returns():
    bars = resample_3m(_minute_frames(_trend_reversal_prices()), "BTC/USDT")
    realistic, _ = _simulate_symbol("BTC/USDT", bars, EmaVwapCosts(10, 1, 2))
    conservative, _ = _simulate_symbol("BTC/USDT", bars, EmaVwapCosts(15, 3, 5))
    assert len(realistic) == len(conservative)
    assert sum(t["net_return_pct"] for t in conservative) <= sum(
        t["net_return_pct"] for t in realistic
    )


def test_study_is_spot_long_only_and_reports_costs():
    frames = {"BTC/USDT": _minute_frames(_trend_reversal_prices())}
    result = run_ema_vwap_study(frames, costs=EmaVwapCosts(10, 1, 2))
    assert result["research_only"] is True
    assert result["timeframe"] == "3m aggregated from complete 1m bars"
    assert result["round_trip_cost_bps"] == 26
    assert result["rules"]["direction"] == "long-only Spot proxy; no shorting or options"
    assert len(result["chronological_thirds_pct"]) == 3
    assert result["trades"] == result["per_symbol"]["BTC/USDT"]["trades"]


def test_rejects_naive_or_invalid_ohlcv():
    frame = _minute_frames([100.0] * 10)
    frame.index = frame.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        resample_3m(frame, "BTC/USDT")
