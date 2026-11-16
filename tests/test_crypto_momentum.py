from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.momentum import (
    MomentumConfig, MomentumCosts, run_momentum_screen,
)

pytestmark = pytest.mark.unit


def _frames(size: int = 420) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC")
    result = {}
    for number, symbol in enumerate(("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT")):
        drift = 0.0008 + number * 0.00015
        close = 100 * np.exp(np.arange(size) * drift)
        result[symbol] = pd.DataFrame({
            "open": close * 0.9995, "high": close * 1.002,
            "low": close * 0.998, "close": close, "volume": np.full(size, 1000.0),
        }, index=index)
    return result


def test_momentum_screen_uses_costs_and_next_open_fills():
    frames = _frames()
    result = run_momentum_screen(frames, costs=MomentumCosts(10, 1, 2))
    assert result["research_only"] is True
    assert result["round_trip_cost_bps"] == 26
    assert result["trades"] > 0
    assert all(t["entry_time"] > t["signal_time"] for t in result["trade_log"])
    assert all(t["exit_time"] > t["entry_time"] for t in result["trade_log"])
    assert all(t["cost_round_trip_bps"] == 26 for t in result["trade_log"])


def test_conservative_costs_cannot_improve_returns():
    frames = _frames()
    realistic = run_momentum_screen(frames, costs=MomentumCosts(10, 1, 2))
    conservative = run_momentum_screen(frames, costs=MomentumCosts(15, 3, 5))
    assert conservative["compounded_portfolio_return_pct"] <= realistic["compounded_portfolio_return_pct"]


def test_momentum_screen_rejects_missing_common_hour():
    frames = _frames()
    frames["ETH/USDT"] = frames["ETH/USDT"].drop(frames["ETH/USDT"].index[250])
    with pytest.raises(ValueError, match="missing or irregular"):
        run_momentum_screen(frames, config=MomentumConfig(lookback_bars=24, trend_bars=48))


def test_momentum_screen_requires_btc_market_filter():
    frames = _frames()
    frames["BTC/USDT"]["close"] = np.linspace(100, 20, len(frames["BTC/USDT"]))
    frames["BTC/USDT"]["open"] = frames["BTC/USDT"]["close"] * 0.999
    frames["BTC/USDT"]["high"] = frames["BTC/USDT"]["close"] * 1.002
    frames["BTC/USDT"]["low"] = frames["BTC/USDT"]["close"] * 0.998
    result = run_momentum_screen(frames, costs=MomentumCosts(10, 1, 2))
    assert result["trades"] == 0
