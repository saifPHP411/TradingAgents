"""Causal alignment, fee accounting and liquidation checks for the replay."""

import numpy as np
import pandas as pd

from weekly import SYMBOLS, episode, features


def test_signal_uses_only_completed_candles():
    idx = pd.date_range("2025-01-01", periods=90, freq="4h", tz="UTC")
    close = np.linspace(25, 50, len(idx))
    bars = pd.DataFrame(
        {"close": close, "high": close + 0.1, "low": close - 0.1}, index=idx
    )
    a = features(bars)
    bars.loc[idx[60] :] *= 2
    b = features(bars)
    pd.testing.assert_frame_equal(a.iloc[:60], b.iloc[:60])
    assert a.index[0] == idx[0] + pd.Timedelta(minutes=15)


def test_perpetual_funding_and_margin():
    n = 7 * 96
    data = np.full((len(SYMBOLS), n, 9), 100.0)
    data[:, :, 8] = 0
    signals = np.zeros((len(SYMBOLS), n, 4))
    signals[2, 0] = [0, 1, 0, 10]
    data[2, 32, 8] = 0.001
    trades = []
    out = episode(data, signals, 0, 1, 10, False, audit=trades)
    assert len(trades) == 1 and out["liquidations"] == 0
    qty = trades[0]["qty"]
    assert np.isclose(out["funding"], -qty * 100 * 0.001)
    assert np.isclose(
        out["end"], 50 + qty * ((99.97 - 100.03) - 0.00055 * 200) - qty * 0.1
    )
    data[2, 5, 6] = 80
    out = episode(data, signals, 0, 1, 10, False)
    assert out["liquidations"] == 1 and 24 < out["end"] < 26
