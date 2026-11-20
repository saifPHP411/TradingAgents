import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_ema import PerpetualEMAStrategy, run_backtest
from tradingagents.crypto.perpetuals import LinearPerpetualInstrument, PerpetualRiskSettings


def test_enabled_research_backtest_requires_aligned_mark_and_funding_data():
    index = pd.date_range("2024-01-01", periods=221, freq="4h", tz="UTC")
    close = 100 + np.linspace(0, 15, len(index)) + np.sin(np.arange(len(index)) / 7)
    traded = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000.0},
        index=index,
    )
    mark = traded[["open", "high", "low", "close"]].copy()
    funding = pd.Series(0.0001, index=index[::2])
    instrument = LinearPerpetualInstrument(
        "BTCUSDT", "BTC", "USDT", "USDT", "Trading", "LinearPerpetual",
        0.001, 100.0, 0.001, 5.0, 0.1, 1.0, 50.0, 0.01, 480,
    )
    result = run_backtest(
        traded, mark, funding, instrument,
        strategy=PerpetualEMAStrategy(enabled=True, risk=PerpetualRiskSettings()),
    )
    assert result.final_equity == 1_000.0
    assert result.assumptions["demo_ready"] is False
    assert result.assumptions["mark_price_liquidation"] is True


@pytest.mark.parametrize("entry_bar_low", [99.0, 90.0])
def test_external_research_signal_executes_on_next_open(entry_bar_low):
    index = pd.date_range("2024-01-01", periods=240, freq="4h", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    mark = bars.copy()
    bars.loc[index[226], "low"] = entry_bar_low
    signals = pd.Series(0, index=index)
    signals.iloc[225] = 1
    instrument = LinearPerpetualInstrument(
        "BTCUSDT", "BTC", "USDT", "USDT", "Trading", "LinearPerpetual",
        0.001, 100.0, 0.001, 5.0, 0.1, 1.0, 50.0, 0.01, 480,
    )
    result = run_backtest(
        bars, mark, pd.Series(0.0, index=index[::2]), instrument,
        strategy=PerpetualEMAStrategy(enabled=True, signal_mode="donchian_breakout"),
        entry_signals=signals, trade_start_index=220,
    )
    assert len(result.trades) == 1
    assert result.trades[0].entry_time == index[226].isoformat()
    if entry_bar_low == 90.0:
        assert result.trades[0].exit_time == index[226].isoformat()
        assert result.trades[0].exit_reason == "atr_stop"
