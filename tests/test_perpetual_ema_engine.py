import numpy as np
import pandas as pd

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
