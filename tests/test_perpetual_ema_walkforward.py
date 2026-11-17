from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualEMAStrategy,
    run_backtest,
)
from tradingagents.crypto.perpetuals import (
    LinearPerpetualInstrument,
    PerpetualRiskSettings,
)


def _instrument():
    return LinearPerpetualInstrument(
        "BTCUSDT", "BTC", "USDT", "USDT", "Trading", "LinearPerpetual",
        0.001, 100.0, 0.001, 5.0, 0.1, 1.0, 50.0, 0.01, 480,
    )


def _history():
    index = pd.date_range("2025-01-01", periods=320, freq="4h", tz="UTC")
    close = 100.0 + np.sin(np.arange(len(index)) / 10.0) * 0.1
    traded = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=index,
    )
    mark = traded.copy()
    funding = pd.Series(0.0, index=index[::2], name="funding_rate")
    return traded, mark, funding, index


def test_walkforward_warmup_signals_are_not_traded(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _history()
    signals = pd.Series(0, index=index, dtype=int)
    signals.loc[index[230]] = 1  # warm-up signal: must be ignored
    signals.loc[index[270]] = 1  # first OOS signal: may enter next bar

    def forced_features(data, cfg):
        return pd.DataFrame(
            {"atr": pd.Series(1.0, index=data.index), "signal": signals.loc[data.index]},
            index=data.index,
        )

    monkeypatch.setattr(engine, "_features", forced_features)
    result = run_backtest(
        traded,
        mark,
        funding,
        _instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            risk=PerpetualRiskSettings(leverage=1.0),
        ),
        trade_start_index=270,
    )

    assert len(result.trades) == 1
    assert result.trades[0].entry_time == index[271].isoformat()


@pytest.mark.parametrize("value", [True, 0, -1, 320, 1.5])
def test_trade_start_index_rejects_invalid_values(value):
    traded, mark, funding, _ = _history()
    with pytest.raises(ValueError, match="trade_start_index"):
        run_backtest(
            traded,
            mark,
            funding,
            _instrument(),
            strategy=PerpetualEMAStrategy(
                enabled=True,
                risk=PerpetualRiskSettings(leverage=1.0),
            ),
            trade_start_index=value,
        )
