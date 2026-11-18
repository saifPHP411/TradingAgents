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
    index = pd.date_range("2025-01-01", periods=330, freq="4h", tz="UTC")
    close = np.full(len(index), 100.0)
    traded = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=index,
    )
    return traded, traded.copy(), pd.Series(0.0, index=index[::2]), index


def _forced_features(index, *, target=105.0, adx=None, plus_di=None, minus_di=None):
    signal = pd.Series(0, index=index, dtype=int)
    signal.iloc[270] = 1
    adx_values = pd.Series(10.0, index=index)
    if adx:
        for position, value in adx.items():
            adx_values.iloc[position] = value
    plus_values = pd.Series(20.0, index=index)
    minus_values = pd.Series(10.0, index=index)
    for position, value in (plus_di or {}).items():
        plus_values.iloc[position] = value
    for position, value in (minus_di or {}).items():
        minus_values.iloc[position] = value
    return pd.DataFrame(
        {
            "atr": 1.0,
            "signal": signal,
            "adx": adx_values,
            "plus_di": plus_values,
            "minus_di": minus_values,
            "mean_target": target,
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mean_reversion_exit_adx": 17.0}, "mean_reversion_exit_adx"),
        ({"mean_reversion_exit_confirmation_bars": 0}, "exit_confirmation_bars"),
        ({"mean_reversion_exit_confirmation_bars": True}, "exit_confirmation_bars"),
        ({"mean_reversion_min_reward_risk": -0.1}, "min_reward_risk"),
        ({"mean_reversion_min_reward_risk": 5.1}, "min_reward_risk"),
        ({"mean_reversion_min_rsi_recovery": -1.0}, "min_rsi_recovery"),
        ({"mean_reversion_require_directional_confirmation": 1}, "directional_confirmation"),
        ({"mean_reversion_exit_requires_adverse_di": "yes"}, "adverse_di"),
    ],
)
def test_mean_reversion_controls_validate_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PerpetualEMAStrategy(**kwargs)


def test_hysteresis_exit_requires_consecutive_completed_adx_bars(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _history()
    features = _forced_features(index, adx={272: 27.0, 273: 28.0})
    monkeypatch.setattr(engine, "_features", lambda data, cfg: features.loc[data.index])

    result = run_backtest(
        traded,
        mark,
        funding,
        _instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            signal_mode="bollinger_reversion",
            mean_reversion_max_adx=18.0,
            mean_reversion_exit_adx=26.0,
            mean_reversion_exit_confirmation_bars=2,
            risk=PerpetualRiskSettings(leverage=1.0),
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "trend_regime_exit"
    assert result.trades[0].exit_time == index[274].isoformat()


def test_single_adx_spike_does_not_trigger_hysteresis_exit(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _history()
    features = _forced_features(index, adx={272: 27.0, 273: 10.0})
    monkeypatch.setattr(engine, "_features", lambda data, cfg: features.loc[data.index])

    result = run_backtest(
        traded,
        mark,
        funding,
        _instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            signal_mode="bollinger_reversion",
            mean_reversion_max_adx=18.0,
            mean_reversion_exit_adx=26.0,
            mean_reversion_exit_confirmation_bars=2,
            risk=PerpetualRiskSettings(leverage=1.0),
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_window"


def test_minimum_reward_risk_filter_rejects_weak_mid_band_target(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _history()
    features = _forced_features(index, target=100.5)
    monkeypatch.setattr(engine, "_features", lambda data, cfg: features.loc[data.index])

    result = run_backtest(
        traded,
        mark,
        funding,
        _instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            signal_mode="bollinger_reversion",
            mean_reversion_min_reward_risk=0.5,
            risk=PerpetualRiskSettings(leverage=1.0),
        ),
    )

    assert result.trades == []
    assert result.rejections["target_reward_risk_below_minimum"] == 1


def test_adverse_di_exit_only_triggers_when_directional_movement_opposes_position(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _history()
    adverse = _forced_features(
        index,
        adx={272: 27.0, 273: 28.0},
        plus_di={272: 8.0, 273: 7.0},
        minus_di={272: 20.0, 273: 21.0},
    )
    favorable = _forced_features(
        index,
        adx={272: 27.0, 273: 28.0},
        plus_di={272: 20.0, 273: 21.0},
        minus_di={272: 8.0, 273: 7.0},
    )

    cfg = PerpetualEMAStrategy(
        enabled=True,
        signal_mode="bollinger_reversion",
        mean_reversion_max_adx=18.0,
        mean_reversion_exit_adx=26.0,
        mean_reversion_exit_confirmation_bars=2,
        mean_reversion_exit_requires_adverse_di=True,
        risk=PerpetualRiskSettings(leverage=1.0),
    )

    monkeypatch.setattr(engine, "_features", lambda data, strategy: adverse.loc[data.index])
    adverse_result = run_backtest(traded, mark, funding, _instrument(), strategy=cfg)
    assert len(adverse_result.trades) == 1
    assert adverse_result.trades[0].exit_reason == "trend_regime_exit"
    assert adverse_result.trades[0].exit_time == index[274].isoformat()

    monkeypatch.setattr(engine, "_features", lambda data, strategy: favorable.loc[data.index])
    favorable_result = run_backtest(traded, mark, funding, _instrument(), strategy=cfg)
    assert len(favorable_result.trades) == 1
    assert favorable_result.trades[0].exit_reason == "end_of_window"


def test_features_expose_directional_indicators_for_regime_diagnostics():
    from tradingagents.crypto.backtest.perpetual_ema import _features

    traded, _, _, _ = _history()
    features = _features(
        traded,
        PerpetualEMAStrategy(enabled=True, signal_mode="bollinger_reversion"),
    )
    assert {"plus_di", "minus_di"}.issubset(features.columns)
    assert features["plus_di"].notna().any()
    assert features["minus_di"].notna().any()
