from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.ensemble import EnsembleSettings, evaluate_ensemble


pytestmark = pytest.mark.unit


def _frame(close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })


def test_requires_completed_history_and_valid_ohlcv():
    with pytest.raises(ValueError, match="220"):
        evaluate_ensemble(_frame(np.linspace(100, 110, 100)))
    bad = _frame(np.linspace(100, 110, 240))
    bad.loc[239, "high"] = 1
    with pytest.raises(ValueError, match="high"):
        evaluate_ensemble(bad)


def test_ambiguous_regime_fails_flat():
    close = np.array([100 + (i % 2) * 0.02 for i in range(260)], dtype=float)
    result = evaluate_ensemble(_frame(close))
    assert result.action == "flat"
    assert result.regime in {"range", "uncertain", "trend"}
    assert result.stop is None


def test_trend_module_can_emit_a_long_signal():
    close = np.linspace(100, 130, 260)
    close[-1] = close[-2] + 1.1
    volume = np.full(len(close), 1000.0)
    volume[-1] = 1500.0
    frame = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })
    result = evaluate_ensemble(frame)
    assert result.regime == "trend"
    assert result.action == "long"
    assert result.module == "trend_donchian_momentum"
    assert result.stop is not None and result.stop < result.close


def test_trend_no_entry_is_not_mistaken_for_an_exit():
    close = np.linspace(100, 130, 260)
    close[-1] = close[-2] + 0.1
    frame = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })
    result = evaluate_ensemble(frame)
    assert result.regime == "trend"
    assert result.action == "flat"
    assert result.should_exit is False


def test_trend_pullback_module_can_emit_a_reclaim_entry():
    close = np.linspace(100, 109, 300)
    close[-3] = close[-4] - 1.2
    close[-2] = close[-3] - 0.2
    close[-1] = close[-2] + 1.8
    frame = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })
    frame.loc[frame.index[-10], "high"] = close[-1] + 0.5
    frame.loc[frame.index[-1], "volume"] = 1500.0
    assert evaluate_ensemble(frame, mode="trend").action == "flat"
    result = evaluate_ensemble(frame, mode="trend_pullback")
    assert result.regime == "trend"
    assert result.action == "long"
    assert result.module == "trend_pullback_momentum"
    assert result.stop is not None and result.stop < result.close


def test_breakout_module_requires_atr_expansion():
    close = np.linspace(100, 130, 260)
    close[-1] = close[-2] + 1.1
    volume = np.full(len(close), 1000.0)
    volume[-1] = 1500.0
    frame = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })
    no_expansion = evaluate_ensemble(frame, mode="breakout")
    assert no_expansion.action == "flat"
    frame.loc[259, "high"] = close[-1] + 4.0
    frame.loc[259, "low"] = close[-1] - 4.0
    expanded = evaluate_ensemble(frame, mode="breakout")
    assert expanded.action == "long"
    assert expanded.module == "volatility_breakout"


def test_range_module_can_emit_a_long_reversal():
    close = 100 + np.sin(np.arange(260) * 0.7) * 0.1
    close[-2] = 97.8
    close[-1] = 98.6
    frame = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })
    result = evaluate_ensemble(frame)
    assert result.regime == "range"
    assert result.action == "long"
    assert result.module == "range_mean_reversion"


def test_signal_is_based_on_last_bar_and_returns_a_defined_stop():
    # Strong rising series. The final close clears the previous 20-bar high.
    close = np.linspace(100, 150, 260)
    close[-1] = close[-2] * 1.025
    result = evaluate_ensemble(_frame(close))
    assert result.close == pytest.approx(close[-1])
    if result.action == "long":
        assert result.stop is not None and result.stop < result.close
        assert result.module == "trend_donchian_momentum"


def test_current_bar_high_is_excluded_from_its_donchian_breakout_threshold():
    close = np.linspace(100, 130, 260)
    close[-1] = close[-2] + 1.1
    volume = np.full(len(close), 1000.0)
    volume[-1] = 1500.0
    frame = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })
    frame.loc[259, "high"] = close[-1] + 4.0
    result = evaluate_ensemble(frame)
    assert result.action == "long"
    assert result.module == "trend_donchian_momentum"


def test_rejects_infinite_and_negative_volume():
    frame = _frame(np.linspace(100, 110, 240))
    frame.loc[239, "close"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_ensemble(frame)
    frame = _frame(np.linspace(100, 110, 240))
    frame.loc[239, "volume"] = -1
    with pytest.raises(ValueError, match="volume"):
        evaluate_ensemble(frame)


def test_rejects_unsorted_and_duplicate_bar_indexes():
    frame = _frame(np.linspace(100, 110, 240))
    frame.index = frame.index[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_ensemble(frame)
    frame = _frame(np.linspace(100, 110, 240))
    frame.index = list(range(239)) + [238]
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_ensemble(frame)


def test_range_only_mode_does_not_trade_an_uncertain_regime():
    close = 100 + np.sin(np.arange(260) * 0.7) * 0.1
    close[-2] = 98.0
    close[-1] = 98.5
    frame = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })
    settings = EnsembleSettings(
        adx_range_max=0.0,
        adx_trend_min=100.0,
        range_zscore=0.1,
        range_rsi_max=100.0,
    )
    result = evaluate_ensemble(frame, settings, mode="range")
    assert result.regime == "uncertain"
    assert result.action == "flat"
    assert result.should_exit is True


def test_rejects_non_finite_indicator_thresholds():
    frame = _frame(np.linspace(100, 110, 240))
    with pytest.raises(ValueError, match="finite"):
        evaluate_ensemble(frame, EnsembleSettings(volume_multiple=float("nan")))
    with pytest.raises(ValueError, match="finite"):
        evaluate_ensemble(frame, EnsembleSettings(atr_stop_multiple=float("inf")))


def test_rejects_invalid_indicator_settings():
    frame = _frame(np.linspace(100, 110, 240))
    with pytest.raises(ValueError, match="fast_ema"):
        evaluate_ensemble(frame, EnsembleSettings(fast_ema=200, slow_ema=50))


def test_no_lookahead_in_donchian_reference():
    close = np.linspace(100, 130, 260)
    frame = _frame(close)
    settings = EnsembleSettings()
    # The final bar's high is not part of its own breakout threshold.
    baseline = evaluate_ensemble(frame, settings)
    changed = frame.copy()
    changed.loc[changed.index[-1], "high"] *= 1.5
    altered = evaluate_ensemble(changed, settings)
    assert baseline.action == altered.action
    assert baseline.regime == altered.regime
