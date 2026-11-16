from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.factor_interaction import (
    FactorConfig,
    FactorCosts,
    run_factor_interaction_study,
)

pytestmark = pytest.mark.unit


def _frames(size: int = 1200) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC")
    result = {}
    for number, symbol in enumerate(
        ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT")
    ):
        t = np.arange(size, dtype=float)
        close = (100 + number * 17) * np.exp(
            t * (0.00015 + number * 0.000025)
            + 0.012 * np.sin(t / (31 + number * 3))
        )
        open_price = close * (1 + 0.0004 * np.sin(t / 7 + number))
        high = np.maximum(open_price, close) * 1.002
        low = np.minimum(open_price, close) * 0.998
        volume = 1000 + 100 * np.sin(t / 19 + number) + number * 50
        result[symbol] = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
    return result


def _config() -> FactorConfig:
    return FactorConfig(
        warmup_bars=336,
        train_window_bars=432,
        test_fold_bars=168,
        holding_bars=24,
        sample_step_bars=24,
        refit_bars=168,
        min_train_rows=30,
    )


def test_factor_interaction_is_research_only_and_uses_next_open_fills():
    report = run_factor_interaction_study(
        _frames(), costs=FactorCosts(0, 0, 0), config=_config()
    )
    assert report["research_only"] is True
    assert report["strategy"] == "weekly_refit_cross_sectional_factor_interaction_ridge"
    assert report["folds"] >= 1
    assert report["aggregate"]["trades"] == len(report["trade_log"])
    assert all(t["entry_time"] > t["signal_time"] for t in report["trade_log"])
    assert all(t["exit_time"] > t["entry_time"] for t in report["trade_log"])
    assert all(0 < t["allocation_pct"] <= 10 for t in report["trade_log"])


def test_training_labels_are_purged_before_each_oos_fold():
    report = run_factor_interaction_study(
        _frames(), costs=FactorCosts(0, 0, 0), config=_config()
    )
    for fold in report["fold_audit"]:
        assert fold["train_label_end"] < fold["test_start"]
        assert fold["train_start"] < fold["train_label_end"]


def test_conservative_cost_threshold_is_higher_than_realistic():
    frames = _frames()
    realistic = run_factor_interaction_study(
        frames, costs=FactorCosts(10, 1, 2), config=_config()
    )
    conservative = run_factor_interaction_study(
        frames, costs=FactorCosts(15, 3, 5), config=_config()
    )
    assert (
        conservative["costs"]["minimum_predicted_gross_return_pct"]
        > realistic["costs"]["minimum_predicted_gross_return_pct"]
    )
    assert conservative["costs"]["round_trip_bps"] == 46
    assert realistic["costs"]["round_trip_bps"] == 26


def test_factor_interaction_rejects_missing_common_hour():
    frames = _frames()
    frames["ETH/USDT"] = frames["ETH/USDT"].drop(frames["ETH/USDT"].index[700])
    with pytest.raises(ValueError, match="common hourly timestamps contain gaps"):
        run_factor_interaction_study(frames, config=_config())


def test_factor_interaction_rejects_short_history():
    with pytest.raises(ValueError, match="at least 1000 common hourly bars"):
        run_factor_interaction_study(_frames(900), config=_config())
