from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.regime_breakout import ExecutionCosts
from tradingagents.crypto.backtest.regime_breakout_study import (
    _buy_and_hold_return,
    load_hourly_frames,
    run_walk_forward_study,
)

pytestmark = pytest.mark.unit


def _flat_frame(size: int = 420) -> pd.DataFrame:
    close = np.full(size, 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(size, 1000.0),
        },
        index=pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC"),
    )


def test_walkforward_uses_disjoint_oos_windows_and_reports_all_cost_scenarios():
    frame = _flat_frame()
    report = run_walk_forward_study(
        {"BTC/USDT": frame, "ETH/USDT": frame.copy()},
        warmup_bars=250,
        test_bars=60,
    )
    assert report["parameters_fitted_on_oos"] is False
    assert report["bar_count_per_symbol"] == 420
    assert len(report["folds"]) == 2
    first, second = report["folds"]
    assert first["oos_end"] < second["oos_start"]
    assert set(report["summary"]) == {
        "frictionless", "realistic_estimate", "conservative_estimate"
    }
    assert report["summary"]["realistic_estimate"]["compounded_oos_return_pct"] == pytest.approx(0)
    assert first["scenarios"]["realistic_estimate"]["symbols"]["BTC/USDT"]["trades"] == 0


def test_walkforward_rejects_misaligned_symbol_history():
    first = _flat_frame()
    second = _flat_frame(421)
    with pytest.raises(ValueError, match="aligned bar counts"):
        run_walk_forward_study(
            {"BTC/USDT": first, "ETH/USDT": second},
            warmup_bars=250,
            test_bars=60,
        )


def test_walkforward_rejects_insufficient_history():
    with pytest.raises(ValueError, match="not enough history"):
        run_walk_forward_study(
            {"BTC/USDT": _flat_frame(260)},
            warmup_bars=250,
            test_bars=60,
        )


def test_walkforward_rejects_misaligned_timestamps_even_when_bar_counts_match():
    first = _flat_frame()
    second = _flat_frame()
    second.index = second.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="aligned timestamps"):
        run_walk_forward_study(
            {"BTC/USDT": first, "ETH/USDT": second},
            warmup_bars=250,
            test_bars=60,
        )


def test_summary_reports_compounded_returns_for_each_symbol_and_symbol_fold_breadth():
    frame = _flat_frame()
    shifted = frame.copy()
    shifted["close"] = shifted["close"] * 1.001
    shifted["open"] = shifted["open"] * 1.001
    shifted["high"] = shifted["high"] * 1.001
    shifted["low"] = shifted["low"] * 1.001
    report = run_walk_forward_study(
        {"BTC/USDT": frame, "ETH/USDT": shifted},
        warmup_bars=250,
        test_bars=60,
    )
    summary = report["summary"]["realistic_estimate"]
    assert set(summary["per_symbol_compounded_oos_return_pct"]) == {"BTC/USDT", "ETH/USDT"}
    assert summary["symbol_fold_count"] == len(report["folds"]) * 2
    assert 0 <= summary["positive_symbol_folds"] <= summary["symbol_fold_count"]
    assert isinstance(summary["all_symbol_folds_positive"], bool)


def test_walkforward_rejects_uniformly_skipped_hourly_bars_by_default():
    frame = _flat_frame(420).iloc[::2]
    with pytest.raises(ValueError, match="expected every bar to be 0 days 01:00:00"):
        run_walk_forward_study(
            {"BTC/USDT": frame}, warmup_bars=100, test_bars=20
        )


def test_walkforward_accepts_explicit_non_hourly_interval():
    frame = _flat_frame(420).iloc[::2]
    report = run_walk_forward_study(
        {"BTC/USDT": frame}, warmup_bars=100, test_bars=20,
        expected_interval="2h",
    )
    assert report["expected_interval"] == "0 days 02:00:00"


def test_buy_and_hold_benchmark_starts_at_next_open_like_strategy():
    frame = _flat_frame(4)
    frame.iloc[0, frame.columns.get_loc("open")] = 100.0
    frame.iloc[1, frame.columns.get_loc("open")] = 200.0
    frame.iloc[3, frame.columns.get_loc("close")] = 210.0
    result = _buy_and_hold_return(
        frame, 0, 3, 1000.0, ExecutionCosts(0, 0, 0)
    )
    assert result == pytest.approx(5.0)


def test_hourly_loader_uses_cached_hourly_history_and_discards_open_candle(monkeypatch):
    from tradingagents.crypto.backtest import history

    now = pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end=now, periods=420, freq="h", tz="UTC")
    prices = np.full(len(index), 100.0)
    hourly_frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.full(len(index), 600.0),
        },
        index=index,
    )

    class FakeHistoryCache:
        def __init__(self, root):
            self.root = root

        def load(self, symbol, interval):
            assert interval == "1h"
            return hourly_frame.copy()

    monkeypatch.setattr(history, "HistoryCache", FakeHistoryCache)
    frames = load_hourly_frames("/unused", ("BTC/USDT",))
    loaded = frames["BTC/USDT"]
    assert len(loaded) == 419
    assert loaded.index[-1] == now - pd.Timedelta(hours=1)
    assert loaded.iloc[-1]["volume"] == pytest.approx(600.0)
