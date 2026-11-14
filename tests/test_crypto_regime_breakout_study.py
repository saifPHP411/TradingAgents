from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.backtest.regime_breakout_study import run_walk_forward_study

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
