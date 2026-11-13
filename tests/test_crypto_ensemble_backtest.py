from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.ensemble_backtest import BacktestSettings, compare_modes, run_backtest


pytestmark = pytest.mark.unit


def _flat_frame(size: int = 260) -> pd.DataFrame:
    close = np.full(size, 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(size, 1000.0),
        },
        index=pd.date_range("2025-01-01", periods=size, freq="h", tz="UTC"),
    )


def test_flat_market_stays_out_and_preserves_capital():
    result = run_backtest(_flat_frame())
    assert result.trades == 0
    assert result.final_equity == pytest.approx(10_000.0)
    assert result.net_return_pct == pytest.approx(0.0)
    assert result.fees_paid == 0.0
    assert result.slippage_paid == 0.0


def test_comparison_runs_all_four_variants_on_same_bars():
    report = compare_modes(_flat_frame())
    assert set(report["results"]) == {"trend", "range", "breakout", "ensemble"}
    assert report["bar_count"] == 260
    assert report["execution"] == "signal_on_close_next_bar_open_stop_gap_aware"
    for result in report["results"].values():
        assert result["trades"] == 0
        assert result["final_equity"] == pytest.approx(10_000.0)


def test_backtest_rejects_bad_bar_order_and_negative_fees():
    frame = _flat_frame()
    frame.index = frame.index[::-1]
    with pytest.raises(ValueError, match="increasing"):
        run_backtest(frame)
    with pytest.raises(ValueError, match="fees"):
        run_backtest(_flat_frame(), settings=BacktestSettings(fee_bps=-1))
