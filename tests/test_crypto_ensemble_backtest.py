from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.ensemble import EnsembleSettings
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


def _rising_breakout_frame() -> pd.DataFrame:
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
    }, index=pd.date_range("2025-01-01", periods=260, freq="h", tz="UTC"))
    last = frame.iloc[-1].copy()
    next_bar = pd.DataFrame([{
        "open": last.close + 0.1,
        "high": last.close + 1.1,
        "low": last.close - 0.9,
        "close": last.close + 0.1,
        "volume": 1000.0,
    }], index=[frame.index[-1] + pd.Timedelta(hours=1)])
    return pd.concat([frame, next_bar])


def test_flat_market_stays_out_and_preserves_capital():
    result = run_backtest(_flat_frame(), settings=BacktestSettings(warmup_bars=220))
    assert result.trades == 0
    assert result.final_equity == pytest.approx(10_000.0)
    assert result.net_return_pct == pytest.approx(0.0)
    assert result.fees_paid == 0.0
    assert result.slippage_paid == 0.0


def test_comparison_runs_all_four_variants_on_same_bars():
    report = compare_modes(_flat_frame(), settings=BacktestSettings(warmup_bars=220))
    assert set(report["results"]) == {"trend", "trend_pullback", "range", "breakout", "ensemble"}
    assert report["bar_count"] == 260
    assert report["execution"] == "signal_on_close_next_bar_open_stop_gap_aware"
    for result in report["results"].values():
        assert result["trades"] == 0
        assert result["final_equity"] == pytest.approx(10_000.0)


def test_backtest_rejects_bad_bar_order_and_negative_fees():
    frame = _flat_frame()
    frame.index = frame.index[::-1]
    with pytest.raises(ValueError, match="increasing"):
        run_backtest(frame, settings=BacktestSettings(warmup_bars=220))
    with pytest.raises(ValueError, match="fees"):
        run_backtest(_flat_frame(), settings=BacktestSettings(fee_bps=-1, warmup_bars=220))
    gap_frame = _flat_frame()
    gapped = gap_frame.drop(index=gap_frame.index[120])
    with pytest.raises(ValueError, match="contiguous 1h"):
        run_backtest(gapped, settings=BacktestSettings(warmup_bars=220))



def test_max_hold_bars_exits_after_exactly_n_completed_bars():
    frame = _rising_breakout_frame()
    for _ in range(4):
        previous = frame.iloc[-1]
        open_price = float(previous.close) + 0.1
        close_price = open_price + 0.1
        next_index = frame.index[-1] + pd.Timedelta(hours=1)
        extra = pd.DataFrame([{
            "open": open_price,
            "high": close_price + 0.5,
            "low": open_price - 0.2,
            "close": close_price,
            "volume": 1000.0,
        }], index=[next_index])
        frame = pd.concat([frame, extra])
    result = run_backtest(
        frame,
        settings=BacktestSettings(warmup_bars=220, max_hold_bars=2),
    )
    assert result.trades == 1
    assert result.trade_log[0]["reason"] == "max_hold_bars"
    assert result.trade_log[0]["raw_entry_price"] == pytest.approx(
        _rising_breakout_frame().iloc[260].open
    )
    assert result.trade_log[0]["exit_time"] == frame.index[262].isoformat()


def test_backtest_rejects_non_finite_or_non_integer_risk_settings():
    with pytest.raises(ValueError, match="finite"):
        run_backtest(
            _flat_frame(),
            settings=BacktestSettings(fee_bps=float("nan"), warmup_bars=220),
        )
    with pytest.raises(ValueError, match="positive integer"):
        run_backtest(
            _flat_frame(),
            settings=BacktestSettings(max_hold_bars=1.5, warmup_bars=220),
        )


def test_single_symbol_backtest_rejects_invalid_signal_settings():
    with pytest.raises(ValueError, match="finite"):
        run_backtest(
            _flat_frame(),
            settings=BacktestSettings(warmup_bars=220),
            signal_settings=EnsembleSettings(volume_multiple=float("nan")),
        )


def test_btc_filter_blocks_altcoin_entries_when_btc_trend_is_not_bullish():
    alt = _rising_breakout_frame()
    btc = _flat_frame(size=len(alt))
    btc.index = alt.index
    settings = BacktestSettings(warmup_bars=220)
    unfiltered = run_backtest(alt, settings=settings)
    filtered = run_backtest(alt, settings=settings, btc_frame=btc)
    assert unfiltered.trades == 1
    assert unfiltered.fees_paid > 0
    assert unfiltered.slippage_paid > 0
    assert unfiltered.trade_log[0]["net_pnl"] < unfiltered.trade_log[0]["gross_pnl"]
    assert filtered.trades == 0



def test_chandelier_trailing_stop_tightens_only_for_following_bar():
    frame = _rising_breakout_frame()
    # This entry candle dips below the trailing level computed from its own
    # high, but remains above the original stop: it must not stop itself out.
    frame.loc[frame.index[-1], "low"] = 129.5
    entry_bar_time = frame.index[-1]
    next_index = frame.index[-1] + pd.Timedelta(hours=1)
    extra = pd.DataFrame(
        [{
            "open": 130.8,
            "high": 131.0,
            "low": 129.0,
            "close": 130.8,
            "volume": 1000.0,
        }],
        index=[next_index],
    )
    frame = pd.concat([frame, extra])
    tight = run_backtest(
        frame,
        settings=BacktestSettings(
            warmup_bars=220,
            max_hold_bars=100,
            trailing_stop_atr_multiple=1.0,
        ),
    )
    loose = run_backtest(
        frame,
        settings=BacktestSettings(
            warmup_bars=220,
            max_hold_bars=100,
            trailing_stop_atr_multiple=10.0,
        ),
    )
    assert tight.trades == 1
    assert tight.trade_log[0]["reason"] == "atr_stop"
    assert tight.trade_log[0]["final_stop"] > tight.trade_log[0]["initial_stop"]
    assert tight.trade_log[0]["exit_time"] == next_index.isoformat()
    assert tight.trade_log[0]["exit_time"] != entry_bar_time.isoformat()
    assert tight.trade_log[0]["raw_exit_price"] > tight.trade_log[0]["raw_entry_price"] * 0.97
    assert loose.trades == 1
    assert loose.trade_log[0]["reason"] == "end_of_data"
    assert loose.trade_log[0]["final_stop"] == pytest.approx(
        loose.trade_log[0]["initial_stop"]
    )


def test_single_symbol_stop_risk_budget_includes_fees_and_adverse_slippage():
    settings = BacktestSettings(
        warmup_bars=220,
        risk_fraction=0.005,
        max_notional_fraction=1.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = run_backtest(_rising_breakout_frame(), settings=settings)
    assert result.trades == 1
    trade = result.trade_log[0]
    fee_rate = settings.fee_bps / 10_000.0
    slip_rate = settings.slippage_bps / 10_000.0
    stop_fill = trade["initial_stop"] * (1.0 - slip_rate)
    estimated_loss = trade["quantity"] * (
        trade["entry_price"] * (1.0 + fee_rate)
        - stop_fill * (1.0 - fee_rate)
    )
    assert estimated_loss == pytest.approx(trade["estimated_stop_risk"])
    assert estimated_loss <= trade["entry_equity"] * settings.risk_fraction + 1e-8
    assert trade["entry_price"] * trade["quantity"] + trade["entry_fee"] <= trade["entry_equity"] + 1e-8


def test_single_symbol_backtest_does_not_trade_before_oos_start():
    frame = _rising_breakout_frame()
    result = run_backtest(
        frame,
        settings=BacktestSettings(warmup_bars=220),
        trading_start_ts=frame.index[260],
    )
    assert result.trades == 0


def test_single_symbol_backtest_requires_timezone_aware_oos_start():
    with pytest.raises(ValueError, match="timezone-aware"):
        run_backtest(
            _rising_breakout_frame(),
            settings=BacktestSettings(warmup_bars=220),
            trading_start_ts="2025-01-12 00:00:00",
        )
