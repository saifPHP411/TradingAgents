from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.ensemble_backtest import BacktestSettings
from tradingagents.crypto.ensemble_portfolio import PortfolioSettings
from tradingagents.crypto.ensemble_walkforward import (
    WalkForwardSettings,
    block_bootstrap_equity,
    run_walk_forward,
    _profit_concentration,
)


pytestmark = pytest.mark.unit


def _flat_frame(size: int = 300) -> pd.DataFrame:
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


def test_walkforward_uses_disjoint_oos_windows_after_purge_and_embargo():
    frame = _flat_frame()
    report = run_walk_forward(
        {"BTC/USDT": frame},
        walk_forward_settings=WalkForwardSettings(
            train_bars=220,
            test_bars=20,
            purge_bars=3,
            embargo_bars=4,
            monte_carlo_simulations=100,
            monte_carlo_block_bars=4,
        ),
        backtest_settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert report["fold_count"] == 3
    first, second, third = report["folds"]
    assert first["train_context_end"] == frame.index[219].isoformat()
    assert first["oos_start"] == frame.index[227].isoformat()
    assert second["oos_start"] == frame.index[247].isoformat()
    assert third["oos_start"] == frame.index[267].isoformat()
    assert first["oos_end"] < second["oos_start"]
    assert report["summary"]["trades"] == 0
    assert report["summary"]["compounded_net_return_pct"] == pytest.approx(0.0)


def test_walkforward_rejects_overlapping_test_windows():
    with pytest.raises(ValueError, match="prevent overlapping"):
        run_walk_forward(
            {"BTC/USDT": _flat_frame()},
            walk_forward_settings=WalkForwardSettings(
                train_bars=220,
                test_bars=20,
                step_bars=10,
                monte_carlo_simulations=100,
            ),
            backtest_settings=BacktestSettings(warmup_bars=220),
        )


def test_walkforward_rejects_misaligned_symbol_timestamps():
    btc = _flat_frame()
    eth = btc.copy()
    eth.index = eth.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="not aligned"):
        run_walk_forward(
            {"BTC/USDT": btc, "ETH/USDT": eth},
            walk_forward_settings=WalkForwardSettings(
                train_bars=220, test_bars=20, monte_carlo_simulations=100
            ),
            backtest_settings=BacktestSettings(warmup_bars=220),
        )


def test_block_bootstrap_is_reproducible_and_reports_negative_paths():
    returns = [-0.01, -0.02, -0.005, -0.015]
    first = block_bootstrap_equity(
        returns, initial_equity=10_000, simulations=200, block_bars=2, seed=7
    )
    second = block_bootstrap_equity(
        returns, initial_equity=10_000, simulations=200, block_bars=2, seed=7
    )
    assert first == second
    assert first["probability_of_loss_pct"] == 100.0
    assert first["terminal_equity_p50"] < 10_000
    assert first["max_drawdown_p95_pct"] > 0


def test_block_bootstrap_handles_no_returns_and_rejects_total_loss_return():
    empty = block_bootstrap_equity([], initial_equity=10_000, simulations=100)
    assert empty["terminal_equity_p50"] == 10_000
    with pytest.raises(ValueError, match="greater than -100%"):
        block_bootstrap_equity([-1.0], initial_equity=10_000, simulations=100)


def test_walkforward_requires_enough_data_for_a_complete_fold():
    with pytest.raises(ValueError, match="not enough aligned history"):
        run_walk_forward(
            {"BTC/USDT": _flat_frame(240)},
            walk_forward_settings=WalkForwardSettings(
                train_bars=220,
                test_bars=20,
                purge_bars=2,
                embargo_bars=2,
                monte_carlo_simulations=100,
            ),
            backtest_settings=BacktestSettings(warmup_bars=220),
        )

def test_buy_hold_benchmark_uses_same_next_bar_oos_entry_timing():
    frame = _flat_frame()
    frame.loc[frame.index[228]:, ["open", "high", "low", "close"]] = 110.0
    fee_rate = 10.0 / 10_000.0
    expected = (110.0 * (1.0 - fee_rate) / (110.0 * (1.0 + fee_rate)) - 1.0) * 100.0
    report = run_walk_forward(
        {"BTC/USDT": frame},
        walk_forward_settings=WalkForwardSettings(
            train_bars=220,
            test_bars=20,
            purge_bars=3,
            embargo_bars=4,
            monte_carlo_simulations=100,
            monte_carlo_block_bars=4,
        ),
        backtest_settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
        mode="range",
    )
    assert report["folds"][0]["equal_weight_buy_hold_net_return_pct"] == pytest.approx(expected)




def test_profit_concentration_flags_single_symbol_and_fold_dominance():
    btc_win = {"symbol": "BTC/USDT", "net_pnl": 60.0}
    btc_loss = {"symbol": "BTC/USDT", "net_pnl": -10.0}
    eth_win = {"symbol": "ETH/USDT", "net_pnl": 30.0}
    bnb_win = {"symbol": "BNB/USDT", "net_pnl": 10.0}
    result = _profit_concentration(
        [btc_win, btc_loss, eth_win, bnb_win],
        [[btc_win, btc_loss], [eth_win], [bnb_win]],
    )
    assert result["gross_winning_pnl_usdt"] == pytest.approx(100.0)
    assert result["by_symbol"]["BTC/USDT"]["net_pnl_usdt"] == pytest.approx(50.0)
    assert result["largest_symbol_share_pct"] == pytest.approx(60.0)
    assert result["largest_fold_share_pct"] == pytest.approx(60.0)
    assert result["largest_trade_share_pct"] == pytest.approx(60.0)
    assert result["single_symbol_or_fold_dominates"] is True
    assert result["interpretation"] == "concentrated"


def test_profit_concentration_is_not_assessable_without_winning_trades():
    result = _profit_concentration(
        [{"symbol": "BTC/USDT", "net_pnl": -5.0}],
        [[{"symbol": "BTC/USDT", "net_pnl": -5.0}]],
    )
    assert result["gross_winning_pnl_usdt"] == 0.0
    assert result["largest_symbol_share_pct"] is None
    assert result["largest_fold_share_pct"] is None
    assert result["single_symbol_or_fold_dominates"] is None
    assert result["interpretation"] == "not_assessable_no_gross_winning_pnl"



def test_block_bootstrap_rejects_segments_that_do_not_partition_returns():
    with pytest.raises(ValueError, match="partition hourly_returns"):
        block_bootstrap_equity(
            [0.01, -0.01],
            initial_equity=10_000,
            simulations=100,
            segments=[],
        )
    with pytest.raises(ValueError, match="partition hourly_returns"):
        block_bootstrap_equity(
            [0.01, -0.01],
            initial_equity=10_000,
            simulations=100,
            segments=[[0.01]],
        )
