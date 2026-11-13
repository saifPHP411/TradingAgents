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
