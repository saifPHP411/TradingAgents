from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tradingagents.crypto.ensemble_backtest import BacktestSettings
from tradingagents.crypto.ensemble_portfolio import (
    PortfolioSettings,
    run_portfolio_backtest,
)


pytestmark = pytest.mark.unit


def _flat_frame(size: int = 261) -> pd.DataFrame:
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


def _rising_frame() -> pd.DataFrame:
    close = np.linspace(100, 130, 260)
    close[-1] = close[-2] + 1.1
    volume = np.full(len(close), 1000.0)
    volume[-1] = 1500.0
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2025-01-01", periods=260, freq="h", tz="UTC"),
    )
    last = frame.iloc[-1]
    next_bar = pd.DataFrame(
        [{
            "open": last.close + 0.1,
            "high": last.close + 1.1,
            "low": last.close - 0.9,
            "close": last.close + 0.1,
            "volume": 1000.0,
        }],
        index=[frame.index[-1] + pd.Timedelta(hours=1)],
    )
    return pd.concat([frame, next_bar])


def test_shared_portfolio_flat_market_preserves_capital():
    frame = _flat_frame()
    result = run_portfolio_backtest(
        {"BTC/USDT": frame, "ETH/USDT": frame.copy()},
        settings=BacktestSettings(warmup_bars=220),
    )
    assert result.trades == 0
    assert result.final_equity == pytest.approx(10_000.0)
    assert result.net_return_pct == pytest.approx(0.0)
    assert result.fees_paid == 0.0
    assert result.slippage_paid == 0.0


def test_btc_risk_filter_blocks_altcoin_entries():
    alt = _rising_frame()
    btc = _flat_frame(size=len(alt))
    btc.index = alt.index
    result = run_portfolio_backtest(
        {"BTC/USDT": btc, "ETH/USDT": alt},
        settings=BacktestSettings(warmup_bars=220),
    )
    assert result.trades == 0
    assert result.blocked_entries["btc_risk_filter"] > 0


def test_shared_cash_enforces_maximum_open_positions():
    rising = _rising_frame()
    result = run_portfolio_backtest(
        {"BTC/USDT": rising, "ETH/USDT": rising.copy()},
        settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            max_open_positions=1,
            max_alt_positions=1,
            max_gross_exposure_fraction=0.5,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 1
    assert result.trade_log[0]["symbol"] == "BTC/USDT"
    assert result.blocked_entries["max_open_positions"] > 0


def test_shared_portfolio_enforces_altcoin_concentration_and_gross_exposure():
    rising = _rising_frame()
    result = run_portfolio_backtest(
        {
            "BTC/USDT": rising,
            "BNB/USDT": rising.copy(),
            "ETH/USDT": rising.copy(),
        },
        settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            max_open_positions=3,
            max_alt_positions=1,
            max_gross_exposure_fraction=0.05,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    alt_trades = [trade for trade in result.trade_log if trade["symbol"] != "BTC/USDT"]
    assert len(alt_trades) <= 1
    assert result.blocked_entries["max_alt_positions"] > 0
    for trade in result.trade_log:
        assert trade["entry_price"] * trade["quantity"] <= 500.01



def test_btc_has_first_claim_on_simultaneous_entry_capacity():
    rising = _rising_frame()
    result = run_portfolio_backtest(
        {"BNB/USDT": rising.copy(), "BTC/USDT": rising.copy()},
        settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            max_open_positions=1,
            max_alt_positions=1,
            max_gross_exposure_fraction=0.5,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 1
    assert result.trade_log[0]["symbol"] == "BTC/USDT"
    assert result.blocked_entries["max_open_positions"] > 0


def test_portfolio_max_hold_counts_entry_candle_and_exits_next_open():
    frame = _rising_frame()
    previous = frame.iloc[-1]
    next_open = float(previous.close) + 0.1
    next_bar = pd.DataFrame(
        [{
            "open": next_open,
            "high": next_open + 0.5,
            "low": next_open - 0.2,
            "close": next_open + 0.1,
            "volume": 1000.0,
        }],
        index=[frame.index[-1] + pd.Timedelta(hours=1)],
    )
    frame = pd.concat([frame, next_bar])
    result = run_portfolio_backtest(
        {"BTC/USDT": frame},
        settings=BacktestSettings(warmup_bars=220, max_hold_bars=1),
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 1
    assert result.trade_log[0]["reason"] == "max_hold_bars"
    assert result.trade_log[0]["exit_time"] == frame.index[-1].isoformat()

def test_portfolio_requires_aligned_hourly_history_and_btc():
    frame = _flat_frame()
    with pytest.raises(ValueError, match="include the BTC"):
        run_portfolio_backtest({"ETH/USDT": frame})
    misaligned = frame.copy()
    misaligned.index = misaligned.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="identical hourly timestamps"):
        run_portfolio_backtest(
            {"BTC/USDT": frame, "ETH/USDT": misaligned},
            settings=BacktestSettings(warmup_bars=220),
        )
    with pytest.raises(ValueError, match="positive integer"):
        run_portfolio_backtest(
            {"BTC/USDT": frame},
            portfolio_settings=PortfolioSettings(max_open_positions=1.5),
        )



def test_portfolio_blocks_highly_correlated_altcoin_entries():
    rising = _rising_frame()
    result = run_portfolio_backtest(
        {
            "BTC/USDT": rising,
            "BNB/USDT": rising.copy(),
            "ETH/USDT": rising.copy(),
        },
        settings=BacktestSettings(warmup_bars=220),
        portfolio_settings=PortfolioSettings(
            max_open_positions=3,
            max_alt_positions=2,
            max_gross_exposure_fraction=0.5,
            max_alt_correlation=0.85,
            correlation_lookback=20,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.blocked_entries["alt_correlation"] > 0
    alt_trades = [trade for trade in result.trade_log if trade["symbol"] != "BTC/USDT"]
    assert len(alt_trades) <= 1


def test_portfolio_rejects_invalid_correlation_limits_and_trailing_stop():
    with pytest.raises(ValueError, match="correlation"):
        run_portfolio_backtest(
            {"BTC/USDT": _flat_frame()},
            portfolio_settings=PortfolioSettings(max_alt_correlation=1.1),
        )
    with pytest.raises(ValueError, match="trailing stop"):
        run_portfolio_backtest(
            {"BTC/USDT": _flat_frame()},
            settings=BacktestSettings(trailing_stop_atr_multiple=0),
        )


def test_portfolio_caps_aggregate_open_stop_risk():
    rising = _rising_frame()
    result = run_portfolio_backtest(
        {
            "BTC/USDT": rising,
            "BNB/USDT": rising.copy(),
            "ETH/USDT": rising.copy(),
        },
        settings=BacktestSettings(
            warmup_bars=220,
            risk_fraction=0.005,
            max_notional_fraction=1.0,
        ),
        portfolio_settings=PortfolioSettings(
            max_open_positions=3,
            max_alt_positions=2,
            max_gross_exposure_fraction=1.0,
            max_alt_correlation=0.85,
            correlation_lookback=20,
            max_total_open_risk_fraction=0.003,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 1
    assert result.blocked_entries["portfolio_risk"] > 0


def test_portfolio_rejects_invalid_total_open_risk_limit():
    with pytest.raises(ValueError, match="total open risk"):
        run_portfolio_backtest(
            {"BTC/USDT": _flat_frame()},
            portfolio_settings=PortfolioSettings(max_total_open_risk_fraction=0),
        )
