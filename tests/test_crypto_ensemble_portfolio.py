from __future__ import annotations  # noqa: I001

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tradingagents.crypto.ensemble_portfolio as portfolio_module

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



def test_portfolio_stop_risk_budget_includes_both_side_costs_and_reconciles_ledger():
    initial_cash = 10_000.0
    backtest_settings = BacktestSettings(
        warmup_bars=220,
        risk_fraction=0.005,
        max_notional_fraction=1.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = run_portfolio_backtest(
        {"BTC/USDT": _rising_frame()},
        settings=backtest_settings,
        portfolio_settings=PortfolioSettings(
            max_open_positions=1,
            max_alt_positions=1,
            max_gross_exposure_fraction=1.0,
            max_total_open_risk_fraction=0.01,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )

    assert result.trades == 1
    trade = result.trade_log[0]
    fee_rate = backtest_settings.fee_bps / 10_000.0
    slippage_rate = backtest_settings.slippage_bps / 10_000.0
    stop_fill = trade["initial_stop"] * (1.0 - slippage_rate)
    estimated_stop_loss = (
        trade["quantity"] * (trade["entry_price"] - stop_fill)
        + trade["entry_fee"]
        + trade["quantity"] * stop_fill * fee_rate
    )
    assert estimated_stop_loss <= initial_cash * backtest_settings.risk_fraction + 1e-8

    assert result.final_equity == pytest.approx(
        initial_cash + sum(item["net_pnl"] for item in result.trade_log)
    )
    assert result.fees_paid == pytest.approx(
        sum(item["entry_fee"] + item["exit_fee"] for item in result.trade_log)
    )
    assert result.slippage_paid == pytest.approx(
        sum(item["slippage_paid"] for item in result.trade_log)
    )
    for item in result.trade_log:
        assert item["net_pnl"] == pytest.approx(
            item["gross_pnl"]
            - item["slippage_paid"]
            - item["entry_fee"]
            - item["exit_fee"]
        )


def test_portfolio_gap_through_trailing_stop_fills_at_worse_open():
    frame = _rising_frame()
    settings = BacktestSettings(warmup_bars=220, max_hold_bars=100)
    baseline = run_portfolio_backtest(
        {"BTC/USDT": frame},
        settings=settings,
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert baseline.trades == 1
    stop = baseline.trade_log[0]["final_stop"]
    gap_open = stop * 0.95
    gap_bar = pd.DataFrame(
        [{
            "open": gap_open,
            "high": gap_open + 0.5,
            "low": gap_open - 0.2,
            "close": gap_open + 0.1,
            "volume": 1000.0,
        }],
        index=[frame.index[-1] + pd.Timedelta(hours=1)],
    )
    result = run_portfolio_backtest(
        {"BTC/USDT": pd.concat([frame, gap_bar])},
        settings=settings,
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 1
    assert result.trade_log[0]["reason"] == "atr_stop"
    assert result.trade_log[0]["raw_exit_price"] == pytest.approx(gap_open)


def test_portfolio_backtest_does_not_trade_before_oos_start():
    frame = _rising_frame()
    result = run_portfolio_backtest(
        {"BTC/USDT": frame},
        settings=BacktestSettings(warmup_bars=220),
        trading_start_ts=frame.index[260],
        portfolio_settings=PortfolioSettings(
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )
    assert result.trades == 0


def test_portfolio_backtest_requires_timezone_aware_oos_start():
    with pytest.raises(ValueError, match="timezone-aware"):
        run_portfolio_backtest(
            {"BTC/USDT": _rising_frame()},
            settings=BacktestSettings(warmup_bars=220),
            trading_start_ts="2025-01-12 00:00:00",
        )



def _controlled_gap_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=224, freq="h", tz="UTC")
    btc_prices = np.full(len(index), 200.0)
    eth_prices = np.full(len(index), 100.0)
    eth_open = eth_prices.copy()
    eth_high = eth_prices.copy()
    eth_low = eth_prices.copy()
    eth_close = eth_prices.copy()

    # The ETH position is entered at index 221. The next bar opens with a
    # material gap while remaining above the original protective stop.
    eth_open[222] = 94.0
    eth_high[222] = 95.0
    eth_low[222] = 93.0
    eth_close[222] = 94.0

    def frame(prices, opens=None, highs=None, lows=None, closes=None):
        return pd.DataFrame(
            {
                "open": prices if opens is None else opens,
                "high": prices if highs is None else highs,
                "low": prices if lows is None else lows,
                "close": prices if closes is None else closes,
                "volume": np.full(len(index), 1_000.0),
            },
            index=index,
        )

    return frame(btc_prices), frame(
        eth_prices, opens=eth_open, highs=eth_high, lows=eth_low, closes=eth_close
    )


def _install_controlled_signals(monkeypatch):
    def indicators(frame, _settings):
        result = frame.copy()
        result["atr"] = 3.0
        return result

    def signal(indicator_frame, j, _settings, _mode):
        price = float(indicator_frame.close.iloc[0])
        symbol = "BTC/USDT" if price >= 150.0 else "ETH/USDT"
        should_enter = (symbol == "ETH/USDT" and j == 220) or (
            symbol == "BTC/USDT" and j == 221
        )
        return SimpleNamespace(
            action="long" if should_enter else "hold",
            stop=180.0 if symbol == "BTC/USDT" else 90.0,
            should_exit=False,
        )

    monkeypatch.setattr(portfolio_module, "_indicators", indicators)
    monkeypatch.setattr(portfolio_module, "_evaluate_indicator_row", signal)
    monkeypatch.setattr(portfolio_module, "_btc_risk_off", lambda *_args: False)


def test_opening_gap_rechecks_daily_loss_before_filling_pending_entries(monkeypatch):
    _install_controlled_signals(monkeypatch)
    btc, eth = _controlled_gap_frames()
    result = run_portfolio_backtest(
        {"BTC/USDT": btc, "ETH/USDT": eth},
        settings=BacktestSettings(
            warmup_bars=220,
            risk_fraction=0.05,
            max_notional_fraction=0.40,
            trailing_stop_atr_multiple=3.0,
        ),
        portfolio_settings=PortfolioSettings(
            max_open_positions=2,
            max_alt_positions=1,
            max_gross_exposure_fraction=1.0,
            max_total_open_risk_fraction=0.05,
            daily_loss_limit_fraction=0.02,
            max_drawdown_fraction=None,
        ),
    )

    assert result.trades == 1
    assert result.trade_log[0]["symbol"] == "ETH/USDT"
    assert result.blocked_entries["daily_loss_limit"] == 1
    assert result.max_drawdown_pct >= 2.0


def test_opening_gap_through_stop_releases_position_slot_before_new_entries(monkeypatch):
    _install_controlled_signals(monkeypatch)
    btc, eth = _controlled_gap_frames()
    eth.loc[eth.index[222], ["open", "high", "low", "close"]] = [89.0, 90.0, 88.0, 89.0]
    result = run_portfolio_backtest(
        {"BTC/USDT": btc, "ETH/USDT": eth},
        settings=BacktestSettings(
            warmup_bars=220,
            risk_fraction=0.005,
            max_notional_fraction=0.40,
            trailing_stop_atr_multiple=3.0,
        ),
        portfolio_settings=PortfolioSettings(
            max_open_positions=1,
            max_alt_positions=1,
            max_gross_exposure_fraction=1.0,
            max_total_open_risk_fraction=0.05,
            daily_loss_limit_fraction=None,
            max_drawdown_fraction=None,
        ),
    )

    assert result.trades == 2
    eth_trade = next(trade for trade in result.trade_log if trade["symbol"] == "ETH/USDT")
    btc_trade = next(trade for trade in result.trade_log if trade["symbol"] == "BTC/USDT")
    assert eth_trade["reason"] == "atr_stop"
    assert eth_trade["raw_exit_price"] == pytest.approx(89.0)
    assert btc_trade["entry_time"] == eth.index[222].isoformat()
