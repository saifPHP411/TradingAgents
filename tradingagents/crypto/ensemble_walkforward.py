"""Leakage-aware chronological OOS validation for the research-only RAE portfolio.

This runner does not fit or optimize parameters. It evaluates a frozen strategy
on disjoint forward test windows. Purge and embargo bars are deliberately
excluded from trading; prior history remains available only for causal
indicator warm-up. Monte Carlo resamples contiguous hourly-return blocks and
is a robustness diagnostic, not a forecast.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd

from .ensemble_backtest import BacktestSettings
from .ensemble_portfolio import PortfolioSettings, run_portfolio_backtest


@dataclass(frozen=True)
class WalkForwardSettings:
    train_bars: int = 4_000
    test_bars: int = 720
    purge_bars: int = 24
    embargo_bars: int = 24
    step_bars: int | None = None
    monte_carlo_simulations: int = 2_000
    monte_carlo_block_bars: int = 24
    random_seed: int = 42


def _validate_frames(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not frames or "BTC/USDT" not in frames:
        raise ValueError("frames must include BTC/USDT")
    index = frames["BTC/USDT"].index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("walk-forward data requires a timezone-aware DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("walk-forward timestamps must be strictly increasing and unique")
    if not (index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("walk-forward data must contain contiguous hourly bars")
    for symbol, frame in frames.items():
        if not frame.index.equals(index):
            raise ValueError(f"{symbol} timestamps are not aligned with BTC/USDT")
    return index


def _validate_settings(settings: WalkForwardSettings, backtest: BacktestSettings) -> None:
    step = settings.test_bars if settings.step_bars is None else settings.step_bars
    integer_values = (
        settings.train_bars,
        settings.test_bars,
        settings.purge_bars,
        settings.embargo_bars,
        step,
        settings.monte_carlo_simulations,
        settings.monte_carlo_block_bars,
        settings.random_seed,
    )
    if any(type(value) is not int for value in integer_values):
        raise ValueError("walk-forward settings must be integers")
    if settings.train_bars < max(backtest.warmup_bars, 220):
        raise ValueError("train_bars must be at least the backtest warm-up length")
    if settings.test_bars < 2 or settings.purge_bars < 0 or settings.embargo_bars < 0:
        raise ValueError("test_bars must be >= 2 and purge/embargo cannot be negative")
    if step < settings.test_bars:
        raise ValueError("step_bars must be >= test_bars to prevent overlapping OOS windows")
    if settings.monte_carlo_simulations < 100 or settings.monte_carlo_block_bars < 1:
        raise ValueError("Monte Carlo requires >= 100 simulations and a positive block length")


def block_bootstrap_equity(
    hourly_returns: list[float] | np.ndarray,
    *,
    initial_equity: float,
    simulations: int = 2_000,
    block_bars: int = 24,
    seed: int = 42,
) -> dict:
    """Circular moving-block bootstrap of hourly returns, preserving short clusters."""
    returns = np.asarray(hourly_returns, dtype=float)
    if not isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    if type(simulations) is not int or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if type(block_bars) is not int or block_bars < 1:
        raise ValueError("block_bars must be a positive integer")
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("hourly_returns must be a finite one-dimensional sequence")
    if len(returns) == 0:
        return {
            "simulations": simulations,
            "block_bars": block_bars,
            "probability_of_loss_pct": 0.0,
            "terminal_equity_p05": initial_equity,
            "terminal_equity_p50": initial_equity,
            "terminal_equity_p95": initial_equity,
            "max_drawdown_p50_pct": 0.0,
            "max_drawdown_p95_pct": 0.0,
        }
    if (returns <= -1.0).any():
        raise ValueError("hourly returns must be greater than -100%")
    rng = np.random.default_rng(seed)
    n = len(returns)
    terminal = np.empty(simulations, dtype=float)
    drawdowns = np.empty(simulations, dtype=float)
    offsets = np.arange(block_bars)
    for simulation in range(simulations):
        sampled = np.empty(n, dtype=float)
        cursor = 0
        while cursor < n:
            start = int(rng.integers(0, n))
            count = min(block_bars, n - cursor)
            sampled[cursor : cursor + count] = returns[(start + offsets[:count]) % n]
            cursor += count
        path = initial_equity * np.cumprod(1.0 + sampled)
        running_peak = np.maximum.accumulate(np.concatenate(([initial_equity], path)))
        path_with_start = np.concatenate(([initial_equity], path))
        dd = (running_peak - path_with_start) / running_peak
        terminal[simulation] = path[-1]
        drawdowns[simulation] = float(np.max(dd))
    return {
        "simulations": simulations,
        "block_bars": block_bars,
        "probability_of_loss_pct": float(np.mean(terminal < initial_equity) * 100.0),
        "terminal_equity_p05": float(np.percentile(terminal, 5)),
        "terminal_equity_p50": float(np.percentile(terminal, 50)),
        "terminal_equity_p95": float(np.percentile(terminal, 95)),
        "max_drawdown_p50_pct": float(np.percentile(drawdowns, 50) * 100.0),
        "max_drawdown_p95_pct": float(np.percentile(drawdowns, 95) * 100.0),
    }


def _risk_metrics(hourly_returns: list[float], trades: list[dict]) -> dict:
    returns = np.asarray(hourly_returns, dtype=float)
    if len(returns) > 1 and float(np.std(returns, ddof=1)) > 0:
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * sqrt(24 * 365))
    else:
        sharpe = None
    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(returns) else 0.0
    sortino = (
        float(np.mean(returns) / downside_deviation * sqrt(24 * 365))
        if len(returns) and downside_deviation > 0
        else None
    )
    pnls = [float(trade["net_pnl"]) for trade in trades]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    return {
        "trades": len(pnls),
        "wins": sum(value > 0 for value in pnls),
        "losses": sum(value < 0 for value in pnls),
        "win_rate_pct": (sum(value > 0 for value in pnls) / len(pnls) * 100.0) if pnls else 0.0,
        "expectancy_usdt": float(np.mean(pnls)) if pnls else 0.0,
        "profit_factor": gains / losses if losses > 0 else None,
        "sharpe_hourly_annualized": sharpe,
        "sortino_hourly_annualized": sortino,
    }


def run_walk_forward(
    frames: dict[str, pd.DataFrame],
    *,
    walk_forward_settings: WalkForwardSettings | None = None,
    backtest_settings: BacktestSettings | None = None,
    portfolio_settings: PortfolioSettings | None = None,
) -> dict:
    """Evaluate fixed RAE settings on disjoint chronological OOS windows."""
    wf = walk_forward_settings or WalkForwardSettings()
    bt = backtest_settings or BacktestSettings()
    ps = portfolio_settings or PortfolioSettings()
    _validate_settings(wf, bt)
    index = _validate_frames(frames)
    step = wf.test_bars if wf.step_bars is None else wf.step_bars
    folds = []
    all_returns: list[float] = []
    all_trades: list[dict] = []
    compounded_equity = bt.initial_cash
    fold_number = 0
    fold_start = 0
    while True:
        test_start = fold_start + wf.train_bars + wf.purge_bars + wf.embargo_bars
        test_end = test_start + wf.test_bars
        if test_end > len(index):
            break
        fold_number += 1
        start_time = index[test_start]
        end_time = index[test_end - 1]
        fold_frames = {
            symbol: frame.iloc[fold_start:test_end].copy()
            for symbol, frame in frames.items()
        }
        result = run_portfolio_backtest(
            fold_frames,
            settings=bt,
            portfolio_settings=ps,
            trading_start_ts=start_time,
        )
        oos_curve = [
            point for point in result.equity_curve
            if pd.Timestamp(point["time"]) >= start_time
        ]
        values = [bt.initial_cash, *[float(point["equity"]) for point in oos_curve]]
        fold_returns = [
            (values[i] / values[i - 1]) - 1.0
            for i in range(1, len(values))
            if values[i - 1] > 0
        ]
        all_returns.extend(fold_returns)
        all_trades.extend(result.trade_log)
        fold_return = result.final_equity / bt.initial_cash - 1.0
        compounded_equity *= 1.0 + fold_return
        folds.append({
            "fold": fold_number,
            "train_context_start": index[fold_start].isoformat(),
            "train_context_end": index[fold_start + wf.train_bars - 1].isoformat(),
            "purge_bars": wf.purge_bars,
            "embargo_bars": wf.embargo_bars,
            "oos_start": start_time.isoformat(),
            "oos_end": end_time.isoformat(),
            "oos_bars": wf.test_bars,
            "trades": result.trades,
            "net_return_pct": result.net_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "blocked_entries": result.blocked_entries,
        })
        fold_start += step
    if not folds:
        raise ValueError("not enough aligned history for one complete walk-forward OOS fold")
    metrics = _risk_metrics(all_returns, all_trades)
    worst_fold_drawdown = max(fold["max_drawdown_pct"] for fold in folds)
    monte_carlo = block_bootstrap_equity(
        all_returns,
        initial_equity=bt.initial_cash,
        simulations=wf.monte_carlo_simulations,
        block_bars=wf.monte_carlo_block_bars,
        seed=wf.random_seed,
    )
    return {
        "schema_version": 1,
        "kind": "rae_chronological_walk_forward",
        "strategy": "frozen RAE settings; no parameter fitting or retuning",
        "execution": "completed hourly close signals; next hourly open fills; gap-aware stops",
        "assumptions": [
            "Purge and embargo bars are not eligible for entries.",
            "Training context is used only for causal indicator warm-up; no model is fitted.",
            "OOS windows are disjoint and each fold starts with fresh initial cash.",
            "Aggregate return compounds each independent fold return in chronological order.",
            "Hourly Sharpe/Sortino are annualized with 24*365 periods and are descriptive.",
            "Monte Carlo uses circular blocks of hourly returns; it does not model market impact or regime changes.",
        ],
        "settings": {
            "walk_forward": asdict(wf),
            "backtest": asdict(bt),
            "portfolio": asdict(ps),
        },
        "fold_count": len(folds),
        "folds": folds,
        "summary": {
            "oos_start": folds[0]["oos_start"],
            "oos_end": folds[-1]["oos_end"],
            "oos_bars": sum(fold["oos_bars"] for fold in folds),
            "compounded_net_return_pct": (compounded_equity / bt.initial_cash - 1.0) * 100.0,
            "worst_fold_max_drawdown_pct": worst_fold_drawdown,
            **metrics,
        },
        "monte_carlo": monte_carlo,
    }
