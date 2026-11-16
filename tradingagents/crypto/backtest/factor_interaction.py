"""Research-only cross-sectional factor-interaction strategy.

A weekly-refit ridge model ranks a liquid Spot universe using causal multi-horizon
momentum, reversal, volatility and liquidity factors plus predeclared interactions.
It trades long-only, at most two 10% sleeves, holds for 24 hours and leaves the
rest in cash. This is a research candidate, not an execution adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

_RAW = (
    "mom24", "mom72", "mom168", "mom336", "reversal6",
    "vol24", "vol168", "ema_gap", "log_dollar_volume", "volume_spike",
)
_FEATURES = _RAW + (
    "btc_regime", "mom168_x_liquidity", "mom72_x_lowvol",
    "mom168_x_btc", "reversal_x_mom168", "mom24_x_volume",
)


@dataclass(frozen=True)
class FactorCosts:
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("cost assumptions must be finite and non-negative")

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.commission_bps + self.half_spread_bps + self.slippage_bps)


@dataclass(frozen=True)
class FactorConfig:
    warmup_bars: int = 336
    train_window_bars: int = 4320
    test_fold_bars: int = 168
    holding_bars: int = 24
    sample_step_bars: int = 24
    refit_bars: int = 168
    top_n: int = 2
    allocation_per_asset: float = 0.10
    ridge_alpha: float = 10.0
    min_train_rows: int = 500
    cost_edge_multiple: float = 1.25

    def __post_init__(self) -> None:
        for name in (
            "warmup_bars", "train_window_bars", "test_fold_bars",
            "holding_bars", "sample_step_bars", "refit_bars",
            "top_n", "min_train_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_n > 2 or self.allocation_per_asset <= 0 or self.allocation_per_asset > 0.10:
            raise ValueError("the research portfolio is capped at two 10% Spot sleeves")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive and finite")
        if not np.isfinite(self.cost_edge_multiple) or self.cost_edge_multiple < 1:
            raise ValueError("cost_edge_multiple must be finite and at least one")


def _validate_frames(frames: Mapping[str, pd.DataFrame]) -> tuple[list[str], pd.DatetimeIndex]:
    if len(frames) < 5 or "BTC/USDT" not in frames:
        raise ValueError("at least five symbols including BTC/USDT are required")
    symbols = sorted(frames)
    common: pd.DatetimeIndex | None = None
    for symbol, frame in frames.items():
        required = {"open", "high", "low", "close", "volume"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"{symbol} has missing OHLCV data")
        if frame.index.tz is None or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"{symbol} timestamps must be timezone-aware, unique and chronological")
        values = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{symbol} contains non-finite OHLCV values")
        if (values[["open", "high", "low", "close"]] <= 0).any().any() or (values.volume < 0).any():
            raise ValueError(f"{symbol} contains invalid OHLCV values")
        if (values.high < values[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError(f"{symbol} contains invalid high prices")
        if (values.low > values[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError(f"{symbol} contains invalid low prices")
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 1000:
        raise ValueError("at least 1000 common hourly bars are required")
    common = common.sort_values()
    if not (common.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("common hourly timestamps contain gaps")
    if any(not frames[s].index.equals(common) for s in symbols):
        raise ValueError("all symbols must share the exact same hourly timestamps")
    return symbols, common


def _build_feature_cube(
    frames: Mapping[str, pd.DataFrame], symbols: list[str], index: pd.DatetimeIndex
) -> np.ndarray:
    raw: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = frames[symbol].reindex(index)
        close = frame.close.astype(float)
        returns = close.pct_change(fill_method=None)
        ema = close.ewm(span=168, adjust=False, min_periods=168).mean()
        dollar_volume = (close * frame.volume.astype(float)).rolling(24, min_periods=24).median()
        volume_base = frame.volume.astype(float).rolling(168, min_periods=168).median()
        volume_recent = frame.volume.astype(float).rolling(24, min_periods=24).mean()
        raw[symbol] = pd.DataFrame({
            "mom24": close.pct_change(24, fill_method=None),
            "mom72": close.pct_change(72, fill_method=None),
            "mom168": close.pct_change(168, fill_method=None),
            "mom336": close.pct_change(336, fill_method=None),
            "reversal6": -close.pct_change(6, fill_method=None),
            "vol24": returns.rolling(24, min_periods=24).std(),
            "vol168": returns.rolling(168, min_periods=168).std(),
            "ema_gap": close / ema - 1.0,
            "log_dollar_volume": np.log(dollar_volume.clip(lower=1.0)),
            "volume_spike": volume_recent / volume_base.replace(0, np.nan),
        }, index=index)

    ranked: dict[str, dict[str, np.ndarray]] = {s: {} for s in symbols}
    for name in _RAW:
        wide = pd.DataFrame({s: raw[s][name] for s in symbols}, index=index)
        ranks = wide.rank(axis=1, method="average", pct=True) - 0.5
        for symbol in symbols:
            ranked[symbol][name] = ranks[symbol].to_numpy(dtype=float)

    btc_close = frames["BTC/USDT"].reindex(index).close.astype(float)
    btc_ema = btc_close.ewm(span=168, adjust=False, min_periods=168).mean()
    btc_regime = np.tanh((btc_close / btc_ema - 1.0).to_numpy(dtype=float) * 20.0)
    cube = np.full((len(index), len(symbols), len(_FEATURES)), np.nan, dtype=float)
    for j, symbol in enumerate(symbols):
        f = ranked[symbol]
        m24, m72, m168 = f["mom24"], f["mom72"], f["mom168"]
        lowvol = -f["vol168"]
        liquidity = f["log_dollar_volume"]
        volume = f["volume_spike"]
        columns = [
            f[name] for name in _RAW
        ] + [
            btc_regime,
            m168 * liquidity,
            m72 * lowvol,
            m168 * btc_regime,
            f["reversal6"] * m168,
            m24 * volume,
        ]
        cube[:, j, :] = np.column_stack(columns)
    return cube


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    lower, upper = np.quantile(y, [0.01, 0.99])
    target = np.clip(y, lower, upper)
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return coef, mean, scale


def _metrics(values: list[float]) -> dict:
    winners = [v for v in values if v > 0]
    losers = [-v for v in values if v < 0]
    return {
        "trades": len(values),
        "wins": len(winners),
        "win_rate_pct": 100.0 * len(winners) / len(values) if values else 0.0,
        "profit_factor": sum(winners) / sum(losers) if losers else ("infinite" if winners else 0.0),
        "expectancy_per_trade_pct": float(np.mean(values)) if values else 0.0,
    }


def run_factor_interaction_study(
    frames: Mapping[str, pd.DataFrame],
    *,
    costs: FactorCosts | None = None,
    config: FactorConfig | None = None,
) -> dict:
    """Run weekly-refit, purged, chronological OOS factor-interaction research."""
    costs = costs or FactorCosts()
    config = config or FactorConfig()
    symbols, index = _validate_frames(frames)
    n = len(index)
    if config.train_window_bars < config.warmup_bars:
        raise ValueError("training window must be at least as long as warm-up")
    if config.test_fold_bars % config.sample_step_bars:
        raise ValueError("test fold must be divisible by the daily sampling interval")
    if config.refit_bars != config.test_fold_bars:
        raise ValueError("model refit interval must equal the non-overlapping OOS fold")
    if config.holding_bars != 24 or config.sample_step_bars != 24:
        raise ValueError("this pre-registered candidate uses a 24-hour horizon and daily samples")
    if n <= config.warmup_bars + config.train_window_bars + config.test_fold_bars:
        raise ValueError("insufficient history for training plus one complete OOS fold")

    aligned = {s: frames[s].reindex(index) for s in symbols}
    cube = _build_feature_cube(aligned, symbols, index)
    opens = np.column_stack([aligned[s].open.to_numpy(dtype=float) for s in symbols])
    close = aligned["BTC/USDT"].close.to_numpy(dtype=float)
    ema_btc = pd.Series(close).ewm(span=168, adjust=False, min_periods=168).mean().to_numpy()
    sample_indices = list(range(config.warmup_bars, n - config.holding_bars - 1, config.sample_step_bars))
    fold_starts = list(range(
        config.warmup_bars + config.train_window_bars,
        n - config.test_fold_bars,
        config.test_fold_bars,
    ))
    if not fold_starts:
        raise ValueError("no complete chronological OOS folds available")

    impact = (costs.half_spread_bps + costs.slippage_bps) / 10_000.0
    fee = costs.commission_bps / 10_000.0
    round_trip_rate = costs.round_trip_bps / 10_000.0
    threshold = config.cost_edge_multiple * round_trip_rate
    equity = 1.0
    equity_curve = [equity]
    trade_log: list[dict] = []
    fold_returns: list[float] = []
    symbol_returns: dict[str, float] = dict.fromkeys(symbols, 1.0)
    audit: list[dict] = []
    per_symbol_trade_returns: dict[str, list[float]] = {s: [] for s in symbols}

    for fold_number, start in enumerate(fold_starts, start=1):
        train_start = max(config.warmup_bars, start - config.train_window_bars)
        train_times = [
            i for i in sample_indices
            if train_start <= i and i + config.holding_bars + 1 < start
        ]
        train_x: list[np.ndarray] = []
        train_y: list[float] = []
        for i in train_times:
            features = cube[i]
            if not np.isfinite(features).all():
                continue
            entry_i, exit_i = i + 1, i + config.holding_bars + 1
            for j in range(len(symbols)):
                gross = opens[exit_i, j] / opens[entry_i, j] - 1.0
                if np.isfinite(gross):
                    train_x.append(features[j])
                    train_y.append(gross)
        if len(train_y) < config.min_train_rows:
            continue
        x_train = np.asarray(train_x, dtype=float)
        y_train = np.asarray(train_y, dtype=float)
        coef, mean, scale = _fit_ridge(x_train, y_train, config.ridge_alpha)
        train_label_end = index[max(train_times) + config.holding_bars + 1]
        fold_equity_start = equity
        fold_trade_count = len(trade_log)

        # Six daily decisions per seven-day fold. The final day is reserved so
        # every 24h position exits inside this fold, preventing boundary leakage.
        for offset in range(0, config.test_fold_bars - config.holding_bars, config.sample_step_bars):
            signal_i = start + offset
            if signal_i + config.holding_bars + 1 >= n:
                continue
            features = cube[signal_i]
            if not np.isfinite(features).all():
                continue
            if not np.isfinite(ema_btc[signal_i]) or close[signal_i] <= ema_btc[signal_i]:
                equity_curve.append(equity)
                continue
            design = np.column_stack([
                np.ones(len(symbols)),
                (features - mean) / scale,
            ])
            predictions = design @ coef
            selected = np.argsort(predictions)[::-1]
            selected = [
                int(j) for j in selected[:config.top_n]
                if predictions[j] > threshold
            ]
            period_return = 0.0
            entry_i = signal_i + 1
            exit_i = signal_i + config.holding_bars + 1
            for j in selected:
                entry_raw, exit_raw = opens[entry_i, j], opens[exit_i, j]
                entry_fill = entry_raw * (1.0 + impact)
                exit_fill = exit_raw * (1.0 - impact)
                net = (exit_fill * (1.0 - fee)) / (entry_fill * (1.0 + fee)) - 1.0
                weight = config.allocation_per_asset
                period_return += weight * net
                per_symbol_trade_returns[symbols[j]].append(net)
                symbol_returns[symbols[j]] *= 1.0 + net
                trade_log.append({
                    "symbol": symbols[j],
                    "signal_time": index[signal_i].isoformat(),
                    "entry_time": index[entry_i].isoformat(),
                    "exit_time": index[exit_i].isoformat(),
                    "predicted_gross_return_pct": float(predictions[j] * 100.0),
                    "net_return_pct": float(net * 100.0),
                    "allocation_pct": weight * 100.0,
                    "round_trip_cost_bps": costs.round_trip_bps,
                    "fold": fold_number,
                })
            equity *= 1.0 + period_return
            equity_curve.append(equity)
        fold_return = (equity / fold_equity_start - 1.0) * 100.0
        fold_returns.append(fold_return)
        audit.append({
            "fold": fold_number,
            "train_start": index[train_start].isoformat(),
            "train_label_end": train_label_end.isoformat(),
            "test_start": index[start].isoformat(),
            "test_end": index[start + config.test_fold_bars - 1].isoformat(),
            "training_rows": len(train_y),
            "trades": len(trade_log) - fold_trade_count,
            "net_return_pct": fold_return,
        })

    if not audit:
        raise ValueError("no OOS folds met the minimum training sample requirement")
    trade_returns = [float(t["net_return_pct"]) / 100.0 for t in trade_log]
    peak = 1.0
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak if peak else 0.0)
    metrics = _metrics(trade_returns)
    return {
        "strategy": "weekly_refit_cross_sectional_factor_interaction_ridge",
        "data_interval": "1h",
        "symbols": symbols,
        "bars": n,
        "first_bar": index[0].isoformat(),
        "last_bar": index[-1].isoformat(),
        "features": list(_FEATURES),
        "parameters": {
            "train_window_bars": config.train_window_bars,
            "warmup_bars": config.warmup_bars,
            "test_fold_bars": config.test_fold_bars,
            "holding_bars": config.holding_bars,
            "sample_step_bars": config.sample_step_bars,
            "ridge_alpha": config.ridge_alpha,
            "top_n": config.top_n,
            "allocation_per_asset": config.allocation_per_asset,
            "cost_edge_multiple": config.cost_edge_multiple,
            "btc_close_above_ema168_filter": True,
        },
        "costs": {
            "commission_bps_per_fill": costs.commission_bps,
            "half_spread_bps_per_fill": costs.half_spread_bps,
            "slippage_bps_per_fill": costs.slippage_bps,
            "round_trip_bps": costs.round_trip_bps,
            "minimum_predicted_gross_return_pct": threshold * 100.0,
        },
        "folds": len(audit),
        "positive_folds": sum(v > 0 for v in fold_returns),
        "compounded_oos_return_pct": (equity - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "aggregate": metrics,
        "per_symbol_compounded_return_pct": {
            s: (value - 1.0) * 100.0 for s, value in symbol_returns.items()
        },
        "per_symbol_trade_metrics": {
            s: _metrics(values) for s, values in per_symbol_trade_returns.items()
        },
        "fold_audit": audit,
        "trade_log": trade_log,
        "research_only": True,
        "execution_note": (
            "Binance public Spot proxy; simulated next-open fills, long-only, cash remainder. "
            "No Bybit execution or order integration."
        ),
        "validation_note": (
            "Rolling 180-day training, 24-hour purged labels, weekly non-overlapping OOS folds; "
            "no parameter search and no use of future labels in training."
        ),
    }
