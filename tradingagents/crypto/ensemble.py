"""Regime-adaptive crypto signal research module.

Research-only: this module emits candidate long/flat signals from completed OHLCV
bars. It is deliberately not wired to the paper worker or broker.
"""
from __future__ import annotations  # noqa: I001

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Regime = Literal["trend", "range", "uncertain"]
Action = Literal["long", "flat"]
StrategyMode = Literal["ensemble", "trend", "range", "breakout"]


@dataclass(frozen=True)
class EnsembleSignal:
    action: Action
    regime: Regime
    module: str
    score: float
    close: float
    stop: float | None
    reason: str


@dataclass(frozen=True)
class EnsembleSettings:
    fast_ema: int = 50
    slow_ema: int = 200
    adx_period: int = 14
    adx_trend_min: float = 22.0
    adx_range_max: float = 18.0
    donchian_entry: int = 20
    donchian_exit: int = 10
    momentum_bars: int = 20
    range_period: int = 20
    range_zscore: float = 2.0
    atr_period: int = 14
    atr_stop_multiple: float = 2.5
    max_extension_atr: float = 2.0


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    if len(frame) < 220:
        raise ValueError("at least 220 completed bars are required")
    data = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float).copy()
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("OHLCV data must be finite and non-null")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high is inconsistent with OHLC")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low is inconsistent with OHLC")
    return data


def _indicators(data: pd.DataFrame, s: EnsembleSettings) -> pd.DataFrame:
    x = data.copy()
    close, high, low = x.close, x.high, x.low
    previous = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / s.atr_period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / s.adx_period, adjust=False).mean() / x.atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / s.adx_period, adjust=False).mean() / x.atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    x["adx"] = dx.ewm(alpha=1 / s.adx_period, adjust=False).mean()
    x["plus_di"], x["minus_di"] = plus_di, minus_di
    x["fast_ema"] = close.ewm(span=s.fast_ema, adjust=False).mean()
    x["slow_ema"] = close.ewm(span=s.slow_ema, adjust=False).mean()
    x["prior_high"] = high.shift(1).rolling(s.donchian_entry).max()
    x["prior_low_exit"] = low.shift(1).rolling(s.donchian_exit).min()
    x["momentum"] = close.pct_change(s.momentum_bars)
    x["range_mean"] = close.rolling(s.range_period).mean()
    x["range_std"] = close.rolling(s.range_period).std(ddof=0)
    x["zscore"] = (close - x.range_mean) / x.range_std.replace(0, float("nan"))
    return x


def evaluate_ensemble(
    frame: pd.DataFrame,
    settings: EnsembleSettings = EnsembleSettings(),
    *,
    mode: StrategyMode = "ensemble",
) -> EnsembleSignal:
    """Evaluate the latest completed bar; callers must exclude any still-open candle."""
    if mode not in {"ensemble", "trend", "range", "breakout"}:
        raise ValueError("mode must be ensemble, trend, range, or breakout")
    data = _validate(frame)
    positive_periods = (
        settings.fast_ema, settings.slow_ema, settings.adx_period,
        settings.donchian_entry, settings.donchian_exit,
        settings.momentum_bars, settings.range_period, settings.atr_period,
    )
    if any(not isinstance(period, int) or period <= 0 for period in positive_periods):
        raise ValueError("indicator periods must be positive integers")
    if settings.fast_ema >= settings.slow_ema:
        raise ValueError("fast_ema must be smaller than slow_ema")
    if settings.atr_stop_multiple <= 0 or settings.max_extension_atr <= 0:
        raise ValueError("ATR multiples must be positive")
    x = _indicators(data, settings)
    row = x.iloc[-1]
    needed = ["atr", "adx", "fast_ema", "slow_ema", "prior_high", "momentum",
              "range_mean", "range_std", "zscore", "plus_di", "minus_di"]
    if any(pd.isna(row[k]) for k in needed) or row.atr <= 0:
        return EnsembleSignal("flat", "uncertain", "none", 0.0, float(row.close), None,
                              "indicator warm-up or invalid volatility")
    close, atr = float(row.close), float(row.atr)
    if row.adx >= settings.adx_trend_min:
        regime: Regime = "trend"
    elif row.adx <= settings.adx_range_max:
        regime = "range"
    else:
        regime = "uncertain"

    if regime == "trend" or mode == "breakout":
        aligned = row.fast_ema > row.slow_ema and row.plus_di > row.minus_di
        momentum_ok = row.momentum > 0
        breakout = close > row.prior_high
        extension_ok = (close - row.fast_ema) / atr <= settings.max_extension_atr
        if mode == "range":
            return EnsembleSignal("flat", regime, "range_mean_reversion", 0.0, close, None,
                                  "range-only mode: trend regime is disabled")
        if mode == "breakout":
            aligned = True
        if aligned and momentum_ok and breakout and extension_ok:
            stop = close - settings.atr_stop_multiple * atr
            return EnsembleSignal("long", regime, "trend_donchian_momentum", 1.0,
                                  close, stop, "EMA/DI alignment, positive momentum and prior-channel breakout")
        return EnsembleSignal("flat", regime, "trend_donchian_momentum", 0.0,
                              close, None, "trend regime lacks a fresh, non-extended confirmed breakout")

    if regime == "range" or mode == "range":
        if mode in {"trend", "breakout"}:
            return EnsembleSignal("flat", regime, "trend_donchian_momentum", 0.0, close, None,
                                  "trend-only mode: range regime is disabled")
        # Mean reversion is only allowed in a low-ADX regime. Never average down:
        # require a bullish close relative to the prior bar after a lower-band stretch.
        stretched = row.zscore <= -settings.range_zscore
        reversal = close > float(x.close.iloc[-2])
        if stretched and reversal:
            stop = close - settings.atr_stop_multiple * atr
            return EnsembleSignal("long", regime, "range_mean_reversion", 0.7,
                                  close, stop, "lower-band stretch followed by a bullish close")
        return EnsembleSignal("flat", regime, "range_mean_reversion", 0.0,
                              close, None, "no confirmed lower-band reversal")
    return EnsembleSignal("flat", regime, "none", 0.0, close, None,
                          "ambiguous regime: stand aside")
