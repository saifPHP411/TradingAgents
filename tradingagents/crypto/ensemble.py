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
StrategyMode = Literal["ensemble", "trend", "range", "breakout", "trend_pullback"]


@dataclass(frozen=True)
class EnsembleSignal:
    action: Action
    regime: Regime
    module: str
    score: float
    close: float
    stop: float | None
    reason: str
    should_exit: bool = False


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
    rsi_period: int = 14
    volume_period: int = 20
    atr_expansion_period: int = 20
    atr_expansion_multiple: float = 1.05
    trend_rsi_min: float = 55.0
    range_rsi_max: float = 35.0
    volume_multiple: float = 1.1
    atr_stop_multiple: float = 2.5
    max_extension_atr: float = 2.0


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    if len(frame) < 220:
        raise ValueError("at least 220 completed bars are required")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("completed bars must be strictly increasing and unique")
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
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / s.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / s.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    x["rsi"] = 100 - 100 / (1 + rs)
    x.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    x.loc[(loss == 0) & (gain == 0), "rsi"] = 50.0
    x["volume_mean"] = x.volume.rolling(s.volume_period).mean()
    x["atr_mean"] = x.atr.rolling(s.atr_expansion_period).mean()
    x["atr_expansion"] = x.atr / x.atr_mean.replace(0, float("nan"))
    return x


def _validate_settings(settings: EnsembleSettings) -> None:
    positive_periods = (
        settings.fast_ema, settings.slow_ema, settings.adx_period,
        settings.donchian_entry, settings.donchian_exit,
        settings.momentum_bars, settings.range_period, settings.atr_period,
        settings.rsi_period, settings.volume_period, settings.atr_expansion_period,
    )
    if any(type(period) is not int or period <= 0 for period in positive_periods):
        raise ValueError("indicator periods must be positive integers")
    numeric_thresholds = (
        settings.adx_trend_min, settings.adx_range_max, settings.range_zscore,
        settings.atr_expansion_multiple, settings.trend_rsi_min,
        settings.range_rsi_max, settings.volume_multiple,
        settings.atr_stop_multiple, settings.max_extension_atr,
    )
    if not np.isfinite(np.asarray(numeric_thresholds, dtype=float)).all():
        raise ValueError("indicator thresholds must be finite numbers")
    if settings.fast_ema >= settings.slow_ema:
        raise ValueError("fast_ema must be smaller than slow_ema")
    if (
        settings.atr_stop_multiple <= 0
        or settings.max_extension_atr <= 0
        or settings.volume_multiple <= 0
        or settings.range_zscore <= 0
        or settings.atr_expansion_multiple <= 0
    ):
        raise ValueError("ATR, volume and range thresholds must be positive")
    if not 0 <= settings.range_rsi_max <= 100 or not 0 <= settings.trend_rsi_min <= 100:
        raise ValueError("RSI thresholds must be between 0 and 100")
    if not 0 <= settings.adx_range_max < settings.adx_trend_min <= 100:
        raise ValueError("ADX regime thresholds must satisfy 0 <= range < trend <= 100")


def _evaluate_indicator_row(
    indicators: pd.DataFrame,
    position: int,
    settings: EnsembleSettings,
    mode: StrategyMode = "ensemble",
) -> EnsembleSignal:
    """Evaluate one precomputed causal indicator row without recalculating history."""
    if mode not in {"ensemble", "trend", "range", "breakout", "trend_pullback"}:
        raise ValueError("mode must be ensemble, trend, range, or breakout")
    if position <= 0 or position >= len(indicators):
        raise ValueError("indicator position must have a previous completed bar")
    row = indicators.iloc[position]
    needed = [
        "atr", "adx", "fast_ema", "slow_ema", "prior_high", "prior_low_exit",
        "momentum", "range_mean", "range_std", "zscore", "plus_di", "minus_di",
        "rsi", "volume_mean", "atr_expansion",
    ]
    if any(pd.isna(row[key]) for key in needed) or row.atr <= 0:
        return EnsembleSignal(
            "flat", "uncertain", "none", 0.0, float(row.close), None,
            "indicator warm-up or invalid volatility", True,
        )
    close, atr = float(row.close), float(row.atr)
    if row.adx >= settings.adx_trend_min:
        regime: Regime = "trend"
    elif row.adx <= settings.adx_range_max:
        regime = "range"
    else:
        regime = "uncertain"

    if mode == "range" and regime != "range":
        return EnsembleSignal(
            "flat", regime, "range_mean_reversion", 0.0, close, None,
            "range-only mode requires a confirmed range regime", True,
        )

    if mode in {"breakout", "trend_pullback"} and regime != "trend":
        module = "volatility_breakout" if mode == "breakout" else "trend_pullback_momentum"
        return EnsembleSignal(
            "flat", regime, module, 0.0, close, None,
            "trend module requires a confirmed trend regime", True,
        )

    if regime == "trend" or mode in {"breakout", "trend_pullback"}:
        aligned = row.fast_ema > row.slow_ema and row.plus_di > row.minus_di
        momentum_ok = row.momentum > 0
        breakout = close > row.prior_high
        extension_ok = (close - row.fast_ema) / atr <= settings.max_extension_atr
        if mode == "range":
            return EnsembleSignal(
                "flat", regime, "range_mean_reversion", 0.0, close, None,
                "range-only mode: trend regime is disabled", True,
            )
        if mode == "breakout":
            aligned = close > row.slow_ema
        if mode == "trend_pullback":
            aligned = row.fast_ema > row.slow_ema and row.plus_di > row.minus_di
        volume_ok = row.volume >= row.volume_mean * settings.volume_multiple
        rsi_ok = row.rsi >= settings.trend_rsi_min
        volatility_expansion_ok = (
            mode != "breakout" or row.atr_expansion >= settings.atr_expansion_multiple
        )
        if mode != "trend_pullback" and aligned and momentum_ok and breakout and extension_ok and volume_ok and rsi_ok and volatility_expansion_ok:
            stop = close - settings.atr_stop_multiple * atr
            module = "volatility_breakout" if mode == "breakout" else "trend_donchian_momentum"
            return EnsembleSignal(
                "long", regime, module, 1.0, close, stop,
                "Donchian breakout with trend, momentum, RSI, volume and ATR expansion confirmation",
            )

        if mode in {"ensemble", "trend_pullback"}:
            previous = indicators.iloc[position - 1]
            pullback_reclaim = (
                previous.close <= previous.fast_ema
                and close > row.fast_ema
                and close > previous.close
            )
            pullback_volume_ok = row.volume >= row.volume_mean * 0.8
            pullback_rsi_ok = row.rsi >= 50.0
            if (
                aligned and momentum_ok and pullback_reclaim and extension_ok
                and pullback_volume_ok and pullback_rsi_ok
            ):
                stop = close - settings.atr_stop_multiple * atr
                return EnsembleSignal(
                    "long", regime, "trend_pullback_momentum", 0.85, close, stop,
                    "EMA pullback reclaim with positive trend momentum and volume confirmation",
                )
        trend_exit = (
            close < row.prior_low_exit
            or row.fast_ema < row.slow_ema
            or row.plus_di < row.minus_di
        )
        return EnsembleSignal(
            "flat", regime, "trend_donchian_momentum", 0.0, close, None,
            "trend exit condition" if trend_exit else "no fresh breakout; keep existing position",
            bool(trend_exit),
        )

    if regime == "range" or mode == "range":
        if mode in {"trend", "breakout", "trend_pullback"}:
            return EnsembleSignal(
                "flat", regime, "trend_donchian_momentum", 0.0, close, None,
                "trend-only mode: range regime is disabled", True,
            )
        stretched = row.zscore <= -settings.range_zscore
        reversal = close > float(indicators.close.iloc[position - 1])
        oversold = row.rsi <= settings.range_rsi_max
        if stretched and reversal and oversold:
            stop = close - settings.atr_stop_multiple * atr
            return EnsembleSignal(
                "long", regime, "range_mean_reversion", 0.7, close, stop,
                "lower-band stretch with oversold RSI and bullish reversal",
            )
        range_exit = close >= float(row.range_mean)
        return EnsembleSignal(
            "flat", regime, "range_mean_reversion", 0.0, close, None,
            "mean-reversion target reached" if range_exit else "no fresh oversold reversal; keep existing position",
            bool(range_exit),
        )
    return EnsembleSignal(
        "flat", regime, "none", 0.0, close, None,
        "ambiguous regime: stand aside", True,
    )


def evaluate_ensemble(
    frame: pd.DataFrame,
    settings: EnsembleSettings = EnsembleSettings(),
    *,
    mode: StrategyMode = "ensemble",
) -> EnsembleSignal:
    """Evaluate the latest completed bar; callers must exclude any still-open candle."""
    if mode not in {"ensemble", "trend", "range", "breakout", "trend_pullback"}:
        raise ValueError("mode must be ensemble, trend, range, or breakout")
    _validate_settings(settings)
    data = _validate(frame)
    indicators = _indicators(data, settings)
    return _evaluate_indicator_row(indicators, len(indicators) - 1, settings, mode)
