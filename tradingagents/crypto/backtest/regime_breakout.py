"""Research-only regime-aware spot breakout strategy.

This module is deliberately isolated from the paper worker. It produces a
long-only signal from completed OHLCV bars and does not place orders, size a
portfolio, or modify the deployed TJR strategy.

The entry is a trend-filtered Donchian breakout with a relative-volume filter.
All rolling breakout/volume references are shifted by one bar to avoid using
the signal bar in its own threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeBreakoutSettings:
    fast_ema: int = 20
    slow_ema: int = 50
    breakout_lookback: int = 20
    volume_lookback: int = 20
    atr_period: int = 14
    volume_multiple: float = 1.20
    stop_atr_multiple: float = 2.5
    reward_risk: float = 3.0
    max_holding_bars: int = 48

    def __post_init__(self) -> None:
        for name in ("fast_ema", "slow_ema", "breakout_lookback",
                     "volume_lookback", "atr_period", "max_holding_bars"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        if self.volume_multiple <= 0 or self.stop_atr_multiple <= 0 or self.reward_risk <= 0:
            raise ValueError("volume_multiple, stop_atr_multiple and reward_risk must be positive")


def indicators(frame: pd.DataFrame, settings: RegimeBreakoutSettings = RegimeBreakoutSettings()) -> pd.DataFrame:
    """Return causal indicators for a chronological OHLCV frame."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV index must be strictly chronological and unique")
    out = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float).copy()
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out["volume"] < 0).any():
        raise ValueError("prices must be positive and volume cannot be negative")
    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [out["high"] - out["low"],
         (out["high"] - previous_close).abs(),
         (out["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.rolling(settings.atr_period, min_periods=settings.atr_period).mean()
    out["ema_fast"] = out["close"].ewm(span=settings.fast_ema, adjust=False,
                                         min_periods=settings.fast_ema).mean()
    out["ema_slow"] = out["close"].ewm(span=settings.slow_ema, adjust=False,
                                         min_periods=settings.slow_ema).mean()
    out["ema_slow_rising"] = out["ema_slow"] > out["ema_slow"].shift(5)
    out["breakout_level"] = out["high"].rolling(
        settings.breakout_lookback, min_periods=settings.breakout_lookback
    ).max().shift(1)
    out["volume_baseline"] = out["volume"].rolling(
        settings.volume_lookback, min_periods=settings.volume_lookback
    ).mean().shift(1)
    out["long_entry"] = (
        (out["close"] > out["breakout_level"])
        & (out["ema_fast"] > out["ema_slow"])
        & out["ema_slow_rising"]
        & (out["close"] > out["ema_fast"])
        & (out["volume"] >= out["volume_baseline"] * settings.volume_multiple)
        & out["atr"].notna()
    )
    return out


def entry_levels(row: pd.Series, settings: RegimeBreakoutSettings = RegimeBreakoutSettings()) -> tuple[float, float, float] | None:
    """Return (reference entry, stop, target) for a completed signal bar."""
    entry = float(row["close"])
    atr = float(row["atr"])
    if not pd.notna(atr) or atr <= 0 or entry <= 0:
        return None
    stop = entry - settings.stop_atr_multiple * atr
    risk = entry - stop
    target = entry + settings.reward_risk * risk
    if stop <= 0 or not (stop < entry < target):
        return None
    return entry, stop, target
