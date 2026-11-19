"""Multi-timeframe, short-hold scalping signals for Bybit USDT linear perpetuals.

This module only calculates signals and risk distances. It never submits orders.
All indicators use completed candles except the live Mark Price used for the
1-minute reclaim trigger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScalpingConfig:
    leverage: float = 2.0
    max_hold_minutes: int = 15
    cooldown_minutes: int = 3
    atr_period: int = 14
    stop_atr_multiple: float = 1.25
    reward_risk: float = 2.0
    min_stop_fraction: float = 0.0015
    max_stop_fraction: float = 0.0045
    cost_per_fill: float = 0.00085
    min_net_edge_fraction: float = 0.0010

    def __post_init__(self) -> None:
        if self.leverage != 2.0:
            raise ValueError("Demo scalper leverage is hard-limited to 2x")
        if not 1 <= self.max_hold_minutes <= 30:
            raise ValueError("max_hold_minutes must be in [1, 30]")
        if not 1 <= self.cooldown_minutes <= 30:
            raise ValueError("cooldown_minutes must be in [1, 30]")
        if self.atr_period < 5:
            raise ValueError("atr_period must be at least five")
        for name in (
            "stop_atr_multiple",
            "reward_risk",
            "min_stop_fraction",
            "max_stop_fraction",
            "cost_per_fill",
            "min_net_edge_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_stop_fraction >= self.max_stop_fraction:
            raise ValueError("minimum stop distance must be below maximum stop distance")


def _validate(frame: pd.DataFrame, name: str, minimum: int) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"{name} must contain OHLC candles")
    if len(frame) < minimum:
        raise ValueError(f"{name} requires at least {minimum} candles")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be ordered and unique")
    values = frame.loc[:, ["open", "high", "low", "close"]].astype(float)
    if values.isna().any().any() or not values.map(math.isfinite).all().all() or (values <= 0).any().any():
        raise ValueError(f"{name} contains invalid prices")
    if (values.high < values[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{name} high values are inconsistent")
    if (values.low > values[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{name} low values are inconsistent")
    result = values.copy()
    if "volume" in frame.columns:
        volume = frame["volume"].astype(float)
        if volume.isna().any() or not volume.map(math.isfinite).all() or (volume < 0).any():
            raise ValueError(f"{name} contains invalid volume")
        result["volume"] = volume
    return result


def _indicators(frame: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous).abs(),
            (result["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = true_range.rolling(atr_period, min_periods=atr_period).mean()
    result["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    result["ema21"] = close.ewm(span=21, adjust=False, min_periods=21).mean()
    result["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()

    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    denominator = avg_gain + avg_loss
    result["rsi"] = (100.0 * avg_gain / denominator.replace(0, math.nan)).fillna(50.0)
    result.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100.0
    result.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0.0
    return result


def _bias(frame: pd.DataFrame, fast: str, slow: str) -> int:
    last = frame.iloc[-1]
    price = float(last["close"])
    fast_value = float(last[fast])
    slow_value = float(last[slow])
    if not all(math.isfinite(v) for v in (price, fast_value, slow_value)):
        return 0
    if price > fast_value > slow_value:
        return 1
    if price < fast_value < slow_value:
        return -1
    return 0


def evaluate_scalping_signal(
    one_minute: pd.DataFrame,
    five_minute: pd.DataFrame,
    fifteen_minute: pd.DataFrame,
    mark_price: float,
    config: ScalpingConfig | None = None,
) -> dict:
    """Evaluate a 1m live reclaim with completed 5m/15m directional filters."""
    cfg = config or ScalpingConfig()
    mark = float(mark_price)
    if not math.isfinite(mark) or mark <= 0:
        raise ValueError("mark_price must be finite and positive")

    one = _validate(one_minute, "one_minute", 40)
    five = _validate(five_minute, "five_minute", 55)
    fifteen = _validate(fifteen_minute, "fifteen_minute", 55)
    # The 1m input must include its forming candle at the end. Signals only use
    # the previous, completed 1m candle and its live Mark Price reclaim.
    one_closed = one.iloc[:-1]
    if len(one_closed) < 39:
        return {"signal": 0, "reason": "no_completed_1m_candle"}
    one_features = _indicators(one_closed, cfg.atr_period)
    five_features = _indicators(five, cfg.atr_period)
    fifteen_features = _indicators(fifteen, cfg.atr_period)

    long_15m = _bias(fifteen_features, "ema20", "ema50") == 1
    short_15m = _bias(fifteen_features, "ema20", "ema50") == -1
    long_5m = _bias(five_features, "ema9", "ema21") == 1
    short_5m = _bias(five_features, "ema9", "ema21") == -1

    last = one_features.iloc[-1]
    previous = one_features.iloc[-2]
    atr = float(last["atr"])
    ema9 = float(last["ema9"])
    rsi = float(last["rsi"])
    previous_rsi = float(previous["rsi"])
    if not all(math.isfinite(v) and v > 0 for v in (atr, ema9)):
        return {"signal": 0, "reason": "indicator_warmup", "atr": None}

    volume_ok = True
    if "volume" in one_features.columns:
        recent_volumes = one_features["volume"].iloc[-21:-1]
        average_volume = float(recent_volumes.mean()) if len(recent_volumes) else 0.0
        last_volume = float(last["volume"])
        volume_ok = average_volume <= 0 or last_volume >= average_volume * 0.50

    long_reclaim = (
        long_15m
        and long_5m
        and float(last["low"]) <= ema9 * 1.0015
        and float(last["close"]) <= ema9 * 1.0020
        and mark > ema9
        and mark > float(last["close"])
        and 45.0 <= rsi <= 72.0
        and rsi >= previous_rsi
        and volume_ok
    )
    short_reclaim = (
        short_15m
        and short_5m
        and float(last["high"]) >= ema9 * 0.9985
        and float(last["close"]) >= ema9 * 0.9980
        and mark < ema9
        and mark < float(last["close"])
        and 28.0 <= rsi <= 55.0
        and rsi <= previous_rsi
        and volume_ok
    )

    if long_reclaim == short_reclaim:
        reason = "conflicting_filters" if long_reclaim else "no_aligned_pullback_reclaim"
        return {
            "signal": 0,
            "reason": reason,
            "atr": atr,
            "long_bias": long_15m and long_5m,
            "short_bias": short_15m and short_5m,
            "rsi": rsi,
            "ema9": ema9,
        }

    direction = 1 if long_reclaim else -1
    stop_distance = max(atr * cfg.stop_atr_multiple, mark * cfg.min_stop_fraction)
    stop_fraction = stop_distance / mark
    if stop_fraction > cfg.max_stop_fraction:
        return {
            "signal": 0,
            "reason": "atr_stop_too_wide",
            "atr": atr,
            "rsi": rsi,
            "ema9": ema9,
        }

    target_distance = stop_distance * cfg.reward_risk
    minimum_target_fraction = 2 * cfg.cost_per_fill + cfg.min_net_edge_fraction
    if target_distance / mark < minimum_target_fraction:
        return {
            "signal": 0,
            "reason": "insufficient_net_reward_after_costs",
            "atr": atr,
            "rsi": rsi,
            "ema9": ema9,
        }

    return {
        "signal": direction,
        "reason": "long_1m_reclaim" if direction > 0 else "short_1m_reclaim",
        "atr": atr,
        "rsi": rsi,
        "ema9": ema9,
        "stop_loss": mark - direction * stop_distance,
        "take_profit": mark + direction * target_distance,
        "stop_distance": stop_distance,
        "target_distance": target_distance,
        "reward_risk": cfg.reward_risk,
        "long_bias": long_15m and long_5m,
        "short_bias": short_15m and short_5m,
    }
