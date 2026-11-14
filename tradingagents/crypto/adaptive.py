"""Research-only adaptive crypto market analysis; never submits orders."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

Regime = Literal["bull_trend", "bear_trend", "range", "high_volatility", "uncertain"]
Strategy = Literal["trend_pullback", "range_reversion", "breakout", "none"]

@dataclass(frozen=True)
class AdaptiveConfig:
    fast_ema: int = 20
    slow_ema: int = 50
    atr_period: int = 14
    lookback: int = 50
    breakout_lookback: int = 20
    adx_trend_min: float = 20.0
    adx_range_max: float = 17.0
    efficiency_trend_min: float = 0.30
    efficiency_range_max: float = 0.25
    relative_volume_min: float = 1.30
    max_atr_pct: float = 0.08
    max_spread_bps: float = 20.0
    estimated_round_trip_cost_bps: float = 30.0
    min_net_reward_risk: float = 1.25
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10

    def __post_init__(self):
        for name in ("fast_ema", "slow_ema", "atr_period", "lookback", "breakout_lookback"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        numeric_limits = (
            self.adx_trend_min, self.adx_range_max, self.efficiency_trend_min,
            self.efficiency_range_max, self.relative_volume_min, self.max_atr_pct,
            self.max_spread_bps, self.estimated_round_trip_cost_bps,
            self.min_net_reward_risk, self.risk_fraction, self.max_notional_fraction,
        )
        if any(not pd.notna(value) or abs(value) == float("inf") for value in numeric_limits):
            raise ValueError("configuration values must be finite")
        if not 0 < self.risk_fraction <= 0.02 or not 0 < self.max_notional_fraction <= 1:
            raise ValueError("risk and notional fractions exceed research safety bounds")
        if min(self.max_spread_bps, self.estimated_round_trip_cost_bps) < 0:
            raise ValueError("cost limits cannot be negative")
        if self.max_atr_pct <= 0 or self.relative_volume_min <= 0 or self.min_net_reward_risk <= 0:
            raise ValueError("volatility, volume, and reward/risk thresholds must be positive")
        if not 0 <= self.efficiency_range_max < self.efficiency_trend_min <= 1:
            raise ValueError("efficiency thresholds must satisfy 0 <= range < trend <= 1")
        if not 0 <= self.adx_range_max < self.adx_trend_min:
            raise ValueError("ADX thresholds must satisfy 0 <= range < trend")

@dataclass(frozen=True)
class AdaptiveDecision:
    action: Literal["buy_candidate", "no_trade"]
    regime: Regime
    strategy: Strategy
    confidence: float
    reference_price: float | None
    stop_price: float | None
    target_price: float | None
    reasons: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    def to_dict(self):
        return asdict(self)

def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        return frame.copy()
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV index must be strictly chronological and unique")
    data = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
    if data.isna().any().any() or not data.map(lambda x:
        pd.notna(x) and abs(x) != float("inf")).all().all():
        raise ValueError("OHLCV values must be finite and non-null")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data.volume < 0).any():
        raise ValueError("prices must be positive and volume non-negative")
    if (data.high < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be the greatest OHLC price")
    if (data.low > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be the smallest OHLC price")
    return data

def _features(data: pd.DataFrame, cfg: AdaptiveConfig) -> pd.DataFrame:
    x = data.copy()
    prev = x.close.shift()
    tr = pd.concat([x.high-x.low, (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
    x["atr_pct"] = x.atr / x.close
    x["ema_fast"] = x.close.ewm(span=cfg.fast_ema, adjust=False, min_periods=cfg.fast_ema).mean()
    x["ema_slow"] = x.close.ewm(span=cfg.slow_ema, adjust=False, min_periods=cfg.slow_ema).mean()
    up, down = x.high.diff(), -x.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    smooth_atr = tr.ewm(alpha=1/cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    pdi = 100*plus.ewm(alpha=1/cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()/smooth_atr
    mdi = 100*minus.ewm(alpha=1/cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()/smooth_atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, pd.NA)
    x["adx"] = dx.ewm(alpha=1/cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    net = x.close.diff(cfg.lookback).abs()
    path = x.close.diff().abs().rolling(cfg.lookback, min_periods=cfg.lookback).sum()
    x["efficiency"] = net/path.replace(0, pd.NA)
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    relative_strength = gain.div(loss.where(loss > 0))
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    x["rsi"] = rsi
    x["prior_high"] = x.high.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)
    x["prior_low"] = x.low.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).min().shift(1)
    x["volume_base"] = x.volume.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).mean().shift(1)
    x["atr_base"] = x.atr_pct.rolling(cfg.lookback, min_periods=cfg.lookback).median().shift(1)
    return x

def analyze_market(frame: pd.DataFrame, *, config: AdaptiveConfig = AdaptiveConfig(),
                   spread_bps: float | None = None, data_is_fresh: bool = True,
                   closed_bars_only: bool = True) -> AdaptiveDecision:
    """Return a deterministic analysis of the latest completed bar, never an order."""
    data = _validate(frame)
    blockers = []
    if not data_is_fresh:
        blockers.append("market data is stale")
    if not closed_bars_only:
        blockers.append("forming bars cannot drive decisions")
    if spread_bps is None:
        blockers.append("spread is unknown")
    elif spread_bps < 0 or spread_bps > config.max_spread_bps:
        blockers.append("spread exceeds liquidity limit")
    needed = max(config.slow_ema, config.lookback, config.breakout_lookback, config.atr_period)+5
    if len(data) < needed:
        blockers.append("insufficient closed-bar history")
    if blockers:
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None, (), tuple(blockers), {"bars":len(data)})
    x = _features(data, config)
    r = x.iloc[-1]
    required = ("atr", "atr_pct", "ema_fast", "ema_slow", "adx", "efficiency", "rsi",
                "prior_high", "prior_low", "volume_base", "atr_base")
    if any(pd.isna(r[k]) for k in required):
        return AdaptiveDecision("no_trade", "uncertain", "none", 0, None, None, None, (), ("indicator warm-up incomplete",), {"bars":len(data)})
    price, atr = float(r.close), float(r.atr)
    bullish = r.ema_fast > r.ema_slow and price > r.ema_slow
    bearish = r.ema_fast < r.ema_slow and price < r.ema_slow
    expanding = r.atr_base > 0 and r.atr_pct > r.atr_base*2
    if r.atr_pct > config.max_atr_pct or expanding:
        regime: Regime = "high_volatility"
    elif bullish and r.adx >= config.adx_trend_min and r.efficiency >= config.efficiency_trend_min:
        regime = "bull_trend"
    elif bearish and r.adx >= config.adx_trend_min and r.efficiency >= config.efficiency_trend_min:
        regime = "bear_trend"
    elif r.adx <= config.adx_range_max and r.efficiency <= config.efficiency_range_max:
        regime = "range"
    else:
        regime = "uncertain"
    strategy: Strategy = "none"
    confidence, reasons = 0.0, []
    if regime == "bull_trend":
        if price > r.ema_fast and x.close.iloc[-2] <= x.ema_fast.iloc[-2]:
            strategy, confidence = "trend_pullback", min(.90, .55+(float(r.adx)-config.adx_trend_min)/100+float(r.efficiency)*.25)
            reasons += ["bullish EMA structure", "completed bar reclaimed fast EMA after pullback"]
        elif price > r.prior_high and r.volume >= r.volume_base*config.relative_volume_min:
            strategy, confidence = "breakout", min(.90, .58+float(r.efficiency)*.25)
            reasons += ["close broke prior-bar range high", "relative volume confirms breakout"]
    elif regime == "range":
        width = float(r.prior_high-r.prior_low)
        if width > 0 and r.rsi <= 30 and price <= r.prior_low+width*.15:
            strategy, confidence = "range_reversion", .62
            reasons += ["low ADX and path efficiency indicate range", "oversold near lower range boundary"]
        elif width > 0 and r.rsi >= 70 and price >= r.prior_high-width*.15:
            reasons.append("overbought at upper range boundary; long-only engine abstains")
    elif regime == "bear_trend":
        reasons.append("bear trend detected; shorting is disabled")
    elif regime == "high_volatility":
        reasons.append("volatility expansion suspends entries")
    else:
        reasons.append("mixed market structure; no expert has a clear setup")
    stop = target = None
    if strategy != "none":
        if strategy == "range_reversion":
            stop = price - 1.5 * atr
            target = (float(r.prior_high) + float(r.prior_low)) / 2
        else:
            stop = price - 2 * atr
            target = price + 4 * atr
        risk_distance = price - stop
        reward_distance = target - price
        cost_distance = (config.estimated_round_trip_cost_bps / 10_000) * price
        net_reward_risk = (reward_distance - cost_distance) / (risk_distance + cost_distance)
        if stop <= 0 or reward_distance <= 0 or net_reward_risk < config.min_net_reward_risk:
            blockers.append("estimated round-trip costs erase minimum net reward/risk")
            strategy, confidence, stop, target = "none", 0.0, None, None
        else:
            reasons.append("reference risk geometry passes estimated-cost gate")
    diagnostics = {"bar_time":str(data.index[-1]), "bars":len(data), "price":price,
                   "atr":atr, "atr_pct":float(r.atr_pct), "adx":float(r.adx),
                   "efficiency_ratio":float(r.efficiency), "rsi":float(r.rsi),
                   "ema_fast":float(r.ema_fast), "ema_slow":float(r.ema_slow),
                   "prior_range_high":float(r.prior_high), "prior_range_low":float(r.prior_low),
                   "relative_volume":float(r.volume/r.volume_base) if r.volume_base > 0 else None,
                   "spread_bps":spread_bps, "estimated_round_trip_cost_bps":config.estimated_round_trip_cost_bps,
                   "risk_fraction_cap":config.risk_fraction, "max_notional_fraction":config.max_notional_fraction,
                   "estimated_net_reward_risk": (
                       (target - price - (config.estimated_round_trip_cost_bps / 10_000) * price)
                       / (price - stop + (config.estimated_round_trip_cost_bps / 10_000) * price)
                       if stop is not None and target is not None and stop < price else None
                   )}
    action = "buy_candidate" if strategy != "none" and not blockers else "no_trade"
    return AdaptiveDecision(action, regime, strategy, confidence,
                            price if action == "buy_candidate" else None,
                            stop if action == "buy_candidate" else None,
                            target if action == "buy_candidate" else None,
                            tuple(reasons), tuple(blockers), diagnostics)
