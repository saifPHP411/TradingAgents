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
    mean_reversion_period: int = 20
    vwap_period: int = 24
    bollinger_std: float = 2.0
    adx_trend_min: float = 20.0
    adx_range_max: float = 17.0
    efficiency_trend_min: float = 0.30
    efficiency_range_max: float = 0.25
    relative_volume_min: float = 1.30
    max_atr_pct: float = 0.08
    max_spread_bps: float = 20.0
    estimated_round_trip_cost_bps: float = 30.0
    min_net_reward_risk: float = 1.25
    min_edge_cost_multiple: float = 3.0
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10

    def __post_init__(self):
        for name in ("fast_ema", "slow_ema", "atr_period", "lookback", "breakout_lookback", "mean_reversion_period", "vwap_period"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        numeric_limits = (
            self.adx_trend_min, self.adx_range_max, self.efficiency_trend_min,
            self.efficiency_range_max, self.relative_volume_min, self.max_atr_pct,
            self.bollinger_std,
            self.max_spread_bps, self.estimated_round_trip_cost_bps,
            self.min_net_reward_risk, self.min_edge_cost_multiple, self.risk_fraction, self.max_notional_fraction,
        )
        if any(not pd.notna(value) or abs(value) == float("inf") for value in numeric_limits):
            raise ValueError("configuration values must be finite")
        if not 0 < self.risk_fraction <= 0.02 or not 0 < self.max_notional_fraction <= 1:
            raise ValueError("risk and notional fractions exceed research safety bounds")
        if min(self.max_spread_bps, self.estimated_round_trip_cost_bps) < 0:
            raise ValueError("cost limits cannot be negative")
        if self.max_atr_pct <= 0 or self.relative_volume_min <= 0 or self.bollinger_std <= 0 or self.min_net_reward_risk <= 0 or self.min_edge_cost_multiple < 1:
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
    directional_total = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / directional_total.where(directional_total > 0)
    x["adx"] = dx.ewm(alpha=1/cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    net = x.close.diff(cfg.lookback).abs()
    path = x.close.diff().abs().rolling(cfg.lookback, min_periods=cfg.lookback).sum()
    x["efficiency"] = net / path.where(path > 0)
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
    band_mid = x.close.rolling(cfg.mean_reversion_period, min_periods=cfg.mean_reversion_period).mean()
    band_std = x.close.rolling(cfg.mean_reversion_period, min_periods=cfg.mean_reversion_period).std(ddof=0)
    x["bb_lower"] = band_mid - cfg.bollinger_std * band_std
    typical_price = (x.high + x.low + x.close) / 3
    rolling_volume = x.volume.rolling(cfg.vwap_period, min_periods=cfg.vwap_period).sum()
    x["rolling_vwap"] = (typical_price * x.volume).rolling(
        cfg.vwap_period, min_periods=cfg.vwap_period
    ).sum() / rolling_volume.where(rolling_volume > 0)
    x["atr_base"] = x.atr_pct.rolling(cfg.lookback, min_periods=cfg.lookback).median().shift(1)
    return x

def _analyze_market_at(
    data: pd.DataFrame,
    features: pd.DataFrame,
    index: int,
    *,
    config: AdaptiveConfig,
    spread_bps: float | None,
    data_is_fresh: bool,
    closed_bars_only: bool,
) -> AdaptiveDecision:
    """Analyze one bar from causal, precomputed features without recomputing history."""
    history_count = index + 1
    blockers = []
    if not data_is_fresh:
        blockers.append("market data is stale")
    if not closed_bars_only:
        blockers.append("forming bars cannot drive decisions")
    if spread_bps is None or not pd.notna(spread_bps):
        blockers.append("spread is unknown")
    elif spread_bps < 0 or spread_bps > config.max_spread_bps:
        blockers.append("spread exceeds liquidity limit")
    needed = max(config.slow_ema, config.lookback, config.breakout_lookback, config.atr_period) + 5
    if history_count < needed:
        blockers.append("insufficient closed-bar history")
    if blockers:
        return AdaptiveDecision(
            "no_trade", "uncertain", "none", 0, None, None, None,
            (), tuple(blockers), {"bars": history_count},
        )

    row = features.iloc[index]
    required = (
        "atr", "atr_pct", "ema_fast", "ema_slow", "adx", "efficiency", "rsi",
        "prior_high", "prior_low", "volume_base", "atr_base", "bb_lower", "rolling_vwap",
    )
    if any(pd.isna(row[key]) for key in required):
        return AdaptiveDecision(
            "no_trade", "uncertain", "none", 0, None, None, None,
            (), ("indicator warm-up incomplete",), {"bars": history_count},
        )

    price, atr = float(row.close), float(row.atr)
    bullish = row.ema_fast > row.ema_slow and price > row.ema_slow
    bearish = row.ema_fast < row.ema_slow and price < row.ema_slow
    expanding = row.atr_base > 0 and row.atr_pct > row.atr_base * 2
    if row.atr_pct > config.max_atr_pct or expanding:
        regime: Regime = "high_volatility"
    elif bullish and row.adx >= config.adx_trend_min and row.efficiency >= config.efficiency_trend_min:
        regime = "bull_trend"
    elif bearish and row.adx >= config.adx_trend_min and row.efficiency >= config.efficiency_trend_min:
        regime = "bear_trend"
    elif row.adx <= config.adx_range_max and row.efficiency <= config.efficiency_range_max:
        regime = "range"
    else:
        regime = "uncertain"

    strategy: Strategy = "none"
    confidence, reasons = 0.0, []
    if regime == "bull_trend":
        previous_close = float(features.close.iloc[index - 1])
        previous_fast_ema = float(features.ema_fast.iloc[index - 1])
        if price > row.ema_fast and previous_close <= previous_fast_ema:
            strategy = "trend_pullback"
            confidence = min(
                .90,
                .55 + (float(row.adx) - config.adx_trend_min) / 100
                + float(row.efficiency) * .25,
            )
            reasons += ["bullish EMA structure", "completed bar reclaimed fast EMA after pullback"]
        elif price > row.prior_high and row.volume >= row.volume_base * config.relative_volume_min:
            strategy = "breakout"
            confidence = min(.90, .58 + float(row.efficiency) * .25)
            reasons += ["close broke prior-bar range high", "relative volume confirms breakout"]
    elif regime == "range":
        width = float(row.prior_high - row.prior_low)
        lower_range_touch = width > 0 and price <= row.prior_low + width * .15
        vwap_band_reversion = price <= row.bb_lower and price < row.rolling_vwap and row.rsi <= 35
        if width > 0 and row.rsi <= 35 and (lower_range_touch or vwap_band_reversion):
            strategy, confidence = "range_reversion", .62
            reasons += ["low ADX and path efficiency indicate range"]
            if vwap_band_reversion:
                reasons += ["price below lower Bollinger band", "price below rolling volume-weighted average"]
            else:
                reasons += ["oversold near lower range boundary"]
        elif width > 0 and row.rsi >= 70 and price >= row.prior_high - width * .15:
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
            midpoint = (float(row.prior_high) + float(row.prior_low)) / 2
            target = min(midpoint, float(row.rolling_vwap))
        else:
            stop = price - 2 * atr
            target = price + 4 * atr
        risk_distance = price - stop
        reward_distance = target - price
        cost_distance = (config.estimated_round_trip_cost_bps / 10_000) * price
        net_reward_risk = (reward_distance - cost_distance) / (risk_distance + cost_distance)
        if (
            stop <= 0
            or reward_distance <= 0
            or net_reward_risk < config.min_net_reward_risk
            or reward_distance < cost_distance * config.min_edge_cost_multiple
        ):
            blockers.append("estimated round-trip costs erase minimum net reward/risk")
            strategy, confidence, stop, target = "none", 0.0, None, None
        else:
            reasons.append("reference risk geometry passes estimated-cost gate")

    diagnostics = {
        "bar_time": str(data.index[index]),
        "bars": history_count,
        "price": price,
        "atr": atr,
        "atr_pct": float(row.atr_pct),
        "adx": float(row.adx),
        "efficiency_ratio": float(row.efficiency),
        "rsi": float(row.rsi),
        "ema_fast": float(row.ema_fast),
        "ema_slow": float(row.ema_slow),
        "prior_range_high": float(row.prior_high),
        "prior_range_low": float(row.prior_low),
        "bollinger_lower": float(row.bb_lower),
        "rolling_vwap": float(row.rolling_vwap),
        "relative_volume": float(row.volume / row.volume_base) if row.volume_base > 0 else None,
        "spread_bps": spread_bps,
        "estimated_round_trip_cost_bps": config.estimated_round_trip_cost_bps,
        "minimum_expected_edge_cost_multiple": config.min_edge_cost_multiple,
        "risk_fraction_cap": config.risk_fraction,
        "max_notional_fraction": config.max_notional_fraction,
        "estimated_net_reward_risk": (
            (target - price - (config.estimated_round_trip_cost_bps / 10_000) * price)
            / (price - stop + (config.estimated_round_trip_cost_bps / 10_000) * price)
            if stop is not None and target is not None and stop < price else None
        ),
    }
    action = "buy_candidate" if strategy != "none" and not blockers else "no_trade"
    return AdaptiveDecision(
        action,
        regime,
        strategy,
        confidence,
        price if action == "buy_candidate" else None,
        stop if action == "buy_candidate" else None,
        target if action == "buy_candidate" else None,
        tuple(reasons),
        tuple(blockers),
        diagnostics,
    )


def analyze_market(
    frame: pd.DataFrame,
    *,
    config: AdaptiveConfig = AdaptiveConfig(),
    spread_bps: float | None = None,
    data_is_fresh: bool = True,
    closed_bars_only: bool = True,
) -> AdaptiveDecision:
    """Return a deterministic analysis of the latest completed bar, never an order."""
    data = _validate(frame)
    if data.empty:
        features = pd.DataFrame(index=data.index)
        index = -1
    else:
        features = _features(data, config)
        index = len(data) - 1
    return _analyze_market_at(
        data,
        features,
        index,
        config=config,
        spread_bps=spread_bps,
        data_is_fresh=data_is_fresh,
        closed_bars_only=closed_bars_only,
    )
