"""Deterministic regime-aware, research-only strategy portfolio for USDT perpetuals.

All signal features use completed 5m/15m candles. Candidate selections do NOT
authorize Demo orders; unlike execution evidence, they are merely hypotheses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from tradingagents.crypto.scalping import _indicators, _validate


@dataclass(frozen=True)
class StrategyFamily:
    id: str
    regimes: tuple[str, ...]
    description: str


STRATEGIES = (
    StrategyFamily("ema_pullback", ("trend_up", "trend_down"), "EMA continuation pullback"),
    StrategyFamily("donchian_breakout", ("trend_up", "trend_down"), "20-bar breakout"),
    StrategyFamily("macd_momentum", ("trend_up", "trend_down"), "MACD histogram momentum"),
    StrategyFamily("bollinger_rsi_reversion", ("range",), "Bollinger + RSI mean reversion"),
    StrategyFamily("vwap_reversion", ("range",), "VWAP reversion with RSI filter"),
    StrategyFamily("volatility_breakout", ("volatile",), "ATR-confirmed price breakout"),
)
NOT_IMPLEMENTED = {
    "funding_basis_arbitrage": "Requires synchronized funding/basis and hedge execution",
    "cointegration_pairs": "Requires synchronized cross-symbol data and two-leg fills",
    "orderbook_market_making": "Requires recorded order book/queue and inventory models",
    "cross_sectional_momentum": "Requires portfolio ranking and shared position accounting",
}

# Research-screen assumptions, not Bybit's verified fee schedule.
STRESS_ROUND_TRIP_COST = 0.003
MIN_NET_TARGET_FRACTION = 0.001
MAX_MODELED_STOP_FRACTION = 0.015


def _closed(frame: pd.DataFrame, label: str, min_bars: int, minutes: int, now: pd.Timestamp) -> pd.DataFrame:
    candles = _validate(frame, label, min_bars)
    if candles.index[-1] + pd.Timedelta(minutes=minutes) > now:
        raise ValueError(f"{label} contains an unfinished candle")
    if now >= candles.index[-1] + pd.Timedelta(minutes=2 * minutes):
        raise ValueError(f"{label} is stale")
    if not candles.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=minutes)).all():
        raise ValueError(f"{label} has missing candles")
    return candles


def _regime(fifteen: pd.DataFrame) -> tuple[str, dict]:
    data = _indicators(fifteen, 14)
    close = float(data["close"].iloc[-1])
    atr = float(data["atr"].iloc[-1])
    median = float(data["atr"].dropna().iloc[-35:].median())
    if not all(math.isfinite(v) and v > 0 for v in (close, atr, median)):
        raise ValueError("incomplete regime indicators")
    recent = data["close"].iloc[-21:]
    distance = abs(float(recent.iloc[-1] - recent.iloc[0]))
    path = float(recent.diff().abs().sum())
    efficiency = distance / path if path else 0.0
    fraction = atr / close
    expansion = atr / median
    ema20 = float(data["ema20"].iloc[-1])
    ema50 = float(data["ema50"].iloc[-1])
    if fraction >= 0.025:
        regime = "extreme_risk"
    elif fraction >= 0.012 or expansion >= 1.8:
        regime = "volatile"
    elif efficiency >= 0.33 and close > ema20 > ema50:
        regime = "trend_up"
    elif efficiency >= 0.33 and close < ema20 < ema50:
        regime = "trend_down"
    elif efficiency <= 0.25 and expansion <= 1.5:
        regime = "range"
    else:
        regime = "transition"
    return regime, {
        "efficiency_20": round(efficiency, 5),
        "atr_fraction_15m": round(fraction, 6),
        "atr_expansion": round(expansion, 4),
        "ema20": ema20,
        "ema50": ema50,
    }


def _ema_pullback(five: pd.DataFrame, trend: int) -> int:
    if not trend:
        return 0
    data = _indicators(five, 14)
    last, previous = data.iloc[-1], data.iloc[-2]
    rsi = float(last["rsi"])
    if trend == 1 and previous["close"] <= previous["ema9"] and last["close"] > last["ema9"] and 43 <= rsi <= 72:
        return 1
    if trend == -1 and previous["close"] >= previous["ema9"] and last["close"] < last["ema9"] and 28 <= rsi <= 57:
        return -1
    return 0


def _donchian(fifteen: pd.DataFrame, trend: int) -> int:
    last = float(fifteen["close"].iloc[-1])
    if trend == 1 and last > float(fifteen["high"].iloc[-21:-1].max()):
        return 1
    if trend == -1 and last < float(fifteen["low"].iloc[-21:-1].min()):
        return -1
    return 0


def _macd(fifteen: pd.DataFrame, trend: int) -> int:
    close = fifteen["close"]
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    histogram = line - line.ewm(span=9, adjust=False).mean()
    before, after = float(histogram.iloc[-2]), float(histogram.iloc[-1])
    if trend == 1 and after > 0 >= before:
        return 1
    if trend == -1 and after < 0 <= before:
        return -1
    return 0


def _bollinger(five: pd.DataFrame) -> int:
    data = _indicators(five, 14)
    values = data["close"].iloc[-20:]
    average = float(values.mean())
    sigma = float(values.std(ddof=0))
    last, rsi = float(values.iloc[-1]), float(data["rsi"].iloc[-1])
    if sigma <= 0:
        return 0
    if last < average - 2 * sigma and rsi <= 35:
        return 1
    if last > average + 2 * sigma and rsi >= 65:
        return -1
    return 0


def _vwap(five: pd.DataFrame) -> int:
    if "volume" not in five:
        return 0
    recent = five.iloc[-20:]
    size = float(recent["volume"].sum())
    if size <= 0:
        return 0
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3
    vwap = float((typical * recent["volume"]).sum() / size)
    data = _indicators(five, 14).iloc[-1]
    close, atr, rsi = float(data["close"]), float(data["atr"]), float(data["rsi"])
    if close < vwap - 1.3 * atr and rsi < 38:
        return 1
    if close > vwap + 1.3 * atr and rsi > 62:
        return -1
    return 0


def _volatility(fifteen: pd.DataFrame) -> int:
    features = _indicators(fifteen, 14)
    atr = float(features["atr"].iloc[-1])
    historical = float(features["atr"].iloc[-22:-2].median())
    if historical <= 0 or atr < 1.15 * historical:
        return 0
    close = float(fifteen["close"].iloc[-1])
    if close > float(fifteen["high"].iloc[-13:-1].max()):
        return 1
    if close < float(fifteen["low"].iloc[-13:-1].min()):
        return -1
    return 0


def evaluate_regime_menu(
    five_minute: pd.DataFrame,
    fifteen_minute: pd.DataFrame,
    *,
    now_ms: int,
    reference_price: float,
) -> dict:
    """Choose a research candidate by market regime, never place an order.

    Missing/stale/unfinished candles fail closed. Signal/stop/target are
    hypotheses; historical validation and a dedicated execution integration
    are intentionally absent.
    """
    if type(now_ms) is not int or now_ms <= 0:
        raise ValueError("now_ms must be a positive integer")
    if type(reference_price) not in (int, float) or not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("reference_price must be finite and positive")
    now = pd.Timestamp(datetime.fromtimestamp(now_ms / 1000, UTC))
    five = _closed(five_minute, "five_minute", 65, 5, now)
    fifteen = _closed(fifteen_minute, "fifteen_minute", 80, 15, now)
    regime, features = _regime(fifteen)
    trend = 1 if regime == "trend_up" else -1 if regime == "trend_down" else 0
    generators = {
        "ema_pullback": lambda: _ema_pullback(five, trend),
        "donchian_breakout": lambda: _donchian(fifteen, trend),
        "macd_momentum": lambda: _macd(fifteen, trend),
        "bollinger_rsi_reversion": lambda: _bollinger(five),
        "vwap_reversion": lambda: _vwap(five),
        "volatility_breakout": lambda: _volatility(fifteen),
    }
    candidates = [
        {
            "strategy": spec.id, "regime_compatible": regime in spec.regimes,
            "signal": int(generators[spec.id]()) if regime in spec.regimes else 0,
        }
        for spec in STRATEGIES
    ]
    output = {
        "regime": regime, "features": features,
        "timestamp_5m_open": five.index[-1].isoformat(),
        "timestamp_15m_open": fifteen.index[-1].isoformat(),
        "candidates": candidates, "selected_strategy": None, "selected_signal": 0,
        "stop_price_model": None, "target_price_model": None,
        "selection_reason": "no_candidate_signal",
        "shadow_only": True, "execution_authorized": False,
        "unimplemented_families": NOT_IMPLEMENTED,
    }
    if regime in ("extreme_risk", "transition"):
        output["selection_reason"] = "unsafe_or_unclassified_regime"
        return output
    triggered = [item for item in candidates if item["signal"]]
    if not triggered:
        return output
    if len({item["signal"] for item in triggered}) != 1:
        output["selection_reason"] = "conflicting_strategy_signals"
        return output
    priorities = {
        "trend_up": ("donchian_breakout", "ema_pullback", "macd_momentum"),
        "trend_down": ("donchian_breakout", "ema_pullback", "macd_momentum"),
        "range": ("bollinger_rsi_reversion", "vwap_reversion"),
        "volatile": ("volatility_breakout",),
    }
    choice = min(triggered, key=lambda item: priorities[regime].index(item["strategy"]))
    direction = choice["signal"]
    short_horizon = choice["strategy"] in ("ema_pullback", "bollinger_rsi_reversion", "vwap_reversion")
    atr = float(_indicators(five if short_horizon else fifteen, 14)["atr"].iloc[-1])
    distance = max(1.25 * atr, reference_price * 0.0015)
    fraction = distance / reference_price
    if fraction > MAX_MODELED_STOP_FRACTION:
        output["selection_reason"] = "modeled_stop_too_wide"
        return output
    if fraction * 2 <= STRESS_ROUND_TRIP_COST + MIN_NET_TARGET_FRACTION:
        output["selection_reason"] = "modeled_target_does_not_cover_costs"
        return output
    output.update({
        "selected_strategy": choice["strategy"], "selected_signal": direction,
        "stop_price_model": reference_price - direction * distance,
        "target_price_model": reference_price + direction * distance * 2,
        "selection_reason": "research_candidate_only_unvalidated_performance",
    })
    return output
