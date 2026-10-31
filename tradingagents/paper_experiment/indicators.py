"""Deterministic technical-indicator context for the paper-inference request.

The paper experiment makes one bounded model call per decision, and that call is
only as good as the evidence placed in it. A bare OHLCV table asks the model to
re-derive trend, momentum and volatility itself, and a single arithmetic slip
there is indistinguishable from a genuine opinion. This module computes the
standard indicators in code — identically for every decision — and renders them
next to the bars, so the model reasons over stated evidence instead of
recomputing it.

Two properties are load-bearing and are both tested:

- **Point-in-time only.** Every value is a function of the window the caller
  passes in, which is already bounded at the decision session. No later bar can
  reach a value.
- **No fabrication.** When the window is too short for an indicator the value is
  ``None`` and the rendered line says so, naming the bar count it needed and the
  bar count it had. It is never zero-filled or extrapolated.

Only pandas is used, so the paper path stays hermetic and a decision's inputs
remain reproducible offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLL_PERIOD, BOLL_STD = 20, 2.0
ATR_PERIOD = 14
SMA_SHORT, SMA_LONG, SMA_TREND = 20, 50, 200
EMA_SHORT, EMA_LONG = 10, 20
VOLUME_WINDOW = 20
RANGE_WINDOW = 20
RETURN_WINDOWS = (1, 5, 20)
TRADING_DAYS = 252

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class IndicatorError(ValueError):
    """The window cannot carry an indicator: a required value is missing or not finite."""


@dataclass(frozen=True)
class IndicatorSet:
    """The indicators computed from one point-in-time window.

    Every field is ``None`` when the window is too short to compute it, so a
    caller can tell "not available" from a real number and never reads a
    fabricated zero.
    """

    bars: int
    last_close: float | None
    sma_short: float | None
    sma_long: float | None
    sma_trend: float | None
    ema_short: float | None
    ema_long: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    boll_mid: float | None
    boll_upper: float | None
    boll_lower: float | None
    boll_percent_b: float | None
    atr: float | None
    atr_pct: float | None
    volume_ratio: float | None
    realized_vol: float | None
    return_1d: float | None
    return_5d: float | None
    return_20d: float | None
    range_position: float | None


def _series(window, column: str) -> pd.Series:
    try:
        series = window[column].astype(float)
    except (KeyError, TypeError, ValueError) as exc:
        raise IndicatorError(f"the price window has no usable {column!r} value: {exc}") from exc
    if not series.map(math.isfinite).all():
        raise IndicatorError(
            f"the price window carries a non-finite {column!r} value; an indicator "
            "cannot be computed from it"
        )
    return series


def _sma(close: pd.Series, period: int) -> float | None:
    if len(close) < period:
        return None
    value = float(close.rolling(period).mean().iloc[-1])
    return value if math.isfinite(value) else None


def _ema(close: pd.Series, period: int) -> float | None:
    if len(close) < period:
        return None
    value = float(close.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1])
    return value if math.isfinite(value) else None


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> float | None:
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = float(gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1])
    avg_loss = float(loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1])
    if not math.isfinite(avg_gain) or not math.isfinite(avg_loss):
        return None
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(close) < MACD_SLOW + MACD_SIGNAL:
        return None, None, None
    fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd = fast - slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    line = float(macd.iloc[-1])
    sig = float(signal.iloc[-1])
    if not (math.isfinite(line) and math.isfinite(sig)):
        return None, None, None
    return line, sig, line - sig


def _bollinger(close: pd.Series) -> tuple[float | None, float | None, float | None, float | None]:
    if len(close) < BOLL_PERIOD:
        return None, None, None, None
    mid = float(close.rolling(BOLL_PERIOD).mean().iloc[-1])
    std = float(close.rolling(BOLL_PERIOD).std(ddof=0).iloc[-1])
    if not (math.isfinite(mid) and math.isfinite(std)):
        return None, None, None, None
    upper = mid + BOLL_STD * std
    lower = mid - BOLL_STD * std
    width = upper - lower
    percent_b = (float(close.iloc[-1]) - lower) / width if width > 0.0 else None
    return mid, upper, lower, percent_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> float | None:
    if len(close) <= period:
        return None
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    value = float(
        true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    )
    return value if math.isfinite(value) else None


def _volume_ratio(volume: pd.Series) -> float | None:
    if len(volume) < VOLUME_WINDOW:
        return None
    average = float(volume.rolling(VOLUME_WINDOW).mean().iloc[-1])
    if not math.isfinite(average) or average <= 0.0:
        return None
    return float(volume.iloc[-1]) / average


def _realized_vol(close: pd.Series, window: int = VOLUME_WINDOW) -> float | None:
    if len(close) <= window:
        return None
    returns = close.pct_change().dropna().tail(window)
    if len(returns) < 2:
        return None
    deviation = float(returns.std(ddof=1))
    if not math.isfinite(deviation):
        return None
    return deviation * math.sqrt(TRADING_DAYS)


def _pct_change(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    previous = float(close.iloc[-1 - periods])
    if not math.isfinite(previous) or previous == 0.0:
        return None
    return float(close.iloc[-1]) / previous - 1.0


def _range_position(high: pd.Series, low: pd.Series, close: pd.Series) -> float | None:
    if len(close) < RANGE_WINDOW:
        return None
    highest = float(high.tail(RANGE_WINDOW).max())
    lowest = float(low.tail(RANGE_WINDOW).min())
    if not (math.isfinite(highest) and math.isfinite(lowest)) or highest <= lowest:
        return None
    return (float(close.iloc[-1]) - lowest) / (highest - lowest)


def compute_indicators(window) -> IndicatorSet:
    """Compute the indicator set from ``window``, or raise on a malformed window.

    ``window`` is a point-in-time frame ending at the decision session. A
    non-finite or missing OHLCV value raises :class:`IndicatorError` so the
    caller fails closed; a window that is merely too short yields ``None`` fields
    rather than a fabricated number.
    """
    close = _series(window, "close")
    high = _series(window, "high")
    low = _series(window, "low")
    volume = _series(window, "volume")

    last_close = float(close.iloc[-1])
    macd, macd_signal, macd_hist = _macd(close)
    boll_mid, boll_upper, boll_lower, boll_percent_b = _bollinger(close)
    atr = _atr(high, low, close)

    return IndicatorSet(
        bars=len(window),
        last_close=last_close,
        sma_short=_sma(close, SMA_SHORT),
        sma_long=_sma(close, SMA_LONG),
        sma_trend=_sma(close, SMA_TREND),
        ema_short=_ema(close, EMA_SHORT),
        ema_long=_ema(close, EMA_LONG),
        rsi=_rsi(close),
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        boll_mid=boll_mid,
        boll_upper=boll_upper,
        boll_lower=boll_lower,
        boll_percent_b=boll_percent_b,
        atr=atr,
        atr_pct=(atr / last_close) if (atr is not None and last_close > 0.0) else None,
        volume_ratio=_volume_ratio(volume),
        realized_vol=_realized_vol(close),
        return_1d=_pct_change(close, 1),
        return_5d=_pct_change(close, 5),
        return_20d=_pct_change(close, 20),
        range_position=_range_position(high, low, close),
    )


def _number(label: str, value: float | None, *, need: int, bars: int, spec: str = "{:,.2f}") -> str:
    if value is None:
        return f"  {label:<24} insufficient history (need {need} bars, have {bars})"
    return f"  {label:<24} {spec.format(value)}"


def _relative(label: str, value: float | None, *, need: int, bars: int) -> str:
    if value is None:
        return f"  {label:<24} insufficient history (need {need} bars, have {bars})"
    if value > 0.0:
        relation = "above"
    elif value < 0.0:
        relation = "below"
    else:
        relation = "at"
    return f"  {label:<24} {relation}"


def _rsi_state(rsi: float | None) -> str:
    if rsi is None:
        return "insufficient history"
    if rsi >= 70.0:
        return "overbought"
    if rsi <= 30.0:
        return "oversold"
    return "neutral"


def _trend(values: IndicatorSet) -> str:
    if values.sma_short is None or values.sma_long is None:
        return "insufficient history for a trend"
    if values.sma_short > values.sma_long:
        direction = "up"
    elif values.sma_short < values.sma_long:
        direction = "down"
    else:
        direction = "flat"
    return f"{direction} (SMA{SMA_SHORT} vs SMA{SMA_LONG})"


def render_indicator_context(window) -> str:
    """Render the indicator block for the prompt, oldest-to-newest window given.

    A window that is absent or empty renders a stated absence rather than raising:
    the caller decides whether to ask at all. A window that is present but
    malformed raises :class:`IndicatorError`.
    """
    if window is None or len(window) == 0:
        return "No indicator context: no price history is available for this session."

    values = compute_indicators(window)
    bars = values.bars
    lines = [
        f"  bars in window           {bars}",
        _number("last close", values.last_close, need=1, bars=bars),
        _number("return 1d", values.return_1d, need=2, bars=bars, spec="{:+.2%}"),
        _number("return 5d", values.return_5d, need=6, bars=bars, spec="{:+.2%}"),
        _number("return 20d", values.return_20d, need=21, bars=bars, spec="{:+.2%}"),
        _number(f"SMA{SMA_SHORT}", values.sma_short, need=SMA_SHORT, bars=bars),
        _number(f"SMA{SMA_LONG}", values.sma_long, need=SMA_LONG, bars=bars),
        _number(f"SMA{SMA_TREND}", values.sma_trend, need=SMA_TREND, bars=bars),
        _number(f"EMA{EMA_SHORT}", values.ema_short, need=EMA_SHORT, bars=bars),
        _number(f"EMA{EMA_LONG}", values.ema_long, need=EMA_LONG, bars=bars),
        _relative("close vs SMA20", None if values.sma_short is None
                  else values.last_close - values.sma_short, need=SMA_SHORT, bars=bars),
        _relative("close vs SMA50", None if values.sma_long is None
                  else values.last_close - values.sma_long, need=SMA_LONG, bars=bars),
        f"  trend                    {_trend(values)}",
        _number(f"RSI({RSI_PERIOD})", values.rsi, need=RSI_PERIOD + 1, bars=bars)
        + f" ({_rsi_state(values.rsi)})",
        _number(
            f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})",
            values.macd,
            need=MACD_SLOW + MACD_SIGNAL,
            bars=bars,
        ),
        _number("MACD signal", values.macd_signal, need=MACD_SLOW + MACD_SIGNAL, bars=bars),
        _number("MACD histogram", values.macd_hist, need=MACD_SLOW + MACD_SIGNAL, bars=bars,
                spec="{:+.2f}"),
        _number(f"Bollinger({BOLL_PERIOD},{BOLL_STD:g}) mid", values.boll_mid,
                need=BOLL_PERIOD, bars=bars),
        _number("Bollinger upper", values.boll_upper, need=BOLL_PERIOD, bars=bars),
        _number("Bollinger lower", values.boll_lower, need=BOLL_PERIOD, bars=bars),
        _number("Bollinger %B", values.boll_percent_b, need=BOLL_PERIOD, bars=bars),
        _number(f"ATR({ATR_PERIOD})", values.atr, need=ATR_PERIOD + 1, bars=bars),
        _number("ATR % of close", values.atr_pct, need=ATR_PERIOD + 1, bars=bars,
                spec="{:.2%}"),
        _number(f"volume vs {VOLUME_WINDOW}d avg", values.volume_ratio,
                need=VOLUME_WINDOW, bars=bars, spec="{:.2f}x"),
        _number(f"realized vol({VOLUME_WINDOW}d ann)", values.realized_vol,
                need=VOLUME_WINDOW + 1, bars=bars, spec="{:.2%}"),
        _number(f"range position({RANGE_WINDOW}d)", values.range_position,
                need=RANGE_WINDOW, bars=bars),
    ]
    return "\n".join(lines)


__all__ = [
    "ATR_PERIOD",
    "BOLL_PERIOD",
    "IndicatorError",
    "IndicatorSet",
    "MACD_FAST",
    "MACD_SIGNAL",
    "MACD_SLOW",
    "RSI_PERIOD",
    "SMA_LONG",
    "SMA_SHORT",
    "SMA_TREND",
    "compute_indicators",
    "render_indicator_context",
]
