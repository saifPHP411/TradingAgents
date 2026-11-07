"""Market-structure primitives, the vocabulary TJR is written in.

Every function here reads a frame that has already been sliced to what the caller
could see (see :mod:`tradingagents.tjr.frames`), so none of them can look ahead.
They are deliberately small and testable one at a time, because the strategy is a
*sequence* of these and a wrong primitive would make the whole chain wrong while
still looking plausible.

The definitions, stated so they can be argued with:

- **Swing high/low** — a fractal: the bar whose high (low) is the extreme of a
  window of ``lookback`` bars on each side. It is only *confirmed* ``lookback``
  bars later, and the functions return it with the index at which it became
  knowable, never earlier.
- **Break of structure (BOS)** — a *close* beyond the most recent confirmed swing
  in the trade's direction. A wick through a swing is not a break; TJR is explicit
  that a closure is what matters.
- **Fair value gap (FVG)** — a three-candle imbalance: a bullish gap where the
  current low sits above the high two bars back, a bearish gap the mirror. The gap
  is the price band between them.
- **Inverse FVG (IFVG)** — a gap that price has since closed back through, so its
  polarity has flipped: a bullish gap closed below becomes bearish resistance, and
  a bearish gap closed above becomes bullish support.
- **Equilibrium** — the 50% midpoint of a leg, the level TJR names as the
  continuation confluence alongside the FVG.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_EVAL_CACHE: dict[tuple, tuple] = {}
"""Per-evaluation memo of the pure frame primitives.

The rule chain calls :func:`swing_points` and :func:`fair_value_gaps` once per
candidate sweep at each instant, over the *same* frame object, so a session
re-derives identical structure many times. The memo is keyed on the frame's
identity and holds a reference to the frame itself, so an entry can only be reused
for the very frame it was computed from (a recycled ``id`` cannot produce a stale
hit, because the frame is kept alive while cached). It is cleared at the start and
end of every :func:`~tradingagents.tjr.rules.evaluate`, so nothing is carried
between evaluations and no look-ahead can leak through it.
"""


def reset_frame_cache() -> None:
    """Drop the per-evaluation primitive cache. Called around each evaluation."""
    _EVAL_CACHE.clear()


def _cache_get(frame: pd.DataFrame, label: str, extra: tuple):
    entry = _EVAL_CACHE.get((id(frame), label, extra))
    if entry is not None and entry[0] is frame:
        return entry[1]
    return None


def _cache_put(frame: pd.DataFrame, label: str, extra: tuple, value) -> None:
    _EVAL_CACHE[(id(frame), label, extra)] = (frame, value)


@dataclass(frozen=True)
class Swing:
    """A confirmed swing point.

    ``index`` is the bar the swing *is*; ``confirmed_at`` is the bar after which it
    could be known (``index + lookback``). A caller that only wants points it could
    actually have seen filters on ``confirmed_at``.
    """

    index: pd.Timestamp
    price: float
    kind: str  # "high" or "low"
    confirmed_at: pd.Timestamp


@dataclass(frozen=True)
class Gap:
    """A fair value gap: the price band between two candles with an imbalance."""

    index: pd.Timestamp
    kind: str  # "bull" or "bear"
    low: float
    high: float

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


def swing_points(frame: pd.DataFrame, *, lookback: int = 2) -> list[Swing]:
    """Confirmed swing highs and lows in a frame, oldest first.

    A swing high at position ``i`` needs ``i`` to be the strict-or-equal extreme of
    ``[i-lookback, i+lookback]`` and strictly greater than at least one neighbour,
    so a flat run does not manufacture a swing. The swing is stamped as confirmed
    at ``i + lookback``.

    Computed with a centered rolling window rather than a Python loop: the result is
    identical (the edges the loop skipped are the positions a centered window leaves
    ``NaN``), and it is the hot path when a long 24/7 session is re-evaluated at
    every bar.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    cached = _cache_get(frame, "swing_points", (lookback,))
    if cached is not None:
        return cached
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    index = frame.index
    size = len(frame)
    if size <= 2 * lookback:
        return []
    window = 2 * lookback + 1
    high_max = pd.Series(highs).rolling(window, center=True).max().to_numpy()
    high_min = pd.Series(highs).rolling(window, center=True).min().to_numpy()
    low_min = pd.Series(lows).rolling(window, center=True).min().to_numpy()
    low_max = pd.Series(lows).rolling(window, center=True).max().to_numpy()
    # NaN at the edges makes these comparisons False, which is exactly the range
    # the original loop skipped.
    high_cond = (highs >= high_max) & (high_min < highs)
    low_cond = (lows <= low_min) & (low_max > lows)
    swings: list[Swing] = []
    for i in range(lookback, size - lookback):
        if high_cond[i]:
            swings.append(
                Swing(
                    index=index[i],
                    price=float(highs[i]),
                    kind="high",
                    confirmed_at=index[i + lookback],
                )
            )
        if low_cond[i]:
            swings.append(
                Swing(
                    index=index[i],
                    price=float(lows[i]),
                    kind="low",
                    confirmed_at=index[i + lookback],
                )
            )
    result = sorted(swings, key=lambda s: (s.index, s.kind))
    _cache_put(frame, "swing_points", (lookback,), result)
    return result


def last_swing(swings: list[Swing], kind: str, *, at: pd.Timestamp) -> Swing | None:
    """The most recent swing of ``kind`` confirmed at or before ``at``, or ``None``."""
    candidates = [s for s in swings if s.kind == kind and s.confirmed_at <= at]
    return candidates[-1] if candidates else None


def fair_value_gaps(frame: pd.DataFrame, *, kind: str | None = None) -> list[Gap]:
    """Three-candle fair value gaps in a frame, oldest first.

    A bullish gap exists when ``low[i] > high[i-2]``; the band is
    ``(high[i-2], low[i])``. A bearish gap is ``high[i] < low[i-2]``; the band is
    ``(high[i], low[i-2])``. ``kind`` filters to ``"bull"`` or ``"bear"``.
    """
    cached = _cache_get(frame, "fair_value_gaps", (kind,))
    if cached is not None:
        return cached
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    index = frame.index
    gaps: list[Gap] = []
    for i in range(2, len(frame)):
        if lows[i] > highs[i - 2]:
            gaps.append(
                Gap(index=index[i], kind="bull", low=float(highs[i - 2]), high=float(lows[i]))
            )
        if highs[i] < lows[i - 2]:
            gaps.append(
                Gap(index=index[i], kind="bear", low=float(highs[i]), high=float(lows[i - 2]))
            )
    if kind is not None:
        gaps = [g for g in gaps if g.kind == kind]
    _cache_put(frame, "fair_value_gaps", (kind,), gaps)
    return gaps


def break_of_structure(
    frame: pd.DataFrame,
    swings: list[Swing],
    *,
    direction: str,
    since: pd.Timestamp | None = None,
    swing_confirmed_before: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """The first bar at or after ``since`` that *closed* through a confirmed swing.

    ``direction`` is ``"up"`` (a close above the most recent confirmed swing high)
    or ``"down"`` (a close below the most recent confirmed swing low). Returns the
    bar's timestamp, or ``None`` when no such close happened in the frame.

    ``swing_confirmed_before`` restricts the swing that may be broken to one
    confirmed strictly before that instant. The reversal confirmation uses it to
    require the broken structure to *predate the sweep*: without it, a micro-swing
    formed by the sweep bar itself is broken by the very next close, and the
    "confirmation" is just the sweep's own bounce rather than a break of the range
    that was swept from.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    kind = "high" if direction == "up" else "low"
    # Filter the *set* of breakable swings once. Selecting the most recent swing per
    # bar and then discarding it would drop the older structure that is still the
    # relevant one — the very bug that let a sweep's own micro-swing count.
    eligible = [
        swing
        for swing in swings
        if swing.kind == kind
        and (swing_confirmed_before is None or swing.confirmed_at < swing_confirmed_before)
    ]
    if not eligible:
        return None
    for stamp, close in frame["close"].items():
        if since is not None and stamp < since:
            continue
        confirmed = [swing for swing in eligible if swing.confirmed_at <= stamp]
        if not confirmed:
            continue
        swing = max(confirmed, key=lambda item: item.index)
        if direction == "up" and close > swing.price:
            return stamp
        if direction == "down" and close < swing.price:
            return stamp
    return None


@dataclass(frozen=True)
class InverseGap:
    """A gap whose polarity has flipped, and when the flip was confirmed.

    ``gap`` is the original band (its edge is the level); ``flipped_at`` is the bar
    whose close went back through it, which is the instant the inverse gap became
    usable. The formation index is *not* the confirmation instant — using it would
    let a setup confirm on a bar from before the sweep.
    """

    gap: Gap
    flipped_at: pd.Timestamp

    @property
    def index(self) -> pd.Timestamp:
        return self.gap.index

    @property
    def kind(self) -> str:
        return self.gap.kind

    @property
    def low(self) -> float:
        return self.gap.low

    @property
    def high(self) -> float:
        return self.gap.high


def inverse_fair_value_gap(
    frame: pd.DataFrame,
    *,
    direction: str,
    since: pd.Timestamp | None = None,
) -> InverseGap | None:
    """The first gap whose polarity has flipped in ``direction``, or ``None``.

    ``direction="up"`` looks for a *bearish* gap that a later bar closed back above
    (support); ``direction="down"`` looks for a *bullish* gap a later bar closed
    below (resistance). ``flipped_at`` is the bar that closed through, which is when
    the inverse gap is knowable.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    source = "bear" if direction == "up" else "bull"
    gaps = fair_value_gaps(frame, kind=source)
    closes = frame["close"]
    for gap in gaps:
        later = closes[closes.index > gap.index]
        if direction == "up":
            hits = later[later > gap.high]
            if len(hits):
                flip = hits.index[0]
                if since is None or flip >= since:
                    return InverseGap(gap=gap, flipped_at=flip)
        else:
            hits = later[later < gap.low]
            if len(hits):
                flip = hits.index[0]
                if since is None or flip >= since:
                    return InverseGap(gap=gap, flipped_at=flip)
    return None


def equilibrium(leg_low: float, leg_high: float) -> float:
    """The 50% midpoint of a leg."""
    if leg_high < leg_low:
        raise ValueError(f"leg_high {leg_high} is below leg_low {leg_low}")
    return (leg_low + leg_high) / 2.0
