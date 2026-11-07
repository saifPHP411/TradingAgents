"""Liquidity levels — TJR's "draws on liquidity".

TJR's first step is to mark where price is *drawn*: the highs and lows where
resting orders sit. It names session highs/lows, 1-hour and 4-hour highs/lows, and
swing highs/lows, and it treats equal highs/lows as a single level when they line
up objectively.

Those same levels serve two jobs in the strategy, and both are needed:

- a **sweep** is price taking one of them out and failing to hold — the entry side;
- a **target** is the next level price is drawn to — the exit side.

A level is stamped with the instant it became *knowable*, so the builder can be
asked for "the levels visible at 10:00" without leaking a 10:30 high. The builder
takes frames already sliced to that instant, so the guard is at the call site and
this module stays a pure function of what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .frames import IntradaySeries
from .sessions import SessionPolicy
from .structure import Swing, swing_points


@dataclass(frozen=True)
class LiquidityLevel:
    """One draw on liquidity, with its provenance."""

    label: str
    kind: str  # "high" or "low"
    price: float
    formed_at: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "price": round(self.price, 6),
            "formed_at": self.formed_at.isoformat(),
        }


def _session_extreme(frame: pd.DataFrame, *, kind: str) -> float | None:
    if frame.empty:
        return None
    column = "high" if kind == "high" else "low"
    value = float(frame[column].max() if kind == "high" else frame[column].min())
    return value


def previous_session_levels(
    series: IntradaySeries, session: str, as_of: pd.Timestamp
) -> list[LiquidityLevel]:
    """The previous session's high and low, if that session is in the data."""
    prior_sessions = _sessions_before(series, session, as_of)
    if not prior_sessions:
        return []
    previous = prior_sessions[-1]
    bars = series.session_bars(previous, as_of)
    levels: list[LiquidityLevel] = []
    formed = series.session.session_close_utc(previous)
    for kind, label in (("high", "prev_session_high"), ("low", "prev_session_low")):
        value = _session_extreme(bars, kind=kind)
        if value is not None:
            levels.append(LiquidityLevel(label=label, kind=kind, price=value, formed_at=formed))
    return levels


def current_session_levels(
    series: IntradaySeries, session: str, as_of: pd.Timestamp
) -> list[LiquidityLevel]:
    """The current session's running high and low, through ``as_of``.

    Used for *targets* — where price is drawn next. For sweep detection the level
    must predate the bar that sweeps it, so :func:`running_extreme_before` is used
    there instead; a running extreme stamped "now" could never be swept by an
    earlier bar.
    """
    bars = series.session_bars(session, as_of)
    if bars.empty:
        return []
    last = bars.index[-1] + pd.Timedelta(minutes=series.bar_minutes)
    levels: list[LiquidityLevel] = []
    for kind, label in (("high", "session_high"), ("low", "session_low")):
        value = _session_extreme(bars, kind=kind)
        if value is not None:
            levels.append(LiquidityLevel(label=label, kind=kind, price=value, formed_at=last))
    return levels


def running_extreme_before(frame: pd.DataFrame, *, kind: str) -> pd.Series:
    """For each bar, the running high (low) of every *earlier* bar in the frame.

    This is the level a bar can sweep: the extreme as it stood *before* the bar,
    which excludes the bar's own print. The first bar's value is NaN (no prior bar),
    so it cannot sweep anything.
    """
    column = "high" if kind == "high" else "low"
    values = frame[column]
    if kind == "high":
        return values.shift(1).cummax()
    return values.shift(1).cummin()


def _sessions_before(series: IntradaySeries, session: str, as_of: pd.Timestamp) -> list[str]:
    frame = series.available(as_of)
    if frame.empty:
        return []
    days = frame.index.tz_convert(series.session.timezone).strftime("%Y-%m-%d")
    return sorted({d for d in days if d < session})


def completed_bar_levels(
    series: IntradaySeries, *, label_prefix: str, session: str, as_of: pd.Timestamp
) -> list[LiquidityLevel]:
    """The most recent *completed* higher-timeframe bar's high and low.

    For 1h/4h levels TJR looks at the last completed bar of that timeframe. A bar
    that is still forming has no close, so it is excluded; the level is the last bar
    whose own close is at or before ``as_of``.
    """
    frame = series.available(as_of)
    if frame.empty:
        return []
    last_start = frame.index[-1]
    bar_close = last_start + pd.Timedelta(minutes=series.bar_minutes)
    row = frame.loc[last_start]
    return [
        LiquidityLevel(
            label=f"{label_prefix}_high", kind="high", price=float(row["high"]), formed_at=bar_close
        ),
        LiquidityLevel(
            label=f"{label_prefix}_low", kind="low", price=float(row["low"]), formed_at=bar_close
        ),
    ]


def swing_levels(
    series: IntradaySeries, as_of: pd.Timestamp, *, lookback: int = 2
) -> list[LiquidityLevel]:
    """Confirmed swing highs and lows, as liquidity levels."""
    frame = series.available(as_of)
    if frame.empty:
        return []
    out: list[LiquidityLevel] = []
    for swing in swing_points(frame, lookback=lookback):
        label = "swing_high" if swing.kind == "high" else "swing_low"
        out.append(
            LiquidityLevel(
                label=label, kind=swing.kind, price=swing.price, formed_at=swing.confirmed_at
            )
        )
    return out


def equal_levels(swings: list[Swing], *, tolerance: float) -> list[LiquidityLevel]:
    """Equal highs / equal lows: swings whose prices line up within ``tolerance``.

    Objective by construction: two or more swings of the same kind whose prices sit
    within ``tolerance`` (a price fraction) form one level, priced at the extreme of
    the cluster. A single swing is not an "equal" level. ``tolerance`` is a fraction
    of price, so it scales across a cheap and an expensive symbol alike.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    out: list[LiquidityLevel] = []
    for kind, label in (("high", "equal_highs"), ("low", "equal_lows")):
        points = sorted([s for s in swings if s.kind == kind], key=lambda s: s.price)
        cluster: list[Swing] = []
        for point in points:
            if cluster and abs(point.price - cluster[-1].price) <= tolerance * cluster[-1].price:
                cluster.append(point)
            else:
                if len(cluster) >= 2:
                    out.append(_cluster_level(cluster, kind, label))
                cluster = [point]
        if len(cluster) >= 2:
            out.append(_cluster_level(cluster, kind, label))
    return out


def _cluster_level(cluster: list[Swing], kind: str, label: str) -> LiquidityLevel:
    price = max(s.price for s in cluster) if kind == "high" else min(s.price for s in cluster)
    formed = max(s.confirmed_at for s in cluster)
    return LiquidityLevel(label=label, kind=kind, price=price, formed_at=formed)


def liquidity_levels(
    *,
    session: str,
    as_of: pd.Timestamp,
    base: IntradaySeries,
    hourly: IntradaySeries | None = None,
    four_hour: IntradaySeries | None = None,
    swing_lookback: int = 2,
    equal_tolerance: float = 0.0005,
) -> list[LiquidityLevel]:
    """Every draw on liquidity visible at ``as_of``, de-duplicated by label+price.

    The union of the previous session's high/low, the current session's running
    high/low, the last completed 1h and 4h bars, the confirmed swings, and any equal
    highs/lows. Duplicate levels (same label and price to the cent) are collapsed so
    a target is not counted twice.
    """
    levels: list[LiquidityLevel] = []
    levels += previous_session_levels(base, session, as_of)
    levels += current_session_levels(base, session, as_of)
    if hourly is not None:
        levels += completed_bar_levels(hourly, label_prefix="prev_1h", session=session, as_of=as_of)
    if four_hour is not None:
        levels += completed_bar_levels(
            four_hour, label_prefix="prev_4h", session=session, as_of=as_of
        )
    frame = base.available(as_of)
    swings = swing_points(frame, lookback=swing_lookback) if not frame.empty else []
    levels += swing_levels(base, as_of, lookback=swing_lookback)
    levels += equal_levels(swings, tolerance=equal_tolerance)

    seen: set[tuple[str, int]] = set()
    unique: list[LiquidityLevel] = []
    for level in levels:
        key = (level.label, round(level.price, 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append(level)
    return sorted(unique, key=lambda item: (item.kind, item.price))


def nearest_above(levels: list[LiquidityLevel], price: float) -> LiquidityLevel | None:
    """The closest level strictly above ``price``, or ``None``."""
    above = [level for level in levels if level.price > price]
    return min(above, key=lambda level: level.price) if above else None


def nearest_below(levels: list[LiquidityLevel], price: float) -> LiquidityLevel | None:
    """The closest level strictly below ``price``, or ``None``."""
    below = [level for level in levels if level.price < price]
    return max(below, key=lambda level: level.price) if below else None


def session_bounds(
    session: str, policy: SessionPolicy | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The session's open and close instants, in UTC, under ``policy``.

    Defaults to the US equity regular session, so existing callers are unchanged;
    a crypto caller passes its own :class:`~.sessions.SessionPolicy`.
    """
    from .sessions import EQUITY_SESSION

    chosen = policy or EQUITY_SESSION
    return chosen.session_open_utc(session), chosen.session_close_utc(session)
