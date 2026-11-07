"""Hand-built intraday sessions for the TJR tests.

Every test frame here is constructed bar by bar so the expected outcome is known
from the construction, not from running the strategy and recording what it said.
The frames are deliberately tiny — one or two sessions — because TJR is a rule
chain, and a rule is best pinned by the smallest market that exercises it.

Nothing here touches the network: the suite's autouse guard would refuse it, and
the whole point of a hand-built frame is that the data is exactly what the test
says it is. Real data is fetched only by ``tradingagents.tjr.fetch_data``.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.tjr.frames import (
    IntradaySeries,
    session_close_utc,
    session_open_utc,
)

# A fixed session date, a Thursday, so the session clock is stable across runs.
SESSION = "2026-10-29"


def frame_from_bars(session: str, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """A 5-minute frame from ``(open, high, low, close)`` tuples, one per bar.

    Bars start at the 09:30 ET open and step 5 minutes. Volume is a constant; no
    rule in TJR reads it.
    """
    index = pd.date_range(
        session_open_utc(session), session_close_utc(session), freq="5min", inclusive="left"
    )[: len(bars)]
    frame = pd.DataFrame(bars, index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


def one_minute_from_five(five: pd.DataFrame) -> pd.DataFrame:
    """Split each 5m bar into five 1m bars, preserving the bar's open/close extremes.

    The path within the bar is linear from open to close, and the bar's own high/low
    are placed on the middle minute, so a 1m series carries the same structure the 5m
    series does without inventing a different path.
    """
    rows: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    for start, row in five.iterrows():
        open_, close = float(row["open"]), float(row["close"])
        for k in range(5):
            oo = open_ + (close - open_) * k / 5
            cc = open_ + (close - open_) * (k + 1) / 5
            high = max(oo, cc, float(row["high"]) if k == 2 else max(oo, cc))
            low = min(oo, cc, float(row["low"]) if k == 2 else min(oo, cc))
            rows.append({"open": oo, "high": high, "low": low, "close": cc, "volume": 200.0})
            index.append(start + pd.Timedelta(minutes=k))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def series_pair(symbol: str, session: str, bars: list[tuple[float, float, float, float]]):
    """The (5m, 1m) :class:`IntradaySeries` pair for a list of 5m bars."""
    five = frame_from_bars(session, bars)
    one = one_minute_from_five(five)
    return (
        IntradaySeries(symbol, "5m", five),
        IntradaySeries(symbol, "1m", one),
    )


# The canonical bullish session: an opening range, a sweep of its low, a 5m rally
# that breaks structure, a pullback into the leg, and a 1m break back up. By
# construction the chain completes at the pullback bars and not before.
BULLISH_BARS: list[tuple[float, float, float, float]] = [
    (100.0, 100.3, 99.8, 100.1),
    (100.1, 100.6, 100.0, 100.5),   # opening-range high 100.6
    (100.5, 100.6, 100.1, 100.2),
    (100.2, 100.3, 99.7, 99.8),
    (99.8, 99.9, 99.4, 99.5),       # opening-range low 99.4
    (99.5, 99.7, 99.45, 99.6),
    (99.6, 99.8, 99.5, 99.7),
    (99.7, 99.9, 99.6, 99.8),
    (99.8, 99.95, 99.6, 99.7),
    (99.7, 99.9, 99.55, 99.6),
    (99.6, 99.7, 97.9, 99.9),       # sweep: low 97.9 takes 99.4, close 99.9 back above
    (99.9, 100.4, 99.85, 100.3),
    (100.3, 100.9, 100.2, 100.8),   # 5m BOS: close 100.8 above the 100.6 swing
    (100.8, 100.9, 100.1, 100.2),   # retrace into the leg's equilibrium (100.15)
    (100.2, 100.7, 100.15, 100.6),  # 1m break back up out of the zone
    (100.6, 101.4, 100.55, 101.3),
    (101.3, 102.0, 101.2, 101.9),
    (101.9, 102.0, 101.4, 101.5),
    (101.5, 102.2, 101.4, 102.1),
    (102.1, 102.5, 102.0, 102.4),
]

# The mirror: a bearish sweep of the opening-range high, a 5m breakdown, a pullback
# up into the leg, and a 1m break back down. A complete setup the long-only book
# must decline rather than trade.
BEARISH_BARS: list[tuple[float, float, float, float]] = [
    (100.0, 100.2, 99.7, 99.9),
    (99.9, 100.0, 99.4, 99.5),      # opening-range low 99.4
    (99.5, 99.9, 99.45, 99.8),
    (99.8, 100.3, 99.7, 100.2),
    (100.2, 100.6, 100.1, 100.5),   # opening-range high 100.6
    (100.5, 100.55, 100.1, 100.2),
    (100.2, 100.4, 100.0, 100.1),
    (100.1, 100.3, 99.9, 100.0),
    (100.0, 100.2, 99.85, 99.9),
    (99.9, 100.1, 99.8, 100.0),
    (100.0, 102.1, 99.95, 100.1),   # sweep: high 102.1 takes 100.6, close 100.1 back below
    (100.1, 100.15, 99.6, 99.7),
    (99.7, 99.8, 99.2, 99.3),       # 5m BOS down: close 99.3 below the 99.4 swing
    (99.3, 99.85, 99.2, 99.75),     # retrace up into the leg's equilibrium (99.75)
    (99.75, 99.8, 99.35, 99.4),     # 1m break back down out of the zone
    (99.4, 99.45, 98.6, 98.7),
    (98.7, 98.8, 98.0, 98.1),
    (98.1, 98.6, 98.0, 98.5),
    (98.5, 98.9, 98.4, 98.8),
    (98.8, 98.85, 98.3, 98.35),
]
