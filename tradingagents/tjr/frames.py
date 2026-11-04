"""Point-in-time intraday frames for TJR, with the closed-bar rule enforced.

TJR is a multi-timeframe day-trading strategy: it reads 5-minute structure for the
reversal and continuation, then drops to 1 minute for the entry. Every read has to
respect one rule — **a bar is only visible once it has closed**. The 5-minute bar
that starts at 10:00 does not exist until 10:05.

That rule is enforced here by construction, the same way
:class:`~tradingagents.backtesting.data.MarketDataFeed` enforces its date cut: the
only way to get bars out is :meth:`IntradaySeries.available`, which drops every bar
whose *close* is after the instant asked for. There is no method that returns a
later bar, so a strategy cannot see the future by reaching one attribute deeper.

The frame's index is the bar's **start**, in UTC. Timestamps are kept in UTC rather
than the exchange's zone because the closed-bar comparison is then plain
arithmetic; the *session* (which day a bar belongs to, and where the regular
session opens) is derived in the exchange's own timezone, where a day boundary
means what a trader means by it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SESSION_TZ = "America/New_York"
"""US equities trade on Eastern time; the session day and the 9:30 open are its."""

REGULAR_OPEN = (9, 30)
REGULAR_CLOSE = (16, 0)

_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")

_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "4h": 240}


class IntradayFrameError(ValueError):
    """The supplied frame cannot serve as a point-in-time intraday series."""


def interval_minutes(interval: str) -> int:
    """Bar length in minutes for a supported interval label."""
    try:
        return _INTERVAL_MINUTES[interval]
    except KeyError:
        raise IntradayFrameError(
            f"unsupported interval {interval!r}; choose one of {sorted(_INTERVAL_MINUTES)}"
        ) from None


def normalize_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and canonicalize a raw intraday frame to a UTC bar-start index.

    Rejects a frame that is empty, has a duplicate timestamp, is missing a price
    column, or carries a non-finite/non-positive price. A *duplicate* timestamp is
    rejected rather than collapsed, because a repeated bar could restate one the
    run already used and silently change a level.
    """
    if frame is None or frame.empty:
        raise IntradayFrameError(f"no intraday bars for {symbol}")
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    for column in ("open", "high", "low", "close"):
        if column not in data.columns:
            raise IntradayFrameError(f"intraday bars for {symbol} are missing {column!r}")
    if "volume" not in data.columns:
        data["volume"] = pd.NA
    index = pd.to_datetime(data.index, errors="coerce", utc=True)
    if index.isna().any():
        raise IntradayFrameError(f"intraday bars for {symbol} have an unparseable timestamp")
    if index.duplicated().any():
        raise IntradayFrameError(f"intraday bars for {symbol} have a duplicate timestamp")
    data.index = index
    data = data.sort_index()
    prices = data[["open", "high", "low", "close"]].astype(float)
    if not prices.notna().all().all():
        raise IntradayFrameError(f"intraday bars for {symbol} carry a missing price")
    if (prices <= 0).any().any():
        raise IntradayFrameError(f"intraday bars for {symbol} carry a non-positive price")
    if (prices["high"] < prices["low"]).any():
        raise IntradayFrameError(f"intraday bars for {symbol} have a high below its low")
    data[["open", "high", "low", "close"]] = prices
    return data[list(_PRICE_COLUMNS)]


def session_of(stamp: pd.Timestamp) -> str:
    """The US trading session (YYYY-MM-DD) a UTC bar-start belongs to."""
    local = pd.Timestamp(stamp).tz_convert(SESSION_TZ)
    return local.strftime("%Y-%m-%d")


def session_open_utc(session: str) -> pd.Timestamp:
    """The 09:30 ET regular-session open of ``session``, as a UTC timestamp."""
    local = pd.Timestamp(session, tz=SESSION_TZ).replace(
        hour=REGULAR_OPEN[0], minute=REGULAR_OPEN[1], second=0, microsecond=0
    )
    return local.tz_convert("UTC")


def session_close_utc(session: str) -> pd.Timestamp:
    """The 16:00 ET regular-session close of ``session``, as a UTC timestamp."""
    local = pd.Timestamp(session, tz=SESSION_TZ).replace(
        hour=REGULAR_CLOSE[0], minute=REGULAR_CLOSE[1], second=0, microsecond=0
    )
    return local.tz_convert("UTC")


@dataclass(frozen=True)
class IntradaySeries:
    """One symbol's intraday bars, served strictly as of an instant.

    ``interval`` is the bar length label (``"1m"``, ``"5m"``); the frame's index is
    the bar start in UTC. ``available`` is the only reader, and it never returns a
    bar that had not closed by the instant asked for.
    """

    symbol: str
    interval: str
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        # Validate once, at construction, so no reader can be handed a bad frame.
        object.__setattr__(self, "frame", normalize_frame(self.frame, self.symbol))
        interval_minutes(self.interval)

    @property
    def bar_minutes(self) -> int:
        return interval_minutes(self.interval)

    def available(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Bars whose close (start + interval) is at or before ``as_of``.

        The look-ahead guard. A bar starting at ``t`` closes at ``t + interval``;
        it is included only when that close is not after ``as_of``. The comparison
        is inclusive of the close instant, because a bar is complete at its close.
        """
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(minutes=self.bar_minutes)
        return self.frame[self.frame.index <= cutoff]

    def session_bars(self, session: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """The regular-session bars of ``session`` that had closed by ``as_of``.

        Restricted to the 09:30–16:00 ET window, so a pre-market print cannot enter
        a session's structure — TJR is explicit that entries wait for the open.
        """
        frame = self.available(as_of)
        if frame.empty:
            return frame
        local = frame.index.tz_convert(SESSION_TZ)
        open_local = session_open_utc(session).tz_convert(SESSION_TZ)
        close_local = session_close_utc(session).tz_convert(SESSION_TZ)
        in_session = (
            (local.strftime("%Y-%m-%d") == session)
            & (local >= open_local)
            & (local < close_local)
        )
        return frame[in_session]

    def last_close(self, as_of: pd.Timestamp) -> float | None:
        frame = self.available(as_of)
        return None if frame.empty else float(frame["close"].iloc[-1])


def resample(series: IntradaySeries, interval: str) -> IntradaySeries:
    """Aggregate ``series`` to a coarser interval, anchored on the session open.

    Resampling an already point-in-time frame keeps the closed-bar guarantee: the
    coarser bars are built only from bars the caller could see. Each bar is labelled
    by its start, and a partially-formed bucket (a trailing bar whose source rows do
    not fill it) is dropped rather than emitted as a short bar, so the result never
    contains a bar that has not closed.
    """
    target_minutes = interval_minutes(interval)
    if target_minutes < series.bar_minutes:
        raise IntradayFrameError(f"cannot resample {series.interval} up to a finer {interval}")
    if target_minutes == series.bar_minutes:
        return series
    frame = series.frame
    local = frame.index.tz_convert(SESSION_TZ)
    day = pd.Series(local.strftime("%Y-%m-%d"), index=frame.index)
    anchor = pd.DatetimeIndex(
        [pd.Timestamp(d, tz=SESSION_TZ) for d in day]
    ) + pd.Timedelta(hours=REGULAR_OPEN[0], minutes=REGULAR_OPEN[1])
    offset_minutes = ((local - anchor).total_seconds() // 60).astype("int64")
    bucket_start = anchor + pd.to_timedelta((offset_minutes // target_minutes) * target_minutes, unit="m")
    expected = target_minutes // series.bar_minutes

    grouped = frame.groupby(bucket_start, sort=True)
    aggregated = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    full = grouped.size() == expected
    aggregated = aggregated[full.values]
    aggregated.index = pd.DatetimeIndex(aggregated.index).tz_convert("UTC")
    aggregated.index.name = frame.index.name
    return IntradaySeries(symbol=series.symbol, interval=interval, frame=aggregated)
