"""Real intraday OHLCV from Yahoo, for the TJR day-trading strategy.

TJR needs 1-minute and 5-minute bars, plus the session and 1h/4h levels built
from them. The daily loader (``ohlcv.py``) cannot serve any of that: it
normalizes every index to a midnight date, which collapses a whole session to one
row. This module is the intraday counterpart, and it lives in the data layer
because only the data layer may import a vendor library (``tests/test_layering.py``).

What Yahoo actually allows, measured against the live API rather than assumed
(yfinance refuses the request outright past these windows):

- ``1m``  bars: at most **8 days per request**, and only within the **last 30
  days**. Older ranges answer ``1m data not available ... within the last 30 days``.
- ``5m``/``15m`` bars: at most 60 days per request, within the last 60 days.
- ``60m`` bars: up to 730 days.

So an honest intraday study here spans a few recent weeks, not years. The fetch
is chunked to respect the per-request cap, then concatenated and de-duplicated;
it never silently shortens the window a caller asked for — a window that Yahoo
will not serve raises rather than returning a smaller frame that would look like
a complete answer.

Timestamps are the *start* of each bar, kept in UTC. A bar is only ever usable
once it has *closed*: the bar that starts at 10:00 is not available until 10:01.
That rule is enforced downstream (``tjr.frames``), not here — this module's job
is to hand back exactly the rows the vendor reported, validated and ordered.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry

# yfinance caps a single intraday request. These are the documented/observed
# limits; a window longer than the cap is fetched in several requests.
_MAX_DAYS_PER_REQUEST: dict[str, int] = {
    "1m": 7,
    "2m": 55,
    "5m": 55,
    "15m": 55,
    "30m": 55,
    "60m": 55,
    "90m": 55,
    "1h": 55,
}

# The oldest a window may start and still be served, in calendar days back from
# "now". A window reaching past this cannot be answered and is refused.
_MAX_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 30,
    "2m": 55,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
}

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90, "1h": 60,
}

_SUPPORTED = tuple(_MAX_DAYS_PER_REQUEST)

_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def interval_minutes(interval: str) -> int:
    """The bar length in minutes, or raise for an interval this loader cannot fetch."""
    if interval not in _INTERVAL_MINUTES:
        raise ValueError(
            f"unsupported intraday interval {interval!r}; "
            f"choose one of {sorted(_INTERVAL_MINUTES)}"
        )
    return _INTERVAL_MINUTES[interval]


class IntradayDataError(NoMarketDataError):
    """The intraday window could not be served as asked."""


def _chunk_bounds(start: dt.date, end: dt.date, max_days: int) -> list[tuple[dt.date, dt.date]]:
    """``[start, end]`` split into inclusive chunks of at most ``max_days`` days.

    ``end`` is inclusive here; yfinance's own ``end`` is exclusive, so the caller
    adds a day when it builds the request.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    bounds: list[tuple[dt.date, dt.date]] = []
    cursor = start
    step = dt.timedelta(days=max_days)
    while cursor <= end:
        chunk_end = min(cursor + step - dt.timedelta(days=1), end)
        bounds.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return bounds


def _normalize_intraday(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Lower-case columns, a UTC DatetimeIndex of bar starts, sorted and de-duplicated.

    Yahoo returns a tz-aware index in the exchange's timezone. Converting to UTC
    makes two symbols on the same clock comparable and makes the closed-bar rule a
    plain arithmetic comparison. Duplicate bar starts (a boundary shared by two
    chunks) are collapsed keeping the later row, which is the more complete one.
    """
    frame = downloaded.copy()
    # A single-symbol download can carry a MultiIndex column level; drop it.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            raise IntradayDataError(symbol, symbol, f"intraday bars are missing {column!r}")
    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if index.isna().any():
        raise IntradayDataError(symbol, symbol, "intraday bars have an unparseable timestamp")
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[list(_PRICE_COLUMNS)]


def _drop_incomplete_rows(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop rows whose OHLC is missing or inconsistent, mirroring the daily loader.

    A vendor can emit a flat/NaN row for a bar with no trades; keeping it would let
    a non-finite price into a level calculation. The row is dropped, not zeroed.
    """
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    tol = 1e-9
    consistent = (
        prices.notna().all(axis=1)
        & (prices > 0).all(axis=1)
        & (prices["high"] >= prices["low"])
        & (prices["open"] >= prices["low"] - tol) & (prices["open"] <= prices["high"] + tol)
        & (prices["close"] >= prices["low"] - tol) & (prices["close"] <= prices["high"] + tol)
    )
    frame = frame.copy()
    frame[["open", "high", "low", "close"]] = prices
    return frame[consistent]


def load_intraday_range(
    symbol: str,
    start: str,
    end: str,
    *,
    interval: str = "1m",
    now: dt.date | None = None,
) -> pd.DataFrame:
    """Real intraday OHLCV for ``symbol`` over ``[start, end]``, inclusive.

    Fetches in vendor-sized chunks, validates every row, and returns a UTC-indexed
    frame of bar *starts*. Refuses a window Yahoo cannot serve (too far back, or an
    unsupported interval) rather than returning a shorter frame, and refuses an
    empty result as ``NoMarketDataError`` — an absence, not a zero-row dataset.
    """
    if interval not in _SUPPORTED:
        raise ValueError(
            f"unsupported intraday interval {interval!r}; choose one of {sorted(_SUPPORTED)}"
        )
    canonical = normalize_symbol(symbol)
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    reference = now or dt.datetime.now(dt.UTC).date()

    oldest = reference - dt.timedelta(days=_MAX_LOOKBACK_DAYS[interval])
    if start_date < oldest:
        raise IntradayDataError(
            symbol,
            canonical,
            f"Yahoo serves {interval} bars only within the last "
            f"{_MAX_LOOKBACK_DAYS[interval]} days; {start_date} is before {oldest}",
        )

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_bounds(
        start_date, end_date, _MAX_DAYS_PER_REQUEST[interval]
    ):
        # yfinance's end is exclusive: add a day so chunk_end is included.
        end_exclusive = (chunk_end + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        downloaded = yf_retry(
            lambda cs=chunk_start, ee=end_exclusive: _download(
                canonical, cs.strftime("%Y-%m-%d"), ee, interval
            )
        )
        if downloaded is None or downloaded.empty:
            # An empty chunk inside a range is a hole, not a valid answer for a
            # window a caller asked to be complete. It is reported, not skipped.
            raise_for_empty(symbol, canonical, f"{interval} bars for {chunk_start}..{chunk_end}")
        frames.append(downloaded)

    combined = pd.concat(frames)
    frame = _normalize_intraday(combined, symbol)
    frame = _drop_incomplete_rows(frame, symbol)
    if frame.empty:
        raise IntradayDataError(symbol, canonical, f"no usable {interval} rows in the window")
    return frame


def _download(canonical: str, start: str, end_exclusive: str, interval: str) -> pd.DataFrame:
    """One yfinance intraday request. A vendor failure is raised as unavailable."""
    import yfinance as yf

    try:
        return yf.Ticker(canonical).history(
            start=start,
            end=end_exclusive,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )
    except Exception as exc:  # noqa: BLE001 - any transit failure is "unavailable"
        raise VendorUnavailableError(
            f"Yahoo intraday request failed for {canonical} ({interval}): {type(exc).__name__}"
        ) from exc
