"""Point-in-time market data for the backtest, with look-ahead made impossible.

The single rule this module enforces: a decision taken on date ``T`` may only see
bars dated ``T`` or earlier. It is enforced by construction rather than by
discipline — :meth:`MarketDataFeed.history` slices the frame at the requested
date before returning it, and there is no method that returns a later bar. A
strategy that wanted tomorrow's price would have to reach past the feed, which
means writing new code rather than making a mistake.

The feed holds no network dependency. Bars arrive as a dict of frames, so a test
supplies a handful of rows by hand and the engine cannot fetch anything.

Two distinctions the feed keeps separate, because conflating them is how a
backtest lies:

- **A missing bar is not a zero.** A weekend, a holiday, and a symbol that did
  not trade are all "no bar", and every caller is made to handle that case
  rather than receiving a carried-forward price it did not ask for.
- **The trading calendar is the data's own.** Sessions are the union of the
  symbols' bar dates, so a holiday is simply a date with no bar rather than a
  date the feed wrongly claims is tradeable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BarDataError(ValueError):
    """The supplied bars cannot be used as a market data feed."""


class Bar(BaseModel):
    """One session's prices for one symbol.

    ``open`` and ``close`` are what an order can actually be filled at, and are
    the only two prices the engine uses. ``high``/``low`` exist for stop and
    target checks, where the intraday extreme decides whether a level was hit.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    symbol: str
    date: str = Field(description="Session date, YYYY-MM-DD")
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _prices_are_ordered(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"high {self.high} is below low {self.low} for {self.symbol} on {self.date}")
        for name, value in (("open", self.open), ("close", self.close)):
            if not (self.low - 1e-9 <= value <= self.high + 1e-9):
                raise ValueError(
                    f"{name} {value} is outside [{self.low}, {self.high}] for "
                    f"{self.symbol} on {self.date}"
                )
        return self


_REQUIRED_COLUMNS = ("open", "high", "low", "close")


class MarketDataFeed:
    """Bars for a set of symbols, served strictly as of a date.

    Built from a mapping of symbol to a pandas frame indexed by date, with
    ``open``/``high``/``low``/``close`` (case-insensitive) and optional
    ``volume``. Every frame is normalized once at construction, so the serving
    methods do no work beyond a slice.
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], *, symbols: Iterable[str] | None = None) -> None:
        if not frames:
            raise BarDataError("a market data feed needs at least one symbol")
        self._frames: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            self._frames[symbol] = self._normalize(symbol, frame)
        self._symbols = tuple(sorted(symbols)) if symbols is not None else tuple(sorted(self._frames))
        missing = [s for s in self._symbols if s not in self._frames]
        if missing:
            raise BarDataError(f"no bars supplied for {missing}")
        self._dates = self._build_calendar()

    # --- Construction -------------------------------------------------------

    @staticmethod
    def _normalize(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Lower-case the columns, parse the index to dates, sort, and drop bad rows."""
        if frame is None or frame.empty:
            raise BarDataError(f"no bars for {symbol}")
        data = frame.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        for column in _REQUIRED_COLUMNS:
            if column not in data.columns:
                raise BarDataError(f"bars for {symbol} are missing the {column!r} column")
        if "volume" not in data.columns:
            data["volume"] = pd.NA
        index = pd.to_datetime(data.index, errors="coerce")
        if index.isna().any():
            raise BarDataError(f"bars for {symbol} have an unparseable date")
        # A tz-aware index is converted to its own local dates rather than UTC,
        # so a Tokyo bar keeps its Tokyo session date.
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        data.index = index.normalize()
        data = data[~data.index.duplicated(keep="last")].sort_index()
        keep = ["open", "high", "low", "close", "volume"]
        data = data[keep].astype({"open": float, "high": float, "low": float, "close": float})
        # Drop unusable rows *here*, at construction, rather than letting one
        # through and having ``Bar`` raise it mid-run: a backtest that dies on the
        # fortieth session because the vendor emitted an inf — or a bar whose high
        # sits below its low — has hidden the bad row instead of reporting it.
        # Dropping it keeps the run alive and the date simply has no bar, which
        # every caller already handles. The consistency check mirrors ``Bar``'s own
        # validator (same 1e-9 tolerance) so a row that survives construction can
        # never raise when it is served.
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        prices = data[["open", "high", "low", "close"]]
        tol = 1e-9
        consistent = (
            prices.notna().all(axis=1)
            & (prices > 0).all(axis=1)
            & (data["high"] >= data["low"])
            & (data["open"] >= data["low"] - tol) & (data["open"] <= data["high"] + tol)
            & (data["close"] >= data["low"] - tol) & (data["close"] <= data["high"] + tol)
        )
        data = data[consistent]
        if data.empty:
            raise BarDataError(f"bars for {symbol} have no usable rows")
        return data

    def _build_calendar(self) -> tuple[str, ...]:
        """Every session any symbol traded, ascending, as canonical date strings."""
        dates: set[pd.Timestamp] = set()
        for frame in self._frames.values():
            dates |= set(frame.index)
        return tuple(d.strftime("%Y-%m-%d") for d in sorted(dates))

    # --- Serving ------------------------------------------------------------

    @property
    def symbols(self) -> tuple[str, ...]:
        """The tradable symbols, sorted."""
        return self._symbols

    def calendar(self, start: str | None = None, end: str | None = None) -> list[str]:
        """Session dates within ``[start, end]``, both inclusive and optional."""
        dates = self._dates
        if start is not None:
            dates = tuple(d for d in dates if d >= start)
        if end is not None:
            dates = tuple(d for d in dates if d <= end)
        return list(dates)

    def bar(self, symbol: str, on: str) -> Bar | None:
        """The bar for ``symbol`` on ``on``, or ``None`` when it did not trade.

        ``None`` is a real answer — a weekend, a holiday, a halt — and is never
        replaced by a neighbouring bar's price.
        """
        frame = self._frame_for(symbol)
        try:
            stamp = pd.Timestamp(on).normalize()
        except (ValueError, TypeError):
            return None
        if stamp not in frame.index:
            return None
        row = frame.loc[stamp]
        volume = row["volume"]
        return Bar(
            symbol=symbol,
            date=stamp.strftime("%Y-%m-%d"),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=None if pd.isna(volume) else float(volume),
        )

    def history(self, symbol: str, until: str) -> pd.DataFrame:
        """Every bar for ``symbol`` dated ``until`` or earlier. Never a later one.

        This is the only way a strategy sees prices, and the slice is applied
        here, so a caller cannot obtain a future bar from the feed even by
        asking for one.
        """
        frame = self._frame_for(symbol)
        try:
            stamp = pd.Timestamp(until).normalize()
        except (ValueError, TypeError):
            raise BarDataError(f"until must be a date, got {until!r}") from None
        return frame[frame.index <= stamp]

    def latest_bar_at_or_before(self, symbol: str, on: str) -> Bar | None:
        """The most recent bar dated ``on`` or earlier, for marking a stale position.

        Used to value a holding on a session its own market was closed, where
        carrying the last known close forward is the honest choice. It is
        deliberately *not* used to fill orders: an order needs a bar for the day
        it executes, and a carried-forward price would let a backtest trade on a
        day the market did not open.
        """
        history = self.history(symbol, on)
        if history.empty:
            return None
        return self.bar(symbol, history.index[-1].strftime("%Y-%m-%d"))

    def previous_session(self, on: str) -> str | None:
        """The last session strictly before ``on``, or ``None``."""
        earlier = [d for d in self._dates if d < on]
        return earlier[-1] if earlier else None

    def next_session(self, on: str) -> str | None:
        """The first session strictly after ``on``, or ``None`` at the end of the data."""
        later = [d for d in self._dates if d > on]
        return later[0] if later else None

    def _frame_for(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._frames:
            raise BarDataError(
                f"no bars for {symbol}; the feed holds {list(self._frames)}"
            )
        return self._frames[symbol]

    def as_of(self, session: str) -> MarketDataFeed:
        """A feed containing only bars dated ``session`` or earlier.

        Used to hand a strategy a view it cannot look past. Every frame is
        re-cut, so ``bar``, ``history``, ``latest_bar_at_or_before`` and the
        calendar all answer from the same bounded data and none of them can
        return a later date. Returns ``self`` when nothing is cut, so a caller
        that asks for the last session pays nothing for the call.

        The result is a distinct feed, not a proxy: a caller that tried to reach
        the future would have to hold the original object, which the strategy is
        not given.
        """
        bounded: dict[str, pd.DataFrame] = {}
        for symbol in self._symbols:
            frame = self._frames[symbol]
            cut = frame[frame.index <= pd.Timestamp(session).normalize()]
            if not cut.empty:
                bounded[symbol] = cut
        if not bounded:
            raise BarDataError(f"no bars at or before {session}")
        if all(len(bounded[s]) == len(self._frames[s]) for s in bounded):
            return self
        return MarketDataFeed(bounded)


@dataclass(frozen=True)
class SessionClock:
    """The backtest's notion of "now", so a signal's timestamp is not wall-clock.

    A signal created during a backtest must be timestamped with the session it
    came from, not with the instant the sweep happened to run: otherwise two runs
    of the same data produce different signal ids, and the event log cannot be
    compared between them.
    """

    session: str
    at: datetime

    @staticmethod
    def for_session(session: str, *, hour: int = 21) -> SessionClock:
        """The clock for a session, dated at the close of the US trading day (21:00 UTC)."""
        parsed = datetime.strptime(session, "%Y-%m-%d").replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        from datetime import UTC

        return SessionClock(session=session, at=parsed.replace(tzinfo=UTC))


def trading_days_between(feed: MarketDataFeed, start: str, end: str, symbol: str | None = None) -> list[str]:
    """Sessions in ``[start, end]``, optionally restricted to one symbol's own bars."""
    if symbol is None:
        return feed.calendar(start, end)
    frame = feed.history(symbol, end)
    return [d.strftime("%Y-%m-%d") for d in frame.index if d.strftime("%Y-%m-%d") >= start]


def as_date(value: str | date | datetime) -> str:
    """Canonical ``YYYY-MM-DD`` for a date, datetime or date string."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise BarDataError(f"not a date: {value!r}") from exc
