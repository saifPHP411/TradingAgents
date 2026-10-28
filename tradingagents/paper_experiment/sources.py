"""Price sources for a paper experiment, with their provenance made explicit.

A prospective paper experiment needs prices. Where those prices come from decides
what the experiment may honestly claim, so the source is a first-class object
that states its own nature rather than a bare callable that hides it:

- :class:`FrozenPriceSource` reads the committed, historical dataset. It is
  reproducible and point-in-time, but it is **not the current market** — its last
  session is fixed in the past — so an experiment on it cannot be called a live
  paper run. ``is_current_market`` is ``False`` and the report says so.
- :class:`StaticPriceSource` is a fixed, in-memory series for tests.

A source that would represent the *current* market does not ship here: wiring one
would mean a vendor call, and this phase adds no network path. The
:class:`PriceSource` protocol is the seam a future current-market source would
implement; until one exists, :func:`current_market_source_available` returns
``False`` and the experiment refuses to claim live readiness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


class PriceSourceError(ValueError):
    """A price source cannot serve a request: unknown symbol, session, or bad data."""


@dataclass(frozen=True)
class SessionBars:
    """One symbol's OHLC for one session, validated finite and positive."""

    symbol: str
    session: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise PriceSourceError(
                    f"{self.symbol} {self.session}: {name} must be finite and positive, got {value}"
                )
        if self.high < self.low:
            raise PriceSourceError(
                f"{self.symbol} {self.session}: high {self.high} is below low {self.low}"
            )
        for name in ("open", "close"):
            value = getattr(self, name)
            if not (self.low - 1e-9 <= value <= self.high + 1e-9):
                raise PriceSourceError(
                    f"{self.symbol} {self.session}: {name} {value} is outside "
                    f"[{self.low}, {self.high}]"
                )


@runtime_checkable
class PriceSource(Protocol):
    """The price data a paper experiment draws on.

    ``is_current_market`` is the honesty flag the report reads: a source that is
    not current cannot make an experiment a live paper run.
    """

    name: str

    @property
    def is_current_market(self) -> bool:  # pragma: no cover - protocol
        ...

    def sessions(self) -> tuple[str, ...]:  # pragma: no cover - protocol
        ...

    def covers(self, session: str) -> bool:  # pragma: no cover - protocol
        ...

    def bars(self, symbol: str, session: str) -> SessionBars | None:  # pragma: no cover - protocol
        ...

    def window(self, symbol: str, session: str) -> pd.DataFrame:  # pragma: no cover - protocol
        ...


def current_market_source_available() -> bool:
    """Whether a source representing the current market is wired.

    Always ``False`` in this phase: no current-market source ships, because
    wiring one would add a vendor call. The experiment consults this before it
    claims live readiness, so "paper-ready" cannot be reported from frozen data
    alone.
    """
    return False


class FrozenPriceSource:
    """The committed historical dataset, exposed point-in-time.

    Bars are read through a session-bounded feed (``MarketDataFeed.as_of``), so a
    request for session T can never see a bar after T. The dataset ends in the
    past, so ``is_current_market`` is ``False`` and :meth:`covers` is honest about
    the horizon.
    """

    def __init__(self, feed, *, name: str = "frozen_baseline") -> None:
        self._feed = feed
        self.name = name
        calendar = feed.calendar()
        if not calendar:
            raise PriceSourceError("the frozen feed carries no sessions")
        self._sessions = tuple(calendar)
        self._session_set = set(self._sessions)
        self.dataset_start = self._sessions[0]
        self.dataset_end = self._sessions[-1]

    @property
    def is_current_market(self) -> bool:
        return False

    def sessions(self) -> tuple[str, ...]:
        return self._sessions

    def covers(self, session: str) -> bool:
        return session in self._session_set

    def bars(self, symbol: str, session: str) -> SessionBars | None:
        if not self.covers(session):
            raise PriceSourceError(
                f"{session} is outside the frozen dataset "
                f"({self.dataset_start}..{self.dataset_end}); a session the data does "
                "not cover cannot be processed"
            )
        bar = self._feed.as_of(session).bar(symbol, session)
        if bar is None:
            return None
        return SessionBars(
            symbol=bar.symbol,
            session=bar.date,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

    def window(self, symbol: str, session: str) -> pd.DataFrame:
        if not self.covers(session):
            raise PriceSourceError(
                f"{session} is outside the frozen dataset "
                f"({self.dataset_start}..{self.dataset_end})"
            )
        frame = self._feed.as_of(session).history(symbol, session)
        if frame is None or len(frame) == 0:
            raise PriceSourceError(f"no bars for {symbol!r} at or before {session}")
        return frame


class StaticPriceSource:
    """A fixed in-memory series, for tests. Explicitly not the current market.

    ``frames`` maps a symbol to a DataFrame indexed by session date with
    open/high/low/close/volume columns, exactly as a feed would carry it.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], *, name: str = "static_test") -> None:
        if not frames:
            raise PriceSourceError("a static source needs at least one symbol")
        self.name = name
        self._frames: dict[str, pd.DataFrame] = {}
        all_sessions: set[str] = set()
        for symbol, frame in frames.items():
            ordered = frame.sort_index()
            self._frames[symbol] = ordered
            all_sessions |= {str(index)[:10] for index in ordered.index}
        self._sessions = tuple(sorted(all_sessions))
        self._session_set = all_sessions
        # Validate every row eagerly, so a malformed fixture fails at
        # construction rather than part-way through a run. A test double that
        # carries a NaN would otherwise poison the experiment quietly.
        for symbol, session in ((s, d) for s in self._frames for d in self._sessions):
            self.bars(symbol, session)

    @property
    def is_current_market(self) -> bool:
        return False

    def sessions(self) -> tuple[str, ...]:
        return self._sessions

    def covers(self, session: str) -> bool:
        return session in self._session_set

    def bars(self, symbol: str, session: str) -> SessionBars | None:
        frame = self._frames.get(symbol)
        if frame is None:
            raise PriceSourceError(f"the static source does not carry {symbol!r}")
        if not self.covers(session):
            raise PriceSourceError(f"{session} is outside the static source's sessions")
        key = pd.Timestamp(session)
        if key not in frame.index:
            return None
        row = frame.loc[key]
        volume = row.get("volume")
        return SessionBars(
            symbol=symbol,
            session=session,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=None if volume is None or pd.isna(volume) else float(volume),
        )

    def window(self, symbol: str, session: str) -> pd.DataFrame:
        frame = self._frames.get(symbol)
        if frame is None:
            raise PriceSourceError(f"the static source does not carry {symbol!r}")
        bounded = frame[frame.index <= pd.Timestamp(session)]
        if len(bounded) == 0:
            raise PriceSourceError(f"no bars for {symbol!r} at or before {session}")
        return bounded
