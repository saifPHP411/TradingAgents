"""Price sources for a paper experiment, with their provenance made explicit.

A prospective paper experiment needs prices. Where those prices come from decides
what the experiment may honestly claim, so the source is a first-class object
that states its own nature rather than a bare callable that hides it:

- :class:`FrozenPriceSource` reads the committed, historical dataset. It is
  reproducible and point-in-time, but it is **not the current market** — its last
  session is fixed in the past — so an experiment on it cannot be called a live
  paper run. ``is_current_market`` is ``False`` and the report says so.
- :class:`StaticPriceSource` is a fixed, in-memory series for tests.
- the vendor-backed sources in :mod:`tradingagents.paper_experiment.providers`
  (Alpha Vantage daily bars, Finnhub quotes) declare their own provenance and are
  the only ones that touch the network. They are opt-in and never the default.

The :class:`PriceSource` protocol is the seam every source implements. A source
states whether it represents the *current* market; a historical vendor dataset
does not, so ``is_current_market`` stays ``False`` and the experiment refuses to
claim live readiness on it.

Data-provenance rule that every source here obeys: a frame is returned in the
project's canonical shape (a ``DatetimeIndex`` of naive, midnight-normalised
sessions and lowercase ``open/high/low/close/volume`` columns), so the shared
validator and the snapshot fingerprint see the same thing whatever the vendor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# The canonical frame shape every source produces and the validator expects.
BAR_COLUMNS = ("open", "high", "low", "close", "volume")
SESSION_COLUMN = "session"


class PriceSourceError(ValueError):
    """A price source cannot serve a request: unknown symbol, session, or bad data."""


@dataclass(frozen=True)
class SourceProvenance:
    """Where a source's data comes from, recorded so a report cannot imply more.

    ``kind`` is the closed vocabulary that separates the cases a report must not
    conflate: ``frozen`` (the committed dataset), ``static`` (a test fixture),
    ``vendor_daily`` (a vendor's historical daily bars) and ``vendor_quote`` (a
    vendor's current-day quote, which is **not** historical OHLCV).
    """

    name: str
    kind: str
    is_current_market: bool
    symbol: str | None = None
    fetched_at: str | None = None
    last_bar_date: str | None = None
    timezone: str | None = None
    adjusted: bool | None = None
    dataset_start: str | None = None
    dataset_end: str | None = None
    snapshot_id: str | None = None
    content_hash: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "is_current_market": self.is_current_market,
            "symbol": self.symbol,
            "fetched_at": self.fetched_at,
            "last_bar_date": self.last_bar_date,
            "timezone": self.timezone,
            "adjusted": self.adjusted,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SourceProvenance:
        """Rebuild a provenance from its serialized form, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        fields = {k: v for k, v in payload.items() if k in known}
        if "notes" in fields and fields["notes"] is not None:
            fields["notes"] = tuple(fields["notes"])
        return cls(**fields)


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """A validated frame as JSON-safe rows, so a vendor frame survives a restart.

    One row per session, with the session as ``date`` and the OHLCV columns. The
    frame is re-validated on the way back in (:meth:`FramePriceSource.from_records`),
    so a corrupted state file is refused rather than trusted.
    """
    rows: list[dict] = []
    for index, row in frame.sort_index().iterrows():
        record = {"date": str(index)[:10]}
        for column in BAR_COLUMNS:
            if column in frame.columns:
                value = row[column]
                record[column] = None if value is None or pd.isna(value) else float(value)
        rows.append(record)
    return rows


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """The inverse of :func:`frame_to_records`: rows back to an indexed frame."""
    if not records:
        raise PriceSourceError("cannot rebuild a frame from an empty record list")
    frame = pd.DataFrame(records)
    if "date" not in frame.columns:
        raise PriceSourceError("a stored frame row is missing its 'date'")
    frame = frame.set_index(pd.to_datetime(frame.pop("date"), errors="raise"))
    frame.index.name = "session"
    return frame


def _validate_frame(frame: pd.DataFrame, symbol: str, source_name: str) -> pd.DataFrame:
    """Return ``frame`` re-indexed by session, or raise :class:`PriceSourceError`.

    The single validator every source's bars pass through, so a malformed vendor
    payload — NaN, infinity, a non-positive price, a high below its low, an
    open/close outside the range, a duplicate or out-of-order session, or a
    non-numeric column — is refused here rather than poisoning the ledger later.
    """
    if frame is None or len(frame) == 0:
        raise PriceSourceError(f"{source_name}: {symbol} has no bars")
    missing = [c for c in BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise PriceSourceError(
            f"{source_name}: {symbol} frame is missing column(s) {missing}"
        )

    index = pd.to_datetime(frame.index, errors="coerce")
    if index.isna().any():
        raise PriceSourceError(f"{source_name}: {symbol} has an unparseable session")
    if index.has_duplicates:
        duplicated = sorted({str(d)[:10] for d in index[index.duplicated()]})
        raise PriceSourceError(
            f"{source_name}: {symbol} has duplicate session(s) {duplicated[:5]}"
        )
    if not index.is_monotonic_increasing:
        raise PriceSourceError(
            f"{source_name}: {symbol} sessions are not in ascending order"
        )

    clean = frame.copy()
    clean.index = index
    clean.index.name = SESSION_COLUMN

    for column in BAR_COLUMNS:
        values = pd.to_numeric(clean[column], errors="coerce")
        if values.isna().any():
            bad = clean.index[values.isna()][0]
            raise PriceSourceError(
                f"{source_name}: {symbol} has a non-numeric {column} at {str(bad)[:10]}"
            )
        clean[column] = values.astype(float)

    ohlc = clean[list(BAR_COLUMNS[:4])]
    if not ohlc.map(math.isfinite).all().all():
        raise PriceSourceError(f"{source_name}: {symbol} has a non-finite OHLC value")
    for column in ("open", "high", "low", "close"):
        if (clean[column] <= 0).any():
            bad = clean.index[clean[column] <= 0][0]
            raise PriceSourceError(
                f"{source_name}: {symbol} has a non-positive {column} at {str(bad)[:10]}"
            )
    if (clean["high"] < clean["low"]).any():
        bad = clean.index[clean["high"] < clean["low"]][0]
        raise PriceSourceError(
            f"{source_name}: {symbol} has high below low at {str(bad)[:10]}"
        )
    for column in ("open", "close"):
        out_of_range = (clean[column] < clean["low"] - 1e-9) | (
            clean[column] > clean["high"] + 1e-9
        )
        if out_of_range.any():
            bad = clean.index[out_of_range][0]
            raise PriceSourceError(
                f"{source_name}: {symbol} has {column} outside [low, high] at "
                f"{str(bad)[:10]}"
            )
    return clean


class FramePriceSource:
    """A price source backed by in-memory OHLCV frames, one per symbol.

    Shared by :class:`StaticPriceSource` and every vendor-backed source, so the
    validation, the session set and the point-in-time ``window`` are written once
    and a vendor cannot diverge from the frozen dataset's behaviour. Every frame
    is validated eagerly at construction.
    """

    kind = "static"

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        name: str = "frame_source",
        provenance: SourceProvenance | None = None,
    ) -> None:
        if not frames:
            raise PriceSourceError("a price source needs at least one symbol")
        self.name = name
        self._frames: dict[str, pd.DataFrame] = {}
        all_sessions: set[str] = set()
        for symbol, frame in frames.items():
            ordered = _validate_frame(frame, symbol, name).sort_index()
            self._frames[symbol] = ordered
            all_sessions |= {str(index)[:10] for index in ordered.index}
        self._sessions = tuple(sorted(all_sessions))
        self._session_set = all_sessions
        self.dataset_start = self._sessions[0]
        self.dataset_end = self._sessions[-1]
        self.provenance = provenance or SourceProvenance(
            name=name,
            kind=self.kind,
            is_current_market=False,
            dataset_start=self.dataset_start,
            dataset_end=self.dataset_end,
        )

    @property
    def is_current_market(self) -> bool:
        return bool(self.provenance.is_current_market)

    def sessions(self) -> tuple[str, ...]:
        return self._sessions

    def covers(self, session: str) -> bool:
        return session in self._session_set

    def bars(self, symbol: str, session: str) -> SessionBars | None:
        frame = self._frames.get(symbol)
        if frame is None:
            raise PriceSourceError(f"{self.name} does not carry {symbol!r}")
        if not self.covers(session):
            raise PriceSourceError(f"{session} is outside {self.name}'s sessions")
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
            raise PriceSourceError(f"{self.name} does not carry {symbol!r}")
        bounded = frame[frame.index <= pd.Timestamp(session)]
        if len(bounded) == 0:
            raise PriceSourceError(f"no bars for {symbol!r} at or before {session}")
        return bounded

    def frame(self, symbol: str) -> pd.DataFrame:
        """The full stored frame for ``symbol`` (used to hash a snapshot)."""
        frame = self._frames.get(symbol)
        if frame is None:
            raise PriceSourceError(f"{self.name} does not carry {symbol!r}")
        return frame

    def to_dict(self) -> dict:
        """Serialize the source's bars and provenance, so a restart can rebuild it.

        A vendor source fetches once at construction; persisting its validated
        bars is what lets a resumed run continue on the *same* data instead of
        re-fetching (and, on a quota-limited vendor, failing to).
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "provenance": self.provenance.to_dict(),
            "frames": {symbol: frame_to_records(frame) for symbol, frame in self._frames.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FramePriceSource:
        """Rebuild a source from :meth:`to_dict`, re-validating every stored frame."""
        frames = payload.get("frames")
        if not frames:
            raise PriceSourceError("a stored price source carries no frames")
        provenance = payload.get("provenance")
        restored = cls(
            {symbol: records_to_frame(rows) for symbol, rows in frames.items()},
            name=payload.get("name", "restored_source"),
            provenance=SourceProvenance.from_dict(provenance) if provenance else None,
        )
        # Preserve the recorded kind (e.g. ``vendor_daily``) on the instance; the
        # class default is only a fallback for a source with no stored kind.
        stored_kind = payload.get("kind")
        if stored_kind:
            restored.kind = stored_kind
        return restored



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
    """Whether a source representing the *current* market is wired.

    ``True`` now that a vendor-backed current-market source exists (Finnhub's
    quote endpoint reports a same-day price). It does **not** mean a live paper
    run is available: a quote source serves only a same-day price, not the
    historical OHLCV a multi-session experiment needs, and a daily-bar source is
    still a historical dataset. The report keeps both facts apart.
    """
    from .providers import current_market_sources

    return bool(current_market_sources())


class FrozenPriceSource:
    """The committed historical dataset, exposed point-in-time.

    Bars are read through a session-bounded feed (``MarketDataFeed.as_of``), so a
    request for session T can never see a bar after T. The dataset ends in the
    past, so ``is_current_market`` is ``False`` and :meth:`covers` is honest about
    the horizon.
    """

    kind = "frozen"

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
        self.provenance = SourceProvenance(
            name=name,
            kind=self.kind,
            is_current_market=False,
            dataset_start=self.dataset_start,
            dataset_end=self.dataset_end,
            notes=(
                "the committed historical dataset; reproducible and point-in-time, "
                "but its last session is fixed in the past",
            ),
        )

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


class StaticPriceSource(FramePriceSource):
    """A fixed in-memory series, for tests. Explicitly not the current market.

    ``frames`` maps a symbol to a DataFrame indexed by session date with
    open/high/low/close/volume columns, exactly as a feed would carry it. All
    validation and point-in-time behaviour is inherited from
    :class:`FramePriceSource`.
    """

    kind = "static"

    def __init__(self, frames: dict[str, pd.DataFrame], *, name: str = "static_test") -> None:
        super().__init__(frames, name=name)
