"""A versioned, verifiable price snapshot: the input window a record anchors to.

Phase 9's first genuine run produced a record with ``bar_count: 0`` and
``data_snapshot_id: feed:unavailable``. The inference was real, but the record's
*input* was not reproducible: nothing tied the decision to a concrete set of
bars. A record like that cannot anchor a prospective paper experiment, because a
replay has nothing to re-read and no way to prove it re-read the same bars.

This module is the fix. A :class:`PriceSnapshot` is a small, self-contained,
reloadable capture of one symbol's bars up to a session:

- **Explicit identity.** ``snapshot_id`` is ``<symbol>-<start>..<end>-<hash12>``,
  so the window and its content fingerprint are both in the name.
- **Verifiable content.** ``content_hash`` is the same SHA-256 the decision
  record carries (``replay.window_hash``), computed over the dates and each
  bar's open/high/low/close at ten decimal places. ``verify()`` recomputes it
  from the stored bars, so a tampered or drifted snapshot is caught rather than
  trusted. Because it is the *same* function the record uses, a snapshot and the
  record that names it must agree — a check a test asserts.
- **Reloadable.** ``save``/``load`` round-trip the bars, so a replay can rebuild
  the exact frame without the original data vendor.

What a snapshot is *not*: it is not a forecast, not a price series the model was
handed in the smoke path (the smoke prompt is price-free), and not a claim about
strategy quality. It is the input window the record is anchored to, so that
"the model saw these bars" is a checkable statement rather than an assertion.

No live trading path is added here; a snapshot is data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .records import DecisionRecord, DecisionRecordError, assert_secret_free
from .replay import window_hash

SNAPSHOT_SCHEMA_VERSION = 1
_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class SnapshotError(DecisionRecordError):
    """A price snapshot is malformed, or its content does not match its hash."""


@dataclass(frozen=True)
class PriceSnapshot:
    """One symbol's bars up to a session, with a verifiable content hash."""

    snapshot_id: str
    symbol: str
    window_start: str
    window_end: str
    bar_count: int
    content_hash: str
    source: str
    created_at: str
    bars: tuple[dict, ...]
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.symbol:
            raise SnapshotError("a snapshot needs a symbol")
        if self.bar_count != len(self.bars):
            raise SnapshotError(
                f"bar_count {self.bar_count} does not match {len(self.bars)} stored bars"
            )
        if len(self.content_hash) != 64:
            raise SnapshotError("content_hash must be a 64-character SHA-256 hex digest")
        if self.bar_count == 0:
            raise SnapshotError(
                "a snapshot with no bars is not a snapshot; refuse it rather than "
                "record an empty window"
            )
        if self.window_end != str(self.bars[-1]["date"]):
            raise SnapshotError(
                f"window_end {self.window_end!r} does not match the last bar "
                f"{self.bars[-1]['date']!r}"
            )
        if self.window_start != str(self.bars[0]["date"]):
            raise SnapshotError(
                f"window_start {self.window_start!r} does not match the first bar "
                f"{self.bars[0]['date']!r}"
            )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        symbol: str,
        frame: pd.DataFrame,
        *,
        source: str,
        created_at: str | None = None,
    ) -> PriceSnapshot:
        """Capture ``frame`` (dates ascending) as a snapshot of ``symbol``.

        Refuses an empty frame: an empty window is the defect this module exists
        to prevent, so it is an error here rather than a silent ``unavailable``.
        """
        if frame is None or len(frame) == 0:
            raise SnapshotError(f"no bars to snapshot for {symbol!r}")
        for column in _BAR_COLUMNS[:4]:
            if column not in frame.columns:
                raise SnapshotError(f"frame for {symbol!r} is missing the {column!r} column")
        ordered = frame.sort_index()
        bars: list[dict] = []
        for index, row in ordered.iterrows():
            for column in _BAR_COLUMNS[:4]:
                value = float(row[column])
                if value != value or value in (float("inf"), float("-inf")):
                    raise SnapshotError(
                        f"bar {str(index)[:10]} for {symbol!r} has a non-finite {column}"
                    )
                if value <= 0.0:
                    raise SnapshotError(
                        f"bar {str(index)[:10]} for {symbol!r} has a non-positive {column}"
                    )
            volume = row.get("volume")
            bars.append(
                {
                    "date": str(index)[:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": None if pd.isna(volume) else float(volume),
                }
            )
        content_hash = window_hash(ordered)
        window_start, window_end = bars[0]["date"], bars[-1]["date"]
        snapshot_id = f"{symbol.lower()}-{window_start}..{window_end}-{content_hash[:12]}"
        return cls(
            snapshot_id=snapshot_id,
            symbol=symbol,
            window_start=window_start,
            window_end=window_end,
            bar_count=len(bars),
            content_hash=content_hash,
            source=source,
            created_at=created_at or datetime.now(UTC).isoformat(),
            bars=tuple(bars),
        )

    # -- verification -----------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Rebuild the exact frame, so a replay can re-read and re-hash it."""
        index = pd.to_datetime([bar["date"] for bar in self.bars])
        data = {
            column: [bar.get(column) for bar in self.bars] for column in _BAR_COLUMNS
        }
        return pd.DataFrame(data, index=index)

    def recompute_hash(self) -> str:
        return window_hash(self.to_frame())

    def verify(self) -> None:
        """Raise unless the stored bars still hash to ``content_hash``."""
        actual = self.recompute_hash()
        if actual != self.content_hash:
            raise SnapshotError(
                f"snapshot {self.snapshot_id} content does not match its hash: stored "
                f"{self.content_hash[:12]}…, recomputed {actual[:12]}…"
            )

    def matches_record(self, record: DecisionRecord) -> bool:
        """Whether this snapshot is the exact window ``record`` was captured on."""
        return (
            record.ticker == self.symbol
            and record.bar_count == self.bar_count
            and record.data_window_hash == self.content_hash
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "bar_count": self.bar_count,
            "content_hash": self.content_hash,
            "source": self.source,
            "created_at": self.created_at,
            "hash_method": "sha256(replay.window_hash) over date + open/high/low/close @10dp",
            "bars": [dict(bar) for bar in self.bars],
        }
        return assert_secret_free(payload)

    @classmethod
    def from_dict(cls, payload: object) -> PriceSnapshot:
        if not isinstance(payload, dict):
            raise SnapshotError("a price snapshot must be a mapping")
        version = payload.get("schema_version", SNAPSHOT_SCHEMA_VERSION)
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError(
                f"unsupported snapshot schema_version {version!r}; this build reads "
                f"version {SNAPSHOT_SCHEMA_VERSION}"
            )
        required = {
            "snapshot_id",
            "symbol",
            "window_start",
            "window_end",
            "bar_count",
            "content_hash",
            "source",
            "created_at",
            "bars",
        }
        missing = required - set(payload)
        if missing:
            raise SnapshotError(f"snapshot is missing {sorted(missing)}")
        raw_bars = payload["bars"]
        if not isinstance(raw_bars, list):
            raise SnapshotError("snapshot bars must be a list")
        snapshot = cls(
            snapshot_id=payload["snapshot_id"],
            symbol=payload["symbol"],
            window_start=payload["window_start"],
            window_end=payload["window_end"],
            bar_count=payload["bar_count"],
            content_hash=payload["content_hash"],
            source=payload["source"],
            created_at=payload["created_at"],
            bars=tuple(dict(bar) for bar in raw_bars),
        )
        # A loaded snapshot is verified on read, so a tampered file fails here
        # rather than silently seeding a replay with the wrong bars.
        snapshot.verify()
        return snapshot

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        assert_secret_free(text)
        target.write_text(text + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> PriceSnapshot:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SnapshotError(f"no price snapshot at {source}") from exc
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"{source} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)


def snapshot_for_session(
    feed,
    symbol: str,
    session: str,
    *,
    source: str,
    created_at: str | None = None,
) -> PriceSnapshot:
    """Snapshot ``symbol``'s bars up to ``session`` from a point-in-time feed.

    Uses the feed's own slice (``history``), so no bar after ``session`` can
    enter the snapshot — the same guarantee the record's window relies on.
    """
    if feed is None:
        raise SnapshotError("no feed to snapshot from")
    if symbol not in feed.symbols:
        raise SnapshotError(f"feed does not carry {symbol!r}")
    frame = feed.history(symbol, session)
    if frame is None or len(frame) == 0:
        raise SnapshotError(
            f"the feed has no bars for {symbol!r} at or before {session}; a record "
            "captured here would not be reproducible"
        )
    if str(frame.index[-1])[:10] != session:
        raise SnapshotError(
            f"the feed's last bar for {symbol!r} is {str(frame.index[-1])[:10]}, not "
            f"{session}; the window must end at the decision date"
        )
    return PriceSnapshot.from_frame(symbol, frame, source=source, created_at=created_at)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from tradingagents.baseline.data import load_feed

    from .protocol import DEFAULT_PROTOCOL

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a reproducible price snapshot from the frozen dataset. "
            "Offline: reads the committed CSVs, contacts no vendor."
        )
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--session", default=None, help="window end YYYY-MM-DD (default: last frozen session)")
    parser.add_argument("--out", default=None, help="path to write the snapshot JSON")
    parser.add_argument("--verify", default=None, help="path to a snapshot to verify instead")
    args = parser.parse_args(argv)

    if args.verify:
        snapshot = PriceSnapshot.load(args.verify)
        print(f"OK {snapshot.snapshot_id} ({snapshot.bar_count} bars, {snapshot.source})")
        return 0

    feed = load_feed(DEFAULT_PROTOCOL)
    session = args.session or feed.calendar()[-1]
    snapshot = snapshot_for_session(
        feed, args.symbol, session, source=f"frozen_baseline:{args.symbol}.csv"
    )
    print(
        f"{snapshot.snapshot_id} bars={snapshot.bar_count} "
        f"window={snapshot.window_start}..{snapshot.window_end} hash={snapshot.content_hash}"
    )
    if args.out:
        print(f"wrote {snapshot.save(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
