"""Historical crypto bars, cached on disk so a study is not a re-download.

A backtest reads months of 1m/5m/15m/1h/4h bars for ten symbols, which is a lot of
requests. Re-fetching them on every run is slow, rate-limit prone and
non-reproducible — a study must run against *the same bars* every time, or its
results cannot be compared. So this module fetches once, validates, and caches to
disk in a stable format keyed by symbol, interval and the requested window.

Two properties matter and are structural:

- **Live data and historical data are separate.** The cache is only ever written
  by an explicit fetch and only ever read by a backtest; the live view
  (:mod:`tradingagents.crypto.engine`) never reads it and never writes it. A
  backtest that silently used a live bar, or a live view that served a cached one,
  would be a correctness bug the dashboard could not show.
- **A cached window is honest about what it holds.** :meth:`HistoryCache.coverage`
  reports the first and last bar time actually present, so a caller can say "the
  venue served 180 days, not 365" instead of assuming the window it asked for was
  the window it got.

The cache stores newline-delimited JSON (one bar per line) rather than a binary
format, so it is inspectable, diff-able and needs no optional dependency.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..bybit import BybitPublicClient
from ..symbols import parse_pair
from ..transport import CryptoDataError
from ..types import CANDLE_INTERVALS, Candle, candles_to_frame

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path(
    os.environ.get("TRADINGAGENTS_CRYPTO_CACHE_DIR") or "~/.tradingagents/crypto/bybit"
).expanduser()


def _bar_key(symbol: str, interval: str) -> str:
    pair = parse_pair(symbol)
    return f"{pair.base}-{pair.quote}__{interval}".replace("/", "-")


@dataclass(frozen=True)
class Coverage:
    """What a cached window actually holds."""

    symbol: str
    interval: str
    bars: int
    first: str | None
    last: str | None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "bars": self.bars,
            "first": self.first,
            "last": self.last,
        }


class HistoryCache:
    """A disk cache of validated historical bars, keyed by symbol and interval."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, interval: str) -> Path:
        return self.root / f"{_bar_key(symbol, interval)}.jsonl"

    # -- read -------------------------------------------------------------

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """The cached frame for ``symbol``/``interval``, or ``None`` if not cached."""
        path = self.path_for(symbol, interval)
        if not path.exists():
            return None
        rows: list[Candle] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    candle = _candle_from_record(symbol, interval, record)
                    if candle is not None:
                        rows.append(candle)
        except OSError:
            return None
        if not rows:
            return None
        return candles_to_frame(rows)

    def coverage(self, symbol: str, interval: str) -> Coverage | None:
        frame = self.load(symbol, interval)
        if frame is None or frame.empty:
            return None
        return Coverage(
            symbol=parse_pair(symbol).canonical,
            interval=interval,
            bars=len(frame),
            first=str(frame.index[0]),
            last=str(frame.index[-1]),
        )

    # -- write ------------------------------------------------------------

    def store(self, symbol: str, interval: str, candles: Iterable[Candle]) -> int:
        """Merge ``candles`` into the cache, de-duplicated by bar start.

        Merging rather than replacing means a study can page a window in several
        fetches and keep every bar it saw. A bar already present is left as it is
        (the venue's own value), so a re-fetch cannot restate a bar a run used.
        """
        path = self.path_for(symbol, interval)
        existing: dict[str, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and record.get("open_time"):
                        existing[record["open_time"]] = record
        for candle in candles:
            key = candle.open_time.isoformat()
            existing.setdefault(key, _record_from_candle(candle))
        ordered = [existing[key] for key in sorted(existing)]
        self._atomic_write(path, ordered)
        return len(ordered)

    def _atomic_write(self, path: Path, records: list[dict]) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".crypto-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "kind": "header"}) + "\n"
                )
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def _record_from_candle(candle: Candle) -> dict:
    return {
        "open_time": candle.open_time.isoformat(),
        "close_time": candle.close_time.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


def _candle_from_record(symbol: str, interval: str, record: dict) -> Candle | None:
    if record.get("kind") == "header":
        return None
    try:
        return Candle(
            symbol=parse_pair(symbol).canonical,
            interval=interval,
            open_time=datetime.fromisoformat(record["open_time"]),
            close_time=datetime.fromisoformat(record["close_time"]),
            open=float(record["open"]),
            high=float(record["high"]),
            low=float(record["low"]),
            close=float(record["close"]),
            volume=float(record.get("volume") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def fetch_history(
    symbols: Iterable[str],
    *,
    intervals: Iterable[str] = ("1m", "5m"),
    limits: Mapping[str, int] | None = None,
    limit: int = 1500,
    client: BybitPublicClient | None = None,
    cache: HistoryCache | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch (or read from cache) bars for every symbol/interval.

    Returns ``{symbol: {interval: frame}}``. ``limits`` overrides the bar count per
    interval (a 1-minute window is short, a 5-minute one is not, so the two want
    different depths); ``limit`` is the fallback. A symbol/interval already cached is
    read from disk unless ``refresh`` is set, so a second study over the same window
    is a disk read. A fetch failure for one symbol/interval is raised rather than
    hidden, so a study never silently runs on a partial universe.
    """
    client = client or BybitPublicClient()
    cache = cache or HistoryCache()
    per_interval = dict(limits or {})
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for raw in symbols:
        symbol = parse_pair(raw).canonical
        out[symbol] = {}
        for interval in intervals:
            if interval not in CANDLE_INTERVALS:
                raise CryptoDataError(f"unsupported interval {interval!r}")
            frame = None if refresh else cache.load(symbol, interval)
            wanted = per_interval.get(interval, limit)
            if frame is not None and not frame.empty and len(frame) >= wanted:
                out[symbol][interval] = frame
                continue
            candles = client.klines(symbol, interval, limit=wanted)
            if not candles:
                if frame is not None and not frame.empty:
                    out[symbol][interval] = frame
                    continue
                raise CryptoDataError(f"the venue returned no {interval} bars for {symbol}")
            cache.store(symbol, interval, candles)
            out[symbol][interval] = candles_to_frame(candles)
    return out


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_DIR",
    "Coverage",
    "HistoryCache",
    "fetch_history",
]
