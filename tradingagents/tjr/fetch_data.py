"""Fetch real intraday bars for a TJR run. Network, opt-in, run by a human.

The only part of the TJR package that touches the network. It goes through the data
layer (``dataflows.vendors.yahoo.intraday.load_intraday_range``), because only the
data layer may import a vendor library — a rule ``tests/test_layering.py`` enforces.

Yahoo's intraday limits are real and short: 1-minute bars for the last 30 days, and
5-minute bars for the last 60. A TJR study is therefore over a few recent weeks, not
years. The fetch respects those windows and refuses one it cannot serve rather than
returning a shorter frame that would look complete.

Usage (maintainer, once per study window):

    python -m tradingagents.tjr.fetch_data --symbols AAPL MSFT --days 20

The frames are written as CSV under ``--out`` (default ``tradingagents/tjr/data``),
one file per symbol and interval, matching the baseline dataset's format. Nothing is
committed; the point of a fetch is that the data is the vendor's, not the repository's.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.intraday import load_intraday_range

DEFAULT_SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA", "AMD")
DEFAULT_OUT = Path(__file__).resolve().parent / "data"
INTERVALS: tuple[str, ...] = ("1m", "5m")


def fetch(
    symbols: tuple[str, ...],
    *,
    days: int,
    out_dir: Path,
    intervals: tuple[str, ...] = INTERVALS,
    now: dt.date | None = None,
) -> dict[str, Path]:
    """Download ``intervals`` for each symbol over the last ``days`` and write them.

    Returns a mapping of ``"{symbol}-{interval}"`` to the file written. Raises rather
    than writing a partial set: a window the vendor will not serve is reported, and
    the caller sees a failure instead of a study built on missing data.
    """
    reference = now or dt.datetime.now(dt.UTC).date()
    start = (reference - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    end = reference.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol in symbols:
        for interval in intervals:
            frame = load_intraday_range(symbol, start, end, interval=interval, now=reference)
            path = out_dir / f"{symbol.upper()}-{interval}.csv"
            frame.to_csv(path)
            written[f"{symbol.upper()}-{interval}"] = path
    return written


def load_frames(data_dir: Path, symbols: tuple[str, ...], interval: str) -> dict[str, pd.DataFrame]:
    """Read the frames ``fetch`` wrote, for one interval. Missing files are refused."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = data_dir / f"{symbol.upper()}-{interval}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"no {interval} frame for {symbol} at {path}; run the fetch first"
            )
        frames[symbol.upper()] = pd.read_csv(path, index_col=0, parse_dates=True)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch intraday bars for a TJR run.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=20, help="calendar days back from today")
    parser.add_argument("--intervals", nargs="+", default=list(INTERVALS), choices=list(INTERVALS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    written = fetch(
        tuple(args.symbols), days=args.days, out_dir=args.out, intervals=tuple(args.intervals)
    )
    for key, path in sorted(written.items()):
        rows = len(pd.read_csv(path))
        print(f"{key}: {rows} rows -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a maintainer entry point
    raise SystemExit(main())
