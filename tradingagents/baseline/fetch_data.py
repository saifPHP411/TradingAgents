"""Fetch and freeze the official baseline dataset. Network, run once by a human.

This is the *only* part of the baseline package that touches the network, and it
is deliberately not called at test time or at run time. It downloads the
protocol's universe over the protocol's period, validates it against the same
rules the backtest feed applies, and writes one deterministic CSV per symbol to
``tradingagents/baseline/data/``. The download itself goes through the data
layer (``dataflows/vendors/yahoo/ohlcv.load_ohlcv_range``), because only the data
layer may import a vendor library — a rule ``tests/test_layering.py`` enforces.

The frozen files are committed, so every later run is offline and byte-identical.
Re-fetching will change the data (split adjustments and dividend adjustments are
restated), which is exactly why the frozen copy, not the vendor, is the baseline's
source of truth. A re-fetch is a *new* baseline version, not an update of this one.

Usage (maintainer, once):

    python -m tradingagents.baseline.fetch_data

It refuses to overwrite an existing freeze unless ``--force`` is given, so an
accidental re-fetch cannot silently move the baseline under a comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# The frozen universe and period. Kept here rather than imported from the
# protocol so the fetch is self-contained and does not need the strategy stack.
UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD")
BENCHMARK = "SPY"
START = "2015-01-01"
END = "2024-12-30"

_REQUIRED = ("open", "high", "low", "close", "volume")


def _normalize(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, date-indexed, ordered OHLCV, dropping rows the feed would drop.

    Mirrors ``MarketDataFeed._normalize`` closely enough that a fetched file loads
    without the feed having to discard anything: the point of freezing is that the
    committed data is exactly what the backtest used.
    """
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    for column in _REQUIRED:
        if column not in data.columns:
            raise ValueError(f"{symbol}: missing {column!r} column after download")
    index = pd.to_datetime(data.index, errors="coerce")
    if index.isna().any():
        raise ValueError(f"{symbol}: unparseable date in the download")
    data.index = index.normalize()
    data.index.name = "date"
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data[list(_REQUIRED)].astype(float)
    # Drop rows that would not survive the feed's own validation, so the frozen
    # file is loadable as-is rather than needing a filter at read time.
    finite = data.notna().all(axis=1)
    positive = (data[["open", "high", "low", "close"]] > 0).all(axis=1)
    ordered = (data["high"] >= data["low"]) & data["close"].between(
        data["low"] - 1e-9, data["high"] + 1e-9
    ) & data["open"].between(data["low"] - 1e-9, data["high"] + 1e-9)
    return data[finite & positive & ordered]


def fetch(*, force: bool = False) -> list[Path]:
    """Download the universe and write one frozen CSV per symbol."""
    # The vendor call goes through the data layer, which is the only layer that
    # may import a vendor library (tests/test_layering.py enforces this).
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv_range

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for symbol in UNIVERSE:
        target = DATA_DIR / f"{symbol}.csv"
        if target.exists() and not force:
            raise FileExistsError(
                f"{target} already exists; pass force=True to re-fetch and replace it"
            )
        raw = load_ohlcv_range(symbol, START, END)
        if raw is None or raw.empty:
            raise RuntimeError(f"no data returned for {symbol}")
        frame = _normalize(symbol, raw)
        frame.to_csv(target, float_format="%.6f")
        written.append(target)
        print(f"{symbol}: {len(frame)} rows  {frame.index[0].date()} .. {frame.index[-1].date()}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing freeze")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
