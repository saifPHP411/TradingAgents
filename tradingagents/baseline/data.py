"""Load the frozen baseline dataset into a ``MarketDataFeed``. Offline.

Reads the committed CSVs, never the network. The feed the backtest runs on is
therefore byte-identical from one machine to the next, which is what makes the
baseline reproducible without a data vendor being up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradingagents.backtesting import MarketDataFeed

from .protocol import DATA_DIR, BaselineProtocol


class BaselineDataError(ValueError):
    """The frozen dataset is missing or unreadable."""


def load_frames(protocol: BaselineProtocol, *, data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """The frozen OHLCV frames for the protocol's universe, indexed by date."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in protocol.universe:
        path = base / f"{symbol}.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        frames[symbol] = frame
    if missing:
        raise BaselineDataError(
            f"frozen data missing for {missing}; run "
            "'python -m tradingagents.baseline.fetch_data' to create it"
        )
    return frames


def load_feed(protocol: BaselineProtocol, *, data_dir: Path | None = None) -> MarketDataFeed:
    """The frozen dataset as a point-in-time ``MarketDataFeed``."""
    return MarketDataFeed(load_frames(protocol, data_dir=data_dir))
