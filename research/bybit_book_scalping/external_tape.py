"""A released one-minute Binance perpetual close as a reference price."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


def reference_signal(book, symbol):
    weeks = pd.to_datetime(book.ts, unit="ms", utc=True)
    dates = weeks.dt.date.unique()
    parts = []
    for month in sorted({str(day)[:7] for day in dates}):
        if month < "2026-11":
            files = [("monthly", month)]
        else:
            files = [("daily", str(d)) for d in dates if str(d).startswith(month)]
        for granularity, stamp in files:
            name = f"{symbol}-1m-{stamp}.zip"
            path = Path(name)
            if not path.exists():
                url = f"https://data.binance.vision/data/futures/um/{granularity}/klines/{symbol}/1m/{name}"
                with urlopen(url, timeout=60) as response:
                    payload = response.read()
                with ZipFile(BytesIO(payload)) as archive:
                    assert len(archive.namelist()) == 1
                path.write_bytes(payload)
            with (
                ZipFile(path) as archive,
                archive.open(archive.namelist()[0]) as source,
            ):
                frame = pd.read_csv(source, usecols=["open_time", "close"])
            parts.append(frame)
    tape = pd.concat(parts).sort_values("open_time").set_index("open_time").close
    assert tape.index.is_unique
    # Observe only once, at the first sample at least 5 seconds past the minute.
    minute = book.ts // 60000 * 60000
    observed = book.loc[(book.ts - minute >= 5000) & (book.ts - minute < 10000)].copy()
    observed.index = minute.loc[observed.index]
    close = tape.reindex(observed.index - 60000).to_numpy()
    mid = (observed.bid + observed.ask) / 2
    basis = close / mid - 1
    center = basis.rolling(60, min_periods=60).median().shift()
    delta = basis - center
    allowed = (observed.ask - observed.bid) / mid < 0.0005
    signals = np.sign(delta).where(allowed & (delta.abs() >= 0.0025), 0)
    return (
        book.ts.map(dict(zip(observed.ts, signals, strict=True)))
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
