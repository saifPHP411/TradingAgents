"""Replay the evaluation block with one-minute trade and mark prices."""

import argparse
import concurrent.futures
import itertools
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from data import MONTHS as COARSE_MONTHS, ROOT, SYMBOLS, archive
from weekly import DEV, NAMES, START, WEEKS, episode, setup

MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def fetch(job):
    sym, kind, month = job
    name = f"{sym}-1m-{month}.zip"
    path = ROOT / kind / name
    if not path.exists():
        url = f"https://data.binance.vision/data/futures/um/monthly/{kind}/{sym}/1m/{name}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=60) as response:
                    payload = response.read()
                tmp = path.with_suffix(".partial")
                tmp.write_bytes(payload)
                with zipfile.ZipFile(tmp) as archive:
                    assert archive.testzip() is None
                tmp.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication", action="store_true")
    other = parser.parse_args().replication
    months = ("2025-05", "2025-06") if other else MONTHS
    coarse_months = ("2025-04", *months) if other else COARSE_MONTHS
    if other:
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            list(
                pool.map(
                    lambda a: archive(*a),
                    itertools.product(
                        SYMBOLS,
                        ("klines", "markPriceKlines", "fundingRate"),
                        coarse_months,
                    ),
                )
            )
    jobs = itertools.product(SYMBOLS, ("klines", "markPriceKlines"), months)
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        list(pool.map(fetch, jobs))
    original, coarse, scores = setup(coarse_months)
    ix = pd.date_range(
        pd.Period(months[0]).start_time,
        (pd.Period(months[-1]) + 1).start_time,
        freq="min",
        inclusive="left",
        tz="UTC",
    )
    first = original.get_loc(ix[0])
    scores = scores[:, first : first + len(ix) // 15]
    tapes = []
    for k, sym in enumerate(SYMBOLS):
        parts = []
        for kind in ("klines", "markPriceKlines"):
            frames = [pd.read_csv(fetch((sym, kind, m))) for m in months]
            frame = pd.concat(frames, ignore_index=True)
            frame.index = pd.DatetimeIndex(
                pd.to_datetime(frame.open_time, unit="ms", utc=True)
            )
            if not frame.index.equals(ix):
                raise ValueError(f"incomplete {sym} {kind}")
            parts.append(frame[["open", "high", "low", "close"]].to_numpy())
        rates = np.zeros(len(ix))
        rates[::15] = coarse[k, first : first + len(ix) // 15, 8]
        tapes.append(np.column_stack([*parts, rates]))
    tape = np.array(tapes)
    rows = []
    starts = (
        pd.date_range("2025-05-05", periods=8, freq="7D", tz="UTC")
        if other
        else [START + pd.Timedelta(weeks=w) for w in range(DEV, WEEKS)]
    )
    for start in starts:
        for stress in (False, True):
            log = []
            result = episode(
                tape,
                scores,
                ix.get_loc(start),
                NAMES.index("channel_break"),
                50,
                stress,
                scale=15,
                audit=log,
            )
            rows.append({"week": str(start.date()), "stress": stress, **result})
            if start == pd.Timestamp("2026-08-17", tz="UTC"):
                for row in log:
                    row["entry"] = ix[row["entry"]].isoformat()
                    row["exit"] = ix[row["exit"]].isoformat()
                Path(f"august-trades-{'stress' if stress else 'base'}.json").write_text(
                    json.dumps(log, indent=2) + "\n"
                )
    pd.DataFrame(rows).to_csv(
        "replication.csv" if other else "minute-check.csv", index=False
    )


if __name__ == "__main__":
    main()
