"""Cross-venue replication of the selected historical breakout and two later weeks."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from weekly import SYMBOLS, episode, features

BASE = "https://api.bybitglobal.com/v5/market/"
CACHE = Path("public-bybit-cache")
CACHE.mkdir(exist_ok=True)


def request(path, **params):
    url = BASE + path + "?" + urlencode({"category": "linear", **params})
    key = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if key.exists():
        return json.loads(key.read_text())
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                data = json.load(response)
            if data["retCode"] != 0:
                raise ValueError(data["retMsg"])
            rows = data["result"]["list"]
            if not rows:
                raise ValueError(f"empty series {path} {params}")
            key.write_text(json.dumps(rows))
            return rows
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def candles(symbol, path, minutes, index, pool):
    block = 900

    def part(i):
        span = index[i : i + block]
        return request(
            path,
            symbol=symbol,
            interval=minutes,
            start=span[0].value // 10**6,
            end=span[-1].value // 10**6 + minutes * 60000 - 1,
            limit=len(span),
        )

    rows = [
        r[:5] for chunk in pool.map(part, range(0, len(index), block)) for r in chunk
    ]
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.pop("time").astype("int64"), unit="ms", utc=True)
    frame = frame.sort_index().astype(float)
    if not frame.index.equals(index):
        raise ValueError(f"incomplete {symbol} {path}")
    return frame


def run(start, weeks, pool):
    end = start + pd.Timedelta(weeks=weeks)
    ix = pd.date_range(start, end, freq="min", inclusive="left")
    four = pd.date_range(
        start - pd.Timedelta(days=12), end, freq="4h", inclusive="left"
    )
    bars = []
    scores = []
    specs = {}
    for symbol in SYMBOLS:
        instr = request("instruments-info", symbol=symbol)[0]
        tier = min(request("risk-limit", symbol=symbol), key=lambda t: int(t["id"]))
        specs[symbol] = (
            float(instr["lotSizeFilter"]["qtyStep"]),
            float(instr["lotSizeFilter"]["minNotionalValue"]),
            float(tier["maintenanceMargin"]),
            min(
                float(tier["maxLeverage"]),
                float(instr["leverageFilter"]["maxLeverage"]),
            ),
            float(tier["riskLimitValue"]),
        )
        h = candles(symbol, "kline", 240, four, pool)
        h.index += pd.Timedelta(hours=4)
        score = features(h).reindex(ix[::15]).fillna(0).to_numpy()
        trade = candles(symbol, "kline", 1, ix, pool)
        mark = candles(symbol, "mark-price-kline", 1, ix, pool)
        rates = pd.Series(0.0, index=ix)
        history = request(
            "funding/history",
            symbol=symbol,
            startTime=start.value // 10**6,
            endTime=end.value // 10**6 - 1,
            limit=200,
        )
        stamps = pd.DatetimeIndex(
            pd.to_datetime(
                [int(row["fundingRateTimestamp"]) for row in history],
                unit="ms",
                utc=True,
            )
        ).sort_values()
        if (
            stamps.has_duplicates
            or max(np.diff(stamps.asi8)) > pd.Timedelta(hours=8).value
            or stamps[0] - start > pd.Timedelta(hours=8)
            or end - stamps[-1] > pd.Timedelta(hours=8)
        ):
            raise ValueError("funding gap")
        for row in history:
            rates.loc[
                pd.to_datetime(int(row["fundingRateTimestamp"]), unit="ms", utc=True)
            ] = float(row["fundingRate"])
        bars.append(np.column_stack([trade, mark, rates]))
        scores.append(score)
    tape, signals = np.array(bars), np.array(scores)
    rows = []
    for w in range(weeks):
        for stress in (False, True):
            audit = []
            result = episode(
                tape,
                signals,
                w * 7 * 1440,
                1,
                50,
                stress,
                scale=15,
                specs=specs,
                audit=audit,
            )
            if start == pd.Timestamp("2026-08-17", tz="UTC"):
                for order in audit:
                    order["entry"] = ix[order["entry"]].isoformat()
                    order["exit"] = ix[order["exit"]].isoformat()
                filename = (
                    "bybit-august-stress.json" if stress else "bybit-august-base.json"
                )
                Path(filename).write_text(json.dumps(audit, indent=2) + "\n")
            rows.append(
                {
                    "week": str((start + pd.Timedelta(weeks=w)).date()),
                    "stress": stress,
                    **result,
                }
            )
    return rows, specs


if __name__ == "__main__":
    values = []
    with ThreadPoolExecutor(12) as pool:
        for start, weeks in [("2026-08-17", 1), ("2026-11-02", 2)]:
            rows, specs = run(pd.Timestamp(start, tz="UTC"), weeks, pool)
            values.extend(rows)
            pd.DataFrame(values).to_csv("bybit-check.csv", index=False)
            Path("bybit-instrument-assumptions.json").write_text(
                json.dumps(specs, indent=2) + "\n"
            )
