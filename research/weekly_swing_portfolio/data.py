"""Download historical USD-M perpetual trade candles, mark candles, and funding."""

import concurrent.futures
import itertools
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
MONTHS = pd.period_range("2025-10", "2026-10", freq="M").astype(str)
ROOT = Path("data")


def archive(symbol, kind, month):
    if kind == "fundingRate":
        name = f"{symbol}-fundingRate-{month}.zip"
        tail = f"{kind}/{symbol}/{name}"
    else:
        name = f"{symbol}-15m-{month}.zip"
        tail = f"{kind}/{symbol}/15m/{name}"
    path = ROOT / kind / name
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        granularity = "daily" if len(str(month)) == 10 else "monthly"
        url = f"https://data.binance.vision/data/futures/um/{granularity}/{tail}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=45) as response:
                    data = response.read()
                temp = path.with_suffix(".partial")
                temp.write_bytes(data)
                with zipfile.ZipFile(temp) as z:
                    assert z.testzip() is None
                temp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
    return path


if __name__ == "__main__":
    jobs = list(
        itertools.product(SYMBOLS, ("klines", "markPriceKlines", "fundingRate"), MONTHS)
    )
    with concurrent.futures.ThreadPoolExecutor(18) as pool:
        list(pool.map(lambda args: archive(*args), jobs))
    print(f"Verified availability of {len(jobs)} archives")
