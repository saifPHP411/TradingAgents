"""Fetch the daily Bybit linear depth-200 archives into the working directory."""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from time import sleep
from urllib.request import urlopen
from zipfile import ZipFile


def download(symbol, day):
    filename = Path(f"{day}.zip")
    if filename.exists():
        with ZipFile(filename) as z:
            if z.namelist() == [f"{day}_{symbol}_ob200.data"]:
                return
        raise ValueError("Use a separate working directory for each contract")
    url = f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob200.data.zip"
    temp = filename.with_suffix(".partial")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=90) as response, temp.open("wb") as f:
                shutil.copyfileobj(response, f)
            with ZipFile(temp) as z:
                if z.namelist() != [f"{day}_{symbol}_ob200.data"]:
                    raise ValueError(f"Unexpected archive for {symbol} on {day}")
            temp.replace(filename)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            sleep(attempt + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", choices=["LTCUSDT", "XRPUSDT"])
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("days", type=int, default=28, nargs="?")
    args = parser.parse_args()
    dates = [args.start + timedelta(days=i) for i in range(args.days)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda day: download(args.symbol, day), dates))
