"""Independent checks for the first hour of decoded quotes and the oracle."""

import json
from itertools import islice
from zipfile import ZipFile

import numpy as np
import pandas as pd

from study import COLS, oracle


def check_quotes(day):
    grouped = []
    previous = None
    synced = False
    with ZipFile(day + ".zip") as z, z.open(z.namelist()[0]) as f:
        for line in f:
            event = json.loads(line)
            data = event["data"]
            if event["type"] == "snapshot":
                bids, asks = {}, {}
                synced = True
            elif data["u"] != previous + 1:
                synced = False
            previous = data["u"]
            if not synced:
                continue
            for key, side in (("a", asks), ("b", bids)):
                for raw_p, raw_q in data[key]:
                    price, size = float(raw_p), float(raw_q)
                    if size:
                        side[price] = size
                    else:
                        side.pop(price, None)
            if grouped and grouped[-1][0] // 5000 == event["ts"] // 5000:
                continue
            best_bid, best_ask = max(bids), min(asks)
            grouped.append(
                (
                    event["ts"],
                    best_bid,
                    best_ask,
                    bids[best_bid],
                    asks[best_ask],
                    sum(bids[p] for p in islice(sorted(bids, reverse=True), 5)),
                    sum(asks[p] for p in islice(sorted(asks), 5)),
                )
            )
            if len(grouped) == 720:
                break
    stored = pd.read_csv(day + ".csv", names=COLS, nrows=720)
    np.testing.assert_allclose(grouped, stored.to_numpy(), rtol=0, atol=0.000001)


def check_oracle():
    bid = np.array([100.0, 99.0, 103.0, 89.0, 95.0])
    ask = bid + 0.1
    times = np.arange(5) * 300000

    def best(i, cash):
        if i >= len(bid) - 1:
            return cash
        paths = [best(i + 1, cash)]
        for j in range(i + 1, len(bid)):
            if times[j] - times[i] > 900000:
                continue
            for side in (1, -1):
                entry, exit_price = (ask[i], bid[j]) if side == 1 else (bid[i], ask[j])
                units = 0.2 * cash / entry
                pnl = units * (
                    side * (exit_price - entry) - 0.00055 * (exit_price + entry)
                )
                paths.append(best(j, cash + pnl))
        return max(paths)

    np.testing.assert_allclose(oracle(bid, ask, times), best(0, 50), rtol=1e-12)


if __name__ == "__main__":
    import sys

    check_oracle()
    check_quotes(sys.argv[1])
    print(
        "720 sampled quotes matched and optimal balance matched exhaustive enumeration."
    )
