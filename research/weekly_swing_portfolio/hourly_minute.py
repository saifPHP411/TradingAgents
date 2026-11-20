"""Check the historical weekly maxima using one-minute trade and mark candles."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

import flow_weekly
import relative_weekly
from data import SYMBOLS
from minute_check import fetch
from weekly import episode


def check(module, date, month, leverage, ratio, hours):
    period = next(
        months for phase, months, _ in module.PERIODS if phase == "evaluation"
    )
    coarse_index, coarse, scores = module.prepare(period)
    start = pd.Timestamp(date, tz="UTC")
    minute_index = pd.date_range(start, periods=7 * 1440, freq="min")
    coarse_rows = slice(
        coarse_index.get_loc(start), coarse_index.get_loc(start) + 7 * 96
    )
    tapes = []
    for k, symbol in enumerate(SYMBOLS):
        columns = []
        for kind in ("klines", "markPriceKlines"):
            frame = pd.read_csv(fetch((symbol, kind, month)))
            frame.index = pd.DatetimeIndex(
                pd.to_datetime(frame.open_time, unit="ms", utc=True)
            )
            frame = frame.loc[minute_index[0] : minute_index[-1]]
            if not frame.index.equals(minute_index):
                raise ValueError(f"Incomplete minute data for {symbol} {kind}")
            columns.append(frame[["open", "high", "low", "close"]])
        funding = np.zeros(len(minute_index))
        funding[::15] = coarse[k, coarse_rows, 8]
        tapes.append(np.column_stack([*columns, funding]))
    tape = np.array(tapes)
    for stress in (False, True):
        result = episode(
            tape,
            scores[:, coarse_rows],
            0,
            0,
            leverage,
            stress,
            scale=15,
            stop_atr=1.5,
            reward_risk=ratio,
            max_hold_bars=hours * 4,
        )
        yield {
            "rule": module.RULES[0],
            "week": date,
            "leverage": leverage,
            "stress": stress,
            **result,
        }


if __name__ == "__main__":
    jobs = product(SYMBOLS, ("klines", "markPriceKlines"), ("2024-08", "2024-11"))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, jobs))
    rows = [
        *check(flow_weekly, "2024-08-05", "2024-08", 50, 2.5, 6),
        *check(relative_weekly, "2024-11-11", "2024-11", 10, 1.5, 24),
    ]
    pd.DataFrame(rows).to_csv("hourly-minute.csv", index=False)
