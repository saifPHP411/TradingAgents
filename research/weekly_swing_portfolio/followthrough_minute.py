"""Replay the best development week and the separate block at one-minute resolution."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from data import SYMBOLS
from followthrough import RULES, prepare
from minute_check import fetch
from weekly import episode


def run(phase, months, warmup, weeks):
    jobs = product(SYMBOLS, ("klines", "markPriceKlines"), months)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, jobs))
    coarse_index, coarse, scores = prepare([*warmup, *months])
    ix = pd.date_range(
        pd.Period(months[0]).start_time,
        (pd.Period(months[-1]) + 1).start_time,
        freq="min",
        inclusive="left",
        tz="UTC",
    )
    subset = slice(coarse_index.get_loc(ix[0]), coarse_index.get_loc(ix[-15]) + 1)
    tapes = []
    for k, symbol in enumerate(SYMBOLS):
        parts = []
        for kind in ("klines", "markPriceKlines"):
            frame = pd.concat([pd.read_csv(fetch((symbol, kind, m))) for m in months])
            observed = pd.DatetimeIndex(
                pd.to_datetime(frame.open_time, unit="ms", utc=True)
            )
            if not observed.equals(ix):
                raise ValueError(f"Incomplete minute data for {symbol} {kind}")
            parts.append(frame[["open", "high", "low", "close"]])
        funding = np.zeros(len(ix))
        funding[::15] = coarse[k, subset, 8]
        tapes.append(np.column_stack([*parts, funding]))
    tape = np.array(tapes)
    strategies = [(0, 50)] if phase == "development" else [(0, 50), (1, 25)]
    for week, stress, (model, leverage) in product(weeks, (False, True), strategies):
        result = episode(
            tape,
            scores[:, subset],
            ix.get_loc(week),
            model,
            leverage,
            stress,
            scale=15,
            stop_atr=1.0,
            reward_risk=1.25,
        )
        yield {
            "phase": phase,
            "rule": RULES[model],
            "leverage": leverage,
            "week": str(week.date()),
            "stress": stress,
            **result,
        }


if __name__ == "__main__":
    development = list(
        run(
            "development",
            ("2025-04",),
            ("2025-03",),
            [pd.Timestamp("2025-04-21", tz="UTC")],
        )
    )
    evaluation = list(
        run(
            "evaluation",
            ("2025-08", "2025-09", "2025-10"),
            ("2025-07",),
            pd.date_range("2025-08-04", periods=12, freq="7D", tz="UTC"),
        )
    )
    pd.DataFrame(development + evaluation).to_csv(
        "followthrough-minute.csv", index=False
    )
