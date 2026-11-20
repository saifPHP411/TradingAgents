"""Test predefined market-regime and relative-strength gates on unseen periods."""

import json
from itertools import product

import numpy as np
import pandas as pd
from relative_weekly import prepare
from weekly import STEP, episode
from wide_universe import FILTERS

PERIODS = (
    ("development", ("2023-10",), "2023-10-09", 3),
    ("holdout", ("2023-12",), "2023-12-11", 3),
    ("replication", ("2023-08", "2023-09"), "2023-08-14", 6),
)
GATES = ("baseline", "strong_signal", "market_confirmation", "both")


def build_gates(index, tape, scores, btc_index):
    # These closes become available at the right edge of the hourly bar.
    btc = pd.Series(tape[btc_index, :, 3], index=index).resample("h", label="right").last()
    fast = btc.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = btc.ewm(span=48, adjust=False, min_periods=48).mean()
    direction = np.sign(fast - slow).shift(freq="15min").reindex(index).fillna(0).to_numpy()

    signals = scores[:, :, 0]
    breadth = (signals > 0).sum(axis=0) - (signals < 0).sum(axis=0)
    aligns = (np.sign(signals) == direction) & (np.sign(signals) * breadth >= 3)
    strong = np.abs(signals) >= 2.5

    for name, allowed in zip(GATES, (True, strong, aligns, aligns & strong), strict=True):
        screened = scores.copy()
        screened[:, :, 0] = np.where(allowed, signals, 0)
        yield name, screened


def settings():
    filters = json.loads(FILTERS.read_text(encoding="utf-8"))
    symbols = tuple(filters)
    specs = {
        name: (
            STEP.get(name, step),
            {"BTCUSDT": 100, "ETHUSDT": 20}.get(name, minimum),
            0.005,
            50,
            float("inf"),
        )
        for name, (step, minimum) in filters.items()
    }
    return symbols, specs


def main():
    symbols, specs = settings()
    rows = []
    for phase, months, anchor, weeks in PERIODS:
        index, tape, scores = prepare(months, symbols)
        starts = pd.date_range(anchor, periods=weeks, freq="7D", tz="UTC")
        for gate, screened in build_gates(index, tape, scores, symbols.index("BTCUSDT")):
            for start, lev, stress in product(starts, (2, 10, 25, 50), (False, True)):
                result = episode(
                    tape,
                    screened,
                    index.get_loc(start),
                    0,
                    lev,
                    stress,
                    stop_atr=1.5,
                    reward_risk=1.5,
                    max_hold_bars=96,
                    specs=specs,
                    symbols=symbols,
                )
                rows.append(
                    {
                        "phase": phase,
                        "week": str(start.date()),
                        "gate": gate,
                        "leverage": lev,
                        "stress": stress,
                        **result,
                        "joint": result["goal"]
                        and bool(result["trades"])
                        and 4 * result["wins"] >= 3 * result["trades"],
                    }
                )
    pd.DataFrame(rows).to_csv("gated-momentum.csv", index=False)


if __name__ == "__main__":
    main()
