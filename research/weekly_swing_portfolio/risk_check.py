"""Replay a 1% planned stop-risk budget on the frozen breakout signal."""

import pandas as pd
from weekly import START, WEEKS, episode, setup

if __name__ == "__main__":
    index, tape, signals = setup()
    rows = []
    for week in range(WEEKS):
        start = START + pd.Timedelta(weeks=week)
        for stress in (False, True):
            result = episode(
                tape, signals, index.get_loc(start), 1, 2, stress, risk_budget=0.01
            )
            rows.append({"week": str(start.date()), "stress": stress, **result})
    pd.DataFrame(rows).to_csv("risk-check.csv", index=False)
