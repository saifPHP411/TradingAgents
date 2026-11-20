"""Check the frozen September 2023 configuration without fitting any parameters."""

from itertools import product

import pandas as pd
from gated_momentum import build_gates, settings
from relative_weekly import prepare
from weekly import episode

PERIODS = (
    (("2024-08", "2024-09"), "2024-08-05", 8),
    (("2024-10", "2024-11"), "2024-10-14", 6),
    (("2026-09", "2026-10"), "2026-09-14", 6),
)


def main():
    symbols, specs = settings()
    rows = []
    for months, anchor, weeks in PERIODS:
        index, tape, scores = prepare(months, symbols)
        selected = dict(build_gates(index, tape, scores, symbols.index("BTCUSDT")))[
            "strong_signal"
        ]
        starts = pd.date_range(anchor, periods=weeks, freq="7D", tz="UTC")
        for week, stress in product(starts, (False, True)):
            outcome = episode(
                tape,
                selected,
                index.get_loc(week),
                0,
                50,
                stress,
                specs=specs,
                stop_atr=1.5,
                reward_risk=1.5,
                max_hold_bars=96,
                symbols=symbols,
            )
            rows.append(
                {
                    "week": week.date().isoformat(),
                    "stress": stress,
                    **outcome,
                    "joint": outcome["goal"]
                    and outcome["trades"] > 0
                    and 4 * outcome["wins"] >= 3 * outcome["trades"],
                }
            )
    pd.DataFrame(rows).to_csv("binance-confirmation.csv", index=False)


if __name__ == "__main__":
    main()
