"""Compare the original market list with 24 contracts on matching dates."""

import json
from pathlib import Path

import flow_weekly
import numpy as np
import pandas as pd
import relative_weekly
from data import SYMBOLS
from weekly import STEP, episode

FILTERS = Path(__file__).with_name("wide-bybit-filters.json")
CONFIGURATIONS = (
    (flow_weekly, ("2024-08", "2024-09"), "2024-08-05", 8, 50, 2.5, 6),
    (relative_weekly, ("2024-10", "2024-11"), "2024-10-14", 6, 10, 1.5, 24),
)


def main():
    verified_filters = json.loads(FILTERS.read_text(encoding="utf-8"))
    extended = (*SYMBOLS, *(name for name in verified_filters if name not in SYMBOLS))
    specs = {
        name: (
            STEP.get(name, step),
            {"BTCUSDT": 100, "ETHUSDT": 20}.get(name, minimum),
            0.005,
            50,
            float("inf"),
        )
        for name, (step, minimum) in verified_filters.items()
    }
    reports = []
    coverage = []
    for module, months, anchor, count, leverage, target, hours in CONFIGURATIONS:
        index, tape, scores = module.prepare(months, extended)
        starts = pd.date_range(anchor, periods=count, freq="7D", tz="UTC")
        for universe in (SYMBOLS, extended):
            size = len(universe)
            coverage_start = index.get_loc(starts[0])
            coverage_end = index.get_loc(starts[-1]) + 7 * 96
            has_signal = np.any(scores[:size, coverage_start:coverage_end, 0], axis=1)
            traded = set()
            for start in starts:
                for stress in (False, True):
                    audit = []
                    outcome = episode(
                        tape[:size],
                        scores[:size],
                        index.get_loc(start),
                        0,
                        leverage,
                        stress,
                        audit=audit,
                        specs=specs,
                        stop_atr=1.5,
                        reward_risk=target,
                        max_hold_bars=hours * 4,
                        symbols=universe,
                    )
                    traded.update(trade["symbol"] for trade in audit)
                    reports.append(
                        {
                            "strategy": module.RULES[0],
                            "symbols": size,
                            "start": start.date().isoformat(),
                            "stress": stress,
                            "joint": outcome["goal"]
                            and bool(outcome["trades"])
                            and 4 * outcome["wins"] >= 3 * outcome["trades"],
                            **outcome,
                        }
                    )
            coverage.append(
                {
                    "strategy": module.RULES[0],
                    "scanned": list(universe),
                    "signaled": [
                        name for name, present in zip(universe, has_signal, strict=True) if present
                    ],
                    "traded": sorted(traded),
                }
            )
    pd.DataFrame(reports).to_csv("wide-universe-weekly.csv", index=False)
    Path("wide-universe-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")


if __name__ == "__main__":
    main()
