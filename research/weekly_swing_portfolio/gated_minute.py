"""Audit the selected 50x gate on the September weeks with minute execution data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gated_momentum import build_gates, settings
from minute_check import fetch
from relative_weekly import prepare
from weekly import episode


def main():
    symbols, specs = settings()
    index, coarse, scores = prepare(("2023-08", "2023-09"), symbols)
    ix = pd.date_range("2023-09-01", "2023-10-01", freq="min", inclusive="left", tz="UTC")
    begin = index.get_loc(ix[0])
    rows = slice(begin, begin + len(ix) // 15)
    selected = dict(build_gates(index, coarse, scores, symbols.index("BTCUSDT")))["strong_signal"]
    tapes = []
    for asset, symbol in enumerate(symbols):
        columns = []
        for kind in ("klines", "markPriceKlines"):
            frame = pd.read_csv(fetch((symbol, kind, "2023-09")))
            found = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit="ms", utc=True))
            if not found.equals(ix):
                raise ValueError(f"Incomplete minute data: {symbol} {kind}")
            columns.append(frame[["open", "high", "low", "close"]])
        funding = np.zeros(len(ix))
        funding[::15] = coarse[asset, rows, 8]
        tapes.append(np.column_stack([*columns, funding]))
    tape = np.array(tapes)
    results = []
    for start in pd.date_range("2023-09-04", periods=3, freq="7D", tz="UTC"):
        for stress in (False, True):
            audit = []
            outcome = episode(
                tape,
                selected[:, rows],
                ix.get_loc(start),
                0,
                50,
                stress,
                scale=15,
                audit=audit,
                specs=specs,
                stop_atr=1.5,
                reward_risk=1.5,
                max_hold_bars=96,
                symbols=symbols,
            )
            results.append({"week": start.date().isoformat(), "stress": stress, **outcome})
            if start.day == 11:
                for order in audit:
                    order["entry"] = ix[order["entry"]].isoformat()
                    order["exit"] = ix[order["exit"]].isoformat()
                scope = "stress" if stress else "base"
                Path(f"gated-september-{scope}.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(results).to_csv("gated-minute.csv", index=False)


if __name__ == "__main__":
    main()
