"""Cross-venue check of the selected week using public Bybit trade and mark data."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from bybit_check import candles, request
from gated_momentum import build_gates, settings
from relative_weekly import scores_for_asset
from weekly import episode

START = pd.Timestamp("2023-09-11", tz="UTC")
STOP = START + pd.Timedelta(days=7)
INDEX = pd.date_range(START - pd.Timedelta(days=13), STOP, freq="15min", inclusive="left")


def fetch_symbol(symbol, pool):
    trade = candles(symbol, "kline", 15, INDEX, pool)
    mark = candles(symbol, "mark-price-kline", 15, INDEX, pool)
    history = request(
        "funding/history",
        symbol=symbol,
        startTime=INDEX[0].value // 10**6,
        endTime=STOP.value // 10**6 - 1,
        limit=200,
    )
    funding = pd.Series(
        {
            pd.to_datetime(int(row["fundingRateTimestamp"]), unit="ms", utc=True): float(
                row["fundingRate"]
            )
            for row in history
        }
    ).sort_index()
    if (
        len(funding) != len(history)
        or not funding.index.isin(INDEX).all()
        or funding.index[0] - INDEX[0] > pd.Timedelta(hours=8)
        or STOP - funding.index[-1] > pd.Timedelta(hours=8)
        or funding.index.to_series().diff().max() > pd.Timedelta(hours=8)
    ):
        raise ValueError(f"Incomplete Bybit funding history for {symbol}")
    instrument = request("instruments-info", symbol=symbol)[0]
    tier = min(request("risk-limit", symbol=symbol), key=lambda item: int(item["id"]))
    limits = instrument["lotSizeFilter"]
    spec = (
        float(limits["qtyStep"]),
        float(limits["minNotionalValue"]),
        float(tier["maintenanceMargin"]),
        min(float(tier["maxLeverage"]), float(instrument["leverageFilter"]["maxLeverage"])),
        float(tier["riskLimitValue"]),
    )
    return trade, mark, funding, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute", action="store_true")
    minute = parser.parse_args().minute
    symbols, _ = settings()
    with ThreadPoolExecutor(12) as markets, ThreadPoolExecutor(18) as requests:
        feeds = dict(
            zip(symbols, markets.map(lambda s: fetch_symbol(s, requests), symbols), strict=True)
        )
    benchmark = feeds["BTCUSDT"][0].close.resample("h", label="right").last().pct_change()
    tape = []
    scores = []
    specs = {}
    for symbol, (trade, mark, funding, spec) in feeds.items():
        tape.append(np.column_stack([trade, mark, funding.reindex(INDEX, fill_value=0)]))
        scores.append(
            scores_for_asset(trade, benchmark, symbol != "BTCUSDT")
            .reindex(INDEX)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .to_numpy()
        )
        specs[symbol] = spec
    tape = np.array(tape)
    screened = dict(build_gates(INDEX, tape, np.array(scores), symbols.index("BTCUSDT")))[
        "strong_signal"
    ]
    result_index = INDEX
    scale = 1
    output = "gated-bybit"
    if minute:
        result_index = pd.date_range(START, STOP, freq="min", inclusive="left")
        screened = screened[:, INDEX.get_loc(START) :]
        scale = 15
        output += "-minute"

        def minute_tape(symbol, pool):
            trade = candles(symbol, "kline", 1, result_index, pool)
            mark = candles(symbol, "mark-price-kline", 1, result_index, pool)
            funding = feeds[symbol][2].reindex(result_index, fill_value=0)
            return np.column_stack([trade, mark, funding])

        with ThreadPoolExecutor(12) as markets, ThreadPoolExecutor(22) as requests:
            tape = np.array(list(markets.map(lambda s: minute_tape(s, requests), symbols)))

    results = []
    for stress in (False, True):
        audit = []
        result = episode(
            tape,
            screened,
            result_index.get_loc(START),
            0,
            50,
            stress,
            scale=scale,
            audit=audit,
            specs=specs,
            stop_atr=1.5,
            reward_risk=1.5,
            max_hold_bars=96,
            symbols=symbols,
        )
        results.append({"week": START.date().isoformat(), "stress": stress, **result})
        for order in audit:
            order["entry"] = result_index[order["entry"]].isoformat()
            order["exit"] = result_index[order["exit"]].isoformat()
        scope = "stress" if stress else "base"
        Path(f"{output}-{scope}.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(results).to_csv(f"{output}.csv", index=False)


if __name__ == "__main__":
    main()
