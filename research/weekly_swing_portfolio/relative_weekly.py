"""Test hourly asset returns relative to the trailing Bitcoin beta."""

import numpy as np
import pandas as pd
from data import SYMBOLS
from flow_weekly import replay_grid
from weekly import load

RULES = ("relative_momentum", "relative_reversal")
PERIODS = (
    ("development", ("2024-04", "2024-05", "2024-06"), "2024-05-06"),
    ("evaluation", ("2024-10", "2024-11"), "2024-10-14"),
)


def scores_for_asset(trade, benchmark, tradable):
    candles = trade.resample("h", label="right").agg(
        {"high": "max", "low": "min", "close": "last"}
    )
    close = candles.close
    move = close.pct_change()
    beta = (
        move.rolling(120).cov(benchmark).shift() / benchmark.rolling(120).var().shift()
    )
    residual = move - beta * benchmark
    dispersion = residual.rolling(120).std().shift() * np.sqrt(3.0)
    score = residual.rolling(3).sum() / dispersion
    atr = (
        pd.concat(
            [
                candles.high - candles.low,
                (candles.high - close.shift()).abs(),
                (candles.low - close.shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(24)
        .mean()
    )
    momentum = score.where(
        tradable
        & (score.abs() >= 1.5)
        & (np.sign(score) == np.sign(close.pct_change(3))),
        0,
    )
    opposite = -score.shift()
    reversal = opposite.where(
        tradable & (opposite.abs() >= 2.0) & (opposite * residual > 0), 0
    )
    return pd.DataFrame(
        dict(zip((*RULES, "atr"), (momentum, reversal, atr), strict=True))
    ).shift(freq="15min")


def prepare(months, symbols=SYMBOLS):
    tape = []
    scores = []
    btc = load("BTCUSDT", "klines", months)
    benchmark = btc.close.resample("h", label="right").last().pct_change()
    for symbol in symbols:
        trade = btc if symbol == "BTCUSDT" else load(symbol, "klines", months)
        mark = load(symbol, "markPriceKlines", months)
        funding = load(symbol, "fundingRate", months)
        if funding.index[0] > trade.index[0] + pd.Timedelta(hours=8) or funding.index[
            -1
        ] < trade.index[-1] - pd.Timedelta(hours=8):
            raise ValueError(f"Incomplete funding coverage for {symbol}")
        signals = scores_for_asset(trade, benchmark, symbol != "BTCUSDT")
        scores.append(
            signals.reindex(trade.index)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .to_numpy()
        )
        tape.append(
            np.column_stack([trade, mark, funding.reindex(trade.index, fill_value=0)])
        )
    return btc.index, np.array(tape), np.array(scores)


if __name__ == "__main__":
    replay_grid(PERIODS, RULES, prepare).to_csv("relative-weekly.csv", index=False)
