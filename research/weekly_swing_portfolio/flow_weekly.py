"""Causal one-hour trade-flow signals on USD-M perpetuals."""

from itertools import product

import numpy as np
import pandas as pd
from data import SYMBOLS, archive
from weekly import episode, load

RULES = ("aggressive_flow", "reversal_after_absorption")
PERIODS = (
    ("development", ("2024-01", "2024-02", "2024-03"), "2024-02-05"),
    ("evaluation", ("2024-07", "2024-08", "2024-09"), "2024-08-05"),
)


def hourly_signals(trade, volume):
    candles = trade.resample("1h", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    volume = volume.resample("1h", label="right").sum()
    whole = volume.quote_volume
    imbalance = 2 * volume.taker_buy_quote_volume / whole - 1
    activity = whole / whole.shift().rolling(24).median()
    close = candles.close
    true_range = pd.concat(
        [
            candles.high - candles.low,
            (candles.high - close.shift()).abs(),
            (candles.low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(24).mean()
    change = (close - candles.open) / atr
    drive = imbalance.where(
        (imbalance.abs() >= 0.18)
        & (activity >= 1.5)
        & (np.sign(imbalance) == np.sign(change))
        & (change.abs() >= 0.6),
        0,
    )
    reversal = -imbalance.shift()
    absorbed = reversal.where(
        (reversal.abs() >= 0.20)
        & (activity.shift() >= 1.5)
        & (reversal * change.shift() >= 0)
        & (np.sign(reversal) == np.sign(change))
        & (change.abs() >= 0.35),
        0,
    )
    return pd.DataFrame(
        dict(zip((*RULES, "atr"), (drive, absorbed, atr), strict=True))
    ).shift(freq="15min")


def prepare(months, symbols=SYMBOLS):
    tape = []
    signals = []
    for symbol in symbols:
        trade, mark, funding = (
            load(symbol, kind, months)
            for kind in ("klines", "markPriceKlines", "fundingRate")
        )
        flow = pd.concat(
            [
                pd.read_csv(
                    archive(symbol, "klines", month),
                    usecols=["open_time", "quote_volume", "taker_buy_quote_volume"],
                )
                for month in months
            ]
        )
        flow.index = pd.DatetimeIndex(
            pd.to_datetime(flow.pop("open_time"), unit="ms", utc=True)
        )
        if not flow.index.equals(trade.index):
            raise ValueError(f"Trade flow timestamps mismatch for {symbol}")
        if funding.index[0] > trade.index[0] + pd.Timedelta(hours=8) or funding.index[
            -1
        ] < trade.index[-1] - pd.Timedelta(hours=8):
            raise ValueError(f"Incomplete funding coverage for {symbol}")
        if (flow.taker_buy_quote_volume < 0).any() or (
            flow.taker_buy_quote_volume > flow.quote_volume
        ).any():
            raise ValueError(f"Unexpected trade volume for {symbol}")
        signals.append(
            hourly_signals(trade, flow).reindex(trade.index).fillna(0).to_numpy()
        )
        tape.append(
            np.column_stack([trade, mark, funding.reindex(trade.index, fill_value=0)])
        )
    return trade.index, np.array(tape), np.array(signals)


def replay_grid(periods, rules, prepare_data):
    results = []
    for phase, months, anchor in periods:
        index, tape, scores = prepare_data(months)
        weeks = pd.date_range(
            pd.Timestamp(anchor, tz="UTC"),
            index[-1] - pd.Timedelta(days=7, minutes=-15),
            freq="7D",
            tz="UTC",
        )
        for start, rule, leverage, target, hours, stress in product(
            weeks,
            range(len(rules)),
            (10, 25, 50),
            (0.8, 1.5, 2.5),
            (6, 24),
            (False, True),
        ):
            result = episode(
                tape,
                scores,
                index.get_loc(start),
                rule,
                leverage,
                stress,
                stop_atr=1.5,
                reward_risk=target,
                max_hold_bars=hours * 4,
            )
            results.append(
                {
                    "phase": phase,
                    "week": str(start.date()),
                    "signal": rules[rule],
                    "leverage": leverage,
                    "target": target,
                    "hours": hours,
                    "stress": stress,
                    "joint": result["goal"]
                    and result["trades"] > 0
                    and 4 * result["wins"] >= 3 * result["trades"],
                    **result,
                }
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    replay_grid(PERIODS, RULES, prepare).to_csv("flow-weekly.csv", index=False)
