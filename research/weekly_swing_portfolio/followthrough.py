"""Compare two fixed four-hour continuation signals in distinct calendar blocks."""

from itertools import product

import numpy as np
import pandas as pd

from data import SYMBOLS
from weekly import episode, features, load

RULES = ("pullback_reclaim", "volatility_expansion")
PERIODS = (
    ("development", "2024-11", "2025-04", "2024-12-02", "2025-04-28"),
    ("evaluation", "2025-07", "2025-10", "2025-08-04", "2025-10-27"),
)


def signals(bars):
    close = bars.close
    bias = np.sign(
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=42, adjust=False).mean()
    )
    reference = close.ewm(span=6, adjust=False).mean().shift()
    crosses_up = (close > reference) & (close.shift() <= reference.shift())
    crosses_down = (close < reference) & (close.shift() >= reference.shift())
    fade_then_continue = (crosses_up & (bias == 1)).astype(int) - (
        crosses_down & (bias == -1)
    ).astype(int)
    base = features(bars)
    atr = base.atr.shift(freq="-15min")
    distance = (close - reference).abs() / atr
    range_high = bars.high.shift().rolling(6).max()
    range_low = bars.low.shift().rolling(6).min()
    broken_level = (close > range_high).astype(int) - (close < range_low).astype(int)
    recent_range = range_high - range_low
    expansion = broken_level.where(
        (recent_range < 2.5 * atr.shift()) & (bars.high - bars.low > atr.shift()), 0
    )
    frame = pd.DataFrame(
        {
            "pullback_reclaim": fade_then_continue * distance,
            "volatility_expansion": expansion * distance,
            "atr": atr,
        }
    )
    return frame.shift(freq="15min")


def prepare(months):
    tapes = []
    scores = []
    for symbol in SYMBOLS:
        price, mark, funding = (
            load(symbol, kind, months)
            for kind in ("klines", "markPriceKlines", "fundingRate")
        )
        if funding.index[0] > price.index[0] + pd.Timedelta(hours=8) or funding.index[
            -1
        ] < price.index[-1] - pd.Timedelta(hours=8):
            raise ValueError(f"Missing funding coverage for {symbol}")
        bars = price.resample("4h", label="right").agg(
            {"high": "max", "low": "min", "close": "last"}
        )
        scores.append(signals(bars).reindex(price.index).fillna(0).to_numpy())
        tapes.append(
            np.column_stack([price, mark, funding.reindex(price.index, fill_value=0)])
        )
    return price.index, np.array(tapes), np.array(scores)


def main():
    rows = []
    for phase, low, high, start, end in PERIODS:
        months = pd.period_range(low, high, freq="M").astype(str)
        index, tape, scores = prepare(months)
        weeks = pd.date_range(start, end, freq="7D", tz="UTC")
        weeks = [
            week
            for week in weeks
            if week + pd.Timedelta(days=7) <= index[-1] + pd.Timedelta(minutes=15)
        ]
        for week in weeks:
            for model, lev, reward, stress in product(
                range(2), (10, 25, 50), (0.6, 1.25, 2.5), (False, True)
            ):
                result = episode(
                    tape,
                    scores,
                    index.get_loc(week),
                    model,
                    lev,
                    stress,
                    stop_atr=1.0,
                    reward_risk=reward,
                )
                joint = (
                    result["goal"]
                    and result["trades"] > 0
                    and result["wins"] * 4 >= result["trades"] * 3
                )
                rows.append(
                    {
                        "phase": phase,
                        "week": str(week.date()),
                        "rule": RULES[model],
                        "leverage": lev,
                        "reward_risk": reward,
                        "stress": stress,
                        "joint": joint,
                        **result,
                    }
                )
    pd.DataFrame(rows).to_csv("followthrough.csv", index=False)


if __name__ == "__main__":
    main()
