"""Weekly portfolio episodes with one signal-selected USD-M perpetual position."""

import math

import numpy as np
import pandas as pd

from data import MONTHS, SYMBOLS, archive

START = pd.Timestamp("2025-11-03", tz="UTC")
WEEKS = 51
DEV = 35
STEP = dict(zip(SYMBOLS, (0.001, 0.001, 0.1, 1, 1), strict=True))
NAMES = ("trend_rotation", "channel_break", "range_reversion")


def load(symbol, kind, months):
    frames = [pd.read_csv(archive(symbol, kind, m)) for m in months]
    if kind == "markPriceKlines" and "2026-06" in months:
        # The monthly archive omits this day; use the actual daily mark candles.
        daily = pd.read_csv(archive(symbol, kind, "2026-06-29"))
        frames.append(daily.loc[~daily.open_time.isin(pd.concat(frames).open_time)])
    df = pd.concat(frames)
    key = "calc_time" if kind == "fundingRate" else "open_time"
    df.index = pd.to_datetime(df[key].astype("int64"), unit="ms", utc=True).dt.floor(
        "15min"
    )
    df = df.sort_index()
    if not df.index.is_unique:
        raise ValueError(f"duplicate timestamps {symbol} {kind}")
    if kind == "fundingRate":
        if df.index.to_series().diff().max() > pd.Timedelta(hours=8, minutes=15):
            raise ValueError(f"funding gap {symbol}")
        return df.last_funding_rate
    expected = pd.date_range(
        pd.Period(months[0]).start_time,
        (pd.Period(months[-1]) + 1).start_time,
        freq="15min",
        inclusive="left",
        tz="UTC",
    )
    if not df.index.equals(expected):
        raise ValueError(f"incomplete candles {symbol} {kind}")
    return df[["open", "high", "low", "close"]]


def features(hourly):
    close = hourly.close
    atr = (
        pd.concat(
            [
                hourly.high - hourly.low,
                (hourly.high - close.shift()).abs(),
                (hourly.low - close.shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )
    slow = close.ewm(span=42, adjust=False, min_periods=42).mean()
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    std = close.rolling(24).std()
    z = (close - close.rolling(24).mean()) / std
    strength = close.diff(6) / atr
    aligned = np.sign(strength) == np.sign(fast - slow)
    trend = strength.where(aligned & (strength.abs() >= 1), 0)
    upper = hourly.high.shift().rolling(12).max()
    lower = hourly.low.shift().rolling(12).min()
    breakout = ((close - upper) / atr).clip(lower=0) - ((lower - close) / atr).clip(
        lower=0
    )
    reversion = (-z).where((z.abs() >= 2) & ((fast - slow).abs() < 1.5 * atr), 0)
    scores = pd.DataFrame(dict(zip(NAMES, (trend, breakout, reversion), strict=True)))
    scores["atr"] = atr
    # Entry at the following 15-minute open for processing/latency.
    scores.index += pd.Timedelta(minutes=15)
    return scores


def setup(months=MONTHS):
    tapes = []
    entries = []
    for symbol in SYMBOLS:
        trade, mark, funding = (
            load(symbol, k, months)
            for k in ("klines", "markPriceKlines", "fundingRate")
        )
        if funding.index[0] > trade.index[0] + pd.Timedelta(hours=8) or funding.index[
            -1
        ] < trade.index[-1] - pd.Timedelta(hours=8):
            raise ValueError("incomplete funding boundaries")
        hourly = trade.resample("4h", label="right").agg(
            {"high": "max", "low": "min", "close": "last"}
        )
        scores = features(hourly)
        entries.append(scores.reindex(trade.index).fillna(0).to_numpy())
        tapes.append(
            np.column_stack([trade, mark, funding.reindex(trade.index, fill_value=0)])
        )
    return trade.index, np.array(tapes), np.array(entries)


def episode(
    tape,
    scores,
    first,
    model,
    leverage,
    stress,
    scale=1,
    audit=None,
    specs=None,
    risk_budget=None,
    stop_atr=None,
    reward_risk=None,
    max_hold_bars=None,
):
    end = first + 7 * 96 * scale
    hold_bars = (
        max_hold_bars if max_hold_bars is not None else (96 if model == 2 else 192)
    )
    balance = peak = 50.0
    dd = 0.0
    slot = None
    count = won = liqs = 0
    positive = negative = 0.0
    funding_total = 0.0
    slip = 0.001 if stress else 0.0003
    fee = 0.00055
    stop_factor = stop_atr if stop_atr is not None else (1.25 if model == 2 else 1.75)
    target_factor = (
        reward_risk if reward_risk is not None else (1.5 if model == 2 else 2.5)
    )
    for i in range(first, end):
        if slot is None and i % scale == 0:
            ranks = np.argsort(-np.abs(scores[:, i // scale, model]), kind="stable")
            for a in ranks:
                score = scores[a, i // scale, model]
                if not score:
                    break
                side = 1 if score > 0 else -1
                price = tape[a, i, 0] * (1 + side * slip)
                qty_step, min_value, mm_rate, max_lev, tier_cap = (
                    specs[SYMBOLS[a]]
                    if specs
                    else (
                        STEP[SYMBOLS[a]],
                        100 if a == 0 else 20 if a == 1 else 5,
                        0.005,
                        50,
                        float("inf"),
                    )
                )
                if leverage > max_lev:
                    continue
                units = balance * 0.50 * leverage / price
                stop_distance = scores[a, i // scale, -1] * stop_factor
                if risk_budget is not None:
                    units = min(
                        units,
                        balance
                        * risk_budget
                        / (stop_distance + price * 2 * (fee + slip)),
                    )
                amount = math.floor(units / qty_step) * qty_step
                if (
                    amount * price < min_value
                    or amount * price > tier_cap
                    or amount <= 0
                ):
                    continue
                target = price + side * target_factor * stop_distance
                goal_exit = (
                    (side * price + fee * price + (1000 - balance) / amount)
                    / (side - fee)
                    / (1 - side * slip)
                )
                target = min(target, goal_exit) if side == 1 else max(target, goal_exit)
                slot = (a, side, price, amount, price - side * stop_distance, target, i)
                funding = 0.0
                break
        if slot is None:
            continue
        a, side, price, qty, stop, target, opened = slot
        op, hi, lo, cl, mo, mh, ml, mc, rate = tape[a, i]
        if i > opened:
            funding -= side * qty * mo * rate
        move_bad = lo if side == 1 else hi
        move_good = hi if side == 1 else lo
        bad_mark = ml if side == 1 else mh
        margin = qty * price / leverage
        mm = max(0.01, mm_rate) if stress else mm_rate
        liq = margin + funding + side * qty * (bad_mark - price) <= qty * bad_mark * (
            mm + fee
        )
        if liq:
            net = -margin - qty * price * fee + min(funding, 0)
            liqs += 1
        else:
            if side * (move_bad - stop) <= 0:
                fill = min(op, stop) if side == 1 else max(op, stop)
            elif side * (move_good - target) >= 0:
                fill = target
            elif i == end - 1 or i - opened + 1 >= hold_bars * scale:
                fill = cl
            else:
                marked = (
                    balance + funding + qty * (side * (mc - price) - fee * (price + mc))
                )
                peak = max(peak, marked)
                dd = max(dd, 1 - marked / peak)
                continue
            fill *= 1 - side * slip
            net = qty * (side * (fill - price) - fee * (price + fill)) + funding
        if audit is not None:
            audit.append(
                {
                    "entry": int(opened),
                    "exit": int(i),
                    "symbol": SYMBOLS[a],
                    "side": int(side),
                    "qty": float(qty),
                    "entry_price": float(price),
                    "liquidated": bool(liq),
                    "net": float(net),
                    "balance": float(balance + net),
                }
            )
        balance += net
        funding_total += funding
        count += 1
        won += net > 0
        positive += max(0, net)
        negative += max(0, -net)
        peak = max(peak, balance)
        dd = max(dd, 1 - balance / peak)
        slot = None
        if balance >= 1000 - 1e-8:
            break
    return {
        "end": balance,
        "peak": peak,
        "drawdown": dd,
        "trades": count,
        "wins": int(won),
        "liquidations": liqs,
        "goal": balance >= 1000 - 1e-8,
        "funding": funding_total,
        "gross_profit": positive,
        "gross_loss": negative,
    }


def main():
    index, tape, scores = setup()
    rows = []
    for w in range(WEEKS):
        t = START + pd.Timedelta(weeks=w)
        for m, name in enumerate(NAMES):
            for lev in [2, 10, 25, 50]:
                for stress in [False, True]:
                    result = episode(tape, scores, index.get_loc(t), m, lev, stress)
                    rows.append(
                        dict(
                            week=str(t.date()),
                            phase="development" if w < DEV else "evaluation",
                            strategy=name,
                            leverage=lev,
                            stress=stress,
                            **result,
                        )
                    )
    df = pd.DataFrame(rows)
    df.to_csv("swing-weekly.csv", index=False)
    dev = df[(df.phase == "development") & (~df.stress)]
    medians = (
        dev.groupby(["strategy", "leverage"]).end.median().sort_values(ascending=False)
    )
    print("development selection", medians.index[0], medians.iloc[0])
    for stress in (False, True):
        test = df[(df.phase == "evaluation") & (df.stress == stress)]
        print(
            stress,
            test.groupby(["strategy", "leverage"])
            .agg(
                mean=("end", "mean"),
                median=("end", "median"),
                best=("end", "max"),
                worst=("end", "min"),
                hits=("goal", "sum"),
                wins=("end", lambda x: (x > 50).sum()),
                liqs=("liquidations", "sum"),
            )
            .round(2)
            .to_string(),
        )
    print("maximum", df.loc[df.end.idxmax()].to_json())
    print("all hits", int(df.goal.sum()))


if __name__ == "__main__":
    main()
