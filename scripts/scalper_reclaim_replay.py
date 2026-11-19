"""Offline signal comparison using public USD-M minute candles and mark opens."""
import io
import json
import urllib.request
import zipfile

import pandas as pd

from tradingagents.crypto.scalping import ScalpingConfig, _indicators

CFG = ScalpingConfig()
DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2026-11-01", tz="UTC")
SPLIT = pd.Timestamp("2026-11-10", tz="UTC")
END = pd.Timestamp("2026-11-18", tz="UTC")


def candles(symbol, kind):
    frames = []
    for day in pd.date_range(START - DAY, END - DAY):
        name = f"{symbol}-1m-{day:%Y-%m-%d}.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/{kind}/{symbol}/1m/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open(archive.namelist()[0]) as source:
            data = pd.read_csv(source, header=None, usecols=range(6))
        data.columns = ["time", "open", "high", "low", "close", "volume"]
        data = data[pd.to_numeric(data.time, errors="coerce").notna()].astype(float)
        data.index = pd.to_datetime(data.pop("time"), unit="ms", utc=True)
        frames.append(data)
    frame = pd.concat(frames).sort_index()
    expected = pd.date_range(START - DAY, END, freq="min", inclusive="left")
    if not frame.index.equals(expected):
        raise ValueError(f"Incomplete {symbol} {kind} minute history")
    return frame


def signals(price, mark):
    one = _indicators(price, CFG.atr_period).shift()
    result = pd.DataFrame(index=price.index)
    for minutes, fast, slow in [(5, "ema9", "ema21"), (15, "ema20", "ema50")]:
        bars = price.resample(f"{minutes}min", label="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        bars = _indicators(bars, CFG.atr_period)
        up = (bars.close > bars[fast]) & (bars[fast] > bars[slow])
        down = (bars.close < bars[fast]) & (bars[fast] < bars[slow])
        result[minutes] = (up.astype(int) - down.astype(int)).reindex(price.index, method="ffill")
    trend = result[5].where(result[5] == result[15], 0)
    ema = one.ema9
    vol = one.volume.shift().rolling(20).mean()
    long = ((trend == 1) & (one.low <= ema * 1.0015) & (one.close <= ema * 1.002)
            & (mark > ema) & (mark > one.close) & one.rsi.between(45, 72) & (one.rsi.diff() >= 0))
    short = ((trend == -1) & (one.high >= ema * 0.9985) & (one.close >= ema * 0.998)
             & (mark < ema) & (mark < one.close) & one.rsi.between(28, 55) & (one.rsi.diff() <= 0))
    distance = (one.atr * CFG.stop_atr_multiple).clip(lower=mark * CFG.min_stop_fraction)
    eligible = ((one.volume >= vol * 0.5) & (distance <= mark * CFG.max_stop_fraction)
                & (distance * CFG.reward_risk >= mark * (2 * CFG.cost_per_fill + CFG.min_net_edge_fraction)))
    result["signal"] = (long.astype(int) - short.astype(int)).where(eligible, 0)
    result["distance"] = distance
    result["cross"] = ((one.close - ema) * result.signal <= 0)
    result["near"] = (mark - ema).abs() <= one.atr * 0.25
    return result


def replay(price, mark, inputs, variant, start, end):
    selected = inputs.signal.ne(0)
    if variant != "baseline":
        selected &= inputs[variant]
    available = start
    pnl = []
    for t, row in inputs[selected].loc[start:end].iterrows():
        expiry = t + pd.Timedelta(minutes=CFG.max_hold_minutes)
        if t < available or expiry >= end:
            continue
        side = row.signal
        entry = price.at[t, "open"]
        stop = mark.at[t, "open"] - side * row.distance
        target = mark.at[t, "open"] + side * CFG.reward_risk * row.distance
        exit_time = expiry
        fill = price.at[expiry, "open"]
        for stamp, bar in mark.loc[t:expiry].iloc[:-1].iterrows():
            adverse, favorable = (bar.low, bar.high) if side == 1 else (bar.high, bar.low)
            if side * (adverse - stop) <= 0:
                fill = min(stop, bar.open) if side == 1 else max(stop, bar.open)
            elif side * (favorable - target) >= 0:
                fill = target
            else:
                continue
            fill += price.at[stamp, "open"] - bar.open
            exit_time = stamp + pd.Timedelta(minutes=1)
            break
        pnl.append((side * (fill - entry) / entry, 1 + fill / entry))
        available = exit_time + pd.Timedelta(minutes=CFG.cooldown_minutes)
    return pnl


def report(trades, cost):
    net = [gross - cost * turnover for gross, turnover in trades]
    loss = -sum(min(0, x) for x in net)
    return {"n": len(net), "wins": int(sum(x > 0 for x in net)),
            "mean_net_bps": 10000 * sum(net) / len(net) if net else None,
            "pf": sum(max(0, x) for x in net) / loss if loss else None}


def main():
    pooled = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "XRPUSDT", "TRXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]:
        price = candles(symbol, "klines")
        mark = candles(symbol, "markPriceKlines")
        inputs = signals(price, mark.open)
        for variant in ["baseline", "cross", "near"]:
            for phase, start, end in [("development", START, SPLIT), ("holdout", SPLIT, END)]:
                trades = replay(price, mark, inputs, variant, start, end)
                pooled.setdefault((variant, phase), []).extend(trades)
                print(symbol, variant, phase, json.dumps(report(trades, CFG.cost_per_fill)), flush=True)
    for key, trades in pooled.items():
        for cost in [CFG.cost_per_fill, CFG.cost_per_fill * 1.5]:
            print("POOLED", *key, cost, json.dumps(report(trades, cost)), flush=True)


if __name__ == "__main__":
    main()
