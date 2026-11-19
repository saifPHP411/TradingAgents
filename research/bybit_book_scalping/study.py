"""Fixed-rule LTCUSDT order-book study. Times in UTC, costs on both legs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from external_tape import reference_signal

COLS = ["ts", "bid", "ask", "bid_size", "ask_size", "bid5", "ask5"]
WEEK = 7 * 86400_000
PROFILES = ("pressure_break", "exhaustion", "compression", "passive_retest")


def load(start_date, days=28):
    frames = []
    for day in pd.date_range(start_date, periods=days):
        day_start = day.value // 10**6
        f = pd.read_csv(f"{day:%Y-%m-%d}.csv", names=COLS)
        frames.append(f.loc[f.ts.between(day_start, day_start + 86400_000 - 1)])
    df = pd.concat(frames, ignore_index=True).sort_values("ts")
    df = df.loc[~(df.ts // 5000).duplicated()].reset_index(drop=True)
    assert (df.ask > df.bid).all() and (df[COLS[3:]] > 0).all().all()
    assert len(df) > 0.999 * days * 17280 and df.ts.diff().max() < 30000
    return df


def signals(df):
    mid = (df.bid + df.ask) / 2
    imbalance = (df.bid5 - df.ask5) / (df.bid5 + df.ask5)
    pressure = imbalance.rolling(6).mean()
    # Historical range ends before the observation where the breakout is detected.
    high = mid.shift().rolling(60).max()
    low = mid.shift().rolling(60).min()
    impulse = mid.pct_change(12)
    alignment = np.sign(impulse)
    broke = (mid > high).astype(int) - (mid < low).astype(int)
    spread_ok = (df.ask - df.bid) / mid < 0.0005
    flow = broke.where((pressure * broke > 0.20) & (impulse.abs() > 0.0015), 0)
    reversal = -np.sign(mid.pct_change(36))
    extreme = mid.pct_change(36).abs() > 0.004
    fade = reversal.where(
        extreme & (pressure * reversal > 0.15) & (mid.pct_change(3) * reversal > 0), 0
    )
    quiet = ((high - low) / mid < 0.0025) & (pressure * broke > 0.25)
    retest = alignment.where((impulse.abs() > 0.002) & (pressure * alignment > 0.25), 0)
    return {
        k: v.where(spread_ok, 0).fillna(0).astype(int).to_numpy()
        for k, v in zip(
            PROFILES, (flow, fade, broke.where(quiet, 0), retest), strict=True
        )
    }


def run(df, sig, mode, start, stress, quantity_step):
    ts, bid, ask, bsize, asize = (df[k].to_numpy() for k in COLS[:5])
    floor = np.searchsorted(ts, start)
    ceiling = np.searchsorted(ts, start + WEEK)
    capital = peak = 50.0
    drawdown = 0.0
    available = start
    day = None
    trades = []
    taker, maker = 0.00055, 0.00020
    slip = 0.00045 if stress else 0.00015
    for signal_idx in np.flatnonzero(sig[floor:ceiling]) + floor:
        i = signal_idx + 1
        if i >= ceiling or ts[signal_idx] < available:
            continue
        now = ts[i]
        date = now // 86400_000
        if date != day:
            day, day_start, streak = date, capital, 0
        if streak >= 3 or capital <= 0.97 * day_start or drawdown >= 0.10:
            continue
        # Avoid every UTC hour boundary, including possible funding settlements.
        remaining = min(start + WEEK - now, 3600000 - now % 3600000)
        if remaining < 945000:
            continue
        side = sig[signal_idx]
        passive = mode == "passive_retest"
        if passive:
            entry = bid[signal_idx] if side == 1 else ask[signal_idx]
            if (side == 1 and entry >= ask[i]) or (side == -1 and entry <= bid[i]):
                continue
            expires = now + 30000
            while i < ceiling and ts[i] <= expires:
                if (side == 1 and ask[i] < entry) or (side == -1 and bid[i] > entry):
                    break
                i += 1
            if i >= ceiling or ts[i] > expires:
                available = expires
                continue
        else:
            quote = ask[i] if side == 1 else bid[i]
            entry = quote * (1 + side * slip)
        qty = math.floor(capital * 0.2 / entry / quantity_step) * quantity_step
        if qty * entry < 5 or qty <= 0:
            continue
        if not passive and qty > (asize[i] if side == 1 else bsize[i]):
            continue
        fee = maker if passive else taker
        end = min(ceiling - 1, np.searchsorted(ts, ts[i] + 900000))
        quotes = bid[i + 1 : end + 1] if side == 1 else ask[i + 1 : end + 1]
        if not len(quotes):
            continue
        change = side * (quotes / entry - 1)
        hit = np.flatnonzero((change <= -0.003) | (change >= 0.006))
        # One further observation for reactive exits, known time limit submitted on time.
        j = min(end, i + 2 + hit[0]) if len(hit) else end
        exit_sizes = bsize if side == 1 else asize
        while j + 1 < ceiling and exit_sizes[j] < qty:
            j += 1
        if exit_sizes[j] < qty:
            raise ValueError("Insufficient size at final exit")
        exit_quotes = bid[i : j + 1] if side == 1 else ask[i : j + 1]
        fills = exit_quotes * (1 - side * slip)
        equity = capital + qty * (side * (fills - entry) - fee * entry - taker * fills)
        peaks = np.maximum.accumulate(np.maximum(peak, equity))
        drawdown = max(drawdown, float(np.max(1 - equity / peaks)))
        pnl = equity[-1] - capital
        capital = float(equity[-1])
        peak = float(peaks[-1])
        streak = streak + 1 if pnl <= 0 else 0
        available = ts[j] + 180000
        trades.append(float(pnl))
    loss = -sum(min(x, 0) for x in trades)
    return {
        "end": capital,
        "peak": peak,
        "trades": len(trades),
        "wins": sum(x > 0 for x in trades),
        "max_dd": drawdown,
        "pf": sum(max(x, 0) for x in trades) / loss if loss else None,
    }


def oracle(bid, ask, times):
    # Optimum possible closed balance at j. Any final position must have opened
    # at an earlier i. Equity multiples are positive and linear in opening cash.
    cash = np.full(len(times), 50.0)
    fee = 0.00055
    for j in range(1, len(times)):
        first = np.searchsorted(times, times[j] - 900000)
        ix = slice(first, j)
        long = (1 - fee) * bid[j] / ask[ix] - (1 + fee)
        short = (1 - fee) - (1 + fee) * ask[j] / bid[ix]
        cash[j] = max(
            cash[j - 1], np.max(cash[ix] * (1 + 0.2 * np.maximum(long, short)))
        )
    return float(cash[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", choices=["XRPUSDT", "LTCUSDT"])
    parser.add_argument("start_date")
    args = parser.parse_args()
    df = load(args.start_date)
    first = pd.Timestamp(args.start_date, tz="UTC").value // 10**6
    quantity_step = 1 if args.symbol == "XRPUSDT" else 0.1
    variants = signals(df)
    variants["reference_gap"] = reference_signal(df, args.symbol)
    report = {
        "symbol": args.symbol,
        "samples": len(df),
        "max_gap_ms": int(df.ts.diff().max()),
        "weeks": [],
    }
    for week in range(4):
        start = first + week * WEEK
        dfw = df[df.ts.between(start, start + WEEK - 1)]
        out = {
            "start": str(pd.to_datetime(start, unit="ms", utc=True).date()),
            "strategies": {
                k: {
                    stress: run(df, v, k, start, stress == "stress", quantity_step)
                    for stress in ("base", "stress")
                }
                for k, v in variants.items()
            },
        }
        out["perfect_timing_taker"] = oracle(
            dfw.bid.to_numpy(), dfw.ask.to_numpy(), dfw.ts.to_numpy()
        )
        report["weeks"].append(out)
    Path("results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
