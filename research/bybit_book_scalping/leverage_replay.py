"""Research-only leverage sensitivity for the frozen cross-venue price-gap signal."""

import argparse
import json
import math
from contextlib import chdir
from pathlib import Path

import numpy as np
import pandas as pd
from external_tape import reference_signal
from study import WEEK, load


def position(arrays, k, end, side, volatility, cash, free, policy, step, stress):
    ts, bid, ask, bs, az = arrays
    i = k + 1
    fee = 0.00055
    slip = 0.00045 if stress else 0.00015
    # Five-minute realized volatility, known when choosing size.
    sigma = max(volatility, 0.0001)
    margin_fraction = float(np.clip(0.30 * 0.002 / sigma, 0.10, 0.60))
    adaptive = (
        50 if sigma <= 0.001 else 25 if sigma <= 0.002 else 10 if sigma <= 0.004 else 5
    )
    lev = adaptive if policy == "adaptive" else int(policy)
    entry_quote = ask[i] if side == 1 else bid[i]
    entry = entry_quote * (1 + side * slip)
    # Cap entry size by displayed volume at decision AND arrival.
    depth = min(az[k], az[i]) if side == 1 else min(bs[k], bs[i])
    qty = (
        math.floor(
            min(
                cash * margin_fraction * lev / entry,
                free / (1 / lev + 2 * fee) / entry,
                depth,
            )
            / step
        )
        * step
    )
    value = qty * entry
    if value < 5 or value * (1 / lev + 2 * fee) > free:
        return None
    deadline = min(end - 1, ts.searchsorted(ts[i] + 900000))
    quotes = bid if side == 1 else ask
    sizes = bs if side == 1 else az
    move = side * (quotes[i + 1 : deadline + 1] / entry - 1)
    hits = np.flatnonzero((move <= -0.003) | (move >= 0.006))
    j = min(deadline, i + 2 + hits[0]) if len(hits) else deadline
    while j < end - 1 and sizes[j] < qty:
        j += 1
    if sizes[j] < qty or ts[j] // 3600000 != ts[i] // 3600000:
        raise ValueError("Unmodelled size/funding at exit")
    # Isolated-margin approximation using sampled mid as a mark proxy.
    # Stress adds 50 bp adverse mark divergence and a higher maintenance rate.
    mm = 0.01 if stress else 0.005
    mark = (
        (bid[i : j + 1] + ask[i : j + 1]) * 0.5 * (1 - side * (0.005 if stress else 0))
    )
    margin = value / lev
    breached = np.flatnonzero(
        margin + side * qty * (mark - entry) <= qty * mark * (mm + fee)
    )
    liquidated = bool(len(breached))
    if liquidated:
        j = i + breached[0]
    exits = quotes[i : j + 1] * (1 - side * slip)
    pnl = qty * (side * (exits - entry) - fee * (exits + entry))
    if liquidated:
        pnl[-1] = -margin - value * fee
    return ts[i : j + 1], pnl, margin + 2 * fee * value, value, lev, liquidated


def prepare(book, symbol):
    arrays = tuple(
        book[key].to_numpy() for key in ("ts", "bid", "ask", "bid_size", "ask_size")
    )
    mid = (book.bid + book.ask) / 2
    vol = np.sqrt(np.log(mid).diff().pow(2).rolling(60).sum()).to_numpy()
    return (
        arrays,
        reference_signal(book, symbol),
        vol,
        0.1 if symbol == "LTCUSDT" else 1,
    )


def replay(markets, start, policy, stress):
    # At an arrival, release closed collateral and mark all open positions as of now.
    # Future exit paths are used only at their respective simulated timestamps.
    events = []
    for name, (arrays, signal, vol, _step) in markets.items():
        ts = arrays[0]
        first, end = ts.searchsorted([start, start + WEEK])
        for k in np.flatnonzero(signal[first:end]) + first:
            if k + 1 >= end or not np.isfinite(vol[k]):
                continue
            now = ts[k + 1]
            if min(start + WEEK - now, 3600000 - now % 3600000) > 915000:
                events.append((now, name, k, end))
    realized = 50.0
    active = {}
    closed_at = {}
    paths = []
    ledger = []
    max_open = 0
    for now, name, k, end in sorted(events):
        arrays, signal, vol, step = markets[name]
        for sym, (times, pnl, _reserve) in list(active.items()):
            if times[-1] <= now:
                realized += pnl[-1]
                closed_at[sym] = times[-1]
                del active[sym]
        if name in active or arrays[0][k] <= closed_at.get(name, 0):
            continue
        equity = realized + sum(
            path[times.searchsorted(now, side="right") - 1]
            for times, path, _ in active.values()
        )
        # Open profits cannot fund a new isolated position.
        free = min(realized, equity) - sum(reserve for _, _, reserve in active.values())
        if free <= 0 or equity < 0.01:
            continue
        trade = position(
            arrays, k, end, signal[k], vol[k], equity, free, policy, step, stress
        )
        if trade is None:
            continue
        times, pnl, reserve, value, lev, liq = trade
        active[name] = times, pnl, reserve
        max_open = max(max_open, len(active))
        paths.append(pd.Series(np.diff(pnl, prepend=0), index=times))
        ledger.append((float(pnl[-1]), liq, value / equity, lev))
    trajectory = (
        50 + pd.concat(paths).groupby(level=0).sum().sort_index().cumsum().to_numpy()
        if paths
        else np.array([50.0])
    )
    trajectory = np.r_[50.0, trajectory]
    cash = trajectory[-1]
    peak = trajectory.max()
    drawdown = (1 - trajectory / np.maximum.accumulate(trajectory)).max()
    assert np.isclose(cash, 50 + sum(row[0] for row in ledger))
    return {
        "end": cash,
        "max_open": max_open,
        "peak": peak,
        "drawdown": drawdown,
        "trades": len(ledger),
        "wins": sum(p > 0 for p, _, _, _ in ledger),
        "liquidations_proxy": sum(q for _, q, _, _ in ledger),
        "gross_profit": sum(max(0, p) for p, _, _, _ in ledger),
        "gross_loss": -sum(min(0, p) for p, _, _, _ in ledger),
        "max_notional_to_equity": max((x for _, _, x, _ in ledger), default=0),
        "leverage_used": sorted({lev for _, _, _, lev in ledger}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", choices=["XRPUSDT", "LTCUSDT"])
    parser.add_argument("start")
    parser.add_argument("weeks", type=int, choices=[3, 4])
    parser.add_argument("--other-dir", type=Path)
    args = parser.parse_args()
    markets = {args.symbol: prepare(load(args.start, days=7 * args.weeks), args.symbol)}
    if args.other_dir:
        other = "XRPUSDT" if args.symbol == "LTCUSDT" else "LTCUSDT"
        with chdir(args.other_dir):
            markets[other] = prepare(load(args.start, days=7 * args.weeks), other)
    start = pd.Timestamp(args.start, tz="UTC").value // 10**6
    rows = []
    for w in range(args.weeks):
        for policy in ["2", "10", "25", "50", "adaptive"]:
            for stress in [False, True]:
                week = start + w * WEEK
                rows.append(
                    {
                        "symbol": "+".join(sorted(markets)),
                        "week": str(pd.to_datetime(week, unit="ms").date()),
                        "policy": policy,
                        "stress": stress,
                        **replay(markets, week, policy, stress),
                    }
                )
    Path(
        "joint-leverage-results.json" if args.other_dir else "leverage-results.json"
    ).write_text(json.dumps(rows, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
