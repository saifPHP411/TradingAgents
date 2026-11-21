"""Read-only historical replay of the deployed Gemini desk's deterministic rules."""
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from tradingagents.crypto.backtest.weekly_candidates import fetch_daily, parse_candles
from tradingagents.crypto.bybit_linear_demo import AUTO_ENTRY_MAX_ADVERSE_BPS, MAX_OPEN_POSITIONS
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings, size_linear_position
from tradingagents.crypto.scalping import ScalpingConfig, evaluate_scalping_signal
from tradingagents.paper_experiment.perpetual_demo.adaptive import (
    MAX_REFERENCE_DRIFT,
    MIN_TRADES,
    MIN_WIN_RATE,
    STRESS_COST_PER_SIDE,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT")
MINUTE = pd.Timedelta(minutes=1)
CONFIG = ScalpingConfig()
RISK = PerpetualRiskSettings(leverage=CONFIG.leverage)
START_CASH = 50.0


def _milliseconds(stamp):
    return int(stamp.timestamp() * 1000)


def _bybit(symbol, first, last, host):
    client = BybitLinearPublicClient(base_url=host, timeout=6)
    bounds = {"start_ms": _milliseconds(first), "end_ms": _milliseconds(last - MINUTE)}
    traded = client.klines(symbol, interval="1m", **bounds)
    mark = client._candles(
        symbol, bounds["start_ms"], bounds["end_ms"],
        "/v5/market/mark-price-kline", False, interval="1m",
    )
    return traded, mark, client.instrument_info(symbol)


def _binance_day(symbol, day):
    name = f"{symbol}-1m-{day.isoformat()}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/markPriceKlines/{symbol}/1m/{name}"
    with urlopen(url, timeout=12) as response, zipfile.ZipFile(io.BytesIO(response.read(2_000_000))) as archive:
        mark = parse_candles(archive.read(name.replace(".zip", ".csv")), day)
    return fetch_daily(symbol, day), mark


def _binance(symbol, first, last):
    daily = [_binance_day(symbol, first.date() + timedelta(days=i)) for i in range((last - first).days)]
    frames = []
    for j in range(2):
        rows = [bar for day in daily for bar in day[j]]
        frame = pd.DataFrame(
            ((r.open, r.high, r.low, r.close, r.volume) for r in rows),
            columns=["open", "high", "low", "close", "volume"],
            index=pd.to_datetime([r.ms for r in rows], unit="ms", utc=True),
        )
        frames.append(frame)
    return *frames, None


def _fetch(first, last):
    expected = pd.date_range(first, last - MINUTE, freq="min")
    failures = []
    sources = (
        ("bybit", "https://api.bybit.com"),
        ("bybit_demo_public", "https://api-demo.bybit.com"),
        ("binance_usdm_proxy", None),
    )
    for name, host in sources:
        def fetch_symbol(symbol, host=host, name=name):
            data = _binance(symbol, first, last) if host is None else _bybit(symbol, first, last, host)
            if any(not frame.index.as_unit("ms").equals(expected.as_unit("ms")) for frame in data[:2]):
                raise ValueError(f"{symbol} incomplete coverage for {name}")
            return data

        try:
            with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as pool:
                return name, dict(zip(SYMBOLS, pool.map(fetch_symbol, SYMBOLS), strict=True)), failures
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(message, flush=True)
    raise RuntimeError("No complete historical dataset: " + "; ".join(failures))


def _truncated_ema(close, span, n):
    decay = 1 - 2 / (span + 1)
    ew = close.ewm(span=span, adjust=False).mean()
    return ew + (close.shift(n - 1) - ew.shift(n - 1)) * decay ** (n - 1)


def _bias_bounds(frame, freq):
    """Compute exact fast and slow EMAs for the 99/100 most recent closes."""
    close = frame["close"]
    fast, slow = (9, 21) if freq == 5 else (20, 50)
    return {
        n: ((close > _truncated_ema(close, fast, n)) & (_truncated_ema(close, fast, n) > _truncated_ema(close, slow, n))).astype(int)
        - ((close < _truncated_ema(close, fast, n)) & (_truncated_ema(close, fast, n) < _truncated_ema(close, slow, n))).astype(int)
        for n in (99, 100)
    }


def _signals(traded, mark, start):
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    frames = [traded.resample(f"{mins}min").agg(aggregation) for mins in (5, 15)]
    biases = [_bias_bounds(frame, mins) for frame, mins in zip(frames, (5, 15), strict=True)]
    prices = mark["close"].to_numpy()
    close, volume = traded["close"], traded["volume"]
    previous_close = close.shift().to_numpy()
    ema = _truncated_ema(close, 9, 119).shift().to_numpy()
    low, high = traded["low"].shift().to_numpy(), traded["high"].shift().to_numpy()
    volume_ok = (volume >= 0.5 * volume.shift().rolling(20).mean()).shift(fill_value=False)
    epsilon = ema * 1e-12  # Preserve borderline events for the exact evaluator.
    could_long = (prices > previous_close) & (prices > ema - epsilon) & (low <= ema * 1.0015 + epsilon) & (previous_close <= ema * 1.002 + epsilon)
    could_short = (prices < previous_close) & (prices < ema + epsilon) & (high >= ema * 0.9985 - epsilon) & (previous_close >= ema * 0.998 - epsilon)
    signals = {}
    for i in range(traded.index.searchsorted(start), len(traded)):
        if not volume_ok.iat[i] or not (could_long[i] or could_short[i]):
            continue
        quote_time = traded.index[i] + MINUTE - pd.Timedelta(milliseconds=1)
        direction = 1 if could_long[i] else -1
        ends, starts = [], []
        for mins, frame, bias in zip((5, 15), frames, biases, strict=True):
            duration = pd.Timedelta(minutes=mins)
            end = frame.index.searchsorted(quote_time - duration, side="right")
            begin = frame.index.searchsorted(quote_time - duration * 100)
            ends.append(end)
            starts.append(begin)
            if bias[end - begin].iat[end - 1] != direction:
                break
        else:
            result = evaluate_scalping_signal(
                traded.iloc[i - 119:i + 1], frames[0].iloc[starts[0]:ends[0]],
                frames[1].iloc[starts[1]:ends[1]], prices[i], CONFIG,
            )
            if result["signal"] != 0:
                signals[i] = result
    return signals


def _exit_price(position, mark, fill, opposite, age, final, risk_stop):
    side, stop, target = position["side"], position["stop"], position["target"]
    opening, high, low, _ = mark
    if side * (opening - stop) <= 0:
        return "stop_gap", fill[0]
    if side * (opening - target) >= 0:
        return "target", target
    if (low <= stop if side == 1 else high >= stop):
        return "stop", stop
    if (high >= target if side == 1 else low <= target):
        return "target", target
    if risk_stop or opposite or age >= CONFIG.max_hold_minutes or final:
        reason = "risk_stop" if risk_stop else "opposite" if opposite else "time" if not final else "period_end"
        return reason, fill[3]
    return None


def _run_week(tables, start_i, stop_i, cost, *, professor):
    cash, peak, drawdown, day_start = START_CASH, START_CASH, 0.0, START_CASH
    positions, ledger, cooldown = {}, [], {}
    skipped, halted = Counter(), False
    for i in range(start_i, stop_i):
        if (i - start_i) % 1440 == 0:
            day_start = cash + sum(p["qty"] * p["side"] * (tables[s][0][i, 0] - p["entry"]) for s, p in positions.items())
        for symbol, (mark, fills, signals, instrument, index) in tables.items():
            if symbol in positions or i == stop_i - 1 or halted:
                continue
            if len(positions) >= MAX_OPEN_POSITIONS or cooldown.get(symbol, 0) > i:
                continue
            signal = signals.get(i - 1)
            if signal is None or i == start_i:
                continue
            side, stop, target = signal["signal"], signal["stop_loss"], signal["take_profit"]
            reference = mark[i - 1, 3]
            if professor and side * (target - reference) / reference <= 2 * STRESS_COST_PER_SIDE:
                skipped["professor_cost"] += 1
                continue
            entry = fills[i, 0]
            limit = reference * (1 + side * AUTO_ENTRY_MAX_ADVERSE_BPS / 10_000)
            if (
                abs(mark[i, 0] / reference - 1) > MAX_REFERENCE_DRIFT
                or side * (entry - limit) > 0
                or not min(stop, target) < entry < max(stop, target)
                or not min(stop, target) < mark[i, 0] < max(stop, target)
                or not min(stop, target) < limit < max(stop, target)
            ):
                skipped["entry_price_guard"] += 1
                continue
            equity = cash + sum(p["qty"] * p["side"] * (tables[s][0][i, 0] - p["entry"]) for s, p in positions.items())
            if instrument is None:
                unit_risk = abs(limit - stop) + (limit + stop) * CONFIG.cost_per_fill
                qty = min(equity * RISK.risk_fraction / unit_risk, equity * RISK.max_notional_fraction / max(entry, limit))
            else:
                size = size_linear_position(
                    equity=equity, entry_price=limit, stop_price=stop, direction=side,
                    cost_rate_per_fill=CONFIG.cost_per_fill, instrument=instrument, risk=RISK,
                    notional_reference_price=max(mark[i, 0], entry) if side == -1 else None,
                )
                if not size.approved:
                    skipped[size.rejection_reason] += 1
                    continue
                qty = size.quantity
            positions[symbol] = {"entry_i": i, "entry_time": index[i].isoformat(), "entry": entry,
                                 "stop": stop, "target": target, "side": side, "qty": qty}
        equity = cash + sum(p["qty"] * p["side"] * (tables[s][0][i, 3] - p["entry"]) for s, p in positions.items())
        peak = max(peak, equity)
        halted = bool(halted or equity < day_start * 0.97 or equity < peak * 0.90)
        for symbol, p in list(positions.items()):
            mark, fills, signals, _, index = tables[symbol]
            opposite = signals.get(i, {}).get("signal", 0) == -p["side"]
            done = _exit_price(p, mark[i], fills[i], opposite, i - p["entry_i"],
                               i == stop_i - 1, halted)
            if done is None:
                continue
            reason, price = done
            gross = p["qty"] * p["side"] * (price - p["entry"])
            costs = p["qty"] * (price + p["entry"]) * cost
            net = gross - costs
            ledger.append({**p, "symbol": symbol, "exit_time": index[i].isoformat(), "reason": reason,
                           "exit": price, "gross_usdt": gross, "cost_usdt": costs, "net_usdt": net})
            cash += net
            del positions[symbol]
            cooldown[symbol] = i + CONFIG.cooldown_minutes
        equity = cash + sum(
            p["qty"] * p["side"] * (tables[s][0][i, 3] - p["entry"])
            for s, p in positions.items()
        )
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
        assert len(positions) <= MAX_OPEN_POSITIONS
    assert not positions
    assert math.isclose(cash, START_CASH + sum(r["net_usdt"] for r in ledger), abs_tol=1e-8)
    return {"final_usdt": cash, "max_drawdown_pct": 100 * drawdown, "halted": halted,
            "skips": dict(skipped), "trades": ledger}


def _metrics(trades):
    pnl = [row["net_usdt"] for row in trades]
    wins = [v for v in pnl if v > 0]
    losses = -sum(v for v in pnl if v < 0)
    return {"trades": len(pnl), "wins": len(wins), "net_win_rate_pct": 100 * len(wins) / len(pnl) if pnl else None,
            "profit_factor": sum(wins) / losses if losses else None, "net_usdt": sum(pnl)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-11-06")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    first, last = start - pd.Timedelta(days=2), start + pd.Timedelta(days=14)
    source, data, failures = _fetch(first, last)
    print(f"Validated complete minute coverage from {source}.", flush=True)
    series = list(data.values())
    with ProcessPoolExecutor(max_workers=4) as pool:
        signals = list(pool.map(
            _signals, [p[0] for p in series], [p[1] for p in series], [start] * len(series),
        ))
    tables = {
        name: (mark[["open", "high", "low", "close"]].to_numpy(), traded[["open", "high", "low", "close"]].to_numpy(),
               signals[i], info, mark.index)
        for i, (name, (traded, mark, info)) in enumerate(data.items())
    }
    cases = {}
    for name, cost, professor in (
        ("ema_estimated", CONFIG.cost_per_fill, False),
        ("ema_stress", STRESS_COST_PER_SIDE, False),
        ("ema_plus_professor_stress", STRESS_COST_PER_SIDE, True),
    ):
        weeks = []
        for week in range(2):
            begin = (2 + week * 7) * 1440
            result = _run_week(tables, begin, begin + 7 * 1440, cost, professor=professor)
            ledger = result.pop("trades")
            result.update(_metrics(ledger))
            weeks.append({
                **result, "ledger": ledger, "start": (start + pd.Timedelta(days=week * 7)).isoformat(),
            })
        overall = _metrics([row for w in weeks for row in w["ledger"]])
        overall["observed_evidence_threshold_met"] = (
            overall["trades"] >= MIN_TRADES and overall["net_win_rate_pct"] >= MIN_WIN_RATE * 100
        )
        cases[name] = {"cost_per_side": cost, "weeks": weeks, "pooled": overall}
    report = {
        "source": source, "symbols": SYMBOLS, "start_utc": start.isoformat(), "end_utc_exclusive": last.isoformat(),
        "source_errors": failures, "uses_bybit_filters": source.startswith("bybit"),
        "input_minutes_per_symbol_including_warmup": len(next(iter(tables.values()))[0]),
        "signals": {s: len(t[2]) for s, t in tables.items()}, "starting_usdt_per_week": START_CASH,
        "gemini_replayed": False, "funding_included": False, "cases": cases,
    }
    summary = {
        "source": source, "symbols": SYMBOLS, "signals": report["signals"],
        "cases": {key: {"pooled": val["pooled"], "weeks": [
            {k: v for k, v in week.items() if k != "ledger"} for week in val["weeks"]
        ]} for key, val in cases.items()},
    }
    print("RESULT " + json.dumps(summary, allow_nan=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
