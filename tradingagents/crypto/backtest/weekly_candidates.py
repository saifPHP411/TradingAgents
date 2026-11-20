"""Exploratory multi-strategy study on historical Binance USD-M 1m bars.

Strictly research only. Uses completed-minute signals, next-minute open fills,
conservative stop-first OHLC exits, actual public archive candles and estimated
costs. It does not use Mark-triggered Bybit scalper logic or submit any orders.
Exchange-specific minimums, book depth and liquidation tier are not verified.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev

ONE_DAY_MINUTES = 1440
COMMISSION_RATE = 0.00055
ADVERSE_FILL_IMPACT_RATE = 0.00030
MAX_NOTIONAL_FRACTION = 0.20
MAX_LEVERAGE = 2.0
INITIAL_USDT = 50.0
HOLD_MINUTES = 30

STRATEGIES = {
    "ema9_ema21_rsi_momentum": (0.0040, 0.0080),
    "bollinger_rsi_reversion": (0.0045, 0.0060),
    "donchian20_breakout": (0.0055, 0.0100),
    # Predeclared, lower-frequency 1m-executed strategies with longer holds.
    "ema60_rsi_pullback": (0.0060, 0.0120),
    "donchian_15m_trend": (0.0080, 0.0180),
    "vwap_15m_reclaim": (0.0060, 0.0120),
}
HOLD_BY_STRATEGY = {
    "ema60_rsi_pullback": 180,
    "donchian_15m_trend": 240,
    "vwap_15m_reclaim": 120,
}


@dataclass(frozen=True)
class Bar:
    ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_candles(data: bytes, day: date) -> list[Bar]:
    """Require 1440 ordered one-minute valid OHLCV bars for one UTC date."""
    bars = []
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
    for row in reader:
        if not row:
            continue
        if row[0].strip().lower() in ("open_time", "open time"):
            continue
        if len(row) < 6:
            raise ValueError("one-minute archive row is missing required columns")
        ms = int(row[0])
        numbers = tuple(float(value) for value in row[1:6])
        opening, high, low, closing, volume = numbers
        if (not all(math.isfinite(v) for v in numbers)
                or min(opening, high, low, closing) <= 0 or volume < 0
                or high < max(opening, closing, low)
                or low > min(opening, closing)):
            raise ValueError("invalid OHLCV archive row")
        if datetime.fromtimestamp(ms / 1000, UTC).date() != day:
            raise ValueError("unexpected archive candle date")
        if bars and ms - bars[-1].ms != 60000:
            raise ValueError("archive timestamps must be consecutive minutes")
        bars.append(Bar(ms, opening, high, low, closing, volume))
    if len(bars) != ONE_DAY_MINUTES:
        raise ValueError(f"expected exactly 1440 candles; got {len(bars)}")
    return bars


def fetch_daily(symbol: str, day: date, *, transport=None) -> list[Bar]:
    if not symbol.endswith("USDT") or not symbol.isalnum():
        raise ValueError("symbol must be a USDT perpetual")
    name = f"{symbol}-1m-{day.isoformat()}.zip"
    url = (
        f"https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/1m/{name}"
    )
    if transport is None:
        request = urllib.request.Request(url, headers={
            "User-Agent": "TradingAgents-read-only-minute-research/1.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError("historical source HTTP status was not 200")
                raw = response.read(16 * 1024 * 1024 + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise ValueError(f"historical source unavailable: {type(exc).__name__}") from exc
    else:
        raw = transport(url)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("unexpectedly large daily archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        csv_names = [name for name in bundle.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError("historical daily archive must contain one CSV")
        rows = bundle.read(csv_names[0])
    return parse_candles(rows, day)


def indicators(bars: list[Bar]) -> dict[str, list[float]]:
    close = [bar.close for bar in bars]
    ema9 = [close[0]]
    ema21 = [close[0]]
    alpha9, alpha21 = 2 / 10, 2 / 22
    ema60 = [close[0]]
    ema240 = [close[0]]
    for price in close[1:]:
        ema9.append(ema9[-1] * (1 - alpha9) + price * alpha9)
        ema21.append(ema21[-1] * (1 - alpha21) + price * alpha21)
        ema60.append(ema60[-1] * (1 - 2 / 61) + price * (2 / 61))
        ema240.append(ema240[-1] * (1 - 2 / 241) + price * (2 / 241))
    # Anchored UTC-session VWAP; each value includes only the current and past bars.
    vwap = []
    session = None
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0
    for bar in bars:
        day = bar.ms // (86_400_000)
        if day != session:
            session = day
            cumulative_price_volume = 0.0
            cumulative_volume = 0.0
        typical = (bar.high + bar.low + bar.close) / 3
        cumulative_price_volume += typical * bar.volume
        cumulative_volume += bar.volume
        vwap.append(
            cumulative_price_volume / cumulative_volume
            if cumulative_volume else bar.close
        )
    rsi = [50.0] * len(close)
    gains = losses = 0.0
    for i in range(1, len(close)):
        delta = close[i] - close[i - 1]
        gain, loss = max(delta, 0.0), max(-delta, 0.0)
        if i <= 14:
            gains += gain
            losses += loss
            if i == 14:
                gains /= 14
                losses /= 14
        else:
            gains = (13 * gains + gain) / 14
            losses = (13 * losses + loss) / 14
        if i >= 14:
            rsi[i] = (100 * gains / (gains + losses)) if gains + losses else 50.0
    return {
        "ema9": ema9, "ema21": ema21,
        "ema60": ema60, "ema240": ema240, "vwap": vwap,
        "rsi": rsi,
    }


def signal(bars: list[Bar], feat: dict, i: int, strategy: str) -> int:
    """i is the completed candle index; entry may occur at i+1 OPEN only."""
    if i < 100 or (
        strategy in {"ema60_rsi_pullback", "donchian_15m_trend", "vwap_15m_reclaim"}
        and i < 240
    ):
        return 0
    rsi = feat["rsi"][i]
    close = bars[i].close
    if strategy == "ema9_ema21_rsi_momentum":
        fast, slow = feat["ema9"], feat["ema21"]
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1] and 55 <= rsi <= 75:
            return 1
        if fast[i] < slow[i] and fast[i - 1] >= slow[i - 1] and 25 <= rsi <= 45:
            return -1
    elif strategy == "bollinger_rsi_reversion":
        current = [bar.close for bar in bars[i - 19:i + 1]]
        previous = [bar.close for bar in bars[i - 20:i]]
        current_mean, previous_mean = mean(current), mean(previous)
        current_std, previous_std = pstdev(current), pstdev(previous)
        lower = current_mean - 2 * current_std
        upper = current_mean + 2 * current_std
        previous_lower = previous_mean - 2 * previous_std
        previous_upper = previous_mean + 2 * previous_std
        if bars[i - 1].close < previous_lower and close >= lower and rsi < 48:
            return 1
        if bars[i - 1].close > previous_upper and close <= upper and rsi > 52:
            return -1
    elif strategy == "donchian20_breakout":
        previous_high = max(bar.high for bar in bars[i - 20:i])
        previous_low = min(bar.low for bar in bars[i - 20:i])
        average_volume = mean(bar.volume for bar in bars[i - 20:i])
        if bars[i].volume < average_volume * 1.2:
            return 0
        if close > previous_high and 55 < rsi < 85:
            return 1
        if close < previous_low and 15 < rsi < 45:
            return -1
    elif strategy == "ema60_rsi_pullback":
        # Evaluated only at five-minute closing bars (next minute executes).
        if (i + 1) % 5:
            return 0
        fast, slow = feat["ema60"], feat["ema240"]
        body_pct = abs(close - bars[i].open) / close
        if not 0.0001 <= body_pct <= 0.008:
            return 0
        if (
            fast[i] > slow[i] and close > slow[i]
            and bars[i - 1].close <= fast[i - 1] and close > fast[i]
            and 50 <= rsi <= 70
        ):
            return 1
        if (
            fast[i] < slow[i] and close < slow[i]
            and bars[i - 1].close >= fast[i - 1] and close < fast[i]
            and 30 <= rsi <= 50
        ):
            return -1
    elif strategy == "donchian_15m_trend":
        # 15-minute close, previous 3h high/low, 1h/4h EMA regime.
        if (i + 1) % 15:
            return 0
        fast, slow = feat["ema60"][i], feat["ema240"][i]
        previous_high = max(bar.high for bar in bars[i - 180:i])
        previous_low = min(bar.low for bar in bars[i - 180:i])
        recent_volume = sum(bar.volume for bar in bars[i - 14:i + 1])
        baseline_volume = sum(bar.volume for bar in bars[i - 60:i]) / 4
        if recent_volume < baseline_volume * 1.1:
            return 0
        if fast > slow and close > previous_high and 56 <= rsi <= 85:
            return 1
        if fast < slow and close < previous_low and 15 <= rsi <= 44:
            return -1
    elif strategy == "vwap_15m_reclaim":
        if (i + 1) % 15:
            return 0
        vwap, ema60, ema240 = feat["vwap"], feat["ema60"], feat["ema240"]
        if (
            ema60[i] > ema240[i] and close > vwap[i]
            and bars[i - 1].close <= vwap[i - 1] and 52 <= rsi <= 72
        ):
            return 1
        if (
            ema60[i] < ema240[i] and close < vwap[i]
            and bars[i - 1].close >= vwap[i - 1] and 28 <= rsi <= 48
        ):
            return -1
    else:
        raise ValueError("unknown predeclared strategy")
    return 0


def simulate_week(
    by_symbol: dict[str, list[Bar]], strategy: str, *, start: int, end: int,
    commission_rate: float = COMMISSION_RATE,
    adverse_impact_rate: float = ADVERSE_FILL_IMPACT_RATE,
) -> dict:
    """One isolated 50-USDT week, one shared-account position at a time."""
    if strategy not in STRATEGIES or start < 101 or end <= start:
        raise ValueError("invalid strategy or simulation window")
    if (not all(math.isfinite(x) and 0 <= x <= 0.01
                for x in (commission_rate, adverse_impact_rate))):
        raise ValueError("invalid commission or adverse impact assumptions")
    if any(len(bars) < end for bars in by_symbol.values()):
        raise ValueError("bar histories must cover the entire window")
    if not by_symbol:
        raise ValueError("at least one symbol required")
    timeline = [bar.ms for bar in next(iter(by_symbol.values()))]
    if any([bar.ms for bar in bars] != timeline for bars in by_symbol.values()):
        raise ValueError("all symbols must share exact minute timestamps")
    f = {symbol: indicators(bars) for symbol, bars in by_symbol.items()}
    stop_frac, take_frac = STRATEGIES[strategy]
    equity = INITIAL_USDT
    trades = []
    position = None
    snapshots = [{
        "timestamp": datetime.fromtimestamp(timeline[start] / 1000, UTC).isoformat(),
        "equity_usdt": equity, "open_positions_count": 0,
    }]
    for i in range(start, end):
        if i > start and (i - start) % ONE_DAY_MINUTES == 0:
            # Daily account equity includes unrealized PnL, not just balance.
            marked = equity
            if position:
                prior_price = by_symbol[position["symbol"]][i - 1].close
                marked += (prior_price - position["entry"]) * position["qty"] * position["direction"]
            snapshots.append({
                "timestamp": datetime.fromtimestamp(timeline[i] / 1000, UTC).isoformat(),
                "equity_usdt": marked, "open_positions_count": int(position is not None),
            })
        closed_this_bar = False
        if position:
            bar = by_symbol[position["symbol"]][i]
            direction = position["direction"]
            stop, target = position["stop"], position["target"]
            is_stop = bar.low <= stop if direction > 0 else bar.high >= stop
            is_target = bar.high >= target if direction > 0 else bar.low <= target
            deadline = (
                i - position["entry_idx"]
                >= HOLD_BY_STRATEGY.get(strategy, HOLD_MINUTES) - 1
            )
            is_final = i == end - 1
            if is_stop or is_target or deadline or is_final:
                # Worst-case within-candle exit: stop wins any stop/target tie.
                if is_stop:
                    raw_exit = min(bar.open, stop) if direction > 0 else max(bar.open, stop)
                    reason = "stop_loss"
                elif is_target:
                    raw_exit = target
                    reason = "take_profit"
                else:
                    raw_exit = bar.close
                    reason = "time_exit" if deadline else "week_boundary"
                exit_price = raw_exit * (1 - direction * adverse_impact_rate)
                gross = direction * (exit_price - position["entry"]) * position["qty"]
                exit_fee = position["qty"] * exit_price * commission_rate
                net = gross - position["entry_fee"] - exit_fee
                equity += gross - exit_fee
                trades.append({
                    "symbol": position["symbol"],
                    "side": "long" if direction > 0 else "short",
                    "closed_at": datetime.fromtimestamp((bar.ms + 60000 - 1) / 1000, UTC).isoformat(),
                    "net_pnl_usdt": net, "exit_reason": reason,
                })
                position = None
                closed_this_bar = True
        if position is None and not closed_this_bar:
            for symbol in sorted(by_symbol):
                action = signal(by_symbol[symbol], f[symbol], i - 1, strategy)
                if action == 0:
                    continue
                bar = by_symbol[symbol][i]
                price = bar.open * (1 + action * adverse_impact_rate)
                qty = (max(equity, 0) * MAX_NOTIONAL_FRACTION) / price
                fee = qty * price * commission_rate
                if qty <= 0 or fee >= equity:
                    break
                equity -= fee
                position = {
                    "symbol": symbol, "direction": action, "entry": price, "qty": qty,
                    "entry_fee": fee, "entry_idx": i,
                    "stop": price * (1 - action * stop_frac),
                    "target": price * (1 + action * take_frac),
                }
                # Check same entry candle conservatively for immediate exits.
                # The next iteration may not inspect this candle, so do it here.
                touch_stop = bar.low <= position["stop"] if action > 0 else bar.high >= position["stop"]
                touch_target = bar.high >= position["target"] if action > 0 else bar.low <= position["target"]
                if touch_stop or touch_target or i == end - 1:
                    if touch_stop:
                        raw_exit = min(bar.open, position["stop"]) if action > 0 else max(bar.open, position["stop"])
                        reason = "entry_bar_stop"
                    elif touch_target:
                        raw_exit = position["target"]
                        reason = "entry_bar_target"
                    else:
                        raw_exit = bar.close
                        reason = "week_boundary"
                    exit_price = raw_exit * (1 - action * adverse_impact_rate)
                    gross = action * (exit_price - position["entry"]) * qty
                    exit_fee = qty * exit_price * commission_rate
                    net = gross - fee - exit_fee
                    equity += gross - exit_fee
                    trades.append({
                        "symbol": symbol,
                        "side": "long" if action > 0 else "short",
                        "closed_at": datetime.fromtimestamp((bar.ms + 60000 - 1) / 1000, UTC).isoformat(),
                        "net_pnl_usdt": net, "exit_reason": reason,
                    })
                    position = None
                break
    if position is not None:
        raise RuntimeError("all positions must be closed at the seven-day boundary")
    end_time = datetime.fromtimestamp((timeline[end - 1] + 60000) / 1000, UTC).isoformat()
    snapshots.append({"timestamp": end_time, "equity_usdt": equity, "open_positions_count": 0})
    winnings = [t["net_pnl_usdt"] for t in trades if t["net_pnl_usdt"] > 0]
    losses = [t["net_pnl_usdt"] for t in trades if t["net_pnl_usdt"] <= 0]
    gross_loss = -sum(x for x in losses if x < 0)
    return {
        "start": snapshots[0]["timestamp"], "end_exclusive": end_time,
        "initial_equity_usdt": INITIAL_USDT, "final_equity_usdt": equity,
        "net_return_pct": (equity / INITIAL_USDT - 1) * 100,
        "trades": len(trades), "win_rate_pct": 100 * len(winnings) / len(trades) if trades else None,
        "profit_factor": sum(winnings) / gross_loss if gross_loss else None,
        "closed_trades": trades,
        "account_equity": snapshots,
    }


def screen_weekly_net_50usdt(weeks: list[dict]) -> dict:
    """Compare *every* isolated 50-USDT week with the literal goal.

    This is an exploratory score of simulated candles, never an independently
    validated or venue-executable strategy.
    """
    outcomes = []
    for week in weeks:
        trade_count = week["trades"]
        win_rate = week["win_rate_pct"]
        equity = week["final_equity_usdt"]
        weekly_profit = equity - INITIAL_USDT
        profit_goal_met = weekly_profit >= 50.0 - 1e-9
        objective_met = (
            trade_count >= 30
            and win_rate is not None
            and win_rate >= 70.0
            and profit_goal_met
        )
        outcomes.append({
            "start": week["start"],
            "end_exclusive": week["end_exclusive"],
            "trades": trade_count,
            "win_rate_pct": win_rate,
            "final_usdt": equity,
            "net_profit_usdt": weekly_profit,
            "weekly_profit_50usdt_met": profit_goal_met,
            "weekly_profit_50usdt_and_70pct_met": objective_met,
        })
    hits = sum(item["weekly_profit_50usdt_and_70pct_met"] for item in outcomes)
    return {
        "tested_seven_day_periods": len(outcomes),
        "weeks_meeting_both_goals": hits,
        "all_weeks_met_both_goals": bool(outcomes) and hits == len(outcomes),
        "sufficient_four_week_sample": len(outcomes) >= 4,
        "goal_independently_validated": False,
        "results_from_modeled_candles_only": True,
        "weekly_results": outcomes,
    }


def run_scan(*, start_day: date, days: int, symbols: tuple[str, ...], transport=None) -> dict:
    if days < 7 or days % 7 or not symbols:
        raise ValueError("days must be a positive multiple of 7 and symbols required")
    history_start = start_day - timedelta(days=1)
    by_symbol = {}
    for symbol in symbols:
        history = []
        for offset in range(days + 1):
            history.extend(fetch_daily(symbol, history_start + timedelta(days=offset),
                                       transport=transport))
        by_symbol[symbol] = history
    scenarios = {
        "zero_cost_upper_bound": (0.0, 0.0),
        "realistic_estimate": (COMMISSION_RATE, ADVERSE_FILL_IMPACT_RATE),
        "stress_estimate": (0.0008, 0.0007),
    }
    reports = {}
    scenario_reports = {}
    for strategy in STRATEGIES:
        by_scenario = {}
        for scenario, (commission, impact) in scenarios.items():
            weeks = []
            for week_idx in range(days // 7):
                start = ONE_DAY_MINUTES + week_idx * 7 * ONE_DAY_MINUTES
                end = start + 7 * ONE_DAY_MINUTES
                weeks.append(simulate_week(
                    by_symbol, strategy, start=start, end=end,
                    commission_rate=commission, adverse_impact_rate=impact,
                ))
            by_scenario[scenario] = weeks
        reports[strategy] = by_scenario["realistic_estimate"]
        scenario_reports[strategy] = by_scenario
    return {
        "research_only": True,
        "source": "Binance USD-M official public historical daily 1m kline ZIPs",
        "source_venue_is_bybit": False,
        "frozen_candidate_parameters": True,
        "selection_is_exploratory_not_holdout_validated": True,
        "starting_equity_each_week_usdt": INITIAL_USDT,
        "max_leverage": MAX_LEVERAGE,
        "max_notional_fraction_of_account": MAX_NOTIONAL_FRACTION,
        "venue_quantity_filters_verified": False,
        "real_orderbook_fill_verified": False,
        "costs": {
            "commission_per_side_bps": COMMISSION_RATE * 10000,
            "adverse_fill_impact_per_side_bps": ADVERSE_FILL_IMPACT_RATE * 10000,
        },
        "symbols": list(symbols),
        "start_day": start_day.isoformat(),
        "days": days,
        "strategies": reports,
        "cost_scenario_comparison": scenario_reports,
        "weekly_goal_screen": {
            name: {
                scenario: screen_weekly_net_50usdt(weeks)
                for scenario, weeks in comparisons.items()
            }
            for name, comparisons in scenario_reports.items()
        },
        "demo_ready": False,
        "orders_submitted": False,
        "limitations": [
            "This uses Binance USD-M 1m traded OHLCV, not Bybit exchange executions.",
            "Next-minute open entries and worst-case stop-first candles prevent intrabar lookahead.",
            "Ticks, lot steps, funding, venue-specific price protection and partial fills not verified.",
            "No untouched prospective holdout or exchange-confirmed fill evidence.",
            "Do not promote any candidate from these exploratory results.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 11, 1))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "XRPUSDT"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_scan(start_day=args.start, days=args.days,
                          symbols=tuple(args.symbols))
    except Exception as exc:
        result = {
            "status": "not_completed", "error": f"{type(exc).__name__}: {exc}",
            "research_only": True, "demo_ready": False, "orders_submitted": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result.get("status") == "not_completed":
        print(json.dumps(result, indent=2))
        raise SystemExit(2)
    for strategy, scenarios in result["cost_scenario_comparison"].items():
        for scenario, weeks in scenarios.items():
            for index, week in enumerate(weeks):
                print(json.dumps({
                    "strategy": strategy, "scenario": scenario, "week": index + 1,
                    "win_rate_pct": week["win_rate_pct"],
                    "net_return_pct": week["net_return_pct"],
                    "trades": week["trades"], "starting_usdt": INITIAL_USDT,
                    "final_usdt": week["final_equity_usdt"], "real_trading": False,
                }))


if __name__ == "__main__":
    main()
