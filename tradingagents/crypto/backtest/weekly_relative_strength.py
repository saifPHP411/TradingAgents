"""Read-only daily cross-sectional crypto research on minute OHLCV.

Three frozen hypotheses, not previously validated in this repo:
  1) buy the strongest 24h performer, if both 24h and last 4h positive;
  2) short the weakest 24h performer, if both 24h and last 4h negative;
  3) buy a 24h loser after a completed 4h rebound.

Select at UTC 00:00 using *only* historical completed bars, enter at the
next-minute open, use one shared 50-USDT account and no more than 20% total
notional. Positions exit by stop/target or the end of the same UTC day.
Archived Binance USD-M candle opens are NOT actual Bybit executable prices.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

try:
    from .weekly_candidates import Bar, fetch_daily
except ImportError:  # Direct read-only GitHub Actions script
    from weekly_candidates import Bar, fetch_daily

DAY = 1440
HOURS_4 = 240
MINUTE_MS = 60_000
STARTING_USDT = 50.0
NOTIONAL_SHARE = 0.20
MAX_LEVERAGE = 2.0
STOP_FRACTION = 0.04
TARGET_FRACTION = 0.08
COMMISSION = 0.00055
ADVERSE_IMPACT = 0.00030
STRATEGIES = (
    "daily_relative_winner_long",
    "daily_relative_loser_short",
    "daily_relative_loser_rebound_long",
)


def theoretical_weekly_net_profit_ceiling_before_costs() -> float:
    """Best imaginable seven wins, each capped at target on 20% of equity.

    Assumes one entry per day and every trade hits its 8% target, with no
    funding, fees, slippage or losing trades. No higher growth is possible
    *under this particular simulator's entry count and take-profit rules*.
    This is NOT a universal bound for crypto trading strategies.
    """
    return STARTING_USDT * ((1 + NOTIONAL_SHARE * TARGET_FRACTION) ** 7 - 1)


def choose_ranked_signal(
    by_symbol: dict[str, list[Bar]], entry_idx: int, strategy: str,
) -> tuple[str, int] | None:
    """A decision for *entry_idx* uses candles only through entry_idx-1."""
    if strategy not in STRATEGIES:
        raise ValueError("unrecognized frozen cross-sectional candidate")
    if entry_idx < DAY + 1:
        raise ValueError("one full prior day and a completed current bar are required")
    if not by_symbol:
        raise ValueError("cross-sectional research needs at least one symbol")
    ranked = []
    for symbol in sorted(by_symbol):
        bars = by_symbol[symbol]
        if entry_idx >= len(bars):
            raise ValueError("history does not reach entry timestamp")
        now = bars[entry_idx - 1].close
        past_day = bars[entry_idx - 1 - DAY].close
        past_4h = bars[entry_idx - 1 - HOURS_4].close
        if min(now, past_day, past_4h) <= 0:
            raise ValueError("invalid nonpositive price")
        if sum(b.volume for b in bars[entry_idx - HOURS_4:entry_idx]) <= 0:
            continue
        ranked.append((now / past_day - 1, now / past_4h - 1, symbol))
    if not ranked:
        return None
    if strategy == "daily_relative_winner_long":
        # Absolute direction filter stops ranking a winner in an all-down market
        # from being interpreted as a valid bullish signal.
        return_24h, return_4h, symbol = max(ranked, key=lambda row: (row[0], row[2]))
        return (symbol, 1) if return_24h >= .015 and return_4h > 0 else None
    return_24h, return_4h, symbol = min(ranked, key=lambda row: (row[0], row[2]))
    if strategy == "daily_relative_loser_short":
        return (symbol, -1) if return_24h <= -.015 and return_4h < 0 else None
    return (symbol, 1) if return_24h <= -.03 and return_4h >= .002 else None


def simulate_week(
    by_symbol: dict[str, list[Bar]], strategy: str, *,
    start: int, end: int, commission: float = COMMISSION,
    impact: float = ADVERSE_IMPACT,
) -> dict:
    """One 7-day independent account, with full both-side net P&L reconciliation."""
    if strategy not in STRATEGIES:
        raise ValueError("invalid candidate name")
    if not (isinstance(start, int) and isinstance(end, int)
            and start >= DAY + 1 and end - start == 7 * DAY):
        raise ValueError("must provide a seven-day UTC trial with two-day warmup")
    if not all(math.isfinite(c) and 0 <= c <= .01 for c in (commission, impact)):
        raise ValueError("fee and impact rates must be valid and bounded")
    if not by_symbol or any(len(b) < end for b in by_symbol.values()):
        raise ValueError("histories do not cover the seven-day interval")
    bars0 = next(iter(by_symbol.values()))
    timeline = [b.ms for b in bars0[:end]]
    if timeline[start] % (DAY * MINUTE_MS) != 0:
        raise ValueError("seven-day trial must start at UTC midnight")
    if any(timeline[j] - timeline[j-1] != MINUTE_MS for j in range(1, end)):
        raise ValueError("minute timeline must be continuous")
    if any([b.ms for b in bars[:end]] != timeline for bars in by_symbol.values()):
        raise ValueError("each symbol must share the same exact minute timestamps")
    account = STARTING_USDT
    position = None
    ledger = []
    account_snapshots = [{
        "timestamp": datetime.fromtimestamp(timeline[start] / 1000, UTC).isoformat(),
        "equity_usdt": account,
        "open_positions_count": 0,
    }]
    for i in range(start, end):
        if i > start and (i - start) % DAY == 0:
            if position is not None:
                raise RuntimeError("position leaked past mandatory daily close")
            account_snapshots.append({
                "timestamp": datetime.fromtimestamp(timeline[i] / 1000, UTC).isoformat(),
                "equity_usdt": account,
                "open_positions_count": 0,
            })
        if (i - start) % DAY == 0 and account > 0:
            selected = choose_ranked_signal(by_symbol, i, strategy)
            if selected is not None:
                symbol, direction = selected
                bar = by_symbol[symbol][i]
                if bar.volume > 0:
                    entry_price = bar.open * (1 + direction * impact)
                    qty = (account * NOTIONAL_SHARE) / entry_price
                    entry_fee = qty * entry_price * commission
                    if qty > 0 and entry_fee < account:
                        account -= entry_fee
                        position = {
                            "symbol": symbol,
                            "direction": direction,
                            "entry": entry_price,
                            "qty": qty,
                            "entry_fee": entry_fee,
                            "entry_idx": i,
                            "stop": entry_price * (1 - direction * STOP_FRACTION),
                            "target": entry_price * (1 + direction * TARGET_FRACTION),
                        }
        if position is None:
            continue
        bar = by_symbol[position["symbol"]][i]
        direction = position["direction"]
        stop_hit = (bar.low <= position["stop"] if direction > 0
                    else bar.high >= position["stop"])
        target_hit = (bar.high >= position["target"] if direction > 0
                      else bar.low <= position["target"])
        daily_end = (i - start + 1) % DAY == 0
        if not (stop_hit or target_hit or daily_end):
            continue
        if stop_hit:
            raw_exit = (min(bar.open, position["stop"]) if direction > 0
                        else max(bar.open, position["stop"]))
            reason = "stop_first"
        elif target_hit:
            raw_exit = position["target"]
            reason = "take_profit"
        else:
            raw_exit = bar.close
            reason = "daily_time_exit"
        exit_price = raw_exit * (1 - direction * impact)
        gross = (exit_price - position["entry"]) * position["qty"] * direction
        exit_fee = position["qty"] * exit_price * commission
        net = gross - position["entry_fee"] - exit_fee
        account += gross - exit_fee
        ledger.append({
            "closed_at": datetime.fromtimestamp((bar.ms + MINUTE_MS - 1) / 1000, UTC).isoformat(),
            "net_pnl_usdt": net,
            "symbol": position["symbol"],
            "side": "long" if direction > 0 else "short",
            "exit_reason": reason,
            "modeled_fill": True,
        })
        position = None
    if position is not None:
        raise RuntimeError("position was not flattened by the week end")
    account_snapshots.append({
        "timestamp": datetime.fromtimestamp(
            (timeline[end - 1] + MINUTE_MS) / 1000, UTC
        ).isoformat(),
        "equity_usdt": account,
        "open_positions_count": 0,
    })
    if not math.isclose(
        account - STARTING_USDT,
        math.fsum(t["net_pnl_usdt"] for t in ledger),
        abs_tol=1e-8, rel_tol=1e-12,
    ):
        raise RuntimeError("cross-sectional account/closed-trade ledger mismatch")
    winning = sum(t["net_pnl_usdt"] > 0 for t in ledger)
    return {
        "start": account_snapshots[0]["timestamp"],
        "end_exclusive": account_snapshots[-1]["timestamp"],
        "initial_equity_usdt": STARTING_USDT,
        "final_equity_usdt": account,
        "net_return_pct": 100 * (account / STARTING_USDT - 1),
        "trades": len(ledger),
        "win_rate_pct": (winning / len(ledger) * 100) if ledger else None,
        "account_equity": account_snapshots,
        "closed_trades": ledger,
    }


def run_study(*, first_day: date, days: int, symbols: tuple[str, ...],
              transport=None) -> dict:
    if not isinstance(days, int) or days < 7 or days % 7:
        raise ValueError("days must be a positive multiple of 7")
    if len(symbols) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("relative-strength ranking requires distinct symbols")
    history: dict[str, list[Bar]] = {}
    for symbol in symbols:
        bars = []
        for offset in range(-2, days):
            bars.extend(fetch_daily(
                symbol, first_day + timedelta(days=offset), transport=transport
            ))
        history[symbol] = bars
    scenarios = {
        "zero_cost_upper_bound": (0.0, 0.0),
        "realistic_estimate": (COMMISSION, ADVERSE_IMPACT),
        "stress_estimate": (0.0008, 0.0007),
    }
    by_strategy = {}
    for name in STRATEGIES:
        by_strategy[name] = {
            scenario: [
                simulate_week(
                    history, name, start=(week * 7 + 2) * DAY,
                    end=((week+1) * 7 + 2) * DAY,
                    commission=fees, impact=slippage,
                )
                for week in range(days // 7)
            ]
            for scenario, (fees, slippage) in scenarios.items()
        }
    return {
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "source": "public Binance USD-M 1m historical candle ZIP",
        "source_venue_is_bybit": False,
        "real_orderbook_fill_verified": False,
        "venue_quantity_filters_verified": False,
        "funding_included": False,
        "liquidation_tier_model_verified": False,
        "independent_holdout": False,
        "starting_equity_each_week_usdt": STARTING_USDT,
        "max_notional_fraction_of_account": NOTIONAL_SHARE,
        "max_leverage": MAX_LEVERAGE,
        "theoretical_best_weekly_net_profit_no_cost_usdt": (
            theoretical_weekly_net_profit_ceiling_before_costs()
        ),
        "target_50_usdt_attainable_under_frozen_daily_rules": (
            theoretical_weekly_net_profit_ceiling_before_costs() >= 50
        ),
        "symbols": list(symbols),
        "start": first_day.isoformat(),
        "days": days,
        "strategies": {
            name: cases["realistic_estimate"] for name, cases in by_strategy.items()
        },
        "cost_scenario_comparison": by_strategy,
        "limitations": [
            "Directional ranking is determined only by completed 24h and 4h returns.",
            "Historical 1m opens are modeled next-bar fills, NOT Bybit orders.",
            "Max 20% equity notional; seven entries per week at most.",
            "No Bybit order-size/tick filters, order-book slippage or partial fills verified.",
            "Funding, margin liquidation tiers and real Bybit mark stops unverified.",
            "This trial uses an already fixed public four-symbol universe, not a broad market.",
            "Risk and execution safeguards prohibit auto-deploy.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--symbols", nargs="+",
                   default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        results = run_study(
            first_day=a.start, days=a.days, symbols=tuple(a.symbols),
        )
    except Exception as exc:
        results = {
            "research_only": True, "orders_submitted": False,
            "demo_ready": False, "status": "not_completed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    if results.get("status") == "not_completed":
        print(json.dumps(results, sort_keys=True))
        raise SystemExit(2)
    for strategy, scenarios in results["cost_scenario_comparison"].items():
        for scenario, weeks in scenarios.items():
            for i, week in enumerate(weeks):
                print(json.dumps({
                    "strategy": strategy, "scenario": scenario, "week": i + 1,
                    "trades": week["trades"], "win_rate_pct": week["win_rate_pct"],
                    "net_profit_usdt": week["final_equity_usdt"] - STARTING_USDT,
                    "final_usdt": week["final_equity_usdt"],
                    "orders_submitted": False,
                }))


if __name__ == "__main__":
    main()
