"""Read-only dollar-neutral paired perpetual research, never a venue fill.

Two predeclared relationships: BTC/ETH and XRP/LTC. Every trade opens one
long and one short simultaneously on completed-minute z-score confirmation.
Both legs and all fees are accounted inside one 50-USDT simulated portfolio.

Official Binance USD-M 1m candle ZIPs are used for exploratory history only,
not historical Bybit executable bid/ask, funding, lot steps or liquidation.
The simultaneous two-leg fill is assumed and is NOT exchange-confirmed.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

try:
    from .weekly_candidates import Bar, fetch_daily
except ImportError:  # Direct execution by the read-only GitHub Actions job
    from weekly_candidates import Bar, fetch_daily

PAIRS = {
    "btc_eth_relative_value": ("BTCUSDT", "ETHUSDT"),
    "xrp_ltc_relative_value": ("XRPUSDT", "LTCUSDT"),
}
STARTING_USDT = 50.0
MAX_TOTAL_NOTIONAL_FRACTION = 0.20
MAX_LEVERAGE = 2.0
MINUTE = 60_000
BARS_PER_DAY = 1440
LOOKBACK = 240
ENTER_Z = 2.0
EXIT_Z = 0.35
STOP_Z = 3.50
MAX_HOLD_BARS = 180
COMMISSION = 0.00055
ADVERSE_IMPACT = 0.00030


def zscores_for_pair(first: list[Bar], second: list[Bar]) -> list[float | None]:
    """Z-scores use past 240 completed spreads, excluding current bar."""
    if len(first) != len(second):
        raise ValueError("pair legs must have exactly matching historical lengths")
    if any(a.ms != b.ms for a, b in zip(first, second, strict=True)):
        raise ValueError("pair legs must share minute timestamps")
    spreads = [math.log(a.close / b.close) for a, b in zip(first, second, strict=True)]
    result: list[float | None] = [None] * len(spreads)
    if len(spreads) <= LOOKBACK:
        return result
    trailing_sum = math.fsum(spreads[:LOOKBACK])
    trailing_sq = math.fsum(x * x for x in spreads[:LOOKBACK])
    for i in range(LOOKBACK, len(spreads)):
        mean = trailing_sum / LOOKBACK
        variance = max(0.0, trailing_sq / LOOKBACK - mean * mean)
        std = math.sqrt(variance)
        if std >= 1e-8:
            result[i] = (spreads[i] - mean) / std
        oldest = spreads[i - LOOKBACK]
        trailing_sum += spreads[i] - oldest
        trailing_sq += spreads[i] * spreads[i] - oldest * oldest
    return result


def simulate_pair_week(
    first: list[Bar], second: list[Bar], *,
    start: int, end: int, commission: float = COMMISSION,
    adverse_impact: float = ADVERSE_IMPACT,
) -> dict:
    """One independent week, both legs share one 50-USDT account and notional cap."""
    if not (isinstance(start, int) and isinstance(end, int)
            and LOOKBACK + 1 <= start < end <= len(first)):
        raise ValueError("invalid paired study window")
    if end - start != 7 * BARS_PER_DAY:
        raise ValueError("paired trials must comprise exactly seven days")
    if len(second) < end:
        raise ValueError("second leg does not cover the entire week")
    if not all(
        math.isfinite(rate) and 0 <= rate <= 0.01
        for rate in (commission, adverse_impact)
    ):
        raise ValueError("invalid execution cost inputs")
    if any(a.ms != b.ms for a, b in zip(first, second, strict=False)):
        raise ValueError("pair timestamps must match exactly")
    if first[start].ms % (BARS_PER_DAY * MINUTE):
        raise ValueError("weekly study must begin at UTC midnight")
    z = zscores_for_pair(first, second)
    equity = STARTING_USDT
    position = None
    trades: list[dict] = []
    times = [first, second]
    snapshots = [{
        "timestamp": datetime.fromtimestamp(first[start].ms / 1000, UTC).isoformat(),
        "equity_usdt": equity,
        "open_positions_count": 0,
    }]

    def marked_balance(mark_index: int) -> float:
        if position is None:
            return equity
        return equity + math.fsum(
            (times[j][mark_index].close - leg["entry_price"])
            * leg["qty"] * leg["direction"]
            for j, leg in enumerate(position["legs"])
        )

    for i in range(start, end):
        if i > start and (i - start) % BARS_PER_DAY == 0:
            snapshots.append({
                "timestamp": datetime.fromtimestamp(first[i].ms / 1000, UTC).isoformat(),
                "equity_usdt": marked_balance(i - 1),
                "open_positions_count": 0 if position is None else 2,
            })
        # Decisions on i use only COMPLETED candle i-1; no peek into bar i.
        prior_z = z[i - 1]
        if position is not None:
            time_exit = i - position["entry_index"] >= MAX_HOLD_BARS
            changed = prior_z is not None and abs(prior_z) <= EXIT_Z
            divergent = prior_z is not None and abs(prior_z) >= STOP_Z
            boundary = i == end - 1
            if time_exit or changed or divergent or boundary:
                gross = 0.0
                exit_fees = 0.0
                for j, leg in enumerate(position["legs"]):
                    bar = times[j][i]
                    raw_price = bar.close if boundary else bar.open
                    exit_price = raw_price * (1 - leg["direction"] * adverse_impact)
                    gross += leg["direction"] * (exit_price - leg["entry_price"]) * leg["qty"]
                    exit_fees += leg["qty"] * exit_price * commission
                equity += gross - exit_fees
                net = gross - position["entry_fees"] - exit_fees
                reason = (
                    "week_boundary" if boundary else "spread_diverged" if divergent
                    else "spread_reverted" if changed else "time_exit"
                )
                closed_ms = first[i].ms + (MINUTE - 1 if boundary else 0)
                trades.append({
                    "closed_at": datetime.fromtimestamp(closed_ms / 1000, UTC).isoformat(),
                    "net_pnl_usdt": net,
                    "exit_reason": reason,
                    "legs": [
                        {"direction": leg["direction"], "qty": leg["qty"]}
                        for leg in position["legs"]
                    ],
                    "modeled_two_leg_execution": True,
                })
                position = None
                # No same-minute reentry after closing a pair.
                continue

        if position is None and prior_z is not None and abs(prior_z) >= ENTER_Z:
            if abs(prior_z) >= STOP_Z or i == end - 1:
                continue
            if first[i].volume <= 0 or second[i].volume <= 0 or equity <= 0:
                continue
            # Positive spread: sell first leg, buy second leg.
            first_direction = -1 if prior_z >= ENTER_Z else 1
            directions = (first_direction, -first_direction)
            notional_per_leg = equity * MAX_TOTAL_NOTIONAL_FRACTION / 2.0
            legs = []
            fees = 0.0
            for j, direction in enumerate(directions):
                entry = times[j][i].open * (1 + direction * adverse_impact)
                qty = notional_per_leg / entry
                fees += qty * entry * commission
                legs.append({"direction": direction, "qty": qty, "entry_price": entry})
            if fees >= equity:
                continue
            equity -= fees
            position = {
                "entry_index": i,
                "entry_fees": fees,
                "legs": legs,
            }

    if position is not None:
        raise RuntimeError("paired strategy failed to close at weekly boundary")
    finish = datetime.fromtimestamp((first[end - 1].ms + MINUTE) / 1000, UTC).isoformat()
    snapshots.append({"timestamp": finish, "equity_usdt": equity, "open_positions_count": 0})
    wins = sum(t["net_pnl_usdt"] > 0 for t in trades)
    net_sum = math.fsum(t["net_pnl_usdt"] for t in trades)
    if not math.isclose(STARTING_USDT + net_sum, equity, abs_tol=1e-8, rel_tol=1e-12):
        raise RuntimeError("paired P&L ledger failed to reconcile to ending equity")
    return {
        "start": snapshots[0]["timestamp"],
        "end_exclusive": finish,
        "initial_equity_usdt": STARTING_USDT,
        "final_equity_usdt": equity,
        "net_return_pct": (equity / STARTING_USDT - 1) * 100,
        "trades": len(trades),
        "win_rate_pct": (100 * wins / len(trades)) if trades else None,
        "account_equity": snapshots,
        "closed_trades": trades,
    }


def run_pairs(*, start_day: date, days: int, transport=None) -> dict:
    if not isinstance(days, int) or days < 7 or days % 7:
        raise ValueError("days must be a positive multiple of seven")
    names = sorted({symbol for pair in PAIRS.values() for symbol in pair})
    history: dict[str, list[Bar]] = {}
    for symbol in names:
        candles: list[Bar] = []
        for d in range(-1, days):
            candles.extend(fetch_daily(
                symbol, start_day + timedelta(days=d), transport=transport,
            ))
        history[symbol] = candles
    scenarios = {
        "zero_cost_upper_bound": (0.0, 0.0),
        "realistic_estimate": (COMMISSION, ADVERSE_IMPACT),
        "stress_estimate": (0.0008, 0.0007),
    }
    comparisons = {}
    for strategy, (first_symbol, second_symbol) in PAIRS.items():
        comparisons[strategy] = {}
        for scenario, (fee, impact) in scenarios.items():
            comparisons[strategy][scenario] = [
                simulate_pair_week(
                    history[first_symbol], history[second_symbol],
                    start=BARS_PER_DAY + week * 7 * BARS_PER_DAY,
                    end=BARS_PER_DAY + (week + 1) * 7 * BARS_PER_DAY,
                    commission=fee, adverse_impact=impact,
                )
                for week in range(days // 7)
            ]
    return {
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "source": "official Binance USD-M perpetual historical minute OHLCV archive",
        "source_venue_is_bybit": False,
        "venue_quantity_filters_verified": False,
        "real_orderbook_fill_verified": False,
        "paired_atomic_fills_verified": False,
        "funding_included": False,
        "tiered_liquidation_model_verified": False,
        "independent_holdout": False,
        "max_notional_fraction_of_account": MAX_TOTAL_NOTIONAL_FRACTION,
        "max_leverage": MAX_LEVERAGE,
        "starting_equity_each_week_usdt": STARTING_USDT,
        "symbols": names,
        "start_day": start_day.isoformat(),
        "days": days,
        "fixed_pair_design": PAIRS,
        "strategies": {
            name: by_cost["realistic_estimate"]
            for name, by_cost in comparisons.items()
        },
        "cost_scenario_comparison": comparisons,
        "limitations": [
            "Paired fills are assumed simultaneous; real one-leg-only fills are not modeled.",
            "Each leg is approximately $5 at 50 USDT equity, potentially below venue minimums.",
            "Real Bybit quantity filters, book liquidity and mark-triggered stops are unverified.",
            "Two-sided funding, liquidations, maker queue position and adverse selection are not modeled.",
            "Signals are completed 1m candle z-scores; bar-open fills have modeled impact only.",
            "This is an exploratory historical sample, not a fresh independent holdout.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 11, 1))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_pairs(start_day=args.start, days=args.days)
    except Exception as exc:
        report = {
            "status": "not_completed",
            "error": f"{type(exc).__name__}: {exc}",
            "research_only": True,
            "orders_submitted": False,
            "demo_ready": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    if report.get("status") == "not_completed":
        print(json.dumps(report))
        raise SystemExit(2)
    for name, by_cost in report["cost_scenario_comparison"].items():
        for scenario, weeks in by_cost.items():
            for j, week in enumerate(weeks):
                print(json.dumps({
                    "strategy": name, "scenario": scenario, "week": j + 1,
                    "trades": week["trades"], "win_rate_pct": week["win_rate_pct"],
                    "net_return_pct": week["net_return_pct"],
                    "final_usdt": week["final_equity_usdt"],
                    "real_trading": False,
                }))


if __name__ == "__main__":
    main()
