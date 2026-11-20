"""Bridge real archived-candle research results into the strict >=50-USDT net profit per week gate.

Uses previously generated JSON from weekly_candidates.py; never reconstructs
missing fills or claims that public Binance OHLCV are real Bybit executions.
Runs without importing the project's runtime or fetching new market data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .weekly_goal import audit_weekly_goal
except ImportError:  # Standalone GitHub Actions script with no third-party dependencies
    from weekly_goal import audit_weekly_goal

REQUIRED_SCENARIOS = (
    "zero_cost_upper_bound",
    "realistic_estimate",
    "stress_estimate",
)
# The experimental scanner does not implement Bybit lot-size requirements,
# funding events, or tiered margin/liquidation. Those facts MUST be failures,
# not favorable assumptions, even if a winning run is provided.
COMMON_QUALITY = {
    "independent_holdout": False,
    "fees_included": True,
    "slippage_included": True,
    "funding_included": False,
    "liquidations_included": False,
    "unrealized_pnl_included": True,
    "no_external_cash_flows": True,
    "venue_filters_enforced": False,
    "position_risk_limits_enforced": True,
    "no_lookahead": True,
    "order_fills_verified_or_conservatively_modeled": False,
}


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("research artifact must be a JSON object")
    if data.get("orders_submitted") is not False or data.get("research_only") is not True:
        raise ValueError("only verifiably research-only artifacts can be audited")
    if data.get("source_venue_is_bybit") is not False:
        raise ValueError("expected a public Binance historical archive, not Bybit fills")
    if data.get("venue_quantity_filters_verified") is not False:
        raise ValueError("unexpected change in execution evidence assumptions")
    if data.get("real_orderbook_fill_verified") is not False:
        raise ValueError("unexpected claim of actual book fills")
    if data.get("starting_equity_each_week_usdt") != 50.0:
        raise ValueError("every weekly portfolio must start at 50 USDT")
    if data.get("max_notional_fraction_of_account") != 0.2:
        raise ValueError("research notional exposure is not the tested 20% cap")
    if data.get("max_leverage") != 2.0:
        raise ValueError("research leverage policy changed")
    scenarios = data.get("cost_scenario_comparison")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("missing strategy/scenario results")
    if data.get("strategies") is None:
        raise ValueError("missing original candidate summary")
    return data


def _normalize_week(week: dict, *, unique_prefix: str) -> dict:
    if not isinstance(week, dict):
        raise ValueError("invalid weekly result")
    start, end = week.get("start"), week.get("end_exclusive")
    if not isinstance(start, str) or not isinstance(end, str) or start >= end:
        raise ValueError("missing or misordered weekly boundary timestamps")
    snapshots = week.get("account_equity")
    trades = week.get("closed_trades")
    if not isinstance(snapshots, list) or not isinstance(trades, list):
        raise ValueError("missing account/trade ledger")
    if week.get("initial_equity_usdt") != 50.0:
        raise ValueError("weekly starting equity is not 50 USDT")
    if len(trades) != week.get("trades"):
        raise ValueError("trade ledger count differs from reported trades")
    if not math.isclose(float(week.get("final_equity_usdt")), float(snapshots[-1]["equity_usdt"]), rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("reported final equity differs from last snapshot")
    if snapshots[0].get("timestamp") != start or snapshots[-1].get("timestamp") != end:
        raise ValueError("weekly account snapshots do not match trial boundaries")
    # Source candidate scan records no trade ids. These are *model event ids*,
    # not fake venue order ids. Never label them as exchange-confirmed fills.
    mapped = []
    for i, trade in enumerate(trades):
        if not isinstance(trade, dict):
            raise ValueError("malformed simulated trade")
        mapped.append({
            "trade_id": f"modeled:{unique_prefix}:{i}",
            "closed_at": trade.get("closed_at"),
            "net_pnl_usdt": trade.get("net_pnl_usdt"),
            "liquidated": False,  # Not an independently verified liquidation field!
        })
    return {
        "account_equity": snapshots,
        "closed_trades": mapped,
    }


def compare_artifacts(paths: list[Path]) -> dict:
    """Audit ALL supplied periods for every predeclared strategy and scenario."""
    if not paths:
        raise ValueError("at least one concrete research artifact is required")
    loaded = [_load(path) for path in paths]
    first = loaded[0]
    names = set(first["cost_scenario_comparison"])
    if any(set(item["cost_scenario_comparison"]) != names for item in loaded):
        raise ValueError("research artifacts contain different strategy families")
    if any(item.get("symbols") != first.get("symbols") for item in loaded):
        raise ValueError("strategy comparisons used inconsistent symbol universes")
    evaluations = {}
    for strategy in sorted(names):
        by_scenario = {}
        for scenario in REQUIRED_SCENARIOS:
            weeks = []
            for dataset in loaded:
                matching = dataset["cost_scenario_comparison"][strategy].get(scenario)
                if not isinstance(matching, list):
                    raise ValueError("incomplete scenario/week result for " + strategy)
                for week in matching:
                    weeks.append(_normalize_week(
                        week,
                        unique_prefix=f"{strategy}:{scenario}:{week.get('start')}",
                    ))
            quality = COMMON_QUALITY.copy()
            if scenario == "zero_cost_upper_bound":
                quality["fees_included"] = False
                quality["slippage_included"] = False
            audit = audit_weekly_goal({
                "initial_equity_usdt": 50.0,
                "source": "binance_usdm_historical_minute_candle_simulation",
                "data_quality": quality,
                "weekly_runs": weeks,
            })
            if audit["reported_evidence_gate_passed"]:
                raise RuntimeError("unverified venue fills must not satisfy the independent evidence gate")
            by_scenario[scenario] = {
                "tested_seven_day_periods": audit["evaluated_complete_disjoint_weeks"],
                "weeks_meeting_50usdt_profit_and_70pct": audit["weeks_meeting_both_targets"],
                "weeks_meeting_50usdt_profit": sum(
                    w["weekly_profit_goal_met"] for w in audit["windows"]
                ),
                "weeks_net_positive": sum(
                    w["net_return_pct"] > 0 for w in audit["windows"]
                ),
                "closed_trades": sum(w["closed_trades"] for w in audit["windows"]),
                "highest_weekly_win_rate_pct": max(
                    (w["win_rate_pct"] or 0.0 for w in audit["windows"]), default=None,
                ),
                "highest_weekly_net_return_pct": max(
                    (w["net_return_pct"] for w in audit["windows"]), default=None,
                ),
                "missing_evidence": audit["quality_claims_missing_or_unverified"],
                "audit": audit,
            }
        evaluations[strategy] = by_scenario

    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "independently_verified_execution": False,
        "dataset": {
            "archive_type": "Binance USD-M 1-minute historical kline ZIP",
            "symbols": first.get("symbols"),
            "periods": len(loaded),
        },
        "tested_strategies": len(evaluations),
        "strategy_comparisons": evaluations,
        "conclusion": "No model can be approved until the missing venue evidence and untouched holdout have been independently established.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_artifacts(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for strategy, costs in report["strategy_comparisons"].items():
        for scenario, item in costs.items():
            print(json.dumps({
                "strategy": strategy,
                "scenario": scenario,
                "weeks": item["tested_seven_day_periods"],
                "positive_weeks": item["weeks_net_positive"],
                "weeks_meeting_50usdt_profit_and_70pct": item["weeks_meeting_50usdt_profit_and_70pct"],
                "weeks_meeting_50usdt_profit": item["weeks_meeting_50usdt_profit"],
                "independently_verified_execution": False,
                "missing_evidence": item["missing_evidence"],
            }, sort_keys=True))


if __name__ == "__main__":
    main()
