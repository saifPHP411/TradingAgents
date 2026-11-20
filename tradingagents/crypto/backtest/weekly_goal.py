"""Research-only audit of 50-to-1000 USDT in seven days and >=70% wins.

Uses a single shared-account equity curve and net closed trades. The script
never invents missing returns or authorizes actual trading.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

START_USDT = 50.0
TARGET_USDT = 1000.0
TARGET_MULTIPLE = TARGET_USDT / START_USDT
WIN_RATE_MIN = 70.0
DAYS_PER_WEEK = 7
MIN_WEEKLY_TRADES = 30
MIN_INDEPENDENT_WEEKS = 4
REQUIRED_QUALITY_FLAGS = (
    "independent_holdout",
    "fees_included",
    "slippage_included",
    "funding_included",
    "liquidations_included",
    "unrealized_pnl_included",
    "no_external_cash_flows",
    "venue_filters_enforced",
    "position_risk_limits_enforced",
    "no_lookahead",
    "order_fills_verified_or_conservatively_modeled",
)


def _time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return stamp.astimezone(UTC)


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def audit_weekly_goal(report: dict) -> dict:
    """Audit all nonoverlapping seven-day periods; never select the best week."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if "weekly_runs" in report:
        return audit_isolated_weekly_trials(report)
    if report.get("initial_equity_usdt") != START_USDT:
        raise ValueError("initial_equity_usdt must equal 50")
    raw_snapshots = report.get("account_equity")
    raw_trades = report.get("closed_trades")
    if not isinstance(raw_snapshots, list) or not isinstance(raw_trades, list):
        raise ValueError("account_equity and closed_trades lists are required")
    if len(raw_snapshots) < DAYS_PER_WEEK + 1:
        raise ValueError("at least eight daily account equity snapshots are required")

    snapshots: list[tuple[datetime, float, int]] = []
    for index, raw in enumerate(raw_snapshots):
        if not isinstance(raw, dict):
            raise ValueError("account equity snapshots must be objects")
        stamp = _time(raw.get("timestamp"), f"account_equity[{index}].timestamp")
        if stamp != stamp.replace(hour=0, minute=0, second=0, microsecond=0):
            raise ValueError("account_equity snapshots must be at UTC midnight")
        equity = _number(raw.get("equity_usdt"), "equity_usdt", nonnegative=True)
        open_positions = raw.get("open_positions_count")
        if (isinstance(open_positions, bool) or not isinstance(open_positions, int)
                or open_positions < 0):
            raise ValueError("open_positions_count must be a nonnegative integer")
        if snapshots and stamp - snapshots[-1][0] != timedelta(days=1):
            raise ValueError("account_equity timestamps must be daily, consecutive and unique")
        snapshots.append((stamp, equity, open_positions))
    if not math.isclose(snapshots[0][1], START_USDT, abs_tol=1e-9, rel_tol=0):
        raise ValueError("first account equity must be 50 USDT")
    if any(equity == 0 for _, equity, _ in snapshots[:-1]):
        raise ValueError("zero interim equity indicates bankruptcy; cannot continue weeks")

    trades: list[tuple[datetime, float, bool]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_trades):
        if not isinstance(raw, dict):
            raise ValueError("closed_trades must contain objects")
        identifier = raw.get("trade_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("trade_id must be unique and nonempty")
        identifiers.add(identifier)
        when = _time(raw.get("closed_at"), f"closed_trades[{index}].closed_at")
        if not snapshots[0][0] <= when < snapshots[-1][0]:
            raise ValueError("closed trade lies outside the account snapshot interval")
        net = _number(raw.get("net_pnl_usdt"), "net_pnl_usdt")
        liquidated = raw.get("liquidated", False)
        if not isinstance(liquidated, bool):
            raise ValueError("liquidated must be boolean")
        trades.append((when, net, liquidated))

    quality = report.get("data_quality")
    if not isinstance(quality, dict):
        quality = {}
    if any(key in quality and not isinstance(quality[key], bool) for key in REQUIRED_QUALITY_FLAGS):
        raise ValueError("data_quality flags must be boolean")
    missing_quality = [key for key in REQUIRED_QUALITY_FLAGS if quality.get(key) is not True]
    windows: list[dict] = []
    end_index = ((len(snapshots) - 1) // DAYS_PER_WEEK) * DAYS_PER_WEEK
    for start in range(0, end_index, DAYS_PER_WEEK):
        begin, opening, start_positions = snapshots[start]
        end, closing, end_positions = snapshots[start + DAYS_PER_WEEK]
        week_trades = [(pnl, liquidated) for when, pnl, liquidated in trades if begin <= when < end]
        count = len(week_trades)
        wins = sum(pnl > 0 for pnl, _ in week_trades)
        liquidations = sum(bool(liquidated) for _, liquidated in week_trades)
        win_rate = wins / count * 100 if count else None
        growth = closing / opening if opening else 0.0
        week_points = [equity for _, equity, _ in snapshots[start:start + DAYS_PER_WEEK + 1]]
        trade_net_pnl = math.fsum(pnl for pnl, _ in week_trades)
        # Cash changes between any two flat checkpoints must be attributable
        # to closed trades. A deposit and later withdrawal can net to zero
        # over seven days while fabricating the buyable size midweek.
        flat_points = [
            (ts, balance) for ts, balance, positions in snapshots[start:start + DAYS_PER_WEEK + 1]
            if positions == 0
        ]
        reconciled = math.isclose(trade_net_pnl, closing - opening, abs_tol=1e-8, rel_tol=1e-12)
        for (t0, balance0), (t1, balance1) in zip(flat_points, flat_points[1:], strict=False):
            interval_net = math.fsum(pnl for when, pnl, _ in trades if t0 <= when < t1)
            reconciled &= math.isclose(interval_net, balance1 - balance0, abs_tol=1e-8, rel_tol=1e-12)
        flat_boundaries = start_positions == 0 and end_positions == 0
        peak = opening
        sampled_dd = 0.0
        for equity in week_points:
            peak = max(peak, equity)
            sampled_dd = max(sampled_dd, (peak - equity) / peak if peak > 0 else 1.0)
        pass_goal = (
            count >= MIN_WEEKLY_TRADES
            and win_rate is not None and win_rate >= WIN_RATE_MIN
            and math.isclose(opening, START_USDT, abs_tol=1e-9, rel_tol=0)
            and closing >= TARGET_USDT
            and liquidations == 0
            and flat_boundaries
            and reconciled
        )
        windows.append({
            "week_start": begin.isoformat(),
            "week_end_exclusive": end.isoformat(),
            "opening_equity_usdt": opening,
            "closing_equity_usdt": closing,
            "normalized_50_usdt_final": START_USDT * growth,
            "net_return_pct": (growth - 1) * 100,
            "win_rate_pct": win_rate,
            "closed_trades": count,
            "winning_trades": wins,
            "liquidations": liquidations,
            "account_trade_pnl_reconciled": reconciled,
            "realized_net_pnl_usdt": trade_net_pnl,
            "flat_week_boundaries": flat_boundaries,
            "sampled_daily_max_drawdown_pct": sampled_dd * 100,
            "weekly_target_met": pass_goal,
        })

    qualifying = sum(week["weekly_target_met"] for week in windows)
    sample_sufficient = len(windows) >= MIN_INDEPENDENT_WEEKS
    all_weeks_pass = bool(windows) and qualifying == len(windows)
    evidence_pass = sample_sufficient and all_weeks_pass and not missing_quality
    source = str(report.get("source") or "")
    if not windows:
        status = "insufficient_period"
    elif evidence_pass and source == "bybit_demo_confirmed":
        status = "reported_demo_goal_passed_not_independently_verified"
    elif evidence_pass:
        status = "reported_historical_goal_passed_not_forward_verified"
    else:
        status = "goal_not_validated"
    return {
        "schema_version": 1,
        "status": status,
        "research_only": True,
        "orders_submitted": False,
        "auto_deployment_allowed": False,
        "objective": {
            "start_usdt": START_USDT, "end_usdt": TARGET_USDT,
            "seven_day_growth_multiple": TARGET_MULTIPLE,
            "required_win_rate_pct": WIN_RATE_MIN,
            "minimum_closed_trades_per_week": MIN_WEEKLY_TRADES,
            "minimum_independent_weeks": MIN_INDEPENDENT_WEEKS,
        },
        "source": source or None,
        "evaluated_complete_disjoint_weeks": len(windows),
        "weeks_meeting_both_targets": qualifying,
        "all_complete_weeks_pass": all_weeks_pass,
        "quality_claims_missing_or_unverified": missing_quality,
        "reported_evidence_gate_passed": evidence_pass,
        "independently_verified_by_this_tool": False,
        "ignored_partial_days": len(snapshots) - 1 - end_index,
        "windows": windows,
        "limitations": [
            "Quality flags are reporter claims, not independently authenticated by this script.",
            "Meeting this gate does not independently verify venue fills, capital limits, or source data.",
            "Snapshots must be from one shared account, not merged independent per-symbol runs.",
            "Daily samples understate intraday drawdown and omit some margin breaches.",
            "Trade PnL must be net of costs; win rate alone does not establish profitability.",
            "A week only qualifies when both boundaries are flat and trade PnL reconciles to equity.",
            "A single best week or development-only search never establishes strategy quality.",
        ],
    }



def audit_isolated_weekly_trials(report: dict) -> dict:
    """Audit independent 50-USDT replays for chronologically ordered, disjoint weeks.

    This is the correct normalization when a weekly strategy replay starts
    fresh at 50 USDT each week (rather than compounding into millions).
    """
    raw = report.get("weekly_runs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("weekly_runs must be a nonempty list")
    if any(not isinstance(item, dict) for item in raw):
        raise ValueError("weekly_runs must contain account trial objects")
    if report.get("initial_equity_usdt") != START_USDT:
        raise ValueError("initial_equity_usdt must equal 50")
    shared_ids: set[str] = set()
    windows = []
    previous_end = None
    missing_by_week: set[str] = set()
    for index, week in enumerate(raw):
        snaps = week.get("account_equity")
        trades = week.get("closed_trades")
        if not isinstance(snaps, list) or len(snaps) != 8:
            raise ValueError("each isolated weekly trial must have exactly eight daily snapshots")
        if not isinstance(trades, list):
            raise ValueError("each weekly trial needs a closed-trades ledger")
        for trade in trades:
            if not isinstance(trade, dict):
                raise ValueError("closed trade must be an object")
            trade_id = trade.get("trade_id")
            if not isinstance(trade_id, str) or not trade_id or trade_id in shared_ids:
                raise ValueError("trade_id must be globally unique across weekly trials")
            shared_ids.add(trade_id)
        quality = report.get("data_quality", {})
        if not isinstance(quality, dict):
            raise ValueError("data_quality must be an object")
        week_quality = week.get("data_quality", {})
        if not isinstance(week_quality, dict):
            raise ValueError("weekly data_quality must be an object")
        effective_quality = quality.copy()
        for key in REQUIRED_QUALITY_FLAGS:
            if key in week_quality:
                if not isinstance(week_quality[key], bool):
                    raise ValueError("weekly data_quality flags must be boolean")
                # A week can withdraw a global claim but cannot override a
                # missing or negative claim about the whole protocol.
                if not week_quality[key]:
                    effective_quality[key] = False
        result = audit_weekly_goal({
            "initial_equity_usdt": START_USDT,
            "source": report.get("source"),
            "data_quality": effective_quality,
            "account_equity": snaps,
            "closed_trades": trades,
        })
        missing_by_week.update(result["quality_claims_missing_or_unverified"])
        window = result["windows"][0]
        start = _time(window["week_start"], "week_start")
        end = _time(window["week_end_exclusive"], "week_end")
        if previous_end is not None and start < previous_end:
            raise ValueError("isolated weekly trials must be chronological and nonoverlapping")
        if start >= end:
            raise ValueError("weekly trial has invalid date range")
        previous_end = end
        window["trial_number"] = index + 1
        window["capital_reset_for_independent_trial"] = True
        windows.append(window)
    missing_quality = [key for key in REQUIRED_QUALITY_FLAGS if key in missing_by_week]
    passed_weeks = sum(item["weekly_target_met"] for item in windows)
    reported_pass = (
        len(windows) >= MIN_INDEPENDENT_WEEKS
        and passed_weeks == len(windows)
        and not missing_quality
    )
    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "auto_deployment_allowed": False,
        "independently_verified_by_this_tool": False,
        "source": report.get("source"),
        "account_mode": "independent_50_usdt_weekly_replays",
        "objective": {
            "start_usdt": START_USDT, "end_usdt": TARGET_USDT,
            "seven_day_growth_multiple": TARGET_MULTIPLE,
            "required_win_rate_pct": WIN_RATE_MIN,
            "minimum_closed_trades_per_week": MIN_WEEKLY_TRADES,
            "minimum_independent_weeks": MIN_INDEPENDENT_WEEKS,
        },
        "status": (
            "reported_historical_goal_passed_not_forward_verified"
            if reported_pass else "goal_not_validated"
        ),
        "evaluated_complete_disjoint_weeks": len(windows),
        "weeks_meeting_both_targets": passed_weeks,
        "all_complete_weeks_pass": passed_weeks == len(windows),
        "quality_claims_missing_or_unverified": missing_quality,
        "reported_evidence_gate_passed": reported_pass,
        "ignored_partial_days": 0,
        "windows": windows,
        "limitations": [
            "Each isolated weekly replay restarts one account from 50 USDT.",
            "All weeks are reported; the best week is never selected.",
            "Self-reported quality flags and fills need independent validation.",
            "Daily equity snapshots can miss intraday losses and liquidation risks.",
            "This script never authorizes real or Demo trading.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit_weekly_goal(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "weeks": result["evaluated_complete_disjoint_weeks"],
        "weeks_meeting_both_targets": result["weeks_meeting_both_targets"],
        "reported_evidence_gate_passed": result["reported_evidence_gate_passed"],
        "orders_submitted": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
