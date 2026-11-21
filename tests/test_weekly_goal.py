"""Synthetic tests only — winning examples are not observed market performance."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.crypto.backtest.weekly_goal import audit_weekly_goal

T0 = datetime(2026, 10, 1, tzinfo=UTC)
QUALITY = {
    "independent_holdout": True,
    "fees_included": True,
    "slippage_included": True,
    "funding_included": True,
    "liquidations_included": True,
    "unrealized_pnl_included": True,
    "no_external_cash_flows": True,
    "venue_filters_enforced": True,
    "position_risk_limits_enforced": True,
    "no_lookahead": True,
    "order_fills_verified_or_conservatively_modeled": True,
}


def fixture_report(
    *, weeks=4, multiples=None, wins=21, trades_per_week=30,
    quality=None, source="historical_replay",
):
    if multiples is None:
        multiples = [20] * weeks
    if quality is None:
        quality = QUALITY.copy()
    checkpoints = [{"timestamp": T0.isoformat(), "equity_usdt": 50,
                    "open_positions_count": 0}]
    trades = []
    opening = 50.0
    for week in range(weeks):
        closing = opening * multiples[week]
        for day in range(1, 8):
            checkpoints.append({
                "timestamp": (T0 + timedelta(days=7 * week + day)).isoformat(),
                "equity_usdt": closing if day == 7 else opening,
                "open_positions_count": 0,
            })
        # Balanced synthetic trade PnLs exactly reconcile with the account delta.
        losses = trades_per_week - wins
        loser_net = -1.0 if losses else 0.0
        winner_net = (closing - opening - losses * loser_net) / wins if wins else 0.0
        for trade in range(trades_per_week):
            trades.append({
                "trade_id": f"week{week}-trade{trade}",
                "closed_at": (
                    T0 + timedelta(days=7 * week, hours=2 + trade * 5)
                ).isoformat(),
                "net_pnl_usdt": winner_net if trade < wins else loser_net,
                "liquidated": False,
            })
        opening = closing
    return {
        "initial_equity_usdt": 50,
        "source": source,
        "data_quality": quality,
        "account_equity": checkpoints,
        "closed_trades": trades,
    }


def test_four_synthetic_weeks_can_meet_goal_but_never_auto_trade():
    audit = audit_weekly_goal(fixture_report())
    assert audit["reported_evidence_gate_passed"]
    assert audit["evaluated_complete_disjoint_weeks"] == 4
    assert audit["weeks_meeting_both_targets"] == 4
    assert audit["windows"][0]["normalized_50_usdt_final"] == pytest.approx(1000)
    assert audit["windows"][0]["win_rate_pct"] == pytest.approx(70)
    assert audit["windows"][0]["net_return_pct"] == pytest.approx(1900)
    assert audit["status"] == "reported_historical_goal_passed_not_forward_verified"
    assert not audit["independently_verified_by_this_tool"]
    assert not audit["auto_deployment_allowed"]
    assert not audit["orders_submitted"]


def test_one_lucky_week_is_not_independent_validation():
    audit = audit_weekly_goal(fixture_report(weeks=1))
    assert audit["weeks_meeting_both_targets"] == 1
    assert audit["status"] == "goal_not_validated"
    assert not audit["reported_evidence_gate_passed"]


def test_outcome_selection_fails_when_other_weeks_lose():
    audit = audit_weekly_goal(fixture_report(multiples=[20, 0.98, 1.02, 1.01]))
    assert audit["weeks_meeting_both_targets"] == 1
    assert not audit["all_complete_weeks_pass"]
    assert not audit["reported_evidence_gate_passed"]


def test_seventy_percent_wins_does_not_imply_twentyfold_return():
    audit = audit_weekly_goal(fixture_report(multiples=[1.02] * 4))
    assert all(row["win_rate_pct"] == pytest.approx(70) for row in audit["windows"])
    assert audit["weeks_meeting_both_targets"] == 0


def test_small_or_low_accuracy_samples_fail():
    few = audit_weekly_goal(fixture_report(trades_per_week=9, wins=8))
    assert not few["reported_evidence_gate_passed"]
    assert all(row["closed_trades"] == 9 for row in few["windows"])
    low = audit_weekly_goal(fixture_report(wins=20))
    assert all(row["win_rate_pct"] < 70 for row in low["windows"])
    assert not low["reported_evidence_gate_passed"]


def test_missing_cost_evidence_prevents_approval():
    quality = QUALITY.copy()
    quality["slippage_included"] = False
    audit = audit_weekly_goal(fixture_report(quality=quality))
    assert audit["weeks_meeting_both_targets"] == 4
    assert "slippage_included" in audit["quality_claims_missing_or_unverified"]
    assert not audit["reported_evidence_gate_passed"]


def test_deposits_or_mismatch_between_trades_and_equity_cannot_pass():
    sample = fixture_report()
    sample["account_equity"][-1]["equity_usdt"] *= 2
    audit = audit_weekly_goal(sample)
    assert not audit["windows"][-1]["account_trade_pnl_reconciled"]
    assert not audit["reported_evidence_gate_passed"]


def test_overnight_positions_at_week_boundary_prevent_passing():
    sample = fixture_report()
    sample["account_equity"][7]["open_positions_count"] = 1
    audit = audit_weekly_goal(sample)
    assert not audit["windows"][0]["flat_week_boundaries"]
    assert not audit["windows"][1]["flat_week_boundaries"]
    assert not audit["reported_evidence_gate_passed"]


def test_liquidation_disqualifies_a_week():
    sample = fixture_report()
    sample["closed_trades"][0]["liquidated"] = True
    audit = audit_weekly_goal(sample)
    assert audit["windows"][0]["liquidations"] == 1
    assert not audit["reported_evidence_gate_passed"]


def test_rejects_missing_daily_sample_or_non_utc_midnight():
    sample = fixture_report()
    del sample["account_equity"][4]
    with pytest.raises(ValueError, match="daily"):
        audit_weekly_goal(sample)
    sample = fixture_report()
    sample["account_equity"][1]["timestamp"] = "2026-10-02T15:00:00+03:00"
    with pytest.raises(ValueError, match="UTC midnight"):
        audit_weekly_goal(sample)


def test_rejects_duplicate_trade_ids_and_invalid_numeric_values():
    sample = fixture_report()
    sample["closed_trades"][1]["trade_id"] = sample["closed_trades"][0]["trade_id"]
    with pytest.raises(ValueError, match="unique"):
        audit_weekly_goal(sample)
    sample = fixture_report()
    sample["account_equity"][5]["equity_usdt"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        audit_weekly_goal(sample)


def isolated_weekly_trials():
    runs = []
    for week_number in range(4):
        standalone = fixture_report(weeks=1)
        shift = timedelta(days=7 * week_number)
        for item in standalone["account_equity"]:
            item["timestamp"] = (
                datetime.fromisoformat(item["timestamp"]) + shift
            ).isoformat()
        for item in standalone["closed_trades"]:
            item["closed_at"] = (
                datetime.fromisoformat(item["closed_at"]) + shift
            ).isoformat()
            item["trade_id"] = f"independent-{week_number}-" + item["trade_id"]
        runs.append({
            "account_equity": standalone["account_equity"],
            "closed_trades": standalone["closed_trades"],
        })
    return {
        "initial_equity_usdt": 50,
        "source": "historical_replay",
        "data_quality": QUALITY.copy(),
        "weekly_runs": runs,
    }


def test_independent_50_usdt_weekly_trials_do_not_compound_into_millions():
    report = audit_weekly_goal(isolated_weekly_trials())
    assert report["account_mode"] == "independent_50_usdt_weekly_replays"
    assert report["reported_evidence_gate_passed"]
    assert not report["independently_verified_by_this_tool"]
    assert report["evaluated_complete_disjoint_weeks"] == 4
    assert report["weeks_meeting_both_targets"] == 4
    assert all(week["opening_equity_usdt"] == 50 for week in report["windows"])
    assert all(week["closing_equity_usdt"] == 1000 for week in report["windows"])


def test_isolated_weekly_trials_reject_time_gaps_and_duplicate_ids():
    data = isolated_weekly_trials()
    data["weekly_runs"][1]["account_equity"][0]["timestamp"] = "2026-11-01T00:00:00Z"
    with pytest.raises(ValueError, match="daily"):
        audit_weekly_goal(data)
    data = isolated_weekly_trials()
    data["weekly_runs"][1]["closed_trades"][0]["trade_id"] = (
        data["weekly_runs"][0]["closed_trades"][0]["trade_id"]
    )
    with pytest.raises(ValueError, match="globally unique"):
        audit_weekly_goal(data)
