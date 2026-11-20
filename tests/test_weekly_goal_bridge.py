"""No-network bridge tests: a claimed 20x run must remain unverified."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradingagents.crypto.backtest.weekly_goal_bridge import compare_artifacts

BASE = datetime(2026, 10, 4, tzinfo=UTC)
SCENARIOS = ("zero_cost_upper_bound", "realistic_estimate", "stress_estimate")


def candidate_week(start: datetime, *, win: bool) -> dict:
    end = start + timedelta(days=7)
    trades = []
    if win:
        wins, count = 21, 30
        gain = (50.0 + 9.0) / wins
        for i in range(count):
            trades.append({
                "symbol": "BTCUSDT",
                "side": "long",
                "closed_at": (start + timedelta(hours=1, minutes=i * 6)).isoformat(),
                "net_pnl_usdt": gain if i < wins else -1.0,
                "exit_reason": "modeled_exit",
            })
    closing = 100.0 if win else 50.0
    snapshots = [
        {
            "timestamp": (start + timedelta(days=i)).isoformat(),
            "equity_usdt": 50.0 if i == 0 else closing,
            "open_positions_count": 0,
        }
        for i in range(8)
    ]
    return {
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "initial_equity_usdt": 50.0,
        "final_equity_usdt": closing,
        "trades": len(trades),
        "win_rate_pct": 70.0 if win else None,
        "net_return_pct": 100.0 if win else 0.0,
        "account_equity": snapshots,
        "closed_trades": trades,
    }


def artifact(start: datetime, *, number_weeks: int, win: bool = False) -> dict:
    by_scenario = {
        scenario: [
            candidate_week(start + timedelta(days=7 * i), win=win)
            for i in range(number_weeks)
        ]
        for scenario in SCENARIOS
    }
    return {
        "research_only": True,
        "orders_submitted": False,
        "source_venue_is_bybit": False,
        "venue_quantity_filters_verified": False,
        "real_orderbook_fill_verified": False,
        "starting_equity_each_week_usdt": 50.0,
        "max_notional_fraction_of_account": 0.2,
        "max_leverage": 2.0,
        "symbols": ["BTCUSDT", "XRPUSDT"],
        "strategies": {"donchian_15m_trend": by_scenario["realistic_estimate"]},
        "cost_scenario_comparison": {"donchian_15m_trend": by_scenario},
    }


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_four_week_research_stays_unapproved_even_with_50usd_profit_claims(tmp_path):
    path = write(tmp_path / "one.json", artifact(BASE, number_weeks=4, win=True))
    result = compare_artifacts([path])
    assert result["tested_strategies"] == 1
    assert not result["independently_verified_execution"]
    assert not result["demo_ready"]
    realistic = result["strategy_comparisons"]["donchian_15m_trend"]["realistic_estimate"]
    assert realistic["weeks_meeting_50usdt_profit_and_70pct"] == 4
    assert realistic["weeks_meeting_50usdt_profit"] == 4
    assert realistic["closed_trades"] == 120
    assert realistic["audit"]["reported_evidence_gate_passed"] is False
    assert realistic["audit"]["weeks_meeting_both_targets"] == 4
    assert "venue_filters_enforced" in realistic["missing_evidence"]
    assert "funding_included" in realistic["missing_evidence"]
    zero = result["strategy_comparisons"]["donchian_15m_trend"]["zero_cost_upper_bound"]
    assert "fees_included" in zero["missing_evidence"]
    assert "slippage_included" in zero["missing_evidence"]


def test_all_weeks_and_separate_artifacts_are_checked(tmp_path):
    october = write(tmp_path / "oct.json", artifact(BASE, number_weeks=4))
    november = write(
        tmp_path / "nov.json", artifact(BASE + timedelta(days=28), number_weeks=2)
    )
    result = compare_artifacts([october, november])
    checked = result["strategy_comparisons"]["donchian_15m_trend"]["realistic_estimate"]
    assert checked["tested_seven_day_periods"] == 6
    assert checked["weeks_meeting_50usdt_profit_and_70pct"] == 0
    assert checked["weeks_net_positive"] == 0
    assert checked["closed_trades"] == 0


def test_refuses_false_live_status_or_unvalidated_size(tmp_path):
    fake = artifact(BASE, number_weeks=1)
    fake["orders_submitted"] = True
    with pytest.raises(ValueError, match="research-only"):
        compare_artifacts([write(tmp_path / "not-research.json", fake)])
    fake["orders_submitted"] = False
    fake["venue_quantity_filters_verified"] = True
    with pytest.raises(ValueError, match="execution evidence assumptions"):
        compare_artifacts([write(tmp_path / "size.json", fake)])


def test_rejects_missing_trades_and_pnl_mismatch(tmp_path):
    fake = artifact(BASE, number_weeks=1, win=True)
    week = fake["cost_scenario_comparison"]["donchian_15m_trend"]["realistic_estimate"][0]
    week["closed_trades"].pop()
    with pytest.raises(ValueError, match="count"):
        compare_artifacts([write(tmp_path / "missing-trade.json", fake)])
    week["trades"] = len(week["closed_trades"])
    week["final_equity_usdt"] = 99.0
    with pytest.raises(ValueError, match="last snapshot"):
        compare_artifacts([write(tmp_path / "mismatch.json", fake)])


def test_rejects_out_of_order_periods_and_mismatched_symbols(tmp_path):
    first = write(tmp_path / "oct.json", artifact(BASE, number_weeks=4))
    last = artifact(BASE + timedelta(days=28), number_weeks=2)
    last["symbols"] = ["BTCUSDT"]
    with pytest.raises(ValueError, match="symbol universes"):
        compare_artifacts([first, write(tmp_path / "mismatch.json", last)])
    last["symbols"] = ["BTCUSDT", "XRPUSDT"]
    second = write(tmp_path / "nov.json", last)
    with pytest.raises(ValueError, match="chronological"):
        compare_artifacts([second, first])
