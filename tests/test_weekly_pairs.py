"""Synthetic, no-network checks for paired long/short historical research."""
from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime

import pytest

from tradingagents.crypto.backtest import weekly_pairs
from tradingagents.crypto.backtest.weekly_pairs import (
    BARS_PER_DAY,
    STARTING_USDT,
    simulate_pair_week,
    zscores_for_pair,
)
from tradingagents.crypto.backtest.weekly_candidates import Bar
from tradingagents.crypto.backtest.weekly_goal_bridge import compare_artifacts


BASE_MS = int(datetime(2026, 10, 3, tzinfo=UTC).timestamp() * 1000)


def bars_for_pair(minutes=8 * BARS_PER_DAY):
    first, second = [], []
    for i in range(minutes):
        p = 100.0 + 0.25 * math.sin(i / 45)
        q = 20.0 + 0.05 * math.sin(i / 55)
        t = BASE_MS + i * 60_000
        first.append(Bar(t, p, p + .1, p - .1, p, 1000.0))
        second.append(Bar(t, q, q + .1, q - .1, q, 1000.0))
    return first, second


def forced_single_pair_signal(monkeypatch, count):
    z = [None] * count
    z[BARS_PER_DAY - 1] = 2.2
    z[BARS_PER_DAY] = 0.1
    monkeypatch.setattr(weekly_pairs, "zscores_for_pair", lambda first, second: z)


def test_zscores_use_only_completed_past_prices():
    a, b = bars_for_pair(450)
    before = zscores_for_pair(a, b)[300]
    assert before is not None
    a[301] = Bar(a[301].ms, 500, 505, 495, 500, 10000.0)
    after = zscores_for_pair(a, b)[300]
    assert after == pytest.approx(before)


def test_paired_entry_and_exit_is_dollar_neutral_with_realistic_costs(monkeypatch):
    a, b = bars_for_pair()
    forced_single_pair_signal(monkeypatch, len(a))
    realistic = simulate_pair_week(
        a, b, start=BARS_PER_DAY, end=8 * BARS_PER_DAY
    )
    no_fees = simulate_pair_week(
        a, b, start=BARS_PER_DAY, end=8 * BARS_PER_DAY,
        commission=0, adverse_impact=0,
    )
    assert realistic["trades"] == no_fees["trades"] == 1
    trade = realistic["closed_trades"][0]
    assert [leg["direction"] for leg in trade["legs"]] == [-1, 1]
    assert trade["exit_reason"] == "spread_reverted"
    assert trade["modeled_two_leg_execution"]
    assert realistic["final_equity_usdt"] < no_fees["final_equity_usdt"]
    assert realistic["final_equity_usdt"] == pytest.approx(
        STARTING_USDT + trade["net_pnl_usdt"]
    )
    assert realistic["account_equity"][0]["equity_usdt"] == 50.0
    assert realistic["account_equity"][-1]["open_positions_count"] == 0
    assert len(realistic["account_equity"]) == 8


def test_paired_time_exit_not_an_unbounded_position(monkeypatch):
    a, b = bars_for_pair()
    z = [None] * len(a)
    z[BARS_PER_DAY - 1] = 2.2
    z[BARS_PER_DAY:] = [1.9] * (len(a) - BARS_PER_DAY)
    monkeypatch.setattr(weekly_pairs, "zscores_for_pair", lambda _a, _b: z)
    result = simulate_pair_week(a, b, start=BARS_PER_DAY, end=8 * BARS_PER_DAY)
    assert result["trades"] == 1
    assert result["closed_trades"][0]["exit_reason"] == "time_exit"


def test_reject_bad_costs_or_cross_symbol_timestamp_mismatch():
    a, b = bars_for_pair()
    with pytest.raises(ValueError, match="invalid execution cost"):
        simulate_pair_week(a, b, start=BARS_PER_DAY, end=8 * BARS_PER_DAY,
                           commission=float("nan"))
    bad = list(b)
    item = bad[4000]
    bad[4000] = Bar(item.ms + 1000, item.open, item.high, item.low,
                    item.close, item.volume)
    with pytest.raises(ValueError, match="timestamps"):
        simulate_pair_week(a, bad, start=BARS_PER_DAY, end=8 * BARS_PER_DAY)
    with pytest.raises(ValueError, match="seven days"):
        simulate_pair_week(a, b, start=BARS_PER_DAY, end=7 * BARS_PER_DAY)


def test_full_readonly_pair_scan_can_be_audited_in_existing_portfolio_gate(
    monkeypatch, tmp_path,
):
    def fake_archive(symbol, day, transport=None):
        base = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        p = {"BTCUSDT": 100.0, "ETHUSDT": 20.0,
             "XRPUSDT": 2.0, "LTCUSDT": 60.0}[symbol]
        return [
            Bar(base + i * 60_000, p, p * 1.001, p * .999, p, 200.0)
            for i in range(1440)
        ]

    monkeypatch.setattr(weekly_pairs, "fetch_daily", fake_archive)
    result = weekly_pairs.run_pairs(start_day=date(2026, 10, 4), days=7)
    assert set(result["strategies"]) == set(weekly_pairs.PAIRS)
    assert result["orders_submitted"] is False
    assert result["paired_atomic_fills_verified"] is False
    assert result["max_notional_fraction_of_account"] == .20
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    audit = compare_artifacts([path])
    assert audit["tested_strategies"] == 2
    assert audit["demo_ready"] is False
    assert all(
        result["weeks_meeting_20x_and_70pct"] == 0
        for scenarios in audit["strategy_comparisons"].values()
        for result in scenarios.values()
    )
