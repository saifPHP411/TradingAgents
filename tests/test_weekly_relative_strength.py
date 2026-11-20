"""Offline regression tests for fixed cross-sectional 24h/4h rank selection."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from tradingagents.crypto.backtest import weekly_relative_strength as study
from tradingagents.crypto.backtest.weekly_candidates import Bar
from tradingagents.crypto.backtest.weekly_goal_bridge import compare_artifacts

START_MS = int(datetime(2026, 7, 30, tzinfo=UTC).timestamp() * 1000)
START = 2 * study.DAY
END = START + 7 * study.DAY


def make_bars(prices, volumes=None):
    return [
        Bar(START_MS + i * 60_000, p, p * 1.001, p * .999, p,
            1000 if volumes is None else volumes[i])
        for i, p in enumerate(prices)
    ]


def one_symbol_constant(price, length=END):
    return make_bars([price] * length)


def test_frozen_daily_cross_sectional_signal_ranks_past_24h_and_4h():
    length = 3 * study.DAY
    def trending(up):
        return [100 * (1 + (0.000024 if up else -0.000024) * i)
                for i in range(length)]
    prices = {
        "BTCUSDT": make_bars(trending(True)),
        "ETHUSDT": make_bars([100] * length),
        "XRPUSDT": make_bars(trending(False)),
    }
    assert study.choose_ranked_signal(
        prices, START, "daily_relative_winner_long"
    ) == ("BTCUSDT", 1)
    assert study.choose_ranked_signal(
        prices, START, "daily_relative_loser_short"
    ) == ("XRPUSDT", -1)
    before = study.choose_ranked_signal(
        prices, START, "daily_relative_winner_long"
    )
    later = prices["BTCUSDT"][START]
    prices["BTCUSDT"][START] = Bar(
        later.ms, 100, 999, 1, 400, 100_000
    )
    assert study.choose_ranked_signal(
        prices, START, "daily_relative_winner_long"
    ) == before


def test_losing_24h_coin_requires_a_real_last_4h_rebound():
    length = 3 * study.DAY
    prices = [100.0] * length
    for i in range(START - 900, START - study.HOURS_4):
        prices[i] = 100 - 6 * (i - (START - 900)) / (900 - study.HOURS_4)
    for i in range(START - study.HOURS_4, START):
        prices[i] = 94 + .8 * (i - (START - study.HOURS_4)) / study.HOURS_4
    history = {
        "BTCUSDT": one_symbol_constant(100, length),
        "ETHUSDT": make_bars(prices),
    }
    assert study.choose_ranked_signal(
        history, START, "daily_relative_loser_rebound_long"
    ) == ("ETHUSDT", 1)
    history["ETHUSDT"] = make_bars(
        [100 if i < START - study.HOURS_4 else 94 for i in range(length)]
    )
    assert study.choose_ranked_signal(
        history, START, "daily_relative_loser_rebound_long"
    ) is None


def test_stop_first_on_entry_candle_reconciles_fees(monkeypatch):
    data = {"BTCUSDT": one_symbol_constant(100),
            "ETHUSDT": one_symbol_constant(100)}
    b = data["BTCUSDT"][START]
    data["BTCUSDT"][START] = Bar(b.ms, 100, 112, 90, 100, 1000)
    monkeypatch.setattr(
        study, "choose_ranked_signal",
        lambda _history, idx, _name: ("BTCUSDT", 1) if idx == START else None,
    )
    result = study.simulate_week(
        data, "daily_relative_winner_long", start=START, end=END
    )
    assert result["trades"] == 1
    assert result["closed_trades"][0]["exit_reason"] == "stop_first"
    assert result["final_equity_usdt"] < 50
    assert result["final_equity_usdt"] == pytest.approx(
        50 + result["closed_trades"][0]["net_pnl_usdt"]
    )
    assert len(result["account_equity"]) == 8


def test_daily_time_exit_and_realistic_cost_ablation(monkeypatch):
    data = {"BTCUSDT": one_symbol_constant(100),
            "ETHUSDT": one_symbol_constant(100)}
    monkeypatch.setattr(
        study, "choose_ranked_signal",
        lambda _history, idx, _name: ("BTCUSDT", 1) if idx == START else None,
    )
    real = study.simulate_week(
        data, "daily_relative_winner_long", start=START, end=END
    )
    free = study.simulate_week(
        data, "daily_relative_winner_long", start=START, end=END,
        commission=0, impact=0,
    )
    assert real["trades"] == free["trades"] == 1
    assert real["closed_trades"][0]["exit_reason"] == "daily_time_exit"
    assert real["final_equity_usdt"] < free["final_equity_usdt"]
    assert free["final_equity_usdt"] == pytest.approx(50)
    with pytest.raises(ValueError, match="fee and impact"):
        study.simulate_week(data, "daily_relative_winner_long", start=START,
                            end=END, impact=float("nan"))


def test_cross_sectional_rejects_misaligned_archives():
    data = {"BTCUSDT": one_symbol_constant(100),
            "ETHUSDT": one_symbol_constant(100)}
    b = data["ETHUSDT"][4000]
    data["ETHUSDT"][4000] = Bar(b.ms + 1, b.open, b.high, b.low,
                                b.close, b.volume)
    with pytest.raises(ValueError, match="timestamps"):
        study.simulate_week(
            data, "daily_relative_winner_long", start=START, end=END
        )


def test_complete_readonly_study_passes_evidence_bridge_without_demo_approval(
    monkeypatch, tmp_path,
):
    def fake_day(symbol, day, transport=None):
        start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        return [
            Bar(start + i * 60_000, 100, 100.1, 99.9, 100, 1000)
            for i in range(1440)
        ]

    monkeypatch.setattr(study, "fetch_daily", fake_day)
    result = study.run_study(
        first_day=date(2026, 8, 2), days=7,
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    path = tmp_path / "study.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    audit = compare_artifacts([path])
    assert audit["tested_strategies"] == 3
    assert not audit["demo_ready"]
    assert all(
        row["weeks_meeting_50usdt_profit"] == 0
        for scenarios in audit["strategy_comparisons"].values()
        for row in scenarios.values()
    )
    assert all(
        week["trades"] == 0
        for scenarios in result["cost_scenario_comparison"].values()
        for weeks in scenarios.values() for week in weeks
    )
