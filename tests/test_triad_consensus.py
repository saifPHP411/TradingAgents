"""No-network audit tests for all three unanimity voters and research compatibility."""
from __future__ import annotations

from dataclasses import replace

import pytest

from tradingagents.crypto.triad_consensus import (
    TriadInput,
    evaluate_triad,
)


def opportunity(**updates):
    data = TriadInput(
        symbol="BTCUSDT", candle_end_ms=9_000_000, now_ms=9_001_000,
        direction=1, close=100, close_15m_ago=99.9,
        ema60=100.2, ema240=99.9, rsi=66,
        mark=100, mark_ts_ms=9_000_900, best_bid=99.99,
        best_ask=100.01, book_ts_ms=9_000_950,
        account_equity_usdt=50, requested_notional_usdt=10,
        venue_filters_verified=False,
    )
    return replace(data, **updates)


def test_all_three_independently_approve_same_completed_signal_but_not_live_trading():
    result = evaluate_triad(opportunity())
    assert result["event_id"] == "BTCUSDT:9000000:1"
    assert set(result["votes"]) == {"sniper", "analyst", "professor"}
    assert all(vote["approve"] for vote in result["votes"].values())
    assert all(vote["direction"] == 1 for vote in result["votes"].values())
    assert result["all_three_approve"]
    assert not result["execution_authorized"]
    assert not result["venue_filters_verified"]


@pytest.mark.parametrize(
    ("fields", "role"),
    [
        ({"direction": 0}, "sniper"),
        ({"candle_end_ms": 9_000_001}, "sniper"),
        ({"now_ms": 9_060_000}, "sniper"),
        ({"rsi": 25}, "analyst"),
        ({"ema60": 99.2, "ema240": 100.2}, "analyst"),
        ({"close_15m_ago": 101}, "analyst"),
        ({"mark_ts_ms": 8_994_999}, "professor"),
        ({"book_ts_ms": 8_994_999}, "professor"),
        ({"best_bid": 99, "best_ask": 99.01}, "professor"),
        ({"requested_notional_usdt": 11}, "professor"),
        ({"max_leverage": 4}, "professor"),
        ({"modeled_cost_per_side": .005}, "professor"),
    ],
)
def test_any_role_can_veto(fields, role):
    result = evaluate_triad(opportunity(**fields))
    assert not result["all_three_approve"]
    assert result["votes"][role]["approve"] is False
    assert result["execution_authorized"] is False


def test_short_requires_momentum_and_regime_agreement():
    short = opportunity(
        direction=-1, close=100, close_15m_ago=100.1,
        ema60=99.7, ema240=100.1, rsi=37,
    )
    assert evaluate_triad(short)["all_three_approve"]
    assert not evaluate_triad(replace(short, ema60=100.2))["all_three_approve"]


def test_unverified_filters_cannot_enable_execution_even_if_flag_were_true():
    model = evaluate_triad(opportunity(venue_filters_verified=True))
    assert model["all_three_approve"]
    assert model["execution_authorized"] is False


@pytest.mark.parametrize(
    "data",
    [
        {"mark": float("nan")},
        {"best_ask": 0},
        {"requested_notional_usdt": float("inf")},
        {"modeled_cost_per_side": -0.01},
        {"mark_ts_ms": 9_000.5},
        {"direction": 2},
    ],
)
def test_malformed_evidence_fails_closed_without_synthetic_fills(data):
    with pytest.raises(ValueError):
        evaluate_triad(opportunity(**data))
