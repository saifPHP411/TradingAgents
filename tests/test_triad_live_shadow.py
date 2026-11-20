"""Offline checks: public Bybit messages, 1-second checks, and no trade path."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.crypto.backtest.weekly_candidates import Bar
from tradingagents.crypto.triad_live_shadow import TriadShadowWatcher

START = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp()*1000)
CANDLE_END = START + 300*60_000


def source_bars(count=300):
    return [
        Bar(
            ms=START + i*60_000, open=100, high=100.1, low=99.9,
            close=100 if i < 284 else 99.9 + .01*(i-284),
            volume=1000,
        )
        for i in range(count)
    ]


def feed_quotes(watcher, now):
    watcher.ingest({
        "topic": "tickers.BTCUSDT", "ts": now,
        "data": {"markPrice": "100"},
    }, received_ms=now)
    watcher.ingest({
        "topic": "orderbook.1.BTCUSDT", "type": "snapshot", "ts": now,
        "data": {"s": "BTCUSDT", "b": [["99.99", "100"]], "a": [["100.01", "100"]]},
    }, received_ms=now)


def test_checks_every_second_and_emits_single_unanimous_shadow_event(monkeypatch):
    from tradingagents.crypto.backtest import weekly_candidates as study

    watcher = TriadShadowWatcher(("BTCUSDT",))
    assert watcher.bootstrap("BTCUSDT", source_bars()) == 300
    monkeypatch.setattr(study, "signal", lambda *_args: 1)
    monkeypatch.setattr(
        study, "indicators",
        lambda bars: {
            "ema60": [100.2] * len(bars),
            "ema240": [99.9] * len(bars),
            "rsi": [65] * len(bars),
        },
    )
    # close at 15m boundary, first second after close.
    feed_quotes(watcher, CANDLE_END + 1000)
    result = watcher.tick(now_ms=CANDLE_END + 1000)
    assert result["scans_completed"] == 1
    assert len(result["new_unanimous_shadow_opportunities"]) == 1
    decision = result["new_unanimous_shadow_opportunities"][0]
    assert decision["all_three_approve"]
    assert not decision["execution_authorized"]
    assert not result["orders_submitted"]
    # Repeated second-by-second scans do not open duplicate positions.
    result2 = watcher.tick(now_ms=CANDLE_END + 2000)
    assert result2["scans_completed"] == 2
    assert result2["new_unanimous_shadow_opportunities"] == []
    assert result2["orders_submitted"] is False
    watcher.clear_live_quotes()
    again = watcher.tick(now_ms=CANDLE_END + 3000)
    assert again["new_unanimous_shadow_opportunities"] == []
    assert again["vetoes"][0]["reason"] == "unverified_live_market_data"


def test_stale_quote_vetoes_before_the_first_approval(monkeypatch):
    from tradingagents.crypto.backtest import weekly_candidates as study

    watcher = TriadShadowWatcher(("BTCUSDT",))
    watcher.bootstrap("BTCUSDT", source_bars())
    monkeypatch.setattr(study, "signal", lambda *_args: 1)
    monkeypatch.setattr(
        study, "indicators",
        lambda bars: {
            "ema60": [100.2] * len(bars),
            "ema240": [99.9] * len(bars),
            "rsi": [65] * len(bars),
        },
    )
    feed_quotes(watcher, CANDLE_END)
    result = watcher.tick(now_ms=CANDLE_END + 6000)
    assert result["new_unanimous_shadow_opportunities"] == []
    veto = result["vetoes"][0]
    assert veto["votes"]["professor"]["approve"] is False
    assert veto["book_age_ms"] == 6000


def test_incomplete_candle_does_not_become_a_closed_signal():
    watcher = TriadShadowWatcher(("BTCUSDT",))
    watcher.bootstrap("BTCUSDT", source_bars())
    forming = {
        "topic": "kline.1.BTCUSDT",
        "data": [{
            "start": CANDLE_END, "confirm": False,
            "open": "100", "high": "101", "low": "99",
            "close": "101", "volume": "100",
        }],
    }
    watcher.ingest(forming, received_ms=CANDLE_END + 2000)
    assert len(watcher.history["BTCUSDT"]) == 300
    assert watcher.tick(now_ms=CANDLE_END + 2000)["orders_submitted"] is False


def test_closed_kline_appends_once_and_gap_clears_bootstrap():
    watcher = TriadShadowWatcher(("BTCUSDT",))
    watcher.bootstrap("BTCUSDT", source_bars())
    closed = {
        "topic": "kline.1.BTCUSDT",
        "data": [{
            "start": CANDLE_END, "confirm": True,
            "open": "100", "high": "101", "low": "99",
            "close": "100.2", "volume": "100",
        }],
    }
    watcher.ingest(closed, received_ms=CANDLE_END + 60_000)
    assert len(watcher.history["BTCUSDT"]) == 301
    watcher.ingest(closed, received_ms=CANDLE_END + 60_000)
    assert len(watcher.history["BTCUSDT"]) == 301
    unexpected = {
        **closed,
        "data": [{**closed["data"][0], "start": CANDLE_END + 3*60_000}],
    }
    watcher.ingest(unexpected, received_ms=CANDLE_END + 4*60_000)
    assert len(watcher.history["BTCUSDT"]) == 1
    assert watcher.rejected >= 1
    assert watcher.tick(now_ms=CANDLE_END+4*60_000)["vetoes"][0]["reason"] == (
        "warmup_or_history_gap"
    )


def test_old_future_and_crossed_book_events_fail_without_mutating_market():
    watcher = TriadShadowWatcher(("BTCUSDT",))
    now = CANDLE_END + 2000
    feed_quotes(watcher, now)
    good = watcher.book["BTCUSDT"]
    watcher.ingest({
        "topic": "orderbook.1.BTCUSDT", "ts": now + 5000,
        "data": {"b": [["100.5", "10"]], "a": [["100.4", "5"]]},
    }, received_ms=now+5000)
    assert watcher.book["BTCUSDT"] == good
    watcher.ingest({
        "topic": "tickers.BTCUSDT", "ts": now-1000,
        "data": {"markPrice": "99"},
    }, received_ms=now)
    assert watcher.mark["BTCUSDT"] == (100, now)


def test_history_validator_rejects_missing_minute_and_invalid_symbols():
    with pytest.raises(ValueError, match="unique"):
        TriadShadowWatcher(("BTCUSDT", "BTCUSDT"))
    with pytest.raises(ValueError, match="symbol"):
        TriadShadowWatcher(("BTC/USDT",))
    watcher = TriadShadowWatcher(("BTCUSDT",))
    bars = source_bars()
    with pytest.raises(ValueError, match="gaps"):
        watcher.bootstrap("BTCUSDT", bars[:10] + bars[12:])
