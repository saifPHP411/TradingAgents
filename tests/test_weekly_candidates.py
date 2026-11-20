"""Synthetic, no-network tests for the fixed weekly 1m candidate study."""
from __future__ import annotations

import io
import math
import zipfile
from datetime import UTC, date, datetime

import pytest

from tradingagents.crypto.backtest import weekly_candidates
from tradingagents.crypto.backtest.weekly_candidates import (
    Bar,
    fetch_daily,
    indicators,
    parse_candles,
    signal,
    simulate_week,
)

BASE_MS = int(datetime(2026, 10, 31, tzinfo=UTC).timestamp() * 1000)


def sample_bars(count=11520):
    bars = []
    for i in range(count):
        price = 100 + 0.25 * math.sin(i / 9) + 0.01 * i / 2000
        bars.append(Bar(BASE_MS + i * 60000, price, price + 0.08,
                        price - 0.08, price, 1000.0))
    return bars


def one_day_csv(day):
    first = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
    return "".join(
        f"{first + i * 60000},100,101,99,100,1000,0,0,0,0,0,0\n"
        for i in range(1440)
    ).encode()


def test_parse_candle_archive_requires_complete_utc_day():
    day = date(2026, 11, 1)
    rows = parse_candles(one_day_csv(day), day)
    assert len(rows) == 1440
    assert rows[0].open == 100
    broken = one_day_csv(day).decode().splitlines(keepends=True)
    del broken[22]
    with pytest.raises(ValueError, match="consecutive"):
        parse_candles("".join(broken).encode(), day)


def test_fetch_archive_uses_injected_read_only_transport():
    day = date(2026, 11, 1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("XRPUSDT-1m-2026-11-01.csv", one_day_csv(day))
    visited = []
    rows = fetch_daily("XRPUSDT", day, transport=lambda url: (
        visited.append(url) or buf.getvalue()
    ))
    assert len(rows) == 1440
    assert visited[0].startswith("https://data.binance.vision/data/futures/um/daily/")


def test_signals_cannot_peek_into_next_minute():
    bars = sample_bars(300)
    before = signal(bars, indicators(bars), 155, "donchian20_breakout")
    bars[157] = Bar(bars[157].ms, 100, 900, 1, 800, 10000)
    after = signal(bars, indicators(bars), 155, "donchian20_breakout")
    assert before == after


def test_same_bar_stop_and_target_is_conservative(monkeypatch):
    bars = sample_bars()
    i = 1440
    bars[i] = Bar(bars[i].ms, 100, 102, 98, 100, 1000)
    monkeypatch.setattr(
        weekly_candidates, "signal",
        lambda bars, features, index, strategy: 1 if index == 1439 else 0,
    )
    result = simulate_week({"XRPUSDT": bars},
                           "ema9_ema21_rsi_momentum", start=1440, end=11520)
    assert result["trades"] == 1
    assert result["closed_trades"][0]["exit_reason"] == "entry_bar_stop"
    assert result["closed_trades"][0]["net_pnl_usdt"] < 0
    assert result["final_equity_usdt"] < 50


def test_one_account_week_closes_at_boundary_and_reconciles():
    bars = sample_bars()
    result = simulate_week(
        {"XRPUSDT": bars, "BTCUSDT": bars},
        "bollinger_rsi_reversion", start=1440, end=11520,
    )
    assert len(result["account_equity"]) == 8
    assert result["account_equity"][0]["equity_usdt"] == 50
    assert result["account_equity"][-1]["open_positions_count"] == 0
    net = sum(trade["net_pnl_usdt"] for trade in result["closed_trades"])
    assert result["final_equity_usdt"] == pytest.approx(50 + net)


def test_rejects_time_mismatch_between_symbols():
    bars = sample_bars()
    mismatched = list(bars)
    item = mismatched[-1]
    mismatched[-1] = Bar(item.ms + 60000, item.open, item.high,
                         item.low, item.close, item.volume)
    with pytest.raises(ValueError, match="exact minute"):
        simulate_week({"BTCUSDT": bars, "XRPUSDT": mismatched},
                      "donchian20_breakout", start=1440, end=11520)


def test_zero_cost_ablation_vs_realistic_on_same_forced_loss(monkeypatch):
    bars = sample_bars()
    index = 1440
    bars[index] = Bar(bars[index].ms, 100, 102, 98, 100, 1000)
    monkeypatch.setattr(
        weekly_candidates, "signal",
        lambda _bars, _features, i, _strategy: 1 if i == 1439 else 0,
    )
    realistic = simulate_week(
        {"BTCUSDT": bars}, "ema9_ema21_rsi_momentum",
        start=1440, end=11520,
    )
    zero_cost = simulate_week(
        {"BTCUSDT": bars}, "ema9_ema21_rsi_momentum",
        start=1440, end=11520,
        commission_rate=0, adverse_impact_rate=0,
    )
    assert realistic["trades"] == zero_cost["trades"] == 1
    assert zero_cost["final_equity_usdt"] > realistic["final_equity_usdt"]
    with pytest.raises(ValueError, match="commission or adverse"):
        simulate_week(
            {"BTCUSDT": bars}, "ema9_ema21_rsi_momentum",
            start=1440, end=11520, commission_rate=float("nan"),
        )


@pytest.mark.parametrize(
    ("strategy", "misaligned_index"),
    [
        ("ema60_rsi_pullback", 246),
        ("donchian_15m_trend", 246),
        ("vwap_15m_reclaim", 246),
    ],
)
def test_predeclared_multitimeframe_candidates_wait_for_bar_close(strategy, misaligned_index):
    bars = sample_bars(360)
    features = indicators(bars)
    assert signal(bars, features, misaligned_index, strategy) == 0


@pytest.mark.parametrize(
    "strategy",
    ["ema60_rsi_pullback", "donchian_15m_trend", "vwap_15m_reclaim"],
)
def test_new_candidates_do_not_peek_at_future_candles(strategy):
    bars = sample_bars(400)
    index = 284  # last minute of a completed 15m bar
    before = signal(bars, indicators(bars), index, strategy)
    bars[285] = Bar(bars[285].ms, 100, 1000, 1, 500, 100000)
    after = signal(bars, indicators(bars), index, strategy)
    assert before == after


def test_new_candidates_have_longer_hold_limits_and_valid_settings():
    assert set(weekly_candidates.HOLD_BY_STRATEGY).issubset(weekly_candidates.STRATEGIES)
    assert weekly_candidates.HOLD_BY_STRATEGY["donchian_15m_trend"] == 240
    assert weekly_candidates.MAX_NOTIONAL_FRACTION == 0.2
    bars = sample_bars(300)
    assert signal(bars, indicators(bars), 119, "donchian_15m_trend") == 0


def test_daily_vwap_only_contains_current_session_prices():
    bars = sample_bars(1442)
    bars[1439] = Bar(bars[1439].ms, 100, 110, 90, 100, 1000)
    # By the next UTC day cumulative VWAP must restart.
    bars[1440] = Bar(bars[1440].ms, 200, 202, 198, 200, 1000)
    features = indicators(bars)
    assert features["vwap"][1440] == pytest.approx(200)
