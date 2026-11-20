"""Synthetic-only tests for bounded real-L2 quote proxy IOC benchmark."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, date, datetime

import pytest

from tradingagents.crypto.backtest.ioc_quote_prefix_benchmark import (
    build_quote_proxy_signals,
    compare_prefix,
    zip_prefix_lines,
)

DAY = date(2026, 10, 1)
START = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000) + 2000
URL = "https://quote-saver.bycsi.com/orderbook/linear/XRPUSDT/example.zip"


def make_event(ts, *, snapshot=False, future_spread=False):
    first = (["99.95", "0.05"], ["99.86", "0.10"])
    second = (["100.05", "0.05"], ["100.13", "0.10"])
    if future_spread:
        second = (["105.0", "0.05"], ["106.0", "0.10"])
    return {
        "ts": ts,
        "type": "snapshot" if snapshot else "delta",
        "data": {
            "s": "XRPUSDT",
            "b": [list(x) for x in first] if snapshot else [],
            "a": [list(x) for x in second] if snapshot else (
                [["100.05", "0"]] + [list(x) for x in second] if future_spread else []
            ),
        },
    }


def make_zip(events):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "2026-10-01_XRPUSDT_ob200.data",
            "\n".join(json.dumps(row) for row in events) + "\n",
        )
    return buffer.getvalue()


def fixed_events():
    return [
        make_event(START + i * 1000, snapshot=(i == 0))
        for i in range(185)
    ]


def test_quote_benchmark_uses_real_l2_message_shape_without_strategy_claims():
    report = compare_prefix(
        make_zip(fixed_events()), symbol="XRPUSDT", day=DAY,
        source_url=URL,
    )
    assert report["status"] == "sampled_quote_only_not_strategy_backtest"
    assert report["sampled_quote_proxy_signals"] >= 10
    assert report["parsed_complete_jsonl_records"] == 185
    assert not report["full_day_archive_verified"]
    assert not report["venue_qty_steps_verified"]
    assert not report["demo_ready"]
    assert report["orders_submitted"] is False
    for latency in ("0", "250", "1000"):
        summary = report["latency_sensitivity_ms"][latency]
        assert summary["market_ioc"]["any_fill_rate"] == 1.0
        assert summary["limit_ioc"]["any_fill_rate"] == 1.0
        assert summary["market_ioc"]["full_fill_rate"] == 1.0
        assert summary["limit_ioc"]["full_fill_rate"] == 0.0
        assert summary["extra_market_filled_quantity"] > 0
        assert summary["net_strategy_profit_usdt"] is None
        assert summary["limit_ioc"]["conditional_net_pnl_usdt"] is None


def test_each_quote_midpoint_is_from_historical_current_or_earlier_book():
    rows = fixed_events()
    first = build_quote_proxy_signals(
        [json.dumps(e).encode() for e in rows], symbol="XRPUSDT",
        requested_day=DAY,
    )
    rows[-1]["data"]["a"] = [["105", "1"]]
    after = build_quote_proxy_signals(
        [json.dumps(e).encode() for e in rows], symbol="XRPUSDT",
        requested_day=DAY,
    )
    assert first[0] == after[0]
    assert first[1] == after[1]
    assert all(s.exit_price is None for s in first)
    assert all(s.expires_ms == (s.signal_ts_ms // 60000 + 1) * 60000 for s in first)


def test_rejects_bad_zip_and_wrong_symbol():
    with pytest.raises(ValueError, match="ZIP"):
        zip_prefix_lines(b"invalid archive")
    rows = [json.dumps(e).encode() for e in fixed_events()]
    with pytest.raises(ValueError, match="symbol mismatch"):
        build_quote_proxy_signals(
            rows, symbol="LTCUSDT", requested_day=DAY,
        )


def test_insufficient_quote_history_fails_closed():
    with pytest.raises(ValueError, match="too few"):
        compare_prefix(
            make_zip(fixed_events()[:2]), symbol="XRPUSDT", day=DAY,
            source_url=URL,
        )


def test_out_of_order_events_fail_closed():
    rows = fixed_events()
    rows[-1]["ts"] = START - 1
    with pytest.raises(ValueError, match="out-of-order"):
        build_quote_proxy_signals(
            [json.dumps(e).encode() for e in rows],
            symbol="XRPUSDT", requested_day=DAY,
        )
