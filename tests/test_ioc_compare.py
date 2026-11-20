"""Synthetic-only tests for the Bybit IOC replay; no real-trade claims."""
from __future__ import annotations

import csv
import json
import zipfile
from datetime import UTC, datetime

import pytest

from tradingagents.crypto.backtest.ioc_compare import (
    Signal,
    cap_price,
    load_signals,
    replay_archive,
    summarize,
    walk_book,
)

T0 = int(datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)


def signal(**overrides):
    defaults = {
        "signal_id": "test-long",
        "symbol": "XRPUSDT",
        "signal_ts_ms": str(T0 + 2000),
        "side": "long",
        "signal_mark": "100",
        "quantity": "1",
        "tick_size": "0.01",
        "signal_expires_ms": str(T0 + 60000),
        "exit_price": "",
    }
    defaults.update(overrides)
    return Signal.from_row(defaults)


def make_archive(tmp_path, messages, symbol="XRPUSDT"):
    path = tmp_path / f"2026-10-01_{symbol}_ob200.data.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("events.data", "".join(json.dumps(item) + "\n" for item in messages))
    return path


def orderbook(ts, kind="snapshot", bids=None, asks=None, symbol="XRPUSDT"):
    return {
        "ts": ts,
        "type": kind,
        "data": {
            "s": symbol,
            "b": bids if bids is not None else [["99.95", "1"]],
            "a": asks if asks is not None else [
                ["100.05", "0.5"], ["100.10", "0.2"], ["100.20", "1"],
            ],
        },
    }


def test_signal_prices_use_directional_tick_rounding():
    assert cap_price(100.05, 0.1, "long", 10) == 100.1
    assert cap_price(100.05, 0.1, "short", 10) == 100.0


@pytest.mark.parametrize(
    ("side", "levels", "cap", "expected"),
    [
        ("long", [(100.2, 1), (100.05, 0.5), (100.1, 0.2)], 100.1, 0.7),
        ("short", [(99.8, 1), (99.95, 0.5), (99.9, 0.2)], 99.9, 0.7),
    ],
)
def test_limit_ioc_never_fills_outside_price_cap(side, levels, cap, expected):
    result = walk_book(levels, side, 1, cap)
    assert result["filled_quantity"] == pytest.approx(expected)


def test_same_orderbook_gives_more_market_fills_but_worse_price(tmp_path):
    archive = make_archive(tmp_path, [orderbook(T0 + 2300)])
    trades = replay_archive(archive, [signal()], latency_ms=250)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["status"] == "modeled"
    assert trade["entry_cap_price"] == 100.1
    assert trade["market_ioc"]["filled_quantity"] == 1
    assert trade["limit_ioc"]["filled_quantity"] == pytest.approx(0.7)
    assert trade["market_ioc"]["average_price"] > trade["limit_ioc"]["average_price"]
    assert trade["limit_ioc"]["adverse_slippage_bps"] <= 10 + 1e-6
    summary = summarize(trades)
    assert summary["market_ioc"]["full_fill_rate"] == 1
    assert summary["limit_ioc"]["full_fill_rate"] == 0
    assert summary["market_ioc"]["conditional_net_pnl_usdt"] is None


def test_replay_waits_for_new_book_event_and_uses_incremental_updates(tmp_path):
    archive = make_archive(tmp_path, [
        orderbook(T0 + 100),
        orderbook(T0 + 2400, "delta", bids=[], asks=[["100.05", "0"]]),
    ])
    trades = replay_archive(archive, [signal()], latency_ms=250)
    assert trades[0]["book_ts_ms"] == T0 + 2400
    assert trades[0]["limit_ioc"]["filled_quantity"] == pytest.approx(0.2)


def test_signal_expiration_or_stale_book_fails_closed(tmp_path):
    archive = make_archive(tmp_path, [orderbook(T0 + 10000)])
    assert replay_archive(
        archive, [signal(signal_expires_ms=str(T0 + 2500))]
    )[0]["status"] == "unreplayable_book_or_expired"
    assert replay_archive(
        archive, [signal()], max_book_delay_ms=100
    )[0]["status"] == "unreplayable_book_or_expired"


def test_missing_archive_event_is_not_counted_as_unfilled(tmp_path):
    archive = make_archive(tmp_path, [orderbook(T0 + 100)])
    row = replay_archive(archive, [signal()])[0]
    assert row["status"] == "missing_orderbook"
    summary = summarize([row])
    assert summary["modeled_signals"] == 0
    assert summary["market_ioc"]["any_fill_rate"] is None


def test_pnl_only_when_independent_exit_price_provided(tmp_path):
    archive = make_archive(tmp_path, [orderbook(T0 + 2300)])
    reports = replay_archive(archive, [signal(exit_price="101")])
    assert reports[0]["market_ioc"]["conditional_net_pnl_usdt"] is not None
    assert reports[0]["limit_ioc"]["conditional_net_pnl_usdt"] is not None
    assert summarize(reports)["limit_ioc"]["conditional_pnl_samples"] == 1


def test_load_signals_rejects_duplicate_identifiers(tmp_path):
    path = tmp_path / "signals.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[
                "signal_id", "symbol", "signal_ts_ms", "side", "signal_mark",
                "quantity", "tick_size",
            ],
        )
        writer.writeheader()
        row = {
            "signal_id": "duplicate", "symbol": "XRPUSDT",
            "signal_ts_ms": T0, "side": "long", "signal_mark": 100,
            "quantity": 1, "tick_size": 0.01,
        }
        writer.writerow(row)
        writer.writerow(row)
    with pytest.raises(ValueError, match="unique"):
        load_signals(path)
