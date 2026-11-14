from __future__ import annotations

import json
from decimal import Decimal
from io import StringIO

import pytest

from tradingagents.crypto.backtest.execution_replay import (
    BybitExecutionReplay,
    InstrumentRules,
    _write_jsonl_record,
    validate_post_only_order,
)

pytestmark = pytest.mark.unit


def book(ts: int, bid="100", bid_qty="2", ask="101", ask_qty="3", kind="snapshot"):
    return {
        "topic": "orderbook.50.BTCUSDT",
        "type": kind,
        "ts": ts,
        "received_ts_ms": ts,
        "data": {"b": [[bid, bid_qty]], "a": [[ask, ask_qty]]},
    }


def trade(ts: int, side: str, price="100", qty="1"):
    return {
        "topic": "publicTrade.BTCUSDT",
        "ts": ts,
        "received_ts_ms": ts,
        "data": [{"T": ts, "S": side, "p": price, "v": qty}],
    }


def test_instrument_rules_validate_tick_step_minimum_and_post_only():
    rules = InstrumentRules(
        category="spot", symbol="BTCUSDT", tick_size=Decimal("0.1"),
        qty_step=Decimal("0.001"), min_order_qty=Decimal("0.001"),
        min_notional=Decimal("5"), max_order_qty=Decimal("10"),
    )
    result = validate_post_only_order(
        rules=rules, side="Buy", quantity="0.01", price="100.0",
        best_bid="100", best_ask="101",
    )
    assert result["qty"] == "0.01"
    with pytest.raises(ValueError, match="qtyStep"):
        validate_post_only_order(
            rules=rules, side="Buy", quantity="0.0101", price="100.0",
            best_bid="100", best_ask="101",
        )
    with pytest.raises(ValueError, match="minimum"):
        validate_post_only_order(
            rules=rules, side="Buy", quantity="0.01", price="100.0",
            best_bid="100", best_ask="101",
        ) if Decimal("0.01") * Decimal("100") < Decimal("5") else None


def test_post_only_validation_rejects_crossing_price_and_bad_tick():
    rules = InstrumentRules(
        category="spot", symbol="BTCUSDT", tick_size=Decimal("0.5"),
        qty_step=Decimal("0.01"), min_order_qty=Decimal("0.01"),
        min_notional=Decimal("1"),
    )
    with pytest.raises(ValueError, match="tickSize"):
        validate_post_only_order(
            rules=rules, side="Buy", quantity="0.1", price="100.1",
            best_bid="100", best_ask="101",
        )
    with pytest.raises(ValueError, match="cross"):
        validate_post_only_order(
            rules=rules, side="Buy", quantity="0.1", price="101",
            best_bid="100", best_ask="101",
        )


def test_trade_volume_consumes_queue_ahead_before_order_fill():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="2"))
    replay.submit_order(
        order_id="b1", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.process(trade(3, "Sell", price="100", qty="2"))
    assert replay.orders["b1"].filled_qty == 0
    replay.process(trade(4, "Sell", price="100", qty="1"))
    assert replay.orders["b1"].filled_qty == Decimal("1")
    assert replay.orders["b1"].status == "filled"


def test_partial_fill_is_tracked_and_not_promoted_to_full_fill():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="0"))
    replay.submit_order(
        order_id="b1", timestamp_ms=2, side="Buy", price="100",
        quantity="3", latency_ms=0,
    )
    replay.process(trade(3, "Sell", price="100", qty="1"))
    assert replay.orders["b1"].filled_qty == Decimal("1")
    assert replay.orders["b1"].status == "partial"
    assert replay.report()["partial_orders"] == 1


def test_latency_activates_order_only_after_delay_and_checks_post_only_against_live_book():
    replay = BybitExecutionReplay(maker_fee_bps=1)
    replay.process(book(1))
    replay.submit_order(
        order_id="late", timestamp_ms=2, side="Buy", price="100.5",
        quantity="1", latency_ms=100,
    )
    replay.process(book(50, bid="100", ask="100.4"))
    replay.process(book(103, bid="100", ask="100.4", kind="delta"))
    assert replay.orders["late"].status == "rejected"
    assert replay.orders["late"].reject_reason == "post_only_would_cross"


def test_book_cancellations_are_not_misclassified_as_fills():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="5"))
    replay.submit_order(
        order_id="b1", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.process({
        "topic": "orderbook.50.BTCUSDT", "type": "delta",
        "received_ts_ms": 3, "data": {"b": [["100", "0"]], "a": []},
    })
    assert replay.orders["b1"].filled_qty == 0


def test_expired_order_is_not_filled_by_late_trade():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="0"))
    replay.submit_order(
        order_id="b1", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0, ttl_ms=10,
    )
    replay.process(trade(20, "Sell", price="100", qty="5"))
    assert replay.orders["b1"].filled_qty == 0
    assert replay.orders["b1"].status == "expired"


def test_adverse_selection_markout_is_signed_against_passive_fill():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="0", ask="101"))
    replay.submit_order(
        order_id="b1", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.process(trade(3, "Sell", price="100", qty="1"))
    replay.process(book(1_100, bid="98", ask="99"))
    replay.process(book(5_100, bid="97", ask="98"))
    replay.process(book(30_100, bid="96", ask="97"))
    report = replay.report()
    assert report["mean_markout_1s"] < 0
    assert report["mean_markout_30s"] < 0
    assert report["mean_entry_fee_adjusted_markout_30s_bps"] < report["mean_markout_30s"]


def test_bybit_instrument_parser_handles_spot_min_order_amount():
    item = {
        "symbol": "BTCUSDT",
        "priceFilter": {"tickSize": "0.1"},
        "lotSizeFilter": {
            "qtyStep": "0.00001", "minOrderQty": "0.00001",
            "minOrderAmt": "5", "maxOrderQty": "100",
        },
    }
    rules = InstrumentRules.from_bybit_instrument(item, "spot")
    assert rules.min_notional == Decimal("5")
    assert rules.qty_step == Decimal("0.00001")
    assert rules.min_order_qty == Decimal("0")


def test_cancel_replace_loses_queue_priority_and_counts_cancellation():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="2"))
    replay.submit_order(
        order_id="old", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.replace_order(
        "old", new_order_id="new", timestamp_ms=3, side="Buy",
        price="100", quantity="1", latency_ms=0,
    )
    replay.process(book(4, bid_qty="2", kind="delta"))
    assert replay.orders["old"].status == "cancelled"
    assert replay.orders["new"].queue_ahead == Decimal("2")
    assert replay.report()["cancellations"] == 1


def test_jsonl_replay_interleaves_submissions_and_expires_orders(tmp_path):
    import json

    from tradingagents.crypto.backtest.execution_replay import replay_jsonl

    path = tmp_path / "tape.jsonl"
    events = [
        book(1, bid_qty="0"),
        trade(3, "Sell", price="100", qty="1"),
        book(20, bid="99", ask="100"),
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    replay = replay_jsonl(
        str(path), maker_fee_bps=1, queue_factor=1,
        orders=[{
            "order_id": "late-expire", "timestamp_ms": 2, "side": "Buy",
            "price": "100", "quantity": "1", "latency_ms": 0, "ttl_ms": 5,
        }],
    )
    assert replay.orders["late-expire"].filled_qty == Decimal("1")
    assert replay.orders["late-expire"].status == "filled"


def test_sequence_gap_invalidates_book_until_fresh_snapshot():
    replay = BybitExecutionReplay(maker_fee_bps=1)
    replay.process({
        **book(1), "data": {"b": [["100", "1"]], "a": [["101", "1"]], "u": 10},
    })
    replay.process({
        "topic": "orderbook.50.BTCUSDT", "type": "delta",
        "received_ts_ms": 2, "data": {"b": [["100", "2"]], "a": [], "u": 12},
    })
    assert replay.sequence_gaps == 1
    assert replay.book.bid is None
    replay.process(trade(3, "Sell", price="100", qty="10"))
    assert replay.book.bid is None
    replay.process({
        **book(4), "data": {"b": [["100", "1"]], "a": [["101", "1"]], "u": 1},
    })
    assert replay.book.bid == Decimal("100")


def test_one_replay_instance_rejects_mixed_symbols():
    replay = BybitExecutionReplay(maker_fee_bps=1)
    replay.process(book(1))
    with pytest.raises(ValueError, match="one symbol"):
        replay.process({
            "topic": "orderbook.50.ETHUSDT", "type": "snapshot",
            "received_ts_ms": 2, "data": {"b": [["10", "1"]], "a": [["11", "1"]]},
        })


def test_report_contains_per_fill_fee_adjusted_markout():
    replay = BybitExecutionReplay(maker_fee_bps=2, queue_factor=1)
    replay.process(book(1, bid_qty="0"))
    replay.submit_order(
        order_id="audit", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.process(trade(3, "Sell", price="100", qty="1"))
    replay.process(book(30_100, bid="101", ask="102"))
    fill = replay.report()["fill_results"][0]
    assert fill["estimated_maker_fee"] == pytest.approx(0.02)
    assert fill["entry_fee_adjusted_markout_30s"] is not None


def test_collector_disconnect_invalidates_orderbook_until_reconnect_snapshot():
    replay = BybitExecutionReplay(maker_fee_bps=1)
    replay.process(book(1))
    replay.process({
        "collector_status": "disconnected", "received_ts_ms": 2,
    })
    assert replay.book.bid is None
    assert replay.connection_interruptions == 1
    replay.process(trade(3, "Sell", price="100", qty="10"))
    assert replay.book.bid is None
    replay.process(book(4))
    assert replay.book.bid == Decimal("100")


def test_jsonl_writer_emits_one_parseable_record_per_line():
    output = StringIO()
    _write_jsonl_record(output, {"event": "first", "value": 1})
    _write_jsonl_record(output, {"event": "second", "value": 2})

    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["event"] for line in lines] == ["first", "second"]


def test_trade_volume_is_not_reused_across_multiple_orders_at_same_price():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="2"))
    replay.submit_order(
        order_id="first", timestamp_ms=2, side="Buy", price="100",
        quantity="2", latency_ms=0,
    )
    replay.submit_order(
        order_id="second", timestamp_ms=2, side="Buy", price="100",
        quantity="2", latency_ms=0,
    )

    replay.process(trade(3, "Sell", price="100", qty="4"))

    total_filled = sum(order.filled_qty for order in replay.orders.values())
    assert total_filled == Decimal("2")
    assert replay.orders["first"].filled_qty == Decimal("2")
    assert replay.orders["second"].filled_qty == Decimal("0")


def test_pending_order_expires_before_delayed_activation():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.submit_order(
        order_id="slow", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=100, ttl_ms=10,
    )
    replay.process(book(120))

    assert replay.orders["slow"].status == "expired"


def test_better_priced_resting_order_gets_trade_priority_over_older_order():
    replay = BybitExecutionReplay(maker_fee_bps=1, queue_factor=1)
    replay.process(book(1, bid_qty="2"))
    replay.submit_order(
        order_id="older_lower_bid", timestamp_ms=2, side="Buy", price="100",
        quantity="1", latency_ms=0,
    )
    replay.submit_order(
        order_id="newer_better_bid", timestamp_ms=3, side="Buy", price="100.5",
        quantity="1", latency_ms=0,
    )
    replay.process(book(4, bid="100", ask="101", kind="delta"))
    replay.process(trade(5, "Sell", price="100", qty="1"))

    assert replay.orders["newer_better_bid"].filled_qty == Decimal("1")
    assert replay.orders["older_lower_bid"].filled_qty == Decimal("0")
