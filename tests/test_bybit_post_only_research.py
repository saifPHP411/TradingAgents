from __future__ import annotations

from tradingagents.crypto.backtest.bybit_execution import build_post_only_limit_payload

import pytest


pytestmark = pytest.mark.unit


def test_buy_post_only_quote_is_passive_and_tick_aligned():
    order = build_post_only_limit_payload(
        symbol="BTCUSDT", side="Buy", quantity="0.01",
        bid="60000.03", ask="60000.05", tick_size="0.1",
    )
    assert order["price"] == "60000.0"
    assert order["timeInForce"] == "PostOnly"
    assert order["orderType"] == "Limit"


def test_sell_post_only_quote_is_passive_and_tick_aligned():
    order = build_post_only_limit_payload(
        symbol="BTCUSDT", side="Sell", quantity="0.01",
        bid="60000.03", ask="60000.05", tick_size="0.1",
    )
    assert order["price"] == "60000.1"
    assert order["timeInForce"] == "PostOnly"


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_post_only_payload_rejects_invalid_book(side):
    with pytest.raises(ValueError, match="bid below ask"):
        build_post_only_limit_payload(
            symbol="BTCUSDT", side=side, quantity="0.01",
            bid="60000", ask="60000", tick_size="0.1",
        )


@pytest.mark.parametrize("quantity", ["0", "-1", "NaN", "Infinity"])
def test_post_only_payload_rejects_invalid_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        build_post_only_limit_payload(
            symbol="BTCUSDT", side="Buy", quantity=quantity,
            bid="60000", ask="60001", tick_size="0.1",
        )


def test_post_only_payload_is_only_a_payload_and_contains_no_auth_material():
    order = build_post_only_limit_payload(
        symbol="ETHUSDT", side="Buy", quantity="0.1",
        bid="3000", ask="3001", tick_size="0.01",
        category="spot", order_link_id="research-test-001",
    )
    assert order["orderLinkId"] == "research-test-001"
    assert not ({"api_key", "secret", "sign", "timestamp"} & order.keys())
