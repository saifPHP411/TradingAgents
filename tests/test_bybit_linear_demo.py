from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from tradingagents.crypto.bybit_linear_demo import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DEMO_BASE_URL,
    BybitLinearDemoAmbiguous,
    BybitLinearDemoBroker,
    BybitLinearDemoNotConfigured,
    BybitLinearDemoRejected,
)
from tradingagents.crypto.perpetuals import LinearPerpetualInstrument


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def instrument():
    return LinearPerpetualInstrument(
        symbol="BTCUSDT",
        base_coin="BTC",
        quote_coin="USDT",
        settle_coin="USDT",
        status="Trading",
        contract_type="LinearPerpetual",
        min_quantity=0.001,
        max_market_quantity=100,
        quantity_step=0.001,
        min_notional=5,
        tick_size=0.1,
        min_leverage=1,
        max_leverage=100,
        leverage_step=0.01,
        funding_interval_minutes=480,
    )


class Public:
    def instrument_info(self, symbol):
        assert symbol == "BTCUSDT"
        return instrument()


class ScriptedTransport:
    def __init__(
        self, *, position_rows=None, orders=None, closed_rows=None,
        history_rows=None, create_error=None, set_leverage_ret_code=0
    ):
        self.calls = []
        self.position_rows = position_rows or []
        self.orders = orders or []
        self.closed_rows = closed_rows or []
        self.history_rows = history_rows or []
        self.create_error = create_error
        self.set_leverage_ret_code = set_leverage_ret_code

    def __call__(self, *, method, url, headers, params=None, data=None, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "data": data,
                "timeout": timeout,
            }
        )
        if url.startswith("https://api.bybit.com") or "testnet" in url:
            raise AssertionError("non-Demo endpoint was contacted")
        path = url.rsplit("/", 1)[-1]
        if method == "GET" and path == "tickers":
            return Response({"retCode": 0, "result": {"list": [{"markPrice": "100"}]}})
        if method == "GET" and path == "closed-pnl":
            return Response({"retCode": 0, "result": {"list": self.closed_rows}})
        if method == "GET" and path == "history":
            return Response({"retCode": 0, "result": {"list": self.history_rows}})
        if method == "GET" and path in {"list", "realtime"}:
            if url.endswith("/v5/position/list"):
                return Response({"retCode": 0, "result": {"list": self.position_rows}})
            return Response({"retCode": 0, "result": {"list": self.orders}})
        if method == "POST" and path == "set-leverage":
            return Response({"retCode": self.set_leverage_ret_code, "result": {}})
        if method == "POST" and path == "create":
            if self.create_error:
                raise TimeoutError("simulated timeout")
            body = json.loads(data)
            return Response(
                {
                    "retCode": 0,
                    "result": {"orderId": "venue-order-1", "orderLinkId": body["orderLinkId"]},
                }
            )
        raise AssertionError(f"unexpected request: {method} {url}")


def broker(transport):
    return BybitLinearDemoBroker(
        api_key="demo-key",
        api_secret="demo-secret",
        transport=transport,
        clock_ms=lambda: 1_700_000_000_000,
        public_client=Public(),
    )


def test_credentials_are_separate_from_legacy_spot_key_names(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)
    monkeypatch.setenv("BYBIT_DEMO_API_KEY", "old-read-only-key")
    monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "old-read-only-secret")
    instance = BybitLinearDemoBroker(public_client=Public())
    assert not instance.configured
    with pytest.raises(BybitLinearDemoNotConfigured):
        instance.get_positions()


def test_partial_credentials_fail_closed():
    with pytest.raises(BybitLinearDemoNotConfigured):
        BybitLinearDemoBroker(api_key="only-key", api_secret="", public_client=Public())


def test_entry_is_signed_and_linear_only_with_exchange_side_protection():
    transport = ScriptedTransport()
    instance = broker(transport)
    result = instance.open_market_position(
        symbol="BTCUSDT",
        side="long",
        leverage=2,
        equity=50,
        stop_loss=98,
        take_profit=104,
        order_link_id="DI16-BTCUSDT-1700000000",
    )
    assert result["endpoint"] == DEMO_BASE_URL
    assert result["category"] == "linear"
    assert result["quantity"] > 0
    assert result["notional_usdt"] <= 10.0 + 1e-8

    order_call = next(c for c in transport.calls if c["url"].endswith("/v5/order/create"))
    body = json.loads(order_call["data"])
    assert body["category"] == "linear"
    assert body["side"] == "Buy"
    assert body["positionIdx"] == 0
    assert body["reduceOnly"] is False
    assert body["stopLoss"] == "98"
    assert body["takeProfit"] == "104"
    assert body["slTriggerBy"] == "MarkPrice"
    assert body["tpTriggerBy"] == "MarkPrice"
    assert body["tpslMode"] == "Full"
    assert body["slOrderType"] == "Market"
    assert body["tpOrderType"] == "Market"
    assert body["orderLinkId"] == "DI16-BTCUSDT-1700000000"
    assert all(c["url"].startswith(DEMO_BASE_URL) for c in transport.calls)

    timestamp = str(1_700_000_000_000)
    payload = timestamp + "demo-key" + "5000" + order_call["data"]
    expected = hmac.new(b"demo-secret", payload.encode(), hashlib.sha256).hexdigest()
    assert order_call["headers"]["X-BAPI-SIGN"] == expected
    assert order_call["headers"]["X-BAPI-SIGN-TYPE"] == "2"


def test_short_entry_uses_sell_and_reversed_stop_target():
    transport = ScriptedTransport()
    instance = broker(transport)
    instance.open_market_position(
        symbol="BTCUSDT",
        side="short",
        leverage=1,
        equity=50,
        stop_loss=102,
        take_profit=96,
        order_link_id="MNL-BTCUSDT-short-1",
    )
    body = json.loads(next(c for c in transport.calls if c["url"].endswith("/v5/order/create"))["data"])
    assert body["side"] == "Sell"
    assert body["stopLoss"] == "102"
    assert body["takeProfit"] == "96"
    assert body["category"] == "linear"


@pytest.mark.parametrize(
    ("side", "stop", "target"),
    [
        ("long", 101, 104),
        ("long", 98, 99),
        ("short", 99, 96),
        ("short", 102, 101),
    ],
)
def test_invalid_protection_levels_are_rejected_before_orders(side, stop, target):
    transport = ScriptedTransport()
    instance = broker(transport)
    with pytest.raises(BybitLinearDemoRejected):
        instance.open_market_position(
            symbol="BTCUSDT",
            side=side,
            leverage=1,
            equity=50,
            stop_loss=stop,
            take_profit=target,
            order_link_id="MNL-invalid-levels-1",
        )
    assert not any(c["url"].endswith("/v5/order/create") for c in transport.calls)


def test_existing_position_blocks_another_entry():
    position = {
        "symbol": "BTCUSDT",
        "size": "0.01",
        "side": "Buy",
        "positionIdx": 0,
        "avgPrice": "100",
        "markPrice": "101",
        "leverage": "2",
        "unrealisedPnl": "0.01",
        "liqPrice": "50",
        "stopLoss": "98",
        "takeProfit": "104",
    }
    transport = ScriptedTransport(position_rows=[position])
    instance = broker(transport)
    with pytest.raises(BybitLinearDemoRejected, match="open position"):
        instance.open_market_position(
            symbol="BTCUSDT",
            side="long",
            leverage=1,
            equity=50,
            stop_loss=98,
            take_profit=104,
            order_link_id="DI16-BTCUSDT-1700000000",
        )
    assert not any(c["url"].endswith("/v5/order/create") for c in transport.calls)


def test_hedge_mode_is_refused():
    position = {
        "symbol": "BTCUSDT",
        "size": "0.01",
        "side": "Buy",
        "positionIdx": 1,
    }
    instance = broker(ScriptedTransport(position_rows=[position]))
    with pytest.raises(BybitLinearDemoRejected, match="hedge-mode"):
        instance.get_positions()


def test_close_is_reduce_only_and_uses_current_exchange_quantity():
    position = {
        "symbol": "BTCUSDT",
        "size": "0.012",
        "side": "Sell",
        "positionIdx": 0,
        "avgPrice": "100",
        "markPrice": "101",
        "leverage": "2",
        "unrealisedPnl": "-0.012",
        "createdTime": "1700000000000",
        "updatedTime": "1700000000000",
    }
    transport = ScriptedTransport(position_rows=[position])
    instance = broker(transport)
    result = instance.close_market_position(
        "BTCUSDT",
        order_link_id="CLOSE-BTCUSDT-1700000000",
    )
    body = json.loads(next(c for c in transport.calls if c["url"].endswith("/v5/order/create"))["data"])
    assert body["side"] == "Buy"
    assert body["qty"] == "0.012"
    assert body["reduceOnly"] is True
    assert body["closeOnTrigger"] is True
    assert result["reduce_only"] is True


def test_ambiguous_timeout_is_reconciled_without_blind_retry():
    transport = ScriptedTransport(create_error=True)
    instance = broker(transport)
    with pytest.raises(BybitLinearDemoAmbiguous) as err:
        instance.open_market_position(
            symbol="BTCUSDT",
            side="long",
            leverage=1,
            equity=50,
            stop_loss=98,
            take_profit=104,
            order_link_id="DI16-BTCUSDT-ambiguous",
        )
    assert err.value.order_link_id == "DI16-BTCUSDT-ambiguous"
    assert len([c for c in transport.calls if c["url"].endswith("/v5/order/create")]) == 1


def test_leverage_already_set_is_treated_as_noop_and_entry_continues():
    transport = ScriptedTransport(set_leverage_ret_code=110043)
    result = broker(transport).open_market_position(
        symbol="BTCUSDT",
        side="long",
        leverage=2,
        equity=50,
        stop_loss=98,
        take_profit=104,
        order_link_id="EMA-BTCUSDT-110043",
    )
    assert result["status"] == "Created"
    assert any(c["url"].endswith("/v5/position/set-leverage") for c in transport.calls)
    assert any(c["url"].endswith("/v5/order/create") for c in transport.calls)


def test_other_leverage_exchange_rejections_still_block_entry():
    transport = ScriptedTransport(set_leverage_ret_code=110044)
    with pytest.raises(BybitLinearDemoRejected, match="retCode 110044"):
        broker(transport).open_market_position(
            symbol="BTCUSDT",
            side="long",
            leverage=2,
            equity=50,
            stop_loss=98,
            take_profit=104,
            order_link_id="EMA-BTCUSDT-110044",
        )
    assert not any(c["url"].endswith("/v5/order/create") for c in transport.calls)


def test_leverage_hard_cap_is_enforced_before_exchange_mutation():
    transport = ScriptedTransport()
    instance = broker(transport)
    with pytest.raises(BybitLinearDemoRejected, match="2x"):
        instance.open_market_position(
            symbol="BTCUSDT",
            side="long",
            leverage=3,
            equity=50,
            stop_loss=98,
            take_profit=104,
            order_link_id="MNL-BTCUSDT-3x",
        )
    assert not any(c["url"].endswith("/v5/position/set-leverage") for c in transport.calls)


def test_zero_position_rows_are_ignored():
    transport = ScriptedTransport(position_rows=[
        {"symbol": "BTCUSDT", "size": "0.000", "side": "", "positionIdx": 0}
    ])
    assert broker(transport).get_positions() == []


def test_signed_get_query_order_matches_wire_parameter_order():
    transport = ScriptedTransport()
    instance = broker(transport)
    assert instance.get_positions() == []

    call = next(c for c in transport.calls if c["url"].endswith("/v5/position/list"))
    params = call["params"]
    assert list(params) == sorted(params, key=str)

    signed_query = "&".join(f"{key}={params[key]}" for key in sorted(params, key=str))
    signature_payload = "1700000000000" + "demo-key" + "5000" + signed_query
    expected = hmac.new(
        b"demo-secret", signature_payload.encode(), hashlib.sha256
    ).hexdigest()
    assert call["headers"]["X-BAPI-SIGN"] == expected


def test_get_closed_pnl_returns_exchange_prices_pnl_and_fees():
    transport = ScriptedTransport(
        closed_rows=[
            {
                "symbol": "BTCUSDT",
                "orderId": "close-order-1",
                "side": "Sell",
                "qty": "0.001",
                "closedSize": "0.001",
                "avgEntryPrice": "64000",
                "avgExitPrice": "63900",
                "closedPnl": "-0.12",
                "openFee": "0.035",
                "closeFee": "0.035",
                "fillCount": "2",
                "leverage": "2",
                "createdTime": "1700000001000",
                "updatedTime": "1700000002000",
            }
        ]
    )
    result = broker(transport).get_closed_pnl(
        symbol="BTCUSDT",
        start_time_ms=1_700_000_000_000,
        end_time_ms=1_700_000_060_000,
    )

    assert result == [
        {
            "symbol": "BTCUSDT",
            "order_id": "close-order-1",
            "side": "Sell",
            "qty": 0.001,
            "closed_size": 0.001,
            "avg_entry_price": 64000.0,
            "avg_exit_price": 63900.0,
            "closed_pnl": -0.12,
            "open_fee": 0.035,
            "close_fee": 0.035,
            "fill_count": 2.0,
            "leverage": 2.0,
            "created_time_ms": 1_700_000_001_000,
            "updated_time_ms": 1_700_000_002_000,
        }
    ]
    call = next(c for c in transport.calls if c["url"].endswith("/v5/position/closed-pnl"))
    assert call["params"]["category"] == "linear"
    assert call["params"]["symbol"] == "BTCUSDT"


def test_get_order_history_identifies_attached_stop_or_take_profit():
    transport = ScriptedTransport(
        history_rows=[
            {
                "symbol": "SOLUSDT",
                "orderId": "stop-order-1",
                "orderLinkId": "sl-child-order",
                "parentOrderLinkId": "SC-SOLUSDT-1700000000000",
                "side": "Sell",
                "qty": "0.1",
                "orderStatus": "Filled",
                "orderType": "Market",
                "stopOrderType": "Stop",
                "avgPrice": "74.98",
                "cumExecQty": "0.1",
                "cumExecValue": "7.498",
                "triggerPrice": "74.98",
                "takeProfit": "75.31",
                "stopLoss": "74.98",
                "tpslMode": "Full",
                "tpTriggerBy": "MarkPrice",
                "slTriggerBy": "MarkPrice",
                "triggerBy": "MarkPrice",
                "reduceOnly": True,
                "closeOnTrigger": True,
                "rejectReason": "",
                "cancelType": "",
                "positionIdx": 0,
                "createdTime": "1700000001000",
                "updatedTime": "1700000002000",
            }
        ]
    )
    result = broker(transport).get_order_history(
        symbol="SOLUSDT",
        order_id="stop-order-1",
        start_time_ms=1_700_000_000_000,
        end_time_ms=1_700_000_060_000,
    )

    assert result[0]["stop_order_type"] == "Stop"
    assert result[0]["parent_order_link_id"] == "SC-SOLUSDT-1700000000000"
    assert result[0]["trigger_price"] == 74.98
    assert result[0]["reduce_only"] is True
    assert result[0]["close_on_trigger"] is True
    call = next(c for c in transport.calls if c["url"].endswith("/v5/order/history"))
    assert call["params"]["orderId"] == "stop-order-1"


def test_history_endpoints_reject_windows_larger_than_seven_days():
    instance = broker(ScriptedTransport())
    with pytest.raises(ValueError, match="seven days"):
        instance.get_closed_pnl(
            start_time_ms=1_700_000_000_000,
            end_time_ms=1_700_700_000_000,
        )
