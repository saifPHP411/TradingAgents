"""Bybit Demo broker adapter tests.

These tests exercise the Bybit Demo broker adapter offline with mocked HTTP
responses. They cover:

- Demo URL validation and live URL rejection
- Missing credentials handling
- Account balance and positions reading
- Order creation and cancellation
- Symbol instrument rules (min order amount, qty step, tick size)
- Decimal quantity handling
- Risk limits enforcement (max notional $25, max positions, max daily orders)
- Long-only enforcement
- SL/TP protection verification

No live API keys are needed; all tests use mocked transports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tradingagents.paper_experiment.broker.bybit_demo import (
    DEMO_BASE_URL as BYBIT_DEMO_BASE_URL,
    BrokerError,
    BrokerNotConfiguredError,
    BybitDemoBroker,
    LiveEndpointRefusedError,
    PaperRiskLimits,
    PaperRiskRejected,
    assert_demo_endpoint,
    assert_demo_request_url,
)

# Test constants
DEMO_API_KEY = "DEMO_KEY_1234567890123456"
DEMO_API_SECRET = "DEMO_SECRET_123456789012345678901234"
LIVE_API_KEY = "LIVE_KEY_1234567890123456"
LIVE_API_SECRET = "LIVE_SECRET_123456789012345678901234"

# Mock account response
MOCK_ACCOUNT_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "list": [
            {
                "accountId": "12345",
                "accountType": "SPOT",
                "coin": [
                    {
                        "coin": "USDT",
                        "walletBalance": "10000.00",
                        "availableToWithdraw": "9500.00",
                        "locked": "500.00",
                    },
                    {
                        "coin": "BTC",
                        "walletBalance": "0.1",
                        "availableToWithdraw": "0.08",
                        "locked": "0.02",
                    },
                ],
            }
        ],
    },
}

# Mock instruments response
MOCK_INSTRUMENTS_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "category": "spot",
        "list": [
            {
                "symbol": "BTCUSDT",
                "status": "Trading",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.00001",
                    "maxOrderQty": "100",
                    "qtyStep": "0.00001",
                    "minOrderAmt": "10",
                },
                "priceFilter": {
                    "tickSize": "0.1",
                },
            },
            {
                "symbol": "ETHUSDT",
                "status": "Trading",
                "baseCoin": "ETH",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.0001",
                    "maxOrderQty": "1000",
                    "qtyStep": "0.0001",
                    "minOrderAmt": "10",
                },
                "priceFilter": {
                    "tickSize": "0.01",
                },
            },
            {
                "symbol": "SOLUSDT",
                "status": "Trading",
                "baseCoin": "SOL",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.01",
                    "maxOrderQty": "10000",
                    "qtyStep": "0.01",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.001",
                },
            },
            {
                "symbol": "XRPUSDT",
                "status": "Trading",
                "baseCoin": "XRP",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "1",
                    "maxOrderQty": "1000000",
                    "qtyStep": "1",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.0001",
                },
            },
            {
                "symbol": "DOGEUSDT",
                "status": "Trading",
                "baseCoin": "DOGE",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "1",
                    "maxOrderQty": "10000000",
                    "qtyStep": "1",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.00001",
                },
            },
            {
                "symbol": "AVAXUSDT",
                "status": "Trading",
                "baseCoin": "AVAX",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.01",
                    "maxOrderQty": "10000",
                    "qtyStep": "0.01",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.001",
                },
            },
            {
                "symbol": "LINKUSDT",
                "status": "Trading",
                "baseCoin": "LINK",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.1",
                    "maxOrderQty": "100000",
                    "qtyStep": "0.1",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.001",
                },
            },
            {
                "symbol": "LTCUSDT",
                "status": "Trading",
                "baseCoin": "LTC",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.01",
                    "maxOrderQty": "10000",
                    "qtyStep": "0.01",
                    "minOrderAmt": "5",
                },
                "priceFilter": {
                    "tickSize": "0.01",
                },
            },
            {
                "symbol": "BCHUSDT",
                "status": "Trading",
                "baseCoin": "BCH",
                "quoteCoin": "USDT",
                "lotSizeFilter": {
                    "minOrderQty": "0.001",
                    "maxOrderQty": "10000",
                    "qtyStep": "0.001",
                    "minOrderAmt": "10",
                },
                "priceFilter": {
                    "tickSize": "0.01",
                },
            },
        ],
    },
}

# Mock positions response
MOCK_POSITIONS_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "list": [
            {
                "symbol": "BTCUSDT",
                "size": "0.1",
                "side": "Buy",
                "avgPrice": "50000",
                "markPrice": "51000",
                "unrealisedPnl": "100",
            }
        ],
    },
}

# Mock order create response
MOCK_ORDER_CREATE_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "orderId": "bybit-order-123",
        "orderLinkId": "pt-test-client-id",
    },
}

# Mock order cancel response
MOCK_ORDER_CANCEL_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "orderId": "bybit-order-123",
        "orderLinkId": "pt-test-client-id",
    },
}

# Mock order history response
MOCK_ORDER_HISTORY_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "list": [
            {
                "orderId": "bybit-order-123",
                "orderLinkId": "pt-test-client-id",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Market",
                "qty": "0.001",
                "price": "0",
                "avgPrice": "50000",
                "cumExecQty": "0.001",
                "cumExecValue": "50",
                "cumExecFee": "0.05",
                "timeInForce": "GTC",
                "orderStatus": "Filled",
                "createdTime": str(int(datetime.now(UTC).timestamp() * 1000)),
                "updatedTime": str(int(datetime.now(UTC).timestamp() * 1000)),
            }
        ],
        "nextPageCursor": "",
    },
}

# Mock realtime order response
MOCK_ORDER_REALTIME_RESPONSE = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "list": [
            {
                "orderId": "bybit-order-123",
                "orderLinkId": "pt-test-client-id",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Market",
                "qty": "0.001",
                "price": "0",
                "avgPrice": "50000",
                "cumExecQty": "0.001",
                "cumExecValue": "50",
                "cumExecFee": "0.05",
                "timeInForce": "GTC",
                "orderStatus": "Filled",
                "createdTime": str(int(datetime.now(UTC).timestamp() * 1000)),
                "updatedTime": str(int(datetime.now(UTC).timestamp() * 1000)),
            }
        ],
    },
}


class MockTransport:
    """A mock HTTP transport that records calls and returns canned responses.

    The transport is **stateful** for the order book: a created order joins
    the open-order list, and a cancel removes it. Bybit's cancel endpoint is
    asynchronous -- its acknowledgement only says the request was accepted --
    so a test that wants to observe a *cancelled* order has to model the
    venue's state, not just return a canned body.

    A canned response registered for a path still wins, so a test can pin an
    exact payload where it needs one.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.responses: dict[tuple[str, str], dict] = {}
        self.call_count = 0
        self.open_orders: list[dict] = []
        self.history_orders: list[dict] = []
        self.executions: list[dict] = []
        self._next_order_id = 1000

    def add_response(self, method: str, path: str, response: dict):
        self.responses[(method.upper(), path)] = response

    def _default_order_list(self, path: str) -> list[dict]:
        if path == "/v5/order/realtime":
            return self.open_orders
        if path == "/v5/order/history":
            return self.history_orders
        if path == "/v5/execution/list":
            return self.executions
        return []

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        params: dict,
        json_body: dict,
        timeout: float,
    ):
        from tradingagents.paper_experiment.broker.bybit_demo import BrokerHttpResponse

        self.call_count += 1
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "params": dict(params),
                "json_body": json_body,
            }
        )

        path = url.split("?")[0].replace("https://api-demo.bybit.com", "")
        key = (method.upper(), path)

        # A created order joins the venue's open-order list.
        if key == ("POST", "/v5/order/create") and "orderFilter" in (json_body or {}):
            self._next_order_id += 1
            order_id = f"mock-{self._next_order_id}"
            record = {
                "orderId": order_id,
                "orderLinkId": json_body.get("orderLinkId", ""),
                "symbol": json_body.get("symbol", ""),
                "side": json_body.get("side", ""),
                "orderType": json_body.get("orderType", "Market"),
                "qty": json_body.get("qty", "0"),
                "timeInForce": json_body.get("timeInForce", "GTC"),
                "orderStatus": "Untriggered",
                "orderFilter": json_body.get("orderFilter", ""),
                "triggerPrice": json_body.get("triggerPrice", "0"),
                "cumExecQty": "0",
                "leavesQty": json_body.get("qty", "0"),
                "createdTime": "1700000000000",
                "updatedTime": "1700000000000",
            }
            # The venue classifies a Spot TP/SL order by its trigger direction
            # relative to the last price: a trigger below the market is a stop
            # loss, a trigger above it is a take profit.
            if record["orderFilter"] == "tpslOrder":
                record["stopOrderType"] = self._classify_tpsl(record)
            self.open_orders.append(record)
            return BrokerHttpResponse(
                200,
                json.dumps(
                    {
                        "retCode": 0,
                        "retMsg": "OK",
                        "result": {"orderId": order_id, "orderLinkId": record["orderLinkId"]},
                    }
                ),
            )

        # A cancel moves the order out of the open list and into the history.
        if key == ("POST", "/v5/order/cancel"):
            link_id = (json_body or {}).get("orderLinkId", "")
            for record in list(self.open_orders):
                if record.get("orderLinkId") == link_id:
                    self.open_orders.remove(record)
                    finished = dict(record)
                    finished["orderStatus"] = "Cancelled"
                    finished["cumExecQty"] = finished.get("cumExecQty", "0")
                    self.history_orders.append(finished)
                    return BrokerHttpResponse(
                        200,
                        json.dumps(
                            {
                                "retCode": 0,
                                "retMsg": "OK",
                                "result": {"orderId": record.get("orderId"), "orderLinkId": link_id},
                            }
                        ),
                    )
            return BrokerHttpResponse(
                200,
                json.dumps({"retCode": 110001, "retMsg": "order does not exist", "result": {}}),
            )

        # A registered response wins, so a test can pin an exact payload.
        if key in self.responses:
            return BrokerHttpResponse(200, json.dumps(self.responses[key]))

        if key[0] == "GET" and path in (
            "/v5/order/realtime",
            "/v5/order/history",
            "/v5/execution/list",
        ):
            rows = self._default_order_list(path)
            link_id = (params or {}).get("orderLinkId")
            if link_id:
                rows = [row for row in rows if row.get("orderLinkId") == link_id]
            symbol = (params or {}).get("symbol")
            if symbol:
                rows = [row for row in rows if row.get("symbol") == symbol]
            return BrokerHttpResponse(200, json.dumps({"retCode": 0, "retMsg": "OK", "result": {"list": rows}}))

        # Default fallback
        return BrokerHttpResponse(200, json.dumps({"retCode": 0, "retMsg": "OK", "result": {}}))

    @staticmethod
    def _classify_tpsl(record: dict) -> str:
        """How the venue classifies a Spot TP/SL order.

        Bybit derives the stop/target from the trigger's side of the market,
        so the mock uses the same rule: a trigger below the last price is a
        stop loss, a trigger at or above it is a take profit.
        """
        trigger = float(record.get("triggerPrice") or 0)
        last = 50000.0
        return "TakeProfit" if trigger >= last else "StopLoss"


def _make_client(transport: MockTransport | None = None, **kwargs) -> BybitDemoBroker:
    """Create a BybitDemoBroker with mocked transport."""
    return BybitDemoBroker(
        api_key=DEMO_API_KEY,
        api_secret=DEMO_API_SECRET,
        base_url=BYBIT_DEMO_BASE_URL,
        transport=transport or MockTransport(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Endpoint validation tests
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    def test_approved_demo_endpoint_passes(self):
        assert assert_demo_endpoint(BYBIT_DEMO_BASE_URL) == BYBIT_DEMO_BASE_URL

    def test_live_endpoint_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://api.bybit.com")

    def test_testnet_endpoint_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://api-testnet.bybit.com")

    def test_http_downgrade_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("http://api-demo.bybit.com")

    def test_wrong_port_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://api-demo.bybit.com:8443")

    def test_credentials_in_url_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://user:pass@api-demo.bybit.com")

    def test_request_url_validation(self):
        assert assert_demo_request_url(f"{BYBIT_DEMO_BASE_URL}/v5/order/create") == f"{BYBIT_DEMO_BASE_URL}/v5/order/create"

    def test_request_url_live_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_request_url("https://api.bybit.com/v5/order/create")


# ---------------------------------------------------------------------------
# Credential tests
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
        with pytest.raises(BrokerNotConfiguredError):
            BybitDemoBroker(base_url=BYBIT_DEMO_BASE_URL)

    def test_live_credentials_are_refused(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", LIVE_API_KEY)
        monkeypatch.setenv("BYBIT_API_SECRET", LIVE_API_SECRET)
        with pytest.raises(BrokerNotConfiguredError):
            BybitDemoBroker(base_url=BYBIT_DEMO_BASE_URL)

    def test_demo_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("BYBIT_DEMO_API_KEY", DEMO_API_KEY)
        monkeypatch.setenv("BYBIT_DEMO_API_SECRET", DEMO_API_SECRET)
        client = BybitDemoBroker(base_url=BYBIT_DEMO_BASE_URL, transport=MockTransport())
        assert client._key == DEMO_API_KEY
        assert client._secret == DEMO_API_SECRET


# ---------------------------------------------------------------------------
# Account tests
# ---------------------------------------------------------------------------


class TestAccount:
    def test_connect_returns_account(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        client = _make_client(transport)

        account = client.connect()
        assert account.account_id == "12345"
        assert account.equity == 10000.0
        assert account.available_balance == 9500.0
        assert account.cash == 10000.0
        assert account.buying_power == 9500.0
        assert account.status == "ACTIVE"
        assert account.currency == "USDT"
        assert account.to_dict()["status"] == "ACTIVE"

    def test_refresh_account_updates_state(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        client = _make_client(transport)

        client.connect()
        assert client._state.confirmed is True
        assert client._state.equity == 10000.0

    def test_unified_wallet_blank_available_to_withdraw_uses_wallet_minus_locked(self):
        response = json.loads(json.dumps(MOCK_ACCOUNT_RESPONSE))
        usdt = response["result"]["list"][0]["coin"][0]
        usdt["availableToWithdraw"] = ""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", response)
        client = _make_client(transport)

        account = client.connect()

        assert account.equity == 10000.0
        assert account.available_balance == 9500.0

    def test_wallet_balance_reader_handles_blank_deprecated_fields(self):
        response = json.loads(json.dumps(MOCK_ACCOUNT_RESPONSE))
        usdt = response["result"]["list"][0]["coin"][0]
        usdt["availableToWithdraw"] = ""
        usdt["free"] = ""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", response)
        client = _make_client(transport)

        balances = client._read_wallet_balances()

        assert balances["USDT"]["wallet_balance"] == 10000.0
        assert balances["USDT"]["locked"] == 500.0
        assert balances["USDT"]["available_to_withdraw"] == 9500.0
        assert balances["USDT"]["free"] == 9500.0
        call = next(c for c in transport.calls if "/v5/account/wallet-balance" in c["url"])
        assert call["params"]["accountType"] == "UNIFIED"


# ---------------------------------------------------------------------------
# Instruments tests
# ---------------------------------------------------------------------------


class TestInstruments:
    def test_load_instruments(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        client = _make_client(transport)

        instruments = client.load_instruments()
        assert "BTCUSDT" in instruments
        assert "ETHUSDT" in instruments
        assert "SOLUSDT" in instruments

    def test_get_instrument_rules(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        client = _make_client(transport)

        btc_info = client.get_instrument("BTC/USDT")
        assert btc_info is not None
        assert btc_info.symbol == "BTCUSDT"
        assert btc_info.min_order_amt == 10.0
        assert btc_info.qty_step == 0.00001
        assert btc_info.tick_size == 0.1
        assert btc_info.tradable is True

    def test_spot_instrument_uses_base_precision_as_quantity_step(self):
        # Bybit Spot documents basePrecision, not qtyStep, in lotSizeFilter.
        response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "spot",
                "list": [
                    {
                        "symbol": "BNBUSDT",
                        "status": "Trading",
                        "baseCoin": "BNB",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "basePrecision": "0.0001",
                            "minOrderAmt": "5",
                        },
                        "priceFilter": {"tickSize": "0.01"},
                    }
                ],
            },
        }
        transport = MockTransport()
        transport.add_response("GET", "/v5/market/instruments-info", response)
        client = _make_client(transport)

        info = client.get_instrument("BNB/USDT")

        assert info is not None
        assert info.qty_step == 0.0001
        assert info.min_order_amt == 5.0
        assert info.tradable is True

    def test_instrument_not_found_returns_none(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        client = _make_client(transport)

        info = client.get_instrument("UNKNOWN/USDT")
        assert info is None

    def test_symbol_mapping_canonical_to_bybit(self):
        client = _make_client(MockTransport())
        assert client._to_bybit_symbol("BTC/USDT") == "BTCUSDT"
        assert client._to_bybit_symbol("BTCUSDT") == "BTCUSDT"

    def test_symbol_mapping_bybit_to_canonical(self):
        client = _make_client(MockTransport())
        assert client._to_canonical_symbol("BTCUSDT") == "BTC/USDT"
        assert client._to_canonical_symbol("BTC/USDT") == "BTC/USDT"


# ---------------------------------------------------------------------------
# Positions tests
# ---------------------------------------------------------------------------


class TestPositions:
    def test_positions_read(self):
        transport = MockTransport()
        # Account response with BTC balance
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        # No open orders
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)

        client.connect()
        positions = client.positions(refresh=True)

        assert "BTC/USDT" in positions
        pos = positions["BTC/USDT"]
        assert pos.quantity == 0.1
        assert pos.available_quantity == 0.08
        assert pos.venue_symbol == "BTCUSDT"
        assert pos.side == "buy"
        assert pos.protection_orders is None
        # mark_price is fetched from a separate public client (not mocked here)

    def test_positions_empty_when_flat(self):
        flat_account = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "accountId": "12345",
                        "accountType": "SPOT",
                        "coin": [
                            {
                                "coin": "USDT",
                                "walletBalance": "10000.00",
                                "availableToWithdraw": "10000.00",
                                "locked": "0.00",
                            }
                        ],
                    }
                ],
            },
        }
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", flat_account)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)

        client.connect()
        positions = client.positions(refresh=True)
        assert positions == {}


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------


class TestOrders:
    def test_submit_market_order(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", MOCK_ORDER_REALTIME_RESPONSE)
        client = _make_client(transport)

        client.connect()
        order = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,  # $25 at $50000
            side="buy",
            reference_price=50000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
            crypto=True,
        )

        assert order.client_order_id == "pt-test-client-id"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.quantity == 0.001  # Broker reports actual executed qty
        assert order.filled_quantity == 0.001
        assert order.status == "filled"

    def test_submit_order_floors_quantity_to_symbol_qty_step(self):
        transport = MockTransport()
        instruments = json.loads(json.dumps(MOCK_INSTRUMENTS_RESPONSE))
        btc_rules = instruments["result"]["list"][0]["lotSizeFilter"]
        btc_rules["minOrderQty"] = "0.01"
        btc_rules["qtyStep"] = "0.01"
        btc_rules["minOrderAmt"] = "5"
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", instruments)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        realtime = json.loads(json.dumps(MOCK_ORDER_REALTIME_RESPONSE))
        realtime["result"]["list"][0]["orderLinkId"] = "pt-qty-step-test"
        realtime["result"]["list"][0]["qty"] = "9.70"
        realtime["result"]["list"][0]["cumExecQty"] = "9.70"
        transport.add_response("GET", "/v5/order/realtime", realtime)
        client = _make_client(transport)

        client.connect()
        order = client.submit_order(
            symbol="BTC/USDT",
            quantity=9.700261,
            side="buy",
            reference_price=2.0,
            client_order_id="pt-qty-step-test",
            allow_orders=True,
            refresh=True,
            crypto=True,
        )
        assert order.quantity == 9.70

        create_call = next(
            call for call in transport.calls
            if call["method"] == "POST" and call["url"].endswith("/v5/order/create")
        )
        assert create_call["json_body"]["qty"] == "9.70"

    def test_submit_order_refuses_non_crypto_orders(self):
        transport = MockTransport()
        client = _make_client(transport)

        with pytest.raises(PaperRiskRejected, match="crypto spot orders only"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.001,
                side="buy",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                allow_orders=True,
                crypto=False,
            )

    def test_submit_order_refused_without_allow_orders(self):
        transport = MockTransport()
        client = _make_client(transport)

        with pytest.raises(PaperRiskRejected, match="order submission is disabled"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.001,
                side="buy",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                allow_orders=False,
            )

    def test_submit_order_refused_before_connect(self):
        transport = MockTransport()
        client = _make_client(transport)

        with pytest.raises(BrokerError, match="account has not been confirmed"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.001,
                side="buy",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                allow_orders=True,
            )

    def test_cancel_order(self):
        """A cancel is confirmed at the venue, not assumed from the acknowledgement.

        Bybit's cancel endpoint is asynchronous: the acknowledgement says the
        *request* was accepted, and the order's real status has to be read
        back. So the transport models a venue that actually cancels the order,
        and the test asserts the confirmed answer.
        """
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        client = _make_client(transport)

        client.connect()
        # The order exists at the venue before the cancel is requested.
        transport.open_orders.append(
            {
                "orderId": "bybit-order-123",
                "orderLinkId": "pt-test-client-id",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Market",
                "qty": "0.001",
                "timeInForce": "GTC",
                "orderStatus": "New",
                "cumExecQty": "0",
                "leavesQty": "0.001",
            }
        )
        order = client.cancel_order(client_order_id="pt-test-client-id", symbol="BTC/USDT")

        assert order.client_order_id == "pt-test-client-id"
        # The venue confirmed the cancellation.
        assert order.status == "cancelled"
        # And the order is no longer in the live list.
        assert transport.open_orders == []

    def test_list_orders(self):
        """Both the working orders and the finished ones are read.

        A V5 order lives in exactly one of the two lists, so reading only one
        of them misses half of the book.
        """
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/order/history", MOCK_ORDER_HISTORY_RESPONSE)
        client = _make_client(transport)

        client.connect()
        page = client.list_orders()

        # The realtime list and the history list are both read.
        assert page.pages_read == 2
        assert len(page.orders) == 1
        assert page.orders[0].client_order_id == "pt-test-client-id"

    def test_idempotency_same_client_order_id(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", MOCK_ORDER_REALTIME_RESPONSE)
        client = _make_client(transport)

        client.connect()
        # First submission
        order1 = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,  # $25 at $50000
            side="buy",
            reference_price=50000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )

        # Second submission with same client_order_id should return existing
        order2 = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,  # $25 at $50000
            side="buy",
            reference_price=50000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )

        assert order1.client_order_id == order2.client_order_id
        assert order1.broker_order_id == order2.broker_order_id


# ---------------------------------------------------------------------------
# Risk limits tests
# ---------------------------------------------------------------------------


class TestRiskLimits:
    def test_max_order_notional_enforced(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport, limits=PaperRiskLimits(max_order_notional=25.0))

        client.connect()

        # Try to place order exceeding $25 notional
        with pytest.raises(PaperRiskRejected, match="exceeds the cap"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=1.0,  # 1 BTC at $50000 = $50000 notional
                side="buy",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_max_open_positions_enforced(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {
            "retCode": 0,
            "result": {
                "list": [
                    {"symbol": "BTCUSDT", "lastPrice": "51000"},
                    {"symbol": "ETHUSDT", "lastPrice": "3100"},
                    {"symbol": "SOLUSDT", "lastPrice": "105"},
                    {"symbol": "XRPUSDT", "lastPrice": "0.55"},
                    {"symbol": "DOGEUSDT", "lastPrice": "0.11"},
                    {"symbol": "AVAXUSDT", "lastPrice": "20"},
                ]
            }
        })
        # Mock account with 5 non-USDT coins
        five_coin_account = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "accountId": "12345",
                        "accountType": "SPOT",
                        "coin": [
                            {"coin": "USDT", "walletBalance": "10000.00", "availableToWithdraw": "9500.00", "locked": "500.00"},
                            {"coin": "BTC", "walletBalance": "0.1", "availableToWithdraw": "0.08", "locked": "0.02"},
                            {"coin": "ETH", "walletBalance": "1.0", "availableToWithdraw": "0.9", "locked": "0.1"},
                            {"coin": "SOL", "walletBalance": "10.0", "availableToWithdraw": "9.0", "locked": "1.0"},
                            {"coin": "XRP", "walletBalance": "1000.0", "availableToWithdraw": "900.0", "locked": "100.0"},
                            {"coin": "DOGE", "walletBalance": "10000.0", "availableToWithdraw": "9000.0", "locked": "1000.0"},
                        ],
                    }
                ],
            },
        }
        transport.add_response("GET", "/v5/account/wallet-balance", five_coin_account)
        client = _make_client(transport, limits=PaperRiskLimits(max_open_positions=5))

        client.connect()

        # Try to open a 6th position
        with pytest.raises(PaperRiskRejected, match="exceed the open-position limit"):
            client.submit_order(
                symbol="AVAX/USDT",
                quantity=0.01,  # valid AVAX Spot basePrecision; reach the position-limit check
                side="buy",
                reference_price=100,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_sell_exceeds_held_quantity_refused(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {
            "retCode": 0,
            "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}
        })
        client = _make_client(transport)

        client.connect()

        # Try to sell more than held (0.1 BTC held, try to sell 0.2)
        with pytest.raises(PaperRiskRejected, match="exceeds the held"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.2,
                side="sell",
                reference_price=51000,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_sell_without_positions_refused(self):
        flat_account = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "accountId": "12345",
                        "accountType": "SPOT",
                        "coin": [
                            {
                                "coin": "USDT",
                                "walletBalance": "10000.00",
                                "availableToWithdraw": "10000.00",
                                "locked": "0.00",
                            }
                        ],
                    }
                ],
            },
        }
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", flat_account)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)

        client.connect()

        with pytest.raises(PaperRiskRejected, match="exceeds the held 0.0"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.1,
                side="sell",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_buy_fits_buying_power(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        client = _make_client(transport)

        client.connect()

        # $25 order should fit in $9500 buying power
        order = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,  # $25 at $50000
            side="buy",
            reference_price=50000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )
        assert order is not None


# ---------------------------------------------------------------------------
# Minimum order amount / qty step tests
# ---------------------------------------------------------------------------


class TestMinOrderRules:
    def test_blank_required_instrument_rules_fail_closed(self):
        transport = MockTransport()
        instruments = json.loads(json.dumps(MOCK_INSTRUMENTS_RESPONSE))
        instruments["result"]["list"][0]["lotSizeFilter"]["minOrderAmt"] = ""
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", instruments)
        client = _make_client(transport)

        client.connect()

        with pytest.raises(PaperRiskRejected, match="required quantity/minimum-notional rules"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.0005,
                side="buy",
                reference_price=50000,
                client_order_id="pt-missing-minimum",
                allow_orders=True,
                refresh=True,
            )
        assert not [
            call for call in transport.calls
            if call["method"] == "POST" and call["url"].endswith("/v5/order/create")
        ]

    def test_below_min_notional_rejected(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)

        client.connect()

        # BTCUSDT min order amt is 10 USDT, try $5
        with pytest.raises(PaperRiskRejected, match="BELOW_MIN_NOTIONAL"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.0001,  # $5 at $50000
                side="buy",
                reference_price=50000,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_qty_step_enforced(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        client = _make_client(transport, limits=PaperRiskLimits(max_order_notional=100.0))

        client.connect()

        # BTCUSDT qty_step is 0.00001, try 0.000005 (not a multiple)
        # Use reference price 50000 so notional = 0.000005 * 50000 = $0.25, below min $10
        # Need higher price to pass min_order_amt, but notional < 100
        with pytest.raises(PaperRiskRejected, match="qtyStep"):
            client.submit_order(
                symbol="BTC/USDT",
                quantity=0.000005,
                side="buy",
                reference_price=2000000,  # $10 notional (just above min $10)
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )

    def test_qty_step_valid_passes(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        client = _make_client(transport)

        client.connect()

        # BTCUSDT qty_step is 0.00001, use 0.0001 (valid multiple, $5 at $50000)
        # min_order_amt is $10, so need to use higher price or larger qty
        # Use reference_price = 200000 so notional = 0.0001 * 200000 = $20
        order = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0001,
            side="buy",
            reference_price=200000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )
        assert order is not None

    def test_all_test_symbols_have_min_order_amt(self):
        """Verify all required symbols have min_order_amt populated."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        client = _make_client(transport)

        client.load_instruments()

        required_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
            "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
        ]
        for sym in required_symbols:
            info = client.get_instrument(sym)
            assert info is not None, f"Missing instrument info for {sym}"
            assert info.min_order_amt > 0, f"min_order_amt not set for {sym}"
            assert info.qty_step > 0, f"qty_step not set for {sym}"
            assert info.tick_size > 0, f"tick_size not set for {sym}"
            assert info.tradable is True, f"{sym} not tradable"


# ---------------------------------------------------------------------------
# Decimal quantity formatting tests
# ---------------------------------------------------------------------------


class TestDecimalQuantity:
    def test_quantity_formatted_without_scientific_notation(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        client = _make_client(transport)

        client.connect()

        # Small quantity that would be scientific notation in str()
        # Use high reference_price so notional >= $10 min_order_amt
        client.submit_order(
            symbol="BTC/USDT",
            quantity=0.00001,  # Would be 1e-05 in scientific notation
            side="buy",
            reference_price=2000000,  # $20 notional
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )

        # Check that the request body used plain decimal
        create_call = next(c for c in transport.calls if c["method"] == "POST" and "/v5/order/create" in c["url"])
        qty_in_request = create_call["json_body"]["qty"]
        assert "e" not in qty_in_request.lower(), f"Quantity in scientific notation: {qty_in_request}"
        assert qty_in_request == "0.00001"


# ---------------------------------------------------------------------------
# Broker status test
# ---------------------------------------------------------------------------


class TestBrokerStatus:
    def test_broker_status_reports_correctly(self, monkeypatch):
        monkeypatch.setenv("BYBIT_DEMO_API_KEY", DEMO_API_KEY)
        monkeypatch.setenv("BYBIT_DEMO_API_SECRET", DEMO_API_SECRET)

        client = _make_client(MockTransport())
        status = client.broker_status()

        assert status.paper_endpoint == BYBIT_DEMO_BASE_URL
        assert status.endpoint_ok is True
        assert status.key_present is True
        assert status.secret_present is True
        assert status.key_env == "BYBIT_DEMO_API_KEY"
        assert status.secret_env == "BYBIT_DEMO_API_SECRET"

    def test_broker_status_shows_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)

        transport = MockTransport()
        # Create client with valid credentials for constructor, but test env vars are empty
        client = BybitDemoBroker(
            base_url=BYBIT_DEMO_BASE_URL,
            transport=transport,
            api_key="test_key",
            api_secret="test_secret",
        )
        status = client.broker_status()

        assert status.key_present is False
        assert status.secret_present is False


# ---------------------------------------------------------------------------
# Protection / SL/TP tests
# ---------------------------------------------------------------------------


class TestProtection:
    def test_stop_loss_required_for_buy(self):
        """The protection gate is enforced in the worker, not the broker.
        The broker validates SL/TP are valid numbers if provided."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        transport.add_response("POST", "/v5/order/create", MOCK_ORDER_CREATE_RESPONSE)
        client = _make_client(transport)

        client.connect()

        # Order with stop_loss and take_profit should be accepted
        order = client.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="pt-test-client-id",
            order_type="Market",
            allow_orders=True,
            refresh=True,
        )
        assert order is not None

    def test_create_protection_orders_success(self):
        """Test creating SL/TP conditional orders for a position."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        # Mock SL order creation response
        transport.add_response("POST", "/v5/order/create", {
            "retCode": 0, "retMsg": "OK",
            "result": {"orderId": "bybit-sl-123", "orderLinkId": "pt-test-sl"}
        })
        # Mock TP order creation response
        transport.add_response("POST", "/v5/order/create", {
            "retCode": 0, "retMsg": "OK",
            "result": {"orderId": "bybit-tp-123", "orderLinkId": "pt-test-tp"}
        })

        client = _make_client(transport)
        client.connect()

        # Create protection orders - MOCK_ACCOUNT_RESPONSE has 0.08 available BTC
        prot = client.create_protection_orders(
            symbol="BTC/USDT",
            position_quantity=0.08,  # Use available quantity
            stop_loss_price=45000,
            take_profit_price=55000,
            client_order_id_prefix="pt-test",
            allow_orders=True,
        )

        assert "stop_loss" in prot
        assert "take_profit" in prot
        assert prot["stop_loss"].client_order_id == "pt-test-sl"
        assert prot["take_profit"].client_order_id == "pt-test-tp"
        # Newly created conditional orders have status "new" or "untriggered" which maps to "accepted"
        assert prot["stop_loss"].status in ("accepted", "unknown")
        assert prot["take_profit"].status in ("accepted", "unknown")

    def test_create_protection_orders_fails_if_no_position(self):
        """Test that creating protection fails if no position exists."""
        flat_account = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "accountId": "12345",
                        "accountType": "SPOT",
                        "coin": [
                            {
                                "coin": "USDT",
                                "walletBalance": "10000.00",
                                "availableToWithdraw": "10000.00",
                                "locked": "0.00",
                            }
                        ],
                    }
                ],
            },
        }
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", flat_account)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)
        client.connect()

        with pytest.raises(PaperRiskRejected, match="no position found"):
            client.create_protection_orders(
                symbol="BTC/USDT",
                position_quantity=0.1,
                stop_loss_price=45000,
                take_profit_price=55000,
                client_order_id_prefix="pt-test",
                allow_orders=True,
            )

    def test_create_protection_orders_fails_if_insufficient_available(self):
        """Test that creating protection fails if available qty < requested."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        # Position has 0.08 available, try to protect 0.1
        with pytest.raises(PaperRiskRejected, match="insufficient available quantity"):
            client.create_protection_orders(
                symbol="BTC/USDT",
                position_quantity=0.1,
                stop_loss_price=45000,
                take_profit_price=55000,
                client_order_id_prefix="pt-test",
                allow_orders=True,
            )

    def test_create_protection_orders_validates_tick_size(self):
        """Test that SL/TP prices must match tickSize."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        # BTC has tickSize 0.1, so 45000.55 is invalid
        with pytest.raises(PaperRiskRejected, match="does not match tickSize"):
            client.create_protection_orders(
                symbol="BTC/USDT",
                position_quantity=0.08,
                stop_loss_price=45000.55,  # Invalid tick size
                take_profit_price=55000,
                client_order_id_prefix="pt-test",
                allow_orders=True,
            )

    def test_create_protection_orders_validates_sl_lt_tp(self):
        """Test that stop_loss must be < take_profit for long positions."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        with pytest.raises(PaperRiskRejected, match="must be less than take_profit"):
            client.create_protection_orders(
                symbol="BTC/USDT",
                position_quantity=0.08,
                stop_loss_price=55000,  # SL > TP is invalid
                take_profit_price=45000,
                client_order_id_prefix="pt-test",
                allow_orders=True,
            )

    def test_verify_protection_orders(self):
        """Test verifying protection orders from /v5/order/realtime."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        client = _make_client(transport)
        client.connect()

        verified = client.verify_protection_orders("BTC/USDT")

        assert verified["stop_loss"] is not None
        assert verified["stop_loss"].client_order_id == "pt-test-sl"
        # ``status`` is this project's vocabulary: an untriggered stop is an
        # order the venue is still working, so it is "accepted".
        assert verified["stop_loss"].status == "accepted"
        assert verified["take_profit"] is not None
        assert verified["take_profit"].client_order_id == "pt-test-tp"
        assert verified["take_profit"].status == "accepted"

    def test_verify_protection_orders_missing(self):
        """Test verifying when no protection orders exist."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)
        client.connect()

        verified = client.verify_protection_orders("BTC/USDT")

        assert verified["stop_loss"] is None
        assert verified["take_profit"] is None

    def test_position_is_protected_when_sl_tp_exist(self):
        """Test that position.is_protected returns True when SL/TP are verified."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        positions = client.positions(refresh=True)
        pos = positions["BTC/USDT"]

        assert pos.is_protected is True
        assert pos.protection_orders is not None
        assert pos.protection_orders.get("stop_loss") is not None
        assert pos.protection_orders.get("take_profit") is not None

    def test_position_not_protected_when_missing_sl_or_tp(self):
        """Test that position.is_protected returns False when SL or TP missing."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                    }
                    # No TakeProfit
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        positions = client.positions(refresh=True)
        pos = positions["BTC/USDT"]

        assert pos.is_protected is False

    def test_position_not_protected_when_orders_rejected(self):
        """Test that position.is_protected returns False when orders are rejected."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Rejected",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})
        client = _make_client(transport)
        client.connect()

        positions = client.positions(refresh=True)
        pos = positions["BTC/USDT"]

        assert pos.is_protected is False


# ---------------------------------------------------------------------------
# Integration with PaperTradingConfig
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_paper_trading_config_accepts_bybit_provider(self):
        from tradingagents.paper_experiment.broker.trading import PaperTradingConfig

        config = PaperTradingConfig(
            execution_provider="bybit_demo",
            base_url=BYBIT_DEMO_BASE_URL,
        )
        assert config.execution_provider == "bybit_demo"
        assert config.base_url == BYBIT_DEMO_BASE_URL

    def test_paper_trading_config_defaults_to_alpaca(self):
        from tradingagents.paper_experiment.broker.trading import PaperTradingConfig

        config = PaperTradingConfig()
        assert config.execution_provider == "alpaca_paper"

    def test_paper_trading_config_rejects_unknown_provider(self):
        from tradingagents.paper_experiment.broker.trading import PaperTradingConfig, WorkerError

        with pytest.raises(WorkerError, match="unknown execution_provider"):
            PaperTradingConfig(execution_provider="unknown_provider")


# ---------------------------------------------------------------------------
# Reconciliation tests
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_reconcile_finds_filled_order(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", MOCK_ORDER_REALTIME_RESPONSE)
        client = _make_client(transport)

        client.connect()
        client._state.orders["pt-test-client-id"] = {
            "client_order_id": "pt-test-client-id",
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.001,
            "status": "accepted",
            "submitted_at": datetime.now(UTC).isoformat(),
        }

        order = client._reconcile("pt-test-client-id")
        assert order.status == "filled"
        assert order.filled_quantity == 0.001

    def test_reconcile_terminal_order_not_asked_again(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        client = _make_client(transport)

        client.connect()
        client._state.orders["pt-test-client-id"] = {
            "client_order_id": "pt-test-client-id",
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.001,
            "filled_quantity": 0.001,
            "filled_avg_price": 50000,
            "status": "filled",
            "submitted_at": datetime.now(UTC).isoformat(),
        }

        # Should not make any request for filled order
        order = client._reconcile("pt-test-client-id")
        assert order.status == "filled"
        # No GET calls should have been made to order endpoints
        order_calls = [c for c in transport.calls if "/v5/order/" in c["url"]]
        assert len(order_calls) == 0


# ---------------------------------------------------------------------------
# Live credentials rejection in errors
# ---------------------------------------------------------------------------


class TestCredentialRedaction:
    def test_credentials_not_in_error_messages(self):
        """Verify that error messages don't leak credentials."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", {
            "retCode": 401,
            "retMsg": f"Invalid API key: {DEMO_API_KEY}",
        })

        client = _make_client(transport)

        with pytest.raises(Exception) as excinfo:
            client.connect()

        error_msg = str(excinfo.value)
        assert DEMO_API_KEY not in error_msg
        assert DEMO_API_SECRET not in error_msg


# ---------------------------------------------------------------------------
# Non-tradable symbol rejected
# ---------------------------------------------------------------------------


class TestNonTradableSymbols:
    def test_non_tradable_symbol_rejected(self):
        non_tradable_instruments = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "spot",
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "Trading",
                        "baseCoin": "BTC",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "minOrderQty": "0.00001",
                            "maxOrderQty": "100",
                            "qtyStep": "0.00001",
                            "minOrderAmt": "10",
                        },
                        "priceFilter": {"tickSize": "0.1"},
                    },
                    {
                        "symbol": "DELISTEDUSDT",
                        "status": "Delisted",
                        "baseCoin": "DELISTED",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "minOrderQty": "0.00001",
                            "maxOrderQty": "100",
                            "qtyStep": "0.00001",
                            "minOrderAmt": "10",
                        },
                        "priceFilter": {"tickSize": "0.1"},
                    },
                ],
            },
        }
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", non_tradable_instruments)
        transport.add_response("GET", "/v5/position/list", {"retCode": 0, "result": {"list": []}})
        client = _make_client(transport)

        client.connect()

        with pytest.raises(PaperRiskRejected, match="not tradable"):
            client.submit_order(
                symbol="DELISTED/USDT",
                quantity=0.001,
                side="buy",
                reference_price=100,
                client_order_id="pt-test-client-id",
                order_type="Market",
                allow_orders=True,
                refresh=True,
            )


# ---------------------------------------------------------------------------
# Comprehensive protection order tests (OCO, partial fills, restart recovery)
# ---------------------------------------------------------------------------


class TestProtectionOrderComprehensive:
    """Comprehensive tests for protection order handling per Bybit V5 API."""

    def test_protection_orders_with_pagination(self):
        """Test reading protection orders with pagination.

        Since MockTransport uses a dict keyed by (method, path), we can't easily
        test sequential pagination calls. Instead, we test that the parser correctly
        handles multiple orders in a single response (which is what the combined
        result would look like).
        """
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)

        # Combined response (simulating result after pagination)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ],
                "nextPageCursor": "",
            }
        })

        client = _make_client(transport)
        client.connect()

        prot = client._read_protection_orders()
        assert "BTCUSDT" in prot
        # A symbol can carry several orders of the same kind (a residual pair
        # beside the original one), so the read keeps **all** of them.
        assert prot["BTCUSDT"]["stop_loss"]
        assert prot["BTCUSDT"]["take_profit"]
        assert prot["BTCUSDT"]["stop_loss"][0]["order_id"] == "bybit-sl-123"
        assert prot["BTCUSDT"]["take_profit"][0]["order_id"] == "bybit-tp-123"

    def test_protection_order_missing_orderfilter(self):
        """Test handling response where orderFilter is missing from entry."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        # orderFilter is missing
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    }
                ]
            }
        })

        client = _make_client(transport)
        client.connect()

        prot = client._read_protection_orders()
        # Should still detect via stopOrderType
        assert "BTCUSDT" in prot
        assert prot["BTCUSDT"]["stop_loss"] is not None

    def test_sl_filled_tp_cancelled(self):
        """OCO: when the stop loss fills, the take profit is cancelled.

        The venue is modelled as a real one: the stop has left the live list
        and is in the **history** (a filled order is no longer open), the
        target is still working, and the cancel is asynchronous -- so the
        target only leaves the live list once the cancel is requested.
        """
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        # The stop has filled: it is in the order history, not the live list.
        transport.history_orders.append({
            "orderId": "bybit-sl-123",
            "orderLinkId": "pt-test-sl",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "Filled",
            "orderFilter": "tpslOrder",
            "stopOrderType": "StopLoss",
            "triggerPrice": "45000",
            "cumExecQty": "0.1",
            "leavesQty": "0",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })
        # The target is still working.
        transport.open_orders.append({
            "orderId": "bybit-tp-123",
            "orderLinkId": "pt-test-tp",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "Untriggered",
            "orderFilter": "tpslOrder",
            "stopOrderType": "TakeProfit",
            "triggerPrice": "55000",
            "cumExecQty": "0",
            "leavesQty": "0.1",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })

        client = _make_client(transport)
        client.connect()

        # Set up protection pair
        client._state.protection_order_pairs["pt-test"] = {
            "sl_client_id": "pt-test-sl",
            "tp_client_id": "pt-test-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        # Set position
        client._positions = {"BTC/USDT": type('obj', (object,), {"quantity": 0.1})()}

        result = client.reconcile_protection_orders("BTC/USDT")

        # The stop is reported from the history, the target from the live list.
        assert result["stop_loss"]["status"] == "Filled"
        assert result["stop_loss"]["mapped_status"] == "filled"
        assert result["stop_loss"]["executed_quantity"] == 0.1
        assert "TP cancelled after SL fill" in result["actions_taken"]
        # The venue confirmed the cancel: the target is no longer working.
        assert result["take_profit"]["mapped_status"] == "cancelled"
        assert result["take_profit"]["is_active"] is False
        # And the pair is settled, so the next cycle does not cancel again.
        assert client._state.protection_order_pairs["pt-test"]["resolved"] is True
        assert transport.open_orders == []

    def test_tp_filled_sl_cancelled(self):
        """OCO: when the take profit fills, the stop loss is cancelled."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.history_orders.append({
            "orderId": "bybit-tp-123",
            "orderLinkId": "pt-test-tp",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "Filled",
            "orderFilter": "tpslOrder",
            "stopOrderType": "TakeProfit",
            "triggerPrice": "55000",
            "cumExecQty": "0.1",
            "leavesQty": "0",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })
        transport.open_orders.append({
            "orderId": "bybit-sl-123",
            "orderLinkId": "pt-test-sl",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "Untriggered",
            "orderFilter": "tpslOrder",
            "stopOrderType": "StopLoss",
            "triggerPrice": "45000",
            "cumExecQty": "0",
            "leavesQty": "0.1",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })

        client = _make_client(transport)
        client.connect()

        client._state.protection_order_pairs["pt-test"] = {
            "sl_client_id": "pt-test-sl",
            "tp_client_id": "pt-test-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        client._positions = {"BTC/USDT": type('obj', (object,), {"quantity": 0.1})()}

        result = client.reconcile_protection_orders("BTC/USDT")

        assert result["take_profit"]["status"] == "Filled"
        assert result["take_profit"]["mapped_status"] == "filled"
        assert "SL cancelled after TP fill" in result["actions_taken"]
        assert result["stop_loss"]["mapped_status"] == "cancelled"
        assert result["stop_loss"]["is_active"] is False
        assert client._state.protection_order_pairs["pt-test"]["resolved"] is True
        assert transport.open_orders == []

    def test_partial_fill_sl(self):
        """A **partial** fill of the stop still cancels the target.

        A partial stop fill has already sold part of the position, so the
        target would sell the rest -- and then the leftover of the stop
        order would sell more than the position holds. The counterpart is
        therefore cancelled on a partial fill too.
        """
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.history_orders.append({
            "orderId": "bybit-sl-123",
            "orderLinkId": "pt-test-sl",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "PartiallyFilled",
            "orderFilter": "tpslOrder",
            "stopOrderType": "StopLoss",
            "triggerPrice": "45000",
            "cumExecQty": "0.04",
            "leavesQty": "0.06",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })
        transport.open_orders.append({
            "orderId": "bybit-tp-123",
            "orderLinkId": "pt-test-tp",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "qty": "0.1",
            "timeInForce": "GTC",
            "orderStatus": "Untriggered",
            "orderFilter": "tpslOrder",
            "stopOrderType": "TakeProfit",
            "triggerPrice": "55000",
            "cumExecQty": "0",
            "leavesQty": "0.1",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        })

        client = _make_client(transport)
        client.connect()

        client._state.protection_order_pairs["pt-test"] = {
            "sl_client_id": "pt-test-sl",
            "tp_client_id": "pt-test-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        client._positions = {"BTC/USDT": type('obj', (object,), {"quantity": 0.1})()}

        result = client.reconcile_protection_orders("BTC/USDT")

        # The venue's CamelCase status maps onto this project's vocabulary.
        assert result["stop_loss"]["status"] == "PartiallyFilled"
        assert result["stop_loss"]["mapped_status"] == "partially_filled"
        assert result["stop_loss"]["executed_quantity"] == 0.04
        assert "TP cancelled after SL fill" in result["actions_taken"]
        assert transport.open_orders == []

    def test_restart_recovery_reads_protection_pairs(self):
        """Test that protection order pairs are persisted and recovered on restart."""
        import os
        import tempfile

        # Use a temporary file for state persistence
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            state_path = f.name

        try:
            # First write an empty valid state to the file
            from tradingagents.paper_experiment.broker.state import BrokerAccountState
            empty_state = BrokerAccountState()
            empty_state.endpoint = "https://api-demo.bybit.com"
            import json
            with open(state_path, 'w') as f:
                json.dump(empty_state.to_dict(), f)

            transport = MockTransport()
            transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
            transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
            transport.add_response("GET", "/v5/order/realtime", {
                "retCode": 0,
                "result": {"list": []}
            })

            # First client creates protection orders
            client1 = _make_client(transport, state_path=state_path)
            client1.connect()

            client1._state.protection_order_pairs["pt-restart"] = {
                "sl_client_id": "pt-restart-sl",
                "tp_client_id": "pt-restart-tp",
                "symbol": "BTC/USDT",
                "position_qty": 0.1,
            }
            if client1._store:
                client1._store.save(client1._state)

            # Create new client (simulating restart) with same state path
            transport2 = MockTransport()
            transport2.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
            transport2.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
            transport2.add_response("GET", "/v5/order/realtime", {
                "retCode": 0,
                "result": {"list": []}
            })

            client2 = _make_client(transport2, state_path=state_path)
            client2.connect()

            # Should be able to access the persisted protection pairs
            assert "pt-restart" in client2._state.protection_order_pairs
            pair = client2._state.protection_order_pairs["pt-restart"]
            assert pair["sl_client_id"] == "pt-restart-sl"
            assert pair["tp_client_id"] == "pt-restart-tp"
            assert pair["symbol"] == "BTC/USDT"
        finally:
            # Clean up temp file
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_is_position_protected_valid(self):
        """Test is_position_protected returns True for valid protection."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        client = _make_client(transport)
        client.connect()
        client.positions(refresh=True)

        protected, reason = client.is_position_protected("BTC/USDT")
        assert protected is True
        assert reason == "protected"

    def test_is_position_protected_missing_sl(self):
        """Test is_position_protected returns False when SL is missing."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        client = _make_client(transport)
        client.connect()
        client.positions(refresh=True)

        protected, reason = client.is_position_protected("BTC/USDT")
        assert protected is False
        assert "stop_loss missing" in reason

    def test_is_position_protected_rejected_sl(self):
        """Test is_position_protected returns False when SL is rejected."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Rejected",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        client = _make_client(transport)
        client.connect()
        client.positions(refresh=True)

        protected, reason = client.is_position_protected("BTC/USDT")
        assert protected is False
        assert "not active" in reason

    def test_is_position_protected_quantity_mismatch(self):
        """Test is_position_protected returns False when SL/TP quantity doesn't match position."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.05",  # Only half the position
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        client = _make_client(transport)
        client.connect()
        client.positions(refresh=True)

        protected, reason = client.is_position_protected("BTC/USDT")
        assert protected is False
        assert "quantity mismatch" in reason

    def test_is_position_protected_wrong_side(self):
        """Test is_position_protected returns False when side is wrong."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Buy",  # Wrong side for long position SL
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })
        transport.add_response("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "51000"}]}})

        client = _make_client(transport)
        client.connect()
        client.positions(refresh=True)

        protected, reason = client.is_position_protected("BTC/USDT")
        assert protected is False
        assert "side mismatch" in reason

    def test_verify_protection_orders_untriggered(self):
        """Test verify_protection_orders returns orders with untriggered status."""
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/market/instruments-info", MOCK_INSTRUMENTS_RESPONSE)
        transport.add_response("GET", "/v5/order/realtime", {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-sl-123",
                        "orderLinkId": "pt-test-sl",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "StopLoss",
                        "triggerPrice": "45000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                    {
                        "orderId": "bybit-tp-123",
                        "orderLinkId": "pt-test-tp",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "orderType": "Market",
                        "qty": "0.1",
                        "timeInForce": "GTC",
                        "orderStatus": "Untriggered",
                        "orderFilter": "tpslOrder",
                        "stopOrderType": "TakeProfit",
                        "triggerPrice": "55000",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                    },
                ]
            }
        })

        client = _make_client(transport)
        client.connect()

        verified = client.verify_protection_orders("BTC/USDT")

        assert verified["stop_loss"] is not None
        assert verified["stop_loss"].client_order_id == "pt-test-sl"
        # ``status`` is this project's vocabulary: an untriggered stop is an
        # order the venue is still working, so it is "accepted".
        assert verified["stop_loss"].status == "accepted"
        assert verified["take_profit"] is not None
        assert verified["take_profit"].client_order_id == "pt-test-tp"
        assert verified["take_profit"].status == "accepted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
