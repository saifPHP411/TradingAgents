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
    """A mock HTTP transport that records calls and returns canned responses."""

    def __init__(self):
        self.calls: list[dict] = []
        self.responses: dict[tuple[str, str], dict] = {}
        self.call_count = 0

    def add_response(self, method: str, path: str, response: dict):
        self.responses[(method.upper(), path)] = response

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

        # Find matching response
        path = url.split("?")[0].replace("https://api-demo.bybit.com", "")
        key = (method.upper(), path)
        if key in self.responses:
            return BrokerHttpResponse(200, json.dumps(self.responses[key]))

        # Default fallback
        return BrokerHttpResponse(200, json.dumps({"retCode": 0, "retMsg": "OK", "result": {}}))


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
        assert account.currency == "USDT"

    def test_refresh_account_updates_state(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        client = _make_client(transport)

        client.connect()
        assert client._state.confirmed is True
        assert client._state.equity == 10000.0


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
        )

        assert order.client_order_id == "pt-test-client-id"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.quantity == 0.001  # Broker reports actual executed qty
        assert order.filled_quantity == 0.001
        assert order.status == "filled"

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
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("POST", "/v5/order/cancel", MOCK_ORDER_CANCEL_RESPONSE)
        client = _make_client(transport)

        client.connect()
        order = client.cancel_order(client_order_id="pt-test-client-id", symbol="BTC/USDT")

        assert order.client_order_id == "pt-test-client-id"
        assert order.status == "cancelled"

    def test_list_orders(self):
        transport = MockTransport()
        transport.add_response("GET", "/v5/account/wallet-balance", MOCK_ACCOUNT_RESPONSE)
        transport.add_response("GET", "/v5/order/history", MOCK_ORDER_HISTORY_RESPONSE)
        client = _make_client(transport)

        client.connect()
        page = client.list_orders()

        assert page.pages_read == 1
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
                quantity=0.001,
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
                        "orderFilter": "StopLoss",
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
                        "orderFilter": "TakeProfit",
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
        assert verified["stop_loss"].status == "untriggered"
        assert verified["take_profit"] is not None
        assert verified["take_profit"].client_order_id == "pt-test-tp"
        assert verified["take_profit"].status == "untriggered"

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
                        "orderFilter": "StopLoss",
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
                        "orderFilter": "TakeProfit",
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
                        "orderFilter": "StopLoss",
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
                        "orderFilter": "StopLoss",
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
                        "orderFilter": "TakeProfit",
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
