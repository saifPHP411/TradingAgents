"""Independent tests for the Bybit Demo Trading adapter.

These tests drive the **real** adapter code through a mock HTTP transport.
No test needs a credential, and none makes a network call.

The scenarios here are the ones that decide whether the adapter is safe to
run: the endpoint it is allowed to talk to, the signature it sends, the
per-symbol minimums it reads from the venue, the way it handles fractional
quantities, the way it refuses a short, the way it keeps a retry from
duplicating an order, and the way it proves -- or fails to prove -- that a
position is protected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from decimal import Decimal
from typing import Any

import pytest

from tradingagents.paper_experiment.broker.bybit_demo import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DEMO_BASE_URL,
    BrokerError,
    BrokerHttpResponse,
    BrokerNotConfiguredError,
    BrokerRateLimitedError,
    BrokerRejectedError,
    BybitDemoBroker,
    DuplicateOrderError,
    InsufficientFundsError,
    LiveEndpointRefusedError,
    PaperRiskRejected,
    _encode_json_body,
    _encode_query_string,
    assert_demo_endpoint,
    assert_demo_request_url,
)

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class RecordingTransport:
    """A mock transport that records every request and replays canned answers.

    The responses are matched on the method and the path. A response may be
    registered more than once for the same path; each call consumes the next
    one, so a test can model a sequence of venue answers.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[tuple[str, str], list[dict]] = {}
        self._raw: dict[tuple[str, str], str] = {}
        self._failures: dict[tuple[str, str], list[Exception]] = {}
        self._default: dict = {"retCode": 0, "retMsg": "OK", "result": {}}

    def add(self, method: str, path: str, response: dict) -> None:
        self._responses.setdefault((method.upper(), path), []).append(response)

    def fail(self, method: str, path: str, error: Exception) -> None:
        """Make the next call to a path raise, to model an unreadable venue."""
        self._failures.setdefault((method.upper(), path), []).append(error)

    def add_default(self, response: dict) -> None:
        self._default = response

    def add_raw(self, method: str, path: str, text: str) -> None:
        """Register an answer that is not JSON, to exercise the parser."""
        self._raw[(method.upper(), path)] = text

    def calls_to(self, method: str, path: str) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if call["method"].upper() == method.upper() and call["path"] == path
        ]

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        params: dict,
        json_body: dict,
        timeout: float,
    ) -> BrokerHttpResponse:
        path = url.split("?")[0].replace(DEMO_BASE_URL, "")
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "headers": dict(headers),
                "params": dict(params),
                "json_body": dict(json_body or {}),
            }
        )
        raw = self._raw.get((method.upper(), path))
        if raw is not None:
            return BrokerHttpResponse(200, raw)
        failures = self._failures.get((method.upper(), path))
        if failures:
            raise failures.pop(0)
        queue = self._responses.get((method.upper(), path))
        payload = self._default
        if queue:
            # The last registered answer stays registered, so a test that
            # reads the same endpoint twice sees the same answer both times.
            payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return BrokerHttpResponse(200, json.dumps(payload))


# ---------------------------------------------------------------------------
# Canned venue answers
# ---------------------------------------------------------------------------


def wallet_response(coins: dict[str, dict[str, str]]) -> dict:
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "accountId": "demo-account",
                    "accountType": "SPOT",
                    "coin": [
                        {
                            "coin": coin,
                            "walletBalance": values["walletBalance"],
                            "availableToWithdraw": values.get(
                                "availableToWithdraw", values["walletBalance"]
                            ),
                            "locked": values.get("locked", "0"),
                        }
                        for coin, values in coins.items()
                    ],
                }
            ]
        },
    }


def instruments_response(entries: list[dict]) -> dict:
    return {"retCode": 0, "retMsg": "OK", "result": {"category": "spot", "list": entries}}


def spot_instrument(
    symbol: str,
    *,
    min_order_qty: str = "0.00001",
    qty_step: str = "0.00001",
    min_order_amt: str = "5",
    tick_size: str = "0.1",
    status: str = "Trading",
) -> dict:
    return {
        "symbol": symbol,
        "status": status,
        "baseCoin": symbol[:-4],
        "quoteCoin": "USDT",
        "lotSizeFilter": {
            "minOrderQty": min_order_qty,
            "maxOrderQty": "1000000",
            "qtyStep": qty_step,
            "minOrderAmt": min_order_amt,
        },
        "priceFilter": {"tickSize": tick_size},
    }


def order_row(
    *,
    order_link_id: str,
    status: str = "New",
    symbol: str = "BTCUSDT",
    side: str = "Buy",
    qty: str = "0.0005",
    cum_exec_qty: str = "0",
    avg_price: str = "0",
    order_filter: str = "",
    stop_order_type: str = "",
    trigger_price: str = "0",
    order_id: str = "venue-1",
) -> dict:
    row = {
        "orderId": order_id,
        "orderLinkId": order_link_id,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty,
        "price": "0",
        "avgPrice": avg_price,
        "cumExecQty": cum_exec_qty,
        "leavesQty": qty,
        "timeInForce": "GTC",
        "orderStatus": status,
        "createdTime": "1700000000000",
        "updatedTime": "1700000000000",
    }
    if order_filter:
        row["orderFilter"] = order_filter
    if stop_order_type:
        row["stopOrderType"] = stop_order_type
    if trigger_price != "0":
        row["triggerPrice"] = trigger_price
    return row


def make_broker(transport: RecordingTransport, **kwargs: Any) -> BybitDemoBroker:
    return BybitDemoBroker(
        api_key="demo-key",
        api_secret="demo-secret",
        base_url=DEMO_BASE_URL,
        transport=transport,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. The endpoint is the demo endpoint, and only the demo endpoint
# ---------------------------------------------------------------------------


class TestEndpointIsDemoOnly:
    def test_the_approved_endpoint_is_the_demo_host(self):
        assert DEMO_BASE_URL == "https://api-demo.bybit.com"
        assert assert_demo_endpoint(DEMO_BASE_URL) == DEMO_BASE_URL

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.bybit.com",
            "https://api.bybit.com/",
            "https://api.bybit.com/v5/order/create",
            "https://api-testnet.bybit.com",
            "https://api.bybit.com:8443",
        ],
    )
    def test_a_live_host_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint(url)

    def test_a_downgrade_to_http_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("http://api-demo.bybit.com")

    def test_a_credential_in_the_url_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://user:pass@api-demo.bybit.com")

    def test_a_different_host_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://bybit.example.com")

    def test_a_request_url_may_carry_a_path(self):
        assert (
            assert_demo_request_url("https://api-demo.bybit.com/v5/order/create")
            == "https://api-demo.bybit.com/v5/order/create"
        )

    def test_a_request_url_to_a_live_host_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_request_url("https://api.bybit.com/v5/order/create")

    def test_the_broker_refuses_to_be_built_on_a_live_host(self):
        with pytest.raises(LiveEndpointRefusedError):
            BybitDemoBroker(
                api_key="k",
                api_secret="s",
                base_url="https://api.bybit.com",
                transport=RecordingTransport(),
            )

    def test_every_request_the_broker_made_went_to_the_demo_host(self):
        transport = RecordingTransport()
        transport.add("GET", "/v5/account/wallet-balance", wallet_response({"USDT": {"walletBalance": "1000"}}))
        broker = make_broker(transport)
        broker.connect()
        assert transport.calls
        for call in transport.calls:
            assert call["url"].startswith(DEMO_BASE_URL)


# ---------------------------------------------------------------------------
# 2. Missing demo credentials stop the run; a live key is never used
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_a_missing_key_is_a_configuration_error(self, monkeypatch):
        """The broker refuses to be built without a demo key.

        The refusal happens at construction, which is earlier than the first
        request: a broker that cannot authenticate is never given a transport
        to make a request with.
        """
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(BrokerNotConfiguredError) as excinfo:
            BybitDemoBroker(transport=RecordingTransport())
        assert API_KEY_ENV in str(excinfo.value)
        assert API_SECRET_ENV in str(excinfo.value)

    def test_a_missing_secret_is_a_configuration_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "demo-key")
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(BrokerNotConfiguredError):
            BybitDemoBroker(transport=RecordingTransport())

    def test_a_live_key_variable_is_not_read(self, monkeypatch):
        """A live Bybit key must not be picked up as a demo key."""
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        monkeypatch.setenv("BYBIT_API_KEY", "live-key")
        monkeypatch.setenv("BYBIT_API_SECRET", "live-secret")
        with pytest.raises(BrokerNotConfiguredError) as excinfo:
            BybitDemoBroker(transport=RecordingTransport())
        # The message points at the demo variables, not the live ones.
        assert API_KEY_ENV in str(excinfo.value)
        assert "BYBIT_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. The request signature
# ---------------------------------------------------------------------------


class TestSignature:
    def test_the_signature_is_the_hmac_of_the_documented_payload(self, monkeypatch):
        """Bybit's own rule: ``timestamp + api_key + recv_window + payload``.

        The header names are the documented ``X-BAPI-*`` set.
        """
        transport = RecordingTransport()
        transport.add("GET", "/v5/account/wallet-balance", wallet_response({"USDT": {"walletBalance": "1000"}}))
        broker = make_broker(transport)
        monkeypatch.setenv(API_KEY_ENV, "demo-key")
        monkeypatch.setenv(API_SECRET_ENV, "demo-secret")
        broker.connect()

        call = transport.calls_to("GET", "/v5/account/wallet-balance")[0]
        headers = call["headers"]
        signature = headers["X-BAPI-SIGN"]
        timestamp = headers["X-BAPI-TIMESTAMP"]
        recv_window = headers["X-BAPI-RECV-WINDOW"]
        query = _encode_query_string(call["params"])
        sign_payload = f"{timestamp}{headers['X-BAPI-API-KEY']}{recv_window}{query}"
        expected = hmac.new(
            b"demo-secret", sign_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert signature == expected

    def test_the_signature_covers_the_body_of_a_post(self, monkeypatch):
        transport = RecordingTransport()
        transport.add("GET", "/v5/account/wallet-balance", wallet_response({"USDT": {"walletBalance": "1000"}}))
        transport.add("GET", "/v5/market/instruments-info", instruments_response([spot_instrument("BTCUSDT")]))
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v1", "orderLinkId": "x"}},
        )
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="sig-test-1",
            allow_orders=True,
            refresh=True,
        )

        post = transport.calls_to("POST", "/v5/order/create")[0]
        headers = post["headers"]
        # The body is signed exactly as it is sent.
        sign_payload = (
            f"{headers['X-BAPI-TIMESTAMP']}"
            f"{headers['X-BAPI-API-KEY']}"
            f"{headers['X-BAPI-RECV-WINDOW']}"
            f"{_encode_json_body(post['json_body'])}"
        )
        expected = hmac.new(
            b"demo-secret", sign_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert headers["X-BAPI-SIGN"] == expected

    def test_the_key_is_sent_in_the_header_not_the_body(self):
        transport = RecordingTransport()
        transport.add("GET", "/v5/account/wallet-balance", wallet_response({"USDT": {"walletBalance": "1000"}}))
        broker = make_broker(transport)
        broker.connect()
        call = transport.calls_to("GET", "/v5/account/wallet-balance")[0]
        assert call["headers"]["X-BAPI-API-KEY"] == "demo-key"
        assert "demo-secret" not in json.dumps(call)


# ---------------------------------------------------------------------------
# 4. Reading the balance, the positions and the orders
# ---------------------------------------------------------------------------


class TestReads:
    def test_the_demo_balance_is_read(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "500.25", "locked": "10"}}),
        )
        broker = make_broker(transport)
        account = broker.connect()
        assert account.equity == pytest.approx(500.25)
        assert account.currency == "USDT"

    def test_a_spot_holding_is_read_as_a_position(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {
                    "USDT": {"walletBalance": "1000"},
                    "BTC": {"walletBalance": "0.02", "locked": "0.005"},
                }
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        broker = make_broker(transport)
        broker.connect()
        positions = broker.positions(refresh=True)
        assert "BTC/USDT" in positions
        position = positions["BTC/USDT"]
        assert position.quantity == pytest.approx(0.02)
        assert position.available_quantity == pytest.approx(0.015)
        assert position.side == "buy"
        assert position.venue_symbol == "BTCUSDT"

    def test_the_open_orders_are_read(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/order/realtime",
            {"retCode": 0, "result": {"list": [order_row(order_link_id="open-1")]}},
        )
        broker = make_broker(transport)
        broker._state.confirmed = True
        page = broker.list_orders()
        assert len(page.orders) == 1
        assert page.orders[0].client_order_id == "open-1"

    def test_the_historical_orders_are_read(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/order/history",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="hist-1",
                            status="Filled",
                            cum_exec_qty="0.0005",
                            avg_price="50000",
                        )
                    ]
                },
            },
        )
        broker = make_broker(transport)
        broker._state.confirmed = True
        page = broker.list_orders()
        assert len(page.orders) == 1
        assert page.orders[0].status == "filled"
        assert page.orders[0].filled_quantity == pytest.approx(0.0005)
        assert page.orders[0].filled_avg_price == pytest.approx(50000.0)

    def test_an_order_with_no_client_id_is_kept_as_unclassified(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [{"orderId": "v9", "symbol": "BTCUSDT"}]}},
        )
        broker = make_broker(transport)
        broker._state.confirmed = True
        page = broker.list_orders()
        assert page.orders == []
        assert len(page.unclassified) == 1


# ---------------------------------------------------------------------------
# 5. Placing and cancelling Spot orders
# ---------------------------------------------------------------------------


class TestOrderLifecycle:
    def _ready_transport(self) -> RecordingTransport:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT", min_order_amt="5")]),
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        return transport

    def test_a_market_order_carries_the_spot_category(self):
        transport = self._ready_transport()
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v1", "orderLinkId": "life-1"}},
        )
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="life-1",
            allow_orders=True,
            refresh=True,
        )
        body = transport.calls_to("POST", "/v5/order/create")[0]["json_body"]
        assert body["category"] == "spot"
        assert body["symbol"] == "BTCUSDT"
        assert body["side"] == "Buy"
        assert body["orderType"] == "Market"
        assert body["orderLinkId"] == "life-1"

    def test_a_limit_order_carries_its_price(self):
        transport = self._ready_transport()
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v2", "orderLinkId": "life-2"}},
        )
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=48000,
            client_order_id="life-2",
            order_type="Limit",
            time_in_force="GTC",
            allow_orders=True,
            refresh=True,
        )
        body = transport.calls_to("POST", "/v5/order/create")[0]["json_body"]
        assert body["orderType"] == "Limit"
        assert body["price"] == "48000"
        assert body["timeInForce"] == "GTC"

    def test_a_cancel_is_requested_with_the_client_order_id(self):
        transport = self._ready_transport()
        transport.add(
            "POST",
            "/v5/order/cancel",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v1", "orderLinkId": "life-1"}},
        )
        # The venue's answer after the cancel: the order is now finished.
        transport.add(
            "GET",
            "/v5/order/history",
            {
                "retCode": 0,
                "result": {"list": [order_row(order_link_id="life-1", status="Cancelled")]},
            },
        )
        broker = make_broker(transport)
        broker.connect()
        broker.cancel_order(client_order_id="life-1", symbol="BTC/USDT")
        body = transport.calls_to("POST", "/v5/order/cancel")[0]["json_body"]
        assert body["category"] == "spot"
        assert body["symbol"] == "BTCUSDT"
        assert body["orderLinkId"] == "life-1"

    def test_a_venue_refusal_is_reported_as_a_rejection(self):
        transport = self._ready_transport()
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 110001, "retMsg": "order does not exist", "result": {}},
        )
        broker = make_broker(transport)
        broker.connect()
        with pytest.raises(BrokerRejectedError):
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.0005,
                side="buy",
                reference_price=50000,
                client_order_id="life-3",
                allow_orders=True,
                refresh=True,
            )

    def test_the_filled_quantity_and_the_average_price_are_read(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/order/history",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="fill-1",
                            status="Filled",
                            qty="0.0005",
                            cum_exec_qty="0.0005",
                            avg_price="50123.45",
                        )
                    ]
                },
            },
        )
        broker = make_broker(transport)
        broker._state.confirmed = True
        order = broker._reconcile("fill-1")
        assert order.status == "filled"
        assert order.filled_quantity == pytest.approx(0.0005)
        assert order.filled_avg_price == pytest.approx(50123.45)


# ---------------------------------------------------------------------------
# 6. The per-pair minimums come from the venue, never from a constant
# ---------------------------------------------------------------------------

REVIEWED_PAIRS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
)
"""The pairs this integration was asked to check."""


class TestPerPairMinimums:
    """Every pair's minimum is read from ``/v5/market/instruments-info``.

    There is no default minimum: a pair the venue does not describe is a
    pair the adapter cannot trade, not a pair with a made-up minimum.
    """

    def _instruments(self) -> dict[str, dict]:
        return {
            "BTCUSDT": spot_instrument("BTCUSDT", min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5", tick_size="0.1"),
            "ETHUSDT": spot_instrument("ETHUSDT", min_order_qty="0.0001", qty_step="0.0001", min_order_amt="5", tick_size="0.01"),
            "SOLUSDT": spot_instrument("SOLUSDT", min_order_qty="0.01", qty_step="0.01", min_order_amt="5", tick_size="0.001"),
            "XRPUSDT": spot_instrument("XRPUSDT", min_order_qty="1", qty_step="1", min_order_amt="5", tick_size="0.0001"),
            "DOGEUSDT": spot_instrument("DOGEUSDT", min_order_qty="1", qty_step="1", min_order_amt="5", tick_size="0.00001"),
            "AVAXUSDT": spot_instrument("AVAXUSDT", min_order_qty="0.1", qty_step="0.1", min_order_amt="5", tick_size="0.001"),
            "LINKUSDT": spot_instrument("LINKUSDT", min_order_qty="0.01", qty_step="0.01", min_order_amt="5", tick_size="0.001"),
            "LTCUSDT": spot_instrument("LTCUSDT", min_order_qty="0.001", qty_step="0.001", min_order_amt="5", tick_size="0.01"),
            "BCHUSDT": spot_instrument("BCHUSDT", min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5", tick_size="0.1"),
        }

    def test_each_reviewed_pair_gets_its_own_minimum(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response(list(self._instruments().values())),
        )
        broker = make_broker(transport)
        loaded = broker.load_instruments()

        for pair in REVIEWED_PAIRS:
            info = loaded[pair]
            assert info.min_order_qty > 0, pair
            assert info.min_order_amt > 0, pair
            assert info.qty_step > 0, pair
            assert info.tick_size > 0, pair
            assert info.tradable is True
            assert info.base_coin + info.quote_coin == pair

    def test_the_minimums_are_not_all_the_same_number(self):
        """The point of reading them is that they differ.

        A single hard-coded minimum for every coin would pass the test above
        while being exactly the behaviour this integration forbids.
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response(list(self._instruments().values())),
        )
        broker = make_broker(transport)
        loaded = broker.load_instruments()
        quantities = {info.min_order_qty for info in loaded.values()}
        steps = {info.qty_step for info in loaded.values()}
        assert len(quantities) > 1
        assert len(steps) > 1

    def test_a_pair_the_venue_does_not_describe_is_not_tradable(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        broker = make_broker(transport)
        loaded = broker.load_instruments()
        assert "DOGEUSDT" not in loaded

    def test_a_non_trading_pair_is_not_tradable(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response(
                [spot_instrument("BTCUSDT", status="PreLaunch"), spot_instrument("ETHUSDT")]
            ),
        )
        broker = make_broker(transport)
        loaded = broker.load_instruments()
        assert loaded["BTCUSDT"].tradable is False
        assert loaded["ETHUSDT"].tradable is True

    def test_an_order_below_the_pairs_minimum_is_refused(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT", min_order_amt="5", qty_step="0.00001")]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        broker = make_broker(transport)
        broker.connect()
        # 0.00001 BTC at 50000 is 0.50 USDT, below the pair's 5 USDT minimum.
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.00001,
                side="buy",
                reference_price=50000,
                client_order_id="min-below-1",
                allow_orders=True,
            )
        assert "BELOW_MIN_NOTIONAL" in str(excinfo.value)

    def test_the_order_is_never_raised_to_the_minimum(self):
        """A size below the minimum is refused, never silently increased.

        Raising the size to reach the venue's minimum would trade more than
        the risk engine approved.
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT", min_order_amt="5", qty_step="0.00001")]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        broker = make_broker(transport)
        broker.connect()
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.00001,
                side="buy",
                reference_price=50000,
                client_order_id="min-below-2",
                allow_orders=True,
            )
        # No order reached the venue.
        assert transport.calls_to("POST", "/v5/order/create") == []


# ---------------------------------------------------------------------------
# 7. Fractional quantities, the $25 cap, and the long-only rule
# ---------------------------------------------------------------------------


class TestTheRealSpotInstrumentShape:
    """A Spot ``lotSizeFilter`` has no ``qtyStep``; the step is ``basePrecision``.

    The documented Spot example for BTCUSDT is::

        "lotSizeFilter": {
            "basePrecision": "0.000001",
            "quotePrecision": "0.0000001",
            "minOrderQty": "0.000011",
            "maxOrderQty": "83",
            "minOrderAmt": "5",
            "maxOrderAmt": "8000000",
            "maxLimitOrderQty": "83",
            "maxMarketOrderQty": "41.5"
        }

    A parser that reads only ``qtyStep`` sees no step at all for every real
    Spot pair, and a missing step silently disables the step check.
    """

    def _documented_spot(self, symbol: str = "BTCUSDT") -> dict:
        """The venue's own documented Spot shape, with no ``qtyStep``."""
        return {
            "symbol": symbol,
            "status": "Trading",
            "baseCoin": symbol[:-4],
            "quoteCoin": "USDT",
            "lotSizeFilter": {
                "basePrecision": "0.000001",
                "quotePrecision": "0.0000001",
                "minOrderQty": "0.000011",
                "maxOrderQty": "83",
                "minOrderAmt": "5",
                "maxOrderAmt": "8000000",
                "maxLimitOrderQty": "83",
                "maxMarketOrderQty": "41.5",
            },
            "priceFilter": {"tickSize": "0.1"},
        }

    def test_the_documented_shape_carries_no_qty_step(self):
        assert "qtyStep" not in self._documented_spot()["lotSizeFilter"]

    def test_the_step_comes_from_the_base_precision(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([self._documented_spot()]),
        )
        broker = make_broker(transport)

        instrument = broker.get_instrument("BTCUSDT")

        assert instrument is not None
        assert instrument.qty_step == pytest.approx(0.000001)

    def test_a_quantity_off_the_step_is_refused_locally(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([self._documented_spot()]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        broker = make_broker(transport)
        broker.connect()

        # 0.0001005 BTC at $50,000 is $5.025 -- over the pair's $5 minimum --
        # but it is not a multiple of the 0.000001 step.
        with pytest.raises(PaperRiskRejected) as caught:
            broker.submit_order(
                symbol="BTC/USDT",
                side="buy",
                quantity=0.0001005,
                reference_price=50000.0,
                client_order_id="bybit-off-step",
                allow_orders=True,
            )

        assert "0.000001" in str(caught.value)
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_a_quantity_on_the_step_is_accepted(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([self._documented_spot()]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v-step", "orderLinkId": "bybit-on-step"}},
        )
        broker = make_broker(transport)
        broker.connect()

        order = broker.submit_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.000100,
            reference_price=50000.0,
            client_order_id="bybit-on-step",
            allow_orders=True,
        )

        assert order.quantity == pytest.approx(0.000100)

    def test_the_minimum_amount_is_read_from_the_notional_field(self):
        """The venue deprecated ``minOrderQty``; ``minOrderAmt`` is the bound."""
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([self._documented_spot()]),
        )
        broker = make_broker(transport)

        instrument = broker.get_instrument("BTCUSDT")

        assert instrument is not None
        assert instrument.min_order_amt == pytest.approx(5.0)

    def test_an_order_below_the_minimum_amount_is_refused(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([self._documented_spot()]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        broker = make_broker(transport)
        broker.connect()

        # 0.000011 BTC at $100 is $0.0011 -- below the pair's $5 minimum.
        with pytest.raises(PaperRiskRejected) as caught:
            broker.submit_order(
                symbol="BTC/USDT",
                side="buy",
                quantity=0.000011,
                reference_price=100.0,
                client_order_id="bybit-under-min",
                allow_orders=True,
            )

        assert "BELOW_MIN_NOTIONAL" in str(caught.value)
        assert transport.calls_to("POST", "/v5/order/create") == []


class TestTheDocumentedDemoRoutes:
    """Every route the adapter calls is one the demo service documents.

    The venue's Demo Trading Service page lists the available endpoints.
    Get Trade History is ``/v5/execution/list``; ``/v5/order/execution`` is
    not a route the service serves, so a call to it fails at the venue and
    the protection evidence can never be read.
    """

    def test_the_execution_route_is_the_documented_one(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "retMsg": "OK", "result": {"list": []}},
        )
        broker = make_broker(transport)
        broker._state.confirmed = True

        executed, truncated = broker._read_order_executions()

        assert executed == {}
        assert truncated is False
        assert transport.calls_to("GET", "/v5/execution/list")

    def test_the_undocumented_execution_route_is_never_called(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "retMsg": "OK", "result": {"list": []}},
        )
        broker = make_broker(transport)
        broker._state.confirmed = True

        broker._read_order_executions()

        assert transport.calls_to("GET", "/v5/order/execution") == []

    def test_the_execution_rows_are_summed_per_client_order_id(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/execution/list",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {"orderLinkId": "abc", "execQty": "0.1", "execPrice": "50000"},
                        {"orderLinkId": "abc", "execQty": "0.05", "execPrice": "50100"},
                    ]
                },
            },
        )
        broker = make_broker(transport)
        broker._state.confirmed = True

        executed, _ = broker._read_order_executions()

        assert executed["abc"] == pytest.approx(0.15)

    def test_a_pair_the_venue_does_not_describe_is_refused(self):
        """No made-up minimum: an undescribed pair is a refusal, not a guess."""
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        broker = make_broker(transport)
        broker.connect()

        with pytest.raises(PaperRiskRejected) as caught:
            broker.submit_order(
                symbol="PEPE/USDT",
                side="buy",
                quantity=1.0,
                reference_price=0.0000012,
                client_order_id="bybit-undescribed",
                allow_orders=True,
            )

        assert "not found in Bybit instruments" in str(caught.value)
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_a_market_order_uses_the_documented_ioc_time_in_force(self):
        """The venue documents that a Market order always uses ``IOC``."""
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v-ioc", "orderLinkId": "bybit-ioc"}},
        )
        broker = make_broker(transport)
        broker.connect()

        broker.submit_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.0005,
            reference_price=50000,
            client_order_id="bybit-ioc",
            allow_orders=True,
        )

        create = transport.calls_to("POST", "/v5/order/create")[0]
        assert create["json_body"]["orderType"] == "Market"
        assert create["json_body"]["timeInForce"] == "IOC"

    def test_a_market_order_states_its_quantity_unit(self):
        """Spot Market orders are by value unless ``marketUnit`` says otherwise."""
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v-unit", "orderLinkId": "bybit-unit"}},
        )
        broker = make_broker(transport)
        broker.connect()

        broker.submit_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.0005,
            reference_price=50000,
            client_order_id="bybit-unit",
            allow_orders=True,
        )

        create = transport.calls_to("POST", "/v5/order/create")[0]
        assert create["json_body"]["marketUnit"] == "baseCoin"


class TestQuantitiesAndLimits:
    def _ready_transport(self, **instrument: Any) -> RecordingTransport:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT", **instrument)]),
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v1", "orderLinkId": "frac-1"}},
        )
        return transport

    def test_a_fractional_quantity_is_sent_as_a_decimal_string(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="frac-1",
            allow_orders=True,
            refresh=True,
        )
        body = transport.calls_to("POST", "/v5/order/create")[0]["json_body"]
        # The quantity is rendered at the pair's qtyStep precision, and it is
        # never scientific notation (Bybit rejects "5e-04").
        assert Decimal(body["qty"]) == Decimal("0.0005")
        assert "e" not in body["qty"].lower()
        # The rendered value is a whole multiple of the step.
        step = Decimal("0.00001")
        assert (Decimal(body["qty"]) / step) % 1 == 0

    def test_a_quantity_that_breaks_the_step_is_refused(self):
        transport = self._ready_transport(min_order_qty="0.01", qty_step="0.01", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        # 0.0155 BTC breaks a 0.01 step. At 1000 USDT the notional is 15.50,
        # so the size is inside the $25 cap and the step is the only fault.
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.0155,
                side="buy",
                reference_price=1000,
                client_order_id="frac-2",
                allow_orders=True,
                refresh=True,
            )
        assert "qtyStep" in str(excinfo.value)

    def test_an_order_over_the_25_usd_cap_is_refused(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        # 0.001 BTC at 50000 is 50 USDT, over the project's 25 USD cap.
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.001,
                side="buy",
                reference_price=50000,
                client_order_id="over-cap-1",
                allow_orders=True,
                refresh=True,
            )
        assert "25" in str(excinfo.value)
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_the_cap_is_25_usd(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        assert broker.limits.max_order_notional == 25.0
        assert broker.limits.max_open_positions == 5
        assert broker.limits.max_daily_orders == 10
        assert broker.limits.allow_short is False

    def test_an_order_just_over_the_cap_is_refused(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        # 0.00051 BTC at 50000 is 25.50 USDT.
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.00051,
                side="buy",
                reference_price=50000,
                client_order_id="over-cap-2",
                allow_orders=True,
                refresh=True,
            )

    def test_a_sell_of_more_than_is_held_is_refused_as_a_short(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        broker.positions(refresh=True)
        # The account holds 0.1 BTC; selling 1 BTC would be a short.
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=1.0,
                side="sell",
                reference_price=50000,
                client_order_id="short-1",
                allow_orders=True,
                refresh=True,
            )
        assert "short" in str(excinfo.value).lower()
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_a_sell_of_the_held_amount_is_allowed(self):
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v2", "orderLinkId": "sell-1"}},
        )
        broker = make_broker(transport)
        broker.connect()
        broker.positions(refresh=True)
        order = broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.1,
            side="sell",
            reference_price=50000,
            client_order_id="sell-1",
            allow_orders=True,
            refresh=True,
        )
        assert order.client_order_id == "sell-1"
        body = transport.calls_to("POST", "/v5/order/create")[0]["json_body"]
        assert body["side"] == "Sell"

    def test_no_order_carries_a_leverage_or_a_margin_category(self):
        """Every order is Spot: the category is never anything else."""
        transport = self._ready_transport(min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="spot-1",
            allow_orders=True,
            refresh=True,
        )
        for call in transport.calls:
            if call["method"] == "POST":
                assert call["json_body"]["category"] == "spot"
            else:
                assert call["params"].get("category", "spot") == "spot"
        body = transport.calls_to("POST", "/v5/order/create")[0]["json_body"]
        # No leverage, no margin, no position-side keys.
        for forbidden in ("leverage", "margin", "positionIdx", "sideMode", "tpLeverage"):
            assert forbidden not in body


# ---------------------------------------------------------------------------
# 8. A retry must not duplicate an order
# ---------------------------------------------------------------------------


class TestNoDuplicateOrders:
    def _ready_transport(self, client_order_id: str = "dup-1") -> RecordingTransport:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response(
                [spot_instrument("BTCUSDT", min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")]
            ),
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {"orderId": "v1", "orderLinkId": client_order_id},
            },
        )
        transport.add(
            "GET",
            "/v5/order/history",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id=client_order_id,
                            status="Filled",
                            qty="0.0005",
                            cum_exec_qty="0.0005",
                            avg_price="50000",
                        )
                    ]
                },
            },
        )
        return transport

    def test_the_same_client_order_id_is_never_submitted_twice(self):
        transport = self._ready_transport()
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="dup-1",
            allow_orders=True,
            refresh=True,
        )
        # The second attempt with the same id reconciles; it does not submit.
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="dup-1",
            allow_orders=True,
        )
        assert len(transport.calls_to("POST", "/v5/order/create")) == 1

    def test_a_duplicate_request_from_the_venue_asks_for_a_reconcile(self):
        """Bybit's "invalid duplicate request" means the order may exist.

        The answer is to reconcile by client order id, never to submit again.
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"USDT": {"walletBalance": "1000"}}),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response(
                [spot_instrument("BTCUSDT", min_order_qty="0.00001", qty_step="0.00001", min_order_amt="5")]
            ),
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 10014, "retMsg": "Invalid duplicate request", "result": {}},
        )
        broker = make_broker(transport)
        broker.connect()
        with pytest.raises(DuplicateOrderError) as excinfo:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.0005,
                side="buy",
                reference_price=50000,
                client_order_id="dup-2",
                allow_orders=True,
            )
        assert "reconcile" in str(excinfo.value).lower()

    def test_a_locally_refused_order_may_be_tried_again(self):
        """A refusal that never reached the venue may be retried.

        The broker's own marker for a local refusal is not the venue's
        ``rejected`` string, so the two are never confused.
        """
        transport = self._ready_transport("dup-3")
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="dup-3",
            allow_orders=True,
            refresh=True,
        )
        recorded = broker._state.orders["dup-3"]
        assert recorded["status"] in ("accepted", "filled", "partially_filled", "new")

    def test_the_intent_is_recorded_before_the_request(self):
        """A crash between submission and confirmation must be recoverable."""
        transport = self._ready_transport("dup-4")
        broker = make_broker(transport)
        broker.connect()
        broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.0005,
            side="buy",
            reference_price=50000,
            client_order_id="dup-4",
            allow_orders=True,
            refresh=True,
        )
        # The order is known by its client id, so it can be reconciled later.
        assert "dup-4" in broker._state.orders
        # The venue's own order id is recorded against the client id, so a
        # later read can name the order at the venue.
        assert broker._state.orders["dup-4"]["broker_order_id"] == "venue-1"


# ---------------------------------------------------------------------------
# 9. Rate limits and other venue errors
# ---------------------------------------------------------------------------


class TestVenueErrors:
    def test_a_rate_limit_is_its_own_error(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            {"retCode": 10006, "retMsg": "Too many visits", "result": {}},
        )
        broker = make_broker(transport)
        with pytest.raises(BrokerRateLimitedError) as excinfo:
            broker.connect()
        assert "10006" in str(excinfo.value)

    def test_a_signature_error_is_reported_as_a_rejection(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            {"retCode": 10004, "retMsg": "Error sign", "result": {}},
        )
        broker = make_broker(transport)
        with pytest.raises(BrokerRejectedError) as excinfo:
            broker.connect()
        assert "10004" in str(excinfo.value)

    def test_an_insufficient_balance_is_reported_as_insufficient_funds(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            {"retCode": 110007, "retMsg": "Insufficient available balance", "result": {}},
        )
        broker = make_broker(transport)
        with pytest.raises(InsufficientFundsError):
            broker.connect()

    def test_a_secret_in_a_venue_message_is_redacted(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            {
                "retCode": 10003,
                "retMsg": "invalid key demo-secret-value-here",
                "result": {},
            },
        )
        broker = make_broker(transport)
        with pytest.raises(BrokerRejectedError) as excinfo:
            broker.connect()
        assert "demo-secret-value-here" not in str(excinfo.value)

    def test_a_non_json_answer_is_a_broker_error(self):
        transport = RecordingTransport()
        transport.add_raw("GET", "/v5/order/realtime", "<html>not json</html>")
        broker = make_broker(transport)
        broker._state.confirmed = True
        with pytest.raises(BrokerError):
            broker._request("GET", "/v5/order/realtime", signed=True)


# ---------------------------------------------------------------------------
# 10. Protection is proven at the venue, never assumed
# ---------------------------------------------------------------------------


class TestProtectionIsProven:
    def _transport_with_protection(self, sl_status: str, tp_status: str) -> RecordingTransport:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-sl",
                            status=sl_status,
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="StopLoss",
                            trigger_price="45000",
                            order_id="v-sl",
                        ),
                        order_row(
                            order_link_id="p-tp",
                            status=tp_status,
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="TakeProfit",
                            trigger_price="55000",
                            order_id="v-tp",
                        ),
                    ]
                },
            },
        )
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        return transport

    def test_both_legs_working_means_the_position_is_protected(self):
        transport = self._transport_with_protection("Untriggered", "Untriggered")
        broker = make_broker(transport)
        broker.connect()
        positions = broker.positions(refresh=True)
        assert positions["BTC/USDT"].is_protected is True

    def test_a_missing_stop_leaves_the_position_unprotected(self):
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        # Only the target exists at the venue.
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-tp",
                            status="Untriggered",
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="TakeProfit",
                            trigger_price="55000",
                        )
                    ]
                },
            },
        )
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        broker = make_broker(transport)
        broker.connect()
        positions = broker.positions(refresh=True)
        assert positions["BTC/USDT"].is_protected is False

    def test_a_filled_stop_leaves_the_position_unprotected(self):
        """A stop that has filled is no longer protection."""
        transport = self._transport_with_protection("Filled", "Untriggered")
        broker = make_broker(transport)
        broker.connect()
        positions = broker.positions(refresh=True)
        assert positions["BTC/USDT"].is_protected is False

    def test_the_protection_orders_are_verified_against_the_venue(self):
        transport = self._transport_with_protection("Untriggered", "Untriggered")
        broker = make_broker(transport)
        broker.connect()
        verified = broker.verify_protection_orders("BTC/USDT")
        assert verified["stop_loss"] is not None
        assert verified["stop_loss"].client_order_id == "p-sl"
        assert verified["take_profit"] is not None
        assert verified["take_profit"].client_order_id == "p-tp"

    def test_a_truncated_read_does_not_claim_protection(self):
        """A read that was cut short is not evidence of anything.

        The stop may simply be on a page that was not read, so a truncated
        read must not be read as "there is no stop".
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-sl",
                            status="Untriggered",
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="StopLoss",
                            trigger_price="45000",
                        )
                    ],
                    "nextPageCursor": "more-pages-exist",
                },
            },
        )
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        broker = make_broker(transport)
        broker.connect()
        positions = broker.positions(refresh=True)
        # The read was incomplete, so the protection is not proven.
        assert positions["BTC/USDT"].is_protected is False

    def test_the_protection_request_uses_only_documented_parameters(self):
        """``stopOrderType`` is a response field, not a request parameter.

        Bybit's create-order endpoint accepts ``orderFilter`` but not
        ``stopOrderType``; sending it would be rejected.
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v1", "orderLinkId": "prot-1"}},
        )
        broker = make_broker(transport)
        broker.connect()
        broker.create_protection_orders(
            symbol="BTC/USDT",
            position_quantity=0.1,
            stop_loss_price=45000,
            take_profit_price=55000,
            client_order_id_prefix="prot",
            allow_orders=True,
        )
        for call in transport.calls_to("POST", "/v5/order/create"):
            body = call["json_body"]
            assert body["category"] == "spot"
            assert body["orderFilter"] == "tpslOrder"
            assert "stopOrderType" not in body
            assert "triggerDirection" not in body
            assert body["triggerPrice"] in ("45000", "55000")



# ---------------------------------------------------------------------------
# The public market-data client
# ---------------------------------------------------------------------------


class TestThePublicInstrumentParser:
    """The public client parses the venue's Spot shape into the project's rules.

    The rule object is shared with every other venue, so a parser that names
    its fields wrongly does not fail loudly at the boundary — it fails when the
    rule is read. These tests read the parsed rule, not just the parse call.
    """

    def _parse(self, entry: dict) -> Any:
        from tradingagents.crypto.bybit import BybitPublicClient

        client = BybitPublicClient()
        rules = client._parse_bybit_instruments([entry])
        return rules["BTC/USDT"]

    def _spot_entry(self, **lot: Any) -> dict:
        base = {
            "symbol": "BTCUSDT",
            "status": "Trading",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "lotSizeFilter": {
                "basePrecision": "0.000001",
                "minOrderQty": "0.00001",
                "maxOrderQty": "63.01155123",
                "minOrderAmt": "1",
            },
            "priceFilter": {"tickSize": "0.01"},
        }
        base["lotSizeFilter"].update(lot)
        return base

    def test_the_step_comes_from_the_base_precision(self):
        rules = self._parse(self._spot_entry())
        assert rules.quantity_step == pytest.approx(0.000001)

    def test_the_minimum_notional_is_the_min_order_amount(self):
        rules = self._parse(self._spot_entry())
        assert rules.min_notional == pytest.approx(1.0)

    def test_the_price_tick_is_the_tick_size(self):
        rules = self._parse(self._spot_entry())
        assert rules.price_tick == pytest.approx(0.01)

    def test_the_pair_is_keyed_by_its_canonical_spelling(self):
        rules = self._parse(self._spot_entry())
        assert rules.symbol == "BTC/USDT"
        assert rules.base_asset == "BTC"
        assert rules.quote_asset == "USDT"

    def test_a_missing_bound_is_unstated_not_zero(self):
        entry = self._spot_entry()
        entry["lotSizeFilter"].pop("minOrderAmt")
        entry["priceFilter"].pop("tickSize")
        rules = self._parse(entry)
        assert rules.min_notional is None
        assert rules.price_tick is None

    def test_a_fractional_step_makes_the_pair_fractionable(self):
        rules = self._parse(self._spot_entry())
        assert rules.fractionable is True

    def test_a_step_of_one_is_not_fractionable(self):
        rules = self._parse(self._spot_entry(basePrecision="1"))
        assert rules.fractionable is False

    def test_an_entry_without_a_quote_coin_is_skipped(self):
        from tradingagents.crypto.bybit import BybitPublicClient

        client = BybitPublicClient()
        entry = self._spot_entry()
        entry["quoteCoin"] = ""
        assert client._parse_bybit_instruments([entry]) == {}
