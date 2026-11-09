"""The crypto order's time-in-force, and the quantity it carries.

Alpaca's ``TimeInForce`` schema splits the vocabulary by security type:

- **Equity** trading: ``day``, ``gtc``, ``opg``, ``cls``, ``ioc``, ``fok``.
- **Options** trading: ``day``, ``gtc``.
- **Crypto** trading: ``gtc``, ``ioc``.

The broker used to send ``day`` for every order, so a crypto order was answered
with HTTP 422 ``invalid crypto time_in_force`` and the order never reached the
book. The fix is scoped: a crypto order carries ``gtc`` (or ``ioc`` when the
caller asks for it), and an equity order keeps the time-in-force it always sent.

The second defect this pins is the *quantity*. ``str()`` renders a small
quantity in exponent form (``1e-06``), which Alpaca does not accept as a
``qty``; a crypto order's quantity is routinely small, so the notation is
written out in full.

Both are pinned at the wire level — the tests assert the body that would be
POSTed to Alpaca — and at the unit level. Every test drives a mocked transport,
so the suite is hermetic: no network, no credential, no order. The mock is a
fake *transport*, not a fake broker, so the adapter's own endpoint validation,
credential handling, request shaping and error classification all run for real.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.paper_experiment.broker import (
    CRYPTO_TIME_IN_FORCE,
    CRYPTO_TIME_IN_FORCE_VALUES,
    EQUITY_TIME_IN_FORCE,
    AlpacaPaperBroker,
    BrokerRejectedError,
    PaperRiskRejected,
    PaperTradingConfig,
    PaperTradingWorker,
    TradingDecision,
)

EQUITY_SYMBOL = "AAPL"
CRYPTO_SYMBOL = "BTC/USD"


# ---------------------------------------------------------------------------
# A transport that records what would have been sent
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code: int, body) -> None:
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


class _RecordingTransport:
    """Records every request and answers the account/positions reads.

    The order response is configurable, so a rejection can be exercised without
    changing the transport's shape.
    """

    def __init__(self, *, order_response: _Resp | None = None) -> None:
        self.order_response = order_response or _Resp(200, _order_payload())
        self.calls: list[dict] = []
        self.orders: list[dict] = []

    def __call__(self, *, method, url, headers, params, json_body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "params": dict(params or {}),
                "json": json_body,
                "timeout": timeout,
            }
        )
        if method == "GET" and url.endswith("/v2/account"):
            return _Resp(200, _ACCOUNT)
        if method == "GET" and url.endswith("/v2/positions"):
            return _Resp(200, [])
        if method == "GET" and url.endswith("/v2/orders"):
            return _Resp(200, [])
        if method == "POST" and url.endswith("/v2/orders"):
            self.orders.append(json_body or {})
            return self.order_response
        raise AssertionError(f"unexpected request {method} {url}")


_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "25000.00",
    "equity": "25000.00",
    "buying_power": "50000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
}


def _order_payload(**over) -> dict:
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": CRYPTO_SYMBOL,
        "side": "buy",
        "qty": "0.0005",
        "filled_qty": "0.0005",
        "filled_avg_price": "50000.0",
        "status": "filled",
        "type": "market",
        "time_in_force": CRYPTO_TIME_IN_FORCE,
        "submitted_at": "2026-01-01T12:00:00Z",
    }
    payload.update(over)
    return payload


def _broker(transport, **over) -> AlpacaPaperBroker:
    """A confirmed paper broker wired to ``transport``."""
    broker = AlpacaPaperBroker(
        api_key="test-paper-key",
        api_secret="test-paper-secret",
        transport=transport,
        **over,
    )
    broker.connect()
    return broker


# ---------------------------------------------------------------------------
# 1. The vocabulary itself
# ---------------------------------------------------------------------------


class TestTheCryptoTimeInForceVocabulary:
    def test_it_is_exactly_the_two_values_alpaca_accepts_for_crypto(self):
        """Alpaca's schema: *"Crypto trading: gtc, ioc."*"""
        assert CRYPTO_TIME_IN_FORCE_VALUES == ("gtc", "ioc")

    def test_the_crypto_default_is_gtc(self):
        assert CRYPTO_TIME_IN_FORCE == "gtc"

    def test_the_equity_default_is_unchanged(self):
        """The equity path must not lose 'day' — Alpaca still accepts it there."""
        assert EQUITY_TIME_IN_FORCE == "day"


# ---------------------------------------------------------------------------
# 2. The wire body
# ---------------------------------------------------------------------------


class TestTheCryptoOrderCarriesGtc:
    def test_a_crypto_order_is_sent_with_gtc(self):
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=0.0005,
            side="buy",
            reference_price=50000.0,
            client_order_id="c1",
            allow_orders=True,
            crypto=True,
        )

        assert transport.orders, "no order was submitted"
        assert transport.orders[0]["time_in_force"] == "gtc"

    def test_the_equity_order_still_carries_day(self):
        """The fix is scoped to crypto; the equity body is byte-for-byte as before."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        # 2 shares at $10 = $20, inside the $25 notional cap.
        broker.submit_order(
            symbol=EQUITY_SYMBOL,
            quantity=2,
            side="buy",
            reference_price=10.0,
            client_order_id="c2",
            allow_orders=True,
        )

        assert transport.orders[0]["time_in_force"] == "day"

    def test_a_crypto_order_may_ask_for_ioc(self):
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=0.0005,
            side="buy",
            reference_price=50000.0,
            client_order_id="c3",
            time_in_force="ioc",
            allow_orders=True,
            crypto=True,
        )

        assert transport.orders[0]["time_in_force"] == "ioc"

    @pytest.mark.parametrize("bad", ["day", "opg", "cls", "fok"])
    def test_an_equity_only_value_is_refused_before_the_request(self, bad):
        """The unsupported value never reaches Alpaca, so it is never a 422.

        The refusal is local and definitive: the broker is never asked to
        reject an order this adapter already knows is invalid.
        """
        transport = _RecordingTransport()
        broker = _broker(transport)

        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol=CRYPTO_SYMBOL,
                quantity=0.0005,
                side="buy",
                reference_price=50000.0,
                client_order_id="c4",
                time_in_force=bad,
                allow_orders=True,
                crypto=True,
            )

        assert not transport.orders, "the invalid order was sent anyway"
        assert not any(c["method"] == "POST" for c in transport.calls)

    def test_an_equity_order_may_still_use_the_equity_vocabulary(self):
        """'day' is refused for crypto and accepted for equity."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=EQUITY_SYMBOL,
            quantity=2,
            side="buy",
            reference_price=10.0,
            client_order_id="c5",
            time_in_force="day",
            allow_orders=True,
        )

        assert transport.orders[0]["time_in_force"] == "day"

    def test_the_client_order_id_and_symbol_are_untouched(self):
        """The fix changes the time-in-force only; nothing else moves."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=0.0005,
            side="buy",
            reference_price=50000.0,
            client_order_id="c6",
            allow_orders=True,
            crypto=True,
        )

        body = transport.orders[0]
        assert body["symbol"] == CRYPTO_SYMBOL
        assert body["client_order_id"] == "c6"
        assert body["side"] == "buy"
        assert body["type"] == "market"


class TestTheWorkerPassesTheAssetClassThrough:
    """The worker must tell the broker which asset class it is trading.

    The broker cannot guess: ``BTC/USD`` and ``AAPL`` are both strings, and a
    symbol-shaped guess would be a guess. The asset class travels explicitly,
    from the worker's config to the order body.
    """

    def test_a_crypto_worker_produces_a_crypto_order(self, tmp_path):
        transport = _RecordingTransport()
        broker = _broker(transport)

        class _Source:
            def next_decision(self):
                return TradingDecision(
                    decision_id="d1",
                    symbol=CRYPTO_SYMBOL,
                    side="buy",
                    notional=25.0,
                    # Levels the strategy proposed, so the mandatory-protection
                    # gate admits the trade. This test is about the asset class
                    # and the time-in-force, not about protection.
                    stop_loss=45000.0,
                    take_profit=60000.0,
                )

        class _Market:
            provider_name = "static_test"
            kind = "current"

            def available(self):
                return True

            def price(self, symbol):
                return 50000.0

        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="crypto-tjr",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source="crypto",
                crypto=True,
            ),
            decision_source=_Source(),
            market_data=_Market(),
            broker_factory=lambda: broker,
            root=tmp_path,
        )
        worker.start()
        try:
            worker.run_once()
        finally:
            worker.stop()

        assert transport.orders, "no order was submitted"
        assert transport.orders[0]["time_in_force"] == "gtc"

    def test_an_equity_worker_produces_an_equity_order(self, tmp_path):
        transport = _RecordingTransport()
        broker = _broker(transport)

        class _Source:
            def next_decision(self):
                return TradingDecision(
                    decision_id="d2",
                    symbol=EQUITY_SYMBOL,
                    side="buy",
                    notional=25.0,
                    # Levels the strategy proposed, so the mandatory-protection
                    # gate admits the trade. This test is about the asset class
                    # and the time-in-force, not about protection.
                    stop_loss=9.0,
                    take_profit=12.0,
                )

        class _Market:
            provider_name = "static_test"
            kind = "current"

            def available(self):
                return True

            def price(self, symbol):
                return 10.0

        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="equity",
                order_notional=25.0,
                whole_shares=True,
                state_path=tmp_path / "worker_state.json",
                decision_source="history",
            ),
            decision_source=_Source(),
            market_data=_Market(),
            broker_factory=lambda: broker,
            root=tmp_path,
        )
        worker.start()
        try:
            worker.run_once()
        finally:
            worker.stop()

        assert transport.orders, "no order was submitted"
        assert transport.orders[0]["time_in_force"] == "day"


# ---------------------------------------------------------------------------
# 3. The quantity
# ---------------------------------------------------------------------------


class TestTheQuantityIsNeverInScientificNotation:
    def test_a_small_quantity_is_written_out_in_full(self):
        """``str(1e-06)`` is '1e-06', which Alpaca does not accept as a qty."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=1e-06,
            side="buy",
            reference_price=50000.0,
            client_order_id="q1",
            allow_orders=True,
            crypto=True,
        )

        qty = transport.orders[0]["qty"]
        assert qty == "0.000001"
        assert "e" not in qty.lower()

    def test_an_ordinary_quantity_is_unchanged(self):
        """The wire format for a normal quantity is exactly what it always was."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=0.0005,
            side="buy",
            reference_price=50000.0,
            client_order_id="q2",
            allow_orders=True,
            crypto=True,
        )

        assert transport.orders[0]["qty"] == str(0.0005)

    def test_the_quantity_keeps_its_value(self):
        """Only the notation changes; the number does not."""
        transport = _RecordingTransport()
        broker = _broker(transport)

        broker.submit_order(
            symbol=CRYPTO_SYMBOL,
            quantity=1e-06,
            side="buy",
            reference_price=50000.0,
            client_order_id="q3",
            allow_orders=True,
            crypto=True,
        )

        assert float(transport.orders[0]["qty"]) == 1e-06


# ---------------------------------------------------------------------------
# 4. The broker's own answer is still classified correctly
# ---------------------------------------------------------------------------


class TestTheBrokerAnswerIsClassified:
    def test_a_422_is_a_rejection_and_not_a_fill(self):
        """A refusal is never dressed up as a success."""
        transport = _RecordingTransport(
            order_response=_Resp(422, {"code": 42210000, "message": "invalid crypto time_in_force"})
        )
        broker = _broker(transport)

        with pytest.raises(BrokerRejectedError) as caught:
            broker.submit_order(
                symbol=CRYPTO_SYMBOL,
                quantity=0.0005,
                side="buy",
                reference_price=50000.0,
                client_order_id="r1",
                allow_orders=True,
                crypto=True,
            )

        assert "422" in str(caught.value)

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_refused_credential_is_a_rejection(self, status):
        transport = _RecordingTransport(
            order_response=_Resp(status, {"message": "access key verification failed"})
        )
        broker = _broker(transport)

        with pytest.raises(BrokerRejectedError):
            broker.submit_order(
                symbol=CRYPTO_SYMBOL,
                quantity=0.0005,
                side="buy",
                reference_price=50000.0,
                client_order_id="r2",
                allow_orders=True,
                crypto=True,
            )
