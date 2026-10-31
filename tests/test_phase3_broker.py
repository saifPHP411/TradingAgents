"""Phase 3: the Alpaca **paper** broker adapter, and its safety invariants.

Every test drives a **mocked HTTP transport**; no test reaches the network (the
suite's autouse ``_no_network`` fixture would refuse it) and no test needs a real
key. The mock is a fake *transport*, not a fake broker: the adapter's own endpoint
validation, credential handling, order parsing, risk checks, idempotency and
reconciliation run for real.

The tests are written to catch a premise error, not to confirm the code. The
headline claims — that a live endpoint is refused, that a missing paper credential
fails closed, that an order is never placed by construction alone, that a lost
submission is reconciled by client order id rather than re-submitted, that a fill
is applied at most once across a restart, and that the broker ledger never touches
the internal simulator ledger — are each pinned by a test that would fail if the
claim were false.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.paper_experiment.broker import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
    BrokerNotConfiguredError,
    BrokerRejectedError,
    BrokerStateError,
    BrokerStateStore,
    LiveEndpointRefusedError,
    PaperRiskLimits,
    PaperRiskRejected,
    assert_paper_endpoint,
    broker_status,
)

_ACCOUNT = {
    "id": "paper-acct-123",
    "status": "ACTIVE",
    "cash": "25000.00",
    "equity": "25000.00",
    "buying_power": "50000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
}


def _order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": "SPY",
        "side": "buy",
        "qty": "1",
        "filled_qty": "0",
        "filled_avg_price": None,
        "status": "accepted",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _transport(routes, *, calls=None, fail_on=None):
    """A transport that answers from ``routes`` keyed by (method, path-suffix)."""

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "headers": dict(headers),
                          "params": dict(params or {}), "json": json_body})
        if fail_on is not None and fail_on(method, url, json_body):
            raise TimeoutError("simulated timeout")
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _broker(**kwargs):
    kwargs.setdefault("api_key", "test-paper-key")
    kwargs.setdefault("api_secret", "test-paper-secret")
    kwargs.setdefault("transport", _transport({("GET", "/v2/account"): _Resp(200, _ACCOUNT)}))
    return AlpacaPaperBroker(**kwargs)


# ---------------------------------------------------------------------------
# 1. Paper-only endpoint enforcement
# ---------------------------------------------------------------------------


class TestPaperEndpointEnforcement:
    @pytest.mark.parametrize(
        "live",
        [
            "https://api.alpaca.markets",
            "https://alpaca.markets",
            "https://broker-api.alpaca.markets",
            "https://API.ALPACA.MARKETS",
        ],
    )
    def test_a_live_endpoint_is_refused(self, live):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(live)

    def test_the_approved_paper_endpoint_is_accepted(self):
        assert assert_paper_endpoint(PAPER_BASE_URL) == PAPER_BASE_URL

    def test_an_unrelated_host_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://evil.example.com")

    def test_a_live_endpoint_is_refused_at_construction(self):
        with pytest.raises(LiveEndpointRefusedError):
            AlpacaPaperBroker(
                api_key="k", api_secret="s", base_url="https://api.alpaca.markets"
            )

    def test_construction_places_no_request(self):
        calls = []
        AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport({}, calls=calls)
        )
        assert calls == []  # constructing a broker must not talk to the broker

    def test_a_live_endpoint_is_refused_before_a_credential_is_read(self):
        # Even with no credentials set, the endpoint check comes first and is the
        # error raised, so a live URL can never get as far as an auth attempt.
        with pytest.raises(LiveEndpointRefusedError):
            AlpacaPaperBroker(api_key="", api_secret="", base_url="https://api.alpaca.markets")


# ---------------------------------------------------------------------------
# 2. Credentials
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_missing_credentials_fail_closed(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        with pytest.raises(BrokerNotConfiguredError):
            AlpacaPaperBroker()

    def test_a_whitespace_only_credential_is_refused(self):
        with pytest.raises(BrokerNotConfiguredError):
            AlpacaPaperBroker(api_key="   ", api_secret="   ")

    def test_a_live_credential_variable_is_not_used_as_a_fallback(self, monkeypatch):
        # The live variable is set, the paper one is not: the adapter must refuse
        # rather than silently adopt a live credential.
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        monkeypatch.setenv("APCA_API_KEY_ID", "live-key-value")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "live-secret-value")
        with pytest.raises(BrokerNotConfiguredError) as exc:
            AlpacaPaperBroker()
        assert "APCA_API_KEY_ID" in str(exc.value)  # named, but never its value

    def test_credentials_are_read_from_the_paper_env_vars(self, monkeypatch):
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "env-key")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "env-secret")
        calls = []
        broker = AlpacaPaperBroker(
            transport=_transport({("GET", "/v2/account"): _Resp(200, _ACCOUNT)}, calls=calls)
        )
        broker.connect()
        assert calls[0]["headers"]["APCA-API-KEY-ID"] == "env-key"
        assert calls[0]["headers"]["APCA-API-SECRET-KEY"] == "env-secret"

    def test_an_auth_failure_is_reported_without_leaking_the_secret(self):
        broker = AlpacaPaperBroker(
            api_key="supersecretpaperkey",
            api_secret="supersecretpapersecret",
            transport=_transport(
                {("GET", "/v2/account"): _Resp(401, "forbidden: bad key supersecretpaperkey")}
            ),
        )
        with pytest.raises(BrokerRejectedError) as exc:
            broker.connect()
        message = str(exc.value)
        assert "supersecretpaperkey" not in message
        assert "***" in message


# ---------------------------------------------------------------------------
# 3. Read-only account check
# ---------------------------------------------------------------------------


class TestAccountCheck:
    def test_connect_reads_the_account_and_places_no_order(self):
        calls = []
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({("GET", "/v2/account"): _Resp(200, _ACCOUNT)}, calls=calls),
        )
        account = broker.connect()
        assert account.account_id == "paper-acct-123"
        assert account.cash == 25000.0
        assert account.buying_power == 50000.0
        assert account.trading_blocked is False
        assert all(c["method"] == "GET" for c in calls)  # read-only

    def test_connect_records_the_confirmed_account(self, tmp_path):
        store = BrokerStateStore(tmp_path / "broker.json")
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({("GET", "/v2/account"): _Resp(200, _ACCOUNT)}),
            state_path=tmp_path / "broker.json",
        )
        broker.connect()
        restored = store.load()
        assert restored.confirmed is True
        assert restored.account_id == "paper-acct-123"
        assert restored.cash == 25000.0

    def test_a_non_object_account_response_is_refused(self):
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({("GET", "/v2/account"): _Resp(200, {"nope": 1})}),
        )
        with pytest.raises(BrokerError):
            broker.connect()

    def test_an_unconfirmed_account_cannot_trade(self):
        broker = _broker()  # connect() never called
        with pytest.raises(BrokerError):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )

    def test_a_blocked_account_cannot_trade(self):
        blocked = dict(_ACCOUNT, trading_blocked=True)
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({("GET", "/v2/account"): _Resp(200, blocked)}),
        )
        broker.connect()
        with pytest.raises(BrokerError):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )


# ---------------------------------------------------------------------------
# 4. Order lifecycle
# ---------------------------------------------------------------------------


class TestOrderLifecycle:
    def _ready(self, order_payload, *, calls=None, state_path=None):
        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("POST", "/v2/orders"): _Resp(200, order_payload),
        }
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport(routes, calls=calls),
            limits=PaperRiskLimits(max_order_notional=10_000),
            state_path=state_path,
        )
        broker.connect()
        return broker

    def test_an_accepted_order_is_parsed(self):
        broker = self._ready(_order(status="accepted"))
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "accepted"
        assert order.broker_order_id == "broker-order-1"
        assert order.remaining_quantity == 1

    def test_a_filled_order_reports_its_fill(self):
        broker = self._ready(
            _order(status="filled", filled_qty="1", filled_avg_price="100.5")
        )
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "filled"
        assert order.filled_quantity == 1
        assert order.filled_avg_price == 100.5

    def test_a_partial_fill_reports_the_remainder(self):
        broker = self._ready(
            _order(status="partially_filled", filled_qty="0.4", filled_avg_price="100")
        )
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "partially_filled"
        assert order.filled_quantity == 0.4
        assert order.remaining_quantity == pytest.approx(0.6)

    @pytest.mark.parametrize(
        "alpaca,expected",
        [
            ("new", "accepted"),
            ("accepted", "accepted"),
            ("partially_filled", "partially_filled"),
            ("filled", "filled"),
            ("canceled", "canceled"),
            ("expired", "expired"),
            ("rejected", "rejected"),
            ("suspended", "unknown"),
            ("something_new", "unknown"),
        ],
    )
    def test_broker_statuses_map_to_the_closed_vocabulary(self, alpaca, expected):
        broker = self._ready(_order(status=alpaca))
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == expected

    def test_a_rejected_order_is_reported(self):
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({
                ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
                ("POST", "/v2/orders"): _Resp(422, {"message": "insufficient buying power"}),
            }),
            limits=PaperRiskLimits(max_order_notional=10_000),
        )
        broker.connect()
        with pytest.raises(BrokerRejectedError):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )

    def test_a_cancel_then_read_reflects_the_cancelled_state(self):
        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("DELETE", "/v2/orders/broker-order-1"): _Resp(204, ""),
            ("GET", "/v2/orders/broker-order-1"): _Resp(200, _order(status="canceled")),
        }
        broker = AlpacaPaperBroker(api_key="k", api_secret="s", transport=_transport(routes))
        broker.connect()
        order = broker.cancel_order("broker-order-1")
        assert order.status == "canceled"


# ---------------------------------------------------------------------------
# 5. Duplicate submission prevention
# ---------------------------------------------------------------------------


class TestDuplicatePrevention:
    def test_the_same_client_order_id_is_not_submitted_twice(self):
        calls = []
        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("POST", "/v2/orders"): _Resp(200, _order(status="accepted")),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(200, _order(status="accepted")),
        }
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes, calls=calls),
            limits=PaperRiskLimits(max_order_notional=10_000),
        )
        broker.connect()
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1  # the second call reconciled, it did not re-post

    def test_a_restart_does_not_re_submit_a_known_order(self, tmp_path):
        state = tmp_path / "broker.json"
        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("POST", "/v2/orders"): _Resp(200, _order(status="accepted")),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(200, _order(status="accepted")),
        }
        first = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes),
            limits=PaperRiskLimits(max_order_notional=10_000), state_path=state,
        )
        first.connect()
        first.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        # A fresh broker from the same state must not re-submit.
        calls = []
        second = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes, calls=calls),
            limits=PaperRiskLimits(max_order_notional=10_000), state_path=state,
        )
        second.connect()
        second.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert [c for c in calls if c["method"] == "POST"] == []


# ---------------------------------------------------------------------------
# 6. Timeout then reconciliation
# ---------------------------------------------------------------------------


class TestTimeoutReconciliation:
    def test_a_timeout_reconciles_rather_than_re_submitting(self):
        calls = []

        def fail_post(method, url, body):
            return method == "POST"  # every POST times out

        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(
                200, _order(status="filled", filled_qty="1", filled_avg_price="100")
            ),
        }
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport(routes, calls=calls, fail_on=fail_post),
            limits=PaperRiskLimits(max_order_notional=10_000),
        )
        broker.connect()
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "filled"  # reconciled from the broker, not guessed
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1  # exactly one attempt, never a blind retry

    def test_an_unreconcilable_timeout_fails_safely(self):
        def fail_post(method, url, body):
            return method == "POST"

        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(404, {"message": "not found"}),
        }
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes, fail_on=fail_post),
            limits=PaperRiskLimits(max_order_notional=10_000),
        )
        broker.connect()
        with pytest.raises(BrokerError):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )

    def test_intent_is_persisted_before_the_request(self, tmp_path):
        state = tmp_path / "broker.json"
        checked = {"done": False}

        def fail_post(method, url, body):
            if method != "POST":
                return False
            # Inspect the state file *during* the POST: intent must already be there.
            assert state.exists()
            saved = json.loads(state.read_text())
            assert "c1" in saved["orders"]
            checked["done"] = True
            return True

        routes = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(
                200, _order(status="accepted")
            ),
        }
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes, fail_on=fail_post),
            limits=PaperRiskLimits(max_order_notional=10_000), state_path=state,
        )
        broker.connect()
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert checked["done"] is True

    def test_restart_after_acceptance_before_local_confirmation(self, tmp_path):
        state = tmp_path / "broker.json"
        # First attempt: POST times out after the broker accepted it.
        routes1 = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(200, _order(status="accepted")),
        }
        first = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport(routes1, fail_on=lambda m, u, b: m == "POST"),
            limits=PaperRiskLimits(max_order_notional=10_000), state_path=state,
        )
        first.connect()
        first.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        # Restart: the intent is on disk, so the order is known and not re-posted.
        calls = []
        routes2 = {
            ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
            ("GET", "/v2/orders:by_client_order_id"): _Resp(
                200, _order(status="filled", filled_qty="1", filled_avg_price="100")
            ),
        }
        second = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport(routes2, calls=calls),
            limits=PaperRiskLimits(max_order_notional=10_000), state_path=state,
        )
        second.connect()
        order = second.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "filled"
        assert [c for c in calls if c["method"] == "POST"] == []


# ---------------------------------------------------------------------------
# 7. Risk limits
# ---------------------------------------------------------------------------


class TestRiskLimits:
    def _ready(self, limits):
        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s",
            transport=_transport({
                ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
                ("POST", "/v2/orders"): _Resp(200, _order()),
            }),
            limits=limits,
        )
        broker.connect()
        return broker

    def test_order_submission_requires_an_explicit_allow(self):
        broker = self._ready(PaperRiskLimits(max_order_notional=10_000))
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1",  # allow_orders defaults False
            )

    def test_the_notional_cap_is_enforced(self):
        broker = self._ready(PaperRiskLimits(max_order_notional=50))
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )

    def test_an_allowlist_is_enforced(self):
        broker = self._ready(
            PaperRiskLimits(max_order_notional=10_000, allowed_symbols=("AAPL",))
        )
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c1", allow_orders=True,
            )

    def test_the_daily_order_limit_is_enforced(self):
        broker = self._ready(
            PaperRiskLimits(max_order_notional=10_000, max_daily_orders=1)
        )
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=100,
            client_order_id="c1", allow_orders=True,
        )
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="c2", allow_orders=True,
            )

    def test_a_non_positive_reference_price_is_refused(self):
        broker = self._ready(PaperRiskLimits(max_order_notional=10_000))
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=0,
                client_order_id="c1", allow_orders=True,
            )

    def test_a_non_finite_reference_price_is_refused(self):
        broker = self._ready(PaperRiskLimits(max_order_notional=10_000))
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=float("nan"),
                client_order_id="c1", allow_orders=True,
            )

    def test_a_limit_must_be_positive(self):
        with pytest.raises(PaperRiskRejected):
            PaperRiskLimits(max_order_notional=0)

    def test_an_unstable_client_order_id_is_refused(self):
        broker = self._ready(PaperRiskLimits(max_order_notional=10_000))
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100,
                client_order_id="has spaces!", allow_orders=True,
            )


# ---------------------------------------------------------------------------
# 8. Duplicate fill prevention
# ---------------------------------------------------------------------------


class TestDuplicateFillPrevention:
    def test_a_fill_is_applied_at_most_once(self):
        broker = _broker()
        from tradingagents.paper_experiment.broker import BrokerOrder

        order = BrokerOrder(
            client_order_id="c1", broker_order_id="b1", symbol="SPY", side="buy",
            quantity=1, filled_quantity=1, filled_avg_price=100, status="filled",
        )
        assert broker.apply_fill(order) is True
        assert broker.apply_fill(order) is False  # second application refused

    def test_applied_fills_survive_a_restart(self, tmp_path):
        state = tmp_path / "broker.json"
        from tradingagents.paper_experiment.broker import BrokerOrder

        order = BrokerOrder(
            client_order_id="c1", broker_order_id="b1", symbol="SPY", side="buy",
            quantity=1, filled_quantity=1, filled_avg_price=100, status="filled",
        )
        first = _broker(state_path=state)
        assert first.apply_fill(order) is True
        second = _broker(state_path=state)
        assert second.apply_fill(order) is False

    def test_a_non_fill_status_is_never_applied(self):
        broker = _broker()
        from tradingagents.paper_experiment.broker import BrokerOrder

        order = BrokerOrder(
            client_order_id="c1", broker_order_id="b1", symbol="SPY", side="buy",
            quantity=1, filled_quantity=0, filled_avg_price=None, status="accepted",
        )
        assert broker.apply_fill(order) is False


# ---------------------------------------------------------------------------
# 9. Ledger separation
# ---------------------------------------------------------------------------


class TestLedgerSeparation:
    def test_the_broker_module_does_not_import_the_internal_ledger(self):
        """A broker fill must not be able to write the internal simulator ledger."""
        import tradingagents.paper_experiment.broker.alpaca_paper as mod

        source = mod.__file__
        text = Path(source).read_text(encoding="utf-8")
        for forbidden in ("portfolio.ledger", "portfolio.events", "execution.simulator",
                          "execution.paper"):
            assert forbidden not in text, f"broker module must not import {forbidden}"

    def test_the_broker_holds_its_own_account_state(self):
        broker = _broker()
        # The broker's account state is separate from any internal ledger.
        assert hasattr(broker, "_state")
        assert broker._state.cash is None  # unconfirmed until connect()

    def test_a_broker_fill_does_not_touch_the_internal_ledger(self):
        from tradingagents.paper_experiment.broker import BrokerOrder
        from tradingagents.portfolio.ledger import Ledger

        ledger = Ledger(initial_cash=100_000.0)
        broker = _broker()
        order = BrokerOrder(
            client_order_id="c1", broker_order_id="b1", symbol="SPY", side="buy",
            quantity=1, filled_quantity=1, filled_avg_price=100, status="filled",
        )
        before_cash = ledger.cash
        before_events = len(ledger.events)
        broker.apply_fill(order)
        # The broker's own state changed; the internal ledger did not.
        assert ledger.cash == before_cash
        assert len(ledger.events) == before_events
        assert ledger.holdings == {}


# ---------------------------------------------------------------------------
# 10. State integrity
# ---------------------------------------------------------------------------


class TestBrokerStateIntegrity:
    def test_a_missing_state_file_is_an_empty_state(self, tmp_path):
        state = BrokerStateStore(tmp_path / "missing.json").load()
        assert state.confirmed is False
        assert state.orders == {}

    def test_a_truncated_state_file_fails_safely(self, tmp_path):
        path = tmp_path / "broker.json"
        path.write_text('{"schema_version": 1, "orders": {')
        with pytest.raises(BrokerStateError):
            BrokerStateStore(path).load()

    def test_a_newer_schema_version_is_refused(self, tmp_path):
        path = tmp_path / "broker.json"
        path.write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(BrokerStateError):
            BrokerStateStore(path).load()

    def test_a_non_object_state_is_refused(self, tmp_path):
        path = tmp_path / "broker.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(BrokerStateError):
            BrokerStateStore(path).load()

    def test_a_state_round_trips(self, tmp_path):
        store = BrokerStateStore(tmp_path / "broker.json")
        from tradingagents.paper_experiment.broker import BrokerAccountState

        state = BrokerAccountState(
            endpoint=PAPER_BASE_URL, account_id="a1", cash=100.0, equity=100.0,
            confirmed=True, applied_fills={"c1"},
        )
        store.save(state)
        restored = store.load()
        assert restored.account_id == "a1"
        assert restored.applied_fills == {"c1"}


# ---------------------------------------------------------------------------
# 11. Secret-free diagnostic
# ---------------------------------------------------------------------------


class TestBrokerStatusDiagnostic:
    def test_it_reports_presence_not_values(self, monkeypatch):
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "super-secret-paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "super-secret-paper-secret-value")
        status = broker_status()
        text = json.dumps(status.to_dict())
        assert status.key_present is True
        assert status.secret_present is True
        assert "super-secret-paper-key-value" not in text
        assert "super-secret-paper-secret-value" not in text

    def test_it_flags_a_live_credential_without_using_it(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.setenv("APCA_API_KEY_ID", "live-key-value")
        status = broker_status()
        assert "APCA_API_KEY_ID" in status.live_credentials_present
        assert "live-key-value" not in json.dumps(status.to_dict())

    def test_it_confirms_the_paper_endpoint(self):
        status = broker_status()
        assert status.endpoint_ok is True
        assert status.paper_endpoint == PAPER_BASE_URL


# ---------------------------------------------------------------------------
# 12. The CLI diagnostic places no order
# ---------------------------------------------------------------------------


class TestBrokerCheckCommand:
    def test_it_reports_an_unconfigured_broker_without_error(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        assert main(["broker-check"]) == 0
        out = capsys.readouterr().out
        assert "not configured" in out
        assert "No order is placed" in out

    def test_it_reports_a_present_credential_without_leaking_it(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        key = "PAPER-KEY-NOT-REAL-123456"
        secret = "PAPER-SECRET-NOT-REAL-123456"
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", key)
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", secret)
        assert main(["broker-check"]) == 0
        out = capsys.readouterr().out
        assert "present: True" in out
        assert key not in out
        assert secret not in out
        assert str(len(key)) not in out

    def test_the_command_offers_no_order_flag(self):
        from tradingagents.paper_experiment.__main__ import main

        with pytest.raises(SystemExit):
            main(["broker-check", "--allow-orders"])  # unrecognised, so refused

    def test_connect_without_credentials_fails_safe(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        # An explicit request to connect cannot be honoured, so it fails safely
        # with a non-zero exit and a clear, secret-free message; it never falls
        # back to a live host and never places an order.
        assert main(["broker-check", "--connect"]) == 2
        captured = capsys.readouterr()
        assert "not configured" in captured.err
        assert "ALPACA_PAPER_API_KEY" in captured.err
        assert "no order" not in captured.out.lower()

    def test_the_alias_command_behaves_identically(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        assert main(["alpaca-diagnose"]) == 0
        out = capsys.readouterr().out
        assert "approved paper endpoint: https://paper-api.alpaca.markets" in out
        assert "No order is placed" in out


# ---------------------------------------------------------------------------
# 13. The read-only --connect diagnostic, end to end (transport faked only)
# ---------------------------------------------------------------------------


def _patch_transport(monkeypatch, routes, *, calls=None, fail=None):
    """Replace the broker's default transport with a fake one.

    The real broker, the real credential handling, the real response parsing and
    the real CLI rendering all run; only the socket is faked. This is the same
    seam the rest of the suite uses, so nothing about the adapter is mocked away.
    """
    import tradingagents.paper_experiment.broker.alpaca_paper as mod

    monkeypatch.setattr(mod, "_default_transport", _transport(routes, calls=calls, fail_on=fail))


class TestReadOnlyConnectDiagnostic:
    def test_it_reads_the_account_and_reports_the_figures(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        calls = []
        _patch_transport(monkeypatch, {("GET", "/v2/account"): _Resp(200, _ACCOUNT)}, calls=calls)

        assert main(["broker-check", "--connect"]) == 0
        out = capsys.readouterr().out
        assert "paper environment: CONFIRMED" in out
        assert "https://paper-api.alpaca.markets" in out
        assert "cash: 25000.0" in out
        assert "equity: 25000.0" in out
        assert "buying_power: 50000.0" in out
        assert "currency: USD" in out
        assert "restrictions: none reported" in out
        assert "no order was placed" in out
        # Read-only: every request is a GET; there is no POST/PUT/DELETE.
        assert calls and all(c["method"] == "GET" for c in calls)

    def test_it_never_prints_a_secret(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        key, secret = "PAPERKEY-do-not-print-9999", "PAPERSECRET-do-not-print-8888"
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", key)
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", secret)
        _patch_transport(monkeypatch, {("GET", "/v2/account"): _Resp(200, _ACCOUNT)})

        assert main(["broker-check", "--connect"]) == 0
        captured = capsys.readouterr()
        for stream in (captured.out, captured.err):
            assert key not in stream
            assert secret not in stream
            assert str(len(key)) not in stream
            assert str(len(secret)) not in stream

    def test_an_auth_failure_fails_safe_without_leaking(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        key = "PAPERKEY-leaky-1234"
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", key)
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        _patch_transport(
            monkeypatch,
            {("GET", "/v2/account"): _Resp(401, f'{{"message": "unauthorized {key}"}}')},
        )

        assert main(["broker-check", "--connect"]) == 2
        captured = capsys.readouterr()
        assert "refused the credential" in captured.err
        assert key not in captured.err
        assert key not in captured.out

    def test_a_timeout_fails_safe(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        _patch_transport(
            monkeypatch,
            {("GET", "/v2/account"): _Resp(200, _ACCOUNT)},
            fail=lambda m, u, b: True,
        )

        assert main(["broker-check", "--connect"]) == 2
        captured = capsys.readouterr()
        assert "broker check failed" in captured.err

    def test_a_non_object_response_fails_safe(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        _patch_transport(monkeypatch, {("GET", "/v2/account"): _Resp(200, {"nope": 1})})

        assert main(["broker-check", "--connect"]) == 2
        assert "broker check failed" in capsys.readouterr().err

    def test_a_blocked_account_is_reported_as_a_restriction(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        blocked = dict(_ACCOUNT, trading_blocked=True, transfers_blocked=True)
        _patch_transport(monkeypatch, {("GET", "/v2/account"): _Resp(200, blocked)})

        assert main(["broker-check", "--connect"]) == 0
        out = capsys.readouterr().out
        assert "TRADING_BLOCKED" in out
        assert "TRANSFERS_BLOCKED" in out

    def test_it_sends_no_order_request(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key-value")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret-value")
        calls = []
        # Any POST would have no route and raise AssertionError out of the transport.
        _patch_transport(monkeypatch, {("GET", "/v2/account"): _Resp(200, _ACCOUNT)}, calls=calls)

        assert main(["broker-check", "--connect"]) == 0
        assert {c["method"] for c in calls} == {"GET"}
        assert all("/v2/orders" not in c["url"] for c in calls)


# ---------------------------------------------------------------------------
# 14. Account restriction vocabulary
# ---------------------------------------------------------------------------


class TestAccountRestrictions:
    def _account(self, **over):
        from tradingagents.paper_experiment.broker import BrokerAccount

        base = {
            "account_id": "a",
            "status": "ACTIVE",
            "cash": 1.0,
            "equity": 1.0,
            "buying_power": 1.0,
            "currency": "USD",
            "trading_blocked": False,
            "account_blocked": False,
        }
        base.update(over)
        return BrokerAccount(**base)

    def test_a_clean_account_reports_no_restriction(self):
        assert self._account().restrictions() == []

    def test_each_block_is_a_named_restriction(self):
        assert "TRADING_BLOCKED" in self._account(trading_blocked=True).restrictions()
        assert "ACCOUNT_BLOCKED" in self._account(account_blocked=True).restrictions()
        assert "TRANSFERS_BLOCKED" in self._account(transfers_blocked=True).restrictions()
        assert (
            "TRADE_SUSPENDED_BY_USER"
            in self._account(trade_suspended_by_user=True).restrictions()
        )

    def test_shorting_disabled_is_reported_only_when_explicitly_false(self):
        assert "SHORTING_DISABLED" in self._account(shorting_enabled=False).restrictions()
        assert "SHORTING_DISABLED" not in self._account(shorting_enabled=True).restrictions()
        assert "SHORTING_DISABLED" not in self._account().restrictions()  # unknown, not false

    def test_a_missing_optional_field_is_unknown_not_zero(self):
        # portfolio_value absent -> None (unknown), not a fabricated 0.0.
        assert self._account().portfolio_value is None
        assert self._account().to_dict()["portfolio_value"] is None
