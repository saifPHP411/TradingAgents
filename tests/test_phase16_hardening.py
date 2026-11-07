"""Phase 16: hardening before the first manual Alpaca **paper** run.

Two hardening changes are pinned here:

1. ``PaperRiskLimits.max_order_notional`` defaults to ``25.0``, so the local
   per-order cap is a real second layer for a ~50 USD paper account instead of a
   nominal ``1000.0``. ``order_notional <= max_order_notional <= 25`` holds.
2. The daily-order counter is **durable**: it lives in the broker's
   ``state_path`` file with the UTC day it belongs to, so a restart on the same
   day keeps the count, a new day starts fresh, and an order the broker never
   accepted (a local refusal or a broker rejection) does not count.

Every test drives a **mocked HTTP transport**; no test reaches the network and
none needs a real key. The assertions are written to fail if the behaviour is
absent, not to confirm the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.paper_experiment.broker import (
    AlpacaPaperBroker,
    BrokerStateStore,
    PaperRiskLimits,
    PaperRiskRejected,
)
from tradingagents.paper_experiment.broker.state import BrokerAccountState

_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "50.00",
    "equity": "50.00",
    "buying_power": "50.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}

_KEY = "test-paper-key"
_SECRET = "test-paper-secret"


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


def _transport(routes, *, calls=None):
    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                # A callable response sees the request body, so a mock order can
                # echo the client_order_id it was submitted with.
                return response(json_body) if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _echo_order(body):
    """An order response echoing the submitted client_order_id."""
    cid = (body or {}).get("client_order_id", "c1")
    return _Resp(200, _order(client_order_id=cid))


def _ready(*, limits=None, state_path=None, calls=None, orders=None):
    """A connected broker over a mocked transport. Reads no real credential."""
    routes = {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("POST", "/v2/orders"): orders or _echo_order,
    }
    broker = AlpacaPaperBroker(
        api_key=_KEY,
        api_secret=_SECRET,
        transport=_transport(routes, calls=calls),
        limits=limits or PaperRiskLimits(),
        state_path=state_path,
    )
    broker.connect()
    return broker


# ---------------------------------------------------------------------------
# 1. max_order_notional is a real cap at or below 25
# ---------------------------------------------------------------------------


class TestNotionalCapDefault:
    def test_the_default_cap_does_not_exceed_25(self):
        assert PaperRiskLimits().max_order_notional <= 25.0

    def test_the_default_cap_is_positive_and_finite(self):
        cap = PaperRiskLimits().max_order_notional
        assert 0 < cap <= 25.0

    def test_the_default_order_notional_does_not_exceed_the_cap(self):
        from tradingagents.paper_experiment.broker import PaperTradingConfig

        config = PaperTradingConfig()
        assert config.order_notional <= config.limits.max_order_notional <= 25.0

    def test_an_order_above_25_is_refused_locally(self):
        calls = []
        broker = _ready(calls=calls)
        # 1 share at 100 = 100 > the 25 cap.
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=100.0,
                client_order_id="c1", allow_orders=True,
            )
        assert "notional" in str(excinfo.value)
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_an_order_at_or_below_the_cap_is_accepted(self):
        calls = []
        broker = _ready(calls=calls)
        # 2 shares at 10 = 20 <= 25.
        order = broker.submit_order(
            symbol="SPY", quantity=2, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        assert order.status == "accepted"
        assert len([c for c in calls if c["method"] == "POST"]) == 1


# ---------------------------------------------------------------------------
# 2. The daily-order counter is durable
# ---------------------------------------------------------------------------


class TestDurableDailyOrders:
    def test_an_accepted_order_increments_the_counter(self):
        broker = _ready()
        assert broker.daily_orders() == 0
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        assert broker.daily_orders() == 1

    def test_a_locally_refused_order_does_not_increment(self):
        broker = _ready()
        with pytest.raises(PaperRiskRejected):
            broker.submit_order(
                symbol="SPY", quantity=100, side="buy", reference_price=100.0,
                client_order_id="c1", allow_orders=True,
            )
        assert broker.daily_orders() == 0

    def test_a_broker_rejected_order_does_not_increment(self):
        from tradingagents.paper_experiment.broker import BrokerRejectedError

        broker = _ready(orders=_Resp(422, {"message": "insufficient buying power"}))
        with pytest.raises(BrokerRejectedError):
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=10.0,
                client_order_id="c1", allow_orders=True,
            )
        assert broker.daily_orders() == 0

    def test_a_partial_fill_is_one_order_not_many(self):
        broker = _ready(
            orders=_Resp(
                200, _order(status="partially_filled", filled_qty="0.4", filled_avg_price="10.0")
            )
        )
        order = broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        assert broker.daily_orders() == 1
        # Applying the same fill twice (as a reconcile would) must not re-count.
        broker.apply_fill(order)
        broker.apply_fill(order)
        assert broker.daily_orders() == 1

    def test_a_restart_on_the_same_day_keeps_the_counter(self, tmp_path):
        state = tmp_path / "broker_state.json"
        first = _ready(state_path=state)
        for i in range(3):
            first.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=10.0,
                client_order_id=f"c{i}", allow_orders=True,
            )
        assert first.daily_orders() == 3

        # A fresh broker over the same state file — a process restart.
        second = _ready(state_path=state)
        assert second.daily_orders() == 3

    def test_a_new_day_resets_the_counter(self, tmp_path):
        state = tmp_path / "broker_state.json"
        # A state written "yesterday" with three orders already counted.
        store = BrokerStateStore(state)
        store.save(
            BrokerAccountState(
                confirmed=True,
                daily_orders_count=3,
                daily_orders_date="2000-01-01",
            )
        )
        broker = _ready(state_path=state)
        assert broker.daily_orders() == 0

    def test_the_day_is_recorded_so_it_can_roll_over(self, tmp_path):
        state = tmp_path / "broker_state.json"
        broker = _ready(state_path=state)
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        payload = json.loads(Path(state).read_text())
        assert payload["daily_orders_count"] == 1
        assert payload["daily_orders_date"]  # a YYYY-MM-DD day, not None

    def test_reaching_the_limit_refuses_a_new_order(self):
        calls = []
        limits = PaperRiskLimits(max_daily_orders=1, max_order_notional=25.0)
        broker = _ready(limits=limits, calls=calls)
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        with pytest.raises(PaperRiskRejected) as excinfo:
            broker.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=10.0,
                client_order_id="c2", allow_orders=True,
            )
        assert "daily order limit" in str(excinfo.value)
        assert len([c for c in calls if c["method"] == "POST"]) == 1

    def test_the_limit_survives_a_restart_on_the_same_day(self, tmp_path):
        state = tmp_path / "broker_state.json"
        limits = PaperRiskLimits(max_daily_orders=1, max_order_notional=25.0)
        first = _ready(limits=limits, state_path=state)
        first.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        # A restart must not hand back a fresh daily budget.
        second = _ready(limits=limits, state_path=state)
        with pytest.raises(PaperRiskRejected):
            second.submit_order(
                symbol="SPY", quantity=1, side="buy", reference_price=10.0,
                client_order_id="c2", allow_orders=True,
            )


# ---------------------------------------------------------------------------
# 3. Backward-compatible migration and secret-free state
# ---------------------------------------------------------------------------


class TestStateMigrationAndSecrets:
    def test_an_old_state_without_the_new_fields_still_loads(self, tmp_path):
        state = tmp_path / "broker_state.json"
        # A v1 state file written before the daily-order fields existed.
        old = {
            "schema_version": 1,
            "endpoint": "https://paper-api.alpaca.markets",
            "account_id": "paper-acct-1",
            "confirmed": True,
            "orders": {},
            "applied_fills": [],
            "positions": {},
        }
        state.write_text(json.dumps(old))
        broker = _ready(state_path=state)
        assert broker.daily_orders() == 0

    def test_a_malformed_daily_count_falls_back_to_zero(self, tmp_path):
        state = tmp_path / "broker_state.json"
        store = BrokerStateStore(state)
        payload = store.load().to_dict()
        payload["daily_orders_count"] = "not-a-number"
        payload["daily_orders_date"] = "2026-01-01"
        state.write_text(json.dumps(payload))
        broker = _ready(state_path=state)
        assert broker.daily_orders() == 0

    def test_the_persisted_state_holds_no_secret(self, tmp_path):
        state = tmp_path / "broker_state.json"
        broker = _ready(state_path=state)
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        text = Path(state).read_text()
        assert _KEY not in text
        assert _SECRET not in text
        assert "APCA-API" not in text

    def test_the_daily_counter_survives_alongside_order_memory(self, tmp_path):
        state = tmp_path / "broker_state.json"
        broker = _ready(state_path=state)
        broker.submit_order(
            symbol="SPY", quantity=1, side="buy", reference_price=10.0,
            client_order_id="c1", allow_orders=True,
        )
        reloaded = BrokerStateStore(state).load()
        assert reloaded.daily_orders_count == 1
        assert "c1" in reloaded.orders
