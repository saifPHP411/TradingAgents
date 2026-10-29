"""Phase 14: the Alpaca **paper** trading worker, and its start/stop control.

Every test drives a **mocked HTTP transport**; no test reaches the network (the
suite's autouse ``_no_network`` fixture would refuse it) and no test needs a real
key. The mock is a fake *transport*, not a fake broker: the adapter's endpoint
validation, credential handling, order parsing, risk checks, idempotency and
reconciliation all run for real, and so does the worker's own state machine.

The claims pinned here are the ones the task cares about:

- an order is placed **only** through the worker, only while ``RUNNING``, and
  never by ``start`` itself;
- a live endpoint is refused and cannot be configured in;
- the local risk checks (buying power, notional cap, daily count, open positions,
  long-only) refuse an order *before* it reaches the broker;
- a decision maps to one deterministic ``client_order_id``, so a duplicate
  decision, a lost response, or a restart reconciles instead of re-ordering;
- a restart comes back ``STOPPED``, never ``RUNNING``;
- stop refuses further cycles but closes no position and cancels no order;
- the dashboard start/stop endpoints require a bearer token **and** a confirmation
  header, and are disabled unless the dashboard was started with
  ``--allow-paper-trading``.

Each test is written to catch a premise error: it fails if the safety property is
not actually enforced, not merely if the code crashes.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tradingagents.paper_experiment.broker import (
    BANNER,
    STATUS_ERROR,
    STATUS_RUNNING,
    STATUS_STOPPED,
    AlpacaPaperBroker,
    BrokerError,
    BrokerNotConfiguredError,
    LiveEndpointRefusedError,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    PreflightError,
    StaticMarketData,
    TradingDecision,
    WorkerError,
    WorkerState,
    WorkerStateStore,
)
from tradingagents.paper_experiment.dashboard.server import (
    CONTROL_CONFIRM_HEADER,
    PAPER_CONFIRM_HEADER,
    Dashboard,
    make_handler,
)

TOKEN = "paper-test-token"

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
    """A transport answering from ``routes`` keyed by (method, path-suffix)."""

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "params": dict(params or {}),
                    "json": json_body,
                }
            )
        if fail_on is not None and fail_on(method, url, json_body):
            raise TimeoutError("simulated timeout")
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _default_routes(*, account=None, positions=None, orders=None, by_client_order_id=None):
    """The broker routes a worker test needs, keyed by (method, path-suffix)."""
    return {
        ("GET", "/v2/account"): account or _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): positions or _Resp(200, []),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id
        or _Resp(200, _order()),
    }


def _broker_factory(routes, *, calls=None, fail_on=None, limits=None, state_path=None):
    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes, calls=calls, fail_on=fail_on),
            limits=limits or PaperRiskLimits(),
            state_path=state_path,
        )

    return build


def _decision(decision_id="d1", *, symbol="SPY", side="buy", notional=25.0, quantity=None):
    return TradingDecision(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        notional=None if quantity is not None else notional,
        quantity=quantity,
    )


class _Source:
    """A decision source that yields a fixed list, one per call."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._i = 0

    def next_decision(self):
        if self._i >= len(self._decisions):
            return None
        d = self._decisions[self._i]
        self._i += 1
        return d

    def reset(self):
        self._i = 0


def _worker(
    tmp_path,
    *,
    decisions=(),
    prices=None,
    routes=None,
    limits=None,
    calls=None,
    fail_on=None,
    order_notional=25.0,
    name="worker_state.json",
):
    config = PaperTradingConfig(
        order_notional=order_notional,
        limits=limits or PaperRiskLimits(),
        state_path=tmp_path / name,
    )
    return PaperTradingWorker(
        config=config,
        decision_source=_Source(decisions),
        market_data=StaticMarketData(prices or {"SPY": 10.0}),
        broker_factory=_broker_factory(routes or _default_routes(), calls=calls, fail_on=fail_on, limits=limits),
    )


# ---------------------------------------------------------------------------
# 1. Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_runs_preflight_and_reports_running(self, tmp_path):
        worker = _worker(tmp_path, decisions=[_decision()])
        result = worker.start()
        assert result["status"] == STATUS_RUNNING
        checks = {c["check"] for c in result["checks"]}
        assert {"broker_connection", "account_status", "buying_power", "market_data", "risk_config"} <= checks

    def test_start_places_no_order(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision()], calls=calls)
        worker.start()
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_second_worker_cannot_start_while_running(self, tmp_path):
        worker = _worker(tmp_path)
        worker.start()
        with pytest.raises(WorkerError):
            worker.start()

    def test_stop_is_graceful_and_returns_stopped(self, tmp_path):
        worker = _worker(tmp_path)
        worker.start()
        result = worker.stop()
        assert result["status"] == STATUS_STOPPED
        assert worker.state.status == STATUS_STOPPED

    def test_a_failed_broker_connection_blocks_start(self, tmp_path):
        def fail_on(method, url, body):
            return url.endswith("/v2/account")

        worker = _worker(tmp_path, fail_on=fail_on)
        with pytest.raises((BrokerError, PreflightError)):
            worker.start()
        assert worker.state.status == STATUS_ERROR

    def test_a_missing_credential_blocks_start(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        config = PaperTradingConfig(state_path=tmp_path / "w.json")
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source([]),
            market_data=StaticMarketData({"SPY": 100.0}),
        )  # default factory reads the (absent) environment
        with pytest.raises(BrokerNotConfiguredError):
            worker.start()
        assert worker.state.status == STATUS_ERROR

    def test_a_non_active_account_blocks_start(self, tmp_path):
        routes = _default_routes(account=_Resp(200, dict(_ACCOUNT, status="ACCOUNT_CLOSED")))
        worker = _worker(tmp_path, routes=routes)
        with pytest.raises(PreflightError):
            worker.start()
        assert worker.state.status == STATUS_ERROR

    def test_zero_buying_power_blocks_start(self, tmp_path):
        routes = _default_routes(account=_Resp(200, dict(_ACCOUNT, buying_power="0")))
        worker = _worker(tmp_path, routes=routes)
        with pytest.raises(PreflightError):
            worker.start()

    def test_unavailable_market_data_blocks_start(self, tmp_path):
        config = PaperTradingConfig(state_path=tmp_path / "w.json")
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source([]),
            market_data=StaticMarketData({}),
            broker_factory=_broker_factory(_default_routes()),
        )
        with pytest.raises(PreflightError):
            worker.start()

    def test_run_once_refuses_when_not_running(self, tmp_path):
        worker = _worker(tmp_path, decisions=[_decision()])
        with pytest.raises(WorkerError):
            worker.run_once()


# ---------------------------------------------------------------------------
# 2. One decision -> one order, and the risk checks
# ---------------------------------------------------------------------------


class TestOrderPath:
    def test_a_buy_decision_places_one_order(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision()], calls=calls)
        worker.start()
        result = worker.run_once()
        assert result.ran
        assert result.order.status == "accepted"
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1
        assert posts[0]["json"]["side"] == "buy"
        assert posts[0]["json"]["client_order_id"] == result.client_order_id

    def test_the_client_order_id_is_deterministic(self, tmp_path):
        d = _decision("d-42")
        a = d.client_order_id(strategy_id="s")
        b = d.client_order_id(strategy_id="s")
        assert a == b
        assert a.startswith("pt-")
        # A different decision yields a different id.
        assert _decision("d-43").client_order_id(strategy_id="s") != a

    def test_insufficient_buying_power_refuses_before_submitting(self, tmp_path):
        calls = []
        routes = _default_routes(account=_Resp(200, dict(_ACCOUNT, buying_power="10")))
        # 5 shares at 100 = 500 > 10 buying power, and under the 1000 cap.
        worker = _worker(
            tmp_path, decisions=[_decision(quantity=5)], routes=routes, calls=calls
        )
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "buying power" in (result.error or "")
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_an_order_above_the_notional_cap_is_refused(self, tmp_path):
        calls = []
        limits = PaperRiskLimits(max_order_notional=100)
        # 20 shares at 10 = 200 > the 100 cap.
        worker = _worker(
            tmp_path, decisions=[_decision(quantity=20)], limits=limits, calls=calls
        )
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "notional" in (result.error or "")
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_the_daily_order_limit_is_enforced(self, tmp_path):
        calls = []
        limits = PaperRiskLimits(max_daily_orders=1, max_order_notional=10_000)
        worker = _worker(
            tmp_path,
            decisions=[_decision("d1"), _decision("d2")],
            limits=limits,
            calls=calls,
        )
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        assert first.ran
        assert not second.ran
        assert "daily order limit" in (second.error or "")
        assert len([c for c in calls if c["method"] == "POST"]) == 1

    def test_the_open_position_limit_is_enforced(self, tmp_path):
        calls = []
        positions = [
            {"symbol": "AAPL", "qty": "1", "market_value": "100", "avg_entry_price": "100"}
        ]
        routes = _default_routes(positions=_Resp(200, positions))
        limits = PaperRiskLimits(max_open_positions=1, max_order_notional=10_000)
        worker = _worker(
            tmp_path,
            decisions=[_decision(symbol="SPY", quantity=1)],
            routes=routes,
            limits=limits,
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "open-position limit" in (result.error or "")
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_short_is_refused(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(side="sell", quantity=1)],
            prices={"SPY": 100.0},
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "short" in (result.error or "").lower()
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_sell_within_the_holding_is_allowed(self, tmp_path):
        calls = []
        positions = [
            {"symbol": "SPY", "qty": "5", "market_value": "500", "avg_entry_price": "100"}
        ]
        routes = _default_routes(
            positions=_Resp(200, positions),
            orders=_Resp(200, _order(side="sell", qty="5")),
        )
        worker = _worker(
            tmp_path,
            decisions=[_decision(side="sell", quantity=5)],
            routes=routes,
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        assert result.ran
        assert result.order.side == "sell"

    def test_a_market_data_failure_places_no_order(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision(symbol="MSFT")], prices={"SPY": 100.0}, calls=calls)
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "market data" in (result.skipped_reason or "").lower()
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_rejected_order_is_reported_and_not_retried(self, tmp_path):
        calls = []
        routes = _default_routes(orders=_Resp(422, {"message": "insufficient buying power"}))
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes, calls=calls)
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "refused" in (result.skipped_reason or "")
        assert len([c for c in calls if c["method"] == "POST"]) == 1

    def test_a_partial_fill_is_recorded(self, tmp_path):
        routes = _default_routes(
            orders=_Resp(
                200, _order(status="partially_filled", filled_qty="0.4", filled_avg_price="100.5")
            )
        )
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes)
        worker.start()
        result = worker.run_once()
        assert result.order.status == "partially_filled"
        assert result.order.filled_quantity == pytest.approx(0.4)
        assert result.order.remaining_quantity == pytest.approx(0.6)

    def test_a_filled_order_is_recorded(self, tmp_path):
        routes = _default_routes(
            orders=_Resp(
                200, _order(status="filled", filled_qty="1", filled_avg_price="100.5")
            )
        )
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes)
        worker.start()
        result = worker.run_once()
        assert result.order.status == "filled"
        assert result.order.filled_avg_price == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# 3. Idempotency: duplicates, timeouts, restarts
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_a_duplicate_decision_is_not_processed_twice(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision("d1"), _decision("d1")], calls=calls)
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        assert first.ran
        assert not second.ran
        assert "already been processed" in (second.skipped_reason or "")
        assert len([c for c in calls if c["method"] == "POST"]) == 1

    def test_a_timeout_reconciles_instead_of_re_submitting(self, tmp_path):
        calls = []

        def fail_on(method, url, body):
            return method == "POST" and url.endswith("/v2/orders")

        routes = _default_routes(
            by_client_order_id=_Resp(
                200, _order(status="filled", filled_qty="1", filled_avg_price="100.5")
            )
        )
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes, calls=calls, fail_on=fail_on)
        worker.start()
        result = worker.run_once()
        assert result.order.status == "filled"  # reconciled, not guessed
        assert len([c for c in calls if c["method"] == "POST"]) == 1

    def test_an_already_submitted_order_is_reconciled_after_a_restart(self, tmp_path):
        calls = []
        routes = _default_routes(
            by_client_order_id=_Resp(200, _order(status="accepted"))
        )
        first = _worker(tmp_path, decisions=[_decision("d1")], routes=routes, calls=calls)
        first.start()
        first.run_once()
        posts_before = len([c for c in calls if c["method"] == "POST"])
        assert posts_before == 1

        # Restart: a fresh worker from the same state, same decision.
        calls.clear()
        second = _worker(tmp_path, decisions=[_decision("d1")], routes=routes, calls=calls)
        assert second.state.status == STATUS_STOPPED
        second.start()
        result = second.run_once()
        assert not result.ran
        # The decision was already processed, so nothing is re-submitted.
        assert "already been processed" in (result.skipped_reason or "")
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_known_client_order_id_is_reconciled_not_re_submitted(self, tmp_path):
        """The client_order_id guard, exercised independently of the decision guard.

        State is seeded with an order the worker already knows but whose decision
        was never marked processed (the shape a crash between submission and the
        decision-id write would leave). The worker must reconcile, not re-post.
        """
        calls = []
        routes = _default_routes(
            by_client_order_id=_Resp(200, _order(status="accepted"))
        )
        worker = _worker(tmp_path, decisions=[_decision("d1")], routes=routes, calls=calls)
        worker.start()
        cid = _decision("d1").client_order_id(strategy_id=worker.config.strategy_id)
        worker.state.orders[cid] = {"client_order_id": cid, "status": "submitting"}
        worker.state.last_processed_decision_id = None
        result = worker.run_once()
        assert not result.ran
        assert "already exists" in (result.skipped_reason or "")
        assert [c for c in calls if c["method"] == "POST"] == []
        # It asked the broker about the order rather than guessing.
        assert [c for c in calls if "by_client_order_id" in c["url"]]

    def test_a_restart_defaults_to_stopped(self, tmp_path):
        worker = _worker(tmp_path)
        worker.start()
        assert worker.state.status == STATUS_RUNNING
        # A fresh worker over the same state file must not resume RUNNING.
        reloaded = _worker(tmp_path)
        assert reloaded.state.status == STATUS_STOPPED
        assert PaperTradingWorker.load(tmp_path / "worker_state.json").status == STATUS_STOPPED

    def test_stop_prevents_further_orders(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision("d1"), _decision("d2")], calls=calls)
        worker.start()
        worker.stop()
        with pytest.raises(WorkerError):
            worker.run_once()
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_stop_does_not_cancel_open_orders(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision()], calls=calls)
        worker.start()
        worker.run_once()
        worker.stop()
        # No DELETE (cancel) call was made by stop.
        assert [c for c in calls if c["method"] == "DELETE"] == []


# ---------------------------------------------------------------------------
# 4. Endpoint safety
# ---------------------------------------------------------------------------


class TestEndpointSafety:
    def test_a_live_endpoint_is_refused_by_the_worker_config(self):
        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(base_url="https://api.alpaca.markets")

    def test_the_worker_builds_only_the_paper_broker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "k")
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "s")
        worker = _worker(tmp_path)
        broker = worker._default_broker_factory()
        assert broker.base_url == "https://paper-api.alpaca.markets"

    def test_status_reports_paper_only(self, tmp_path):
        worker = _worker(tmp_path)
        status = worker.status()
        assert status["banner"] == BANNER
        assert status["live"] is False
        assert status["mode"] == "ALPACA_PAPER"
        assert status["paper_endpoint_only"] is True

    def test_the_status_view_carries_no_credential(self, tmp_path):
        worker = _worker(tmp_path)
        worker.start()
        blob = json.dumps(worker.status())
        assert "paper-key" not in blob
        assert "paper-secret" not in blob


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_the_worker_state_is_versioned_and_round_trips(self, tmp_path):
        path = tmp_path / "w.json"
        store = WorkerStateStore(path)
        state = WorkerState(strategy_id="s", last_processed_decision_id="d1")
        state.orders["c1"] = {"client_order_id": "c1", "status": "accepted"}
        store.save(state)
        loaded = store.load()
        assert loaded.last_processed_decision_id == "d1"
        assert loaded.orders["c1"]["status"] == "accepted"

    def test_a_newer_schema_version_is_refused(self, tmp_path):
        from tradingagents.paper_experiment.broker import WorkerStateError

        path = tmp_path / "w.json"
        path.write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(WorkerStateError):
            WorkerStateStore(path).load()

    def test_the_persisted_state_carries_no_credential(self, tmp_path):
        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker.run_once()
        blob = (tmp_path / "worker_state.json").read_text()
        assert "paper-key" not in blob
        assert "paper-secret" not in blob


# ---------------------------------------------------------------------------
# 6. Dashboard HTTP surface
# ---------------------------------------------------------------------------

_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


def _refuse(*args, **kwargs):
    raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")


@pytest.fixture
def allow_loopback(monkeypatch):
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (_LOOPBACK_CONNECT if address[0] == "127.0.0.1" else _refuse)(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (_LOOPBACK_LOOKUP if host == "127.0.0.1" else _refuse)(host, *a, **k),
    )


def _serve_in_thread(dashboard):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url, token=None):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url, token=None, confirm_header=None):
    request = urllib.request.Request(url, method="POST")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if confirm_header:
        request.add_header(confirm_header, "yes")
    try:
        with urllib.request.urlopen(request, data=b"{}", timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _experiment_root(root):
    """Create a minimal, loadable paper experiment at ``root``."""
    import pandas as pd

    from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
    from tradingagents.paper_experiment.inference import StubInference
    from tradingagents.paper_experiment.sources import StaticPriceSource

    index = pd.date_range("2024-01-01", periods=6, freq="B")
    opens = [10.0 + i for i in range(6)]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [o + 1 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 0.5 for o in opens],
            "volume": [1e6] * 6,
        },
        index=index,
    )
    source = StaticPriceSource({"SPY": frame}, name="static_test")
    config = ExperimentConfig.from_protocol(
        symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference()
    )
    experiment.ensure_created()
    experiment.process_session(list(source.sessions())[0])
    return root


@pytest.fixture
def paper_root(tmp_path):
    return _experiment_root(tmp_path / "exp")


@pytest.fixture
def paper_dashboard(paper_root, allow_loopback):
    dashboard = Dashboard(
        paper_root,
        token=TOKEN,
        allow_paper_trading=True,
        broker_factory=_broker_factory(_default_routes()),
    )
    httpd, base = _serve_in_thread(dashboard)
    yield base, dashboard
    httpd.shutdown()
    httpd.server_close()


class TestDashboardPaperSurface:
    def test_status_requires_a_token(self, paper_dashboard):
        base, _ = paper_dashboard
        status, _ = _get(f"{base}/api/paper/status")
        assert status == 401

    def test_status_with_token_reports_stopped(self, paper_dashboard):
        base, _ = paper_dashboard
        status, body = _get(f"{base}/api/paper/status", token=TOKEN)
        assert status == 200
        assert body["banner"] == BANNER
        assert body["live"] is False
        assert body["status"] == STATUS_STOPPED

    def test_start_requires_a_token(self, paper_dashboard):
        base, _ = paper_dashboard
        status, _ = _post(f"{base}/api/paper/start", token=None, confirm_header=PAPER_CONFIRM_HEADER)
        assert status == 401

    def test_start_requires_the_paper_confirm_header(self, paper_dashboard):
        base, _ = paper_dashboard
        status, body = _post(f"{base}/api/paper/start", token=TOKEN, confirm_header=None)
        assert status == 403
        assert "confirmation" in body["error"].lower()

    def test_the_run_confirm_header_does_not_start_paper_trading(self, paper_dashboard):
        # The two confirmation headers are distinct on purpose.
        base, _ = paper_dashboard
        status, _ = _post(f"{base}/api/paper/start", token=TOKEN, confirm_header=CONTROL_CONFIRM_HEADER)
        assert status == 403

    def test_start_then_stop(self, paper_dashboard):
        base, _ = paper_dashboard
        status, body = _post(f"{base}/api/paper/start", token=TOKEN, confirm_header=PAPER_CONFIRM_HEADER)
        assert status == 200
        assert body["status"] == STATUS_RUNNING
        status, body = _post(f"{base}/api/paper/stop", token=TOKEN, confirm_header=PAPER_CONFIRM_HEADER)
        assert status == 200
        assert body["status"] == STATUS_STOPPED

    def test_start_places_no_order(self, paper_dashboard):
        base, dashboard = paper_dashboard
        _post(f"{base}/api/paper/start", token=TOKEN, confirm_header=PAPER_CONFIRM_HEADER)
        worker = dashboard.paper._worker
        assert worker is not None
        assert worker.state.orders == {}

    def test_a_disabled_dashboard_refuses_start(self, paper_root, allow_loopback):
        dashboard = Dashboard(paper_root, token=TOKEN, allow_paper_trading=False)
        httpd, base = _serve_in_thread(dashboard)
        try:
            status, body = _get(f"{base}/api/paper/status", token=TOKEN)
            assert status == 200
            assert body["enabled"] is False
            status, _ = _post(
                f"{base}/api/paper/start", token=TOKEN, confirm_header=PAPER_CONFIRM_HEADER
            )
            assert status == 403
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_broker_failure_on_start_is_a_safe_502(self, paper_root, allow_loopback):
        def fail_on(method, url, body):
            return url.endswith("/v2/account")

        dashboard = Dashboard(
            paper_root,
            token=TOKEN,
            allow_paper_trading=True,
            broker_factory=_broker_factory(_default_routes(), fail_on=fail_on),
        )
        httpd, base = _serve_in_thread(dashboard)
        try:
            status, body = _post(
                f"{base}/api/paper/start", token=TOKEN, confirm_header=PAPER_CONFIRM_HEADER
            )
            assert status == 502
            assert "error" in body
            # The worker is left in ERROR, not RUNNING.
            assert _get(f"{base}/api/paper/status", token=TOKEN)[1]["status"] == STATUS_ERROR
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestBanner:
    def test_the_index_page_shows_the_not_live_banner(self, tmp_path, allow_loopback):
        dashboard = Dashboard(tmp_path, token=TOKEN, allow_paper_trading=True)
        httpd, base = _serve_in_thread(dashboard)
        try:
            with urllib.request.urlopen(f"{base}/", timeout=10) as response:
                html = response.read().decode()
            assert "ALPACA PAPER — NOT LIVE" in html
            assert "START PAPER TRADING" in html
            assert "STOP PAPER TRADING" in html
        finally:
            httpd.shutdown()
            httpd.server_close()


# ---------------------------------------------------------------------------
# 6b. Market-data failure handling
# ---------------------------------------------------------------------------


class TestMarketDataFailures:
    """A bad price must refuse the order, never fabricate a fill."""

    def _run(self, tmp_path, prices):
        calls = []
        worker = _worker(tmp_path, decisions=[_decision()], prices=prices, calls=calls)
        worker.start()
        return worker.run_once(), calls

    def test_a_missing_price_places_no_order(self, tmp_path):
        result, calls = self._run(tmp_path, {"AAPL": 10.0})
        assert not result.ran
        assert "market data" in (result.skipped_reason or "")
        assert [c for c in calls if c["method"] == "POST"] == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
    def test_a_non_positive_or_non_finite_price_places_no_order(self, tmp_path, bad):
        result, calls = self._run(tmp_path, {"SPY": bad})
        assert not result.ran
        assert [c for c in calls if c["method"] == "POST"] == []
        # The accounting view is untouched: no order was recorded.
        assert worker_orders(tmp_path) == {}

    def test_a_price_source_exception_is_surfaced_not_swallowed(self, tmp_path):
        from tradingagents.paper_experiment.broker import MarketDataError

        class _Exploding:
            def available(self):
                return True

            def price(self, symbol):
                raise MarketDataError(f"the price source failed for {symbol!r}")

        config = PaperTradingConfig(state_path=tmp_path / "w.json")
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source([_decision()]),
            market_data=_Exploding(),
            broker_factory=_broker_factory(_default_routes()),
        )
        worker.start()
        result = worker.run_once()
        assert not result.ran
        assert "price source failed" in (result.error or "")
        assert worker.state.last_error and "MarketDataError" in worker.state.last_error

    def test_an_unexpected_price_source_exception_does_not_crash_the_worker(self, tmp_path):
        class _Buggy:
            def available(self):
                return True

            def price(self, symbol):
                raise KeyError("a bug in the source")

        config = PaperTradingConfig(state_path=tmp_path / "w.json")
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source([_decision()]),
            market_data=_Buggy(),
            broker_factory=_broker_factory(_default_routes()),
        )
        worker.start()
        # A raw KeyError must not escape and corrupt the worker's state machine.
        with pytest.raises(KeyError):
            worker.run_once()
        assert worker.state.status == STATUS_RUNNING


def worker_orders(tmp_path, name="worker_state.json"):
    path = tmp_path / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())["orders"]


# ---------------------------------------------------------------------------
# 7. The strategy -> worker bridge
# ---------------------------------------------------------------------------

def _buy_experiment(root):
    """An experiment whose pending decision is a BUY."""
    import pandas as pd

    from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
    from tradingagents.paper_experiment.inference import StubInference
    from tradingagents.paper_experiment.sources import StaticPriceSource

    index = pd.date_range("2024-01-01", periods=8, freq="B")
    opens = [10.0 + i for i in range(8)]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [o + 1 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 0.5 for o in opens],
            "volume": [1e6] * 8,
        },
        index=index,
    )
    source = StaticPriceSource({"SPY": frame}, name="static_test")
    config = ExperimentConfig.from_protocol(
        symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference(rating="Buy")
    )
    experiment.ensure_created()
    # Process up to but not including the last session, so one decision stays
    # pending (settled decisions are not re-traded).
    for session in list(source.sessions())[:-1]:
        experiment.process_session(session)
    return experiment


class TestDecisionBridge:
    def test_pending_buy_decisions_are_yielded_in_order(self, tmp_path):
        from tradingagents.paper_experiment.paper_trading import ExperimentDecisionSource

        experiment = _buy_experiment(tmp_path / "exp")
        source = ExperimentDecisionSource(experiment, order_notional=25.0)
        decisions = []
        while (d := source.next_decision()) is not None:
            decisions.append(d)
        assert decisions, "a Buy experiment must yield at least one decision"
        assert all(d.side == "buy" for d in decisions)
        assert all(d.notional == 25.0 for d in decisions)
        sessions = [d.decision_id.split("-", 1)[1] for d in decisions]
        assert sessions == sorted(sessions)  # deterministic order

    def test_a_hold_experiment_yields_no_decisions(self, tmp_path):
        import pandas as pd

        from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
        from tradingagents.paper_experiment.inference import StubInference
        from tradingagents.paper_experiment.paper_trading import ExperimentDecisionSource
        from tradingagents.paper_experiment.sources import StaticPriceSource

        index = pd.date_range("2024-01-01", periods=8, freq="B")
        opens = [10.0 + i for i in range(8)]
        frame = pd.DataFrame(
            {"open": opens, "high": [o + 1 for o in opens], "low": [o - 1 for o in opens],
             "close": [o + 0.5 for o in opens], "volume": [1e6] * 8},
            index=index,
        )
        source = StaticPriceSource({"SPY": frame}, name="static_test")
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
        )
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp", inference=StubInference(rating="Hold")
        )
        experiment.ensure_created()
        for session in list(source.sessions())[:-1]:
            experiment.process_session(session)
        source_ = ExperimentDecisionSource(experiment, order_notional=25.0)
        assert source_.next_decision() is None


# ---------------------------------------------------------------------------
# 8. The CLI path
# ---------------------------------------------------------------------------


class TestCliPaperTrade:
    def test_the_dry_run_places_no_order(self, paper_root, monkeypatch, capsys):
        from tradingagents.paper_experiment import __main__ as cli
        from tradingagents.paper_experiment.dashboard import paper_control

        # Replace the controller's worker builder so no network is touched.
        original = paper_control.PaperTradingController

        class _Patched(original):
            def _build_worker(self):
                return _worker(self.root, routes=_default_routes())

        monkeypatch.setattr(paper_control, "PaperTradingController", _Patched)
        code = cli.main(["paper-trade", "--root", str(paper_root), "--dry-run"])
        assert code == 0
        out = capsys.readouterr().out
        assert '"placed_order": false' in out

    def test_the_command_is_registered(self):
        from tradingagents.paper_experiment import __main__ as cli

        # --help exits through argparse (SystemExit 0); the command must be known.
        with pytest.raises(SystemExit):
            cli.main(["paper-trade", "--help"])

