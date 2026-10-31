"""Phase 15: the continuous paper-trading **loop** behind the dashboard's START.

Phase 14 made START set the worker ``RUNNING`` but left the cycle to an explicit
caller. This phase makes START mean "run continuously": ``start()`` launches one
background loop that waits ``poll_interval_seconds`` and then calls ``run_once()``
once per interval until ``stop()``.

Every test drives a **mocked HTTP transport** and a fake market-data source; no
test reaches the network (the suite's autouse ``_no_network`` fixture would refuse
it) and no test needs a real key. No order is ever sent to Alpaca.

The claims pinned here:

- START starts exactly one loop; a second START does not create a second loop;
- START itself places no order, and neither does the loop before its first
  interval elapses;
- the loop calls ``run_once()`` periodically, using ``poll_interval_seconds``;
- STOP stops the loop and leaves the state ``STOPPED``; it closes no position and
  cancels no order;
- a STOP that lands while a cycle is in flight is handled gracefully, without
  corrupting the state file;
- an unexpected exception inside a cycle is contained: the state goes ``ERROR``,
  the loop stops, and the process keeps running;
- a restart never auto-starts the loop;
- the existing order idempotency (one ``client_order_id``, one order) holds even
  when the loop repeats a decision.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time

import pytest

from tradingagents.paper_experiment.broker import (
    STATUS_ERROR,
    STATUS_RUNNING,
    STATUS_STOPPED,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
    TradingDecision,
    WorkerError,
)
from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController

LOOP_THREAD_NAME = "paper-trading-loop"

# Captured at import, before the suite's autouse network guard patches them, so a
# loopback-only dashboard test can still bind and connect to 127.0.0.1.
_REAL_CONNECT = socket.socket.connect
_REAL_LOOKUP = socket.getaddrinfo

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
        "qty": "2",
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
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _default_routes(*, orders=None, by_client_order_id=None):
    return {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, []),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id or _Resp(200, _order()),
    }


def _broker_factory(routes, *, calls=None):
    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes, calls=calls),
            limits=PaperRiskLimits(),
        )

    return build


def _decision(decision_id="d1", *, symbol="SPY", side="buy", notional=25.0):
    return TradingDecision(decision_id=decision_id, symbol=symbol, side=side, notional=notional)


class _Source:
    """A decision source yielding a fixed list, then ``None`` forever."""

    def __init__(self, decisions=()):
        self._decisions = list(decisions)
        self._i = 0
        self.calls = 0

    def next_decision(self):
        self.calls += 1
        if self._i >= len(self._decisions):
            return None
        d = self._decisions[self._i]
        self._i += 1
        return d


class _RepeatingSource:
    """A source that always yields the *same* decision (a duplicate-order trap)."""

    def __init__(self, decision):
        self._decision = decision
        self.calls = 0

    def next_decision(self):
        self.calls += 1
        return self._decision


class _ExplodingSource:
    """A source whose ``next_decision`` raises an unexpected error."""

    def __init__(self):
        self.calls = 0

    def next_decision(self):
        self.calls += 1
        raise RuntimeError("the decision source blew up")


class _BlockingSource:
    """A source that blocks inside a cycle until ``release`` is set."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def next_decision(self):
        self.entered.set()
        self.release.wait(timeout=5)
        return None


def _worker(tmp_path, *, source, prices=None, routes=None, calls=None, poll_interval=0.05):
    config = PaperTradingConfig(
        order_notional=25.0,
        limits=PaperRiskLimits(),
        state_path=tmp_path / "worker_state.json",
        poll_interval_seconds=poll_interval,
    )
    worker = PaperTradingWorker(
        config=config,
        decision_source=source,
        market_data=StaticMarketData(prices or {"SPY": 10.0}),
        broker_factory=_broker_factory(routes or _default_routes(), calls=calls),
    )
    _WORKERS.append(worker)
    return worker


_WORKERS: list[PaperTradingWorker] = []


@pytest.fixture(autouse=True)
def _stop_workers():
    """Stop every worker this test started, so no loop outlives the test."""
    start = len(_WORKERS)
    yield
    for worker in _WORKERS[start:]:
        with contextlib.suppress(Exception):
            worker.stop()
    del _WORKERS[start:]


def _live_loops():
    return {t for t in threading.enumerate() if t.name == LOOP_THREAD_NAME and t.is_alive()}


def _worker_loop_alive(worker):
    thread = worker._loop  # noqa: SLF001 - asserting the worker's own seam
    return bool(thread is not None and thread.is_alive())


def _wait_for(predicate, *, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _posts(calls):
    return [c for c in calls if c["method"] == "POST"]


# ---------------------------------------------------------------------------
# 1. START starts a loop
# ---------------------------------------------------------------------------


class TestStartStartsLoop:
    def test_start_launches_exactly_one_loop(self, tmp_path):
        worker = _worker(tmp_path, source=_Source([_decision()]))
        try:
            result = worker.start()
            assert result["status"] == STATUS_RUNNING
            assert worker.loop_running is True
            assert _worker_loop_alive(worker)
        finally:
            worker.stop()
        assert worker.loop_running is False

    def test_start_itself_places_no_order(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, source=_Source([_decision()]), calls=calls)
        try:
            worker.start()
            # START must not have placed an order...
            assert _posts(calls) == []
            # ...and neither has the loop, before its first interval has elapsed.
            time.sleep(0.02)
            assert _posts(calls) == []
        finally:
            worker.stop()

    def test_status_reports_loop_fields(self, tmp_path):
        worker = _worker(tmp_path, source=_Source([_decision()]), poll_interval=0.05)
        try:
            worker.start()
            status = worker.status()
            assert status["loop_running"] is True
            assert status["poll_interval_seconds"] == 0.05
            assert status["next_cycle_at"] is not None
            assert status["cycles_completed"] == 0
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 2. The loop calls run_once periodically, on poll_interval_seconds
# ---------------------------------------------------------------------------


class TestLoopIsPeriodic:
    def test_the_loop_calls_run_once_repeatedly(self, tmp_path):
        calls = []
        source = _Source([_decision("d1"), _decision("d2")])
        worker = _worker(tmp_path, source=source, calls=calls, poll_interval=0.02)
        try:
            worker.start()
            assert _wait_for(lambda: worker.state.cycles_completed >= 3, timeout=3.0)
        finally:
            worker.stop()
        # Several cycles ran, so run_once was called repeatedly, not once.
        assert source.calls >= 3
        assert worker.state.cycles_completed >= 3

    def test_the_loop_places_one_order_per_decision(self, tmp_path):
        calls = []
        source = _Source([_decision("d1"), _decision("d2")])
        worker = _worker(tmp_path, source=source, calls=calls, poll_interval=0.02)
        try:
            worker.start()
            assert _wait_for(lambda: len(_posts(calls)) >= 2, timeout=3.0)
        finally:
            worker.stop()
        assert len(_posts(calls)) == 2

    def test_poll_interval_is_actually_used(self, tmp_path):
        """A long interval means no cycle in the same window a short one cycles in."""
        slow = _worker(
            tmp_path / "slow", source=_Source([_decision()]), poll_interval=5.0
        )
        try:
            slow.start()
            time.sleep(0.25)
            assert slow.state.cycles_completed == 0
        finally:
            slow.stop()

        fast = _worker(
            tmp_path / "fast", source=_Source([_decision()]), poll_interval=0.02
        )
        try:
            fast.start()
            assert _wait_for(lambda: fast.state.cycles_completed >= 1, timeout=3.0)
        finally:
            fast.stop()


# ---------------------------------------------------------------------------
# 3. STOP stops the loop
# ---------------------------------------------------------------------------


class TestStopStopsLoop:
    def test_stop_stops_the_loop(self, tmp_path):
        worker = _worker(tmp_path, source=_Source([_decision()]), poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: worker.state.cycles_completed >= 1, timeout=3.0)
        worker.stop()
        assert worker.loop_running is False
        assert worker.state.status == STATUS_STOPPED
        assert not _worker_loop_alive(worker)
        # No further cycles after stop.
        done = worker.state.cycles_completed
        time.sleep(0.1)
        assert worker.state.cycles_completed == done

    def test_stop_does_not_cancel_or_close(self, tmp_path):
        calls = []
        worker = _worker(tmp_path, source=_Source([_decision()]), calls=calls, poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: len(_posts(calls)) >= 1, timeout=3.0)
        worker.stop()
        assert [c for c in calls if c["method"] == "DELETE"] == []

    def test_stop_during_a_cycle_is_graceful(self, tmp_path):
        """A STOP that lands mid-cycle waits for it, then leaves a valid state file."""
        source = _BlockingSource()
        worker = _worker(tmp_path, source=source, poll_interval=0.01)
        worker.start()
        assert source.entered.wait(timeout=3.0), "the loop never entered a cycle"

        stopped = threading.Event()

        def do_stop():
            worker.stop()
            stopped.set()

        stopper = threading.Thread(target=do_stop)
        stopper.start()
        # The cycle is blocked; stop must not have completed yet (graceful wait).
        assert not stopped.wait(timeout=0.15)
        source.release.set()
        stopper.join(timeout=5.0)
        assert stopped.is_set()
        assert worker.state.status == STATUS_STOPPED
        assert worker.loop_running is False
        # The state file is intact JSON, not a torn write.
        payload = json.loads((tmp_path / "worker_state.json").read_text())
        assert payload["status"] == STATUS_STOPPED
        assert isinstance(payload["orders"], dict)


# ---------------------------------------------------------------------------
# 4. One loop only; stop-then-start works
# ---------------------------------------------------------------------------


class TestOneLoopOnly:
    def test_start_twice_does_not_create_two_loops(self, tmp_path):
        worker = _worker(tmp_path, source=_Source([_decision()]), poll_interval=0.02)
        try:
            worker.start()
            with pytest.raises(WorkerError):
                worker.start()
            assert _worker_loop_alive(worker)
        finally:
            worker.stop()
        assert not _worker_loop_alive(worker)

    def test_stop_then_start_runs_one_new_loop(self, tmp_path):
        source = _Source([_decision("d1"), _decision("d2"), _decision("d3")])
        worker = _worker(tmp_path, source=source, poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: worker.state.cycles_completed >= 1, timeout=3.0)
        worker.stop()
        assert not _worker_loop_alive(worker)

        worker.start()
        try:
            assert _worker_loop_alive(worker)
            assert worker.state.cycles_completed == 0  # reset on a fresh start
            assert _wait_for(lambda: worker.state.cycles_completed >= 1, timeout=3.0)
        finally:
            worker.stop()
        assert not _worker_loop_alive(worker)


# ---------------------------------------------------------------------------
# 5. Restart never auto-starts
# ---------------------------------------------------------------------------


class TestRestartNoAutoStart:
    def test_a_restart_does_not_start_the_loop(self, tmp_path):
        worker = _worker(tmp_path, source=_Source([_decision()]), poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: worker.state.cycles_completed >= 1, timeout=3.0)
        worker.stop()

        # A fresh worker over the same state file.
        reloaded = _worker(tmp_path, source=_Source([_decision()]), poll_interval=0.02)
        assert reloaded.state.status == STATUS_STOPPED
        assert reloaded.loop_running is False
        assert not _worker_loop_alive(worker)
        # The persisted counter is kept, but no new cycle runs without a START.
        kept = reloaded.state.cycles_completed
        time.sleep(0.1)
        assert reloaded.state.cycles_completed == kept


# ---------------------------------------------------------------------------
# 6. An unexpected cycle failure is contained
# ---------------------------------------------------------------------------


class TestCycleFailureIsContained:
    def test_an_unexpected_error_sets_error_and_stops_the_loop(self, tmp_path):
        worker = _worker(tmp_path, source=_ExplodingSource(), poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: worker.state.status == STATUS_ERROR, timeout=3.0)
        # The loop stopped itself, the process is alive, and the state is readable.
        assert _wait_for(lambda: not worker.loop_running, timeout=3.0)
        assert worker.state.last_error and "RuntimeError" in worker.state.last_error
        assert not _worker_loop_alive(worker)
        payload = json.loads((tmp_path / "worker_state.json").read_text())
        assert payload["status"] == STATUS_ERROR

    def test_an_error_does_not_leave_a_loop_behind(self, tmp_path):
        worker = _worker(tmp_path, source=_ExplodingSource(), poll_interval=0.02)
        worker.start()
        assert _wait_for(lambda: worker.state.status == STATUS_ERROR, timeout=3.0)
        assert _wait_for(lambda: not worker.loop_running, timeout=3.0)
        time.sleep(0.1)
        assert not _worker_loop_alive(worker)


# ---------------------------------------------------------------------------
# 7. The loop does not duplicate orders
# ---------------------------------------------------------------------------


class TestLoopIdempotency:
    def test_a_repeated_decision_places_one_order(self, tmp_path):
        calls = []
        source = _RepeatingSource(_decision("d1"))
        worker = _worker(tmp_path, source=source, calls=calls, poll_interval=0.02)
        try:
            worker.start()
            # Let several cycles run over the same decision.
            assert _wait_for(lambda: source.calls >= 5, timeout=3.0)
        finally:
            worker.stop()
        # Exactly one order, however many cycles ran.
        assert len(_posts(calls)) == 1
        assert source.calls >= 5


# ---------------------------------------------------------------------------
# 8. The dashboard controller drives the loop
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_root(tmp_path):
    import pandas as pd

    from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
    from tradingagents.paper_experiment.inference import StubInference
    from tradingagents.paper_experiment.sources import StaticPriceSource

    root = tmp_path / "exp"
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


class TestControllerLoop:
    def test_controller_start_runs_a_loop_and_stop_ends_it(self, paper_root):
        calls = []
        controller = PaperTradingController(
            paper_root,
            enabled=True,
            broker_factory=_broker_factory(_default_routes(), calls=calls),
        )
        try:
            started = controller.start()
            assert started["status"] == STATUS_RUNNING
            assert controller.status()["loop_running"] is True
            assert len(_live_loops()) == 1
        finally:
            controller.stop()
        assert controller.status()["loop_running"] is False
        assert _live_loops() == set()

    def test_controller_stop_before_any_start_does_not_build_a_worker(self, paper_root):
        controller = PaperTradingController(paper_root, enabled=True)
        result = controller.stop()
        assert result["status"] == STATUS_STOPPED
        assert result["stopped"] is False
        # No worker was built, so no request could have been made.
        assert controller._worker is None  # noqa: SLF001 - asserting the seam

    def test_controller_shutdown_is_safe_when_idle(self, paper_root):
        controller = PaperTradingController(paper_root, enabled=True)
        controller.shutdown()  # must not raise and must not build a worker
        assert controller._worker is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 8b. The dashboard HTTP endpoints drive the loop end to end
# ---------------------------------------------------------------------------


class TestDashboardHttpLoop:
    def _serve(self, paper_root, monkeypatch):
        from http.server import ThreadingHTTPServer

        from tradingagents.paper_experiment.dashboard.server import (
            PAPER_CONFIRM_HEADER,
            Dashboard,
            make_handler,
        )

        def _refuse(*a, **k):
            raise OSError("the test must not reach the network")

        monkeypatch.setattr(
            socket.socket,
            "connect",
            lambda sock, address: (
                _REAL_CONNECT(sock, address) if address[0] == "127.0.0.1" else _refuse()
            ),
        )
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, *a, **k: _REAL_LOOKUP(host, *a, **k) if host == "127.0.0.1" else _refuse(),
        )
        dashboard = Dashboard(
            paper_root,
            token="t",
            allow_paper_trading=True,
            broker_factory=_broker_factory(_default_routes()),
        )
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}", dashboard, PAPER_CONFIRM_HEADER

    def _post(self, url, header):
        import urllib.request

        request = urllib.request.Request(url, method="POST")
        request.add_header("Authorization", "Bearer t")
        request.add_header(header, "yes")
        with urllib.request.urlopen(request, data=b"{}", timeout=10) as response:
            return response.status, json.loads(response.read())

    def _get(self, url):
        import urllib.request

        request = urllib.request.Request(url)
        request.add_header("Authorization", "Bearer t")
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def test_http_start_runs_a_loop_and_http_stop_ends_it(self, paper_root, monkeypatch):
        httpd, base, dashboard, header = self._serve(paper_root, monkeypatch)
        try:
            # Shorten the interval so the test does not wait 60 seconds. This sets
            # the config the controller builds the worker with.
            original = PaperTradingController._build_worker

            def _fast(self):
                worker = original(self)
                worker.config.poll_interval_seconds = 0.02
                return worker

            monkeypatch.setattr(PaperTradingController, "_build_worker", _fast)

            status, body = self._post(f"{base}/api/paper/start", header)
            assert status == 200
            assert body["status"] == STATUS_RUNNING

            # The status endpoint reports the loop as running.
            status, body = self._get(f"{base}/api/paper/status")
            assert status == 200
            assert body["loop_running"] is True

            # The loop advances on its own.
            worker = dashboard.paper._worker  # noqa: SLF001
            assert _wait_for(lambda: worker.state.cycles_completed >= 1, timeout=3.0)

            status, body = self._post(f"{base}/api/paper/stop", header)
            assert status == 200
            assert body["status"] == STATUS_STOPPED
            assert body["loop_running"] is False
        finally:
            dashboard.paper.shutdown()
            httpd.shutdown()
            httpd.server_close()


# ---------------------------------------------------------------------------
# 9. Persistence stays backward-compatible
# ---------------------------------------------------------------------------


class TestPersistenceBackwardCompatible:
    def test_a_state_file_without_the_loop_counter_still_loads(self, tmp_path):
        path = tmp_path / "worker_state.json"
        legacy = {
            "schema_version": 1,
            "strategy_id": "alpaca-paper",
            "status": "RUNNING",
            "orders": {},
            "applied_fills": [],
            "positions": {},
            "account": {},
        }
        path.write_text(json.dumps(legacy))
        worker = _worker(tmp_path, source=_Source([_decision()]))
        # Reads back as STOPPED with the new counter defaulted, not an error.
        assert worker.state.status == STATUS_STOPPED
        assert worker.state.cycles_completed == 0

    def test_a_malformed_loop_counter_does_not_break_the_load(self, tmp_path):
        path = tmp_path / "worker_state.json"
        legacy = {
            "schema_version": 1,
            "strategy_id": "alpaca-paper",
            "status": "STOPPED",
            "cycles_completed": "not-an-int",
        }
        path.write_text(json.dumps(legacy))
        worker = _worker(tmp_path, source=_Source([_decision()]))
        assert worker.state.cycles_completed == 0
