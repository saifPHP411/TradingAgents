"""Phase 17: a worker must *discover* decisions produced after it was built.

The failure this pins: the dashboard built the worker on a status poll, and the
decision source loaded the experiment's decisions once and cached them. A BUY
recorded later — or a decision the experiment settled internally before the
worker's first cycle — was never seen, so the worker reported
``no decision is due`` forever.

Three fixes are pinned here, and each test is written to fail if the behaviour is
absent rather than to confirm the code:

1. ``ExperimentDecisionSource`` re-reads the experiment (through a caller-supplied
   ``reload``) on every decision, so a decision produced after the build is found.
   It keeps a build-time baseline of decisions that were *already settled*, so
   starting a worker does not replay months of history.
2. ``PaperTradingController.status()`` never builds the worker, so a dashboard
   polling status cannot freeze an empty queue. The worker is built on START.
3. The worker keeps a **durable** set of the decisions it has ordered, so a
   decision is not re-ordered after a retry or a restart.

No test reaches the network (the suite's autouse fixture refuses it); every broker
call goes through a mocked transport, and no test needs a real key.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    STATUS_RUNNING,
    STATUS_STOPPED,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingWorker,
)
from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.paper_trading import ExperimentDecisionSource
from tradingagents.paper_experiment.sources import StaticPriceSource

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


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _order(client_order_id="c1", **over):
    payload = {
        "id": f"broker-{client_order_id}",
        "client_order_id": client_order_id,
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


def _transport(calls):
    def send(*, method, url, headers, params, json_body, timeout):
        calls.append({"method": method, "url": url, "json": json_body})
        if method == "GET" and url.endswith("/v2/account"):
            return _Resp(200, _ACCOUNT)
        if method == "GET" and url.endswith("/v2/positions"):
            return _Resp(200, [])
        if method == "POST" and url.endswith("/v2/orders"):
            return _Resp(200, _order(client_order_id=(json_body or {}).get("client_order_id", "c1")))
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _posts(calls):
    return [c for c in calls if c["method"] == "POST"]


def _posted_client_order_ids(calls):
    return [c["json"]["client_order_id"] for c in _posts(calls)]


def _experiment(root, *, rating="Buy", sessions=6):
    """A loadable experiment directory over a static 6-session source."""
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=sessions, freq="B")
    opens = [10.0 + i for i in range(sessions)]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [o + 1 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 0.5 for o in opens],
            "volume": [1e6] * sessions,
        },
        index=index,
    )
    source = StaticPriceSource({"SPY": frame}, name="static_test")
    config = ExperimentConfig.from_protocol(
        symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=1
    )
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference(rating=rating)
    )
    experiment.ensure_created()
    # The controller and the source reload from disk, so the initial state must be
    # written before either reads it.
    experiment.save_state()
    return experiment


def _sessions(experiment):
    return list(experiment.source.sessions())


def _controller(root, calls, **kwargs):
    return PaperTradingController(
        root,
        enabled=True,
        order_notional=25.0,
        broker_factory=lambda: AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(calls),
            limits=PaperRiskLimits(),
        ),
        **kwargs,
    )


_WORKERS: list[PaperTradingWorker] = []


@pytest.fixture(autouse=True)
def _stop_controllers():
    start = len(_WORKERS)
    yield
    for worker in _WORKERS[start:]:
        with contextlib.suppress(Exception):
            worker.stop()
    del _WORKERS[start:]


def _run(controller):
    """Start (builds the worker), run one cycle, stop. Returns the cycle dict."""
    controller.start()
    try:
        _WORKERS.append(controller._worker)  # noqa: SLF001 - teardown seam
        return controller.run_once()
    finally:
        controller.stop()


# ---------------------------------------------------------------------------
# 1. The source discovers decisions produced after it was built
# ---------------------------------------------------------------------------


class TestDecisionSourceFreshness:
    def test_a_source_built_before_any_decision_finds_one_produced_later(self, tmp_path):
        experiment = _experiment(tmp_path / "exp")
        reload = lambda: PaperExperiment.load(experiment.root, inference=None)  # noqa: E731
        source = ExperimentDecisionSource(experiment, order_notional=25.0, reload=reload)

        # Nothing has been decided yet.
        assert source.next_decision() is None

        # A later writer records a decision (this is what auto-advance does).
        experiment.process_session(_sessions(experiment)[0])

        decision = source.next_decision()
        assert decision is not None
        assert decision.side == "buy"
        assert decision.decision_id == "SPY-2024-01-01"

    def test_a_source_yields_each_new_decision_exactly_once(self, tmp_path):
        experiment = _experiment(tmp_path / "exp")
        reload = lambda: PaperExperiment.load(experiment.root, inference=None)  # noqa: E731
        source = ExperimentDecisionSource(experiment, order_notional=25.0, reload=reload)
        sessions = _sessions(experiment)

        experiment.process_session(sessions[0])
        first = source.next_decision()
        assert first is not None
        # Re-reading does not hand the same decision out twice.
        assert source.next_decision() is None

        experiment.process_session(sessions[1])
        second = source.next_decision()
        assert second is not None
        assert second.decision_id != first.decision_id

    def test_a_hold_decision_is_never_yielded(self, tmp_path):
        experiment = _experiment(tmp_path / "exp", rating="Hold")
        for session in _sessions(experiment):
            experiment.process_session(session)
        source = ExperimentDecisionSource(experiment, order_notional=25.0)
        assert source.next_decision() is None

    def test_a_blocked_decision_is_never_yielded(self, tmp_path):
        experiment = _experiment(tmp_path / "exp")
        experiment.inference = None  # every decision is blocked
        for session in _sessions(experiment):
            experiment.process_session(session)
        source = ExperimentDecisionSource(experiment, order_notional=25.0)
        assert source.next_decision() is None

    def test_settled_history_is_not_replayed_on_build(self, tmp_path):
        """A worker built on a long run offers only the still-pending tail.

        Every decision already settled (executed/evaluated) when the source is
        built is history and is never offered; a decision still awaiting execution
        is, because it is exactly what the worker exists to act on.

        The run stops one session short of the end: a decision taken on the newest
        bar has no T+1 yet, so it is unresolved and correctly withheld. Processing
        all but the last session leaves one decision whose successor bar exists, so
        it is genuinely awaiting execution and must be offered.
        """
        experiment = _experiment(tmp_path / "exp")
        sessions = _sessions(experiment)
        for session in sessions[:-1]:
            experiment.process_session(session)

        settled = {
            d.decision_id for d in experiment.decisions.values() if d.state != "pending"
        }
        assert settled  # the premise: there is history to suppress

        source = ExperimentDecisionSource(experiment, order_notional=25.0)
        offered = []
        while (decision := source.next_decision()) is not None:
            offered.append(decision.decision_id)

        # No settled decision is offered...
        assert not (set(offered) & settled)
        # ...and the one still-pending decision is.
        pending = {d.decision_id for d in experiment.decisions.values() if d.state == "pending"}
        assert pending  # the premise: there is a live tail to offer
        assert set(offered) == pending

    def test_a_reload_failure_falls_back_without_duplicating(self, tmp_path):
        experiment = _experiment(tmp_path / "exp")
        experiment.process_session(_sessions(experiment)[0])

        def boom():
            raise RuntimeError("the state file is momentarily unreadable")

        source = ExperimentDecisionSource(experiment, order_notional=25.0, reload=boom)
        first = source.next_decision()
        assert first is not None
        # The fallback must not re-offer the same decision.
        assert source.next_decision() is None


# ---------------------------------------------------------------------------
# 2. status() never builds the worker
# ---------------------------------------------------------------------------


class TestStatusDoesNotBuildWorker:
    def test_status_before_start_leaves_the_worker_unbuilt(self, tmp_path):
        experiment = _experiment(tmp_path / "exp")
        builds = []

        def factory():
            builds.append(1)
            return AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport([]),
                limits=PaperRiskLimits(),
            )

        controller = PaperTradingController(
            experiment.root, enabled=True, order_notional=25.0, broker_factory=factory
        )
        for _ in range(5):
            status = controller.status()
            assert status["status"] == STATUS_STOPPED
        assert controller._worker is None  # noqa: SLF001 - the seam under test
        assert builds == []  # the broker was never constructed either

    def test_status_then_start_then_a_decision_is_delivered(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        controller = _controller(experiment.root, calls)

        # Polling status several times before any decision must not freeze anything.
        for _ in range(5):
            assert controller.status()["status"] == STATUS_STOPPED
        assert controller._worker is None  # noqa: SLF001

        # Now a decision exists; START discovers and orders it.
        experiment.process_session(_sessions(experiment)[0])
        cycle = _run(controller)
        assert cycle["ran"] is True
        assert cycle["decision_id"] == "SPY-2024-01-01"
        assert len(_posts(calls)) == 1


# ---------------------------------------------------------------------------
# 3. Auto-advance racing the worker does not lose the decision
# ---------------------------------------------------------------------------


class TestAutoAdvanceRace:
    def test_a_decision_settled_after_the_build_is_still_ordered(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        sessions = _sessions(experiment)
        controller = _controller(experiment.root, calls)

        # A decision exists and is pending when the worker is built.
        experiment.process_session(sessions[0])
        controller.start()
        _WORKERS.append(controller._worker)  # noqa: SLF001
        try:
            # Auto-advance races ahead and settles it internally before the cycle.
            experiment.process_session(sessions[1])
            assert experiment.decisions["SPY-2024-01-01"].state != "pending"

            cycle = controller.run_once()
            assert cycle["ran"] is True
            assert cycle["decision_id"] == "SPY-2024-01-01"
        finally:
            controller.stop()
        assert len(_posts(calls)) == 1

    def test_a_decision_produced_after_several_cycles_is_delivered(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        sessions = _sessions(experiment)
        controller = _controller(experiment.root, calls)
        controller.start()
        _WORKERS.append(controller._worker)  # noqa: SLF001
        try:
            # Several empty cycles first: no decision yet.
            for _ in range(3):
                assert controller.run_once()["ran"] is False
            assert _posts(calls) == []

            # Now decisions appear and are picked up.
            experiment.process_session(sessions[0])
            first = controller.run_once()
            assert first["ran"] is True
            experiment.process_session(sessions[1])
            second = controller.run_once()
            assert second["ran"] is True
            assert second["decision_id"] != first["decision_id"]
        finally:
            controller.stop()
        assert len(_posts(calls)) == 2


# ---------------------------------------------------------------------------
# 4. No duplicate order on retry or restart
# ---------------------------------------------------------------------------


class TestNoDuplicateOrders:
    def test_a_retry_does_not_re_order_the_same_decision(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        experiment.process_session(_sessions(experiment)[0])
        controller = _controller(experiment.root, calls)
        controller.start()
        _WORKERS.append(controller._worker)  # noqa: SLF001
        try:
            first = controller.run_once()
            assert first["ran"] is True
            # Repeated cycles over the same state place no second order.
            for _ in range(3):
                assert controller.run_once()["ran"] is False
        finally:
            controller.stop()
        assert len(_posts(calls)) == 1

    def test_a_restart_does_not_re_order_a_processed_decision(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        experiment.process_session(_sessions(experiment)[0])

        first = _controller(experiment.root, calls)
        cycle = _run(first)
        assert cycle["ran"] is True
        assert len(_posts(calls)) == 1

        # A fresh controller over the same directory: new worker, same state file.
        posts_before = len(_posts(calls))
        second = _controller(experiment.root, calls)
        cycle2 = _run(second)
        assert cycle2["ran"] is False
        assert len(_posts(calls)) == posts_before  # nothing re-submitted

    def test_the_processed_set_is_persisted(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        experiment.process_session(_sessions(experiment)[0])
        controller = _controller(experiment.root, calls)
        _run(controller)
        blob = (experiment.root / "worker_state.json").read_text()
        assert "SPY-2024-01-01" in blob


# ---------------------------------------------------------------------------
# 5. A refusal is not re-attempted on every cycle (bounded retry)
# ---------------------------------------------------------------------------


class TestRefusalIsBounded:
    def test_a_refused_decision_is_not_re_submitted_every_cycle(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        experiment.process_session(_sessions(experiment)[0])

        def factory():
            return AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport(calls),
                # A notional cap below the order size forces a local refusal.
                limits=PaperRiskLimits(max_order_notional=0.5),
            )

        controller = PaperTradingController(
            experiment.root, enabled=True, order_notional=25.0, broker_factory=factory
        )
        controller.start()
        _WORKERS.append(controller._worker)  # noqa: SLF001
        try:
            first = controller.run_once()
            assert first["ran"] is False
            assert "refused" in (first["skipped_reason"] or "")
            # The refusal is remembered durably...
            assert "SPY-2024-01-01" in controller._worker.state.processed_decisions  # noqa: SLF001
            # ...so the next cycle does not re-attempt the same decision.
            second = controller.run_once()
            assert second["ran"] is False
        finally:
            controller.stop()
        assert _posts(calls) == []  # never reached the broker


# ---------------------------------------------------------------------------
# 6. The worker still refuses to run unless RUNNING
# ---------------------------------------------------------------------------


class TestSafetyUnchanged:
    def test_a_paper_only_broker_is_still_built(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        controller = _controller(experiment.root, calls)
        controller.start()
        _WORKERS.append(controller._worker)  # noqa: SLF001
        try:
            status = controller.status()
            assert status["mode"] == "ALPACA_PAPER"
            assert status["live"] is False
            assert status["paper_endpoint_only"] is True
            assert controller._worker.state.status == STATUS_RUNNING  # noqa: SLF001
        finally:
            controller.stop()

    def test_the_status_view_carries_no_credential(self, tmp_path):
        calls = []
        experiment = _experiment(tmp_path / "exp")
        controller = _controller(experiment.root, calls)
        blob = json.dumps(controller.status())
        assert "paper-key" not in blob
        assert "paper-secret" not in blob
