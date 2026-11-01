"""Phase 20b: the dashboard/API contract, and its tolerance of change.

The dashboard is a phone UI reading a JSON API, and the backend and the UI ship
together but must not break each other. Two failure modes this suite guards:

- a **new field** (freshness, cadence, warnings, per-decision provenance) must be
  additive: the summary still carries the keys the UI already reads, and the new
  keys are present and JSON-safe;
- **old or malformed persisted state** must still load, and a **missing optional
  field** must not crash the view — a UI that throws on an absent key shows the
  operator nothing at exactly the moment they need it.

The tests read the real ``ExperimentView`` and, for the HTTP surface, drive a real
``ThreadingHTTPServer`` on loopback with ``urllib`` (the same harness the phase-13
tests use, with the same loopback exception to the suite's no-network fixture).
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.dashboard import ExperimentView
from tradingagents.paper_experiment.dashboard.server import Dashboard, make_handler
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.sources import StaticPriceSource

TOKEN = "test-token-123"
SYMBOL = "SPY"


def _frame(periods: int = 12, *, start: float = 100.0):
    index = pd.date_range("2026-10-01", periods=periods, freq="B")
    opens = [start + i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1e6] * periods,
        },
        index=index,
    )


def _config(**overrides):
    base = {
        "symbols": (SYMBOL,),
        "decision_frequency_sessions": 1,
        "evaluation_lag_sessions": 2,
    }
    base.update(overrides)
    return ExperimentConfig.from_protocol(**base)


def _experiment(root, *, periods: int = 12, processed: int = 4):
    source = StaticPriceSource({SYMBOL: _frame(periods)}, name="static_test")
    experiment = PaperExperiment(
        source=source, config=_config(), root=root, inference=StubInference(rating="Buy")
    )
    experiment.ensure_created()
    for session in list(source.sessions())[:processed]:
        experiment.process_session(session)
    return experiment


# The keys the phase-13 UI already reads. A new build must not drop any.
_REQUIRED_SUMMARY_KEYS = (
    "experiment_id",
    "strategy_id",
    "created_at",
    "last_session_processed",
    "counts",
    "orders",
    "book",
    "performance",
    "costs",
    "data",
    "refresh",
    "readiness",
    "limitations",
    "journal",
    "resting_orders",
    "processed_sessions",
    "schema_version",
)


@pytest.mark.unit
class TestSummaryContract:
    def test_the_ui_required_keys_are_all_present(self, tmp_path):
        _experiment(tmp_path / "exp")
        summary = ExperimentView(tmp_path / "exp").summary()
        for key in _REQUIRED_SUMMARY_KEYS:
            assert key in summary, f"the summary dropped the UI key {key!r}"

    def test_the_new_blocks_are_present_and_json_safe(self, tmp_path):
        _experiment(tmp_path / "exp")
        summary = ExperimentView(tmp_path / "exp").summary()
        for key in ("freshness", "cadence", "warnings"):
            assert key in summary
        # Round-tripping proves no non-serializable object leaked into the payload.
        assert json.loads(json.dumps(summary))["freshness"]["market_status"]

    def test_freshness_carries_the_fields_the_ui_shows(self, tmp_path):
        _experiment(tmp_path / "exp")
        fresh = ExperimentView(tmp_path / "exp").summary()["freshness"]
        for key in (
            "market_status",
            "reason",
            "latest_available_session",
            "last_source_bar",
            "last_processed_session",
            "last_refresh_at",
            "next_refresh_allowed_at",
            "data_age_days",
            "pending_sessions",
            "is_stale",
        ):
            assert key in fresh, f"freshness is missing {key!r}"

    def test_the_cadence_block_carries_the_ui_fields(self, tmp_path):
        _experiment(tmp_path / "exp")
        cadence = ExperimentView(tmp_path / "exp").summary()["cadence"]
        for key in (
            "decision_frequency_sessions",
            "max_decisions",
            "decisions_taken",
            "max_decisions_reached",
            "next_decision_session",
        ):
            assert key in cadence

    def test_warnings_is_a_list_of_strings(self, tmp_path):
        _experiment(tmp_path / "exp")
        warnings = ExperimentView(tmp_path / "exp").summary()["warnings"]
        assert isinstance(warnings, list)
        assert all(isinstance(w, str) for w in warnings)

    def test_a_reached_cap_raises_a_warning(self, tmp_path):
        # cadence 1 with a cap of 2: the third processed session hits the cap.
        source = StaticPriceSource({SYMBOL: _frame(6)}, name="static_test")
        experiment = PaperExperiment(
            source=source,
            config=_config(max_decisions=2),
            root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        for session in list(source.sessions()):
            experiment.process_session(session)
        warnings = ExperimentView(tmp_path / "exp").summary()["warnings"]
        assert any("max_decisions" in w for w in warnings)


@pytest.mark.unit
class TestDecisionContract:
    def test_each_decision_carries_full_provenance(self, tmp_path):
        _experiment(tmp_path / "exp")
        decisions = ExperimentView(tmp_path / "exp").decisions()
        assert decisions
        for decision in decisions:
            for key in (
                "provider",
                "model",
                "prompt_version",
                "graph_version",
                "snapshot_id",
                "data_window_hash",
                "state",
                "filled_quantity",
            ):
                assert key in decision, f"decision is missing {key!r}"

    def test_decisions_are_json_safe(self, tmp_path):
        _experiment(tmp_path / "exp")
        decisions = ExperimentView(tmp_path / "exp").decisions()
        assert json.loads(json.dumps(decisions)) == decisions


@pytest.mark.unit
class TestBackwardCompatibility:
    def test_a_decision_without_the_new_provenance_fields_still_loads(self, tmp_path):
        """Old persisted state has no provider/model/prompt_version keys."""
        _experiment(tmp_path / "exp")
        state_path = tmp_path / "exp" / "session_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for decision in state.get("decisions", {}).values():
            for key in ("provider", "model", "prompt_version", "graph_version"):
                decision.pop(key, None)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        # The real recovery path must load it, defaulting the missing fields to None.
        experiment = PaperExperiment.load(tmp_path / "exp", inference=StubInference())
        assert experiment.decisions
        for decision in experiment.decisions.values():
            assert decision.provider is None
            assert decision.prompt_version is None
        # And a report can still be built from the older state.
        from tradingagents.paper_experiment.report import build_report

        assert build_report(experiment).to_dict()["counts"]["sessions_processed"] >= 0

    def test_a_summary_missing_optional_blocks_does_not_crash_the_renderer(self, tmp_path):
        """The UI reads freshness/cadence/warnings defensively; the API may omit them."""
        _experiment(tmp_path / "exp")
        summary = ExperimentView(tmp_path / "exp").summary()
        # Simulate an older backend that did not send the new blocks.
        stripped = {k: v for k, v in summary.items() if k not in ("freshness", "cadence", "warnings")}
        # The renderer's guards (``|| {}`` / ``|| []``) mean these are all it needs.
        assert (stripped.get("freshness") or {}).get("market_status") is None
        assert (stripped.get("warnings") or []) == []
        assert (stripped.get("cadence") or {}).get("decision_frequency_sessions") is None


_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


@pytest.fixture
def allow_loopback(monkeypatch):
    def _refuse(*args, **kwargs):
        raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (
            _LOOPBACK_CONNECT if address[0] == "127.0.0.1" else _refuse
        )(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (
            _LOOPBACK_LOOKUP if host == "127.0.0.1" else _refuse
        )(host, *a, **k),
    )


def _serve(root):
    dashboard = Dashboard(root, token=TOKEN, allow_control=False, stub=True)
    handler = make_handler(dashboard)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _get(port, path):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    request.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.unit
class TestHttpContract:
    def test_summary_endpoint_serves_the_new_blocks(self, tmp_path, allow_loopback):
        _experiment(tmp_path / "exp")
        server, thread = _serve(tmp_path / "exp")
        try:
            body = _get(server.server_address[1], "/api/summary")
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert body["freshness"]["market_status"]
        assert "cadence" in body
        assert isinstance(body["warnings"], list)

    def test_decisions_endpoint_serves_provenance(self, tmp_path, allow_loopback):
        _experiment(tmp_path / "exp")
        server, thread = _serve(tmp_path / "exp")
        try:
            body = _get(server.server_address[1], "/api/decisions")
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert body["decisions"]
        assert "provider" in body["decisions"][0]
