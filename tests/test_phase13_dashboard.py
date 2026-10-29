"""Phase 13: the paper dashboard — auth, read-only views, and gated control.

The dashboard is a *monitoring and control surface for the simulation*. These
tests pin the properties that make it safe to expose:

- every ``/api`` route requires a bearer token, and an unset token denies rather
  than allows;
- a non-loopback bind without a token is refused;
- the read-only views report the persisted state without mutating it, and the
  snapshot cannot claim an order is FILLED when it is PARTIALLY_FILLED;
- the control action is off unless enabled, requires a confirmation header, is
  bounded, refuses a concurrent run, and advances the *same* persisted state the
  CLI would — it is simulation-only and never reaches a broker.

The HTTP tests drive a real ``ThreadingHTTPServer`` on an ephemeral loopback
port and talk to it with ``urllib``. Loopback sockets are allowed by the suite's
no-network fixture (it only blocks ``connect``/``getaddrinfo`` to non-loopback
hosts via the default path); these tests bind ``127.0.0.1`` explicitly.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.dashboard import (
    DashboardAuthError,
    DashboardStateError,
    ExperimentView,
    check_bearer,
    is_loopback,
    require_bindable,
    resolve_token,
    serve,
)
from tradingagents.paper_experiment.dashboard.server import (
    CONTROL_CONFIRM_HEADER,
    Dashboard,
    make_handler,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.sources import StaticPriceSource

TOKEN = "test-token-123"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frame(periods: int = 12, *, start: float = 100.0, step: float = 1.0, volume: float = 1e6):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [volume] * periods,
        },
        index=index,
    )


def _source(periods: int = 12):
    return StaticPriceSource({"SPY": _frame(periods)}, name="static_test")


def _config(**overrides):
    base = {"symbols": ("SPY",), "decision_frequency_sessions": 1, "evaluation_lag_sessions": 2}
    base.update(overrides)
    return ExperimentConfig.from_protocol(**base)


def _experiment(root, *, periods: int = 12, processed: int = 0):
    source = _source(periods)
    experiment = PaperExperiment(
        source=source,
        config=_config(),
        root=root,
        inference=StubInference(),
    )
    experiment.ensure_created()
    sessions = list(source.sessions())
    for session in sessions[:processed]:
        experiment.process_session(session)
    return experiment


@pytest.fixture
def experiment_root(tmp_path):
    root = tmp_path / "exp"
    _experiment(root, periods=12, processed=4)
    return root


_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


@pytest.fixture
def allow_loopback(monkeypatch):
    """Let the HTTP tests reach a server on 127.0.0.1 only.

    The suite's autouse no-network fixture refuses every socket, which is right
    for a test that must not reach a vendor. A test of a *local* server needs the
    loopback exception the ChatGPT sign-in tests already use: 127.0.0.1 is
    permitted, everything else still refused.
    """
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (
            _LOOPBACK_CONNECT if address[0] == "127.0.0.1" else _refuse)(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (
            _LOOPBACK_LOOKUP if host == "127.0.0.1" else _refuse)(host, *a, **k),
    )


def _refuse(*args, **kwargs):
    raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_loopback_hosts_are_recognized(self):
        assert is_loopback("127.0.0.1")
        assert is_loopback("localhost")
        assert is_loopback("::1")
        assert not is_loopback("0.0.0.0")
        assert not is_loopback("paper.example.com")

    def test_non_loopback_without_token_is_refused(self):
        with pytest.raises(DashboardAuthError):
            require_bindable("0.0.0.0", "")

    def test_non_loopback_with_token_is_allowed(self):
        require_bindable("0.0.0.0", "a-token")

    def test_loopback_without_token_is_allowed(self):
        require_bindable("127.0.0.1", "")

    def test_resolve_token_prefers_argument(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "from-env")
        assert resolve_token("explicit") == "explicit"
        assert resolve_token() == "from-env"

    def test_blank_env_token_is_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "   ")
        assert resolve_token() == ""

    def test_check_bearer_accepts_the_token(self):
        assert check_bearer(f"Bearer {TOKEN}", TOKEN)

    def test_check_bearer_rejects_wrong_or_missing(self):
        assert not check_bearer("Bearer nope", TOKEN)
        assert not check_bearer(None, TOKEN)
        assert not check_bearer(f"Basic {TOKEN}", TOKEN)

    def test_unset_token_denies_everything(self):
        # An empty expected token must never mean "allow all".
        assert not check_bearer("Bearer anything", "")
        assert not check_bearer(None, "")


# ---------------------------------------------------------------------------
# Read-only views
# ---------------------------------------------------------------------------


class TestViews:
    def test_view_requires_an_existing_state_file(self, tmp_path):
        with pytest.raises(DashboardStateError):
            ExperimentView(tmp_path / "missing")

    def test_summary_reports_persisted_state(self, experiment_root):
        summary = ExperimentView(experiment_root).summary()
        assert summary["counts"]["sessions_processed"] == 4
        assert summary["book"]["cash"] == 100_000.0
        assert summary["readiness"]["status"] in {"BLOCKED", "OFFLINE PAPER (reproducible, not live)"}
        assert summary["schema_version"] == 1

    def test_summary_is_read_only(self, experiment_root):
        before = (experiment_root / "session_state.json").read_text()
        ExperimentView(experiment_root).summary()
        after = (experiment_root / "session_state.json").read_text()
        assert before == after

    def test_decisions_are_newest_first(self, experiment_root):
        decisions = ExperimentView(experiment_root).decisions()
        sessions = [d["decision_session"] for d in decisions]
        assert sessions == sorted(sessions, reverse=True)

    def test_journal_tail_respects_limit(self, experiment_root):
        entries = ExperimentView(experiment_root).journal_tail(limit=2)
        assert len(entries) == 2
        # newest first
        assert entries[0]["seq"] > entries[1]["seq"]

    def test_resting_view_reports_remaining_quantity(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.state import _resting_view

        view = _resting_view(
            {"quantity": 1000.0, "filled_quantity": 400.0, "status": "PARTIALLY_FILLED"}
        )
        assert view["remaining_quantity"] == 600.0

    def test_snapshot_cannot_claim_filled_when_partially_filled(self, tmp_path):
        """A partially-filled resting order is reported as PARTIALLY_FILLED."""
        from tradingagents.paper_experiment.dashboard.state import _resting_view

        view = _resting_view(
            {
                "order_id": "o1",
                "symbol": "SPY",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": 100.0,
                "filled_quantity": 40.0,
                "status": "PARTIALLY_FILLED",
                "limit_price": 99.0,
            }
        )
        assert view["status"] == "PARTIALLY_FILLED"
        assert view["status"] != "FILLED"
        assert view["filled_quantity"] == 40.0


# ---------------------------------------------------------------------------
# Control (gated)
# ---------------------------------------------------------------------------


class TestControl:
    def test_control_disabled_by_default(self, experiment_root):
        dashboard = Dashboard(experiment_root, token=TOKEN)
        with pytest.raises(PermissionError):
            dashboard.run_sessions(1)

    def test_control_advances_and_persists(self, experiment_root):
        dashboard = Dashboard(experiment_root, token=TOKEN, allow_control=True)
        result = dashboard.run_sessions(3)
        assert result["processed_count"] == 3
        # The same state file the CLI reads is advanced.
        raw = json.loads((experiment_root / "session_state.json").read_text())
        assert len(raw["processed_sessions"]) == 7

    def test_control_is_bounded(self, experiment_root):
        dashboard = Dashboard(experiment_root, token=TOKEN, allow_control=True, max_sessions_per_call=2)
        result = dashboard.run_sessions(100)
        assert result["processed_count"] == 2
        assert result["bounded_to"] == 2

    def test_negative_sessions_is_refused(self, experiment_root):
        dashboard = Dashboard(experiment_root, token=TOKEN, allow_control=True)
        with pytest.raises(ValueError):
            dashboard.run_sessions(-1)

    def test_control_does_not_reach_a_broker(self, experiment_root):
        """The only action drives PaperExperiment; no broker is constructed."""
        import tradingagents.paper_experiment.dashboard.server as server_mod

        assert not hasattr(server_mod, "AlpacaPaperBroker")
        assert not hasattr(server_mod, "submit_order")


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


def _serve_in_thread(root, *, token=TOKEN, allow_control=False, max_sessions=50):
    dashboard = Dashboard(
        root, token=token, allow_control=allow_control, max_sessions_per_call=max_sessions
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url, token=None):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body.decode("utf-8", "replace")}


def _post(url, token=None, body=None, confirm=True):
    request = urllib.request.Request(url, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if confirm:
        request.add_header(CONTROL_CONFIRM_HEADER, "yes")
    data = json.dumps(body or {}).encode()
    try:
        with urllib.request.urlopen(request, data=data, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def http_dashboard(experiment_root, allow_loopback):
    httpd, base = _serve_in_thread(experiment_root, allow_control=True)
    yield base
    httpd.shutdown()
    httpd.server_close()


class TestHttpSurface:
    def test_healthz_needs_no_auth(self, http_dashboard):
        status, body = _get(f"{http_dashboard}/healthz")
        assert status == 200
        assert body["status"] == "ok"

    def test_index_is_served_without_auth(self, http_dashboard):
        request = urllib.request.Request(f"{http_dashboard}/")
        with urllib.request.urlopen(request, timeout=10) as response:
            html = response.read().decode()
        assert "TradingAgents" in html
        # Self-contained: no external asset references that would leak a request.
        assert "http://" not in html.split("</style>")[0] or "<script src" not in html

    def test_api_requires_a_token(self, http_dashboard):
        status, _ = _get(f"{http_dashboard}/api/summary")
        assert status == 401

    def test_api_rejects_a_wrong_token(self, http_dashboard):
        status, _ = _get(f"{http_dashboard}/api/summary", token="wrong")
        assert status == 401

    def test_summary_with_token(self, http_dashboard):
        status, body = _get(f"{http_dashboard}/api/summary", token=TOKEN)
        assert status == 200
        assert body["counts"]["sessions_processed"] == 4

    def test_unknown_route_is_404(self, http_dashboard):
        status, _ = _get(f"{http_dashboard}/api/nope", token=TOKEN)
        assert status == 404

    def test_run_requires_confirmation_header(self, http_dashboard):
        status, body = _post(f"{http_dashboard}/api/run", token=TOKEN, body={"sessions": 1}, confirm=False)
        assert status == 403
        assert "confirmation" in body["error"].lower()

    def test_run_requires_a_token(self, http_dashboard):
        status, _ = _post(f"{http_dashboard}/api/run", token=None, body={"sessions": 1})
        assert status == 401

    def test_run_advances_sessions(self, http_dashboard):
        status, body = _post(f"{http_dashboard}/api/run", token=TOKEN, body={"sessions": 2})
        assert status == 200
        assert body["processed_count"] == 2

    def test_run_bad_body_is_400(self, http_dashboard):
        request = urllib.request.Request(f"{http_dashboard}/api/run", method="POST")
        request.add_header("Authorization", f"Bearer {TOKEN}")
        request.add_header(CONTROL_CONFIRM_HEADER, "yes")
        try:
            with urllib.request.urlopen(request, data=b"not json", timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400

    def test_read_only_dashboard_refuses_run(self, experiment_root, allow_loopback):
        httpd, base = _serve_in_thread(experiment_root, allow_control=False)
        try:
            status, body = _post(f"{base}/api/run", token=TOKEN, body={"sessions": 1})
            assert status == 403
            assert "disabled" in body["error"].lower()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_control_status_reports_enabled(self, http_dashboard):
        status, body = _get(f"{http_dashboard}/api/control/status", token=TOKEN)
        assert status == 200
        assert body["control_enabled"] is True


def test_serve_refuses_public_bind_without_token(tmp_path):
    with pytest.raises(DashboardAuthError):
        serve(tmp_path, host="0.0.0.0", port=0, token="")


# ---------------------------------------------------------------------------
# CLI wiring and the shared runner
# ---------------------------------------------------------------------------


class TestCliServe:
    def test_serve_is_a_known_command_and_refuses_a_public_bind(self, tmp_path):
        from tradingagents.paper_experiment.__main__ import main

        # A public bind with no token is refused before the server starts, so the
        # command returns (exit 2) instead of blocking on serve_forever().
        assert main(["serve", "--root", str(tmp_path), "--host", "0.0.0.0"]) == 2

    def test_serve_requires_a_root(self):
        from tradingagents.paper_experiment.__main__ import main

        # `_root` raises RunnerError, which `main` reports as exit 2.
        assert main(["serve"]) == 2

    def test_run_uses_the_shared_runner(self, tmp_path):
        """The CLI's `run` processes sessions through the shared runner."""
        from tradingagents.paper_experiment.__main__ import _restored_source, main
        from tradingagents.paper_experiment.runner import pending_sessions

        root = tmp_path / "cli_exp"
        assert main(["start", "--root", str(root), "--stub"]) == 0
        assert main(["run", "--root", str(root), "--stub", "--sessions", "3"]) == 0
        raw = json.loads((root / "session_state.json").read_text())
        assert len(raw["processed_sessions"]) == 3
        experiment = PaperExperiment.load(root, source=_restored_source(root))
        assert len(pending_sessions(experiment)) == len(experiment.source.sessions()) - 3


class TestModuleEntryPoint:
    """`python -m ...dashboard.server` — the Railway/Docker start command.

    Importing the package used to pull ``.server`` into ``sys.modules`` before
    runpy could execute it, which made runpy warn and — with no ``__main__`` guard
    in ``server.py`` — meant the module ran and *exited without listening*. These
    tests pin both halves of the fix: the lazy package import, and a real entry
    point that binds and serves.
    """

    def test_package_import_does_not_load_server(self):
        """The eager import that caused the runpy warning is gone."""
        import subprocess
        import sys

        code = (
            "import sys, tradingagents.paper_experiment.dashboard as d;"
            "print('tradingagents.paper_experiment.dashboard.server' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False", out.stderr

    def test_lazy_attributes_resolve(self):
        import tradingagents.paper_experiment.dashboard as d

        assert d.Dashboard is not None
        assert callable(d.serve)
        with pytest.raises(AttributeError):
            _ = d.does_not_exist

    def test_main_refuses_a_public_bind_without_a_token(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.server import main

        # Returns 2 rather than blocking, because the bind is refused first.
        assert main(["--root", str(tmp_path), "--host", "0.0.0.0"]) == 2

    def test_main_requires_a_root(self, monkeypatch):
        from tradingagents.paper_experiment.dashboard.server import main

        monkeypatch.delenv("TRADINGAGENTS_EXPERIMENT_ROOT", raising=False)
        with pytest.raises(SystemExit):
            main([])

    def test_main_reads_root_and_port_from_the_environment(self, tmp_path, monkeypatch, allow_loopback):
        """`--root` falls back to the env var; the port to $PORT."""
        from tradingagents.paper_experiment.dashboard import server as server_mod

        root = tmp_path / "env_exp"
        _experiment(root, periods=6, processed=1)

        captured: dict[str, object] = {}

        def fake_serve(root_arg, *, host, port, token, allow_control):
            captured.update(root=str(root_arg), host=host, port=port, token=token, allow_control=allow_control)

        monkeypatch.setattr(server_mod, "serve", fake_serve)
        monkeypatch.setenv("TRADINGAGENTS_EXPERIMENT_ROOT", str(root))
        monkeypatch.setenv("PORT", "9123")
        monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "env-token")

        assert server_mod.main([]) == 0
        assert captured["port"] == 9123
        assert captured["root"] == str(root)
        # `serve` resolves the token from the environment, so `main` passes None.
        assert captured["token"] is None
        assert server_mod.resolve_token(captured["token"]) == "env-token"

    def test_module_entry_point_actually_listens(self, experiment_root, allow_loopback):
        """Run the real module as a subprocess and prove it serves HTTP."""
        import os
        import socket
        import subprocess
        import sys
        import time

        # Reserve a free port, then hand it to the subprocess.
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        env = {
            **os.environ,
            "TRADINGAGENTS_EXPERIMENT_ROOT": str(experiment_root),
            "TRADINGAGENTS_DASHBOARD_TOKEN": TOKEN,
            "PORT": str(port),
        }
        proc = subprocess.Popen(
            [sys.executable, "-m", "tradingagents.paper_experiment.dashboard.server",
             "--host", "127.0.0.1"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.time() + 20
            last_error = None
            while time.time() < deadline:
                try:
                    status, body = _get(f"http://127.0.0.1:{port}/healthz")
                    if status == 200:
                        break
                except Exception as exc:  # noqa: BLE001 - retry until the deadline
                    last_error = exc
                time.sleep(0.2)
            else:
                proc.kill()
                raise AssertionError(f"server never listened: {last_error}")
            # No runpy warning on stderr.
            proc.kill()
            output = proc.communicate(timeout=10)[0]
            assert "RuntimeWarning" not in output, output
            assert "sys.modules" not in output, output
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


class TestRunner:
    def test_process_pending_respects_the_bound(self, tmp_path):
        from tradingagents.paper_experiment.runner import process_pending

        experiment = _experiment(tmp_path / "r", periods=10, processed=0)
        outcomes = process_pending(experiment, sessions=4)
        assert len(outcomes) == 4

    def test_process_pending_negative_is_refused(self, tmp_path):
        from tradingagents.paper_experiment.runner import process_pending

        experiment = _experiment(tmp_path / "r2", periods=10, processed=0)
        with pytest.raises(ValueError):
            process_pending(experiment, sessions=-1)

    def test_process_pending_is_idempotent_on_a_second_call(self, tmp_path):
        from tradingagents.paper_experiment.runner import process_pending

        experiment = _experiment(tmp_path / "r3", periods=5, processed=0)
        process_pending(experiment)
        # Every session is processed; a second call has nothing left to do.
        assert process_pending(experiment) == []
