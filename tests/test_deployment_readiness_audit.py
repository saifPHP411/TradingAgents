"""Phase 2.4 deployment-readiness audit: the deployment must say what it does.

These tests pin the properties a deployment document and a deployment config
must have for an operator to trust them:

1. the deployment files describe **one** service — the two start commands and
   the two health checks agree;
2. the start command is safe: ``$PORT`` is expanded by a shell, ``exec`` hands
   the process the platform's signals, and the process is the dashboard (the
   only program that serves ``/healthz``);
3. the deployment does not claim to be simulation-only — the paper worker really
   does place paper orders — and it does not claim to be live;
4. the documentation states plainly what was **not** verified.

Every test is offline and reads the repository's own files.
"""

from __future__ import annotations

import json
import re
import socket
import tempfile
from pathlib import Path

import pytest

_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


@pytest.fixture
def allow_loopback(monkeypatch):
    """Permit loopback sockets only; every other destination is refused.

    The suite's autouse guard blocks the network, so a test that binds a
    loopback server needs this narrow exception — and only for 127.0.0.1.
    """

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
            _LOOPBACK_LOOKUP if host in ("127.0.0.1", "localhost") else _refuse
        )(host, *a, **k),
    )

REPO = Path(__file__).resolve().parents[1]
RAILWAY_JSON = REPO / "railway.json"
RAILWAY_TS = REPO / ".railway" / "railway.ts"
DOCKERFILE = REPO / "Dockerfile"
DEPLOY_DOC = REPO / "docs" / "RAILWAY_DEPLOYMENT.md"


def _railway_json() -> dict:
    return json.loads(RAILWAY_JSON.read_text())


def _experiment_root(root: Path) -> Path:
    """A minimal, loadable paper experiment at ``root``.

    The dashboard's START path reads the experiment, so an empty directory
    would fail with a runner error instead of reaching the credential check
    this test is about.
    """
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
    source = StaticPriceSource({"BTC/USDT": frame}, name="static_test")
    config = ExperimentConfig.from_protocol(
        symbols=("BTC/USDT",), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference()
    )
    experiment.ensure_created()
    experiment.process_session(list(source.sessions())[0])
    return root


def _ts_start_command() -> str:
    """The start command the TypeScript file assembles, reassembled."""
    text = RAILWAY_TS.read_text()
    match = re.search(r"start:\s*\n?\s*(.*?),\n\s*healthcheck:", text, re.S)
    assert match, "the TypeScript deployment file no longer has a start command"
    parts = re.findall(r'"([^"]*)"|\'([^\']*)\'', match.group(1))
    return "".join(a if a else b for a, b in parts)


class TestTheDeploymentFilesAgree:
    def test_the_two_start_commands_are_identical(self):
        """Two files describing one service must not describe two services."""
        assert _ts_start_command() == _railway_json()["deploy"]["startCommand"]

    def test_the_health_check_path_is_served_by_the_dashboard(self):
        assert _railway_json()["deploy"]["healthcheckPath"] == "/healthz"

    def test_the_health_check_path_appears_in_the_type_script_file(self):
        assert '"/healthz"' in RAILWAY_TS.read_text()

    def test_both_files_build_from_the_same_dockerfile(self):
        assert _railway_json()["build"]["dockerfilePath"] == "Dockerfile"
        assert 'dockerfilePath: "Dockerfile"' in RAILWAY_TS.read_text()

    def test_the_volume_mount_path_is_the_same_in_both_files(self):
        """The container's data path and the mount path must agree."""
        mount = "/home/appuser/app/data"
        assert mount in _railway_json()["deploy"]["requiredMountPath"]
        assert mount in RAILWAY_TS.read_text()
        assert mount in _railway_json()["deploy"]["startCommand"]


class TestTheStartCommandIsSafe:
    def test_the_port_is_expanded_by_a_shell(self):
        """Railway runs a Dockerfile service's start command in exec form, where
        ``$PORT`` is not expanded, so the command must go through a shell."""
        command = _railway_json()["deploy"]["startCommand"]
        assert command.startswith('/bin/sh -c "')
        assert "$PORT" in command
        # The shell must not expand $PORT itself into the command string.
        assert "'$PORT'" not in command

    def test_exec_hands_the_platform_signals_to_python(self):
        """Without ``exec`` the shell stays in the foreground and the platform's
        SIGTERM never reaches the server."""
        command = _railway_json()["deploy"]["startCommand"]
        assert 'exec python -m tradingagents.paper_experiment.dashboard.server' in command

    def test_the_process_is_the_dashboard(self):
        """The dashboard is the only program that serves /healthz."""
        command = _railway_json()["deploy"]["startCommand"]
        assert "tradingagents.paper_experiment.dashboard.server" in command

    def test_the_command_binds_the_public_interface(self):
        """Railway reaches the service through the container's public interface."""
        assert "--host 0.0.0.0" in _railway_json()["deploy"]["startCommand"]

    def test_the_command_uses_the_data_root(self):
        assert "--root /home/appuser/app/data" in _railway_json()["deploy"]["startCommand"]


class TestTheImageEntryPoint:
    def test_the_entry_point_is_the_dashboard(self):
        """The image's default process must be the one that serves /healthz."""
        text = DOCKERFILE.read_text()
        entry = re.search(r"^ENTRYPOINT\s+(.+)$", text, re.M)
        assert entry, "the Dockerfile has no ENTRYPOINT"
        assert "tradingagents.paper_experiment.dashboard.server" in entry.group(1)

    def test_the_entry_point_is_not_the_analysis_cli(self):
        """The analysis CLI is a Typer app, so it would exit with
        "No such command" when a platform prepended its own arguments."""
        text = DOCKERFILE.read_text()
        entry = re.search(r"^ENTRYPOINT\s+(.+)$", text, re.M).group(1)
        assert "tradingagents\"" not in entry
        assert "cli.main" not in entry

    def test_the_health_check_probes_the_health_path(self):
        text = DOCKERFILE.read_text()
        health = re.search(r"^HEALTHCHECK.*?exit 1", text, re.M | re.S)
        assert health, "the Dockerfile has no HEALTHCHECK"
        assert "/healthz" in health.group(0)


class TestTheDocumentationIsHonest:
    def test_the_doc_does_not_claim_to_be_simulation_only(self):
        """The deployed service *can* place paper orders. The document must not
        describe it as a simulation that places none."""
        text = DEPLOY_DOC.read_text()
        assert "simulation-only workload" not in text
        assert "it is simulation-only" not in text

    def test_the_doc_does_not_claim_the_credentials_place_no_order(self):
        """The Alpaca paper credentials are trading credentials."""
        text = DEPLOY_DOC.read_text()
        assert "the read-only Alpaca **paper** diagnostic" not in text

    def test_the_doc_does_not_say_no_order_is_placed_by_the_deployment(self):
        text = DEPLOY_DOC.read_text()
        assert "Nothing here places an order" not in text

    def test_the_doc_states_that_paper_orders_can_be_placed(self):
        """The opposite must be said explicitly, so an operator is not misled."""
        text = DEPLOY_DOC.read_text()
        flat = " ".join(text.split())
        assert "places **paper** orders to Alpaca" in flat
        assert "submits **paper** orders to Alpaca" in flat

    def test_the_doc_says_the_deployment_was_not_exercised(self):
        """No Railway account and no Alpaca credentials were available, so the
        document must not imply they were."""
        text = DEPLOY_DOC.read_text()
        assert "## What is not verified" in text
        assert "No Railway account was accessible" in text
        flat = " ".join(text.split())
        assert "no real paper order was ever placed" in flat

    def test_the_doc_says_stop_does_not_close_positions(self):
        """Stopping prevents new orders; it must not silently liquidate."""
        text = DEPLOY_DOC.read_text()
        assert "stopping does not close a position" in text

    def test_the_doc_describes_the_symbol_validation_states(self):
        """The operator must know that validation is deferred and what it shows."""
        text = DEPLOY_DOC.read_text()
        assert "## Symbol validation" in text or "### Symbol validation" in text
        assert "NOT_VALIDATED" in text
        assert "VALIDATED" in text
        assert "EXECUTION_REGISTRY_UNAVAILABLE" in text

    def test_the_doc_names_the_rejection_codes(self):
        text = DEPLOY_DOC.read_text()
        for code in (
            "MARKET_DATA_UNAVAILABLE",
            "MARKET_DATA_NOT_TRADABLE",
            "EXECUTION_UNAVAILABLE",
            "EXECUTION_NOT_TRADABLE",
            "EXECUTION_NOT_FRACTIONABLE",
        ):
            assert code in text, f"{code} is not documented"


class TestNoLivePathIsConfigured:
    def test_the_start_command_names_no_live_host(self):
        command = _railway_json()["deploy"]["startCommand"]
        for live in ("api.alpaca.markets", "broker-api.alpaca.markets"):
            assert live not in command

    def test_no_live_credential_variable_is_configured(self):
        """Only the paper credential names may appear in the deployment."""
        text = RAILWAY_JSON.read_text() + RAILWAY_TS.read_text()
        for live in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY"):
            assert live not in text

    def test_the_paper_credentials_are_the_ones_configured(self):
        text = RAILWAY_JSON.read_text() + RAILWAY_TS.read_text()
        # The paper credentials are set in the platform dashboard, not here, so
        # their *names* must not be a configured value in either file.
        assert "ALPACA_PAPER_API_KEY" not in text
        assert "ALPACA_PAPER_API_SECRET" not in text

    def test_the_dashboard_token_is_a_placeholder_not_a_secret(self):
        """The JSON file carries a placeholder; the TypeScript file sets no
        token at all, because the real one is entered in the platform."""
        text = RAILWAY_JSON.read_text()
        assert "{{DASHBOARD_TOKEN}}" in text
        ts = RAILWAY_TS.read_text()
        assert "TRADINGAGENTS_DASHBOARD_TOKEN:" not in ts
        assert "set in the Railway dashboard, not here" in ts


class TestTheDeploymentCannotLeakASecret:
    @pytest.mark.parametrize(
        "path",
        [RAILWAY_JSON, RAILWAY_TS],
    )
    def test_no_credential_looking_value_is_committed(self, path):
        """A value that looks like a key or a secret must not be in the repo."""
        text = path.read_text()
        assert "sk-" not in text
        assert "AKIA" not in text
        assert not re.search(r"APCA_API_SECRET_KEY\s*[:=]\s*['\"][A-Za-z0-9]{16,}", text)


class TestAMissingCredentialIsAConfigurationProblem:
    """A missing credential is not a server fault, and must not read as one.

    A 500 says "the server broke"; a missing ``ALPACA_PAPER_API_KEY`` says
    "the deployment is not configured to trade". The operator needs to see the
    difference, and the response must still name the variable to set without
    ever revealing a value.
    """

    def _serve(self, dashboard):
        import json as _json
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer

        from tradingagents.paper_experiment.dashboard.server import (
            PAPER_CONFIRM_HEADER,
            make_handler,
        )

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            request = urllib.request.Request(f"{base}/api/paper/start", method="POST")
            request.add_header("Authorization", "Bearer t")
            request.add_header(PAPER_CONFIRM_HEADER, "yes")
            try:
                with urllib.request.urlopen(request, data=b"{}", timeout=10) as response:
                    status, body = response.status, _json.loads(response.read())
            except urllib.error.HTTPError as exc:
                status, body = exc.code, _json.loads(exc.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        return status, body

    def test_a_missing_credential_is_reported_as_503_not_500(self, allow_loopback):
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        root = _experiment_root(Path(tempfile.mkdtemp()))
        dashboard = Dashboard(root, token="t", allow_paper_trading=True)
        try:
            status, body = self._serve(dashboard)
        finally:
            dashboard.paper.shutdown()
        assert status == 503, body
        assert "not configured" in body.get("reason", "").lower()
        # The message names the variable to set, and no value.
        assert "ALPACA_PAPER_API_KEY" in body["error"]
        assert "ALPACA_PAPER_API_SECRET" in body["error"]

    def test_the_response_carries_no_credential_value(self, allow_loopback):
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        root = _experiment_root(Path(tempfile.mkdtemp()))
        dashboard = Dashboard(root, token="t", allow_paper_trading=True)
        try:
            _status, body = self._serve(dashboard)
        finally:
            dashboard.paper.shutdown()
        blob = str(body)
        assert "APCA-API-KEY-ID" not in blob
        assert "APCA-API-SECRET-KEY" not in blob
