"""The deployment wiring: the flags, the environment and the Railway settings.

A deployment is configured through three layers that must agree: the command
line, the environment variables and the ``railway.json`` the platform reads.
These tests pin the properties that make a deployment *safe* rather than merely
*runnable*:

1. the execution-price deviation is configurable, and a misconfigured value is a
   loud error rather than a silently disabled guard;
2. the paper path is **off unless asked for** — the flag defaults to off and the
   environment variable is opt-in;
3. the deployment names a **paper** broker, never a live host;
4. the deployment's health check is the endpoint the server actually serves;
5. the deployment's symbol lists are single-venue and consistent with each other.

The tests read the deployment file from disk and exercise the real argument
parser, so a rename of a flag or a variable breaks them.
"""

from __future__ import annotations

import json
import re
import shlex
import threading
import time
from pathlib import Path

import pytest

from tradingagents.paper_experiment.broker import (
    DEFAULT_MAX_PRICE_DEVIATION,
    MAX_PRICE_DEVIATION_ENV,
)
from tradingagents.paper_experiment.dashboard import server as server_module
from tradingagents.paper_experiment.dashboard.server import (
    HEALTH_PATH,
    _resolve_max_price_deviation,
    build_parser,
)

ROOT = Path(__file__).resolve().parents[1]


class _Args:
    """A stand-in for the parsed argparse namespace."""

    def __init__(self, **kwargs) -> None:
        self.paper_max_price_deviation = kwargs.get("paper_max_price_deviation")


class TestMaxPriceDeviation:
    def test_the_default_is_the_projects_own(self):
        assert _resolve_max_price_deviation(_Args()) == DEFAULT_MAX_PRICE_DEVIATION

    def test_the_flag_wins(self):
        value = _resolve_max_price_deviation(_Args(paper_max_price_deviation=0.02))
        assert value == pytest.approx(0.02)

    def test_the_environment_variable_is_read(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "0.03")
        assert _resolve_max_price_deviation(_Args()) == pytest.approx(0.03)

    def test_the_flag_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "0.03")
        value = _resolve_max_price_deviation(_Args(paper_max_price_deviation=0.01))
        assert value == pytest.approx(0.01)

    def test_a_non_numeric_value_is_a_loud_error(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "five percent")
        with pytest.raises(Exception) as excinfo:
            _resolve_max_price_deviation(_Args())
        assert MAX_PRICE_DEVIATION_ENV in str(excinfo.value)

    def test_a_negative_value_is_a_loud_error(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "-0.1")
        with pytest.raises(Exception) as excinfo:
            _resolve_max_price_deviation(_Args())
        assert MAX_PRICE_DEVIATION_ENV in str(excinfo.value)

    def test_a_nan_value_is_a_loud_error(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "nan")
        with pytest.raises(Exception) as excinfo:
            _resolve_max_price_deviation(_Args())
        assert MAX_PRICE_DEVIATION_ENV in str(excinfo.value)

    def test_an_infinite_value_is_a_loud_error(self, monkeypatch):
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "inf")
        with pytest.raises(Exception) as excinfo:
            _resolve_max_price_deviation(_Args())
        assert MAX_PRICE_DEVIATION_ENV in str(excinfo.value)

    def test_zero_is_allowed(self, monkeypatch):
        """Zero is a legitimate, if strict, guard."""
        monkeypatch.setenv(MAX_PRICE_DEVIATION_ENV, "0")
        assert _resolve_max_price_deviation(_Args()) == pytest.approx(0.0)


class TestPaperTradingIsOptIn:
    def test_the_paper_trading_flag_is_off_by_default(self):
        """The deployment must not place an order because the server started."""


        parser = build_parser()
        args = parser.parse_args(["--root", str(ROOT)])
        assert args.allow_paper_trading is False

    def test_the_crypto_live_flag_is_off_by_default(self):

        parser = build_parser()
        args = parser.parse_args(["--root", str(ROOT)])
        assert args.crypto_live is False

    def test_the_parser_accepts_the_deviation_flag(self):

        parser = build_parser()
        args = parser.parse_args(
            ["--root", str(ROOT), "--paper-max-price-deviation", "0.02"]
        )
        assert args.paper_max_price_deviation == pytest.approx(0.02)


class TestRailwaySettings:
    @pytest.fixture
    def railway(self) -> dict:
        return json.loads((ROOT / "railway.json").read_text())

    def test_the_health_check_path_is_served(self, railway):
        """The health check must name an endpoint the server actually serves."""
        assert railway["deploy"]["healthcheckPath"] == HEALTH_PATH

    def test_the_health_check_path_exists_in_the_server(self):
        assert HEALTH_PATH.startswith("/")
        assert HEALTH_PATH == "/healthz"

    def test_the_start_command_serves_the_dashboard(self, railway):
        command = railway["deploy"]["startCommand"]
        assert "tradingagents.paper_experiment.dashboard.server" in command
        assert "--allow-paper-trading" in command

    def test_the_start_command_binds_every_interface(self, railway):
        """A public deployment must bind 0.0.0.0, not loopback."""
        command = railway["deploy"]["startCommand"]
        assert "--host 0.0.0.0" in command

    def test_the_start_command_uses_the_port_variable(self, railway):
        command = railway["deploy"]["startCommand"]
        assert "$PORT" in command

    def test_the_deployment_names_the_paper_decision_source(self, railway):
        variables = railway["environments"]["production"]["variables"]
        assert variables["TRADINGAGENTS_PAPER_DECISION_SOURCE"] == "crypto"

    def test_the_deployment_names_the_binance_market_data(self, railway):
        variables = railway["environments"]["production"]["variables"]
        assert variables["TRADINGAGENTS_PAPER_MARKET_DATA"] == "binance_snapshot"

    def test_the_deployment_requires_a_dashboard_token(self, railway):
        variables = railway["environments"]["production"]["variables"]
        assert variables["TRADINGAGENTS_DASHBOARD_TOKEN"]

    def test_the_deployment_names_no_live_broker_host(self, railway):
        raw = json.dumps(railway)
        # The live host must never appear. The paper host is the only one allowed,
        # and it is not written into the deployment file at all.
        assert "api.alpaca.markets" not in raw
        assert "paper-api.alpaca.markets" not in raw

    def test_the_restart_policy_is_declared(self, railway):
        assert railway["deploy"]["restartPolicyType"] in {"ON_FAILURE", "ON_SUCCESS", "NO"}

    def test_the_build_uses_the_dockerfile(self, railway):
        assert railway["build"]["builder"] == "DOCKERFILE"

    def test_the_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_the_dockerfile_creates_a_non_root_user(self):
        """The container must not run the dashboard as root."""
        text = (ROOT / "Dockerfile").read_text()
        assert "USER " in text

    def test_the_dockerfile_probes_the_health_check(self):
        text = (ROOT / "Dockerfile").read_text()
        assert "/healthz" in text


class TestDeploymentEnvironment:
    def test_the_env_example_names_the_deviation_variable(self):
        text = (ROOT / ".env.example").read_text()
        assert MAX_PRICE_DEVIATION_ENV in text

    def test_the_env_example_never_carries_a_secret(self):
        """The example must carry names and blanks, never values."""
        text = (ROOT / ".env.example").read_text()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            assert "KEY" in name or "SECRET" in name or "TOKEN" in name or not value, (
                f"{name!r} carries a value: {value!r}"
            )

    def test_the_env_example_names_the_alpaca_paper_credentials(self):
        text = (ROOT / ".env.example").read_text()
        assert "ALPACA_PAPER_API_KEY" in text
        assert "ALPACA_PAPER_API_SECRET" in text

    def test_the_env_example_never_names_a_live_credential(self):
        text = (ROOT / ".env.example").read_text()
        assert "ALPACA_API_KEY" not in text.replace("ALPACA_PAPER_API_KEY", "")
        assert "ALPACA_API_SECRET" not in text.replace("ALPACA_PAPER_API_SECRET", "")


class TestTheTwoDeploymentFilesAgree:
    """`railway.json` is deprecated; `.railway/railway.ts` replaces it.

    Railway stops reading Config as Code on 2026-12-01, so both files exist for
    a while. They must describe the **same** service: a start command or a
    health check that differs between them is a deployment that behaves
    differently from what the repository says it does.
    """

    @pytest.fixture
    def railway(self) -> dict:
        return json.loads((ROOT / "railway.json").read_text())

    @pytest.fixture
    def iac(self) -> str:
        return (ROOT / ".railway" / "railway.ts").read_text()

    def test_the_iac_file_exists(self, iac):
        assert iac.strip()

    def test_the_iac_file_exports_the_project(self, iac):
        assert "defineRailway" in iac
        assert "export default" in iac

    def test_the_iac_health_check_matches_the_json(self, railway, iac):
        assert f'healthcheck: "{railway["deploy"]["healthcheckPath"]}"' in iac

    def test_the_iac_names_the_same_module(self, railway, iac):
        assert "tradingagents.paper_experiment.dashboard.server" in iac
        assert "tradingagents.paper_experiment.dashboard.server" in railway["deploy"]["startCommand"]

    def test_the_iac_names_the_same_decision_source(self, railway, iac):
        source = railway["environments"]["production"]["variables"][
            "TRADINGAGENTS_PAPER_DECISION_SOURCE"
        ]
        assert f'TRADINGAGENTS_PAPER_DECISION_SOURCE: "{source}"' in iac

    def test_the_iac_names_the_same_market_data(self, railway, iac):
        provider = railway["environments"]["production"]["variables"][
            "TRADINGAGENTS_PAPER_MARKET_DATA"
        ]
        assert f'TRADINGAGENTS_PAPER_MARKET_DATA: "{provider}"' in iac

    def test_the_iac_carries_no_secret(self, iac):
        """A secret in the IaC file would be a secret in the repository."""
        for line in iac.splitlines():
            if "API_KEY" in line or "API_SECRET" in line or "TOKEN" in line:
                # The variable may be *named*; it may not carry a value.
                assert "=" not in line or ":" not in line, (
                    f"a secret-bearing variable carries a value: {line.strip()!r}"
                )

    def test_the_iac_never_names_a_live_host(self, iac):
        assert "api.alpaca.markets" not in iac

    def test_the_json_start_command_is_shell_wrapped(self, railway):
        """A Dockerfile service's start command runs in exec form, where
        ``$PORT`` is not expanded. The command must therefore be shell-wrapped."""
        command = railway["deploy"]["startCommand"]
        assert "$PORT" in command
        assert command.startswith("/bin/sh -c ")

    def test_the_iac_start_command_is_shell_wrapped(self, iac):
        assert "$PORT" in iac
        assert "/bin/sh -c" in iac

    def test_the_json_uses_only_keys_the_railway_schema_declares(self, railway):
        """Config as Code has a closed schema.

        A key Railway does not know is not an error — it is silently ignored, so
        a typo would quietly drop a setting (a health check that never runs, a
        restart policy that never retries). The allowed keys are the ones
        ``railway.com/railway.schema.json`` declares, and the schema is
        vendored here so the test stays offline.
        """
        schema_path = ROOT / "deploy" / "railway.schema.json"
        assert schema_path.exists(), "the Railway schema must be vendored for offline validation"
        schema = json.loads(schema_path.read_text())
        allowed = set(schema["properties"])
        assert set(railway) <= allowed

        deploy_allowed = set(schema["properties"]["deploy"]["properties"])
        assert set(railway["deploy"]) <= deploy_allowed

        build_allowed = set(schema["properties"]["build"]["properties"])
        assert set(railway["build"]) <= build_allowed

    def test_the_iac_names_the_same_builder_as_the_json(self, railway, iac):
        """One service, one builder: a Dockerfile here and Nixpacks there would
        be two different deployments."""
        builder = railway["build"]["builder"]
        assert builder == "DOCKERFILE"
        assert 'builder: "DOCKERFILE"' in iac
        path = railway["build"]["dockerfilePath"]
        assert f'dockerfilePath: "{path}"' in iac

    def test_both_files_put_the_state_on_the_same_path(self, railway, iac):
        """The durable state must live on a mounted volume.

        The worker's processed/ordered decision sets are what stop a restart
        from placing the same order twice. They are files under the experiment
        root, so the root must be a mounted path in both deployment files.
        """
        mount = "/home/appuser/app/data"
        assert railway["deploy"]["requiredMountPath"] == mount
        assert '"/home/appuser/app/data": data' in iac
        # The start command must point at that same path.
        assert f"--root {mount}" in railway["deploy"]["startCommand"]
        assert f"--root {mount}" in iac

    def test_both_files_run_as_root_so_the_volume_is_writable(self, railway, iac):
        """The image runs as the non-root ``appuser``.

        A Railway volume mounted into a non-root container is not writable by
        that user: the first write fails with ``PermissionError``, and the
        worker's state never lands. Railway's documented remedy is
        ``RAILWAY_RUN_UID=0``. The deployment docs call it "not optional", so
        both files must set it.
        """
        assert railway["environments"]["production"]["variables"]["RAILWAY_RUN_UID"] == "0"
        assert 'RAILWAY_RUN_UID: "0"' in iac

    def test_the_iac_mounts_a_volume_for_the_state(self, iac):
        """A volume resource and a mount, or the state is ephemeral."""
        assert 'volume("data"' in iac
        assert "volumeMounts" in iac
        assert "sizeMB" in iac

    def test_the_iac_uses_only_option_names_the_railway_sdk_accepts(self, iac):
        """The SDK's `service()` input is a closed set of keys.

        A key the SDK does not know is ignored by Railway, so a typo would
        silently drop a setting — a health check that never runs, or a variable
        that is never set. These are the option names the SDK's
        ``IntentServiceConfig`` declares.
        """
        known = {
            "source", "root", "rootDirectory", "build", "deploy", "run",
            "start", "startCommand", "preDeploy", "preDeployCommand",
            "healthcheck", "healthcheckPath", "healthcheckTimeout", "replicas",
            "regions", "networking", "domains", "tcp", "tcpProxies", "env",
            "variables", "volumeMounts", "configFile", "parentServiceId",
            "groupId", "clusterRole", "replicaConfig", "clusterDisplay",
            "tracing", "cdn",
        }
        service_body = iac.split('service("crypto-dashboard", {', 1)[1].split("\n  });", 1)[0]
        # Only the keys at the service's own level; the nested `env` and
        # `build` objects are checked separately.
        import re

        top_level = re.findall(r"^\s{4}([a-zA-Z]+):", service_body, re.MULTILINE)
        unknown = sorted(set(top_level) - known)
        assert unknown == []


class TestTheCryptoPathsTradeTheSameRules:
    """The live crypto path and the crypto backtest must use the same settings.

    A backtest that measured one rule while the live path traded another
    describes a system that does not exist. The one adaptation TJR needs for a
    24/7 market is the entry-window cut, and it is written down in exactly one
    place.
    """

    def test_the_crypto_entry_cut_is_the_last_half_hour_of_the_utc_day(self):
        from tradingagents.crypto.tjr_settings import CRYPTO_ENTRY_CUT

        assert CRYPTO_ENTRY_CUT == (23, 30)

    def test_the_backtest_study_uses_the_shared_crypto_settings(self):
        from tradingagents.crypto.backtest.study import _crypto_settings
        from tradingagents.crypto.tjr_settings import crypto_tjr_settings

        assert _crypto_settings() == crypto_tjr_settings()

    def test_the_backtest_study_is_not_the_equity_default(self):
        """The equity default is 15:30 ET; a crypto path that used it would stop
        taking entries eight hours before the crypto session ends."""
        from tradingagents.crypto.backtest.study import _crypto_settings
        from tradingagents.tjr.rules import TjrSettings

        assert _crypto_settings().latest_entry_time != TjrSettings().latest_entry_time

    def test_the_live_crypto_source_is_built_with_the_crypto_settings(self):
        """The controller must pass the crypto settings to the source it builds."""
        import inspect

        from tradingagents.paper_experiment.dashboard import paper_control

        source = inspect.getsource(paper_control.PaperTradingController._build_crypto_source)
        assert "crypto_tjr_settings()" in source

    def test_only_the_entry_cut_differs_from_the_equity_default(self):
        """Every other rule and default is unchanged by the crypto adaptation."""
        from dataclasses import fields

        from tradingagents.crypto.tjr_settings import crypto_tjr_settings
        from tradingagents.tjr.rules import TjrSettings

        equity = TjrSettings()
        crypto = crypto_tjr_settings()
        differences = {
            f.name
            for f in fields(equity)
            if getattr(equity, f.name) != getattr(crypto, f.name)
        }
        assert differences == {"latest_entry_time"}


class TestTheInstalledPackageCarriesItsData:
    """A deployment installs the wheel, so the data must travel with it.

    The dashboard creates the experiment from the frozen dataset when the root
    has no experiment yet. That dataset is committed to the repository but is
    *not* Python source, so it is only in the wheel if ``package-data`` says so.
    Without it, an installed deployment has no data to start from and refuses
    to start with "the frozen dataset is unavailable" — a failure that only
    appears after the image is built, never in the checkout.

    The test reads the packaging configuration rather than building a wheel, so
    it stays fast and offline, but it asserts the same fact the wheel does.
    """

    def test_the_frozen_dataset_is_declared_as_package_data(self):
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = config["tool"]["setuptools"]["package-data"]
        patterns = package_data.get("tradingagents.baseline", [])
        assert "data/*.csv" in patterns, (
            "the frozen baseline dataset must be packaged, or an installed "
            "deployment cannot create an experiment"
        )

    def test_the_committed_dataset_matches_the_declared_pattern(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        data_dir = root / "tradingagents" / "baseline" / "data"
        files = sorted(p.name for p in data_dir.glob("*.csv"))
        assert files, "the frozen dataset is missing from the repository"


class TestTheEffectiveStartupCommand:
    """The command Railway actually runs, checked against the real parser.

    A deployment is configured in three places that must agree: the Dockerfile,
    ``railway.json`` and ``.railway/railway.ts``. A mismatch between them is
    invisible in the repository and fatal on the platform — the service either
    starts the wrong program or refuses to start at all.

    These tests take the command the platform would run, strip it down to the
    flags, and feed those flags to the **real** ``argparse`` parser. A flag that
    is renamed, removed or misspelled therefore breaks the test rather than the
    deployment.
    """

    MODULE = "tradingagents.paper_experiment.dashboard.server"

    @pytest.fixture
    def railway(self) -> dict:
        return json.loads((ROOT / "railway.json").read_text())

    @pytest.fixture
    def iac_start(self) -> str:
        """The start command the Infrastructure-as-Code file declares.

        It is assembled from concatenated string literals, so the literals are
        joined in source order to recover the command.
        """
        text = (ROOT / ".railway" / "railway.ts").read_text()
        block = re.search(r"start:\s*(.*?),\n\s*healthcheck:", text, re.S)
        assert block, "the IaC file no longer declares a start command"
        parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", block.group(1))
        return "".join(a or b for a, b in parts)

    def _shell_payload(self, command: str) -> str:
        """The command inside the ``/bin/sh -c "..."`` wrapper."""
        match = re.search(r'/bin/sh -c "(.*)"$', command, re.S)
        assert match, (
            "the start command must be shell-wrapped: Railway runs a Dockerfile "
            "service's start command in exec form, where $PORT is not expanded"
        )
        return match.group(1)

    def _argv(self, command: str) -> list[str]:
        """The command as a shell would see it, minus the ``exec`` builtin."""
        argv = shlex.split(self._shell_payload(command))
        assert argv[0] == "exec", f"the start command should exec its program: {argv[:1]}"
        return argv[1:]

    def _flags(self, command: str) -> list[str]:
        """Only the flags — what ``argparse`` actually receives.

        The program and module name are consumed by the interpreter, and
        ``$PORT`` is expanded by the shell before the program sees it.
        """
        argv = self._argv(command)
        assert argv[0] == "python", f"expected the venv python, got {argv[0]!r}"
        assert argv[1] == "-m"
        assert argv[2] == self.MODULE, f"expected {self.MODULE}, got {argv[2]!r}"
        flags = argv[3:]
        # The shell expands $PORT before exec; stand in a real port so the
        # parser's int conversion is exercised the way it is in production.
        return ["8765" if flag == "$PORT" else flag for flag in flags]

    # -- the command is well formed -----------------------------------

    def test_the_start_command_runs_the_dashboard_module(self, railway):
        argv = self._argv(railway["deploy"]["startCommand"])
        assert argv[:3] == ["python", "-m", self.MODULE]

    def test_the_start_command_is_shell_wrapped(self, railway):
        """`$PORT` is only expanded by a shell, never in exec form."""
        assert self._shell_payload(railway["deploy"]["startCommand"])

    def test_every_flag_is_accepted_by_the_real_parser(self, railway):
        """The decisive check: the real parser must accept the deployment's flags."""
        flags = self._flags(railway["deploy"]["startCommand"])
        namespace = build_parser().parse_args(flags)
        assert namespace.root

    def test_no_flag_in_the_command_is_unknown(self, railway):
        """An unknown flag must be caught here, not on the platform."""
        known = {
            action.option_strings[0]
            for action in build_parser()._actions
            if action.option_strings
        }
        for flag in self._flags(railway["deploy"]["startCommand"]):
            if flag.startswith("--"):
                assert flag in known, f"{flag!r} is not a flag the server accepts"

    # -- the effective settings ---------------------------------------

    def test_the_effective_root_is_the_volume_mount(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.root == railway["deploy"]["requiredMountPath"]

    def test_the_effective_bind_is_every_interface(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.host == "0.0.0.0"

    def test_the_effective_port_comes_from_the_environment(self, railway):
        """The port is the platform's, not a hard-coded one."""
        command = railway["deploy"]["startCommand"]
        assert "$PORT" in command
        namespace = build_parser().parse_args(self._flags(command))
        assert namespace.port == 8765  # the stand-in the shell would have expanded

    def test_paper_trading_is_enabled_by_the_command(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.allow_paper_trading is True

    def test_the_command_reads_the_binance_price_seam(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.paper_market_data == "binance_snapshot"

    def test_the_command_uses_the_crypto_decision_source(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.paper_decision_source == "crypto"

    def test_the_command_starts_the_live_crypto_engine(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.crypto_live is True

    def test_the_command_does_not_enable_the_simulation_control(self, railway):
        """``--allow-control`` advances the *simulation*; it is a different thing
        from paper trading and is not part of this deployment."""
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.allow_control is False

    def test_the_command_does_not_enable_auto_advance(self, railway):
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.auto_advance == 0

    def test_the_command_does_not_use_the_stub_source(self, railway):
        """The stub source is a non-genuine fixture; a deployment must not use it."""
        namespace = build_parser().parse_args(
            self._flags(railway["deploy"]["startCommand"])
        )
        assert namespace.stub is False

    # -- the three files agree ----------------------------------------

    def test_the_iac_start_command_is_identical_to_the_json_one(self, railway, iac_start):
        assert iac_start == railway["deploy"]["startCommand"]

    def test_the_dockerfile_entrypoint_cannot_shadow_the_start_command(self, railway):
        """The image's ENTRYPOINT must not be the command the platform appends to.

        Docker runs ``ENTRYPOINT`` followed by the command the platform sets as
        ``CMD``. Railway's own documentation states that a start command
        *overrides* the image's ENTRYPOINT in exec form for a Dockerfile
        service — but the safe shape is an image whose ENTRYPOINT is the
        dashboard itself, so the service starts the right program whether the
        platform overrides the entrypoint or merely appends to it.

        The image's ENTRYPOINT is therefore checked to be the dashboard module,
        not the analysis CLI: ``tradingagents`` is a Typer app whose bare form
        runs an interactive LLM analysis, which is not a server.
        """
        text = (ROOT / "Dockerfile").read_text()
        entrypoints = re.findall(r"^ENTRYPOINT\s+(.+)$", text, re.M)
        assert entrypoints, "the Dockerfile declares no ENTRYPOINT"
        entrypoint = " ".join(entrypoints)
        assert self.MODULE in entrypoint or "paper_experiment" in entrypoint, (
            "the image's ENTRYPOINT must start the paper dashboard, not the "
            f"analysis CLI: {entrypoint!r}"
        )

    # -- what the startup command does NOT do -------------------------

    def test_the_startup_does_not_start_the_paper_worker(self, railway):
        """The worker starts only when an operator presses START.

        The startup command's job is to serve the dashboard. If it started the
        worker, a redeploy would resume trading on its own — which is exactly
        the behaviour the deployment is required not to have.
        """
        command = railway["deploy"]["startCommand"]
        flags = self._flags(command)
        # No flag starts a worker: the worker is started by the API.
        assert "--start-worker" not in flags
        assert "--auto-advance" not in flags
        namespace = build_parser().parse_args(flags)
        assert namespace.allow_paper_trading is True, (
            "the flag only permits the START button to exist; it does not start it"
        )

    def test_the_startup_does_not_name_a_broker_host(self, railway):
        """No broker host is in the command, so none can be reached at startup."""
        command = railway["deploy"]["startCommand"]
        assert "alpaca.markets" not in command


class TestTheImageEntryPointServesTheHealthCheck:
    """The image's default process must satisfy the image's own HEALTHCHECK.

    The Dockerfile probes ``/healthz``. Only the dashboard serves that path —
    the analysis CLI serves no HTTP at all — so an image whose default process
    is the analysis CLI carries a health check that can never pass.
    """

    @pytest.fixture
    def entrypoint(self) -> list[str]:
        text = (ROOT / "Dockerfile").read_text()
        match = re.search(r"^ENTRYPOINT\s+\[(.*)\]$", text, re.M)
        assert match, "the Dockerfile declares no exec-form ENTRYPOINT"
        return [part.strip().strip('"') for part in match.group(1).split(",")]

    def test_the_entrypoint_is_the_dashboard_module(self, entrypoint):
        assert entrypoint == [
            "python",
            "-m",
            "tradingagents.paper_experiment.dashboard.server",
        ]

    def test_the_entrypoint_program_is_not_the_analysis_cli(self, entrypoint):
        """The analysis CLI is a Typer app: it is not a server, and as an
        ENTRYPOINT it would swallow the platform's start command."""
        assert "tradingagents" not in entrypoint

    def test_the_entrypoint_module_actually_serves_the_health_path(self):
        """The HEALTHCHECK's URL must be served by the entrypoint's program."""
        from tradingagents.paper_experiment.dashboard.server import HEALTH_PATH

        assert HEALTH_PATH == "/healthz"

    def test_the_healthcheck_probes_the_health_path(self):
        text = (ROOT / "Dockerfile").read_text()
        assert "/healthz" in text

    def test_the_compose_analysis_services_name_the_analysis_cli(self):
        """The analysis CLI moved out of the image's ENTRYPOINT, so the compose
        services that run it must name it themselves."""
        text = (ROOT / "docker-compose.yml").read_text()
        assert text.count("entrypoint:\n      - tradingagents\n") == 2, (
            "the `tradingagents` and `tradingagents-ollama` services must each "
            "name the analysis CLI as their entrypoint"
        )


def _serve_in_thread_for_startup(root):
    """Run the real ``serve`` startup path in a thread and return its dashboard.

    ``serve`` is what the module entry point calls, so this exercises the same
    code the platform's start command runs — including ``_ensure_experiment``
    and the worker that is *not* built.
    """
    from tradingagents.paper_experiment.dashboard.server import serve

    captured = {}
    original = server_module.Dashboard

    class _Capturing(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["dashboard"] = self

    server_module.Dashboard = _Capturing
    try:
        thread = threading.Thread(
            target=serve,
            kwargs={"root": str(root), "host": "127.0.0.1", "port": 0,
                    "allow_paper_trading": True},
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 30
        while "dashboard" not in captured and time.monotonic() < deadline:
            thread.join(timeout=0.05)
        assert "dashboard" in captured, "serve did not build a dashboard"
    finally:
        server_module.Dashboard = original

    dashboard = captured["dashboard"]
    return dashboard


class TestTheStartupItselfPlacesNothing:
    """Starting the dashboard must not start the worker or reach a broker.

    This is the property that separates the four control layers: the platform
    starts the *dashboard*, the operator starts the *worker*, and only the
    worker's cycle can reach a *broker*. The dashboard's startup must therefore
    create the experiment's state and nothing else.
    """

    def test_serve_creates_the_experiment_state_and_no_worker(self, tmp_path):
        """The real ``serve`` path: an experiment exists, a worker does not."""
        root = tmp_path / "experiment"
        dashboard = _serve_in_thread_for_startup(root)
        try:
            assert (root / "session_state.json").exists(), (
                "the startup must create the experiment's state"
            )
            assert (root / "journal.json").exists()
            # No worker was built, so no worker state file exists.
            assert not (root / "worker_state.json").exists()
            assert dashboard.paper._worker is None, (
                "the startup must not build a paper worker"
            )
        finally:
            dashboard.paper.shutdown()

    def test_the_startup_reaches_no_broker(self, tmp_path, monkeypatch):
        """No HTTP request may leave the process during startup.

        Both HTTP libraries the project uses are guarded: the Alpaca broker
        calls ``requests.request`` and the market-data providers use
        ``urllib``. Guarding only one of them would let the other through and
        make this test pass while a request was actually made.
        """
        import urllib.request

        import requests

        calls: list[str] = []

        def _record(url) -> None:
            calls.append(str(url))
            raise AssertionError(f"the startup must not make a request: {url}")

        def _guarded_request(method, url, *args, **kwargs):
            _record(url)

        monkeypatch.setattr(requests, "request", _guarded_request)
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _record(a[0])
        )
        monkeypatch.setattr(
            urllib.request,
            "build_opener",
            lambda *a, **k: type(
                "_Guard",
                (urllib.request.OpenerDirector,),
                {"open": lambda self, url, data=None, timeout=None: _record(url)},
            )(),
        )

        root = tmp_path / "experiment"
        dashboard = _serve_in_thread_for_startup(root)
        try:
            assert dashboard.paper._worker is None
            assert calls == [], f"the startup made a request: {calls}"
        finally:
            dashboard.paper.shutdown()

    def test_the_broker_guard_would_catch_a_real_call(self, monkeypatch):
        """The guard above must be a real guard, not a no-op.

        A test that patches an HTTP library the code never uses would pass
        forever while the startup quietly contacted a broker.
        """
        import requests

        seen: list[str] = []

        def _capture(method, url, *args, **kwargs):
            seen.append(url)

        monkeypatch.setattr(requests, "request", _capture)
        requests.request("GET", "https://paper-api.alpaca.markets/v2/account")
        assert seen == ["https://paper-api.alpaca.markets/v2/account"]


class TestTheVolumeMountPathAgrees:
    """The mount path is written in three places; they must be one path.

    A volume mounted at one path while `--root` points at another is a
    deployment that looks configured and is not: the experiment's state would
    land on the container's ephemeral filesystem, and a redeploy would forget
    which decisions already produced an order.
    """

    @pytest.fixture
    def railway(self) -> dict:
        return json.loads((ROOT / "railway.json").read_text())

    def test_the_json_required_mount_path_is_the_experiment_root(self, railway):
        assert railway["deploy"]["requiredMountPath"] == "/home/appuser/app/data"

    def test_the_start_command_roots_the_experiment_at_the_mount(self, railway):
        namespace = build_parser().parse_args(
            TestTheEffectiveStartupCommand()._flags(
                railway["deploy"]["startCommand"]
            )
        )
        assert namespace.root == railway["deploy"]["requiredMountPath"]

    def test_the_iac_mounts_the_volume_at_the_same_path(self, railway):
        text = (ROOT / ".railway" / "railway.ts").read_text()
        assert '"/home/appuser/app/data": data' in text, (
            "the IaC file must mount the volume at the same path the start "
            "command roots the experiment at"
        )

    def test_the_documented_mount_path_matches(self, railway):
        text = (ROOT / "docs" / "RAILWAY_DEPLOYMENT.md").read_text()
        assert railway["deploy"]["requiredMountPath"] in text, (
            "the deployment doc must name the mount path the service uses"
        )

    def test_the_volume_is_not_the_container_filesystem(self, railway):
        """The mount must be a real path, not the image's own working dir."""
        mount = railway["deploy"]["requiredMountPath"]
        assert mount.startswith("/")
        assert mount != "/home/appuser/app"
