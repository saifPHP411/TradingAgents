"""The deployed Railway dashboard must be explicitly wired to Bybit Demo."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAILWAY_JSON = ROOT / "railway.json"
RAILWAY_TS = ROOT / ".railway" / "railway.ts"
ENV_EXAMPLE = ROOT / ".env.example"


def test_railway_command_explicitly_selects_bybit_demo():
    railway = json.loads(RAILWAY_JSON.read_text(encoding="utf-8"))
    command = railway["deploy"]["startCommand"]
    match = re.search(r'/bin/sh -c "(.*)"$', command)
    assert match, "the Railway command must remain shell-wrapped"
    argv = shlex.split(match.group(1))
    assert "--paper-execution-provider" in argv
    index = argv.index("--paper-execution-provider")
    assert argv[index + 1] == "bybit_demo"
    assert "--allow-paper-trading" in argv
    assert "--paper-decision-source" in argv
    assert argv[argv.index("--paper-decision-source") + 1] == "crypto"


def test_railway_environment_also_selects_bybit_demo():
    railway = json.loads(RAILWAY_JSON.read_text(encoding="utf-8"))
    variables = railway["environments"]["production"]["variables"]
    assert variables["TRADINGAGENTS_PAPER_EXECUTION_PROVIDER"] == "bybit_demo"


def test_railway_iac_matches_the_bybit_demo_selection():
    text = RAILWAY_TS.read_text(encoding="utf-8")
    assert "--paper-execution-provider bybit_demo" in text
    assert 'TRADINGAGENTS_PAPER_EXECUTION_PROVIDER: "bybit_demo"' in text
    assert "Bybit's **Demo Spot** endpoint" in text


def test_bybit_demo_credentials_are_named_but_not_committed():
    example = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "#BYBIT_DEMO_API_KEY=" in example
    assert "#BYBIT_DEMO_API_SECRET=" in example
    deployment = RAILWAY_JSON.read_text(encoding="utf-8") + RAILWAY_TS.read_text(
        encoding="utf-8"
    )
    assert "BYBIT_DEMO_API_KEY" not in deployment
    assert "BYBIT_DEMO_API_SECRET" not in deployment


def test_dashboard_deployment_never_selects_a_live_bybit_host():
    railway = json.loads(RAILWAY_JSON.read_text(encoding="utf-8"))
    command = railway["deploy"]["startCommand"]
    assert "api.bybit.com" not in command
    assert "api-testnet.bybit.com" not in command
