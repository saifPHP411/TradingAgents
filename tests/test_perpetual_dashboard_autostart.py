from __future__ import annotations

import json

from tradingagents.paper_experiment.dashboard.server import (
    AUTO_START_PERPETUAL_LAB_ENV,
    AUTO_START_PERPETUAL_STUDY_ENV,
    Dashboard,
    _maybe_auto_start_perpetual_lab,
    _maybe_auto_start_perpetual_study,
)


class _LabStub:
    def __init__(self) -> None:
        self.calls = 0

    def start_perpetual_lab(self) -> dict:
        self.calls += 1
        return {"status": "running"}


class _LegacyStudyStub:
    def __init__(self) -> None:
        self.calls = 0

    def start_perpetual_study(self) -> dict:
        self.calls += 1
        return {"status": "running"}


def test_legacy_fixed_ema_holdout_never_auto_starts_after_being_inspected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(AUTO_START_PERPETUAL_STUDY_ENV, "true")
    dashboard = _LegacyStudyStub()

    _maybe_auto_start_perpetual_study(dashboard, tmp_path)

    assert dashboard.calls == 0


def test_perpetual_lab_does_not_auto_start_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(AUTO_START_PERPETUAL_LAB_ENV, raising=False)
    dashboard = _LabStub()

    _maybe_auto_start_perpetual_lab(dashboard, tmp_path)

    assert dashboard.calls == 0


def test_perpetual_lab_auto_start_requires_explicit_environment_flag(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(AUTO_START_PERPETUAL_LAB_ENV, "true")
    dashboard = _LabStub()

    _maybe_auto_start_perpetual_lab(dashboard, tmp_path)

    assert dashboard.calls == 1


def test_perpetual_lab_auto_start_does_not_repeat_when_report_exists(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(AUTO_START_PERPETUAL_LAB_ENV, "1")
    (tmp_path / "perpetual_strategy_lab.json").write_text("{}", encoding="utf-8")
    dashboard = _LabStub()

    _maybe_auto_start_perpetual_lab(dashboard, tmp_path)

    assert dashboard.calls == 0


def test_completed_perpetual_study_status_is_recovered_from_volume_after_restart(
    tmp_path,
):
    report = {
        "reports": {
            "1x": {
                "untouched_final_holdout": {
                    "win_rate_pct": 80.0,
                    "equal_weight_return_pct": 1.25,
                    "profit_factor": 1.4,
                    "trades": 40,
                    "liquidations": 0,
                },
                "acceptance_gate": {"passed": False},
            }
        }
    }
    (tmp_path / "perpetual_ema_walkforward.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    dashboard = Dashboard(tmp_path, token="test-token")

    status = dashboard.perpetual_study_status()

    assert status["status"] == "completed"
    assert status["orders_submitted"] is False
    assert status["demo_ready"] is False
    assert status["summary"]["1x"]["win_rate_pct"] == 80.0
    assert status["summary"]["1x"]["passed"] is False
