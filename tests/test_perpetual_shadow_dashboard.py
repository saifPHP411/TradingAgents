from __future__ import annotations

from tradingagents.paper_experiment.dashboard.server import (
    AUTO_START_PERPETUAL_SHADOW_ENV,
    PERPETUAL_INDEX_HTML,
    Dashboard,
    _maybe_auto_start_perpetual_shadow,
)


def test_shadow_forward_is_research_only_and_refuses_non_perpetual_dashboard():
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.perpetual_only_mode = False

    result = dashboard.start_perpetual_shadow()

    assert result["status"] == "error"
    assert result["orders_submitted"] is False
    assert result["execution_enabled"] is False
    assert result["demo_ready"] is False
    assert not hasattr(dashboard, "_perpetual_shadow_thread")


def test_shadow_forward_panel_polls_authenticated_read_only_status():
    assert "id=\"shadow-badge\"" in PERPETUAL_INDEX_HTML
    assert "id=\"shadow-results\"" in PERPETUAL_INDEX_HTML
    assert '/api/perpetual/shadow/status' in PERPETUAL_INDEX_HTML
    assert "SHADOW_TIMER=setInterval(refreshShadow,60000)" in PERPETUAL_INDEX_HTML
    assert "no candidate is selected" in PERPETUAL_INDEX_HTML.lower()
    assert "No orders." in PERPETUAL_INDEX_HTML


def test_shadow_forward_auto_start_is_opt_in(monkeypatch, tmp_path):
    class FakeDashboard:
        calls = 0

        def start_perpetual_shadow(self):
            self.calls += 1
            return {"status": "starting"}

    dashboard = FakeDashboard()
    monkeypatch.delenv(AUTO_START_PERPETUAL_SHADOW_ENV, raising=False)
    _maybe_auto_start_perpetual_shadow(dashboard, tmp_path)
    assert dashboard.calls == 0

    monkeypatch.setenv(AUTO_START_PERPETUAL_SHADOW_ENV, "1")
    _maybe_auto_start_perpetual_shadow(dashboard, tmp_path)
    assert dashboard.calls == 1
