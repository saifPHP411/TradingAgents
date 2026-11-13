"""Dashboard contract tests for provider identity and truthful fallbacks."""

from tradingagents.paper_experiment.dashboard.engine_view import engine_view


def test_engine_view_does_not_guess_alpaca_when_provider_mode_is_missing():
    view = engine_view({})
    assert view["mode"] == "UNKNOWN"
    assert view["live"] is False


def test_engine_view_preserves_explicit_bybit_demo_mode():
    view = engine_view({"mode": "BYBIT_DEMO"})
    assert view["mode"] == "BYBIT_DEMO"
    assert view["live"] is False
