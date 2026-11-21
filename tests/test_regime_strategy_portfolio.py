"""Forward-safe regime selection must never authorize an automatic Bybit entry."""
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.crypto import regime_selector as router
from tradingagents.crypto.scalping import ScalpingConfig
from tradingagents.paper_experiment.perpetual_demo.adaptive import AdaptiveDesk, GeminiAnalyst
from tradingagents.paper_experiment.perpetual_demo.server import HTML, PerpetualDemoController


def candles(period, count, direction=1, step=0.2):
    index = pd.date_range("2026-11-20T00:00:00Z", periods=count, freq=f"{period}min")
    prices = [110 + direction * i * step for i in range(count)]
    return pd.DataFrame({
        "open": prices, "close": prices,
        "high": [p + 0.05 for p in prices],
        "low": [p - 0.05 for p in prices],
        "volume": [100.0] * count,
    }, index=index)


def fixtures(direction=1, step=0.2):
    five = candles(5, 300, direction=direction, step=step / 3)
    fifteen = candles(15, 100, direction=direction, step=step)
    now = fifteen.index[-1] + pd.Timedelta(minutes=17)
    return five, fifteen, int(now.timestamp() * 1000)


def test_trending_up_routes_to_donchian_without_trade_permission():
    five, fifteen, now = fixtures()
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    assert result["regime"] == "trend_up"
    assert result["selected_strategy"] == "donchian_breakout"
    assert result["selected_signal"] == 1
    assert result["shadow_only"] is True
    assert result["execution_authorized"] is False
    assert len(result["candidates"]) == 6
    assert not next(c for c in result["candidates"] if c["strategy"] == "bollinger_rsi_reversion")["regime_compatible"]
    assert "funding_basis_arbitrage" in result["unimplemented_families"]


def test_downtrend_routes_short_without_opening_orders():
    five, fifteen, now = fixtures(direction=-1)
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=90)
    assert result["regime"] == "trend_down"
    assert result["selected_strategy"] == "donchian_breakout"
    assert result["selected_signal"] == -1
    assert result["stop_price_model"] > result["target_price_model"]
    assert result["execution_authorized"] is False


def test_range_chooses_predeclared_mean_reversion_priority(monkeypatch):
    five, fifteen, now = fixtures()
    monkeypatch.setattr(router, "_regime", lambda _frame: ("range", {"test_only": True}))
    monkeypatch.setattr(router, "_bollinger", lambda _frame: 1)
    monkeypatch.setattr(router, "_vwap", lambda _frame: 1)
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    assert result["selected_strategy"] == "bollinger_rsi_reversion"
    assert result["selected_signal"] == 1
    assert result["execution_authorized"] is False


def test_opposing_range_signals_abstain(monkeypatch):
    five, fifteen, now = fixtures()
    monkeypatch.setattr(router, "_regime", lambda _frame: ("range", {}))
    monkeypatch.setattr(router, "_bollinger", lambda _frame: 1)
    monkeypatch.setattr(router, "_vwap", lambda _frame: -1)
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    assert result["selection_reason"] == "conflicting_strategy_signals"
    assert result["selected_strategy"] is None
    assert result["selected_signal"] == 0


@pytest.mark.parametrize("regime", ["transition", "extreme_risk"])
def test_unknown_or_extreme_market_abstains(monkeypatch, regime):
    five, fifteen, now = fixtures()
    monkeypatch.setattr(router, "_regime", lambda _frame: (regime, {}))
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    assert result["selection_reason"] == "unsafe_or_unclassified_regime"
    assert result["selected_strategy"] is None


def test_small_target_cannot_pay_stressed_round_trip_cost(monkeypatch):
    five, fifteen, now = fixtures(step=0.01)
    monkeypatch.setattr(router, "_regime", lambda _frame: ("trend_up", {}))
    monkeypatch.setattr(router, "_donchian", lambda _frame, _trend: 1)
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    assert result["selection_reason"] == "modeled_target_does_not_cover_costs"
    assert result["selected_signal"] == 0


def test_completed_bars_are_mandatory_and_future_gaps_fail_closed():
    five, fifteen, now = fixtures()
    for changed_five, changed_fifteen, at in (
        (five, fifteen, now - 20 * 60_000),
        (five, fifteen, now + 60 * 60_000),
        (five.drop(five.index[-10]), fifteen, now),
        (five, fifteen.drop(fifteen.index[-20]), now),
    ):
        with pytest.raises(ValueError):
            router.evaluate_regime_menu(changed_five, changed_fifteen, now_ms=at, reference_price=130)
    with pytest.raises(ValueError):
        router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=float("nan"))


def test_shadow_journal_is_idempotent_persistent_and_never_demo_evidence(tmp_path):
    five, fifteen, now = fixtures()
    report = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    desk = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    assert desk.record_regime_choice("ETHUSDT", report, now)
    assert not desk.record_regime_choice("ETHUSDT", report, now + 10)
    status = desk.status(now + 1)
    assert status["latest_regime_choices"][0]["selected_strategy"] == "donchian_breakout"
    assert status["latest_regime_choices"][0]["execution_authorized"] is False
    assert status["performance"] == {}
    assert status["demo_evidence_ready"] is False
    invalid = {**report, "execution_authorized": True}
    assert not desk.record_regime_choice("BTCUSDT", invalid, now)
    desk.close()
    reopened = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    assert reopened.status(now + 1)["latest_regime_choices"] == status["latest_regime_choices"]
    reopened.close()


def test_authenticated_dashboard_has_research_only_regime_labels(tmp_path):
    assert "id=\"regime-choices\"" in HTML
    assert "NO ORDERS" in HTML
    assert "RESEARCH ONLY" in HTML
    class Broker:
        configured = False
        public = object()

    controller = PerpetualDemoController(tmp_path, broker=Broker(), symbols=("ETHUSDT",), adaptive=True)
    five, fifteen, now = fixtures()
    result = router.evaluate_regime_menu(five, fifteen, now_ms=now, reference_price=130)
    candle_ms = now // 60_000 * 60_000
    market = {
        "mark_price": 130.0, "signal": 0,
        "forming_candle_open": datetime.fromtimestamp(candle_ms / 1000, UTC).isoformat(),
        "mark_received_ms": now, "regime_research": result,
    }
    controller._observe_candidate("ETHUSDT", market, now)
    status = controller.learning_status()
    assert status["latest_regime_choices"][0]["selected_strategy"] == "donchian_breakout"
    assert status["demo_evidence_ready"] is False
    controller.shutdown()
