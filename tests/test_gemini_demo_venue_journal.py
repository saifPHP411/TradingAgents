"""Bybit Demo closed PnL is a separate, idempotent evidence channel."""
from tradingagents.crypto.scalping import ScalpingConfig
from tradingagents.paper_experiment.perpetual_demo.adaptive import AdaptiveDesk, GeminiAnalyst
from tradingagents.paper_experiment.perpetual_demo.server import HTML, PerpetualDemoController

NOW = 1_800_000_000_000
SYMBOL = "BTCUSDT"
ORDER = "close-trade-123456789"


def bybit_close(pnl, *, order_id=ORDER):
    return {
        "attribution_status": "matched",
        "close_order_id": order_id,
        "closed_pnl": pnl,
        "fees_total": 0.04,
        "closed_size": "0.01",
        "avg_entry_price": "100",
        "avg_exit_price": "101",
    }


def test_fee_adjusted_venue_net_pnl_is_not_charged_twice(tmp_path):
    desk = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    assert desk.record_venue_close(SYMBOL, "long", bybit_close("1.25"), NOW)
    assert not desk.record_venue_close(SYMBOL, "long", bybit_close("1.25"), NOW + 10)
    current = desk.status(NOW + 100)["venue_performance"]
    assert current["source"] == "bybit_demo_closed_pnl_not_live_money"
    assert current["recorded_closes"] == 1
    assert current["winning_closes"] == 1
    assert current["net_realized_usdt"] == 1.25  # 0.04 fee NOT deducted again
    assert current["profit_factor"] is None
    assert desk.status(NOW + 100)["performance"] == {}
    assert not desk.ready(NOW + 100)
    desk.close()

    reopened = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    assert reopened.status(NOW + 100)["venue_performance"] == current
    assert reopened.record_venue_close(
        SYMBOL, "short", bybit_close(-1.5, order_id="close-trade-222222222"), NOW + 200
    )
    net = reopened.status(NOW + 300)["venue_performance"]
    assert net["recorded_closes"] == 2
    assert net["net_realized_usdt"] == -0.25
    assert net["profit_factor"] == 1.25 / 1.5
    assert net["net_win_rate"] == 0.5
    reopened.close()


def test_outcomes_must_be_matching_valid_venue_closes(tmp_path):
    desk = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    for detail in [
        {**bybit_close(1), "attribution_status": "lookup_error"},
        {**bybit_close(1), "close_order_id": None},
        {**bybit_close(1), "closed_pnl": float("nan")},
        {**bybit_close(1), "closed_pnl": True},
        {**bybit_close(1), "closed_size": "0"},
        {**bybit_close(1), "avg_entry_price": "not-a-number"},
    ]:
        assert not desk.record_venue_close(SYMBOL, "long", detail, NOW)
    assert not desk.record_venue_close("INVALID!@#", "long", bybit_close(1), NOW)
    assert not desk.record_venue_close(SYMBOL, "long", bybit_close(1), -1)
    assert desk.status(NOW + 100)["venue_performance"]["recorded_closes"] == 0
    desk.close()


def test_venue_ledger_appears_only_after_record_is_observed(tmp_path):
    desk = AdaptiveDesk(tmp_path, GeminiAnalyst(""), ScalpingConfig())
    assert desk.record_venue_close(SYMBOL, "long", bybit_close(2), NOW)
    assert desk.status(NOW)["venue_performance"]["recorded_closes"] == 0
    assert desk.status(NOW + 1)["venue_performance"]["recorded_closes"] == 1
    desk.close()


def test_controller_capture_and_mobile_labels_are_separate_from_shadow(tmp_path):
    class Broker:
        configured = False
        public = object()

    controller = PerpetualDemoController(
        tmp_path, broker=Broker(), symbols=(SYMBOL,), adaptive=True,
    )
    controller._remember_closed_venue(SYMBOL, "long", bybit_close("0.12"), NOW)
    view = controller.adaptive.status(NOW + 100)
    assert view["venue_performance"]["net_realized_usdt"] == 0.12
    assert view["demo_evidence_ready"] is False
    assert "id=\"venue-learning-net\"" in HTML
    assert "id=\"venue-learning-pf\"" in HTML
    assert "separate from modeled shadow trades" in HTML
    controller.shutdown()
