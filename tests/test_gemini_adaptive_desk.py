import json
import time
from datetime import UTC, datetime

import pytest

from tradingagents.crypto.bybit_linear_demo import BybitLinearDemoRejected
from tradingagents.crypto.scalping import ScalpingConfig
from tradingagents.paper_experiment.perpetual_demo import adaptive
from tradingagents.paper_experiment.perpetual_demo.server import PerpetualDemoController

CANDLE = 1_800_000_000_000
SYMBOL = "ETHUSDT"


def market(mark=100.0, signal=1):
    return {
        "mark_price": mark, "signal": signal, "suggested_long_stop": 99.7,
        "suggested_long_target": 100.65, "long_bias": True, "rsi": 55.0,
    }


def observe(desk, offset, mark=100.0, signal=1, symbol=SYMBOL):
    now = CANDLE + offset
    return desk.observe(symbol, market(mark, signal), CANDLE, now, now)


def test_shadow_fills_after_signal_learns_only_from_settled_data_and_survives_restart(tmp_path):
    desk = adaptive.AdaptiveDesk(tmp_path, adaptive.GeminiAnalyst(""), ScalpingConfig())
    assert observe(desk, 1_000) == "gemini_key_missing"
    assert observe(desk, 11_000, mark=100.03) == "gemini_key_missing"
    assert observe(desk, 21_000, mark=101.0, signal=0) == "no_sniper_signal"
    assert desk.status(CANDLE + 21_000)["performance"] == {}
    desk.close()

    reopened = adaptive.AdaptiveDesk(tmp_path, adaptive.GeminiAnalyst(""), ScalpingConfig())
    result = reopened.status(CANDLE + 22_000)
    assert result["sample_counts"] == {"target": 1}
    assert result["performance"]["baseline"]["mean_net_return_pct"] == pytest.approx(
        (100.65 - 100.03 - (100.65 + 100.03) * adaptive.STRESS_COST_PER_SIDE) / 100.03 * 100
    )
    assert result["demo_evidence_ready"] is False
    reopened.close()


def test_missing_quotes_and_nonfillable_entries_never_train_the_policy(tmp_path):
    desk = adaptive.AdaptiveDesk(tmp_path, adaptive.GeminiAnalyst(""), ScalpingConfig())
    observe(desk, 1_000)
    observe(desk, 40_000, signal=0)
    observe(desk, 1_000, symbol="BTCUSDT")
    observe(desk, 9_000, mark=100.4, symbol="BTCUSDT")
    state = desk.status(CANDLE + 41_000)
    assert state["performance"] == {}
    assert state["sample_counts"] == {"entry_not_fillable": 1, "missing_quotes": 1}
    desk.close()


def test_provider_vote_respects_durable_budget_and_binds_to_fresh_reference(tmp_path, monkeypatch):
    class Analyst(adaptive.GeminiAnalyst):
        def vote(self, evidence):
            assert evidence["direction"] == "long"
            assert evidence["prior_settled_shadow_mark_model"]["baseline"]["observations"] == 30
            return {"approve": True, "reason": "الاتجاه مؤكد", "model_version": self.model}

    monkeypatch.setattr(adaptive.time, "time_ns", lambda: (CANDLE + 2_000) * 1_000_000)
    desk = adaptive.AdaptiveDesk(tmp_path, Analyst("fixture"), ScalpingConfig(), calls_per_day=1)
    with desk._db() as db:
        db.executemany("""
            INSERT INTO shadow VALUES (?, ?, 'long', ?, ?, 100, 99, 102, ?, ?, ?, 'target')
        """, (
            (str(i), SYMBOL, CANDLE - 180_000, CANDLE - 180_000,
             CANDLE - 180_000, CANDLE - 179_500, 0.5)
            for i in range(30)
        ))
    assert observe(desk, 1_000) == "analyst_requested"
    desk._inflight.result(timeout=2)
    assert observe(desk, 2_000) == "waiting_for_quote_after_vote"
    assert observe(desk, 3_000, mark=100.09) == "professor_price_moved"
    assert observe(desk, 4_000) == "unanimous"
    desk.close()
    reopened = adaptive.AdaptiveDesk(tmp_path, Analyst("fixture"), ScalpingConfig(), calls_per_day=1)
    assert observe(reopened, 5_000, symbol="BTCUSDT") == "gemini_daily_budget"
    assert observe(reopened, 60_000) == "stale_or_invalid_market"
    assert reopened.status(CANDLE + 5_000)["recent_votes"][0]["status"] == "analyst_approved"
    reopened.close()


def test_gemini_http_uses_credential_header_and_rejects_non_boolean_vote(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return json.dumps({"candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"approve":"yes","reason":"test"}'}]},
            }]}).encode()

    def transport(request, timeout):
        assert request.full_url.startswith("https://generativelanguage.googleapis.com/v1beta/")
        assert request.get_header("X-goog-api-key") == "test-key"
        assert b"test-key" not in request.data
        assert json.loads(request.data)["generationConfig"]["responseJsonSchema"] == adaptive.SCHEMA
        assert timeout == 8
        return Response()

    monkeypatch.setattr(adaptive, "urlopen", transport)
    with pytest.raises(ValueError, match="Invalid Gemini vote"):
        adaptive.GeminiAnalyst("test-key").vote({"test": 1})


def test_demo_entry_never_reaches_broker_without_adaptive_consensus(tmp_path, monkeypatch):
    class Broker:
        configured = True
        public = object()

        def open_position(self, **kwargs):
            pytest.fail("adaptive entry should never reach order transport")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = PerpetualDemoController(
        tmp_path, broker=Broker(), symbols=(SYMBOL,), adaptive=True, adaptive_demo_enabled=True,
    )
    controller._mode = "demo"
    now = time.time_ns() // 1_000_000
    candle = now // 60_000 * 60_000
    snapshot = {
        **market(), "forming_candle_open": datetime.fromtimestamp(candle / 1000, UTC).isoformat(),
        "mark_received_ms": now,
    }
    controller._process_scan((SYMBOL,), [(snapshot, None)], 50, {}, {}, {}, {}, {})
    assert controller._strategy_status["last_learning_decision"]["status"] == "gemini_key_missing"
    with pytest.raises(BybitLinearDemoRejected, match="qualifying forward shadow evidence"):
        controller.start_strategy()
    controller.shutdown()
