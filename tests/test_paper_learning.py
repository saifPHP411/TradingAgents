from tradingagents.paper_experiment.learning import (
    build_learning_report,
    normalize_execution_rows,
)


def _fill(exec_id, order_id, side, qty, price, fee, time):
    return {
        "execId": exec_id,
        "orderLinkId": order_id,
        "symbol": "BTCUSDT",
        "side": side,
        "execQty": str(qty),
        "execPrice": str(price),
        "execFee": str(fee),
        "feeCurrency": "USDT",
        "execTime": str(time),
    }


def test_normalize_execution_rows_deduplicates_and_rejects_invalid_rows():
    rows = [
        _fill("e1", "buy-1", "Buy", 0.1, 100, 0.01, 1000),
        _fill("e1", "buy-1", "Buy", 0.1, 100, 0.01, 1000),
        _fill("bad", "bad", "Buy", 0, 100, 0.01, 1000),
    ]
    result = normalize_execution_rows(rows)
    assert len(result) == 1
    assert result[0]["execution_id"] == "e1"


def test_fifo_pairing_calculates_net_pnl_from_actual_quote_fees():
    worker = {
        "orders": {
            "buy-1": {
                "decision_id": "d1",
                "decision_snapshot": {"strategy_name": "tjr", "stop_loss": 95},
            }
        }
    }
    rows = [
        _fill("b1", "buy-1", "Buy", 1, 100, 0.1, 1000),
        _fill("s1", "sell-1", "Sell", 1, 110, 0.1, 2000),
    ]
    report = build_learning_report(worker, rows, history_complete=True, minimum_trades=1)
    assert report["closed_trades"] == 1
    assert report["metrics_ready"] is True
    assert report["trades"][0]["gross_pnl"] == 10
    assert report["trades"][0]["fees"] == 0.2
    assert abs(report["trades"][0]["net_pnl"] - 9.8) < 1e-9
    assert report["trades"][0]["decision_id"] == "d1"


def test_net_metrics_are_withheld_when_fee_currency_is_not_quote_asset():
    worker = {"orders": {}}
    buy = _fill("b1", "buy-1", "Buy", 1, 100, 0.1, 1000)
    sell = _fill("s1", "sell-1", "Sell", 1, 110, 0.1, 2000)
    buy["feeCurrency"] = "BNB"
    report = build_learning_report(worker, [buy, sell], history_complete=True, minimum_trades=1)
    assert report["closed_trades"] == 1
    assert report["metrics_ready"] is False
    assert report["net_pnl"] is None
    assert report["trades"][0]["net_pnl"] is None


def test_incomplete_history_never_marks_metrics_ready():
    worker = {"orders": {}}
    rows = [
        _fill("b1", "buy-1", "Buy", 1, 100, 0.1, 1000),
        _fill("s1", "sell-1", "Sell", 1, 110, 0.1, 2000),
    ]
    report = build_learning_report(worker, rows, history_complete=False, minimum_trades=1)
    assert report["metrics_ready"] is False
    assert report["net_profit_factor"] is None
    assert any("truncated" in item.lower() for item in report["suggestions"])


def test_insufficient_sample_does_not_claim_a_strategy_edge():
    worker = {"orders": {}}
    rows = [
        _fill("b1", "buy-1", "Buy", 1, 100, 0.1, 1000),
        _fill("s1", "sell-1", "Sell", 1, 110, 0.1, 2000),
    ]
    report = build_learning_report(worker, rows, history_complete=True, minimum_trades=20)
    assert report["metrics_ready"] is False
    assert report["net_win_rate"] is None
    assert any("more completed trades" in item for item in report["suggestions"])


def test_worker_order_reconciliation_preserves_decision_attribution():
    from types import SimpleNamespace

    from tradingagents.paper_experiment.broker.trading import PaperTradingWorker, WorkerState

    worker = PaperTradingWorker.__new__(PaperTradingWorker)
    worker.state = WorkerState()
    worker._store = None
    order = SimpleNamespace(
        client_order_id="entry-1",
        broker_order_id="venue-1",
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        filled_quantity=0.1,
        filled_avg_price=100.0,
        status="filled",
        submitted_at="2026-01-01T00:00:00+00:00",
    )
    snapshot = {"strategy_name": "tjr", "stop_loss": 95.0}
    worker._record_order(order, decision_id="decision-1", decision_snapshot=snapshot)
    order.filled_avg_price = 101.0
    worker._record_order(order)
    record = worker.state.orders["entry-1"]
    assert record["decision_id"] == "decision-1"
    assert record["decision_snapshot"] == snapshot
    assert record["filled_avg_price"] == 101.0


def test_worker_loop_feedback_refresh_is_read_only_and_persisted(tmp_path):
    import json
    import time
    from types import SimpleNamespace

    from tradingagents.paper_experiment.broker.trading import PaperTradingWorker, WorkerState

    class FakeBroker:
        calls = 0

        def read_execution_history(self, *, max_pages):
            self.calls += 1
            assert max_pages == 10
            return [], False

    worker = PaperTradingWorker.__new__(PaperTradingWorker)
    worker.config = SimpleNamespace(
        execution_provider="bybit_demo",
        state_path=tmp_path / "worker_state.json",
    )
    worker._broker = FakeBroker()
    worker.state = WorkerState()
    worker._last_learning_refresh_monotonic = 0.0
    worker._refresh_learning_report_if_due()
    report_path = tmp_path / "learning_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert worker._broker.calls == 1
    assert report["available"] is True
    assert report["strategy_mutation_enabled"] is False
    assert report["automatic_promotion_enabled"] is False

    worker._last_learning_refresh_monotonic = time.monotonic()
    worker._refresh_learning_report_if_due()
    assert worker._broker.calls == 1
