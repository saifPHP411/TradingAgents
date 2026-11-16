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
