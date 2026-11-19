from __future__ import annotations

from tradingagents.paper_experiment.perpetual_demo.server import (
    PerpetualDemoController,
)


class ClosedTradeBroker:
    def __init__(self, closed_rows, order_rows):
        self.closed_rows = closed_rows
        self.order_rows = order_rows
        self.closed_query = None
        self.order_query = None

    def get_closed_pnl(self, **kwargs):
        self.closed_query = kwargs
        return self.closed_rows

    def get_order_history(self, **kwargs):
        self.order_query = kwargs
        return self.order_rows


def _controller(broker):
    controller = PerpetualDemoController.__new__(PerpetualDemoController)
    controller.broker = broker
    return controller


def _meta():
    return {
        "side": "long",
        "entry_time_ms": 1_700_000_000_000,
        "venue_entry_time_ms": 1_700_000_000_000,
        "venue_entry_price": 75.10,
        "entry_price_mark": 75.10,
    }


def test_closed_trade_reconciler_identifies_exchange_stop_loss_fill():
    broker = ClosedTradeBroker(
        [
            {
                "symbol": "SOLUSDT",
                "order_id": "exit-order-1",
                "side": "Sell",
                "closed_pnl": -0.021,
                "open_fee": 0.003,
                "close_fee": 0.003,
                "closed_size": 0.1,
                "avg_entry_price": 75.10,
                "avg_exit_price": 74.98,
                "created_time_ms": 1_700_000_600_000,
                "updated_time_ms": 1_700_000_600_100,
            }
        ],
        [
            {
                "order_id": "exit-order-1",
                "order_link_id": "sl-child-order",
                "order_status": "Filled",
                "stop_order_type": "Stop",
                "trigger_price": 74.98,
            }
        ],
    )
    result = _controller(broker)._closed_trade_details(
        "SOLUSDT", _meta(), 1_700_000_700_000
    )

    assert result["attribution_status"] == "matched"
    assert result["exit_reason"] == "stop_loss"
    assert result["closed_pnl"] == -0.021
    assert result["fees_total"] == 0.006
    assert result["avg_entry_price"] == 75.10
    assert result["avg_exit_price"] == 74.98
    assert broker.closed_query["symbol"] == "SOLUSDT"
    assert broker.order_query["order_id"] == "exit-order-1"


def test_closed_trade_reconciler_identifies_exchange_take_profit_fill():
    broker = ClosedTradeBroker(
        [
            {
                "symbol": "BNBUSDT",
                "order_id": "exit-order-2",
                "side": "Sell",
                "closed_pnl": 0.017,
                "open_fee": 0.002,
                "close_fee": 0.002,
                "closed_size": 0.01,
                "avg_entry_price": 592.65,
                "avg_exit_price": 594.42,
                "created_time_ms": 1_700_000_600_000,
                "updated_time_ms": 1_700_000_600_100,
            }
        ],
        [
            {
                "order_id": "exit-order-2",
                "order_link_id": "tp-child-order",
                "order_status": "Filled",
                "stop_order_type": "TakeProfit",
                "trigger_price": 594.42,
            }
        ],
    )
    meta = _meta()
    meta.update({"venue_entry_price": 592.65, "entry_price_mark": 592.65})
    result = _controller(broker)._closed_trade_details(
        "BNBUSDT", meta, 1_700_000_700_000
    )

    assert result["exit_reason"] == "take_profit"
    assert result["closed_pnl"] == 0.017
    assert result["close_order_id"] == "exit-order-2"


def test_closed_trade_reconciler_does_not_guess_when_history_is_missing():
    broker = ClosedTradeBroker([], [])
    result = _controller(broker)._closed_trade_details(
        "SOLUSDT", _meta(), 1_700_000_700_000
    )

    assert result == {"attribution_status": "closed_pnl_not_found"}


def test_dashboard_includes_closed_pnl_and_reasons_in_execution_activity():
    from tradingagents.paper_experiment.perpetual_demo.server import HTML

    assert "Closed trades" in HTML
    assert "renderClosedPnl" in HTML
    assert "reason=" in HTML
    assert "Protection type" in HTML
