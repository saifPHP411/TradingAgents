"""Regression coverage for the live broker execution dashboard view."""

from tradingagents.paper_experiment.dashboard.engine_view import trades_view


def test_filled_broker_order_is_reported_as_execution_before_trade_settlement():
    status = {
        "orders": {
            "client-1": {
                "client_order_id": "client-1",
                "broker_order_id": "broker-1",
                "symbol": "BNB/USDT",
                "side": "buy",
                "quantity": 0.0153,
                "filled_quantity": 0.0153,
                "filled_avg_price": 652.57,
                "status": "filled",
                "submitted_at": "2026-11-12T14:54:01.723000+00:00",
            }
        }
    }

    result = trades_view([], status)

    assert result["submitted_orders"] == 1
    assert result["filled_orders"] == 1
    assert result["filled_executions"] == 1
    assert result["settled_trades"] == 0
    assert result["executions"] == [
        {
            "client_order_id": "client-1",
            "broker_order_id": "broker-1",
            "symbol": "BNB/USDT",
            "side": "buy",
            "filled_quantity": 0.0153,
            "average_fill_price": 652.57,
            "order_timestamp": "2026-11-12T14:54:01.723000+00:00",
            "order_status": "filled",
        }
    ]


def test_partial_fill_is_an_execution_but_unfilled_order_is_not():
    status = {
        "orders": {
            "partial": {
                "client_order_id": "partial",
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.01,
                "filled_quantity": 0.004,
                "filled_avg_price": 90000.0,
                "status": "partially_filled",
                "submitted_at": "2026-11-12T14:00:00+00:00",
            },
            "accepted": {
                "client_order_id": "accepted",
                "symbol": "ETH/USDT",
                "side": "buy",
                "quantity": 0.1,
                "filled_quantity": 0,
                "status": "accepted",
                "submitted_at": "2026-11-12T14:01:00+00:00",
            },
        }
    }

    result = trades_view([], status)

    assert result["submitted_orders"] == 2
    assert result["filled_orders"] == 1
    assert len(result["executions"]) == 1
    assert result["executions"][0]["client_order_id"] == "partial"
    assert result["executions"][0]["filled_quantity"] == 0.004
    assert result["settled_trades"] == 0


def test_execution_is_not_mislabeled_as_a_settled_trade():
    status = {
        "orders": {
            "entry": {
                "client_order_id": "entry",
                "symbol": "BNB/USDT",
                "side": "buy",
                "quantity": 1,
                "filled_quantity": 1,
                "filled_avg_price": 650,
                "status": "filled",
            }
        }
    }

    result = trades_view([], status)

    assert result["executions"][0]["average_fill_price"] == 650
    assert result["trades"] == []
    assert result["settled_trades"] == 0
