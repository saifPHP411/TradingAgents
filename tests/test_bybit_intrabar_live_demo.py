from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.transport import HttpResponse
from tradingagents.paper_experiment.perpetual_demo import server as demo_server


def test_public_klines_include_forming_candle_only_when_requested():
    interval_ms = 4 * 60 * 60 * 1000
    server_ms = 1_800_000_000_000
    current_open = server_ms // interval_ms * interval_ms
    previous_open = current_open - interval_ms
    payload = {
        "retCode": 0,
        "result": {
            "list": [
                [str(current_open), "100", "102", "99", "101", "20", "2020"],
                [str(previous_open), "98", "101", "97", "100", "30", "3000"],
            ]
        },
        "time": server_ms,
    }

    def transport(*, url, params, headers, timeout):
        assert url.endswith("/v5/market/kline")
        assert params["interval"] == "240"
        return HttpResponse(200, json.dumps(payload))

    client = BybitLinearPublicClient(
        base_url="https://api.bybit.com",
        transport=transport,
    )
    completed = client.klines(
        "BTCUSDT",
        start_ms=previous_open,
        end_ms=server_ms,
    )
    including_current = client.klines(
        "BTCUSDT",
        start_ms=previous_open,
        end_ms=server_ms,
        include_current=True,
    )

    assert len(completed) == 1
    assert len(including_current) == 2
    assert int(including_current.index[-1].timestamp() * 1000) == current_open


class _IntrabarPublic:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.include_current_values: list[bool] = []

    def klines(self, symbol, *, start_ms, end_ms, include_current=False):
        self.include_current_values.append(include_current)
        return self.frame.copy()


class _IntrabarBroker:
    configured = True

    def __init__(self):
        self.marks = iter((104.0, 100.0, 99.0))
        self.order_calls = []

    def get_wallet_balance(self):
        return {"status": "ok", "total_equity_usd": 50.0}

    def get_positions(self):
        return []

    def mark_price(self, symbol):
        return next(self.marks)

    def open_market_position(self, **kwargs):
        self.order_calls.append(kwargs)
        return {
            "quantity": 0.01,
            "order_id": "demo-order-1",
            "notional_usdt": 1.0,
        }


def test_demo_controller_enters_on_live_intrabar_signal_once_per_4h_candle(
    tmp_path,
    monkeypatch,
):
    now = datetime(2026, 11, 19, 13, 0, tzinfo=UTC)
    monkeypatch.setattr(demo_server.time, "time", lambda: now.timestamp())
    current_open = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
    index = pd.date_range(end=current_open, periods=230, freq="4h")
    frame = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
            "volume": [10.0] * len(index),
        },
        index=index,
    )
    public = _IntrabarPublic(frame)
    broker = _IntrabarBroker()
    controller = demo_server.PerpetualDemoController(
        tmp_path,
        broker=broker,
        public_client=public,
        symbols=("BTCUSDT",),
    )

    def live_features(data, config):
        result = data.copy()
        result["atr"] = 2.0
        result["mean_target"] = 103.0
        result["signal"] = 0
        result.loc[result.index[-1], "signal"] = 1
        return result

    monkeypatch.setattr(demo_server, "_features", live_features)

    # At 104 the long mean target is no longer favorable; at 100 it becomes valid.
    controller._strategy_cycle()
    assert broker.order_calls == []
    controller._strategy_cycle()
    assert len(broker.order_calls) == 1
    order = broker.order_calls[0]
    assert order["side"] == "long"
    assert order["stop_loss"] == 96.8
    assert order["take_profit"] == 103.0

    # The persisted per-candle attempt prevents a duplicate order on later polls.
    controller._strategy_cycle()
    assert len(broker.order_calls) == 1
    assert public.include_current_values == [True, True, True]
