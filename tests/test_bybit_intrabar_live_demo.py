from __future__ import annotations

import json

from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.transport import HttpResponse


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
