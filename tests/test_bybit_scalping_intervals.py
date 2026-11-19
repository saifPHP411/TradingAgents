from __future__ import annotations

import json

import pytest

from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.transport import HttpResponse


@pytest.mark.parametrize(
    ("interval", "code", "duration_ms"),
    [("1m", "1", 60_000), ("5m", "5", 300_000), ("15m", "15", 900_000)],
)
def test_public_klines_support_scalping_intervals(interval, code, duration_ms):
    server_ms = 1_800_000_000_000
    current_open = server_ms // duration_ms * duration_ms
    previous_open = current_open - duration_ms
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
        assert params["category"] == "linear"
        assert params["interval"] == code
        return HttpResponse(200, json.dumps(payload))

    client = BybitLinearPublicClient(base_url="https://api-demo.bybit.com", transport=transport)
    frame = client.klines(
        "BTCUSDT",
        start_ms=previous_open,
        end_ms=server_ms,
        include_current=True,
        interval=interval,
    )
    assert len(frame) == 2
    assert int(frame.index[-1].timestamp() * 1000) == current_open


def test_public_klines_reject_unknown_scalping_interval():
    client = BybitLinearPublicClient(base_url="https://api-demo.bybit.com")
    with pytest.raises(ValueError, match="unsupported Bybit kline interval"):
        client.klines("BTCUSDT", start_ms=1_800_000_000_000, end_ms=1_800_000_060_000, interval="2m")
