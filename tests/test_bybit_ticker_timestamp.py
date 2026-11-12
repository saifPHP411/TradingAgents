"""Regression tests for Bybit V5 ticker response timestamps.

Bybit puts the server timestamp on the response envelope, not in each
result.list ticker row. The execution-price seam must receive that timestamp
without inventing one locally.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tradingagents.crypto.bybit import BybitPublicClient
from tradingagents.crypto.transport import HttpResponse
from tradingagents.paper_experiment.broker import BybitDemoExecutionPrice, MarketDataError


def _client(payload):
    def transport(*, url, params, headers, timeout):
        assert url.endswith("/v5/market/tickers")
        return HttpResponse(200, json.dumps(payload))

    return BybitPublicClient(transport=transport)


def _payload(*, response_time=None):
    payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "spot",
            "list": [
                {
                    "symbol": "XRPUSDT",
                    "lastPrice": "1.0371",
                    "bid1Price": "1.0370",
                    "ask1Price": "1.0372",
                }
            ],
        },
    }
    if response_time is not None:
        payload["time"] = response_time
    return payload


def test_bybit_tickers_propagates_response_envelope_time_to_each_symbol_alias():
    now = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
    now_ms = str(int(now.timestamp() * 1000))
    client = _client(_payload(response_time=now_ms))

    tickers = client.tickers(["XRP/USDT"])

    assert tickers["XRP/USDT"]["time"] == now_ms
    assert tickers["XRPUSDT"]["time"] == now_ms
    assert tickers["XRP/USDT"]["lastPrice"] == "1.0371"


def test_bybit_execution_price_accepts_fresh_envelope_timestamp():
    now = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
    now_ms = str(int(now.timestamp() * 1000))
    client = _client(_payload(response_time=now_ms))
    seam = BybitDemoExecutionPrice(
        ["XRP/USDT"],
        client=client,
        max_age_seconds=5.0,
        clock=lambda: now,
    )

    assert seam.price("XRP/USDT") == pytest.approx(1.0371)
    details = seam.describe()
    assert details["is_current_price"] is True
    assert details["execution_data_last_snapshot_age_seconds"] == pytest.approx(0.0)


def test_bybit_execution_price_still_fails_closed_when_envelope_time_is_missing():
    now = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
    client = _client(_payload())
    seam = BybitDemoExecutionPrice(
        ["XRP/USDT"],
        client=client,
        max_age_seconds=5.0,
        clock=lambda: now,
    )

    with pytest.raises(MarketDataError, match="carries no timestamp"):
        seam.price("XRP/USDT")

    assert seam.describe()["is_current_price"] is False
