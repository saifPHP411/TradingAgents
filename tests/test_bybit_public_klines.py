from __future__ import annotations

import json
from datetime import UTC, datetime

from tradingagents.crypto.bybit import BybitPublicClient
from tradingagents.crypto.transport import CryptoDataError, HttpResponse
from tradingagents.crypto.types import candle_from_bybit_kline


def _row(open_ms: int) -> list[str]:
    return [str(open_ms), "100", "102", "99", "101", "1.25", "126.25"]


def test_bybit_kline_parser_derives_the_close_time():
    candle = candle_from_bybit_kline("BTC/USDT", "1m", _row(60_000))
    assert candle is not None
    assert candle.open_time == datetime.fromtimestamp(60, tz=UTC)
    assert candle.close_time == datetime.fromtimestamp(119.999, tz=UTC)
    assert candle.volume == 1.25


def test_bybit_kline_parser_rejects_malformed_rows():
    assert candle_from_bybit_kline("BTC/USDT", "1m", ["60000", "100"]) is None
    assert candle_from_bybit_kline(
        "BTC/USDT", "1m", ["60000", "bad", "102", "99", "101", "1", "126"]
    ) is None


def test_bybit_spot_klines_page_backwards_and_return_oldest_first():
    first_page = [_row(stamp * 60_000) for stamp in range(2000, 1000, -1)]
    second_page = [_row(1000 * 60_000)]
    pages = [first_page, second_page]
    calls = []

    def transport(*, url, params, headers, timeout):
        calls.append({"url": url, "params": dict(params)})
        page = pages.pop(0)
        return HttpResponse(
            200,
            json.dumps({"retCode": 0, "retMsg": "OK", "result": {"list": page}}),
        )

    client = BybitPublicClient(transport=transport)
    candles = client.klines("BTC/USDT", "1m", limit=1001)

    assert len(candles) == 1001
    assert candles[0].open_time == datetime.fromtimestamp(1000 * 60, tz=UTC)
    assert candles[-1].open_time == datetime.fromtimestamp(2000 * 60, tz=UTC)
    assert all(left.open_time < right.open_time for left, right in zip(candles, candles[1:]))
    assert calls[0]["url"].endswith("/v5/market/kline")
    assert calls[0]["params"] == {
        "category": "spot",
        "symbol": "BTCUSDT",
        "interval": "1",
        "limit": 1000,
    }
    assert calls[1]["params"]["end"] == 1001 * 60_000 - 1


def test_bybit_kline_nonzero_ret_code_is_a_data_error():
    def transport(*, url, params, headers, timeout):
        return HttpResponse(
            200,
            json.dumps(
                {
                    "retCode": 10001,
                    "retMsg": "invalid interval",
                    "result": {"list": []},
                }
            ),
        )

    client = BybitPublicClient(transport=transport)
    try:
        client.klines("BTC/USDT", "1m", limit=1)
    except CryptoDataError as exc:
        assert "10001" in str(exc)
    else:
        raise AssertionError("a Bybit retCode error must not become an empty candle set")


def test_bybit_client_uses_demo_public_market_data_without_credentials():
    client = BybitPublicClient()
    described = client.describe()
    assert described["crypto_data_provider"] == "bybit_public"
    assert described["crypto_data_requires_key"] is False
    assert described["crypto_data_rest_host"] == "https://api-demo.bybit.com"
