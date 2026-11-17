from __future__ import annotations

import pytest

from tradingagents.crypto.binance_perpetual import BinanceUSDMPublicClient
from tradingagents.crypto.transport import HttpResponse


def _transport_for(payloads):
    def transport(*, url, params, headers, timeout):
        for suffix, payload in payloads.items():
            if url.endswith(suffix):
                return HttpResponse(200, __import__("json").dumps(payload))
        raise AssertionError(f"unexpected public endpoint: {url}")
    return transport


def test_binance_instrument_info_uses_market_rules_and_research_leverage_cap():
    exchange_info = {
        "symbols": [{
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100", "stepSize": "0.001"},
                {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "50", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }]
    }
    client = BinanceUSDMPublicClient(
        transport=_transport_for({
            "/fapi/v1/exchangeInfo": exchange_info,
            "/fapi/v1/fundingInfo": [],
        })
    )

    instrument = client.instrument_info("BTCUSDT")

    assert instrument.status == "Trading"
    assert instrument.contract_type == "LinearPerpetual"
    assert instrument.max_market_quantity == 50
    assert instrument.min_notional == 5
    assert instrument.max_leverage == 2
    assert instrument.funding_interval_minutes == 480


def test_binance_funding_reader_preserves_negative_funding_rates():
    start = 1_735_689_600_000
    payload = [
        {"symbol": "BTCUSDT", "fundingRate": "-0.00010000", "fundingTime": start},
        {"symbol": "BTCUSDT", "fundingRate": "0.00020000", "fundingTime": start + 8 * 60 * 60 * 1000},
    ]
    client = BinanceUSDMPublicClient(
        transport=_transport_for({"/fapi/v1/fundingRate": payload})
    )

    rates = client.funding_rates(
        "BTCUSDT",
        start_ms=start,
        end_ms=start + 8 * 60 * 60 * 1000,
        interval_minutes=480,
    )

    assert rates.iloc[0] == pytest.approx(-0.0001)
    assert rates.iloc[1] == pytest.approx(0.0002)


def test_binance_candles_exclude_incomplete_bars_and_return_utc_index():
    start = 1_735_689_600_000
    interval = 4 * 60 * 60 * 1000
    payload = [
        [start, "100", "102", "99", "101", "10", start + interval - 1, "1000", 100, "5", "500", "0"],
        [start + interval, "101", "103", "100", "102", "11", start + 2 * interval - 1, "1100", 110, "6", "600", "0"],
        [start + 2 * interval, "102", "104", "101", "103", "12", start + 3 * interval - 1, "1200", 120, "7", "700", "0"],
    ]
    client = BinanceUSDMPublicClient(
        transport=_transport_for({"/fapi/v1/klines": payload})
    )

    frame = client.klines(
        "BTCUSDT",
        start_ms=start,
        end_ms=start + 2 * interval,
    )

    assert len(frame) == 2
    assert str(frame.index.tz) == "UTC"
    assert frame.iloc[-1]["close"] == 102


def test_binance_adapter_rejects_unapproved_host():
    with pytest.raises(ValueError, match="official Binance"):
        BinanceUSDMPublicClient(base_url="https://api-demo.bybit.com")
