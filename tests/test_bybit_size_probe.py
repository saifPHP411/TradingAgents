"""No-network unit tests for read-only Bybit instrument size audit."""
from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from tradingagents.crypto.backtest.bybit_size_probe import (
    evaluate_public_filters,
    fetch_market_object,
    probe,
)


def instrument(symbol="XRPUSDT", *, step="1", min_qty="1", min_notional="5"):
    return {
        "symbol": symbol,
        "status": "Trading",
        "contractType": "LinearPerpetual",
        "quoteCoin": "USDT",
        "settleCoin": "USDT",
        "lotSizeFilter": {
            "qtyStep": step,
            "minOrderQty": min_qty,
            "minNotionalValue": min_notional,
        },
        "priceFilter": {"tickSize": "0.0001"},
    }


def ticker(symbol="XRPUSDT", mark="2"):
    return {"symbol": symbol, "markPrice": mark}


def test_tiny_50usdt_account_floors_quantity_before_checking_filters():
    info = evaluate_public_filters(instrument(), ticker(), "XRPUSDT")
    assert info["max_position_notional_usdt"] == "10.00"
    assert info["max_qty_at_cap"] == "4"
    assert info["estimated_notional_at_mark_usdt"] == "8"
    assert info["passes_public_quantity_and_notional_filters"] is True
    assert info["orders_submitted"] is False
    assert info["account_permissions_and_historical_filters_verified"] is False


def test_price_and_step_can_make_nominally_cheap_account_ineligible():
    info = evaluate_public_filters(
        instrument("BTCUSDT", step="0.001", min_qty="0.001"),
        ticker("BTCUSDT", mark="100000"),
        "BTCUSDT",
    )
    assert info["max_qty_at_cap"] == "0.000"
    assert info["passes_public_quantity_and_notional_filters"] is False


def test_min_order_notional_disqualifies_lots_even_when_qty_above_min():
    info = evaluate_public_filters(
        instrument(min_notional="9"),
        ticker(),
        "XRPUSDT",
    )
    assert info["max_qty_at_cap"] == "4"
    assert info["passes_public_quantity_and_notional_filters"] is False


@pytest.mark.parametrize("bad", ["0", "-1", "NaN", "Infinity", "unknown"])
def test_bad_steps_fail_closed(bad):
    with pytest.raises(ValueError, match="invalid qtyStep"):
        evaluate_public_filters(instrument(step=bad), ticker(), "XRPUSDT")


def test_nonperpetual_or_symbol_mismatch_fail_closed():
    with pytest.raises(ValueError, match="active linear"):
        evaluate_public_filters(instrument("BTCUSDT"), ticker(), "XRPUSDT")


def test_public_gets_are_read_only_and_check_response_identity():
    urls = []

    def transport(url):
        urls.append(url)
        sample = instrument() if "instruments-info" in url else ticker()
        return json.dumps({"retCode": 0, "result": {"list": [sample]}}).encode()

    result = probe("XRPUSDT", transport=transport)
    assert result["passes_public_quantity_and_notional_filters"]
    assert len(urls) == 2
    assert all("/v5/market/" in url and "category=linear" in url for url in urls)
    assert all("symbol=XRPUSDT" in url for url in urls)

    def wrong_transport(_url):
        return json.dumps({"retCode": 0, "result": {"list": [ticker("BTCUSDT")]}}).encode()

    with pytest.raises(ValueError, match="symbol mismatch"):
        fetch_market_object("/v5/market/tickers", "XRPUSDT", transport=wrong_transport)


def test_backup_host_is_official_and_provenance_is_recorded():
    visited = []

    def transport(url):
        visited.append(url)
        if url.startswith("https://api.bybit.com"):
            raise HTTPError(url, 403, "region blocked", {}, None)
        sample = instrument() if "instruments-info" in url else ticker()
        return json.dumps({"retCode": 0, "result": {"list": [sample]}}).encode()

    result = probe("XRPUSDT", transport=transport)
    assert result["source_api_host"] == "https://api.bytick.com"
    assert result["passes_public_quantity_and_notional_filters"]
    assert len(visited) == 3
    assert all("api.bybit.com" in url or "api.bytick.com" in url for url in visited)


def test_public_api_rejects_unapproved_host():
    with pytest.raises(ValueError, match="unsupported"):
        fetch_market_object(
            "/v5/market/tickers", "XRPUSDT",
            host="https://example.com",
            transport=lambda url: b"{}",
        )
