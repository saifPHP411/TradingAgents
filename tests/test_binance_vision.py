from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from tradingagents.crypto.binance_vision import (
    BinanceVisionDataError,
    BinanceVisionPublicClient,
)


def _zip_csv(filename: str, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, csv_text)
    return buffer.getvalue()


def _client(archives: dict[str, bytes]) -> BinanceVisionPublicClient:
    def transport(*, url: str, timeout: float) -> bytes:
        for suffix, payload in archives.items():
            if url.endswith(suffix):
                return payload
        raise AssertionError(f"unexpected archive URL: {url}")

    return BinanceVisionPublicClient(transport=transport)


def test_instrument_info_is_research_only_linear_usdt_perpetual():
    instrument = BinanceVisionPublicClient(transport=lambda **_: b"x").instrument_info(
        "BTCUSDT"
    )

    assert instrument.symbol == "BTCUSDT"
    assert instrument.contract_type == "LinearPerpetual"
    assert instrument.quote_coin == "USDT"
    assert instrument.settle_coin == "USDT"
    assert instrument.max_leverage == 2.0


def test_binance_vision_traded_and_mark_candles_are_parsed_as_utc():
    start = 1_735_689_600_000
    interval = 4 * 60 * 60 * 1000
    end = start + 3 * interval
    rows = "\n".join(
        [
            f"{start},100,102,99,101,12,{start + interval - 1},0,0,0,0,0",
            f"{start + interval},101,103,100,102,13,{start + 2 * interval - 1},0,0,0,0,0",
            f"{start + 2 * interval},102,104,101,103,14,{start + 3 * interval - 1},0,0,0,0,0",
        ]
    )
    archives = {}
    for dataset in ("klines", "markPriceKlines"):
        suffix = (
            f"/{dataset}/BTCUSDT/4h/BTCUSDT-4h-2025-01.zip"
        )
        archives[suffix] = _zip_csv("BTCUSDT-4h-2025-01.csv", rows)
    client = _client(archives)

    traded = client.klines("BTCUSDT", start_ms=start, end_ms=end)
    mark = client.mark_price_klines("BTCUSDT", start_ms=start, end_ms=end)

    assert len(traded) == 3
    assert len(mark) == 3
    assert str(traded.index.tz) == "UTC"
    assert traded.index.equals(mark.index)
    assert traded.iloc[-1]["close"] == pytest.approx(103.0)
    assert "volume" in traded.columns
    assert "volume" not in mark.columns


def test_binance_vision_funding_keeps_negative_and_positive_rates():
    start = 1_735_689_600_000
    interval = 8 * 60 * 60 * 1000
    rows = (
        "calc_time,funding_interval_hours,last_funding_rate\n"
        f"{start},8,-0.0001\n"
        f"{start + interval},8,0.0002\n"
    )
    suffix = "/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2025-01.zip"
    client = _client(
        {suffix: _zip_csv("BTCUSDT-fundingRate-2025-01.csv", rows)}
    )

    rates = client.funding_rates(
        "BTCUSDT",
        start_ms=start,
        end_ms=start + interval,
        interval_minutes=480,
    )

    assert rates.iloc[0] == pytest.approx(-0.0001)
    assert rates.iloc[1] == pytest.approx(0.0002)
    assert str(rates.index.tz) == "UTC"


def test_binance_vision_rejects_invalid_host_and_missing_archives():
    with pytest.raises(ValueError, match="official Binance Vision"):
        BinanceVisionPublicClient(base_url="https://api.binance.com")

    client = BinanceVisionPublicClient(
        transport=lambda **_: b"not a zip archive"
    )
    with pytest.raises(BinanceVisionDataError, match="valid ZIP"):
        client.klines(
            "BTCUSDT",
            start_ms=1_735_689_600_000,
            end_ms=1_735_689_600_000 + 4 * 60 * 60 * 1000,
        )
