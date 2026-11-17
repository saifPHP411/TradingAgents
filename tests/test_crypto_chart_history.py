"""Tests for the bounded, read-only dashboard chart history service."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tradingagents.paper_experiment.dashboard.chart import CryptoChartService


def _candle(symbol="BTC/USDT", interval="1h", *, offset=0, close_delta=-1):
    opened = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=offset)
    closed = opened + timedelta(hours=1, seconds=close_delta)
    base = 100.0 + offset
    return SimpleNamespace(
        symbol=symbol, interval=interval, open_time=opened, close_time=closed,
        open=base, high=base + 3, low=base - 2, close=base + 1,
    )


class _Client:
    def __init__(self, candles):
        self.candles = candles
        self.calls = []

    def klines(self, symbol, interval, *, limit):
        self.calls.append((symbol, interval, limit))
        return self.candles


@pytest.mark.unit
def test_chart_returns_only_completed_normalised_candles():
    client = _Client([
        _candle(offset=0), _candle(offset=1),
        _candle(offset=2, close_delta=3600),
    ])
    service = CryptoChartService(client_factory=lambda: client)
    result = service.get("BTCUSDT", "1h", 60)
    assert result["status"] == "ok"
    assert result["symbol"] == "BTC/USDT"
    assert result["provider"] == "binance_public_spot"
    assert len(result["candles"]) == 2
    assert result["candles"][0]["time"] < result["candles"][1]["time"]
    assert client.calls == [("BTC/USDT", "1h", 60)]


@pytest.mark.unit
def test_chart_caches_requests_and_returns_defensive_copies():
    clock = [10.0]
    client = _Client([_candle()])
    service = CryptoChartService(
        client_factory=lambda: client, clock=lambda: clock[0], ttl_seconds=20,
    )
    first = service.get("BTC/USDT", "1h", 60)
    first["candles"][0]["close"] = -1
    second = service.get("BTC/USDT", "1h", 60)
    assert len(client.calls) == 1
    assert second["candles"][0]["close"] > 0
    clock[0] += 21
    service.get("BTC/USDT", "1h", 60)
    assert len(client.calls) == 2


@pytest.mark.unit
def test_chart_uses_last_good_data_as_explicitly_stale_on_refresh_failure():
    clock = [10.0]
    client = _Client([_candle()])
    service = CryptoChartService(
        client_factory=lambda: client, clock=lambda: clock[0], ttl_seconds=1,
    )
    assert service.get("BTC/USDT", "1h", 60)["status"] == "ok"
    clock[0] += 2
    client.klines = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError())
    stale = service.get("BTC/USDT", "1h", 60)
    assert stale["status"] == "stale"
    assert stale["candles"]
    assert "TimeoutError" in stale["detail"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "interval", "limit"),
    [
        ("BTC/USDT", "1d", 60), ("BTC/USDT", "1h", 1),
        ("BTC/USDT", "1h", 999), ("BTC/USDT", "1h", True),
    ],
)
def test_chart_rejects_invalid_query_values(symbol, interval, limit):
    service = CryptoChartService(client_factory=lambda: _Client([_candle()]))
    with pytest.raises(ValueError):
        service.get(symbol, interval, limit)


@pytest.mark.unit
def test_chart_rejects_malformed_price_data():
    bad = _candle()
    bad.high = float("nan")
    service = CryptoChartService(client_factory=lambda: _Client([bad]))
    result = service.get("BTC/USDT", "1h", 60)
    assert result["status"] == "error"
    assert result["candles"] == []
    assert "ValueError" in result["detail"]


@pytest.mark.unit
def test_chart_rejects_unaware_timestamps():
    candle = _candle()
    candle.open_time = datetime(2026, 1, 1)
    service = CryptoChartService(client_factory=lambda: _Client([candle]))
    result = service.get("BTC/USDT", "1h", 60)
    assert result["status"] == "error"
    assert "ValueError" in result["detail"]
