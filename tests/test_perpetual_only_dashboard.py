from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.crypto.bybit_perpetual_account import (
    BybitLinearDemoPositionClient,
    BybitPerpetualAccountError,
    BybitPerpetualAccountNotConfigured,
)
from tradingagents.paper_experiment.dashboard import server as dashboard_server
from tradingagents.paper_experiment.dashboard.server import Dashboard, make_handler


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _position_row(**overrides):
    row = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "size": "0.015",
        "avgPrice": "65000",
        "markPrice": "65200",
        "leverage": "2",
        "liqPrice": "33500",
        "stopLoss": "63000",
        "takeProfit": "71000",
        "unrealisedPnl": "3.0",
        "updatedTime": "1794931200000",
        "positionIdx": 0,
    }
    row.update(overrides)
    return row


def test_demo_position_reader_is_signed_get_only_and_returns_real_levels():
    captured = {}

    def transport(**kwargs):
        captured.update(kwargs)
        return Response({"retCode": 0, "result": {"list": [_position_row()]}})

    client = BybitLinearDemoPositionClient(
        api_key="demo-key",
        api_secret="demo-secret",
        transport=transport,
        clock_ms=lambda: 1234567890000,
    )
    result = client.get_position("BTCUSDT")

    assert result["status"] == "open"
    position = result["position"]
    assert position["side"] == "long"
    assert position["quantity"] == pytest.approx(0.015)
    assert position["leverage"] == pytest.approx(2.0)
    assert position["entry_price"] == pytest.approx(65000)
    assert position["mark_price"] == pytest.approx(65200)
    assert position["stop_loss"] == pytest.approx(63000)
    assert position["take_profit"] == pytest.approx(71000)
    assert position["liquidation_price"] == pytest.approx(33500)
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api-demo.bybit.com/v5/position/list"
    assert captured["params"] == {
        "category": "linear", "settleCoin": "USDT", "symbol": "BTCUSDT"
    }
    assert captured["headers"]["X-BAPI-SIGN"]
    assert captured["headers"]["X-BAPI-API-KEY"] == "demo-key"
    assert not hasattr(client, "place_order")
    assert not hasattr(client, "set_leverage")
    assert not hasattr(client, "cancel_order")


def test_demo_position_reader_reports_flat_only_when_venue_returns_empty_positions():
    client = BybitLinearDemoPositionClient(
        api_key="key",
        api_secret="secret",
        transport=lambda **_: Response({"retCode": 0, "result": {"list": []}}),
        clock_ms=lambda: 1,
    )
    assert client.get_position("ETHUSDT") == {"status": "flat", "position": None}


def test_demo_position_reader_requires_both_credentials(monkeypatch):
    monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    client = BybitLinearDemoPositionClient()
    with pytest.raises(BybitPerpetualAccountNotConfigured):
        client.get_position("BTCUSDT")


def test_demo_position_reader_rejects_hedged_or_multiple_open_position_legs():
    client = BybitLinearDemoPositionClient(
        api_key="key",
        api_secret="secret",
        transport=lambda **_: Response({
            "retCode": 0,
            "result": {"list": [_position_row(), _position_row(side="Sell")]},
        }),
        clock_ms=lambda: 1,
    )
    with pytest.raises(BybitPerpetualAccountError, match="multiple open position legs"):
        client.get_position("BTCUSDT")


def test_demo_position_reader_never_uses_a_live_or_testnet_host():
    # The client does not accept a URL argument at all; its sole target is fixed.
    client = BybitLinearDemoPositionClient(api_key="key", api_secret="secret")
    assert client.base_url == "https://api-demo.bybit.com"


class FakePublicPerpetualClient:
    def instrument_info(self, symbol):
        assert symbol == "BTCUSDT"
        return SimpleNamespace(
            status="Trading",
            contract_type="LinearPerpetual",
            quote_coin="USDT",
            settle_coin="USDT",
        )

    def klines(self, symbol, *, start_ms, end_ms):
        assert symbol == "BTCUSDT"
        assert end_ms > start_ms
        index = pd.date_range("2026-11-01", periods=3, freq="4h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [64000.0, 64500.0, 65000.0],
                "high": [64600.0, 65100.0, 65500.0],
                "low": [63800.0, 64200.0, 64800.0],
                "close": [64500.0, 65000.0, 65200.0],
                "volume": [10.0, 11.0, 12.0],
            },
            index=index,
        )


class FakeAccountClient:
    def get_position(self, symbol):
        assert symbol == "BTCUSDT"
        return {
            "status": "open",
            "position": {
                "symbol": symbol,
                "side": "short",
                "quantity": 0.01,
                "leverage": 2.0,
                "entry_price": 65000.0,
                "mark_price": 65200.0,
                "stop_loss": 67000.0,
                "take_profit": 61000.0,
                "liquidation_price": 97500.0,
                "unrealized_pnl": -2.0,
                "updated_at": "1794931200000",
            },
        }


def test_perpetual_chart_combines_completed_candles_with_open_futures_position(tmp_path):
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_client_factory=FakePublicPerpetualClient,
        perpetual_account_client_factory=FakeAccountClient,
    )

    result = dashboard.perpetual_chart("BTCUSDT")

    assert result["category"] == "linear"
    assert result["timeframe"] == "4h"
    assert len(result["candles"]) == 3
    assert result["position_status"] == "open"
    assert result["position"]["side"] == "short"
    assert result["position"]["entry_price"] == pytest.approx(65000)
    assert result["position"]["stop_loss"] == pytest.approx(67000)
    assert result["position"]["take_profit"] == pytest.approx(61000)
    assert result["position"]["liquidation_price"] == pytest.approx(97500)
    assert result["orders_submitted"] is False


def test_perpetual_only_mode_never_reads_or_returns_spot_positions(tmp_path):
    dashboard = Dashboard(tmp_path, token="test-token", perpetual_only_mode=True)

    result = dashboard.positions()

    assert result["positions"] == []
    assert result["count"] == 0
    assert result["quote_refresh_status"] == "disabled"
    with pytest.raises(PermissionError, match="Spot/equity simulation controls"):
        dashboard.run_sessions(1)


def test_perpetual_only_mode_never_calls_a_missing_position_feed_flat(tmp_path, monkeypatch):
    monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    dashboard = Dashboard(tmp_path, token="test-token", perpetual_only_mode=True)

    status, position, source = dashboard._perpetual_position_view("BTCUSDT")

    assert status == "unavailable"
    assert position is None
    assert source is None


def test_invalid_or_unknown_chart_symbols_are_rejected_before_data_fetch(tmp_path):
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_client_factory=FakePublicPerpetualClient,
        perpetual_account_client_factory=FakeAccountClient,
    )
    with pytest.raises(ValueError, match="unsupported linear perpetual"):
        dashboard.perpetual_chart("BTC/USDT")


_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


@pytest.fixture
def allow_loopback(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("tests may only connect to loopback")
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (
            _LOOPBACK_CONNECT if address[0] == "127.0.0.1" else refuse
        )(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (
            _LOOPBACK_LOOKUP if host == "127.0.0.1" else refuse
        )(host, *a, **k),
    )


def test_perpetual_only_http_surface_blocks_legacy_spot_api(tmp_path, allow_loopback):
    dashboard = Dashboard(tmp_path, token="test-token", perpetual_only_mode=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/paper/status",
            headers={"Authorization": "Bearer test-token"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 410
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_perpetual_only_dashboard_does_not_construct_legacy_spot_components(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Spot/paper component was constructed in perpetual-only mode")

    monkeypatch.setattr(dashboard_server, "PaperTradingController", forbidden)
    monkeypatch.setattr(dashboard_server, "CryptoViewService", forbidden)
    monkeypatch.setattr(dashboard_server, "CryptoChartService", forbidden)

    dashboard = Dashboard(tmp_path, token="test-token", perpetual_only_mode=True)

    assert dashboard.paper is None
    assert dashboard.crypto is None
    assert dashboard.chart_service is None


def test_perpetual_only_entrypoint_skips_spot_provider_resolution(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRADINGAGENTS_PERPETUAL_ONLY", "1")
    captured = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("Spot/paper settings were resolved in perpetual-only mode")

    def fake_serve(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(dashboard_server, "resolve_settings", forbidden)
    monkeypatch.setattr(dashboard_server, "resolve_paper_settings", forbidden)
    monkeypatch.setattr(dashboard_server, "_resolve_execution_provider", forbidden)
    monkeypatch.setattr(dashboard_server, "_market_data_factory_for", forbidden)
    monkeypatch.setattr(dashboard_server, "_crypto_symbols", forbidden)
    monkeypatch.setattr(dashboard_server, "serve", fake_serve)

    assert dashboard_server.main(["--root", str(tmp_path)]) == 0

    assert captured["crypto_symbols"] == ()
    assert captured["crypto_live"] is False
    assert captured["market_data_factory"] is None
