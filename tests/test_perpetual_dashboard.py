from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.paper_experiment.dashboard.server import Dashboard, INDEX_HTML


class FakePerpetualClient:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def instrument_info(self, symbol):
        self.calls.append(("instrument", symbol))
        return SimpleNamespace(
            symbol=symbol,
            status="Trading",
            contract_type="LinearPerpetual",
            quote_coin="USDT",
            settle_coin="USDT",
        )

    def klines(self, symbol, *, start_ms, end_ms):
        self.calls.append(("klines", symbol))
        return self.frame.copy()


def _frame():
    index = pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [10.0, 12.0, 14.0],
        },
        index=index,
    )


@pytest.mark.unit
def test_perpetual_chart_uses_public_linear_data_and_does_not_invent_position(tmp_path):
    client = FakePerpetualClient(_frame())
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_client_factory=lambda: client,
    )

    result = dashboard.perpetual_chart("BTCUSDT")

    assert result["symbol"] == "BTCUSDT"
    assert result["category"] == "linear"
    assert result["timeframe"] == "4h"
    assert result["orders_submitted"] is False
    assert len(result["candles"]) == 3
    assert result["candles"][-1]["close"] == 103.0
    assert result["position_status"] == "unavailable"
    assert result["position"] is None
    assert [call[0] for call in client.calls] == ["instrument", "klines"]


@pytest.mark.unit
def test_perpetual_chart_overlays_a_validated_open_position_snapshot(tmp_path):
    client = FakePerpetualClient(_frame())
    (tmp_path / "perpetual_open_position.json").write_text(
        json.dumps(
            {
                "state": "OPEN",
                "symbol": "BTCUSDT",
                "side": "long",
                "quantity": 0.01,
                "leverage": 2,
                "entry_price": 100.0,
                "mark_price": 103.0,
                "stop_loss": 97.0,
                "take_profit": 109.0,
                "liquidation_price": 51.0,
                "unrealized_pnl": 0.03,
                "updated_at": "2026-01-01T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_client_factory=lambda: client,
    )

    result = dashboard.perpetual_chart("BTCUSDT")

    assert result["position_status"] == "open"
    assert result["position"]["side"] == "long"
    assert result["position"]["leverage"] == 2.0
    assert result["position"]["stop_loss"] == 97.0
    assert result["position"]["take_profit"] == 109.0
    assert result["position"]["liquidation_price"] == 51.0


@pytest.mark.unit
def test_perpetual_position_flat_state_must_match_chart_symbol(tmp_path):
    dashboard = Dashboard(tmp_path, token="test-token", perpetual_only_mode=True)
    path = tmp_path / "perpetual_open_position.json"
    path.write_text(json.dumps({"state": "FLAT", "symbol": "ETHUSDT"}), encoding="utf-8")

    status, position = dashboard._perpetual_position_view("BTCUSDT")

    assert status == "invalid"
    assert position is None


@pytest.mark.unit
def test_perpetual_chart_rejects_symbols_outside_the_linear_usdt_allowlist(tmp_path):
    client = FakePerpetualClient(_frame())
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_client_factory=lambda: client,
    )

    with pytest.raises(ValueError, match="unsupported linear perpetual"):
        dashboard.perpetual_chart("BTCUSD")

    assert client.calls == []


@pytest.mark.unit
def test_perpetual_only_mode_disables_equity_simulation_and_spot_positions(tmp_path):
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        allow_control=True,
        allow_paper_trading=True,
        perpetual_only_mode=True,
    )

    assert dashboard.positions()["count"] == 0
    assert dashboard.positions()["quote_refresh_status"] == "disabled"
    with pytest.raises(PermissionError, match="perpetual-only"):
        dashboard.run_sessions(1)


@pytest.mark.unit
def test_dashboard_html_is_perpetual_only_and_has_position_chart_overlays():
    assert "SPOT ENGINE" not in INDEX_HTML
    assert "NORMAL PAPER" not in INDEX_HTML
    assert "BINANCE" not in INDEX_HTML
    assert "/api/perpetual/chart" in INDEX_HTML
    assert "drawChart" in INDEX_HTML
    assert "TAKE PROFIT" in INDEX_HTML
    assert "No authenticated Bybit Futures position feed is connected" in INDEX_HTML
