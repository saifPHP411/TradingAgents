from __future__ import annotations

import json

import pytest

from tradingagents.crypto.bybit_perpetual_account import (
    BybitLinearDemoPositionClient,
    BybitPerpetualAccountError,
)
from tradingagents.paper_experiment.dashboard.server import PERPETUAL_INDEX_HTML, Dashboard


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_wallet_balance_uses_only_signed_demo_get_and_normalizes_unified_usdt():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return Response({
            "retCode": 0,
            "result": {
                "list": [{
                    "accountType": "UNIFIED",
                    "totalEquity": "1250.50",
                    "totalWalletBalance": "1200.25",
                    "totalMarginBalance": "1185.25",
                    "totalAvailableBalance": "1000.00",
                    "totalPerpUPL": "-15.00",
                    "totalInitialMargin": "200.00",
                    "totalMaintenanceMargin": "40.00",
                    "coin": [{
                        "coin": "USDT",
                        "walletBalance": "1200.25",
                        "equity": "1185.25",
                        "unrealisedPnl": "-15.00",
                        "usdValue": "1185.25",
                        "locked": "0",
                        "borrowAmount": "0",
                    }],
                }]
            },
        })

    client = BybitLinearDemoPositionClient(
        api_key="demo-key",
        api_secret="demo-secret",
        transport=transport,
        clock_ms=lambda: 1234567890000,
    )
    result = client.get_wallet_balance()

    assert result["status"] == "ok"
    assert result["total_equity_usd"] == pytest.approx(1250.50)
    assert result["total_available_balance_usd"] == pytest.approx(1000.00)
    assert result["total_perpetual_unrealized_pnl_usd"] == pytest.approx(-15.00)
    assert result["usdt_wallet_balance"] == pytest.approx(1200.25)
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://api-demo.bybit.com/v5/account/wallet-balance"
    assert calls[0]["params"] == {"accountType": "UNIFIED", "coin": "USDT"}
    assert calls[0]["headers"]["X-BAPI-SIGN"]
    assert calls[0]["headers"]["X-BAPI-API-KEY"] == "demo-key"
    assert "demo-secret" not in repr(calls[0])


def test_wallet_balance_fails_closed_on_invalid_equity():
    client = BybitLinearDemoPositionClient(
        api_key="key",
        api_secret="secret",
        transport=lambda **_: Response({
            "retCode": 0,
            "result": {"list": [{
                "accountType": "UNIFIED",
                "totalEquity": "NaN",
                "totalWalletBalance": "10",
                "coin": [],
            }]},
        }),
        clock_ms=lambda: 1,
    )
    with pytest.raises(BybitPerpetualAccountError, match="totalEquity"):
        client.get_wallet_balance()


def test_closed_pnl_is_read_only_sorted_and_keeps_fee_attribution():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return Response({
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "ETHUSDT", "side": "Sell", "closedSize": "0.2",
                        "avgEntryPrice": "3000", "avgExitPrice": "2900",
                        "closedPnl": "20", "openFee": "0.3", "closeFee": "0.29",
                        "leverage": "2", "createdTime": "2000", "updatedTime": "2100",
                        "execType": "Trade",
                    },
                    {
                        "symbol": "BTCUSDT", "side": "Buy", "closedSize": "0.01",
                        "avgEntryPrice": "60000", "avgExitPrice": "61000",
                        "closedPnl": "10", "openFee": "0.2", "closeFee": "0.2",
                        "leverage": "1", "createdTime": "1000", "updatedTime": "1100",
                        "execType": "Trade",
                    },
                ],
                "nextPageCursor": "",
            },
        })

    client = BybitLinearDemoPositionClient(
        api_key="key", api_secret="secret", transport=transport, clock_ms=lambda: 1
    )
    result = client.get_closed_pnl(limit=25)

    assert result["status"] == "ok"
    assert [row["symbol"] for row in result["records"]] == ["BTCUSDT", "ETHUSDT"]
    assert result["records"][0]["closed_pnl"] == pytest.approx(10)
    assert result["records"][1]["side"] == "short"
    assert result["records"][1]["open_fee"] == pytest.approx(0.3)
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://api-demo.bybit.com/v5/position/closed-pnl"
    assert calls[0]["params"] == {"category": "linear", "limit": 25}


class FakeAccountClient:
    equity = 1000.0

    def get_wallet_balance(self):
        return {
            "status": "ok",
            "total_equity_usd": self.equity,
            "total_wallet_balance_usd": self.equity - 5,
            "total_margin_balance_usd": self.equity - 2,
            "total_available_balance_usd": self.equity - 100,
            "total_perpetual_unrealized_pnl_usd": 3,
            "total_initial_margin_usd": 100,
            "total_maintenance_margin_usd": 20,
            "usdt_wallet_balance": self.equity - 5,
            "updated_at_ms": 1234567890000,
        }

    def get_closed_pnl(self, *, limit):
        assert limit == 100
        return {
            "status": "ok",
            "window": "Bybit default recent 7 days",
            "next_page_available": False,
            "records": [{
                "symbol": "BTCUSDT", "side": "long", "quantity": 0.01,
                "entry_price": 60000, "exit_price": 61000,
                "closed_pnl": 8.5, "open_fee": 0.2, "close_fee": 0.2,
                "created_time_ms": 1234567890000, "updated_time_ms": 1234567890000,
            }],
        }


def test_dashboard_tracks_first_verified_baseline_and_calculates_return(tmp_path):
    fake = FakeAccountClient()
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_account_client_factory=lambda: fake,
    )

    first = dashboard.perpetual_account()
    assert first["status"] == "ok"
    assert first["initial_capital_usd"] == pytest.approx(1000)
    assert first["return_since_initial_capital_pct"] == pytest.approx(0)
    assert first["capital_flow_adjusted"] is False

    fake.equity = 1125
    second = dashboard.perpetual_account()
    assert second["initial_capital_usd"] == pytest.approx(1000)
    assert second["net_pnl_since_initial_capital_usd"] == pytest.approx(125)
    assert second["return_since_initial_capital_pct"] == pytest.approx(12.5)
    assert len(second["equity_history"]) == 1  # snapshots are rate-limited to one per minute
    saved = json.loads((tmp_path / "perpetual_account_baseline.json").read_text())
    assert saved["initial_capital_usd"] == pytest.approx(1000)


def test_dashboard_honors_explicit_original_capital_and_rejects_invalid_value(
    tmp_path, monkeypatch
):
    fake = FakeAccountClient()
    monkeypatch.setenv("TRADINGAGENTS_INITIAL_CAPITAL_USDT", "800")
    dashboard = Dashboard(
        tmp_path,
        token="test-token",
        perpetual_only_mode=True,
        perpetual_account_client_factory=lambda: fake,
    )
    result = dashboard.perpetual_account()
    assert result["initial_capital_source"] == "configured_initial_capital"
    assert result["initial_capital_usd"] == pytest.approx(800)
    assert result["return_since_initial_capital_pct"] == pytest.approx(25)

    monkeypatch.setenv("TRADINGAGENTS_INITIAL_CAPITAL_USDT", "-1")
    invalid = dashboard.perpetual_account()
    assert invalid["status"] == "error"
    assert invalid["error_type"] == "InvalidInitialCapital"


def test_perpetual_dashboard_uses_burgundy_theme_and_account_analytics():
    assert "--bg:#10070c" in PERPETUAL_INDEX_HTML
    assert "id=\"account-metrics\"" in PERPETUAL_INDEX_HTML
    assert "id=\"equity-chart\"" in PERPETUAL_INDEX_HTML
    assert "id=\"realized-chart\"" in PERPETUAL_INDEX_HTML
    assert "/api/perpetual/account" in PERPETUAL_INDEX_HTML
    assert "No Spot engine" in PERPETUAL_INDEX_HTML
    assert "var gridY=top+ph*i/3" in PERPETUAL_INDEX_HTML
    assert "var y=top+ph*i/3" not in PERPETUAL_INDEX_HTML
