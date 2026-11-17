from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualCosts,
    PerpetualEMAStrategy,
    run_backtest,
)
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import (
    MAX_RESEARCH_LEVERAGE,
    LinearPerpetualInstrument,
    PerpetualRiskSettings,
    estimate_isolated_liquidation_price,
    linear_funding_cashflow,
    size_linear_position,
)


def instrument(**overrides):
    values = {
        "symbol": "BTCUSDT", "base_coin": "BTC", "quote_coin": "USDT", "settle_coin": "USDT",
        "status": "Trading", "contract_type": "LinearPerpetual", "min_quantity": 0.001,
        "max_market_quantity": 100.0, "quantity_step": 0.001, "min_notional": 5.0,
        "tick_size": 0.1, "min_leverage": 1.0, "max_leverage": 50.0, "leverage_step": 0.01,
        "funding_interval_minutes": 480,
    }
    values.update(overrides)
    return LinearPerpetualInstrument(**values)


def test_leverage_ceiling_and_conservative_default():
    assert MAX_RESEARCH_LEVERAGE == 2.0
    assert PerpetualRiskSettings().leverage == 1.0
    with pytest.raises(ValueError, match="hard 2x ceiling"):
        PerpetualRiskSettings(leverage=3.0, max_leverage=3.0)


def test_sizing_respects_stop_risk_margin_and_notional_caps():
    sized = size_linear_position(
        equity=1000, entry_price=100, stop_price=97, direction=1,
        cost_rate_per_fill=0.00085, instrument=instrument(),
        risk=PerpetualRiskSettings(leverage=2.0),
    )
    assert sized.approved
    assert sized.initial_margin <= 100
    assert sized.notional <= 200
    assert sized.modeled_stop_risk <= 5


def test_isolated_liquidation_estimate_and_funding_signs():
    assert estimate_isolated_liquidation_price(
        entry_price=100, leverage=2, maintenance_margin_rate=0.005, direction=1
    ) == pytest.approx(50.5)
    assert linear_funding_cashflow(
        direction=1, quantity=2, mark_price=100, funding_rate=0.001
    ) == pytest.approx(-0.2)
    assert linear_funding_cashflow(
        direction=-1, quantity=2, mark_price=100, funding_rate=0.001
    ) == pytest.approx(0.2)


def test_strategy_is_disabled_and_costs_are_validated():
    assert PerpetualEMAStrategy().enabled is False
    with pytest.raises(ValueError, match="disabled"):
        run_backtest(None, None, None, instrument())
    with pytest.raises(ValueError, match="finite and non-negative"):
        PerpetualCosts(slippage_bps=-1)


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)


def transport_for(payload):
    def send(**kwargs):
        assert kwargs["headers"] == {}
        assert "api.bybit.com" in kwargs["url"]
        return Response(payload)
    return send


def test_bybit_public_client_rejects_unapproved_hosts_and_has_no_order_api():
    with pytest.raises(ValueError, match="approved Bybit"):
        BybitLinearPublicClient("http://api.bybit.com")
    client = BybitLinearPublicClient()
    assert not hasattr(client, "place_order")
    assert not hasattr(client, "set_leverage")


def test_bybit_instrument_rules_are_loaded_from_public_v5_response():
    row = {
        "symbol": "BTCUSDT", "baseCoin": "BTC", "quoteCoin": "USDT",
        "settleCoin": "USDT", "status": "Trading", "contractType": "LinearPerpetual",
        "fundingInterval": "480",
        "lotSizeFilter": {"minOrderQty": "0.001", "maxMktOrderQty": "100",
            "qtyStep": "0.001", "minNotionalValue": "5"},
        "priceFilter": {"tickSize": "0.1"},
        "leverageFilter": {"minLeverage": "1", "maxLeverage": "50", "leverageStep": "0.01"},
    }
    client = BybitLinearPublicClient(transport=transport_for({
        "retCode": 0, "result": {"list": [row]},
    }))
    parsed = client.instrument_info("BTCUSDT")
    assert parsed.max_leverage == 50
    assert parsed.funding_interval_minutes == 480


def test_bybit_klines_exclude_the_still_forming_bar():
    start = 1_700_000_000_000
    bar = 14_400_000
    client = BybitLinearPublicClient(transport=transport_for({
        "retCode": 0,
        "time": start + bar + 60_000,
        "result": {"list": [
            [str(start + bar), "101", "103", "100", "102", "4", "408"],
            [str(start), "99", "102", "98", "101", "5", "505"],
        ]},
    }))
    data = client.klines("BTCUSDT", start_ms=start, end_ms=start + bar * 2)
    assert len(data) == 1
    assert data.iloc[0]["close"] == 101
    assert isinstance(data.index, pd.DatetimeIndex)


def _synthetic_perpetual_window():
    index = pd.date_range("2024-01-01 04:00", periods=230, freq="4h", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
        },
        index=index,
    )
    # Draw down enough to trip the daily loss guard, but stay above the ATR stop.
    bars.loc[index[202], ["high", "low", "close"]] = [100.5, 97.7, 97.8]
    bars.loc[index[203], ["open", "high", "low", "close"]] = [97.8, 98.2, 97.6, 98.0]
    mark = bars.copy()
    funding = pd.Series(0.0, index=index[::2], name="funding_rate")
    return bars, mark, funding, index


def test_daily_risk_exit_is_cleared_before_a_later_position(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _synthetic_perpetual_window()
    signals = pd.Series(0, index=index, dtype=int)
    signals.loc[index[200]] = 1
    signals.loc[index[203]] = 1

    def forced_features(data, cfg):
        return pd.DataFrame(
            {"atr": pd.Series(1.0, index=data.index), "signal": signals},
            index=data.index,
        )

    monkeypatch.setattr(engine, "_features", forced_features)
    result = engine.run_backtest(
        traded,
        mark,
        funding,
        instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            risk=PerpetualRiskSettings(
                leverage=2.0,
                max_daily_loss_fraction=0.003,
                max_drawdown_fraction=0.10,
            ),
        ),
    )

    assert len(result.trades) == 2
    assert result.trades[0].exit_reason == "daily_loss_limit"
    assert result.trades[1].exit_reason == "end_of_window"


def test_entry_is_rejected_when_liquidation_is_too_close_to_atr_stop(monkeypatch):
    import tradingagents.crypto.backtest.perpetual_ema as engine

    traded, mark, funding, index = _synthetic_perpetual_window()
    signals = pd.Series(0, index=index, dtype=int)
    signals.loc[index[200]] = 1

    def forced_features(data, cfg):
        return pd.DataFrame(
            {"atr": pd.Series(1.0, index=data.index), "signal": signals},
            index=data.index,
        )

    monkeypatch.setattr(engine, "_features", forced_features)
    monkeypatch.setattr(engine, "estimate_isolated_liquidation_price", lambda **_: 98.0)
    result = engine.run_backtest(
        traded,
        mark,
        funding,
        instrument(),
        strategy=PerpetualEMAStrategy(
            enabled=True,
            risk=PerpetualRiskSettings(leverage=2.0),
        ),
    )

    assert result.trades == []
    assert result.rejections["liquidation_too_close_to_atr_stop"] == 1


def test_liquidation_buffer_is_validated_and_conservative_by_default():
    assert PerpetualRiskSettings().liquidation_buffer_fraction == pytest.approx(0.02)
    with pytest.raises(ValueError, match="liquidation_buffer_fraction"):
        PerpetualRiskSettings(liquidation_buffer_fraction=0.0)
    with pytest.raises(ValueError, match="liquidation_buffer_fraction"):
        PerpetualRiskSettings(liquidation_buffer_fraction=0.11)
