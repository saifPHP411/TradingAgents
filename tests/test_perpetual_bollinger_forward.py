from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_bollinger_forward import (
    FORWARD_CANDIDATES,
    _load_symbol_window,
    _parse_forward_start,
)


@pytest.mark.unit
def test_forward_start_requires_timezone_and_four_hour_alignment():
    assert _parse_forward_start("2026-11-01T00:00:00+00:00") == pd.Timestamp(
        "2026-11-01T00:00:00Z"
    )
    with pytest.raises(ValueError, match="timezone"):
        _parse_forward_start("2026-11-01T00:00:00")
    with pytest.raises(ValueError, match="4-hour"):
        _parse_forward_start("2026-11-01T01:00:00+00:00")


@pytest.mark.unit
def test_forward_candidates_are_fixed_and_do_not_mutate_development_definitions():
    di_candidate = FORWARD_CANDIDATES["bollinger_rsi_quality20_atr16_di16"]
    std_candidate = FORWARD_CANDIDATES["bollinger_rsi_quality20_atr16_std18"]

    assert di_candidate["atr_multiple"] == 1.6
    assert di_candidate["mean_reversion_max_adverse_di_spread"] == 16.0
    assert std_candidate["atr_multiple"] == 1.6
    assert std_candidate["mean_reversion_std"] == 1.8
    assert len(FORWARD_CANDIDATES) == 2


@pytest.mark.unit
def test_forward_loader_uses_vision_placeholders_and_starts_trading_at_cutoff():
    index = pd.date_range("2026-09-01", periods=300, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
        },
        index=index,
    )
    instrument = SimpleNamespace(
        min_quantity=0.001,
        max_market_quantity=10.0,
        quantity_step=0.001,
        min_notional=5.0,
        tick_size=0.1,
        funding_interval_minutes=480,
    )
    funding = pd.Series(0.0, index=index[::2])

    class FakeClient:
        def instrument_info(self, symbol):
            assert symbol == "BTCUSDT"
            return instrument

        def klines(self, symbol, *, start_ms, end_ms):
            return frame

        def mark_price_klines(self, symbol, *, start_ms, end_ms):
            return frame.copy()

        def funding_rates(
            self,
            symbol,
            *,
            start_ms,
            end_ms,
            interval_minutes,
        ):
            assert symbol == "BTCUSDT"
            assert interval_minutes == 480
            assert end_ms == int(index[260].timestamp() * 1000)
            return funding.loc[funding.index < index[260]]

    loaded = _load_symbol_window(
        FakeClient(),
        "BTCUSDT",
        start_ms=int(index[0].timestamp() * 1000),
        end_ms=int((index[-1] + pd.Timedelta(hours=4)).timestamp() * 1000),
        forward_start=index[260],
    )
    _instrument, traded, mark, rates, trade_start_index, filters = loaded

    assert traded.index.equals(mark.index)
    assert rates.equals(funding.loc[funding.index < index[260]])
    assert trade_start_index == 260
    assert filters["min_notional"] == 5.0
    assert filters["quantity_step"] == 0.001
    assert filters["executable_filters_verified"] is False
    assert filters["source"].startswith("Binance Vision research placeholders")

@pytest.mark.unit
def test_forward_funding_assumptions_are_explicit_and_do_not_rewrite_history():
    from tradingagents.crypto.backtest.perpetual_bollinger_forward import (
        _funding_with_forward_assumption,
    )

    historical_index = pd.date_range(
        "2026-10-31T00:00:00Z", periods=3, freq="8h"
    )
    historical = pd.Series(
        [-0.0001, 0.0002, -0.0001],
        index=historical_index,
        name="funding_rate",
    )
    historical.attrs["funding_interval_minutes"] = pd.Series(
        480, index=historical_index, dtype=int
    )
    cutoff = pd.Timestamp("2026-11-01T00:00:00Z")
    last_candle = pd.Timestamp("2026-11-03T20:00:00Z")

    positive = _funding_with_forward_assumption(
        historical,
        forward_start=cutoff,
        last_candle=last_candle,
        interval_minutes=480,
        forward_rate=0.0001,
    )

    assert positive.loc[historical_index].tolist() == historical.tolist()
    assert positive.loc[cutoff] == pytest.approx(0.0001)
    assert positive.loc[pd.Timestamp("2026-11-02T00:00:00Z")] == pytest.approx(0.0001)
    assert positive.attrs["forward_funding_assumption"] == pytest.approx(0.0001)
    assert positive.attrs["funding_interval_minutes"].eq(480).all()

    zero = _funding_with_forward_assumption(
        historical,
        forward_start=cutoff,
        last_candle=last_candle,
        interval_minutes=480,
        forward_rate=0.0,
    )
    negative = _funding_with_forward_assumption(
        historical,
        forward_start=cutoff,
        last_candle=last_candle,
        interval_minutes=480,
        forward_rate=-0.0001,
    )
    assert zero.loc[cutoff] == 0.0
    assert negative.loc[cutoff] == pytest.approx(-0.0001)


@pytest.mark.unit
def test_forward_report_does_not_claim_real_exchange_filters():
    # The loader contract is tested separately with a fake Vision client;
    # this assertion guards against reintroducing exchangeInfo claims.
    source = inspect.getsource(_load_symbol_window)
    assert "Binance Vision research placeholders" in source
    assert "executable_filters_verified" in source
