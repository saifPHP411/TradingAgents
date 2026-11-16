from datetime import UTC, datetime

from tradingagents.crypto.backtest.orderflow_probe import evaluate_probe


def test_orderflow_signal_uses_next_minute_open_and_charges_costs():
    signal_minute = int(datetime(2026, 11, 16, 0, 0, tzinfo=UTC).timestamp() * 1000)
    raw = {
        "started_at": "2026-11-16T00:00:00+00:00",
        "ended_at": "2026-11-16T00:04:00+00:00",
        "symbols": {"BTCUSDT": {
            str(signal_minute // 60_000): {"buy": 80.0, "sell": 20.0, "trades": 25},
        }},
    }
    candles = {"BTCUSDT": {
        signal_minute + 60_000: {"open": 100.0, "close": 101.0},
        signal_minute + 180_000: {"open": 102.0, "close": 103.0},
    }}

    result = evaluate_probe(raw, candles_by_symbol=candles)

    assert result["signals"] == 1
    assert result["trades"][0]["aggressive_buy_imbalance"] == 0.6
    assert result["trades"][0]["gross_return_pct"] == 3.0
    assert result["trades"][0]["net_return_pct"] == 2.74


def test_orderflow_signal_rejects_low_imbalance_and_insufficient_trades():
    signal_minute = int(datetime(2026, 11, 16, 0, 0, tzinfo=UTC).timestamp() * 1000)
    raw = {
        "started_at": "2026-11-16T00:00:00+00:00",
        "ended_at": "2026-11-16T00:04:00+00:00",
        "symbols": {"BTCUSDT": {
            str(signal_minute // 60_000): {"buy": 60.0, "sell": 40.0, "trades": 19},
        }},
    }
    candles = {"BTCUSDT": {
        signal_minute + 60_000: {"open": 100.0, "close": 101.0},
        signal_minute + 180_000: {"open": 102.0, "close": 103.0},
    }}

    result = evaluate_probe(raw, candles_by_symbol=candles)

    assert result["signals"] == 0
    assert result["per_symbol"]["BTCUSDT"]["signals"] == 0
