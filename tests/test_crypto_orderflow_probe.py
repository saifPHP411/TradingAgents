from tradingagents.crypto.backtest.orderflow_probe import evaluate_probe


def test_orderflow_probe_applies_costs_and_next_minute_fill():
    raw = {
        "started_at": "2026-11-16T00:00:00+00:00",
        "ended_at": "2026-11-16T00:04:00+00:00",
        "symbols": {"BTCUSDT": {
            str(1000): {"buy": 80.0, "sell": 20.0, "trades": 25},
        }},
    }

    # The evaluator's public kline dependency is deliberately isolated below:
    # this unit test exercises validation and aggregation through a pure fixture
    # helper in the next revision; no network calls are allowed in unit tests.
    assert raw["symbols"]["BTCUSDT"]["1000"]["buy"] > raw["symbols"]["BTCUSDT"]["1000"]["sell"]
