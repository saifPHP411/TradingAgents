from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from tradingagents.paper_experiment.perpetual_demo.server import (
    DEFAULT_SYMBOLS,
    SYMBOLS_ENV,
    PerpetualDemoController,
    build_parser,
)


def controller_for(tmp_path, **kwargs):
    broker = Mock()
    broker.get_wallet_balance.return_value = {"status": "ok", "total_equity_usd": 50}
    broker.get_positions.return_value = []
    return PerpetualDemoController(tmp_path, broker=broker, **kwargs)


def test_default_scan_evaluates_more_than_twenty_distinct_contracts(tmp_path, monkeypatch):
    controller = controller_for(tmp_path)
    candle_open = datetime.now(UTC).replace(second=0, microsecond=0).isoformat()
    get_market = Mock(return_value={"forming_candle_open": candle_open, "signal": 0})
    monkeypatch.setattr(controller, "_scalping_market", get_market)

    controller._strategy_cycle()

    assert len(DEFAULT_SYMBOLS) == 24
    assert {call.args[0] for call in get_market.call_args_list} == set(DEFAULT_SYMBOLS)
    assert get_market.call_count == 24
    assert controller._state["last_scalper_scan"]["configured_symbol_count"] == 24
    assert controller._state["last_scalper_scan"]["symbols_scanned"] == 24
    controller.broker.open_market_position.assert_not_called()


def test_custom_universe_reports_normalized_symbols_and_rejects_alias_duplicates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(SYMBOLS_ENV, "ethusdt, linkusdt")
    args = build_parser().parse_args([])
    controller = controller_for(tmp_path, symbols=tuple(args.symbols.replace(",", " ").split()))
    controller.broker.configured = False
    assert controller.status()["strategy"]["symbols"] == ["ETHUSDT", "LINKUSDT"]
    with pytest.raises(ValueError, match="unique"):
        controller_for(tmp_path, symbols=("ETHUSDT", " ethusdt "))


def test_last_configured_contract_can_produce_an_entry(tmp_path, monkeypatch):
    controller = controller_for(tmp_path)
    controller.broker.open_market_position.return_value = {"quantity": 1}
    candle_open = datetime.now(UTC).replace(second=0, microsecond=0)
    monkeypatch.setattr(
        "tradingagents.paper_experiment.perpetual_demo.server.time.time",
        lambda: candle_open.timestamp() + 10,
    )

    def get_market(symbol, **kwargs):
        return {
            "forming_candle_open": candle_open.isoformat(),
            "signal": int(symbol == DEFAULT_SYMBOLS[-1]),
            "mark_price": 6,
            "suggested_long_stop": 5.98,
            "suggested_long_target": 6.04,
        }

    monkeypatch.setattr(controller, "_scalping_market", get_market)
    controller._strategy_cycle()

    controller.broker.open_market_position.assert_called_once()
    request = controller.broker.open_market_position.call_args.kwargs
    assert request["symbol"] == DEFAULT_SYMBOLS[-1]
    assert request["leverage"] == 2.0


def test_old_snapshot_from_wide_scan_cannot_produce_an_entry(tmp_path, monkeypatch):
    controller = controller_for(tmp_path, symbols=("ETHUSDT",))
    candle_open = datetime.now(UTC).replace(second=0, microsecond=0)
    monkeypatch.setattr(
        "tradingagents.paper_experiment.perpetual_demo.server.time.time",
        lambda: candle_open.timestamp() + 60,
    )
    monkeypatch.setattr(
        controller,
        "_scalping_market",
        Mock(return_value={"forming_candle_open": candle_open.isoformat(), "signal": 1}),
    )

    controller._strategy_cycle()

    controller.broker.open_market_position.assert_not_called()
