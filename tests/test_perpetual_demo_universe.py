from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from tradingagents.crypto.bybit_linear_demo import BybitLinearDemoAmbiguous
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
    broker.get_order_history.return_value = []
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
    controller.broker.open_position.assert_not_called()


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
    controller.broker.open_position.return_value = {"quantity": 1, "limit_price": 6.006}
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

    controller.broker.open_position.assert_called_once()
    request = controller.broker.open_position.call_args.kwargs
    assert request["symbol"] == DEFAULT_SYMBOLS[-1]
    assert request["leverage"] == 2.0
    assert request["signal_price"] == 6
    assert request["signal_expires_ms"] == int(candle_open.timestamp() * 1000) + 60_000


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

    controller.broker.open_position.assert_not_called()


@pytest.mark.parametrize("filled", [0, 0.04])
def test_cancelled_ioc_uses_confirmed_execution_quantity_after_restart(
    tmp_path, monkeypatch, filled,
):
    symbol = "ETHUSDT"
    controller = controller_for(tmp_path, symbols=(symbol,))
    candle_open = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    now = candle_open.timestamp() + 10
    monkeypatch.setattr(
        "tradingagents.paper_experiment.perpetual_demo.server.time.time", lambda: now,
    )
    market = {
        "forming_candle_open": candle_open.isoformat(),
        "signal": 1,
        "mark_price": 100,
        "suggested_long_stop": 99.6,
        "suggested_long_target": 100.8,
    }
    monkeypatch.setattr(controller, "_scalping_market", Mock(return_value=market))
    controller.broker.open_position.side_effect = BybitLinearDemoAmbiguous("uncertain")
    controller._strategy_cycle()

    restarted = PerpetualDemoController(tmp_path, broker=controller.broker, symbols=(symbol,))
    monkeypatch.setattr(restarted, "_scalping_market", Mock(return_value=market))
    meta = restarted._state["scalper_positions"][symbol]
    assert meta["entry_submission_unconfirmed"]
    assert not any(event["type"] == "strategy_entry_fill" for event in restarted._state["events"])
    order = {
        "order_link_id": meta["order_link_id"],
        "order_status": "Cancelled",
        "qty": 0.1,
        "cum_exec_qty": filled,
        "avg_price": 100.05 if filled else None,
        "price": 100.1,
    }
    if filled:
        restarted.broker.get_positions.return_value = [
            {"symbol": symbol, "side": "long", "quantity": filled, "entry_price": 100.05},
        ]

    # A position in the same contract is insufficient to attribute an unknown order.
    restarted.broker.get_order_history.return_value = [dict(order, order_link_id="other-order")]
    now += 5
    restarted._strategy_cycle()
    assert not meta["venue_position_confirmed"]
    restarted.broker.get_order_history.return_value = [order]
    restarted._strategy_cycle()
    restarted._strategy_cycle()

    events = restarted._state["events"]
    restarted.broker.open_position.assert_called_once()
    fills = [event for event in events if event["type"] == "strategy_entry_fill"]
    if filled:
        assert len(fills) == 1
        assert fills[0]["requested_quantity"] == 0.1
        assert fills[0]["filled_quantity"] == filled
        assert fills[0]["average_fill_price"] == 100.05
        assert fills[0]["entry_limit_price"] == 100.1
        assert fills[0]["adverse_slippage_bps"] == pytest.approx(5)
        assert meta["venue_position_confirmed"]
    else:
        assert fills == []
        assert symbol not in restarted._state["scalper_positions"]
        assert sum(event["type"] == "strategy_entry_unfilled" for event in events) == 1
