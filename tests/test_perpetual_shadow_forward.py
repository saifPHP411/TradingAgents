from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.crypto.backtest.perpetual_strategy_lab import CANDIDATES
from tradingagents.crypto.backtest.perpetual_shadow_forward import (
    BAR_MS,
    DEFAULT_SYMBOLS,
    INITIAL_EQUITY_PER_SYMBOL,
    STATE_FILE,
    _load_state,
    _save_state,
    _summarize,
    read_shadow_status,
    run_shadow_cycle,
)


@dataclass
class FakeTrade:
    side: str
    entry_time: str
    exit_time: str
    quantity: float
    leverage: float
    margin: float
    entry_price: float
    exit_price: float
    liquidation_price_estimate: float
    exit_reason: str
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float

    def to_dict(self):
        return self.__dict__.copy()


class FakeResult:
    def __init__(self, trades, final_equity=1003.0):
        self.trades = trades
        self.final_equity = final_equity

    def metrics(self):
        return {"max_drawdown_pct": 1.25, "liquidations": 0}


def _trade(exit_reason: str, net_pnl: float) -> FakeTrade:
    return FakeTrade(
        side="long",
        entry_time="2026-01-01T04:00:00+00:00",
        exit_time="2026-01-01T08:00:00+00:00",
        quantity=0.01,
        leverage=1.0,
        margin=100.0,
        entry_price=100.0,
        exit_price=101.0,
        liquidation_price_estimate=1.0,
        exit_reason=exit_reason,
        gross_pnl=net_pnl + 0.1,
        fees=0.1,
        funding=0.0,
        net_pnl=net_pnl,
    )


def _now_ms(value: str) -> int:
    return int(pd.Timestamp(value).value // 1_000_000)


def test_forward_boundary_is_persisted_before_network_access_and_never_moves(tmp_path):
    initial_now = _now_ms("2026-01-01T00:30:00Z")
    state = _load_state(tmp_path, initial_now)
    original_start = state["forward_start_ms"]
    assert original_start == _now_ms("2026-01-01T04:00:00Z")

    # The persisted anchor must survive a public API outage on the first cycle.
    with pytest.raises(RuntimeError, match="public data unavailable"):
        run_shadow_cycle(
            tmp_path,
            now_ms=initial_now,
            client_factory=lambda: object(),
            history_fetcher=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("public data unavailable")
            ),
        )

    saved = read_shadow_status(tmp_path)
    assert saved["forward_start_ms"] == original_start
    assert (Path(tmp_path) / STATE_FILE).is_file()

    later = _load_state(tmp_path, initial_now + 60 * 60 * 1000)
    assert later["forward_start_ms"] == original_start


def test_summary_excludes_end_of_window_mark_from_realized_win_rate_and_pnl():
    result = FakeResult([_trade("atr_stop", 2.0), _trade("end_of_window", -3.0)])
    summary, rows = _summarize(
        [("BTCUSDT", result)], "ema_14_50_adx18", 1.0, "realistic_estimate"
    )

    assert summary["closed_trades"] == 1
    assert summary["wins"] == 1
    assert summary["win_rate_pct"] == 100.0
    assert summary["realized_net_pnl_usd"] == pytest.approx(2.0)
    assert summary["open_positions"] == 1
    assert summary["estimated_open_pnl_after_exit_costs_usd"] == pytest.approx(-3.0)
    assert len(rows) == 2
    assert summary["orders_submitted"] is False
    assert summary["demo_ready"] is False


def test_shadow_cycle_scores_only_future_bars_for_all_frozen_candidates(tmp_path):
    initial_now = _now_ms("2026-01-01T00:30:00Z")
    state = _load_state(tmp_path, initial_now)
    _save_state(tmp_path, state)
    forward_start = state["forward_start_ms"]
    history_start = state["history_start_ms"]
    latest_open = forward_start + BAR_MS
    now_ms = forward_start + 2 * BAR_MS

    index = pd.to_datetime(
        [history_start + i * BAR_MS for i in range(262)],
        unit="ms",
        utc=True,
    )
    assert int(index[-1].value // 1_000_000) == latest_open
    candles = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [102.0] * len(index),
            "low": [99.0] * len(index),
            "close": [101.0] * len(index),
            "volume": [10.0] * len(index),
        },
        index=index,
    )
    funding_index = index[::2]
    funding = pd.Series([0.0001] * len(funding_index), index=funding_index)

    history_calls = []
    def history_fetcher(_client, symbol, start_ms, end_ms):
        history_calls.append(symbol)
        assert start_ms == history_start
        assert end_ms == now_ms
        return SimpleNamespace(symbol=symbol), candles.copy(), candles[["open", "high", "low", "close"]].copy(), funding.copy()

    backtest_calls = []
    def backtest_runner(traded, mark, funding_rates, instrument, **kwargs):
        backtest_calls.append({
            "trade_start_index": kwargs["trade_start_index"],
            "leverage": kwargs["strategy"].risk.leverage,
            "symbol": instrument.symbol,
        })
        assert len(traded) == 262
        assert len(mark) == 262
        assert not funding_rates.empty
        assert kwargs["initial_equity"] == INITIAL_EQUITY_PER_SYMBOL
        return FakeResult([_trade("atr_stop", 2.0), _trade("end_of_window", 1.0)])

    result = run_shadow_cycle(
        tmp_path,
        now_ms=now_ms,
        client_factory=lambda: object(),
        history_fetcher=history_fetcher,
        backtest_runner=backtest_runner,
    )

    assert history_calls == list(DEFAULT_SYMBOLS)
    assert len(backtest_calls) == len(CANDIDATES) * 2 * 3 * len(DEFAULT_SYMBOLS)
    assert {call["trade_start_index"] for call in backtest_calls} == {260}
    assert {call["leverage"] for call in backtest_calls} == {1.0, 2.0}
    assert result["status"] == "running"
    assert result["forward_start_ms"] == forward_start
    assert result["forward_bars"] == 2
    assert len(result["results"]) == len(CANDIDATES) * 2 * 3
    assert all(row["closed_trades"] == 4 for row in result["results"])
    assert all(row["open_positions"] == 4 for row in result["results"])
    assert all(row["orders_submitted"] is False for row in result["results"])
    assert result["orders_submitted"] is False
    assert result["execution_enabled"] is False
    assert result["demo_ready"] is False
    saved = read_shadow_status(tmp_path)
    assert saved["last_processed_bar_ms"] == latest_open
    assert len(saved["recent_trades"]) == 120
