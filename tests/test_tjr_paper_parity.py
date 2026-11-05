"""The TJR backtest must enforce the *paper account's* caps, not weaker ones.

The backtest and the live paper worker are two paths over one strategy. If they
size, count or bound differently, the backtest measures a book nobody will run —
and the gap stays invisible until real money is at stake. These tests pin the
shared limits in :mod:`tradingagents.risk.paper_parity` against the paper
broker's own dataclass, and pin each cap's enforcement in the backtest engine.

Every case here is checked against the ledger's own fills: an entry price times a
quantity is a real fill, not an intended one, so "the cap held" means the money
moved under it.
"""

from __future__ import annotations

import math

import pytest

from tests.tjr_helpers import BEARISH_BARS, BULLISH_BARS, SESSION, series_pair
from tradingagents.paper_experiment.broker.alpaca_paper import PaperRiskLimits
from tradingagents.risk import RiskEngine, RiskLimits, RiskReason, paper_parity_limits
from tradingagents.risk.paper_parity import (
    MAX_DAILY_ORDERS,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
)
from tradingagents.signals import SignalAction, TradingSignal
from tradingagents.tjr.engine import TjrBacktester, TjrConfig
from tradingagents.tjr.report import build_report

pytestmark = pytest.mark.unit

# A session extended so the entry's target is reached before the bell: one clean
# winning long per symbol when the caps allow it.
_WIN = BULLISH_BARS[:15] + [
    (100.6, 110.0, 100.55, 109.0),
    (109.0, 110.0, 108.0, 109.5),
]


def _run(symbols, *, limits: RiskLimits | None = None, cash: float = 100_000.0):
    one = {}
    five = {}
    for symbol, bars in symbols.items():
        five[symbol], one[symbol] = series_pair(symbol, SESSION, bars)
    config = TjrConfig(
        initial_cash=cash,
        risk_limits=limits or paper_parity_limits(),
    )
    return TjrBacktester(one_minute=one, five_minute=five, config=config).run()


def _same_bars(n: int) -> dict[str, list]:
    return {f"S{i:02d}": _WIN for i in range(n)}


# ---------------------------------------------------------------------------
# The parity constants are the broker's own numbers
# ---------------------------------------------------------------------------


class TestParityConstants:
    def test_the_parity_numbers_are_the_broker_defaults(self):
        broker = PaperRiskLimits()
        assert MAX_ORDER_NOTIONAL == broker.max_order_notional == 25.0
        assert MAX_OPEN_POSITIONS == broker.max_open_positions == 5
        assert MAX_DAILY_ORDERS == broker.max_daily_orders == 10

    def test_the_engine_limits_carry_the_broker_caps(self):
        limits = paper_parity_limits()
        assert limits.max_order_value == MAX_ORDER_NOTIONAL
        assert limits.target_order_value == MAX_ORDER_NOTIONAL
        assert limits.max_open_positions == MAX_OPEN_POSITIONS
        assert limits.max_daily_orders == MAX_DAILY_ORDERS
        assert limits.long_only is True

    def test_the_backtest_default_is_the_paper_parity_config(self):
        assert TjrConfig().risk_limits == paper_parity_limits()


# ---------------------------------------------------------------------------
# max_order_notional = 25
# ---------------------------------------------------------------------------


class TestOrderNotionalCap:
    def test_every_fill_is_at_or_below_the_notional_cap(self):
        result = _run(_same_bars(3))
        assert result.trades
        for trade in result.trades:
            notional = trade.entry_price * trade.quantity
            assert notional <= MAX_ORDER_NOTIONAL + 1e-9

    def test_the_signal_is_sized_to_the_target_notional(self):
        # The strategy requests no size, so the engine sizes the entry to the
        # target notional (25) rather than a fraction of the 100k book.
        result = _run(_same_bars(1))
        trade = result.trades[0]
        assert trade.entry_price * trade.quantity == pytest.approx(MAX_ORDER_NOTIONAL, rel=1e-6)

    def test_a_smaller_cap_shrinks_the_fill(self):
        limits = paper_parity_limits().model_copy(update={"max_order_value": 10.0, "target_order_value": 10.0})
        result = _run(_same_bars(1), limits=limits)
        trade = result.trades[0]
        assert trade.entry_price * trade.quantity <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# max_daily_orders = 10
# ---------------------------------------------------------------------------


class TestDailyOrderCap:
    def test_the_eleventh_entry_is_refused(self):
        # Twelve names each produce a clean setup. The open-position cap would
        # bind first at 5, so it is lifted here to isolate the daily cap: only ten
        # entries may be placed in the session.
        limits = paper_parity_limits().model_copy(update={"max_open_positions": 20})
        result = _run(_same_bars(MAX_DAILY_ORDERS + 2), limits=limits)
        assert result.entries_per_session[SESSION] == MAX_DAILY_ORDERS
        assert len(result.trades) == MAX_DAILY_ORDERS
        assert any("MAX_DAILY_ORDERS" in r["reasons"] for r in result.rejections)

    def test_the_cap_is_reported_per_session(self):
        result = _run(_same_bars(3))
        assert result.entries_per_session == {SESSION: 3}
        report = build_report(result)
        assert report.max_entries_in_a_session == 3

    def test_a_looser_cap_admits_more(self):
        limits = paper_parity_limits().model_copy(update={"max_daily_orders": 20})
        result = _run(_same_bars(12), limits=limits)
        # The open-position cap still binds at 5, so this only proves the daily
        # cap is no longer the binding one.
        assert result.entries_per_session[SESSION] == 5


# ---------------------------------------------------------------------------
# max_open_positions = 5
# ---------------------------------------------------------------------------


class TestOpenPositionCap:
    def test_no_more_than_five_positions_are_held_at_once(self):
        result = _run(_same_bars(12))
        events: list[tuple[str, int]] = []
        for trade in result.trades:
            events.append((trade.entry_time, 1))
            events.append((trade.exit_time, -1))
        events.sort()
        concurrent = 0
        peak = 0
        for _, delta in events:
            concurrent += delta
            peak = max(peak, concurrent)
        assert peak <= MAX_OPEN_POSITIONS
        assert any("MAX_OPEN_POSITIONS" in r["reasons"] for r in result.rejections)


# ---------------------------------------------------------------------------
# Long-only
# ---------------------------------------------------------------------------


class TestLongOnly:
    def test_a_bearish_setup_is_declined_by_the_long_only_book(self):
        result = _run({"X": BEARISH_BARS})
        assert result.trades == []
        assert result.ledger.position_quantity("X") == 0

    def test_the_engine_cannot_be_configured_to_short(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="long-only"):
            RiskLimits(long_only=False)


# ---------------------------------------------------------------------------
# Fractional vs whole shares
# ---------------------------------------------------------------------------


class TestShareGranularity:
    def test_fractional_shares_are_the_default(self):
        result = _run(_same_bars(1))
        trade = result.trades[0]
        # A 25 USD target on a ~100 USD name is 0.25 units, which only a
        # fractional book can hold.
        assert trade.quantity == pytest.approx(MAX_ORDER_NOTIONAL / trade.entry_price, rel=1e-6)

    def test_whole_shares_floor_the_quantity(self):
        # A ~10 USD name: a 25 USD target is 2.48 units, which floors to 2 shares.
        limits = paper_parity_limits().model_copy(update={"fractional_shares": False})
        scaled = [(o * 0.1, h * 0.1, lo * 0.1, c * 0.1) for o, h, lo, c in _WIN]
        result = _run({"X": scaled}, limits=limits)
        trade = result.trades[0]
        assert trade.quantity == math.floor(trade.quantity)
        assert trade.quantity == 2

    def test_whole_shares_refuse_a_size_that_floors_to_zero(self):
        # A 25 USD target on a 40 USD name floors to zero whole shares.
        limits = paper_parity_limits().model_copy(update={"fractional_shares": False})
        one = {}
        five = {}
        scaled = [(o * 0.4, h * 0.4, lo * 0.4, c * 0.4) for o, h, lo, c in _WIN]
        five["X"], one["X"] = series_pair("X", SESSION, scaled)
        result = TjrBacktester(
            one_minute=one, five_minute=five, config=TjrConfig(risk_limits=limits)
        ).run()
        assert result.trades == []
        assert any("NON_POSITIVE_QUANTITY" in r["reasons"] for r in result.rejections)


# ---------------------------------------------------------------------------
# The risk engine honours the new limits directly
# ---------------------------------------------------------------------------


class TestRiskEngineLimits:
    def _signal(self) -> TradingSignal:
        return TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.8,
            timestamp="2026-01-05T21:00:00+00:00",
            as_of_date="2026-01-05",
        )

    def test_max_daily_orders_rejects_at_the_count(self):
        from tradingagents.portfolio import Ledger

        engine = RiskEngine(paper_parity_limits())
        ledger = Ledger(initial_cash=100_000.0)
        decision = engine.evaluate(
            self._signal(), ledger, price=100.0, daily_orders_placed=MAX_DAILY_ORDERS
        )
        assert RiskReason.MAX_DAILY_ORDERS in decision.reasons
        assert not decision.approved

    def test_max_daily_orders_passes_below_the_count(self):
        from tradingagents.portfolio import Ledger

        engine = RiskEngine(paper_parity_limits())
        ledger = Ledger(initial_cash=100_000.0)
        decision = engine.evaluate(
            self._signal(), ledger, price=100.0, daily_orders_placed=MAX_DAILY_ORDERS - 1
        )
        assert decision.approved

    def test_target_order_value_sizes_the_entry(self):
        from tradingagents.portfolio import Ledger

        engine = RiskEngine(paper_parity_limits())
        ledger = Ledger(initial_cash=100_000.0)
        decision = engine.evaluate(self._signal(), ledger, price=100.0)
        assert decision.quantity * 100.0 == pytest.approx(MAX_ORDER_NOTIONAL)

    def test_whole_shares_floor_through_the_engine(self):
        from tradingagents.portfolio import Ledger

        limits = paper_parity_limits().model_copy(update={"fractional_shares": False})
        engine = RiskEngine(limits)
        ledger = Ledger(initial_cash=100_000.0)
        decision = engine.evaluate(self._signal(), ledger, price=10.0)
        # 25 / 10 = 2.5 units -> 2 whole shares.
        assert decision.quantity == 2.0
