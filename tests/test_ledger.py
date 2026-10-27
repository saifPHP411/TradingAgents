"""The simulated ledger: cash, positions, average entry, P&L, fees and events.

These are accounting tests, so they check conservation identities rather than
spot values: cash moves by exactly the trade's value plus its fee, a round trip
realises exactly the price change less costs, and equity equals cash plus marked
value. A ledger that satisfies those cannot quietly lose money.

The book is long-only, so the rejections matter as much as the successes: a sell
larger than the holding and a buy the cash cannot cover must both leave the book
untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tradingagents.portfolio import (
    EventType,
    InsufficientFundsError,
    Ledger,
    LedgerEvent,
)
from tradingagents.portfolio.ledger import InsufficientPositionError, LedgerError

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 7, 21, 0, tzinfo=UTC)


def _ledger(cash: float = 100_000.0) -> Ledger:
    return Ledger(initial_cash=cash)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_a_new_ledger_holds_only_cash(self):
        ledger = _ledger(50_000.0)
        assert ledger.cash == 50_000.0
        assert ledger.equity == 50_000.0
        assert ledger.open_positions() == {}
        assert ledger.total_fees == 0.0
        assert ledger.events == []

    def test_a_non_positive_starting_cash_is_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(LedgerError):
                Ledger(initial_cash=bad)


# ---------------------------------------------------------------------------
# Opening and increasing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuying:
    def test_a_buy_moves_cash_and_opens_a_position(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        assert ledger.cash == pytest.approx(9_000.0)
        assert ledger.position_quantity("AAPL") == 10
        assert ledger.average_entry_price("AAPL") == 100.0
        assert ledger.market_value == pytest.approx(1_000.0)

    def test_a_second_buy_blends_the_average_entry(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=120.0, fee=0.0, timestamp=T1)
        assert ledger.position_quantity("AAPL") == 20
        assert ledger.average_entry_price("AAPL") == pytest.approx(110.0)
        assert ledger.cash == pytest.approx(7_800.0)

    def test_a_fee_leaves_cash_and_is_recorded(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.5, timestamp=T0)
        assert ledger.cash == pytest.approx(8_998.5)
        assert ledger.total_fees == pytest.approx(1.5)
        assert any(e.type is EventType.FEE_CHARGED for e in ledger.events)

    def test_a_buy_beyond_the_cash_is_refused_and_changes_nothing(self):
        ledger = _ledger(1_000.0)
        with pytest.raises(InsufficientFundsError):
            ledger.apply_fill(symbol="AAPL", quantity=20, price=100.0, fee=0.0, timestamp=T0)
        assert ledger.cash == 1_000.0
        assert ledger.open_positions() == {}
        assert ledger.events == []

    def test_a_buy_whose_fee_overruns_the_cash_is_refused(self):
        ledger = _ledger(1_000.0)
        with pytest.raises(InsufficientFundsError):
            ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=5.0, timestamp=T0)
        assert ledger.cash == 1_000.0


# ---------------------------------------------------------------------------
# Selling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelling:
    def test_a_partial_sell_reduces_the_quantity_and_realises_pnl(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-4, price=110.0, fee=0.0, timestamp=T1)
        assert ledger.position_quantity("AAPL") == 6
        assert ledger.average_entry_price("AAPL") == pytest.approx(100.0)
        assert ledger.realized_pnl == pytest.approx(40.0)

    def test_a_full_sell_closes_the_position(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-10, price=110.0, fee=0.0, timestamp=T1)
        assert ledger.position_quantity("AAPL") == 0.0
        assert ledger.open_positions() == {}
        assert ledger.average_entry_price("AAPL") == 0.0
        assert ledger.realized_pnl == pytest.approx(100.0)
        assert ledger.cash == pytest.approx(10_100.0)

    def test_realised_pnl_is_gross_and_fees_are_separate(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-10, price=110.0, fee=3.0, timestamp=T1)
        assert ledger.realized_pnl == pytest.approx(100.0)
        assert ledger.net_realized_pnl == pytest.approx(97.0)
        assert ledger.total_fees == pytest.approx(3.0)

    def test_a_sell_larger_than_the_holding_is_refused(self):
        """Long-only: an oversized sell must not become a short."""
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=5, price=100.0, fee=0.0, timestamp=T0)
        with pytest.raises(InsufficientPositionError):
            ledger.apply_fill(symbol="AAPL", quantity=-6, price=110.0, fee=0.0, timestamp=T1)
        assert ledger.position_quantity("AAPL") == 5
        assert ledger.cash == pytest.approx(9_500.0)

    def test_selling_a_symbol_never_held_is_refused(self):
        ledger = _ledger(10_000.0)
        with pytest.raises(InsufficientPositionError):
            ledger.apply_fill(symbol="TSLA", quantity=-1, price=200.0, fee=0.0, timestamp=T0)
        assert ledger.cash == 10_000.0


# ---------------------------------------------------------------------------
# Invalid fills
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInvalidFills:
    def test_a_zero_quantity_fill_is_refused(self):
        with pytest.raises(LedgerError):
            _ledger().apply_fill(symbol="AAPL", quantity=0, price=100.0, fee=0.0, timestamp=T0)

    def test_a_non_positive_price_is_refused(self):
        for bad in (0.0, -1.0):
            with pytest.raises(LedgerError):
                _ledger().apply_fill(symbol="AAPL", quantity=1, price=bad, fee=0.0, timestamp=T0)

    def test_a_negative_fee_is_refused(self):
        with pytest.raises(LedgerError):
            _ledger().apply_fill(symbol="AAPL", quantity=1, price=100.0, fee=-1.0, timestamp=T0)


# ---------------------------------------------------------------------------
# Marks and unrealised P&L
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarks:
    def test_unrealised_pnl_is_zero_until_the_position_is_marked(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.holdings["AAPL"].last_price = None
        assert ledger.unrealized_pnl == 0.0

    def test_a_mark_moves_unrealised_pnl_and_equity(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.mark("AAPL", 110.0)
        assert ledger.unrealized_pnl == pytest.approx(100.0)
        assert ledger.equity == pytest.approx(10_100.0)

    def test_a_non_positive_mark_is_refused(self):
        ledger = _ledger()
        with pytest.raises(LedgerError):
            ledger.mark("AAPL", 0.0)

    def test_marking_one_symbol_leaves_another_alone(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="MSFT", quantity=5, price=200.0, fee=0.0, timestamp=T0)
        ledger.mark("AAPL", 110.0)
        assert ledger.unrealized_pnl == pytest.approx(100.0)
        ledger.mark_many({"MSFT": 210.0})
        assert ledger.unrealized_pnl == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Conservation identities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConservation:
    def test_cash_conservation_over_a_buy(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=2.0, timestamp=T0)
        assert ledger.cash == pytest.approx(10_000.0 - 1_000.0 - 2.0)

    def test_a_round_trip_realises_exactly_the_price_change_less_fees(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-10, price=105.0, fee=1.0, timestamp=T1)
        # gross 10 * 5 = 50; realised_pnl is gross, and net_realized_pnl takes
        # both fees off it.
        assert ledger.realized_pnl == pytest.approx(50.0)
        assert ledger.net_realized_pnl == pytest.approx(48.0)
        assert ledger.total_fees == pytest.approx(2.0)
        # A flat book's equity is its cash: the round trip moved cash by exactly
        # the gross realised P&L less both fees.
        assert ledger.equity == pytest.approx(10_000.0 + 50.0 - 2.0)
        assert ledger.total_pnl == pytest.approx(48.0)

    def test_equity_equals_cash_plus_marked_value(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        ledger.mark("AAPL", 123.0)
        assert ledger.equity == pytest.approx(ledger.cash + ledger.market_value)
        assert ledger.market_value == pytest.approx(1_230.0)

    def test_a_flat_book_has_no_exposure(self):
        ledger = _ledger(10_000.0)
        assert ledger.total_exposure == 0.0

    def test_exposure_is_the_marked_share_of_equity(self):
        ledger = _ledger(1_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=5, price=100.0, fee=0.0, timestamp=T0)
        # cash 500, market value 500, equity 1000 -> exposure 0.5
        assert ledger.total_exposure == pytest.approx(0.5)

    def test_many_trades_preserve_the_accounting_identity(self):
        """cash + cost basis + fees - gross realised P&L == initial cash."""
        ledger = _ledger(10_000.0)
        fills = [
            ("AAPL", 10, 100.0, 1.0), ("MSFT", 5, 200.0, 1.0),
            ("AAPL", -4, 110.0, 1.0), ("AAPL", 6, 105.0, 1.0),
            ("MSFT", -5, 190.0, 1.0),
        ]
        for symbol, qty, price, fee in fills:
            ledger.apply_fill(symbol=symbol, quantity=qty, price=price, fee=fee,
                              timestamp=T0)
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert (
            ledger.cash + basis + ledger.total_fees - ledger.realized_pnl
            == pytest.approx(10_000.0)
        )

    def test_the_identity_holds_with_a_standalone_fee(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        ledger.charge_fee(amount=4.0, timestamp=T1, reason="custody")
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert (
            ledger.cash + basis + ledger.total_fees - ledger.realized_pnl
            == pytest.approx(10_000.0)
        )


# ---------------------------------------------------------------------------
# Fees with no market side
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStandaloneFees:
    def test_a_fee_reduces_cash(self):
        ledger = _ledger(1_000.0)
        ledger.charge_fee(amount=5.0, timestamp=T0, reason="custody")
        assert ledger.cash == pytest.approx(995.0)
        assert ledger.total_fees == pytest.approx(5.0)

    def test_a_fee_beyond_the_cash_is_refused(self):
        ledger = _ledger(10.0)
        with pytest.raises(InsufficientFundsError):
            ledger.charge_fee(amount=20.0, timestamp=T0)
        assert ledger.cash == 10.0

    def test_a_negative_fee_is_refused(self):
        with pytest.raises(LedgerError):
            _ledger().charge_fee(amount=-1.0, timestamp=T0)


# ---------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventLog:
    def test_a_buy_writes_ordered_events(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        types = [e.type for e in ledger.events]
        assert types == [EventType.ORDER_FILLED, EventType.POSITION_OPENED, EventType.FEE_CHARGED]

    def test_sequence_numbers_are_dense_and_monotonic(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-4, price=110.0, fee=1.0, timestamp=T1)
        ledger.apply_fill(symbol="AAPL", quantity=-6, price=120.0, fee=1.0, timestamp=T2)
        assert [e.seq for e in ledger.events] == list(range(1, len(ledger.events) + 1))

    def test_position_events_distinguish_open_increase_reduce_and_close(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=5, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-4, price=100.0, fee=0.0, timestamp=T0)
        ledger.apply_fill(symbol="AAPL", quantity=-11, price=100.0, fee=0.0, timestamp=T0)
        position_events = [e.type for e in ledger.events if e.type.value.startswith("POSITION_")]
        assert position_events == [
            EventType.POSITION_OPENED,
            EventType.POSITION_INCREASED,
            EventType.POSITION_REDUCED,
            EventType.POSITION_CLOSED,
        ]

    def test_events_are_immutable(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=1, price=100.0, fee=0.0, timestamp=T0)
        with pytest.raises(ValidationError):
            ledger.events[0].seq = 99

    def test_events_carry_no_free_text_from_a_model(self):
        """The detail dict is figures and flags; a model's prose must not land in it."""
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=1, price=100.0, fee=0.0, timestamp=T0,
                          reason="signal ab12cd34")
        for event in ledger.events:
            assert isinstance(event.detail, dict)
            for value in event.detail.values():
                assert isinstance(value, (int, float, str, bool))

    def test_a_rejected_fill_writes_no_event(self):
        ledger = _ledger(10.0)
        with pytest.raises(InsufficientFundsError):
            ledger.apply_fill(symbol="AAPL", quantity=1, price=100.0, fee=0.0, timestamp=T0)
        assert ledger.events == []


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshot:
    def test_a_snapshot_reports_the_whole_book(self):
        ledger = _ledger(10_000.0)
        ledger.apply_fill(symbol="AAPL", quantity=10, price=100.0, fee=1.0, timestamp=T0)
        ledger.mark("AAPL", 110.0)
        snap = ledger.snapshot(T1)
        assert snap.cash == pytest.approx(8_999.0)
        assert snap.market_value == pytest.approx(1_100.0)
        assert snap.equity == pytest.approx(10_099.0)
        assert snap.unrealized_pnl == pytest.approx(100.0)
        assert snap.total_fees == pytest.approx(1.0)
        assert snap.positions == {"AAPL": 10}
        assert snap.prices == {"AAPL": 110.0}

    def test_a_snapshot_has_no_margin_buying_power(self):
        ledger = _ledger(1_000.0)
        snap = ledger.snapshot(T0)
        assert snap.available_cash == snap.cash
        assert snap.buying_power == snap.cash


@pytest.mark.unit
def test_ledger_event_is_exported_from_the_package():
    assert LedgerEvent is not None
