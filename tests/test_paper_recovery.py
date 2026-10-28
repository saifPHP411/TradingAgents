"""Paper-session persistence, restart recovery and reliability.

Phase 3 of the simulated trading system: the paper session must survive a
process shutdown, a restart, a price-source failure, malformed market data, a
duplicate step, and a partially filled resting order — without changing the
accounting or filling an order twice.

The tests here are written to *catch a premise error*, not to confirm the code.
Each scenario runs the same sequence twice: once straight through, and once with
a save/restore in the middle. The two runs must agree on every accounting figure
and on the event semantics, or the recovery mechanism is a story rather than a
proof. A JSON reporting snapshot is explicitly *not* treated as recovery; the
state file is a separate, versioned format, and the tests show the snapshot alone
would lose the ledger's event log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.execution import (
    PAPER_STATE_SCHEMA_VERSION,
    ExecutionSimulator,
    PaperSession,
    PaperSessionError,
    PaperStateError,
    Quote,
    dict_quote_source,
    new_session,
)
from tradingagents.execution.costs import CostModel
from tradingagents.risk import RiskLimits
from tradingagents.signals import (
    OrderIntent,
    SignalAction,
    SignalTimeInForce,
    TradingSignal,
)

NOW = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def _signal(action=SignalAction.BUY, *, symbol="AAPL", size=0.2, at=NOW) -> TradingSignal:
    kwargs = {} if action is SignalAction.HOLD else {"position_size": size}
    return TradingSignal(
        symbol=symbol, action=action, confidence=0.8, timestamp=at,
        as_of_date=at.date().isoformat(), **kwargs,
    )


def _limit(symbol="AAPL", limit_price=95.0, *, size=1.0, at=NOW, expires=None,
           entry_price=100.0) -> TradingSignal:
    return TradingSignal(
        symbol=symbol, action=SignalAction.BUY, confidence=0.8, timestamp=at,
        as_of_date=at.date().isoformat(), position_size=size,
        order_intent=OrderIntent.LIMIT, limit_price=limit_price, entry_price=entry_price,
        time_in_force=SignalTimeInForce.GTC, expires_session=expires,
    )


def _session(*, cash=100_000.0, quotes=None, costs=None, participation=0.0, limits=None):
    session = new_session(
        initial_cash=cash,
        quotes=dict_quote_source(quotes or {"AAPL": 100.0}, timestamp=NOW),
        limits=limits or RiskLimits(max_position_size=0.5, max_portfolio_exposure=None),
        costs=costs or CostModel(),
    )
    if participation:
        session.simulator = ExecutionSimulator(
            session.ledger, costs=session.simulator.costs, mode="PAPER",
            max_volume_participation=participation,
        )
    return session


def _accounting(session: PaperSession) -> dict:
    """The accounting figures two runs must agree on."""
    ledger = session.ledger
    return {
        "cash": ledger.cash,
        "equity": ledger.equity,
        "realized_pnl": ledger.realized_pnl,
        "unrealized_pnl": ledger.unrealized_pnl,
        "total_fees": ledger.total_fees,
        "positions": {s: h.quantity for s, h in ledger.open_positions().items()},
    }


def _event_semantics(session: PaperSession) -> list[tuple]:
    """The event log with volatile fields removed: type, symbol and reason."""
    return [(e.type.value, e.symbol, e.reason) for e in session.ledger.events]


# ---------------------------------------------------------------------------
# Scenario A vs B: restart at the same point changes nothing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRestartEqualsNoRestart:
    """The headline guarantee: a restart must not change the outcome."""

    def _run(self, *, restart: bool):
        session = _session(costs=CostModel(commission_rate=0.001))
        session.on_signal(_signal(size=0.2))
        if restart:
            session = PaperSession.restore(
                session.state(),
                quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW),
            )
        session.on_signal(_signal(size=0.2, at=NOW + timedelta(minutes=1)))
        return session

    def test_scenario_a_and_b_agree_on_every_figure(self):
        straight = self._run(restart=False)
        restarted = self._run(restart=True)
        assert _accounting(straight) == _accounting(restarted)

    def test_scenario_a_and_b_agree_on_the_event_semantics(self):
        straight = self._run(restart=False)
        restarted = self._run(restart=True)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_the_restored_event_sequence_continues_rather_than_restarting(self):
        restarted = self._run(restart=True)
        seqs = [e.seq for e in restarted.ledger.events]
        assert seqs == list(range(1, len(seqs) + 1))

    def test_no_fill_is_duplicated_across_the_restart(self):
        straight = self._run(restart=False)
        restarted = self._run(restart=True)
        filled = [
            (e.symbol, e.detail.get("quantity"))
            for e in restarted.ledger.events if e.type.value == "ORDER_FILLED"
        ]
        straight_filled = [
            (e.symbol, e.detail.get("quantity"))
            for e in straight.ledger.events if e.type.value == "ORDER_FILLED"
        ]
        assert filled == straight_filled


# ---------------------------------------------------------------------------
# The reporting snapshot is not recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotIsNotRecovery:
    def test_the_snapshot_does_not_carry_the_event_log(self):
        session = _session()
        session.on_signal(_signal())
        # A snapshot is a reporting view; it deliberately omits the ledger's
        # event log, so it cannot rebuild the book by replay.
        assert "events" not in session.snapshot()["ledger"]

    def test_the_state_does_carry_the_event_log(self):
        session = _session()
        session.on_signal(_signal())
        assert session.state()["ledger"]["events"]

    def test_the_state_is_versioned(self):
        assert _session().state()["schema_version"] == PAPER_STATE_SCHEMA_VERSION

    def test_the_state_is_json_safe(self):
        session = _session()
        session.on_signal(_signal())
        json.dumps(session.state())

    def test_a_snapshot_cannot_be_restored_as_state(self):
        session = _session()
        session.on_signal(_signal())
        with pytest.raises(PaperStateError):
            PaperSession.restore(session.snapshot(), quotes=dict_quote_source({"AAPL": 100.0}))

    def test_an_unknown_schema_version_is_refused(self):
        state = _session().state()
        state["schema_version"] = 99
        with pytest.raises(PaperStateError):
            PaperSession.restore(state, quotes=dict_quote_source({"AAPL": 100.0}))

    def test_a_malformed_state_file_is_refused(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json")
        with pytest.raises(PaperStateError):
            PaperSession.restore({}, quotes=dict_quote_source({"AAPL": 100.0}), path=path)


# ---------------------------------------------------------------------------
# GTC LIMIT recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRestingLimitRecovery:
    def test_a_resting_limit_fills_once_after_a_restart(self):
        session = _session(costs=CostModel(commission_rate=0.001))
        session.on_signal(_limit(limit_price=95.0))
        assert len(session.resting) == 1
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1)),
        )
        steps = restored.work_resting()
        assert [s.order_status for s in steps] == ["FILLED"]
        assert restored.ledger.position_quantity("AAPL") > 0
        assert restored.resting == []
        filled = [e for e in restored.ledger.events if e.type.value == "ORDER_FILLED"]
        assert len(filled) == 1

    def test_the_restored_resting_order_keeps_its_identity(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        original = session.resting[0]
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert restored.resting[0].order_id == original.order_id
        assert restored.resting[0].status.value == "ACCEPTED"
        assert restored.resting[0].remaining_quantity == original.remaining_quantity

    def test_the_restored_fill_price_matches_the_uninterrupted_run(self):
        def run(restart):
            session = _session(costs=CostModel(commission_rate=0.001))
            session.on_signal(_limit(limit_price=95.0))
            if restart:
                session = PaperSession.restore(
                    session.state(),
                    quotes=dict_quote_source(
                        {"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1)
                    ),
                )
            else:
                session.quotes = dict_quote_source(
                    {"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1)
                )
            session.work_resting()
            return session

        straight = run(False)
        restarted = run(True)
        assert _accounting(straight) == _accounting(restarted)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_a_resting_limit_that_never_fills_stays_live_across_a_restart(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 99.0}, timestamp=NOW + timedelta(hours=1)),
        )
        restored.work_resting()
        assert len(restored.resting) == 1
        assert restored.ledger.position_quantity("AAPL") == 0.0


# ---------------------------------------------------------------------------
# Expiration across a restart
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpirationAfterRestart:
    def test_an_order_expires_exactly_once_after_a_restart(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0, expires="2026-01-05"))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 94.0}, timestamp=NOW + timedelta(days=3)),
        )
        steps = restored.work_resting()
        assert [s.order_status for s in steps] == ["CANCELLED"]
        assert restored.resting == []
        assert restored.ledger.position_quantity("AAPL") == 0.0
        cancelled = [e for e in restored.ledger.events if e.type.value == "ORDER_CANCELLED"]
        assert len(cancelled) == 1
        assert cancelled[0].detail.get("reason") == "EXPIRED"

    def test_an_expired_order_is_not_filled_on_a_later_session(self):
        """A price that reaches the limit after expiry must not fill it."""
        session = _session()
        session.on_signal(_limit(limit_price=95.0, expires="2026-01-05"))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 90.0}, timestamp=NOW + timedelta(days=3)),
        )
        restored.work_resting()
        assert restored.ledger.position_quantity("AAPL") == 0.0
        assert not [e for e in restored.ledger.events if e.type.value == "ORDER_FILLED"]

    def test_an_order_with_no_expiry_never_expires(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 99.0}, timestamp=NOW + timedelta(days=30)),
        )
        restored.work_resting()
        assert len(restored.resting) == 1

    def test_expiry_is_not_retriggered_by_a_second_work(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0, expires="2026-01-05"))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 94.0}, timestamp=NOW + timedelta(days=3)),
        )
        restored.work_resting()
        restored.work_resting()
        cancelled = [e for e in restored.ledger.events if e.type.value == "ORDER_CANCELLED"]
        assert len(cancelled) == 1


# ---------------------------------------------------------------------------
# Partial-fill recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialFillRecovery:
    def _partial_session(self, *, size=0.94, price=94.0, volume=4000.0, costs=None):
        # 0.94 * 100,000 = 94,000 notional; at 94 that is 1,000 units, and 10% of
        # a 4,000-volume quote is 400, so the first session fills 400.
        session = _session(
            cash=100_000.0, quotes={"AAPL": price},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            participation=0.1, costs=costs,
        )
        quote = Quote(symbol="AAPL", price=price, timestamp=NOW, average_daily_volume=volume)
        session.quotes = lambda symbol: quote
        session.on_signal(_limit(size=size))
        return session

    def test_a_four_hundred_then_six_hundred_fill_totals_one_thousand(self):
        session = self._partial_session()
        assert session.resting[0].filled_quantity == pytest.approx(400.0)
        restored = PaperSession.restore(
            session.state(),
            quotes=lambda symbol: Quote(
                symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1),
                average_daily_volume=6000.0,
            ),
        )
        restored.work_resting()
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(1000.0)
        assert restored.resting == []

    def test_the_restored_remainder_is_the_original_remainder(self):
        session = self._partial_session()
        original = session.resting[0]
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW),
        )
        assert restored.resting[0].filled_quantity == pytest.approx(original.filled_quantity)
        assert restored.resting[0].remaining_quantity == pytest.approx(
            original.remaining_quantity
        )
        assert restored.resting[0].average_fill_price == pytest.approx(
            original.average_fill_price
        )

    def test_a_restart_does_not_double_the_position(self):
        session = self._partial_session()
        restored = PaperSession.restore(
            session.state(),
            quotes=lambda symbol: Quote(
                symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1),
                average_daily_volume=6000.0,
            ),
        )
        restored.work_resting()
        # 400 + 600 = 1,000, not 1,400: the first piece is not replayed.
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(1000.0)

    def test_fees_are_charged_once_per_actual_fill(self):
        session = self._partial_session(costs=CostModel(commission_per_unit=0.01))
        restored = PaperSession.restore(
            session.state(),
            quotes=lambda symbol: Quote(
                symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1),
                average_daily_volume=6000.0,
            ),
        )
        restored.work_resting()
        # 400 * 0.01 + 600 * 0.01 = 10.0, not 14.0.
        assert restored.ledger.total_fees == pytest.approx(10.0)

    def test_the_weighted_average_fill_price_is_correct(self):
        session = self._partial_session()
        restored = PaperSession.restore(
            session.state(),
            quotes=lambda symbol: Quote(
                symbol="AAPL", price=90.0, timestamp=NOW + timedelta(hours=1),
                average_daily_volume=6000.0,
            ),
        )
        restored.work_resting()
        # 400 at 94 and 600 at 90 -> (400*94 + 600*90) / 1000 = 91.6.
        assert restored.ledger.average_entry_price("AAPL") == pytest.approx(91.6)

    def test_a_three_piece_fill_totals_the_order(self):
        session = self._partial_session()
        restored = PaperSession.restore(
            session.state(),
            quotes=lambda symbol: Quote(
                symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1),
                average_daily_volume=2000.0,
            ),
        )
        restored.work_resting()  # +200 -> 600
        assert restored.resting[0].filled_quantity == pytest.approx(600.0)
        restored.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=2),
            average_daily_volume=4000.0,
        )
        restored.work_resting()  # +400 -> 1000
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(1000.0)
        assert restored.resting == []

    def test_a_partial_fill_then_cancel_leaves_the_filled_part_only(self):
        session = self._partial_session()
        order = session.resting[0]
        session.cancel_resting(order, detail="manual cancel")
        assert session.resting == []
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)
        assert len([e for e in session.ledger.events if e.type.value == "ORDER_CANCELLED"]) == 1

    def test_a_partial_fill_then_expiry_keeps_the_filled_part(self):
        session = _session(
            cash=100_000.0, quotes={"AAPL": 94.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            participation=0.1,
        )
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW, average_daily_volume=4000.0
        )
        session.on_signal(_limit(size=0.94, expires="2026-01-05"))
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW + timedelta(days=2),
            average_daily_volume=100_000.0,
        )
        steps = session.work_resting()
        assert [s.order_status for s in steps] == ["CANCELLED"]
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)

    def test_a_partial_fill_then_risk_rejection_of_the_remainder(self):
        """A tightened cap can leave no room for a resting order's remainder."""
        # The first session fills 400 at 94; the remainder is worked after the cap
        # is tightened, so no positive size keeps the caps and it is rejected.
        session = _session(
            cash=1_000_000.0, quotes={"AAPL": 94.0},
            limits=RiskLimits(max_position_size=0.5, max_portfolio_exposure=None),
            participation=0.1,
        )
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW, average_daily_volume=4000.0
        )
        session.on_signal(_limit(size=0.5))  # ~5,300 units requested; first fill 400
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)
        # Tighten the cap so the remainder has no headroom.
        session.risk.limits = RiskLimits(
            max_position_size=0.0001, max_portfolio_exposure=None,
            default_position_size=0.0001,
        )
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1),
            average_daily_volume=100_000.0,
        )
        steps = session.work_resting()
        assert [s.order_status for s in steps] == ["REJECTED"]
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)
        assert session.resting == []
        rejected = [e for e in session.ledger.events if e.type.value == "ORDER_REJECTED"]
        assert rejected and rejected[-1].detail.get("reason") == "RISK_LIMIT"


# ---------------------------------------------------------------------------
# Duplicate processing / idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateProcessing:
    def test_the_same_signal_driven_twice_fills_once(self):
        session = _session()
        signal = _signal(size=0.2)
        session.on_signal(signal)
        session.on_signal(signal)
        assert session.ledger.position_quantity("AAPL") == pytest.approx(200.0)
        assert len([e for e in session.ledger.events if e.type.value == "ORDER_FILLED"]) == 1

    def test_a_duplicate_is_recorded_not_silent(self):
        session = _session()
        signal = _signal(size=0.2)
        session.on_signal(signal)
        second = session.on_signal(signal)
        assert second.order_status == "DUPLICATE_SIGNAL"
        assert not second.approved
        assert "already processed" in second.reason

    def test_a_distinct_signal_still_processes(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        session.on_signal(_signal(size=0.1, at=NOW + timedelta(minutes=1)))
        assert session.ledger.position_quantity("AAPL") > 0

    def test_the_processed_set_survives_a_restart(self):
        session = _session()
        signal = _signal(size=0.2)
        session.on_signal(signal)
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        step = restored.on_signal(signal)
        assert step.order_status == "DUPLICATE_SIGNAL"
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(200.0)

    def test_working_the_resting_queue_rolls_the_day(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        first_day = session.day
        session.quotes = dict_quote_source(
            {"AAPL": 99.0}, timestamp=NOW + timedelta(days=2)
        )
        session.work_resting()
        assert session.day != first_day
        assert session.day == (NOW + timedelta(days=2)).date().isoformat()

    def test_a_duplicate_does_not_charge_a_second_fee(self):
        session = _session(costs=CostModel(commission_rate=0.001))
        signal = _signal(size=0.2)
        session.on_signal(signal)
        fees = session.ledger.total_fees
        session.on_signal(signal)
        assert session.ledger.total_fees == pytest.approx(fees)

    def test_working_the_resting_queue_twice_fills_once(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        session.quotes = dict_quote_source({"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1))
        session.work_resting()
        held = session.ledger.position_quantity("AAPL")
        session.work_resting()
        assert session.ledger.position_quantity("AAPL") == pytest.approx(held)

    def test_repeated_work_after_restart_does_not_refill(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source({"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1)),
        )
        restored.work_resting()
        held = restored.ledger.position_quantity("AAPL")
        restored.work_resting()
        restored.work_resting()
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(held)


# ---------------------------------------------------------------------------
# Market-data failure handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarketDataFailures:
    def _resting_session(self, *, symbol="AAPL"):
        session = _session(quotes={symbol: 100.0})
        session.on_signal(_limit(symbol=symbol, limit_price=95.0))
        return session

    def test_a_nan_quote_is_refused_at_construction(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=float("nan"), timestamp=NOW)

    def test_an_infinite_quote_is_refused_at_construction(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=float("inf"), timestamp=NOW)

    def test_a_non_positive_quote_is_refused_at_construction(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=0.0, timestamp=NOW)

    def test_a_malformed_range_is_refused(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=100.0, timestamp=NOW, low=105.0, high=95.0)

    def test_a_non_finite_range_is_refused(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=100.0, timestamp=NOW, low=float("nan"))

    def test_a_missing_quote_leaves_the_order_resting_not_cancelled(self):
        session = self._resting_session()
        session.quotes = dict_quote_source({"MSFT": 100.0}, timestamp=NOW)
        session.work_resting()
        assert len(session.resting) == 1
        assert session.ledger.position_quantity("AAPL") == 0.0
        assert not [e for e in session.ledger.events if e.type.value == "ORDER_CANCELLED"]

    def test_a_price_source_exception_preserves_the_book_and_the_order(self):
        session = self._resting_session()

        def boom(symbol):
            raise RuntimeError("vendor down")

        session.quotes = boom
        with pytest.raises(RuntimeError):
            session.work_resting()
        # The book did not move, the order is still live, and no event was faked.
        assert session.ledger.position_quantity("AAPL") == 0.0
        assert len(session.resting) == 1
        assert session.resting[0].status.value == "ACCEPTED"

    def test_a_price_source_exception_on_one_symbol_does_not_stall_others(self):
        session = _session(quotes={"AAPL": 100.0, "MSFT": 100.0})
        session.on_signal(_limit(symbol="AAPL", limit_price=95.0))
        session.on_signal(_limit(symbol="MSFT", limit_price=95.0))

        def flaky(symbol):
            if symbol == "AAPL":
                return Quote(symbol="AAPL", price=94.0, timestamp=NOW + timedelta(hours=1))
            raise RuntimeError("vendor down")

        session.quotes = flaky
        with pytest.raises(RuntimeError):
            session.work_resting()
        # The healthy symbol filled; the broken one kept its order, and the queue
        # still holds exactly the order that could not be worked.
        assert session.ledger.position_quantity("AAPL") > 0
        assert session.ledger.position_quantity("MSFT") == 0.0
        assert [o.symbol for o in session.resting] == ["MSFT"]

    def test_a_source_returning_the_wrong_symbol_is_refused(self):
        session = _session()
        session.quotes = lambda symbol: Quote(symbol="MSFT", price=100.0, timestamp=NOW)
        with pytest.raises(PaperSessionError):
            session.on_signal(_signal())

    def test_a_failed_source_does_not_consume_the_signal(self):
        """A data outage is not a processed decision; the signal stays retryable."""
        session = _session()

        def boom(symbol):
            raise RuntimeError("vendor down")

        session.quotes = boom
        signal = _signal(size=0.2)
        with pytest.raises(RuntimeError):
            session.on_signal(signal)
        session.quotes = dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        step = session.on_signal(signal)
        assert step.order_status == "FILLED"

    def test_a_closed_market_leaves_a_resting_order_live(self):
        session = self._resting_session()
        session.quotes = dict_quote_source(
            {"AAPL": 94.0}, timestamp=NOW + timedelta(hours=1), is_open=False
        )
        session.work_resting()
        assert len(session.resting) == 1
        assert session.resting[0].status.value == "ACCEPTED"
        assert session.ledger.position_quantity("AAPL") == 0.0
        assert not [e for e in session.ledger.events if e.type.value == "ORDER_REJECTED"]

    def test_a_closed_market_does_not_corrupt_the_book(self):
        session = _session()
        session.quotes = dict_quote_source(
            {"AAPL": 100.0}, timestamp=NOW, is_open=False
        )
        step = session.on_signal(_signal(size=0.2))
        assert step.order_status != "FILLED"
        assert session.ledger.position_quantity("AAPL") == 0.0
        assert session.ledger.cash == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Snapshot correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotCorrectness:
    def test_a_resting_order_reads_accepted_not_filled(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        order = session.snapshot()["resting_orders"][0]
        assert order["status"] == "ACCEPTED"
        assert order["filled_quantity"] == 0.0
        assert order["remaining_quantity"] == order["quantity"]

    def test_a_partially_filled_order_reads_partially_filled(self):
        session = _session(
            cash=100_000.0, quotes={"AAPL": 94.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            participation=0.1,
        )
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW, average_daily_volume=4000.0
        )
        session.on_signal(_limit(size=0.94))
        order = session.snapshot()["resting_orders"][0]
        assert order["status"] == "PARTIALLY_FILLED"
        assert order["filled_quantity"] == pytest.approx(400.0)
        assert order["remaining_quantity"] == pytest.approx(order["quantity"] - 400.0)
        assert order["average_fill_price"] == pytest.approx(94.0)

    def test_the_snapshot_reports_time_in_force_and_expiry(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0, expires="2026-02-01"))
        order = session.snapshot()["resting_orders"][0]
        assert order["time_in_force"] == "GTC"
        assert order["expires_session"] == "2026-02-01"

    def test_the_snapshot_reports_the_book_and_the_clock(self):
        session = _session()
        session.on_signal(_signal(size=0.2))
        snap = session.snapshot()
        assert snap["cash"] == pytest.approx(session.ledger.cash)
        assert snap["total_fees"] == pytest.approx(session.ledger.total_fees)
        assert snap["realized_pnl"] == pytest.approx(session.ledger.realized_pnl)
        assert snap["unrealized_pnl"] == pytest.approx(session.ledger.unrealized_pnl)
        assert snap["current_time"] == NOW.isoformat()

    def test_the_snapshot_is_json_safe_with_orders_present(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0))
        json.dumps(session.snapshot())

    def test_the_snapshot_positions_match_the_ledger(self):
        session = _session()
        session.on_signal(_signal(size=0.2))
        snap = session.snapshot()
        for symbol, holding in session.ledger.open_positions().items():
            assert snap["positions"][symbol]["quantity"] == pytest.approx(holding.quantity)
            assert snap["positions"][symbol]["average_entry_price"] == pytest.approx(
                holding.average_entry_price
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def _sequence(self, session: PaperSession):
        session.on_signal(_signal(size=0.2))
        session.on_signal(_limit(limit_price=95.0, size=0.2, at=NOW + timedelta(minutes=1)))
        session.quotes = dict_quote_source(
            {"AAPL": 94.0}, timestamp=NOW + timedelta(minutes=2)
        )
        session.work_resting()
        session.on_signal(_signal(SignalAction.SELL, size=1.0, at=NOW + timedelta(minutes=3)))
        return session

    def test_run_a_equals_run_b(self):
        first = self._sequence(_session(costs=CostModel(commission_rate=0.001)))
        second = self._sequence(_session(costs=CostModel(commission_rate=0.001)))
        assert _accounting(first) == _accounting(second)
        assert _event_semantics(first) == _event_semantics(second)

    def test_run_c_restart_in_the_middle_matches_run_a(self):
        straight = self._sequence(_session(costs=CostModel(commission_rate=0.001)))

        partial = _session(costs=CostModel(commission_rate=0.001))
        partial.on_signal(_signal(size=0.2))
        resumed = PaperSession.restore(
            partial.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        resumed.on_signal(_limit(limit_price=95.0, size=0.2, at=NOW + timedelta(minutes=1)))
        resumed.quotes = dict_quote_source(
            {"AAPL": 94.0}, timestamp=NOW + timedelta(minutes=2)
        )
        resumed.work_resting()
        resumed.on_signal(
            _signal(SignalAction.SELL, size=1.0, at=NOW + timedelta(minutes=3))
        )

        assert _accounting(straight) == _accounting(resumed)
        assert _event_semantics(straight) == _event_semantics(resumed)

    def test_a_state_round_trip_is_byte_stable(self):
        session = _session()
        session.on_signal(_signal(size=0.2))
        once = json.dumps(session.state(), sort_keys=True)
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert json.dumps(restored.state(), sort_keys=True) == once


# ---------------------------------------------------------------------------
# Multiple resting orders and multiple symbols
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultipleRestingOrders:
    def test_multiple_resting_orders_survive_a_restart(self):
        session = _session(quotes={"AAPL": 100.0, "MSFT": 200.0})
        session.on_signal(_limit(symbol="AAPL", limit_price=95.0))
        session.on_signal(_limit(symbol="MSFT", limit_price=190.0, entry_price=200.0))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source(
                {"AAPL": 94.0, "MSFT": 189.0}, timestamp=NOW + timedelta(hours=1)
            ),
        )
        assert {o.symbol for o in restored.resting} == {"AAPL", "MSFT"}
        steps = restored.work_resting()
        assert {s.symbol for s in steps} == {"AAPL", "MSFT"}
        assert restored.resting == []
        assert restored.ledger.position_quantity("AAPL") > 0
        assert restored.ledger.position_quantity("MSFT") > 0

    def test_only_the_symbol_whose_limit_is_reached_fills(self):
        session = _session(quotes={"AAPL": 100.0, "MSFT": 200.0})
        session.on_signal(_limit(symbol="AAPL", limit_price=95.0))
        session.on_signal(_limit(symbol="MSFT", limit_price=190.0, entry_price=200.0))
        session.quotes = dict_quote_source(
            {"AAPL": 94.0, "MSFT": 199.0}, timestamp=NOW + timedelta(hours=1)
        )
        session.work_resting()
        assert session.ledger.position_quantity("AAPL") > 0
        assert session.ledger.position_quantity("MSFT") == 0.0
        assert [o.symbol for o in session.resting] == ["MSFT"]

    def test_multiple_symbols_survive_a_restart_with_their_positions(self):
        session = _session(quotes={"AAPL": 100.0, "MSFT": 200.0})
        session.on_signal(_signal(symbol="AAPL", size=0.1))
        session.on_signal(_signal(symbol="MSFT", size=0.1))
        restored = PaperSession.restore(
            session.state(),
            quotes=dict_quote_source(
                {"AAPL": 100.0, "MSFT": 200.0}, timestamp=NOW + timedelta(hours=1)
            ),
        )
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(
            session.ledger.position_quantity("AAPL")
        )
        assert restored.ledger.position_quantity("MSFT") == pytest.approx(
            session.ledger.position_quantity("MSFT")
        )
        assert set(restored.ledger.open_positions()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# The safety boundary is unchanged by recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecoveryKeepsTheSafetyBoundary:
    def test_a_restored_session_is_still_paper_only(self):
        session = _session()
        session.on_signal(_signal())
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert restored.mode.value == "PAPER"
        assert restored.simulator.mode.value == "PAPER"

    def test_a_state_claiming_a_live_mode_is_refused(self):
        state = _session().state()
        state["mode"] = "LIVE"
        with pytest.raises(PaperStateError):
            PaperSession.restore(state, quotes=dict_quote_source({"AAPL": 100.0}))

    def test_a_state_with_a_live_simulator_mode_is_refused(self):
        state = _session().state()
        state["simulator"]["mode"] = "live"
        with pytest.raises(PaperStateError):
            PaperSession.restore(state, quotes=dict_quote_source({"AAPL": 100.0}))

    def test_saving_and_restoring_preserves_the_risk_limits(self):
        session = _session(
            limits=RiskLimits(max_position_size=0.25, max_portfolio_exposure=None)
        )
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert restored.risk.limits.max_position_size == pytest.approx(0.25)

    def test_saving_and_restoring_preserves_the_cost_schedule(self):
        session = _session(costs=CostModel(commission_rate=0.001, slippage_bps=5.0))
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert restored.simulator.costs.commission_rate == pytest.approx(0.001)
        assert restored.simulator.costs.slippage_bps == pytest.approx(5.0)

    def test_saving_and_restoring_preserves_participation(self):
        session = _session(participation=0.1)
        restored = PaperSession.restore(
            session.state(), quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )
        assert restored.simulator.max_volume_participation == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# File-based persistence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilePersistence:
    def test_state_round_trips_through_a_file(self, tmp_path):
        session = _session(costs=CostModel(commission_rate=0.001))
        session.on_signal(_signal(size=0.2))
        path = session.save_state(tmp_path / "paper" / "state.json")
        restored = PaperSession.restore(
            {}, quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW), path=path
        )
        assert _accounting(restored) == _accounting(session)
        assert restored.processed_signal_ids == session.processed_signal_ids

    def test_a_restored_file_continues_the_session(self, tmp_path):
        session = _session(costs=CostModel(commission_rate=0.001))
        session.on_signal(_signal(size=0.2))
        path = session.save_state(tmp_path / "state.json")
        restored = PaperSession.restore(
            {}, quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW), path=path
        )
        restored.on_signal(_signal(size=0.2, at=NOW + timedelta(minutes=1)))
        assert restored.ledger.position_quantity("AAPL") == pytest.approx(399.96, rel=1e-3)

    def test_the_written_state_is_readable_json(self, tmp_path):
        session = _session()
        session.on_signal(_signal())
        path = session.save_state(tmp_path / "state.json")
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == PAPER_STATE_SCHEMA_VERSION
        assert payload["ledger"]["events"]


# ---------------------------------------------------------------------------
# Corrupt / inconsistent state must be refused, not silently restored
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInconsistentStateIsRefused:
    """A state a session could not have reached must fail, not resume.

    The recovery format's whole promise is that a corrupt file is rejected rather
    than turned into a different portfolio. These are the ways a state can be
    internally impossible while still being well-formed JSON, and each must raise
    ``PaperStateError``.
    """

    def _state(self):
        session = _session(costs=CostModel(commission_rate=0.001))
        session.on_signal(_signal(size=0.2))
        return session.state()

    def _restore(self, state):
        return PaperSession.restore(
            state, quotes=dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        )

    def test_negative_cash_is_refused(self):
        state = self._state()
        state["ledger"]["cash"] = -5000.0
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_non_finite_cash_is_refused(self):
        state = self._state()
        state["ledger"]["cash"] = float("nan")
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_negative_holding_is_refused(self):
        state = self._state()
        for holding in state["ledger"]["holdings"].values():
            holding["quantity"] = -42.0
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_non_finite_holding_quantity_is_refused(self):
        state = self._state()
        for holding in state["ledger"]["holdings"].values():
            holding["quantity"] = float("inf")
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_negative_fees_are_refused(self):
        state = self._state()
        for holding in state["ledger"]["holdings"].values():
            holding["fees_paid"] = -10.0
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_a_rolled_back_sequence_counter_is_refused(self):
        """A seq below the log length would mint duplicate sequence numbers."""
        state = self._state()
        state["ledger"]["seq"] = 1
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_duplicate_event_sequence_numbers_are_refused(self):
        state = self._state()
        events = state["ledger"]["events"]
        assert len(events) >= 2
        events[1]["seq"] = events[0]["seq"]
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_non_dense_event_sequence_is_refused(self):
        state = self._state()
        for offset, event in enumerate(state["ledger"]["events"]):
            event["seq"] = (offset + 1) * 2  # 2, 4, 6 ... not dense
        state["ledger"]["seq"] = 2 * len(state["ledger"]["events"])
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_a_filled_order_with_remaining_quantity_is_refused(self):
        state = self._state()
        state["resting"] = [{
            "order_id": "x-paper", "symbol": "AAPL", "side": "BUY", "quantity": 100.0,
            "order_type": "LIMIT", "limit_price": 95.0, "created_at": NOW.isoformat(),
            "as_of_date": "2026-01-05", "time_in_force": "GTC", "status": "FILLED",
            "filled_quantity": 40.0, "average_fill_price": 95.0,
        }]
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_an_accepted_order_with_fill_data_is_refused(self):
        state = self._state()
        state["resting"] = [{
            "order_id": "y-paper", "symbol": "AAPL", "side": "BUY", "quantity": 100.0,
            "order_type": "LIMIT", "limit_price": 95.0, "created_at": NOW.isoformat(),
            "as_of_date": "2026-01-05", "time_in_force": "GTC", "status": "ACCEPTED",
            "filled_quantity": 40.0, "average_fill_price": 95.0,
        }]
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_a_partially_filled_order_with_zero_filled_is_refused(self):
        state = self._state()
        state["resting"] = [{
            "order_id": "z-paper", "symbol": "AAPL", "side": "BUY", "quantity": 100.0,
            "order_type": "LIMIT", "limit_price": 95.0, "created_at": NOW.isoformat(),
            "as_of_date": "2026-01-05", "time_in_force": "GTC",
            "status": "PARTIALLY_FILLED", "filled_quantity": 0.0,
        }]
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_a_negative_resting_quantity_is_refused(self):
        state = self._state()
        state["resting"] = [{
            "order_id": "n-paper", "symbol": "AAPL", "side": "BUY", "quantity": -5.0,
            "order_type": "LIMIT", "limit_price": 95.0, "created_at": NOW.isoformat(),
            "as_of_date": "2026-01-05", "time_in_force": "GTC", "status": "ACCEPTED",
        }]
        with pytest.raises(PaperStateError):
            self._restore(state)

    def test_a_valid_state_still_restores(self):
        """The validation must not reject a state the session actually produced."""
        restored = self._restore(self._state())
        assert restored.ledger.cash > 0
        assert restored.ledger.position_quantity("AAPL") > 0


# ---------------------------------------------------------------------------
# The per-session liquidity budget is shared, exactly as in a backtest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionLiquidityBudgetIsShared:
    """Two orders for one symbol share one session's volume, as in a backtest."""

    def _two_resting(self, *, participation=0.1, volume=4000.0):
        # A price above the limit so neither order fills when it is created; both
        # rest, and are worked together on the next session.
        session = _session(
            cash=10_000_000.0, quotes={"AAPL": 99.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            participation=participation,
        )
        session.on_signal(_limit(size=0.30, limit_price=95.0))
        session.on_signal(_limit(size=0.31, limit_price=95.0))
        assert len(session.resting) == 2
        # Now the market trades through the limit, with a thin session volume.
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW + timedelta(days=1),
            average_daily_volume=volume,
        )
        return session

    def test_two_resting_orders_share_one_session_budget(self):
        session = self._two_resting()
        before = session.ledger.position_quantity("AAPL")
        session.work_resting()
        filled = session.ledger.position_quantity("AAPL") - before
        # 10% of a 4,000-volume session is 400 units for the symbol, shared by
        # both orders, so the session fills 400 in total, not 800.
        assert filled == pytest.approx(400.0)

    def test_a_shared_budget_is_consumed_in_queue_order(self):
        session = self._two_resting()
        steps = session.work_resting()
        # The first order takes the whole budget; the second gets nothing this
        # session, so the allocation is deterministic. Both remain live: the first
        # as a partial fill, the second still accepted with its whole size left.
        assert steps[0].quantity == pytest.approx(400.0)
        assert steps[1].quantity == pytest.approx(0.0)
        assert [o.symbol for o in session.resting] == ["AAPL", "AAPL"]
        assert session.resting[1].filled_quantity == pytest.approx(0.0)

    def test_participation_off_means_no_liquidity_limit(self):
        session = self._two_resting(participation=0.0)
        before = session.ledger.position_quantity("AAPL")
        session.work_resting()
        # Both orders fill their whole size, because no budget was configured.
        assert session.ledger.position_quantity("AAPL") - before > 400.0


# ---------------------------------------------------------------------------
# The session clock never runs backwards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionClockMonotonicity:
    def test_an_earlier_quote_does_not_move_the_day_backwards(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0, at=NOW))
        day = session.day
        # A quote stamped in the past is stale, not a new session.
        session.quotes = dict_quote_source(
            {"AAPL": 94.0}, timestamp=NOW - timedelta(days=5)
        )
        session.work_resting()
        assert session.day == day

    def test_the_daily_loss_baseline_is_not_reset_by_a_stale_quote(self):
        session = _session()
        session.on_signal(_limit(limit_price=95.0, at=NOW))
        baseline = session.day_start_equity
        session.quotes = dict_quote_source(
            {"AAPL": 94.0}, timestamp=NOW - timedelta(days=5)
        )
        session.work_resting()
        assert session.day_start_equity == baseline


# ---------------------------------------------------------------------------
# Backtest and Paper agree on a shared, liquidity-limited session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBacktestPaperParity:
    """The same orders against the same thin session must fill the same size.

    The two engines share the simulator and the ledger, so they agree by
    construction *except* where the caller's per-session liquidity accounting
    differs. This pins the one place that could diverge: two orders competing for
    one symbol's volume on one session.
    """

    def test_competing_limits_fill_the_same_size_in_both_engines(self):
        import pandas as pd

        from tradingagents.backtesting import (
            BacktestConfig,
            MarketDataFeed,
            PortfolioBacktester,
        )

        dates = ["2026-01-05", "2026-01-06", "2026-01-07"]

        def feed():
            frame = pd.DataFrame(
                {
                    "open": [100.0, 100.0, 94.0],
                    "high": [100.0, 100.0, 94.0],
                    "low": [100.0, 100.0, 94.0],
                    "close": [100.0, 100.0, 94.0],
                    "volume": [1_000_000, 1_000_000, 4000],
                },
                index=pd.to_datetime(dates),
            )
            return MarketDataFeed({"AAPL": frame})

        def signal(session, size):
            return TradingSignal(
                symbol="AAPL", action=SignalAction.BUY, confidence=0.8,
                timestamp=datetime.strptime(session, "%Y-%m-%d").replace(hour=21, tzinfo=UTC),
                as_of_date=session, position_size=size, order_intent=OrderIntent.LIMIT,
                limit_price=95.0, entry_price=100.0, time_in_force=SignalTimeInForce.GTC,
            )

        def strategy(context):
            if context.session == dates[0]:
                return signal(context.session, 0.30)
            if context.session == dates[1]:
                return signal(context.session, 0.31)
            return None

        config = BacktestConfig(
            max_volume_participation=0.1, resting_limit_orders=True,
            initial_cash=10_000_000.0,
            risk_limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
        )
        backtester = PortfolioBacktester(feed(), strategy, config=config)
        backtester.run()

        # The equivalent paper run: two resting limits, worked together on the
        # thin session that reaches the limit.
        session = _session(
            cash=10_000_000.0, quotes={"AAPL": 99.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            participation=0.1,
        )
        session.on_signal(_limit(size=0.30, limit_price=95.0))
        session.on_signal(_limit(size=0.31, limit_price=95.0))
        session.quotes = lambda symbol: Quote(
            symbol="AAPL", price=94.0, timestamp=NOW + timedelta(days=1),
            average_daily_volume=4000.0,
        )
        session.work_resting()

        assert backtester.ledger.position_quantity("AAPL") == pytest.approx(
            session.ledger.position_quantity("AAPL")
        )
        # 10% of 4,000 units is 400 for the symbol, shared by both orders.
        assert session.ledger.position_quantity("AAPL") == pytest.approx(400.0)
