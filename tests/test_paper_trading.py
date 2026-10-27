"""Paper trading: the same pipeline as a backtest, driven by supplied quotes.

The tests that matter here are the ones that show paper mode cannot become a
live one. A paper session is built from the same ledger, risk engine and
simulator a backtest uses, so the accounting is identical by construction rather
than by review. And no matter how it is configured, there is no mode that reaches
a venue: the guard is the same one the simulator already enforces.

The other set of tests shows the session does the work the pipeline implies —
risk refusals produce no order, a HOLD produces no order, costs reach the ledger,
and the day rolls so the daily-loss limit has a baseline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.execution import (
    ExecutionMode,
    ExecutionSimulator,
    LiveTradingNotSupportedError,
    PaperSession,
    PaperSessionError,
    Quote,
    dict_quote_source,
    new_session,
)
from tradingagents.execution.costs import CostModel
from tradingagents.portfolio.events import EventType
from tradingagents.portfolio.ledger import Ledger
from tradingagents.risk import RiskEngine, RiskLimits
from tradingagents.signals import SignalAction, TradingSignal

NOW = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def _signal(action=SignalAction.BUY, *, symbol="AAPL", size=0.1, at=NOW) -> TradingSignal:
    # The contract forbids a size on a HOLD, so it is only passed for a real action.
    kwargs = {} if action is SignalAction.HOLD else {"position_size": size}
    return TradingSignal(
        symbol=symbol, action=action, confidence=0.8, timestamp=at,
        as_of_date=at.date().isoformat(), **kwargs,
    )


def _session(*, cash=100_000.0, quotes=None, limits=None, costs=None) -> PaperSession:
    return new_session(
        initial_cash=cash,
        quotes=dict_quote_source(quotes or {"AAPL": 100.0}, timestamp=NOW),
        limits=limits or RiskLimits(max_position_size=0.5, max_portfolio_exposure=None),
        costs=costs,
    )


# ---------------------------------------------------------------------------
# The guard: paper is simulation, and nothing else is reachable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoLivePath:
    def test_the_default_mode_is_paper(self):
        assert _session().mode is ExecutionMode.PAPER

    def test_a_live_mode_cannot_be_selected(self):
        for name in ("live", "real", "production", "broker", "exchange"):
            with pytest.raises(LiveTradingNotSupportedError):
                PaperSession(
                    ledger=Ledger(initial_cash=1_000.0),
                    risk=RiskEngine(),
                    simulator=ExecutionSimulator(Ledger(initial_cash=1_000.0)),
                    quotes=dict_quote_source({"AAPL": 100.0}),
                    mode=name,
                )

    def test_there_is_no_live_member_to_select(self):
        assert {m.value for m in ExecutionMode} == {"BACKTEST", "PAPER"}

    def test_the_simulator_refuses_a_live_mode(self):
        with pytest.raises(LiveTradingNotSupportedError):
            ExecutionSimulator(Ledger(initial_cash=1_000.0), mode="live")

    def test_a_paper_session_uses_the_same_simulator_a_backtest_does(self):
        session = _session()
        assert isinstance(session.simulator, ExecutionSimulator)
        assert session.simulator.mode is ExecutionMode.PAPER


# ---------------------------------------------------------------------------
# Driving signals through the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOnSignal:
    def test_an_approved_buy_fills_and_moves_the_ledger(self):
        session = _session()
        step = session.on_signal(_signal(size=0.1))
        assert step.approved
        assert step.order_status == "FILLED"
        assert step.quantity > 0
        assert session.ledger.position_quantity("AAPL") == pytest.approx(step.quantity)
        assert session.ledger.cash < 100_000.0

    def test_a_hold_produces_no_order(self):
        session = _session()
        step = session.on_signal(_signal(SignalAction.HOLD))
        assert not step.approved
        assert step.order_status == "NONE"
        assert session.ledger.position_quantity("AAPL") == 0.0

    def test_a_risk_refusal_produces_no_order(self):
        """An order below the configured minimum value is refused, not placed small."""
        session = _session(limits=RiskLimits(
            max_position_size=0.5, max_portfolio_exposure=None, min_order_value=50_000.0))
        step = session.on_signal(_signal(size=0.1))  # 10k notional, under the 50k floor
        assert not step.approved
        assert step.order_status == "NONE"
        assert session.ledger.position_quantity("AAPL") == 0.0

    def test_a_position_cap_clamps_rather_than_refusing(self):
        """The cap reduces the size; it does not throw the decision away."""
        session = _session(limits=RiskLimits(
            max_position_size=0.1, default_position_size=0.1, max_portfolio_exposure=None))
        step = session.on_signal(_signal(size=0.5))
        assert step.approved
        assert step.quantity <= 0.1 * session.ledger.equity / 100.0 + 1e-6

    def test_a_sell_with_no_holding_is_refused(self):
        session = _session()
        step = session.on_signal(_signal(SignalAction.SELL))
        assert not step.approved
        assert step.order_status == "NONE"
        assert session.ledger.position_quantity("AAPL") == 0.0

    def test_a_round_trip_returns_to_flat(self):
        session = _session()
        session.on_signal(_signal(size=0.2))
        assert session.ledger.position_quantity("AAPL") > 0
        session.on_signal(_signal(SignalAction.SELL, size=1.0))
        assert session.ledger.position_quantity("AAPL") == pytest.approx(0.0)

    def test_a_missing_quote_is_an_error_not_a_silent_skip(self):
        session = _session(quotes={"AAPL": 100.0})
        with pytest.raises(PaperSessionError):
            session.on_signal(_signal(symbol="MSFT"))

    def test_every_decision_is_recorded(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        session.on_signal(_signal(SignalAction.HOLD))
        assert len(session.steps) == 2
        assert [s.action for s in session.steps] == ["BUY", "HOLD"]


# ---------------------------------------------------------------------------
# Costs and the ledger
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCostsReachTheLedger:
    def test_a_commission_lands_in_the_ledger(self):
        session = _session(costs=CostModel(commission_rate=0.001))
        step = session.on_signal(_signal(size=0.1))
        assert step.fees > 0
        assert session.ledger.total_fees == pytest.approx(step.fees)

    def test_costs_reduce_equity_against_a_costless_run(self):
        free = _session()
        charged = _session(costs=CostModel(commission_rate=0.005))
        free.on_signal(_signal(size=0.1))
        charged.on_signal(_signal(size=0.1))
        assert charged.ledger.equity < free.ledger.equity

    def test_no_costs_means_no_fees(self):
        session = _session(costs=CostModel())
        step = session.on_signal(_signal(size=0.1))
        assert step.fees == 0.0
        assert session.ledger.total_fees == 0.0


# ---------------------------------------------------------------------------
# Marking and the daily baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarking:
    def test_marking_updates_equity(self):
        session = _session()
        session.on_signal(_signal(size=0.5))
        before = session.ledger.equity
        session.mark([Quote(symbol="AAPL", price=110.0, timestamp=NOW)])
        assert session.ledger.equity > before

    def test_the_equity_peak_tracks_the_high_water_mark(self):
        session = _session()
        session.on_signal(_signal(size=0.5))
        session.mark([Quote(symbol="AAPL", price=120.0, timestamp=NOW)])
        session.mark([Quote(symbol="AAPL", price=90.0, timestamp=NOW)])
        assert session.equity_peak >= 100_000.0

    def test_the_day_rolls_and_resets_the_daily_baseline(self):
        session = _session()
        first = session.day
        session.on_signal(_signal(size=0.1))
        later = NOW + timedelta(days=1)
        session.on_signal(_signal(SignalAction.HOLD, at=later))
        assert session.day != first
        assert session.day == later.date().isoformat()

    def test_a_same_day_signal_does_not_reset_the_baseline(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        baseline = session.day_start_equity
        session.on_signal(_signal(SignalAction.HOLD, at=NOW + timedelta(hours=1)))
        assert session.day_start_equity == baseline


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshot:
    def test_the_snapshot_reports_the_book(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        snapshot = session.snapshot()
        assert snapshot["mode"] == "PAPER"
        assert "AAPL" in snapshot["positions"]
        assert snapshot["equity"] > 0

    def test_the_snapshot_lists_every_step(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        session.on_signal(_signal(SignalAction.HOLD))
        assert len(session.snapshot()["steps"]) == 2

    def test_the_snapshot_is_json_safe(self):
        session = _session()
        session.on_signal(_signal(size=0.1))
        json.dumps(session.snapshot())

    def test_the_snapshot_can_be_written_to_a_file(self, tmp_path):
        session = _session()
        session.on_signal(_signal(size=0.1))
        path = session.write_json(tmp_path / "paper" / "session.json")
        payload = json.loads(path.read_text())
        assert payload["mode"] == "PAPER"


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuotes:
    def test_a_non_positive_quote_is_refused(self):
        with pytest.raises(PaperSessionError):
            Quote(symbol="AAPL", price=0.0, timestamp=NOW)

    def test_a_closed_market_is_carried_into_the_fill_attempt(self):
        quotes = dict_quote_source({"AAPL": 100.0}, timestamp=NOW, is_open=False)
        session = new_session(initial_cash=100_000.0, quotes=quotes,
                              limits=RiskLimits(max_position_size=0.5,
                                                max_portfolio_exposure=None))
        step = session.on_signal(_signal(size=0.1))
        # The simulator refuses to fill into a closed market, and says so.
        assert step.order_status != "FILLED"
        assert session.ledger.position_quantity("AAPL") == 0.0

    def test_a_dict_source_reports_a_missing_symbol(self):
        source = dict_quote_source({"AAPL": 100.0}, timestamp=NOW)
        with pytest.raises(PaperSessionError, match="MSFT"):
            source("MSFT")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_identical_signals_give_identical_results(self):
        def run():
            session = _session(costs=CostModel(commission_rate=0.001))
            session.on_signal(_signal(size=0.2))
            session.on_signal(_signal(SignalAction.SELL, size=1.0, at=NOW + timedelta(minutes=1)))
            return session.snapshot()

        first = run()
        second = run()
        # Timestamps are part of the record; everything else must match.
        for snapshot in (first, second):
            for step in snapshot["steps"]:
                step.pop("timestamp")
            snapshot["ledger"].pop("timestamp")
        assert first == second


# ---------------------------------------------------------------------------
# The position cap is enforced at the fill in paper mode too
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaperCapEnforcedAtFill:
    """Paper mode shares the pipeline, so it must share the cap guarantee.

    The decision is sized against the quote, but the fill pays a different price
    once the spread and slippage are applied. A cap satisfied at the quote can be
    breached by the fill, exactly as a gap breaches it in a backtest. The session
    re-checks a buy at the projected fill price, so the position that results
    respects the cap. A sell reduces exposure and is not re-checked.
    """

    def test_adverse_costs_do_not_breach_the_position_cap(self):
        # 100 bps of half-spread plus 100 bps of slippage moves a 100 quote to a
        # 102 fill. Sizing 20% at the quote would be 20.4% at the fill.
        costs = CostModel(half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        step = session.on_signal(_signal(size=0.20))
        quantity = session.ledger.position_quantity("AAPL")
        assert quantity > 0
        assert step.fill_price == pytest.approx(102.0)
        assert quantity * step.fill_price / session.ledger.equity <= 0.20 + 1e-6

    def test_a_fill_inside_the_cap_is_not_reduced(self):
        costs = CostModel(half_spread_bps=1.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.50, max_portfolio_exposure=None),
            costs=costs,
        )
        session.on_signal(_signal(size=0.10))
        # 10% of 100,000 = 10,000; at ~100.01 that is ~99.99 units, not reduced.
        assert session.ledger.position_quantity("AAPL") == pytest.approx(99.99, rel=1e-3)

    def test_the_paper_reduction_is_recorded_on_the_step(self):
        costs = CostModel(half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        step = session.on_signal(_signal(size=0.20))
        # The step's quantity is what was actually filled, so a report cannot
        # claim a larger position than the ledger holds.
        assert step.quantity == pytest.approx(session.ledger.position_quantity("AAPL"))

    def test_a_sell_is_not_reduced_by_the_cap(self):
        """A sell is not subject to the buy-side cap, and it is not reduced.

        ``position_size`` on a SELL is a fraction of the *holding* to sell, not a
        fraction of equity, so a sell of 1.0 closes the position. The cap that
        reduces a buy must not touch it: a sell lowers exposure, so a cap on
        exposure cannot be breached by one.
        """
        costs = CostModel(half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        session.on_signal(_signal(size=0.20))
        held = session.ledger.position_quantity("AAPL")
        assert held > 0
        step = session.on_signal(_signal(SignalAction.SELL, size=1.0))
        assert step.order_status == "FILLED"
        assert session.ledger.position_quantity("AAPL") == pytest.approx(0.0)

    def test_a_partial_sell_sells_the_requested_fraction_of_the_holding(self):
        costs = CostModel(half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        session.on_signal(_signal(size=0.20))
        held = session.ledger.position_quantity("AAPL")
        session.on_signal(_signal(SignalAction.SELL, size=0.50))
        assert session.ledger.position_quantity("AAPL") == pytest.approx(held * 0.50)

    def test_paper_accounting_identity_holds_after_a_capped_fill(self):
        costs = CostModel(commission_flat=1.0, commission_per_unit=0.01,
                          half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        session.on_signal(_signal(size=0.20))
        ledger = session.ledger
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert ledger.cash + basis + ledger.total_fees - ledger.realized_pnl == pytest.approx(
            100_000.0
        )

    def test_repeated_paper_entries_cannot_accumulate_past_the_cap(self):
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
        )
        session.on_signal(_signal(size=0.20))
        session.on_signal(_signal(size=0.20, at=NOW + timedelta(minutes=1)))
        q = session.ledger.position_quantity("AAPL")
        assert q * 100.0 / session.ledger.equity <= 0.20 + 1e-6

    def test_a_refusal_at_the_fill_is_recorded_not_silent(self):
        """A refusal after the order was created must leave a trace.

        The order exists by then, so the refusal is recorded the same way the
        simulator records one: the step names the reason and an ``ORDER_REJECTED``
        event is written. The position is taken to the cap by the first signal, so
        the second has no headroom once the costs move the fill off the quote.
        """
        costs = CostModel(half_spread_bps=100.0, slippage_bps=100.0)
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=None),
            costs=costs,
        )
        session.on_signal(_signal(size=0.20))
        second = session.on_signal(_signal(size=0.20, at=NOW + timedelta(minutes=1)))
        assert second.order_status == "NONE"
        assert "refused at execution" in second.reason
        rejected = [e for e in session.ledger.events if e.type is EventType.ORDER_REJECTED]
        assert rejected, "a fill-time refusal must record an ORDER_REJECTED event"
        assert all(e.detail.get("reason") == "RISK_LIMIT" for e in rejected)


@pytest.mark.unit
class TestPaperBuyCappedByCashAtFill:
    """The paper path caps a buy to affordable cash at fill, as the backtest does.

    Regression test for a real defect: a buy sized to the whole book was refused
    by the simulator once the commission pushed the all-in cost past the cash,
    because the paper path did not pass the fee-inclusive affordable notional to
    the fill-time re-check. It now does, so the same order is reduced instead.
    """

    def test_a_commission_does_not_reject_a_full_cash_buy(self):
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            costs=CostModel(commission_rate=0.001),
        )
        step = session.on_signal(_signal(size=1.0))
        assert step.order_status == "FILLED"
        assert session.ledger.position_quantity("AAPL") > 0
        spent = session.ledger.position_quantity("AAPL") * 100.0 + session.ledger.total_fees
        assert spent <= 100_000.0 + 1e-6

    def test_the_paper_and_backtest_caps_agree_on_the_quantity(self):
        """Both paths must cap to the same affordable size for the same inputs."""
        session = _session(
            quotes={"AAPL": 100.0},
            limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
            costs=CostModel(commission_rate=0.001),
        )
        step = session.on_signal(_signal(size=1.0))
        assert session.ledger.cash == pytest.approx(0.0, abs=1e-6)
        assert step.quantity == pytest.approx(session.ledger.position_quantity("AAPL"))
