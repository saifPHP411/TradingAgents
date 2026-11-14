"""The reported symptom, pinned: a settled decision is not a filled order.

The live experiment reported ``executed: 1`` with ``filled orders: 0`` and read
as if an order had gone missing. It had not. ``counts.executed`` counts
*decisions carried to their execution session* — a HOLD included — and a HOLD
never creates an order. The defect was in the report's vocabulary, not in the
execution path, and these tests fix both halves of that statement:

- the execution path *does* produce simulated fills when a decision is tradeable
  (so a zero is never "fills are broken");
- the report now states the order funnel explicitly, so ``executed`` can never be
  read as ``filled``.

No test here calls a provider or a broker: the inference seam is a deterministic
stub and the price source is a fixed in-memory frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    PaperExperiment,
    StaticPriceSource,
    build_report,
)
from tradingagents.paper_experiment.inference import StubInference

# The live experiment's shape: 100 sessions, a decision every 5, lag 5.
SESSIONS = 100
FREQUENCY = 5
LAG = 5


def _frame(periods: int = SESSIONS, *, start: float = 100.0, step: float = 1.0):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * periods,
        },
        index=index,
    )


def _experiment(tmp_path, *, rating: str, periods: int = SESSIONS):
    source = StaticPriceSource({"SPY": _frame(periods)})
    config = ExperimentConfig.from_protocol(
        symbols=("SPY",),
        decision_frequency_sessions=FREQUENCY,
        max_decisions=120,
        evaluation_lag_sessions=LAG,
    )
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=tmp_path / "exp",
        inference=StubInference(rating=rating),
    )
    experiment.ensure_created()
    for session in source.sessions():
        experiment.process_session(session)
    return experiment, source


@pytest.mark.unit
class TestASettledDecisionIsNotAFilledOrder:
    def test_a_hold_settles_every_decision_but_creates_no_order(self, tmp_path):
        """The exact reported shape: executed present, filled_orders zero."""
        experiment, _ = _experiment(tmp_path, rating="Hold")
        counts = build_report(experiment).to_dict()["counts"]

        assert counts["decisions_total"] == 20
        assert counts["filled_orders"] == 0
        # The last decision's evaluation session lies past the data end, so it is
        # settled (executed) but not yet evaluated. This is the "executed: 1".
        assert counts["executed"] == 1
        assert counts["evaluated"] == 19

    def test_every_settled_hold_is_counted_as_without_order(self, tmp_path):
        experiment, _ = _experiment(tmp_path, rating="Hold")
        orders = build_report(experiment).to_dict()["orders"]

        assert orders["settled"] == 20
        assert orders["submitted"] == 0
        assert orders["filled"] == 0
        assert orders["without_order"] == 20
        assert orders["without_order_reasons"] == {"no order: HOLD": 20}

    def test_a_tradeable_buy_does_produce_a_simulated_fill(self, tmp_path):
        """A zero fill is a HOLD/risk outcome, never a broken execution path."""
        experiment, _ = _experiment(tmp_path, rating="Buy")
        payload = build_report(experiment).to_dict()

        assert payload["orders"]["filled"] > 0
        assert payload["counts"]["filled_orders"] > 0
        assert payload["book"]["positions"]["SPY"]["quantity"] > 0
        assert payload["book"]["cash"] < payload["book"]["initial_cash"]
        assert payload["book"]["total_fees"] > 0

    def test_the_funnel_separates_decision_from_order_outcomes(self, tmp_path):
        """Every stage is a distinct number, and the identity holds."""
        experiment, _ = _experiment(tmp_path, rating="Buy")
        orders = build_report(experiment).to_dict()["orders"]

        # A settled decision either produced an order or did not.
        assert orders["settled"] == orders["submitted"] + orders["without_order"]
        # A submitted order landed in exactly one outcome bucket.
        outcomes = (
            orders["filled"]
            + orders["partially_filled"]
            + orders["rejected"]
            + orders["cancelled"]
            + orders["resting"]
        )
        assert orders["submitted"] == outcomes

    def test_a_hold_is_never_counted_as_a_submitted_order(self, tmp_path):
        experiment, _ = _experiment(tmp_path, rating="Hold")
        orders = build_report(experiment).to_dict()["orders"]
        assert orders["submitted"] == 0
        assert orders["filled"] == 0

    def test_the_sell_without_a_position_is_refused_with_a_reason(self, tmp_path):
        """A SELL with nothing held settles, but creates no order and says why."""
        experiment, _ = _experiment(tmp_path, rating="Sell")
        orders = build_report(experiment).to_dict()["orders"]
        assert orders["submitted"] == 0
        assert orders["without_order"] == 20
        assert any(
            "INSUFFICIENT_POSITION" in reason
            for reason in orders["without_order_reasons"]
        )


@pytest.mark.unit
class TestTheExecutedDecisionIsIdentifiable:
    def test_the_un_evaluated_decision_is_the_last_one(self, tmp_path):
        """The single ``executed`` decision is the tail, not a lost order."""
        experiment, source = _experiment(tmp_path, rating="Hold")
        executed = [d for d in experiment.decisions.values() if d.state == "executed"]
        assert len(executed) == 1
        tail = executed[0]
        # Its evaluation session is one lag past its execution session, and the
        # data ends before it, so it cannot be evaluated yet.
        sessions = source.sessions()
        assert tail.execution_session == sessions[sessions.index(tail.decision_session) + 1]
        assert tail.evaluation_session is None
        assert tail.order_status == "NONE"
        assert tail.order_created is False

    def test_order_reason_is_persisted_on_the_decision(self, tmp_path):
        experiment, _ = _experiment(tmp_path, rating="Hold")
        reasons = {d.order_reason for d in experiment.decisions.values()}
        assert reasons == {"no order: HOLD"}

    def test_the_funnel_survives_a_state_round_trip(self, tmp_path):
        """A restored experiment reports the same funnel as the original."""
        experiment, _ = _experiment(tmp_path, rating="Buy")
        before = build_report(experiment).to_dict()["orders"]

        restored = PaperExperiment.load(tmp_path / "exp")
        after = build_report(restored).to_dict()["orders"]
        assert after == before

    def test_the_report_declares_the_executed_counter_is_not_fills(self, tmp_path):
        experiment, _ = _experiment(tmp_path, rating="Hold")
        payload = build_report(experiment).to_dict()
        # The report carries its own note so the two numbers cannot be conflated.
        assert "counts.executed" in payload["orders"]["note"]
        assert payload["schema_version"] == "1.1.0"
