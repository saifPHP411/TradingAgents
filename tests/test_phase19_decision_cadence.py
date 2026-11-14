"""Phase 19: a refreshed session that advances without a decision must say why.

The reported symptom was that a Railway data refresh *succeeded* — the logs showed
``[auto] advanced 2 session(s): 2026-10-29..2026-10-30`` and ``[data] ... status=
appended new=2`` — yet no new decision appeared. The advance was real; the sessions
were processed. What the logs could not show is *why* neither session produced a
decision, because a processed session carried no reason and the report carried no
cadence block.

The cause is the decision cadence, not a lost decision. ``_is_decision_session``
returns True only every ``decision_frequency_sessions`` sessions, measured from the
source's *first* session, and returns False once ``max_decisions`` is reached. A run
therefore advances many sessions that settle the book and work resting orders but
never decide. That is correct — a decision on every bar would be a different
experiment — but it was silent.

These tests pin the diagnosability fix and the invariants it must not break:

- a non-cadence session is processed and *named* as such, in the outcome, the
  journal and the report's cadence block;
- a cadence session still decides, so the naming never suppresses a real decision;
- the reached cap is named distinctly from the cadence gap, because the two need
  different operator responses;
- the cadence is anchored to the source's first session, so it does not drift when
  the vendor's sliding ``compact`` window moves its earliest bar forward — the
  failure mode that would silently shift which sessions decide;
- ``next_decision_session`` points at a session that will actually decide, so the
  field cannot claim a decision is coming when the cap has been reached.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.experiment import (
    DECISION_SKIP_REASONS,
    SKIP_MAX_DECISIONS_REACHED,
    SKIP_NOT_DECISION_SESSION,
    PaperExperiment,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.journal import KIND_SESSION_PROCESSED
from tradingagents.paper_experiment.refresh import (
    STATUS_APPENDED,
    RefreshPolicy,
    refresh_source,
)
from tradingagents.paper_experiment.report import build_report
from tradingagents.paper_experiment.runner import advance, process_pending
from tradingagents.paper_experiment.sources import StaticPriceSource

SYMBOL = "SPY"
NO_THROTTLE = RefreshPolicy(min_interval_seconds=0.0, max_fetches_per_day=24)


def _frame(sessions, *, base: float = 600.0):
    index = pd.to_datetime(list(sessions))
    epoch = pd.Timestamp("2026-01-01")
    opens = [base + float((ts - epoch).days) for ts in index]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * len(sessions),
        },
        index=index,
    )


def _vendor_source(sessions, *, name: str = "alpha_vantage_daily"):
    source = StaticPriceSource({SYMBOL: _frame(sessions)}, name=name)
    source.kind = "vendor_daily"
    return source


def _sessions(start: str, end: str):
    return tuple(str(d)[:10] for d in pd.bdate_range(start, end))


def _experiment(tmp_path, sessions, *, cadence: int = 5, max_decisions: int = 120):
    config = ExperimentConfig.from_protocol(
        symbols=(SYMBOL,),
        decision_frequency_sessions=cadence,
        max_decisions=max_decisions,
        evaluation_lag_sessions=2,
    )
    experiment = PaperExperiment(
        source=_vendor_source(sessions),
        config=config,
        root=tmp_path / "exp",
        inference=StubInference(rating="Buy"),
    )
    experiment.ensure_created()
    experiment.save_state()
    return experiment


def _fetcher(sessions):
    source = _vendor_source(sessions)

    def fetch(experiment, *, today=None):
        return source

    return fetch


@pytest.mark.unit
class TestAnAdvancedSessionWithoutADecisionIsNamed:
    def test_a_non_cadence_session_is_processed_and_named(self, tmp_path):
        """The reported shape: the session advances, no decision, and it says why."""
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions)
        # Process every session; index 0 and 5 are cadence, the rest are not.
        outcomes = {o.session: o for o in (experiment.process_session(s) for s in sessions)}

        assert outcomes[sessions[1]].processed is True
        assert outcomes[sessions[1]].decision_id is None
        assert outcomes[sessions[1]].decision_skip_reason == SKIP_NOT_DECISION_SESSION
        # The premise: a cadence session in the same run still decides, so the
        # naming did not suppress it.
        assert outcomes[sessions[0]].decision_id == f"{SYMBOL}-{sessions[0]}"
        assert outcomes[sessions[0]].decision_skip_reason is None

    def test_the_skip_reason_is_in_the_closed_vocabulary(self, tmp_path):
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions)
        for session in sessions:
            outcome = experiment.process_session(session)
            if outcome.decision_skip_reason is not None:
                assert outcome.decision_skip_reason in DECISION_SKIP_REASONS

    def test_the_journal_records_the_skip_reason(self, tmp_path):
        """A reader of the journal alone can tell a skipped session from a decided one."""
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions)
        for session in sessions:
            experiment.process_session(session)

        entries = [
            e for e in experiment.journal.entries if e.kind == KIND_SESSION_PROCESSED
        ]
        by_session = {e.payload["session"]: e.payload for e in entries}
        assert by_session[sessions[1]]["decision_skip_reason"] == SKIP_NOT_DECISION_SESSION
        assert by_session[sessions[0]]["decision_skip_reason"] is None

    def test_a_refresh_then_advance_names_the_non_cadence_sessions(self, tmp_path):
        """The exact reported sequence: fetch two new sessions, advance, no decision."""
        initial = _sessions("2026-10-01", "2026-10-28")
        grown = _sessions("2026-10-01", "2026-10-30")
        experiment = _experiment(tmp_path, initial)
        process_pending(experiment)

        result = advance(experiment, refresh_policy=NO_THROTTLE, fetcher=_fetcher(grown))

        assert result.refresh.status == STATUS_APPENDED
        assert result.refresh.new_sessions == ("2026-10-29", "2026-10-30")
        advanced = {o.session: o for o in result.outcomes}
        # 2026-10-29 is index 20 (cadence), 2026-10-30 is index 21 (not).
        assert advanced["2026-10-29"].decision_id == f"{SYMBOL}-2026-10-29"
        assert advanced["2026-10-30"].decision_skip_reason == SKIP_NOT_DECISION_SESSION


@pytest.mark.unit
class TestTheReachedCapIsNamedDistinctly:
    def test_a_session_after_the_cap_names_the_cap_not_the_cadence(self, tmp_path):
        """The two reasons need different operator responses, so they are distinct."""
        sessions = _sessions("2026-10-01", "2026-11-30")
        # cadence 1 so every session is a cadence session; cap at 3.
        experiment = _experiment(tmp_path, sessions, cadence=1, max_decisions=3)
        outcomes = [experiment.process_session(s) for s in sessions]

        decided = [o for o in outcomes if o.decision_id]
        capped = [o for o in outcomes if o.decision_skip_reason == SKIP_MAX_DECISIONS_REACHED]
        assert len(decided) == 3
        assert capped, "sessions past the cap must be processed and named"
        # A capped session is never mislabelled as merely off-cadence.
        assert all(o.decision_skip_reason != SKIP_NOT_DECISION_SESSION for o in capped)


@pytest.mark.unit
class TestTheCadenceBlockExplainsTheState:
    def test_the_report_carries_the_cadence_block(self, tmp_path):
        initial = _sessions("2026-10-01", "2026-10-28")
        grown = _sessions("2026-10-01", "2026-10-30")
        experiment = _experiment(tmp_path, initial)
        process_pending(experiment)
        advance(experiment, refresh_policy=NO_THROTTLE, fetcher=_fetcher(grown))

        cadence = build_report(experiment).to_dict()["cadence"]
        assert cadence["decision_frequency_sessions"] == 5
        assert cadence["max_decisions"] == 120
        assert cadence["max_decisions_reached"] is False
        assert cadence["decisions_taken"] == 5
        assert cadence["next_decision_session"] is None  # no cadence bar remains

    def test_next_decision_session_is_one_that_will_actually_decide(self, tmp_path):
        """It must point at a future cadence bar, never at one the cap forbids."""
        # 12 sessions, cadence 5: cadence bars are index 0, 5, 10.
        sessions = _sessions("2026-10-01", "2026-10-16")
        experiment = _experiment(tmp_path, sessions)
        # Process only through index 3, so index 5 is still due.
        for session in sessions[:4]:
            experiment.process_session(session)

        cadence = experiment.decision_cadence()
        assert cadence["next_decision_session"] == sessions[5]
        # And it really is a cadence session, and not capped.
        assert sessions.index(cadence["next_decision_session"]) % 5 == 0
        assert cadence["max_decisions_reached"] is False

    def test_a_reached_cap_reports_no_next_decision_session(self, tmp_path):
        sessions = _sessions("2026-10-01", "2026-11-30")
        experiment = _experiment(tmp_path, sessions, cadence=1, max_decisions=3)
        process_pending(experiment)

        cadence = experiment.decision_cadence()
        assert cadence["max_decisions_reached"] is True
        assert cadence["next_decision_session"] is None


@pytest.mark.unit
class TestTheCadenceDoesNotDriftWithASlidingWindow:
    def test_the_anchor_survives_a_window_that_moves_its_earliest_bar(self, tmp_path):
        """Alpha Vantage's ``compact`` window slides; the anchor must not.

        If the anchor moved with the window, which sessions decide would shift
        silently under a running experiment — the reported "no decision" would
        then be a real defect rather than a cadence gap.
        """
        anchor_window = _sessions("2026-01-23", "2026-06-11")  # 100 bars
        slid_window = _sessions("2026-06-15", "2026-10-30")  # 100 bars, earliest moved
        experiment = _experiment(tmp_path, anchor_window)
        process_pending(experiment)
        assert experiment.source.sessions()[0] == anchor_window[0]

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(slid_window)
        )

        assert outcome.status == STATUS_APPENDED
        # The anchor is preserved, so the cadence is unchanged.
        assert experiment.source.sessions()[0] == anchor_window[0]
        cadence = experiment.decision_cadence()
        assert cadence["decision_frequency_sessions"] == 5
        # Every decision taken sits on the original anchor's cadence.
        sessions = experiment.source.sessions()
        for decision in experiment.decisions.values():
            assert sessions.index(decision.decision_session) % 5 == 0

    def test_a_resumed_run_computes_the_same_cadence_as_a_fresh_one(self, tmp_path):
        """The check is a function of the session, not of how many were processed."""
        sessions = _sessions("2026-10-01", "2026-10-16")
        straight = _experiment(tmp_path / "straight", sessions)
        process_pending(straight)

        resumed = _experiment(tmp_path / "resumed", sessions)
        for session in sessions[:3]:
            resumed.process_session(session)
        resumed = PaperExperiment.load(
            tmp_path / "resumed" / "exp", inference=StubInference(rating="Buy")
        )
        for session in sessions[3:]:
            resumed.process_session(session)

        assert set(straight.decisions) == set(resumed.decisions)
        assert straight.decision_cadence() == resumed.decision_cadence()


@pytest.mark.unit
class TestTheAutoAdvanceLogNamesTheDecisionOutcome:
    def test_an_advanced_tick_reports_its_decision_count(self, tmp_path):
        """The reported log line must say how many decisions the tick produced.

        ``[auto] advanced 2 session(s)`` alone reads as progress with no decision.
        The line now carries the decision count and, when it is zero, the reason.
        """
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        initial = _sessions("2026-10-01", "2026-10-28")
        grown = _sessions("2026-10-01", "2026-10-30")
        experiment = _experiment(tmp_path, initial)
        process_pending(experiment)

        dashboard = Dashboard(
            tmp_path / "exp",
            allow_control=True,
            stub=True,
            refresh_policy=NO_THROTTLE,
            refresh_fetcher=_fetcher(grown),
        )
        result = dashboard.run_sessions(100)

        assert result["processed"] == ["2026-10-29", "2026-10-30"]
        assert result["decision_count"] == 1  # only 2026-10-29 is a cadence session
        assert result["no_decision_reasons"] == [SKIP_NOT_DECISION_SESSION]

    def test_a_tick_with_no_decision_names_the_cadence(self, tmp_path):
        """A tick that advances only off-cadence sessions must not look like progress."""
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        # 8 sessions from an anchor; advance a single off-cadence bar (index 1).
        initial = _sessions("2026-10-01", "2026-10-02")
        grown = _sessions("2026-10-01", "2026-10-05")
        experiment = _experiment(tmp_path, initial)
        process_pending(experiment)

        dashboard = Dashboard(
            tmp_path / "exp",
            allow_control=True,
            stub=True,
            refresh_policy=NO_THROTTLE,
            refresh_fetcher=_fetcher(grown),
        )
        result = dashboard.run_sessions(100)

        assert result["decision_count"] == 0
        assert result["no_decision_reasons"] == [SKIP_NOT_DECISION_SESSION]
        line = result["refresh"]["summary"]
        assert "[data]" in line
