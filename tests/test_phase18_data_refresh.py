"""Phase 18: growing a running experiment with new vendor sessions, safely.

The reported symptom was an auto-advance loop that printed, forever,
``[auto] no unprocessed sessions remain; nothing to advance``. The cause is that a
vendor daily source fetches once and freezes, and a resume rebuilds that frozen
source from the state without re-fetching — so once every session of the original
fetch is processed, the session set can never grow.

These tests pin the fix: an append-only, quota-aware, best-effort refresh that
adds the vendor's newer sessions to the *same* experiment. They are written to
catch a premise error, not to confirm the code:

- a refresh must never re-add or re-order a session the run has already processed
  (that is what keeps a duplicate from being filled twice);
- a refresh must never restate a bar the ledger already booked;
- a refresh must never change accounting, identity, the journal chain or the
  processed-session set;
- a fetch failure of any kind must leave the experiment exactly as it was and must
  not crash the engine that is already running;
- the quota guard must hold, so a retry storm cannot exhaust the free tier.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.experiment import PaperExperiment
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.journal import ExperimentJournal
from tradingagents.paper_experiment.providers.base import (
    ProviderRateLimitedError,
    ProviderStaleDataError,
)
from tradingagents.paper_experiment.refresh import (
    REFRESH_STATE_FILENAME,
    STATUS_APPENDED,
    STATUS_CONFLICT,
    STATUS_DISABLED,
    STATUS_NO_NEW_SESSIONS,
    STATUS_NOT_REFRESHABLE,
    STATUS_QUOTA_GUARD,
    STATUS_THROTTLED,
    STATUS_UNAVAILABLE,
    RefreshPolicy,
    RefreshState,
    read_refresh_state,
    refresh_source,
)
from tradingagents.paper_experiment.runner import advance, pending_sessions, process_pending
from tradingagents.paper_experiment.sources import StaticPriceSource

SYMBOL = "SPY"
# Business days, so the frame is a plausible trading calendar without weekends.
SESSION_SET = (
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-08",
    "2024-01-09",
)


def _frame(sessions, *, base: float = 100.0):
    """A frame whose values depend on the *date*, not the row position.

    So a subset and a superset agree on the bars they share, which is what a real
    vendor does and what the append/restate tests need.
    """
    index = pd.to_datetime(list(sessions))
    epoch = pd.Timestamp("2024-01-01")
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
    """A frame source that declares itself a vendor daily series."""
    source = StaticPriceSource({SYMBOL: _frame(sessions)}, name=name)
    source.kind = "vendor_daily"
    return source


def _config():
    return ExperimentConfig.from_protocol(
        symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )


def _experiment(tmp_path, sessions=SESSION_SET[:3]):
    root = tmp_path / "exp"
    experiment = PaperExperiment(
        source=_vendor_source(sessions),
        config=_config(),
        root=root,
        inference=StubInference(rating="Buy"),
    )
    experiment.ensure_created()
    experiment.save_state()
    return experiment


def _accounting(experiment) -> dict:
    state = experiment.session.state()
    ledger = state["ledger"]
    return {
        "cash": ledger["cash"],
        "fees": ledger.get("fees_paid"),
        "positions": {k: v.get("quantity") for k, v in ledger.get("positions", {}).items()},
        "events": len(ledger.get("events", [])),
    }


def _lifecycle(experiment) -> dict:
    """The deterministic lifecycle of every decision: state and both sessions."""
    return {
        decision_id: (d.state, d.execution_session, d.evaluation_session)
        for decision_id, d in sorted(experiment.decisions.items())
    }


def _fetcher(sessions, *, name: str = "alpha_vantage_daily"):
    source = _vendor_source(sessions, name=name)

    def fetch(experiment, *, today=None):
        return source

    return fetch


NOW = datetime(2024, 1, 10, tzinfo=UTC)
NO_THROTTLE = RefreshPolicy(min_interval_seconds=0.0, max_fetches_per_day=24)


# ---------------------------------------------------------------------------
# The fix: a new vendor session reaches the experiment
# ---------------------------------------------------------------------------


class TestAppendNewSessions:
    def test_a_new_vendor_session_becomes_a_pending_session(self, tmp_path):
        """The whole point: after the original fetch is exhausted, a newer bar lands."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        assert pending_sessions(experiment) == []

        outcome = refresh_source(
            experiment,
            policy=NO_THROTTLE,
            fetcher=_fetcher(SESSION_SET[:4]),
            now=NOW,
        )
        assert outcome.status == STATUS_APPENDED
        assert outcome.new_sessions == ("2024-01-05",)
        assert pending_sessions(experiment) == ["2024-01-05"]

    def test_advance_refreshes_then_processes_the_new_session(self, tmp_path):
        """One call both fetches and advances, so an auto tick is never idle by design."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)

        result = advance(
            experiment,
            refresh_policy=NO_THROTTLE,
            fetcher=_fetcher(SESSION_SET[:4]),
            now=NOW,
        )
        assert [o.session for o in result.outcomes] == ["2024-01-05"]
        assert result.refresh.status == STATUS_APPENDED

    def test_several_new_sessions_are_appended_in_order(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET), now=NOW
        )
        assert outcome.new_sessions == ("2024-01-05", "2024-01-08", "2024-01-09")
        assert pending_sessions(experiment) == ["2024-01-05", "2024-01-08", "2024-01-09"]

    def test_refresh_preserves_identity_accounting_and_journal(self, tmp_path):
        """A refresh grows the data; it must not touch the book, id or journal."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        before_id = experiment.experiment_id
        before_created = experiment.created_at
        before_processed = set(experiment.processed_sessions)
        before_accounting = _accounting(experiment)
        before_entries = len(experiment.journal)

        refresh_source(experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET), now=NOW)

        assert experiment.experiment_id == before_id
        assert experiment.created_at == before_created
        assert experiment.processed_sessions == before_processed
        # A refresh books nothing: no fill, no fee, no position change.
        assert _accounting(experiment) == before_accounting
        # The journal is appended to (state_saved), never rewritten; chain verifies.
        assert len(experiment.journal) >= before_entries
        ExperimentJournal.load(experiment.journal_path)

    def test_restart_after_refresh_sees_the_merged_sessions(self, tmp_path):
        """The appended bars are persisted, so a fresh load has them without a fetch."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        refresh_source(experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET), now=NOW)

        reloaded = PaperExperiment.load(tmp_path / "exp")
        assert set(reloaded.source.sessions()) == set(SESSION_SET)
        assert reloaded.experiment_id == experiment.experiment_id
        assert reloaded.processed_sessions == experiment.processed_sessions
        assert reloaded.source.kind == "vendor_daily"

    def test_refresh_is_deterministic_against_a_source_that_already_had_the_session(
        self, tmp_path
    ):
        """Refreshing then processing == starting with the session already present."""
        # Run A: start with 3, process, refresh to 4, process the new one.
        a = _experiment(tmp_path / "a", SESSION_SET[:3])
        process_pending(a)
        refresh_source(a, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW)
        process_pending(a)

        # Run B: start with all 4 and process them.
        b_root = tmp_path / "b"
        b = PaperExperiment(
            source=_vendor_source(SESSION_SET[:4]),
            config=_config(),
            root=b_root,
            inference=StubInference(rating="Buy"),
        )
        b.ensure_created()
        b.save_state()
        process_pending(b)

        assert _accounting(a) == _accounting(b)
        assert a.processed_sessions == b.processed_sessions
        # The lifecycle must match too, not just the book: a refreshed run must not
        # leave a decision executed forever where the straight run evaluated it.
        assert _lifecycle(a) == _lifecycle(b)

    def test_growing_one_session_at_a_time_matches_a_whole_frame_run(self, tmp_path):
        """The strongest determinism case: refresh repeatedly, one bar per step.

        An incremental run must resolve each frontier decision's execution *and*
        evaluation session as the frame grows, so its lifecycle equals a run that
        had every bar from the start. Before the evaluation-session fix, every
        decision stayed ``executed`` here and the two runs disagreed.
        """
        straight = _experiment(tmp_path / "straight", SESSION_SET)
        process_pending(straight)

        grown = _experiment(tmp_path / "grown", SESSION_SET[:2])
        process_pending(grown)
        for count in range(3, len(SESSION_SET) + 1):
            refresh_source(
                grown,
                policy=NO_THROTTLE,
                fetcher=_fetcher(SESSION_SET[:count]),
                now=NOW,
            )
            process_pending(grown)

        assert _accounting(grown) == _accounting(straight)
        assert grown.processed_sessions == straight.processed_sessions
        assert _lifecycle(grown) == _lifecycle(straight)
        # The premise: the grown run really did resolve evaluations, not leave them
        # all executed.
        assert any(state == "evaluated" for state, _, _ in _lifecycle(grown).values())


# ---------------------------------------------------------------------------
# Never re-add, never re-order, never restate
# ---------------------------------------------------------------------------


class TestAppendOnlyAndForward:
    def test_a_processed_session_is_never_re_added(self, tmp_path):
        """The vendor re-sending a session the run already used must be a no-op."""
        experiment = _experiment(tmp_path, SESSION_SET[:4])
        process_pending(experiment)
        before = _accounting(experiment)

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS
        assert outcome.new_sessions == ()
        assert _accounting(experiment) == before

    def test_an_older_backfilled_session_is_ignored_not_inserted(self, tmp_path):
        """A bar before the current last session must not reorder the run."""
        experiment = _experiment(tmp_path, ["2024-01-03", "2024-01-04"])
        process_pending(experiment)
        before_sessions = experiment.source.sessions()

        # The vendor now also reports 2024-01-02, an older bar.
        outcome = refresh_source(
            experiment,
            policy=NO_THROTTLE,
            fetcher=_fetcher(["2024-01-02", "2024-01-03", "2024-01-04"]),
            now=NOW,
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS
        assert experiment.source.sessions() == before_sessions
        assert "2024-01-02" not in experiment.source.sessions()

    def test_a_restated_bar_is_refused_and_changes_nothing(self, tmp_path):
        """A vendor changing a bar the ledger already booked must be refused."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        before = _accounting(experiment)
        before_sessions = experiment.source.sessions()

        # Same sessions, but 2024-01-03's volume is restated. Volume is a bar
        # column, so a revision to it must still be refused: the run has booked the
        # session already.
        restated = _frame(SESSION_SET[:3])
        restated.loc[pd.Timestamp("2024-01-03"), "volume"] = 9_000_000.0
        changed = StaticPriceSource({SYMBOL: restated}, name="alpha_vantage_daily")
        changed.kind = "vendor_daily"

        outcome = refresh_source(
            experiment,
            policy=NO_THROTTLE,
            fetcher=lambda e, today=None: changed,
            now=NOW,
        )
        assert outcome.status == STATUS_CONFLICT
        assert "refusing to restate" in (outcome.message or "")
        assert experiment.source.sessions() == before_sessions
        assert _accounting(experiment) == before

    def test_an_identical_existing_bar_is_not_a_conflict(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:3]), now=NOW
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS


# ---------------------------------------------------------------------------
# Quota, throttle and failure handling
# ---------------------------------------------------------------------------


class TestQuotaAndFailure:
    def test_a_second_fetch_inside_the_interval_is_throttled(self, tmp_path):
        experiment = _experiment(tmp_path)
        calls = {"n": 0}

        def fetch(experiment, *, today=None):
            calls["n"] += 1
            return _vendor_source(SESSION_SET[:4])

        policy = RefreshPolicy(min_interval_seconds=3600.0)
        first = refresh_source(experiment, policy=policy, fetcher=fetch, now=NOW)
        second = refresh_source(experiment, policy=policy, fetcher=fetch, now=NOW)
        assert first.status == STATUS_APPENDED
        assert second.status == STATUS_THROTTLED
        assert calls["n"] == 1, "a throttled refresh must not spend a call"

    def test_the_daily_quota_guard_holds(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=0.0, max_fetches_per_day=2)
        calls = {"n": 0}

        def fetch(experiment, *, today=None):
            calls["n"] += 1
            return _vendor_source(SESSION_SET[:3])

        for _ in range(5):
            refresh_source(experiment, policy=policy, fetcher=fetch, now=NOW)
        assert calls["n"] == 2, "the daily cap must bound the calls"
        last = refresh_source(experiment, policy=policy, fetcher=fetch, now=NOW)
        assert last.status == STATUS_QUOTA_GUARD

    def test_a_new_utc_day_resets_the_daily_counter(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=0.0, max_fetches_per_day=1)
        refresh_source(
            experiment, policy=policy, fetcher=_fetcher(SESSION_SET[:3]), now=NOW
        )
        blocked = refresh_source(
            experiment, policy=policy, fetcher=_fetcher(SESSION_SET[:3]), now=NOW
        )
        assert blocked.status == STATUS_QUOTA_GUARD

        tomorrow = datetime(2024, 1, 11, tzinfo=UTC)
        allowed = refresh_source(
            experiment, policy=policy, fetcher=_fetcher(SESSION_SET[:4]), now=tomorrow
        )
        assert allowed.status == STATUS_APPENDED

    def test_a_fetch_exception_is_reported_not_raised_and_changes_nothing(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        before = _accounting(experiment)
        before_sessions = experiment.source.sessions()

        def boom(experiment, *, today=None):
            raise RuntimeError("socket exploded")

        outcome = refresh_source(experiment, policy=NO_THROTTLE, fetcher=boom, now=NOW)
        assert outcome.status == STATUS_UNAVAILABLE
        assert "socket exploded" in (outcome.message or "")
        assert experiment.source.sessions() == before_sessions
        assert _accounting(experiment) == before

    def test_a_quota_notice_is_reported_as_unavailable(self, tmp_path):
        experiment = _experiment(tmp_path)

        def limited(experiment, *, today=None):
            raise ProviderRateLimitedError("Alpha Vantage quota/limit reached")

        outcome = refresh_source(experiment, policy=NO_THROTTLE, fetcher=limited, now=NOW)
        assert outcome.status == STATUS_UNAVAILABLE
        assert "quota" in (outcome.message or "").lower()

    def test_a_stale_feed_is_reported_as_unavailable(self, tmp_path):
        experiment = _experiment(tmp_path)

        def stale(experiment, *, today=None):
            raise ProviderStaleDataError("newest bar is older than the limit")

        outcome = refresh_source(experiment, policy=NO_THROTTLE, fetcher=stale, now=NOW)
        assert outcome.status == STATUS_UNAVAILABLE

    def test_a_nan_bar_in_the_new_frame_is_refused(self, tmp_path):
        """A malformed vendor bar must not reach the ledger; the refresh is refused."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        before_sessions = experiment.source.sessions()

        bad = _frame(SESSION_SET[:4])
        bad.loc[pd.Timestamp("2024-01-05"), "close"] = float("nan")
        # Bypass the fixture's eager validation by building the frame source
        # directly and swapping the stored frame, exactly as a corrupted vendor
        # payload would look after parsing.
        source = StaticPriceSource({SYMBOL: _frame(SESSION_SET[:4])}, name="alpha_vantage_daily")
        source.kind = "vendor_daily"
        source._frames[SYMBOL] = bad

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=lambda e, today=None: source, now=NOW
        )
        assert outcome.status == STATUS_UNAVAILABLE
        assert experiment.source.sessions() == before_sessions

    def test_an_infinite_bar_in_the_new_frame_is_refused(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        before_sessions = experiment.source.sessions()

        bad = _frame(SESSION_SET[:4])
        bad.loc[pd.Timestamp("2024-01-05"), "high"] = float("inf")
        source = StaticPriceSource({SYMBOL: _frame(SESSION_SET[:4])}, name="alpha_vantage_daily")
        source.kind = "vendor_daily"
        source._frames[SYMBOL] = bad

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=lambda e, today=None: source, now=NOW
        )
        assert outcome.status == STATUS_UNAVAILABLE
        assert experiment.source.sessions() == before_sessions

    def test_disabled_policy_never_fetches(self, tmp_path):
        experiment = _experiment(tmp_path)
        calls = {"n": 0}

        def fetch(experiment, *, today=None):
            calls["n"] += 1
            return _vendor_source(SESSION_SET[:4])

        outcome = refresh_source(
            experiment, policy=RefreshPolicy(enabled=False), fetcher=fetch, now=NOW
        )
        assert outcome.status == STATUS_DISABLED
        assert calls["n"] == 0

    def test_a_frozen_source_is_not_refreshable_and_writes_no_sidecar(self, tmp_path):
        """A frozen run has no vendor to ask; a monitor of it stays read-only."""
        root = tmp_path / "frozen"
        experiment = PaperExperiment(
            source=StaticPriceSource({SYMBOL: _frame(SESSION_SET)}, name="frozen_baseline"),
            config=_config(),
            root=root,
            inference=StubInference(rating="Hold"),
        )
        experiment.ensure_created()
        experiment.save_state()

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET), now=NOW
        )
        assert outcome.status == STATUS_NOT_REFRESHABLE
        assert not (root / REFRESH_STATE_FILENAME).exists()

    def test_a_two_symbol_source_cannot_be_refreshed(self, tmp_path):
        """The refresh cannot guess a symbol, so it refuses rather than fetching one."""
        root = tmp_path / "multi"
        frames = {
            "SPY": _frame(SESSION_SET),
            "QQQ": _frame(SESSION_SET, base=300.0),
        }
        source = StaticPriceSource(frames, name="alpha_vantage_daily")
        source.kind = "vendor_daily"
        experiment = PaperExperiment(
            source=source, config=_config(), root=root, inference=StubInference(rating="Hold")
        )
        experiment.ensure_created()
        experiment.save_state()

        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET), now=NOW
        )
        assert outcome.status == STATUS_NOT_REFRESHABLE
        assert "symbol" in (outcome.message or "")


# ---------------------------------------------------------------------------
# Holidays, weekends, and a delayed vendor
# ---------------------------------------------------------------------------


class TestCalendarEdges:
    def test_a_weekend_with_no_new_bar_is_not_an_error(self, tmp_path):
        """On a Saturday the vendor's newest bar is still Friday: no new session."""
        friday = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
        experiment = _experiment(tmp_path, friday)
        process_pending(experiment)
        saturday = datetime(2024, 1, 6, tzinfo=UTC)
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(friday), now=saturday
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS
        assert outcome.last_bar_date == "2024-01-05"

    def test_a_vendor_delay_leaves_the_last_bar_unchanged(self, tmp_path):
        """A delayed feed (no newer bar) is a no-op, not a fabricated session."""
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:3]), now=NOW
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS
        assert outcome.last_bar_date == "2024-01-04"

    def test_a_holiday_gap_in_the_vendor_calendar_is_fine(self, tmp_path):
        """A calendar that simply skips a holiday is appended as-is."""
        experiment = _experiment(tmp_path, ["2024-01-02", "2024-01-03"])
        process_pending(experiment)
        # 2024-01-04 is treated as a skipped session here; 2024-01-05 follows.
        outcome = refresh_source(
            experiment,
            policy=NO_THROTTLE,
            fetcher=_fetcher(["2024-01-02", "2024-01-03", "2024-01-05"]),
            now=NOW,
        )
        assert outcome.status == STATUS_APPENDED
        assert outcome.new_sessions == ("2024-01-05",)


# ---------------------------------------------------------------------------
# The frontier decision: no fill at the decision price, resolved on refresh
# ---------------------------------------------------------------------------


class TestFrontierDecisionIsNotLookAhead:
    """A decision on the newest bar has no T+1 yet.

    Before the refresh feature the frame never grew, so this case was only ever
    the final, never-settled decision. Once the frame *can* grow, pinning a
    frontier decision's execution to its own bar would let a later bar fill it at
    the decision price — look-ahead. These tests pin the fix: the execution
    session stays unresolved until a successor exists, and then resolves to it.
    """

    def test_a_frontier_decision_has_no_execution_session_yet(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        frontier = experiment.decisions[f"{SYMBOL}-2024-01-04"]
        assert frontier.execution_session is None
        assert frontier.state == "pending"

    def test_no_decision_executes_at_its_own_decision_bar(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET)
        process_pending(experiment)
        for decision in experiment.decisions.values():
            if decision.execution_session is not None:
                assert decision.execution_session > decision.decision_session

    def test_the_frontier_decision_resolves_to_the_refreshed_successor(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        assert experiment.decisions[f"{SYMBOL}-2024-01-04"].execution_session is None

        refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        process_pending(experiment)

        frontier = experiment.decisions[f"{SYMBOL}-2024-01-04"]
        assert frontier.execution_session == "2024-01-05"
        assert frontier.state in ("executed", "evaluated")

    def test_a_blocked_frontier_decision_is_also_left_unresolved(self, tmp_path):
        root = tmp_path / "blocked"
        experiment = PaperExperiment(
            source=_vendor_source(SESSION_SET[:3]),
            config=_config(),
            root=root,
            inference=StubInference(rating="not-a-real-rating"),
        )
        experiment.ensure_created()
        experiment.save_state()
        process_pending(experiment)
        blocked = experiment.decisions[f"{SYMBOL}-2024-01-04"]
        assert blocked.state == "blocked"
        assert blocked.execution_session is None


# ---------------------------------------------------------------------------
# The log line and the sidecar
# ---------------------------------------------------------------------------


class TestStatusReporting:
    def test_the_summary_line_carries_the_required_fields(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        process_pending(experiment)
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        line = outcome.summary_line()
        assert "[data]" in line
        assert "source_last_bar=2024-01-05" in line
        assert "last_new_session=2024-01-05" in line
        assert "last_processed=2024-01-04" in line
        assert "status=appended" in line

    def test_the_summary_line_never_carries_a_secret(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        line = outcome.summary_line()
        for token in ("apikey", "api_key", "token", "secret", "ALPHA_VANTAGE_API_KEY"):
            assert token not in line.lower()

    def test_the_sidecar_round_trips_and_is_additive(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        path = tmp_path / "exp" / REFRESH_STATE_FILENAME
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 1
        assert payload["last_status"] == STATUS_APPENDED
        # An unknown key from a newer build must not break the load.
        payload["future_field"] = 1
        path.write_text(json.dumps(payload))
        state = read_refresh_state(tmp_path / "exp")
        assert state.last_status == STATUS_APPENDED

    def test_a_corrupt_sidecar_does_not_break_a_refresh(self, tmp_path):
        experiment = _experiment(tmp_path, SESSION_SET[:3])
        (tmp_path / "exp" / REFRESH_STATE_FILENAME).write_text("{not json")
        outcome = refresh_source(
            experiment, policy=NO_THROTTLE, fetcher=_fetcher(SESSION_SET[:4]), now=NOW
        )
        assert outcome.status == STATUS_APPENDED

    def test_the_refresh_state_defaults_are_safe(self):
        state = RefreshState.load(Path("/nonexistent/data_refresh.json"))
        assert state.fetches_today == 0
        assert state.last_status is None


# ---------------------------------------------------------------------------
# The dashboard control path reports the refresh
# ---------------------------------------------------------------------------


class TestDashboardRefresh:
    def test_run_sessions_reports_a_json_safe_refresh_block(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        _experiment(tmp_path, SESSION_SET[:3])
        dashboard = Dashboard(
            tmp_path / "exp",
            allow_control=True,
            stub=True,
            refresh_policy=NO_THROTTLE,
            refresh_fetcher=_fetcher(SESSION_SET[:4]),
        )
        result = dashboard.run_sessions(100)
        # The result is serializable (the /api/run route sends it as JSON).
        encoded = json.dumps(result)
        assert "refresh" in result
        assert result["refresh"]["status"] == STATUS_APPENDED
        assert "2024-01-05" in encoded

    def test_auto_advance_logs_the_data_line(self, tmp_path, capsys):
        """The operator sees the last data date and status, not just 'nothing to advance'."""
        from tradingagents.paper_experiment.dashboard.server import Dashboard

        _experiment(tmp_path, SESSION_SET[:3])
        dashboard = Dashboard(
            tmp_path / "exp",
            allow_control=True,
            stub=True,
            refresh_policy=NO_THROTTLE,
            refresh_fetcher=_fetcher(SESSION_SET[:4]),
        )
        dashboard.run_sessions(100)
        # A second run has nothing new to fetch and nothing pending; the loop's line
        # must still explain why.
        result = dashboard.run_sessions(100)
        line = result["refresh"]["summary"]
        assert "[data]" in line
        assert "status=" in line
        assert "source_last_bar=2024-01-05" in line
