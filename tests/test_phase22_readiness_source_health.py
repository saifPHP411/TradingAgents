"""Phase 22: readiness must not claim currency while the source is unreachable.

The reported inconsistency: the freshness view could report
``market_status = data_source_unavailable`` (``source_status = failed``) after a
refresh that could not reach the vendor, while ``readiness`` still read
``READY FOR LIVE PAPER`` because it looked only at the source's *currency* flag and
never at the source's *health*. Two views of the same run disagreed.

These tests pin the corrected behaviour and, just as importantly, the behaviour
that must **not** change: a weekend or a holiday with no new session is not a
failure, and a skip (throttled/quota) is not a failure either. A test that fails
here should first be checked for a wrong premise: the point is that "the source
could not be reached" and "there is simply nothing new today" are different facts
and must stay different.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.experiment import PaperExperiment
from tradingagents.paper_experiment.freshness import (
    MARKET_STATUS_SOURCE_UNAVAILABLE,
    market_freshness,
)
from tradingagents.paper_experiment.providers.alpha_vantage import (
    AlphaVantageDailySource,
)
from tradingagents.paper_experiment.providers.base import (
    HttpResponse,
    ProviderAuthError,
    ProviderNotConfiguredError,
    ProviderStaleDataError,
)
from tradingagents.paper_experiment.refresh import (
    STATUS_AUTH_FAILED,
    STATUS_NO_NEW_SESSIONS,
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    RefreshPolicy,
    refresh_source,
)
from tradingagents.paper_experiment.report import build_report

SYMBOL = "SPY"


def _transport(body: str, *, status_code: int = 200):
    def send(*, url, params, headers, timeout):
        return HttpResponse(status_code, body)

    return send


def _payload(latest: date, *, days: int = 120) -> str:
    """An Alpha Vantage daily body whose newest bar is ``latest`` (weekdays only)."""
    series: dict[str, dict] = {}
    cursor = latest
    made = 0
    while made < days:
        if cursor.weekday() < 5:
            series[cursor.isoformat()] = {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. volume": "1000000",
            }
            made += 1
        cursor -= timedelta(days=1)
    return json.dumps(
        {
            "Meta Data": {"2. Symbol": SYMBOL, "5. Time Zone": "US/Eastern"},
            "Time Series (Daily)": series,
        }
    )


def _source(latest: date, *, today: date) -> AlphaVantageDailySource:
    return AlphaVantageDailySource(
        SYMBOL, api_key="k", transport=_transport(_payload(latest)), today=today
    )


def _experiment(tmp_path, *, latest: date, today: date) -> PaperExperiment:
    root = tmp_path / "exp"
    experiment = PaperExperiment(
        source=_source(latest, today=today),
        config=ExperimentConfig.from_protocol(symbols=(SYMBOL,)),
        root=root,
        inference=None,
    )
    experiment.ensure_created()
    experiment.save_state()
    return experiment


def _with_genuine_inference(experiment: PaperExperiment) -> PaperExperiment:
    """Record one genuine-inference decision, so readiness can reach its top verdict.

    The verdict this phase guards is the *top* one ("READY FOR LIVE PAPER"), which
    requires a genuine inference as well as a healthy current source. Rather than
    run a real provider, a single decision is written with the same ``source``
    label a genuine call records; the readiness rule under test reads only that
    label. A synthetic-fixture decision is used nowhere here, so a passing
    "not ready" assertion can never come from the fixture path.
    """
    from tradingagents.llm_evaluation.records import SOURCE_LLM_INFERENCE
    from tradingagents.paper_experiment.experiment import (
        STATE_EVALUATED,
        DecisionState,
    )

    session = experiment.source.sessions()[-1]
    decision_id = f"{SYMBOL}-{session}"
    experiment.decisions[decision_id] = DecisionState(
        decision_id=decision_id,
        symbol=SYMBOL,
        decision_session=session,
        execution_session=None,
        evaluation_session=None,
        state=STATE_EVALUATED,
        source=SOURCE_LLM_INFERENCE,
    )
    experiment.save_state()
    return experiment


# ---------------------------------------------------------------------------
# A current, reachable source reads ready
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCurrentSourceIsReady:
    def test_a_fresh_bar_is_current_market(self, tmp_path):
        today = date(2026, 11, 2)  # a Monday
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=today, today=today)
        )
        assert experiment.source.is_current_market is True
        assert build_report(experiment).to_dict()["readiness"]["status"] == (
            "READY FOR LIVE PAPER"
        )

    def test_a_weekend_gap_is_still_current(self, tmp_path):
        # Newest bar is Friday; "today" is the following Monday (a 3-day gap).
        monday = date(2026, 11, 2)
        friday = date(2026, 10, 30)
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=friday, today=monday)
        )
        assert experiment.source.is_current_market is True
        assert build_report(experiment).to_dict()["readiness"]["status"] == (
            "READY FOR LIVE PAPER"
        )


# ---------------------------------------------------------------------------
# An unreachable source never reads ready
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnreachableSourceIsNotReady:
    def _after_refresh(self, tmp_path, fetcher, *, latest=None, today=date(2026, 11, 2)):
        experiment = _with_genuine_inference(
            _experiment(
                tmp_path, latest=latest or (today - timedelta(days=5)), today=today
            )
        )
        outcome = refresh_source(
            experiment,
            policy=RefreshPolicy(),
            fetcher=fetcher,
            now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
        )
        return experiment, outcome

    def test_an_auth_failure_drops_readiness_from_ready(self, tmp_path):
        def fetcher(experiment, *, today=None):
            raise ProviderAuthError("rejected")

        experiment, outcome = self._after_refresh(tmp_path, fetcher)
        assert outcome.status == STATUS_AUTH_FAILED
        assert outcome.source_status == "failed"
        # Freshness says unavailable ...
        assert (
            market_freshness(experiment, now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC)).market_status
            == MARKET_STATUS_SOURCE_UNAVAILABLE
        )
        # ... and readiness agrees, instead of claiming READY.
        report = build_report(experiment).to_dict()
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"
        assert any("data_source_unavailable" in b for b in report["readiness"]["blockers"])

    def test_a_missing_credential_is_reported_and_not_ready(self, tmp_path):
        def fetcher(experiment, *, today=None):
            raise ProviderNotConfiguredError("no key")

        experiment, outcome = self._after_refresh(tmp_path, fetcher)
        assert outcome.status == STATUS_NOT_CONFIGURED
        report = build_report(experiment).to_dict()
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"

    def test_a_stale_feed_is_reported_and_not_ready(self, tmp_path):
        def fetcher(experiment, *, today=None):
            raise ProviderStaleDataError("newest bar is older than the limit")

        experiment, outcome = self._after_refresh(tmp_path, fetcher)
        assert outcome.status == STATUS_UNAVAILABLE
        report = build_report(experiment).to_dict()
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"
        assert any("data_source_unavailable" in b for b in report["readiness"]["blockers"])

    def test_the_readiness_and_freshness_views_agree(self, tmp_path):
        """The precise bug: the two views must never disagree about reachability."""

        def fetcher(experiment, *, today=None):
            raise ProviderAuthError("rejected")

        experiment, _ = self._after_refresh(tmp_path, fetcher)
        freshness = market_freshness(
            experiment, now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
        )
        report = build_report(experiment).to_dict()
        freshness_unavailable = (
            freshness.market_status == MARKET_STATUS_SOURCE_UNAVAILABLE
        )
        readiness_ready = report["readiness"]["status"] == "READY FOR LIVE PAPER"
        assert not (freshness_unavailable and readiness_ready)


# ---------------------------------------------------------------------------
# A no-new-session tick is not a failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoNewSessionsIsNotAFailure:
    def test_a_holiday_with_the_same_newest_bar_is_not_a_failure(self, tmp_path):
        today = date(2026, 11, 2)  # Monday
        friday = date(2026, 10, 30)
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=friday, today=today)
        )

        # The refresh returns the *same* newest bar: no new session today.
        def fetcher(experiment, *, today=None):
            return AlphaVantageDailySource(
                SYMBOL, api_key="k", transport=_transport(_payload(friday)), today=today
            )

        outcome = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=fetcher,
            now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
        )
        assert outcome.status == STATUS_NO_NEW_SESSIONS
        assert outcome.source_status == "ok"
        assert outcome.failure_kind is None
        # Readiness is unaffected: the source is healthy and current.
        assert build_report(experiment).to_dict()["readiness"]["status"] == (
            "READY FOR LIVE PAPER"
        )

    def test_a_skip_does_not_count_as_a_source_failure(self, tmp_path):
        """A throttled tick is "not asked", not "failed"; readiness stays ready."""
        today = date(2026, 11, 2)
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=today, today=today)
        )
        # First refresh succeeds (records the attempt time); the second is throttled.
        refresh_source(
            experiment, policy=RefreshPolicy(),
            fetcher=lambda e, today=None: _source(today, today=today),
            now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
        )
        throttled = refresh_source(
            experiment, policy=RefreshPolicy(),
            fetcher=lambda e, today=None: _source(today, today=today),
            now=datetime(2026, 11, 2, 14, 5, tzinfo=UTC),
        )
        assert throttled.attempted is False
        assert build_report(experiment).to_dict()["readiness"]["status"] == (
            "READY FOR LIVE PAPER"
        )


# ---------------------------------------------------------------------------
# A refresh that appends a new bar keeps the run current
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefreshAppendsAndStaysCurrent:
    def test_appending_a_new_session_keeps_readiness_ready(self, tmp_path):
        today = date(2026, 11, 2)
        start = today - timedelta(days=4)  # an older newest bar at start
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=start, today=today)
        )

        def fetcher(experiment, *, today=None):
            return _source(today, today=today)

        outcome = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=fetcher,
            now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
        )
        assert outcome.status == "appended"
        assert outcome.new_sessions
        assert build_report(experiment).to_dict()["readiness"]["status"] == (
            "READY FOR LIVE PAPER"
        )

    def test_a_second_refresh_adds_no_duplicate_session(self, tmp_path):
        today = date(2026, 11, 2)
        experiment = _with_genuine_inference(
            _experiment(tmp_path, latest=today, today=today)
        )

        def fetcher(experiment, *, today=None):
            return _source(today, today=today)

        first = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=fetcher,
            now=datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
        )
        before = experiment.source.sessions()
        second = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=fetcher,
            now=datetime(2026, 11, 2, 15, 0, tzinfo=UTC),
        )
        assert second.status == STATUS_NO_NEW_SESSIONS
        assert experiment.source.sessions() == before
        # No duplicate session label was introduced by the first append either.
        assert len(set(before)) == len(before)
        del first
