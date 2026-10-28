"""Point-in-time integrity and leakage tests for the LLM evaluation layer.

Two things are checked here that the existing suites do not:

1. **The audit matrix matches the code.** The dependency matrix is derived from
   the graph spec, the router's vendor map and the PIT registry. If any of those
   change, a test here fails, so the audit cannot quietly go stale.

2. **The boundary rejects leakage.** Future-dated data, an unavailable feed, a
   record whose window runs past its decision date, a hash mismatch, a
   credential in a record — each is a distinct way information from after the
   decision could enter a historical run, and each has a test that it is refused.

Nothing here calls a model or the network. The provider boundary is mocked only
where a provider must be simulated, and a mock is never presented as genuine
model performance.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_evaluation import (  # noqa: E402
    CLOSES,
    _frame,
    make_record as _make_record_raw,
    make_set as _make_set_raw,
)

# ---------------------------------------------------------------------------
# Fixtures: a small session-bounded feed, reusing the llm_evaluation helpers
# ---------------------------------------------------------------------------
from tradingagents.backtesting import MarketDataFeed
from tradingagents.dataflows.pit import PIT_GUARANTEES, PITLevel, get_guarantee
from tradingagents.llm_evaluation import (
    Classification,
    DecisionRecordError,
    build_evaluation,
    graph_readiness,
    point_in_time_report,
    source_facts,
)
from tradingagents.llm_evaluation.evaluate import STATUS_BLOCKED
from tradingagents.llm_evaluation.point_in_time import (
    agent_dependency_matrix,
    agent_sources,
    tool_vendors,
)
from tradingagents.llm_evaluation.records import SOURCE_LLM_INFERENCE


@pytest.fixture
def feed() -> MarketDataFeed:
    return MarketDataFeed(
        {"AAPL": _frame(), "SPY": _frame([200.0 + i for i in range(len(CLOSES))])}
    )


def _make_record(feed, **kwargs):
    return _make_record_raw(feed, **kwargs)


def _make_set(decisions, **kwargs):
    return _make_set_raw(decisions, **kwargs)


# ---------------------------------------------------------------------------
# 1. The audit matrix is derived from, and consistent with, the code
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatrixMatchesTheCode:
    def test_every_analyst_in_the_graph_is_in_the_matrix(self):
        from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

        matrix = agent_dependency_matrix()
        assert set(matrix) == set(ANALYST_NODE_SPECS)

    def test_every_source_in_the_matrix_is_classified_in_the_registry(self):
        """A source the matrix names but the registry does not is uncheckable."""
        matrix = agent_dependency_matrix()
        for entry in matrix.values():
            for source in entry["sources"]:
                assert source in PIT_GUARANTEES, f"{source} is not in the PIT registry"

    def test_the_market_analyst_reads_only_verified_sources(self):
        matrix = agent_dependency_matrix()
        for source in matrix["market"]["sources"]:
            assert source_facts()[source].classification is Classification.PIT_VERIFIED

    def test_the_sentiment_analysts_prefetched_sources_are_recorded(self):
        # Its tool tuple is empty because it fetches before calling the model, so
        # the sources must be recorded explicitly or its dependencies vanish.
        sources = agent_sources()["social"]
        assert "stocktwits" in sources and "reddit" in sources and "yahoo_news" in sources

    def test_every_tool_vendor_is_reflected_in_the_router(self):
        from tradingagents.dataflows.router import VENDOR_METHODS

        for tool, vendors in tool_vendors().items():
            assert set(vendors) == set(VENDOR_METHODS[tool])

    def test_a_required_source_is_not_marked_optional(self):
        facts = source_facts()
        # Prices are the one input nothing can proceed without.
        assert facts["yahoo_prices"].required_for_graph is True
        # A prediction market is enrichment; it degrades to a sentinel.
        assert facts["polymarket"].required_for_graph is False
        assert facts["polymarket"].disables_cleanly is True

    def test_the_classification_mapping_is_conservative(self):
        """Only a request-bounded source is PIT_VERIFIED."""
        from tradingagents.llm_evaluation.point_in_time import _LEVEL_TO_CLASSIFICATION

        assert _LEVEL_TO_CLASSIFICATION[PITLevel.REQUEST_BOUNDED] is Classification.PIT_VERIFIED
        assert (
            _LEVEL_TO_CLASSIFICATION[PITLevel.WINDOW_FILTERED]
            is Classification.PIT_PARTIALLY_VERIFIED
        )
        assert _LEVEL_TO_CLASSIFICATION[PITLevel.NOT_PIT_SAFE] is Classification.CURRENT_ONLY

    def test_every_source_gets_a_classification_from_its_registry_level(self):
        for source, fact in source_facts().items():
            level = get_guarantee(source).level
            if level is PITLevel.REQUEST_BOUNDED:
                assert fact.classification is Classification.PIT_VERIFIED
            elif level is PITLevel.WINDOW_FILTERED:
                assert fact.classification is Classification.PIT_PARTIALLY_VERIFIED
            else:
                assert fact.classification is Classification.CURRENT_ONLY


# ---------------------------------------------------------------------------
# 2. Readiness verdict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadinessVerdict:
    def test_a_price_only_historical_run_is_ready(self):
        verdict = graph_readiness("price_only")
        assert verdict.ready is True
        assert verdict.blockers == ()
        assert set(verdict.verified_sources) == {"yahoo_prices", "yahoo_indicators"}

    def test_a_full_graph_historical_run_is_not_ready(self):
        verdict = graph_readiness("full")
        assert verdict.ready is False
        assert verdict.blockers
        # The weak and current-only sources are named, not just counted.
        assert "yahoo_news" in verdict.weak_sources
        assert "yahoo_profile" in verdict.current_only_sources

    def test_an_unknown_scope_is_refused(self):
        with pytest.raises(ValueError, match="scope"):
            graph_readiness("everything")

    def test_the_verdict_is_json_safe(self):
        json.dumps(graph_readiness("full").to_dict())

    def test_the_report_carries_the_matrix_and_the_counts(self):
        report = point_in_time_report("full")
        assert report["agent_dependency_matrix"]
        assert report["source_classifications"]
        assert report["verified_source_count"] >= 1
        assert report["current_only_source_count"] >= 1
        json.dumps(report)


# ---------------------------------------------------------------------------
# 3. Leakage: future-dated data is refused
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFutureDataIsRefused:
    def test_a_record_window_past_its_decision_date_is_refused(self, feed):
        # The model may not have seen a bar after the decision. A window ending
        # later is look-ahead, so the record cannot be built.
        with pytest.raises(DecisionRecordError, match="window_end must equal as_of_date"):
            _make_record(feed, session="2022-03-06", window_end="2022-03-20")

    def test_the_feed_never_returns_a_bar_after_the_session(self, feed):
        assert feed.bar("AAPL", "2027-01-01") is None

    def test_the_protocol_refuses_a_full_scope_that_needs_unavailable_pit_data(self):
        from tradingagents.llm_evaluation import LLMEvaluationProtocol

        with pytest.raises(ValueError, match="point-in-time"):
            LLMEvaluationProtocol(input_scope="full")


# ---------------------------------------------------------------------------
# 4. Leakage: missing / malformed availability and hashes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingAndMalformedInputs:
    def test_a_short_window_hash_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="64-character"):
            _make_record(feed, data_window_hash="deadbeef")

    def test_a_short_config_hash_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="64-character"):
            _make_record(feed, config_hash="not-a-hash")

    def test_a_hash_mismatch_is_refused_by_the_replay(self, feed):
        from tradingagents.llm_evaluation import ReplayError, ReplayStrategy

        record = _make_record(feed, session="2022-03-06", data_window_hash="a" * 64)
        strategy = ReplayStrategy(_make_set([record]))
        # The replay recomputes the window hash from the session-bounded feed and
        # refuses a record whose stored hash does not match.
        context = _context(feed, "2022-03-06")
        with pytest.raises(ReplayError, match="window"):
            strategy(context)

    def test_an_unparseable_date_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="yyyy-mm-dd"):
            _make_record(feed, session="2022-03-06", as_of_date="06/03/2022")


def _context(feed, session):
    """A minimal StrategyContext over the session-bounded feed."""
    from tradingagents.backtesting.data import SessionClock
    from tradingagents.backtesting.engine import StrategyContext
    from tradingagents.portfolio.ledger import Ledger

    return StrategyContext(
        session=session,
        feed=feed,
        ledger=Ledger(initial_cash=100_000.0, currency="USD"),
        clock=SessionClock.for_session(session),
        symbols=feed.symbols,
    )


# ---------------------------------------------------------------------------
# 5. Credentials never reach a record or the artifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoCredentialsInOutputs:
    def test_a_credential_in_the_rationale_is_refused(self, feed):
        from tradingagents.llm_evaluation import CredentialLeakError

        with pytest.raises(CredentialLeakError):
            _make_record(feed, reasoning_excerpt="key sk-" + "a" * 40)

    def test_the_artifact_never_carries_an_environment_secret(self, monkeypatch, feed):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "b" * 40)
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert "sk-" + "b" * 40 not in json.dumps(outcome.artifact)

    def test_credential_status_reports_names_not_values(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "c" * 40)
        from tradingagents.llm_evaluation import credential_status

        status = credential_status()
        assert status["env_var_by_provider"]["openai"] == "OPENAI_API_KEY"
        assert "sk-" + "c" * 40 not in json.dumps(status)


# ---------------------------------------------------------------------------
# 6. Provider failure produces no fabricated decision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderFailureDoesNotFabricate:
    def test_a_blocked_build_has_null_llm_results(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert outcome.status == STATUS_BLOCKED
        assert outcome.artifact["llm_results"] is None

    def test_a_replay_with_no_credential_is_never_marked_genuine_inference(self, feed):
        record_set = _make_set(
            [_make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        )
        outcome = build_evaluation(
            record_set=record_set, feed=feed, include_reference_context=False
        )
        assert outcome.artifact["genuine_inference"] is False

    def test_only_an_llm_inference_set_is_genuine(self, feed):
        record_set = _make_set(
            [_make_record(feed, session="2022-03-06", action="BUY", rating="Buy",
                          source=SOURCE_LLM_INFERENCE)],
            mode=SOURCE_LLM_INFERENCE,
        )
        outcome = build_evaluation(
            record_set=record_set, feed=feed, include_reference_context=False
        )
        assert outcome.artifact["genuine_inference"] is True


# ---------------------------------------------------------------------------
# 7. The artifact records the restricted scope honestly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArtifactScopeHonesty:
    def test_the_artifact_names_the_omitted_agents(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        scope = outcome.artifact["evaluation_scope"]
        assert scope["input_scope"] == "price_only"
        assert scope["restricted"] is True
        assert scope["agents_evaluated"] == ["market"]
        assert set(scope["agents_omitted"]) == {"social", "news", "fundamentals"}

    def test_the_artifact_carries_the_point_in_time_block(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        pit = outcome.artifact["point_in_time"]
        assert pit["scope"] == "price_only"
        assert pit["readiness"]["ready"] is True
        assert pit["agent_dependency_matrix"]

    def test_a_restricted_scope_is_recorded_as_a_blocker(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert any("RESTRICTED" in b for b in outcome.blockers)

    def test_the_limitations_block_says_the_full_graph_was_not_measured(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        limitations = outcome.artifact["limitations"]
        assert limitations["news_sentiment_fundamentals_evaluated"] is False


# ---------------------------------------------------------------------------
# 8. Determinism: the audit is a property of the code, not of the moment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuditDeterminism:
    def test_two_point_in_time_reports_are_identical(self):
        assert point_in_time_report("full") == point_in_time_report("full")

    def test_the_artifact_is_identical_apart_from_the_timestamp(self, feed):
        first = build_evaluation(record_set=None, feed=feed, include_reference_context=False)
        second = build_evaluation(record_set=None, feed=feed, include_reference_context=False)
        a, b = dict(first.artifact), dict(second.artifact)
        a.pop("generated_at"), b.pop("generated_at")
        assert a == b


# ---------------------------------------------------------------------------
# 9. Duplicate decisions are refused
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateDecisions:
    def test_a_duplicate_decision_id_is_refused(self, feed):
        record = _make_record(feed, session="2022-03-06", decision_id="dup-1")
        with pytest.raises(DecisionRecordError, match="duplicate decision_id"):
            _make_set([record, record])


# ---------------------------------------------------------------------------
# 10. The audit does not weaken the safety boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafetyBoundaryUnchanged:
    def test_no_live_mode_is_reachable_from_the_audit(self):
        from tradingagents.execution import ExecutionMode

        assert {m.value for m in ExecutionMode} == {"BACKTEST", "PAPER"}

    def test_the_audit_adds_no_model_call(self, monkeypatch):
        # Importing and running the audit must not build an LLM client.
        from tradingagents.llm_clients import factory

        def _boom(*a, **k):
            raise AssertionError("the audit must not create an LLM client")

        monkeypatch.setattr(factory, "create_llm_client", _boom)
        point_in_time_report("full")
