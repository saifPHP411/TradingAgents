"""Tests for the Phase 8 decision-capture path, smoke test and provenance gates.

These are offline: no model is called. The fake client is a seam, and every test
that uses it asserts the capture is labelled a fixture, never genuine inference.
The tests that matter most are the ones that could catch a lie:

- a graph with no observed model call cannot yield an ``llm_inference`` record,
  even when the caller passes ``mode=REAL_INFERENCE_CAPTURE``;
- an injected (test) smoke client is never reported as a genuine observation;
- a rating the parser cannot read (REVIEW) is refused, not degraded to Hold;
- replay never constructs an LLM client and never emits a fabricated signal;
- a tampered window hash and a tampered config hash are both refused;
- a credential-shaped value cannot reach a record set.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.backtesting import MarketDataFeed
from tradingagents.llm_evaluation import (
    DEFAULT_PROTOCOL as LLM_PROTOCOL,
    MODE_MOCK_TEST,
    MODE_REAL_INFERENCE_CAPTURE,
    SOURCE_LLM_INFERENCE,
    SOURCE_SYNTHETIC_FIXTURE,
    CaptureError,
    DecisionRecordSet,
    ReplayError,
    ReplayStrategy,
    SmokeBlocked,
    build_record,
    capture_decision,
    graph_observed_real_inference,
    observed_decision,
    run_smoke,
    signal_from_record,
    window_hash,
)

SESSIONS = [f"2022-03-{d:02d}" for d in range(1, 15)]
CLOSES = [100.0 + i for i in range(14)]


def _frame(closes=None, dates=None) -> pd.DataFrame:
    closes = closes if closes is not None else CLOSES
    dates = dates if dates is not None else SESSIONS
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=pd.to_datetime(dates),
    )


@pytest.fixture
def feed() -> MarketDataFeed:
    return MarketDataFeed(
        {"SPY": _frame([200.0 + i for i in range(len(CLOSES))]), "AAPL": _frame()}
    )


class _FakeGraph:
    """A graph double: reports a real model call only when told to."""

    def __init__(self, *, observed: bool = False, provider="groq", model="openai/gpt-oss-20b"):
        self._observed = observed
        self._provider = provider
        self._model = model

    @property
    def inference_observed(self) -> bool:
        return self._observed

    def run_settings(self) -> dict:
        return {
            "llm_provider": self._provider,
            "deep_think_provider": self._provider,
            "deep_think_llm": self._model,
            "quick_think_llm": self._model,
        }


def _state(**overrides) -> dict:
    base = {
        "company_of_interest": "SPY",
        "trade_date": "2022-03-06",
        "final_rating": "Overweight",
        "pm_executive_summary": "Mildly constructive; sized modestly.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The genuine-inference discriminator
# ---------------------------------------------------------------------------


class TestInferenceDiscriminator:
    def test_a_graph_with_no_observed_call_is_not_genuine(self):
        assert graph_observed_real_inference(_FakeGraph(observed=False)) is False

    def test_a_graph_that_observed_a_call_is_genuine(self):
        assert graph_observed_real_inference(_FakeGraph(observed=True)) is True

    def test_an_object_without_the_marker_fails_closed(self):
        class Bare:
            pass

        assert graph_observed_real_inference(Bare()) is False

    def test_a_stubbed_run_cannot_be_labelled_genuine_even_when_asked(self, feed):
        """The caller's mode is a request; the graph decides the source."""
        result = capture_decision(
            final_state=_state(),
            graph=_FakeGraph(observed=False),
            feed=feed,
            mode=MODE_REAL_INFERENCE_CAPTURE,
        )
        assert result.record.source == SOURCE_SYNTHETIC_FIXTURE
        assert result.record.is_genuine_inference is False
        assert result.mode == MODE_MOCK_TEST
        assert any("did not report a real model call" in n for n in result.notes)

    def test_an_observed_run_is_labelled_genuine(self, feed):
        result = capture_decision(
            final_state=_state(),
            graph=_FakeGraph(observed=True),
            feed=feed,
            mode=MODE_REAL_INFERENCE_CAPTURE,
        )
        assert result.record.source == SOURCE_LLM_INFERENCE
        assert result.record.is_genuine_inference is True
        assert result.mode == MODE_REAL_INFERENCE_CAPTURE
        assert result.record.model_provider == "groq"
        assert result.record.model_name == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Capture validation
# ---------------------------------------------------------------------------


class TestCaptureValidation:
    def test_a_capture_records_the_window_and_provenance(self, feed):
        result = capture_decision(
            final_state=_state(), graph=_FakeGraph(observed=True), feed=feed
        )
        record = result.record
        assert record.ticker == "SPY"
        assert record.as_of_date == "2022-03-06"
        assert record.window_end == record.as_of_date
        assert record.bar_count == 6
        assert len(record.data_window_hash) == 64
        assert record.config_hash == LLM_PROTOCOL.config_hash()
        assert record.window_start == "2022-03-01"
        assert result.window_reproducible is True

    def test_a_capture_invents_no_size_or_levels(self, feed):
        record = capture_decision(
            final_state=_state(), graph=_FakeGraph(observed=True), feed=feed
        ).record
        assert record.position_size is None
        assert record.stop_loss is None
        assert record.take_profit is None
        assert record.limit_price is None
        assert record.order_intent == "MARKET"
        assert record.time_in_force == "DAY"

    def test_the_captured_rating_maps_to_the_production_action(self, feed):
        for rating, action in (
            ("Buy", "BUY"),
            ("Overweight", "BUY"),
            ("Hold", "HOLD"),
            ("Underweight", "SELL"),
            ("Sell", "SELL"),
        ):
            record = capture_decision(
                final_state=_state(final_rating=rating),
                graph=_FakeGraph(observed=True),
                feed=feed,
            ).record
            assert record.action == action, rating

    def test_a_review_rating_is_refused_not_degraded_to_hold(self, feed):
        with pytest.raises(CaptureError, match="no tradeable action"):
            capture_decision(
                final_state=_state(final_rating="REVIEW"),
                graph=_FakeGraph(observed=True),
                feed=feed,
            )

    def test_a_missing_rating_is_refused(self, feed):
        with pytest.raises(CaptureError, match="final_rating"):
            capture_decision(
                final_state=_state(final_rating=""),
                graph=_FakeGraph(observed=True),
                feed=feed,
            )

    def test_a_missing_ticker_is_refused(self, feed):
        with pytest.raises(CaptureError, match="company_of_interest"):
            capture_decision(
                final_state=_state(company_of_interest=""),
                graph=_FakeGraph(observed=True),
                feed=feed,
            )

    def test_a_missing_trade_date_is_refused(self, feed):
        with pytest.raises(CaptureError, match="trade_date"):
            capture_decision(
                final_state=_state(trade_date=""),
                graph=_FakeGraph(observed=True),
                feed=feed,
            )

    def test_an_unknown_mode_is_refused(self, feed):
        with pytest.raises(CaptureError, match="unknown capture mode"):
            capture_decision(
                final_state=_state(),
                graph=_FakeGraph(observed=True),
                feed=feed,
                mode="NOT_A_MODE",
            )

    def test_a_ticker_outside_the_feed_captures_but_flags_unreproducible(self, feed):
        result = capture_decision(
            final_state=_state(company_of_interest="ZZZZ"),
            graph=_FakeGraph(observed=True),
            feed=feed,
        )
        assert result.window_reproducible is False
        assert result.record.bar_count == 0
        assert any("not reproducible" in n for n in result.notes)

    def test_the_captured_window_hashes_to_the_replay_window(self, feed):
        """A captured record's hash must match the replay's recomputation."""
        record = capture_decision(
            final_state=_state(), graph=_FakeGraph(observed=True), feed=feed
        ).record
        assert record.data_window_hash == window_hash(feed.history("SPY", "2022-03-06"))

    def test_a_captured_record_round_trips_through_json(self, feed):
        record = capture_decision(
            final_state=_state(), graph=_FakeGraph(observed=True), feed=feed
        ).record
        payload = json.loads(json.dumps(record.to_dict()))
        from tradingagents.llm_evaluation import DecisionRecord

        assert DecisionRecord.from_dict(payload) == record


# ---------------------------------------------------------------------------
# Observed-response parsing
# ---------------------------------------------------------------------------


class TestObservedDecision:
    def test_a_labelled_rating_is_read(self):
        rating, reasoning = observed_decision("**Rating**: Buy\n**Executive Summary**: go")
        assert rating == "Buy"
        assert "Rating" in reasoning

    def test_a_response_with_no_rating_is_refused(self):
        with pytest.raises(CaptureError, match="no readable rating"):
            observed_decision("I would rather not say.")

    def test_an_empty_response_is_refused(self):
        with pytest.raises(CaptureError, match="no text"):
            observed_decision("   ")

    def test_a_credential_shaped_value_is_scrubbed_from_the_excerpt(self):
        rating, reasoning = observed_decision(
            "**Rating**: Hold\nkey sk-abcdefghijklmnopqrstuvwxyz"
        )
        assert rating == "Hold"
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in reasoning
        assert "<redacted>" in reasoning


# ---------------------------------------------------------------------------
# Replay isolation and provenance separation
# ---------------------------------------------------------------------------


def _captured_set(feed, *, observed=True, sessions=("2022-03-06",)) -> DecisionRecordSet:
    decisions = tuple(
        capture_decision(
            final_state=_state(trade_date=s, company_of_interest="SPY"),
            graph=_FakeGraph(observed=observed),
            feed=feed,
        ).record
        for s in sessions
    )
    return DecisionRecordSet(
        strategy_id="capture-test",
        graph_version=LLM_PROTOCOL.graph_version,
        protocol_version=LLM_PROTOCOL.version,
        mode=decisions[0].source,
        created_at="2026-01-01T00:00:00+00:00",
        decisions=decisions,
    )


class TestReplayIsolation:
    def test_replay_never_constructs_an_llm_client(self, feed, monkeypatch):
        from tradingagents.llm_clients import factory

        def _boom(*a, **k):  # pragma: no cover - only reached on failure
            raise AssertionError("replay created an LLM client")

        monkeypatch.setattr(factory, "create_llm_client", _boom)
        strategy = ReplayStrategy(_captured_set(feed))
        # Building and calling the strategy must not reach the factory.
        assert strategy.name.startswith("llm_replay:")

    def test_a_tampered_window_hash_is_refused(self, feed):
        record_set = _captured_set(feed)
        record = record_set.decisions[0]
        from tradingagents.llm_evaluation import DecisionRecord

        tampered = DecisionRecord.from_dict(
            {**record.to_dict(), "data_window_hash": "0" * 64}
        )
        tampered_set = DecisionRecordSet(
            strategy_id="t",
            graph_version=LLM_PROTOCOL.graph_version,
            protocol_version=LLM_PROTOCOL.version,
            mode=tampered.source,
            created_at="2026-01-01T00:00:00+00:00",
            decisions=(tampered,),
        )
        strategy = ReplayStrategy(tampered_set)
        context = _context(feed, "2022-03-06")
        with pytest.raises(ReplayError, match="window mismatch"):
            strategy(context)

    def test_a_tampered_config_hash_is_refused(self, feed):
        record_set = _captured_set(feed)
        record = record_set.decisions[0]
        from tradingagents.llm_evaluation import DecisionRecord

        tampered = DecisionRecord.from_dict(
            {**record.to_dict(), "config_hash": "1" * 64}
        )
        tampered_set = DecisionRecordSet(
            strategy_id="t",
            graph_version=LLM_PROTOCOL.graph_version,
            protocol_version=LLM_PROTOCOL.version,
            mode=tampered.source,
            created_at="2026-01-01T00:00:00+00:00",
            decisions=(tampered,),
        )
        strategy = ReplayStrategy(
            tampered_set, expected_config_hash=LLM_PROTOCOL.config_hash()
        )
        context = _context(feed, "2022-03-06")
        with pytest.raises(ReplayError, match="config mismatch"):
            strategy(context)

    def test_a_matching_config_hash_replays(self, feed):
        record_set = _captured_set(feed)
        strategy = ReplayStrategy(
            record_set, expected_config_hash=LLM_PROTOCOL.config_hash()
        )
        signal = strategy(_context(feed, "2022-03-06"))
        assert signal is not None
        assert signal.action.value == "BUY"

    def test_a_genuine_and_a_fixture_set_have_distinct_provenance(self, feed):
        genuine = _captured_set(feed, observed=True)
        fixture = _captured_set(feed, observed=False)
        assert genuine.mode == SOURCE_LLM_INFERENCE
        assert genuine.is_genuine_inference is True
        assert fixture.mode == SOURCE_SYNTHETIC_FIXTURE
        assert fixture.is_genuine_inference is False

    def test_replaying_a_valid_record_is_deterministic(self, feed):
        record_set = _captured_set(feed)
        first = ReplayStrategy(record_set)(_context(feed, "2022-03-06"))
        second = ReplayStrategy(record_set)(_context(feed, "2022-03-06"))
        assert first.action == second.action
        assert first.rating == second.rating
        assert first.position_size == second.position_size

    def test_a_session_with_no_record_yields_no_signal(self, feed):
        record_set = _captured_set(feed)
        strategy = ReplayStrategy(record_set)
        assert strategy(_context(feed, "2022-03-07")) is None

    def test_signal_from_record_uses_the_production_contract(self, feed):
        record = _captured_set(feed).decisions[0]
        signal = signal_from_record(record)
        assert signal is not None
        assert signal.symbol == "SPY"
        assert signal.source == "llm_portfolio_manager"


class _LedgerViewStub:
    def position_quantity(self, symbol):  # pragma: no cover - not read here
        return 0.0


class _Context:
    def __init__(self, feed, session):
        self.feed = feed.as_of(session)
        self.session = session
        self.symbols = feed.symbols
        self.ledger = _LedgerViewStub()

    def history(self, symbol):
        return self.feed.history(symbol, self.session)


def _context(feed, session):
    return _Context(feed, session)


# ---------------------------------------------------------------------------
# Bounded smoke test (offline: no real request)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content):
        self.llm = _FakeLLM(content)

    def get_llm(self):
        return self.llm


class TestSmokeConfigGate:
    def test_no_credential_stops_before_any_request(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(SmokeBlocked, match="GROQ_API_KEY"):
            run_smoke(provider="groq")

    def test_the_blocker_names_the_variable_and_makes_no_request(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        try:
            run_smoke(provider="groq")
        except SmokeBlocked as exc:
            assert "GROQ_API_KEY" in str(exc)
            assert "no request" in str(exc).lower()


class TestSmokeInjectedClient:
    """An injected client is a test seam and must never be a genuine observation."""

    def _client(self, content):
        client = _FakeClient(content)
        return lambda **kwargs: client, client

    def test_a_valid_injected_response_is_a_fixture_not_genuine(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, client = self._client("**Rating**: Buy\n**Executive Summary**: go long")
        report = run_smoke(provider="groq", feed=feed, client_factory=factory, session="2022-03-06")
        assert report.succeeded is True
        assert report.genuine is False
        assert report.mode == MODE_MOCK_TEST
        assert report.requests_made == 1
        assert client.llm.calls == 1
        assert any("fixture" in n for n in report.notes)

    def test_an_injected_client_writes_a_fixture_record(self, monkeypatch, feed, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, _ = self._client("**Rating**: Hold\n**Executive Summary**: wait")
        out = tmp_path / "records.json"
        report = run_smoke(
            provider="groq",
            feed=feed,
            client_factory=factory,
            session="2022-03-06",
            record_out=out,
        )
        assert report.record_path is not None
        loaded = DecisionRecordSet.load(out)
        assert loaded.mode == SOURCE_SYNTHETIC_FIXTURE
        assert loaded.decisions[0].source == SOURCE_SYNTHETIC_FIXTURE

    def test_an_unreadable_response_fails_and_writes_nothing(self, monkeypatch, feed, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, _ = self._client("I decline to answer.")
        out = tmp_path / "records.json"
        report = run_smoke(
            provider="groq",
            feed=feed,
            client_factory=factory,
            session="2022-03-06",
            record_out=out,
        )
        assert report.succeeded is False
        assert report.status == "SMOKE FAILED"
        assert report.record_path is None
        assert not out.exists()

    def test_a_provider_error_is_reported_not_swallowed(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)

        def _boom(**kwargs):
            raise RuntimeError("provider unavailable")

        report = run_smoke(
            provider="groq", feed=feed, client_factory=_boom, session="2022-03-06"
        )
        assert report.succeeded is False
        assert report.status == "SMOKE FAILED"
        assert "provider unavailable" in report.error

    def test_the_report_never_carries_the_credential_value(self, monkeypatch, feed):
        secret = "g" * 40
        monkeypatch.setenv("GROQ_API_KEY", secret)
        factory, _ = self._client("**Rating**: Buy\n**Executive Summary**: go")
        report = run_smoke(
            provider="groq", feed=feed, client_factory=factory, session="2022-03-06"
        )
        blob = json.dumps(report.to_dict())
        assert secret not in blob
        assert report.api_key_env == "GROQ_API_KEY"

    def test_only_one_request_is_made(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, client = self._client("**Rating**: Buy\n**Executive Summary**: go")
        run_smoke(provider="groq", feed=feed, client_factory=factory, session="2022-03-06")
        assert client.llm.calls == 1


# ---------------------------------------------------------------------------
# build_record direct (shared by capture and smoke)
# ---------------------------------------------------------------------------


class TestBuildRecord:
    def test_it_hashes_the_frame_it_is_given(self, feed):
        from tradingagents.signals import action_for_rating

        frame = feed.history("SPY", "2022-03-06")
        record, reproducible = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=frame,
            protocol=LLM_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
        )
        assert reproducible is True
        assert record.data_window_hash == window_hash(frame)

    def test_an_empty_frame_is_recorded_as_unreproducible(self):
        from tradingagents.signals import action_for_rating

        record, reproducible = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=None,
            protocol=LLM_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
        )
        assert reproducible is False
        assert record.bar_count == 0
        assert record.data_snapshot_id.endswith("unavailable")


# ---------------------------------------------------------------------------
# Evaluation artifact: the capture-status block distinguishes the modes
# ---------------------------------------------------------------------------


class TestCaptureStatus:
    def test_the_blocked_artifact_claims_nothing(self, feed):
        from tradingagents.llm_evaluation import build_evaluation

        outcome = build_evaluation(feed=feed, include_reference_context=False)
        status = outcome.artifact["capture_status"]
        assert outcome.artifact["evaluation_mode"] == "blocked"
        assert status["genuine_inference_request_succeeded"] is False
        assert status["decision_record_captured"] is False
        assert status["historical_evaluation_allowed"] is False
        assert outcome.artifact["llm_results"] is None

    def test_a_fixture_replay_is_not_historical_evaluation(self, feed):
        from tradingagents.llm_evaluation import build_evaluation

        record_set = _captured_set(feed, observed=False)
        outcome = build_evaluation(
            feed=feed, record_set=record_set, include_reference_context=False
        )
        status = outcome.artifact["capture_status"]
        assert outcome.artifact["evaluation_mode"] == "REPLAY"
        assert status["genuine_inference_request_succeeded"] is False
        assert status["decision_record_captured"] is True
        assert status["historical_evaluation_allowed"] is False

    def test_a_genuine_capture_opens_only_the_restricted_scope(self, feed):
        """A genuine capture opens the price-only scope, never the full graph."""
        from tradingagents.llm_evaluation import build_evaluation

        record_set = _captured_set(feed, observed=True)
        outcome = build_evaluation(
            feed=feed, record_set=record_set, include_reference_context=False
        )
        status = outcome.artifact["capture_status"]
        assert outcome.artifact["evaluation_mode"] == "REAL_INFERENCE_CAPTURE"
        assert status["genuine_inference_request_succeeded"] is True
        assert status["inputs_reproducible"] is True
        assert status["historical_evaluation_scope"] == "price_only"
        assert status["historical_evaluation_allowed"] is True
        # The full graph stays blocked: news/sentiment/fundamentals are not
        # point-in-time safe for a historical run, capture or not.
        assert status["full_graph_historical_evaluation_allowed"] is False
        assert outcome.artifact["reproducibility"]["inference_determinism_claimed"] is False

    def test_a_fixture_capture_never_opens_historical_evaluation(self, feed):
        from tradingagents.llm_evaluation import build_evaluation

        record_set = _captured_set(feed, observed=False)
        outcome = build_evaluation(
            feed=feed, record_set=record_set, include_reference_context=False
        )
        status = outcome.artifact["capture_status"]
        assert status["historical_evaluation_allowed"] is False
        assert status["full_graph_historical_evaluation_allowed"] is False

    def test_the_artifact_never_carries_a_credential(self, monkeypatch, feed):
        from tradingagents.llm_evaluation import build_evaluation

        secret = "g" * 40
        monkeypatch.setenv("GROQ_API_KEY", secret)
        outcome = build_evaluation(feed=feed, include_reference_context=False)
        assert secret not in json.dumps(outcome.artifact)
