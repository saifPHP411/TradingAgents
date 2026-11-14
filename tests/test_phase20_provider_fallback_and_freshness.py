"""Phase 20: provider fallback, honest provenance, and market-data freshness.

Two gaps this phase closes, both about a paper run telling the truth about itself:

1. **Provider resilience.** The paper seam made one call to one configured
   provider. A missing credential or a provider error blocked the decision and
   that was that. The operator asked for Groq with a Gemini fallback, so the seam
   now tries the primary and, only if it cannot answer, a *named* fallback. The
   switch is recorded and — crucially — each decision records the provider that
   **actually answered**, not the one that was configured. A fallback that
   produced a decision must be visible on the record, or the provenance is a lie.

2. **Freshness.** A daily vendor source has a newest *session*, not a live price.
   Between sessions there is legitimately nothing new, and that must read as
   "up to date", not as a fault — while a source whose newest bar has gone stale
   must not read as current. The statuses are a closed vocabulary and each names
   the data condition rather than claiming an exchange clock.

These tests are written to catch a premise error, not to confirm the code. They
pin the properties that make the provenance trustworthy: a fallback is never
tried when the primary answers; a fallback never fabricates a second decision; a
fallback that fails too blocks rather than invents; the recorded provider is the
one that answered; and freshness is derived from recorded facts, never a fetch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.experiment import PaperExperiment
from tradingagents.paper_experiment.freshness import (
    MARKET_STATUS_HISTORICAL,
    MARKET_STATUS_NO_DATA,
    MARKET_STATUS_PENDING,
    MARKET_STATUS_STALE,
    MARKET_STATUS_UP_TO_DATE,
    MARKET_STATUSES,
    market_freshness,
)
from tradingagents.paper_experiment.inference import (
    InferenceUnavailable,
    ProviderInference,
    StubInference,
)
from tradingagents.paper_experiment.runner import build_inference, process_pending
from tradingagents.paper_experiment.sources import (
    SourceProvenance,
    StaticPriceSource,
)

SYMBOL = "SPY"


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content, *, error=None):
        self._content = content
        self._error = error

    def invoke(self, prompt):
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content, *, error=None):
        self._llm = _FakeLLM(content, error=error)

    def get_llm(self):
        return self._llm


_DECISION_TEXT = (
    "**Rating**: Buy\n"
    "**Supporting factors**: uptrend.\n"
    "**Opposing factors**: valuation.\n"
    "**Executive Summary**: accumulate.\n"
)


def _window(rows=30):
    index = pd.bdate_range("2026-01-02", periods=rows)
    epoch = pd.Timestamp("2026-01-01")
    opens = [100.0 + float((ts - epoch).days) for ts in index]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * rows,
        },
        index=index,
    )


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


def _sessions(start: str, end: str):
    return tuple(str(d)[:10] for d in pd.bdate_range(start, end))


def _experiment(tmp_path, sessions, *, inference=None, current=False, stale=None):
    source = StaticPriceSource({SYMBOL: _frame(sessions)}, name="alpha_vantage_daily")
    source.kind = "vendor_daily"
    source.provenance = SourceProvenance(
        name="alpha_vantage_daily",
        kind="vendor_daily",
        is_current_market=current,
        last_bar_date=sessions[-1],
        max_stale_days=stale,
    )
    config = ExperimentConfig.from_protocol(symbols=(SYMBOL,))
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=tmp_path / "exp",
        inference=inference or StubInference(rating="Buy"),
    )
    experiment.ensure_created()
    experiment.save_state()
    return experiment


@pytest.mark.unit
class TestProviderFallback:
    def _inference(self, *, primary, primary_error=None, fallback=None, fallback_error=None,
                   fallback_provider="google", fallback_model="gemini-3.8-flash"):
        calls: list[str] = []

        def factory(*, provider, model, **kwargs):
            calls.append(provider)
            if provider == primary:
                # A readable decision unless the test injects an error.
                content = _DECISION_TEXT if primary_error is None else None
                return _FakeClient(content, error=primary_error)
            content = fallback if fallback is not None else _DECISION_TEXT
            return _FakeClient(content, error=fallback_error)

        inference = ProviderInference(
            provider=primary,
            model="primary-model",
            client_factory=factory,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )
        return inference, calls

    def test_primary_answers_and_the_fallback_is_never_tried(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "present")
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, calls = self._inference(primary="groq", primary_error=None)
        rating, _ = inference(symbol=SYMBOL, session="2026-01-02", window=_window())
        assert rating == "Buy"
        assert calls == ["groq"]  # the fallback was not built at all
        assert inference.last_provider == "groq"
        assert inference.last_fallback_reason is None

    def test_primary_error_switches_to_the_named_fallback(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "present")
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, calls = self._inference(
            primary="groq", primary_error=RuntimeError("timeout"),
            fallback=_DECISION_TEXT, fallback_error=None,
        )
        rating, _ = inference(symbol=SYMBOL, session="2026-01-02", window=_window())
        assert rating == "Buy"
        assert calls == ["groq", "google"]  # primary first, then fallback
        # The record names the provider that *answered*, not the configured one.
        assert inference.last_provider == "google"
        assert inference.last_model == "gemini-3.8-flash"
        assert "timeout" in inference.last_fallback_reason

    def test_missing_primary_credential_uses_the_fallback(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, calls = self._inference(
            primary="groq", fallback=_DECISION_TEXT,
        )
        rating, _ = inference(symbol=SYMBOL, session="2026-01-02", window=_window())
        assert rating == "Buy"
        assert inference.last_provider == "google"
        # The primary was skipped without a call, so only the fallback was built.
        assert calls == ["google"]

    def test_every_provider_failing_blocks_rather_than_invents(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "present")
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, calls = self._inference(
            primary="groq", primary_error=RuntimeError("down"),
            fallback_error=RuntimeError("also down"),
        )
        with pytest.raises(InferenceUnavailable) as exc:
            inference(symbol=SYMBOL, session="2026-01-02", window=_window())
        assert calls == ["groq", "google"]
        assert "no provider produced a decision" in str(exc.value)
        # A total failure records no answering provider.
        assert inference.last_provider is None

    def test_no_fallback_configured_behaves_as_before(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "present")
        inference = ProviderInference(
            provider="groq", model="m",
            client_factory=lambda **kw: _FakeClient("x", error=RuntimeError("down")),
        )
        with pytest.raises(InferenceUnavailable):
            inference(symbol=SYMBOL, session="2026-01-02", window=_window())

    def test_a_fallback_never_produces_two_decisions(self, tmp_path, monkeypatch):
        """The first readable rating is returned; the fallback does not add one."""
        monkeypatch.setenv("GROQ_API_KEY", "present")
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, calls = self._inference(
            primary="groq", primary_error=RuntimeError("timeout"),
            fallback=_DECISION_TEXT,
        )
        sessions = _sessions("2026-10-01", "2026-10-30")
        experiment = _experiment(tmp_path, sessions, inference=inference)
        process_pending(experiment)
        # One decision per cadence session, never a duplicate from the fallback.
        assert len(experiment.decisions) == len(
            [s for s in sessions if s in experiment.processed_sessions][::5]
        ) or len(experiment.decisions) == 5
        assert calls.count("google") == len(experiment.decisions)

    def test_the_decision_records_the_answering_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "present")
        monkeypatch.setenv("GOOGLE_API_KEY", "present")
        inference, _ = self._inference(
            primary="groq", primary_error=RuntimeError("timeout"),
            fallback=_DECISION_TEXT,
        )
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions, inference=inference)
        process_pending(experiment)
        decisions = list(experiment.decisions.values())
        assert decisions
        for decision in decisions:
            # The record names the fallback that answered, not the configured groq.
            assert decision.provider == "google"
            assert decision.model == "gemini-3.8-flash"
            assert decision.prompt_version
            assert decision.graph_version


@pytest.mark.unit
class TestFallbackIsNotInTheConfigIdentity:
    def test_adding_a_fallback_does_not_change_the_config_hash(self):
        base = ExperimentConfig.from_protocol(symbols=(SYMBOL,))
        with_fallback = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), fallback_provider="google", fallback_model="gemini-3.8-flash"
        )
        assert base.config_hash() == with_fallback.config_hash()

    def test_a_state_written_before_the_fallback_field_still_loads(self, tmp_path):
        """Backward compatibility: an old config payload has no fallback keys."""
        config = ExperimentConfig.from_protocol(symbols=(SYMBOL,))
        payload = config.to_dict()
        payload.pop("fallback_provider")
        payload.pop("fallback_model")
        restored = ExperimentConfig.from_dict(payload)
        assert restored.fallback_provider is None
        assert restored.config_hash() == config.config_hash()

    def test_the_fallback_round_trips_through_the_config(self):
        config = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), fallback_provider="google", fallback_model="gemini-3.8-flash"
        )
        restored = ExperimentConfig.from_dict(config.to_dict())
        assert restored.fallback_provider == "google"
        assert restored.fallback_model == "gemini-3.8-flash"

    def test_build_inference_passes_the_fallback_through(self):
        config = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), fallback_provider="google", fallback_model="gemini-3.8-flash"
        )
        inference = build_inference(config, stub=False)
        assert inference.fallback_provider == "google"
        assert inference.fallback_model == "gemini-3.8-flash"


@pytest.mark.unit
class TestMarketFreshness:
    def test_a_frozen_dataset_is_historical_not_current(self, tmp_path):
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions, current=False, stale=None)
        fresh = market_freshness(experiment)
        assert fresh.market_status == MARKET_STATUS_HISTORICAL
        assert fresh.is_stale is False

    def test_an_unprocessed_current_source_is_pending(self, tmp_path):
        today = datetime.now(UTC).date()
        sessions = tuple(
            str(d)[:10] for d in pd.bdate_range(today - timedelta(days=6), today)
        )
        experiment = _experiment(tmp_path, sessions, current=True, stale=10)
        fresh = market_freshness(experiment)
        assert fresh.market_status == MARKET_STATUS_PENDING
        assert fresh.pending_sessions > 0

    def test_a_processed_current_source_is_up_to_date(self, tmp_path):
        today = datetime.now(UTC).date()
        sessions = tuple(
            str(d)[:10] for d in pd.bdate_range(today - timedelta(days=6), today)
        )
        experiment = _experiment(tmp_path, sessions, current=True, stale=10)
        process_pending(experiment)
        fresh = market_freshness(experiment)
        assert fresh.market_status == MARKET_STATUS_UP_TO_DATE
        assert "no newer market session" in fresh.reason
        assert fresh.is_stale is False

    def test_a_bar_beyond_the_freshness_rule_is_stale(self, tmp_path):
        today = datetime.now(UTC).date()
        sessions = tuple(
            str(d)[:10] for d in pd.bdate_range(today - timedelta(days=40), today - timedelta(days=30))
        )
        experiment = _experiment(tmp_path, sessions, current=False, stale=10)
        fresh = market_freshness(experiment)
        assert fresh.market_status == MARKET_STATUS_STALE
        assert fresh.is_stale is True
        assert fresh.data_age_days > 10

    def test_every_status_is_in_the_closed_vocabulary(self, tmp_path):
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions)
        assert market_freshness(experiment).market_status in MARKET_STATUSES

    def test_freshness_reports_the_age_against_a_supplied_reference(self, tmp_path):
        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions, current=True, stale=3650)
        reference = datetime(2026, 10, 20, tzinfo=UTC)
        fresh = market_freshness(experiment, now=reference)
        assert fresh.data_age_days == 12
        assert fresh.reference_date == "2026-10-20"

    def test_the_freshness_block_is_json_safe(self, tmp_path):
        import json

        sessions = _sessions("2026-10-01", "2026-10-08")
        experiment = _experiment(tmp_path, sessions, current=True, stale=10)
        payload = market_freshness(experiment).to_dict()
        assert json.loads(json.dumps(payload)) == payload


@pytest.mark.unit
class TestNoDataSource:
    def test_a_source_with_no_sessions_reports_no_data(self, tmp_path):
        """A source carrying no sessions must report no_data, never a false currency."""
        sessions = _sessions("2026-10-01", "2026-10-01")
        experiment = _experiment(tmp_path, sessions)
        # Drop the only session, leaving a source with an empty session set. This
        # stands in for a provider that returned nothing usable.
        experiment.source._sessions = ()
        experiment.source._session_set = set()
        fresh = market_freshness(experiment)
        assert fresh.market_status == MARKET_STATUS_NO_DATA
        assert fresh.latest_available_session is None
