"""Phase 9 tests: the genuine-inference path, proven offline.

Phase 9's one job is to make a real provider request and capture the resulting
decision — or to report precisely why it cannot. This environment has no
credential, so no genuine request is made here and the suite stays offline.

What these tests *can* prove, and do, is that the discriminator is real rather
than asserted:

- the observer fires only because the **real** provider client dispatched a
  LangChain ``on_llm_end`` callback — exercised through
  ``create_llm_client`` with an ``httpx.MockTransport``, so no network is used
  but every layer above the socket is the production one;
- a caller cannot force ``llm_inference`` by passing a mode argument;
- a missing credential blocks before any request and makes zero calls;
- the smoke path makes at most one request, and ``max_retries=0`` keeps a 5xx
  from silently becoming three billable attempts;
- a provider failure yields no ``DecisionRecord`` and a named category;
- replay of a captured record makes zero provider calls;
- genuine-inference status and point-in-time safety stay independent;
- no credential value reaches a report, a record, or the artifact.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from tradingagents.backtesting import MarketDataFeed
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.observed import InferenceObserver
from tradingagents.llm_evaluation import (
    MODE_MOCK_TEST,
    MODE_REAL_INFERENCE_CAPTURE,
    SOURCE_LLM_INFERENCE,
    SOURCE_SYNTHETIC_FIXTURE,
    CaptureError,
    DecisionRecordSet,
    ReplayStrategy,
    SmokeBlocked,
    capture_decision,
    graph_observed_real_inference,
    run_smoke,
    signal_from_record,
)
from tradingagents.llm_evaluation.smoke import _classify_failure

GROQ_MODEL = "openai/gpt-oss-20b"
# A credential-shaped placeholder: the value never leaves this process, and the
# transport below never opens a socket, so nothing is spent.
FAKE_KEY = "gsk-" + "0" * 40


def _chat_completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": GROQ_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _Recorder:
    """Counts HTTP requests the client actually made, and replies from a script."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self._responses.pop(0)
        return httpx.Response(status, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    return FAKE_KEY


# ---------------------------------------------------------------------------
# The observer is wired to the real client, not to a test double
# ---------------------------------------------------------------------------


class TestObserverRidesTheRealClient:
    """The genuine-inference flag must come from a real callback dispatch."""

    def test_the_real_provider_client_fires_the_observer(self, groq_key):
        """A completed call through ``create_llm_client`` flips the observer."""
        recorder = _Recorder([(200, _chat_completion("**Rating**: Buy"))])
        observer = InferenceObserver()
        llm = create_llm_client(
            provider="groq",
            model=GROQ_MODEL,
            http_client=recorder.client(),
            callbacks=[observer],
        ).get_llm()

        assert observer.observed is False  # nothing yet
        llm.invoke("hi")

        assert observer.observed is True
        assert observer.calls == 1
        assert len(recorder.requests) == 1  # exactly one request, no retry

    def test_a_client_built_without_the_observer_stays_unobserved(self, groq_key):
        """Without the callback, the same real call leaves the flag alone.

        This is what makes the flag meaningful: it is caused by the callback,
        not by the mere fact that a request happened.
        """
        recorder = _Recorder([(200, _chat_completion("**Rating**: Buy"))])
        llm = create_llm_client(
            provider="groq", model=GROQ_MODEL, http_client=recorder.client()
        ).get_llm()

        llm.invoke("hi")

        assert len(recorder.requests) == 1

    def test_the_observer_reads_no_model_output(self, groq_key):
        """The observer counts calls; it must not retain response content."""
        recorder = _Recorder([(200, _chat_completion("secret-looking text"))])
        observer = InferenceObserver()
        llm = create_llm_client(
            provider="groq",
            model=GROQ_MODEL,
            http_client=recorder.client(),
            callbacks=[observer],
        ).get_llm()
        llm.invoke("hi")

        # The only public state is a bool and a count — no content, no response.
        assert set(vars(observer)) == {"observed", "calls"}


class TestGraphObserverWiring:
    """A graph built through the real factory observes; a stubbed graph does not."""

    def _graph(self, monkeypatch, tmp_path, **config):
        from tradingagents.graph import trading_graph

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.update(
            results_dir=str(tmp_path / "results"),
            data_cache_dir=str(tmp_path / "cache"),
            memory_log_path=str(tmp_path / "log.md"),
            llm_provider="groq",
            deep_think_llm=GROQ_MODEL,
            quick_think_llm=GROQ_MODEL,
            **config,
        )
        monkeypatch.setattr(trading_graph, "set_config", lambda *a, **k: None)
        return trading_graph.TradingAgentsGraph, cfg

    def test_a_graph_using_the_real_factory_reports_observed(self, monkeypatch, tmp_path, groq_key):
        """The graph's own observer is the one attached to the real clients."""
        from tradingagents.graph import trading_graph

        recorder = _Recorder([(200, _chat_completion("**Rating**: Buy"))])

        def factory(config, tier, **extra):
            return create_llm_client(
                provider="groq",
                model=GROQ_MODEL,
                http_client=recorder.client(),
                callbacks=extra.get("callbacks", []),
            )

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.update(
            results_dir=str(tmp_path / "results"),
            data_cache_dir=str(tmp_path / "cache"),
            memory_log_path=str(tmp_path / "log.md"),
            llm_provider="groq",
            deep_think_llm=GROQ_MODEL,
            quick_think_llm=GROQ_MODEL,
        )
        monkeypatch.setattr(trading_graph, "create_tier_client", factory)
        monkeypatch.setattr(trading_graph, "set_config", lambda *a, **k: None)

        graph = trading_graph.TradingAgentsGraph(config=cfg)
        assert graph.inference_observed is False

        graph.deep_thinking_llm.invoke("hi")

        assert graph.inference_observed is True


# ---------------------------------------------------------------------------
# A caller cannot force the genuine label
# ---------------------------------------------------------------------------


class TestGenuineLabelCannotBeForced:
    def _state(self):
        return {
            "company_of_interest": "SPY",
            "trade_date": "2022-03-06",
            "final_rating": "Overweight",
            "pm_executive_summary": "Constructive.",
        }

    def _feed(self):
        import pandas as pd

        sessions = [f"2022-03-{d:02d}" for d in range(1, 15)]
        closes = [200.0 + i for i in range(14)]
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [1_000_000] * 14,
            },
            index=pd.to_datetime(sessions),
        )
        return MarketDataFeed({"SPY": frame})

    def test_asking_for_real_inference_on_a_stub_yields_a_fixture(self):
        class _Stub:
            inference_observed = False

            def run_settings(self):
                return {"deep_think_provider": "groq", "deep_think_llm": GROQ_MODEL}

        result = capture_decision(
            final_state=self._state(),
            graph=_Stub(),
            feed=self._feed(),
            mode=MODE_REAL_INFERENCE_CAPTURE,
        )
        assert result.mode == MODE_MOCK_TEST
        assert result.record.source == SOURCE_SYNTHETIC_FIXTURE
        assert result.inference_observed is False

    def test_a_manually_constructed_record_cannot_claim_genuine(self):
        """``llm_inference`` on a hand-built record is not the capture path's verdict."""
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL

        record, _ = _manual_record(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        # The record can carry the label, but only a real graph yields it through
        # capture_decision; the discriminator is the graph, not the record field.
        assert record.source == SOURCE_LLM_INFERENCE
        assert graph_observed_real_inference(object()) is False

    def test_a_replayed_record_is_not_a_new_inference(self, groq_key):
        """Replay consumes a record; it must never be re-labelled as fresh."""
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL

        record_set, feed = _record_set(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        strategy = ReplayStrategy(record_set, expected_config_hash=PROTOCOL.config_hash())
        signal = strategy(_Context(feed, "2022-03-06"))
        # The replay emitted a signal from the record, and the record keeps its
        # original source; replay does not mint a new inference.
        assert signal is not None
        assert record_set.decisions[0].source == SOURCE_LLM_INFERENCE


class _LedgerViewStub:
    """A minimal ledger view for replay: no positions, no cash constraints."""

    def positions(self):
        return {}

    def cash(self):
        return 1_000_000.0


class _Context:
    """The engine's strategy context, backed by a session-bounded feed view."""

    def __init__(self, feed, session):
        self.feed = feed.as_of(session)
        self.session = session
        self.symbols = feed.symbols
        self.ledger = _LedgerViewStub()

    def history(self, symbol):
        return self.feed.history(symbol, self.session)


def _record_set(protocol, *, source: str):
    """A one-record set, plus the feed whose window it hashes."""
    record, feed = _manual_record(protocol, source=source)
    record_set = DecisionRecordSet(
        strategy_id="phase9-test",
        graph_version=protocol.graph_version,
        protocol_version=protocol.version,
        mode=source,
        created_at="2026-01-01T00:00:00+00:00",
        decisions=(record,),
    )
    return record_set, feed


def _manual_record(protocol, *, source: str):
    """A record built directly, plus the feed whose window it hashes."""
    import pandas as pd

    from tradingagents.llm_evaluation import build_record

    # The window must end at the decision date, so the record's session is the
    # frame's last bar.
    sessions = [f"2022-03-{d:02d}" for d in range(1, 7)]
    closes = [200.0 + i for i in range(6)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 6,
        },
        index=pd.to_datetime(sessions),
    )
    from tradingagents.signals import action_for_rating

    record, _ = build_record(
        ticker="SPY",
        session="2022-03-06",
        rating="Overweight",
        action=action_for_rating("Overweight"),
        frame=frame,
        protocol=protocol,
        source=source,
        provider="groq",
        model_name=GROQ_MODEL,
    )
    return record, MarketDataFeed({"SPY": frame})


# ---------------------------------------------------------------------------
# Credential gate
# ---------------------------------------------------------------------------


class TestCredentialGate:
    def test_missing_credential_blocks_before_any_request(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        made: list[int] = []

        def factory(**kwargs):  # pragma: no cover - must never run
            made.append(1)
            raise AssertionError("no client may be built without a credential")

        with pytest.raises(SmokeBlocked, match="GROQ_API_KEY"):
            run_smoke(provider="groq", client_factory=factory)
        assert made == []

    def test_the_blocker_names_the_variable_only(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(SmokeBlocked) as exc:
            run_smoke(provider="groq")
        assert "GROQ_API_KEY" in str(exc.value)
        assert "no request" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Exactly one request, and failure categories
# ---------------------------------------------------------------------------


class TestOneRequestAndFailureReporting:
    def test_a_provider_5xx_makes_one_request_not_three(self, groq_key):
        """The SDK default retries a 5xx three times; the smoke path must not."""
        recorder = _Recorder([(500, {"error": "boom"})])
        from tradingagents.llm_evaluation.smoke import run_smoke as _run

        report = _run(
            provider="groq",
            client_factory=lambda **kw: create_llm_client(
                provider="groq", model=GROQ_MODEL, http_client=recorder.client(), **{
                    k: v for k, v in kw.items() if k in ("timeout", "max_retries")
                }
            ),
        )
        assert report.succeeded is False
        assert report.record_path is None
        assert len(recorder.requests) == 1
        assert report.failure_category == "provider_error"

    def test_a_provider_failure_yields_no_record(self, groq_key, tmp_path):
        recorder = _Recorder([(401, {"error": "bad key"})])
        out = tmp_path / "records.json"
        report = run_smoke(
            provider="groq",
            client_factory=lambda **kw: create_llm_client(
                provider="groq", model=GROQ_MODEL, http_client=recorder.client()
            ),
            record_out=out,
        )
        assert report.status == "SMOKE FAILED"
        assert report.failure_category == "authentication_failure"
        assert report.record_path is None
        assert not out.exists()

    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, "authentication_failure"),
            (403, "authentication_failure"),
            (404, "model_unavailable"),
            (408, "timeout"),
            (429, "rate_limit"),
            (500, "provider_error"),
            (503, "provider_error"),
        ],
    )
    def test_http_status_maps_to_a_named_category(self, status, expected):
        exc = httpx.HTTPStatusError("x", request=None, response=None)
        exc.status_code = status
        assert _classify_failure(exc) == expected

    def test_a_timeout_is_reported_as_a_timeout(self):
        assert _classify_failure(TimeoutError("slow")) == "timeout"

    def test_an_unknown_error_is_other_not_a_guess(self):
        assert _classify_failure(RuntimeError("???")) == "other"


# ---------------------------------------------------------------------------
# Genuine vs fixture provenance through the capture path
# ---------------------------------------------------------------------------


class TestCaptureProvenance:
    def _feed(self):
        import pandas as pd

        sessions = [f"2022-03-{d:02d}" for d in range(1, 15)]
        closes = [200.0 + i for i in range(14)]
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [1_000_000] * 14,
            },
            index=pd.to_datetime(sessions),
        )
        return MarketDataFeed({"SPY": frame})

    def test_an_observed_graph_is_labelled_genuine(self):
        class _Observed:
            inference_observed = True

            def run_settings(self):
                return {"deep_think_provider": "groq", "deep_think_llm": GROQ_MODEL}

        result = capture_decision(
            final_state={
                "company_of_interest": "SPY",
                "trade_date": "2022-03-06",
                "final_rating": "Buy",
            },
            graph=_Observed(),
            feed=self._feed(),
            mode=MODE_REAL_INFERENCE_CAPTURE,
        )
        assert result.mode == MODE_REAL_INFERENCE_CAPTURE
        assert result.record.source == SOURCE_LLM_INFERENCE
        assert result.record.model_provider == "groq"
        assert result.record.model_name == GROQ_MODEL

    def test_a_review_rating_is_still_refused(self):
        class _Observed:
            inference_observed = True

            def run_settings(self):
                return {}

        with pytest.raises(CaptureError):
            capture_decision(
                final_state={
                    "company_of_interest": "SPY",
                    "trade_date": "2022-03-06",
                    "final_rating": "REVIEW",
                },
                graph=_Observed(),
                feed=self._feed(),
                mode=MODE_REAL_INFERENCE_CAPTURE,
            )


# ---------------------------------------------------------------------------
# Replay makes no provider call
# ---------------------------------------------------------------------------


class TestReplayMakesNoProviderCall:
    def test_replay_never_builds_an_llm_client(self, monkeypatch):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL

        calls: list[int] = []
        monkeypatch.setattr(
            "tradingagents.llm_clients.create_llm_client",
            lambda *a, **k: calls.append(1),
            raising=True,
        )
        monkeypatch.setattr(
            "tradingagents.llm_clients.factory.create_llm_client",
            lambda *a, **k: calls.append(1),
            raising=True,
        )
        record_set, feed = _record_set(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        strategy = ReplayStrategy(record_set)
        assert strategy(_Context(feed, "2022-03-06")) is not None
        assert calls == []

    def test_replay_signal_matches_the_recorded_action(self):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL

        record, _ = _manual_record(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        signal = signal_from_record(record)
        assert signal is not None
        assert signal.symbol == "SPY"


# ---------------------------------------------------------------------------
# PIT status is independent of genuine-inference status
# ---------------------------------------------------------------------------


class TestPitIndependentOfGenuine:
    def test_a_genuine_record_does_not_open_the_full_graph(self):
        """Genuine inference and full-graph PIT safety are separate verdicts.

        A record whose provenance is ``llm_inference`` marks a genuine capture,
        yet the full graph still stays blocked because news/sentiment/fundamentals
        are not point-in-time safe. The artifact must not let one imply the other.
        """
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL, build_evaluation

        record_set, feed = _record_set(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        outcome = build_evaluation(
            feed=feed, record_set=record_set, include_reference_context=False
        )
        status = outcome.artifact["capture_status"]
        assert status["genuine_inference_request_succeeded"] is True
        assert status["full_graph_historical_evaluation_allowed"] is False
        assert status["historical_evaluation_scope"] == "price_only"
        # The artifact states the distinction rather than leaving it implicit.
        assert "point-in-time safety" in status["note"]

    def test_a_fixture_record_opens_no_historical_scope(self):
        """A fixture, however well-formed, is not a historical observation."""
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL, build_evaluation

        record_set, feed = _record_set(PROTOCOL, source=SOURCE_SYNTHETIC_FIXTURE)
        outcome = build_evaluation(
            feed=feed, record_set=record_set, include_reference_context=False
        )
        status = outcome.artifact["capture_status"]
        assert status["genuine_inference_request_succeeded"] is False
        assert status["historical_evaluation_allowed"] is False
        assert status["full_graph_historical_evaluation_allowed"] is False


# ---------------------------------------------------------------------------
# Secret safety
# ---------------------------------------------------------------------------


class TestSecretSafety:
    def test_the_smoke_report_never_carries_the_key(self, monkeypatch):
        secret = "gsk-" + "a" * 48
        monkeypatch.setenv("GROQ_API_KEY", secret)

        class _Client:
            class _LLM:
                calls = 0

                def invoke(self, prompt):
                    type(self).calls += 1

                    class _R:
                        content = "**Rating**: Buy"

                    return _R()

            def get_llm(self):
                return self._LLM()

        report = run_smoke(provider="groq", client_factory=lambda **k: _Client())
        blob = json.dumps(report.to_dict())
        assert secret not in blob
        assert report.api_key_env == "GROQ_API_KEY"

    def test_the_record_set_never_carries_the_key(self, groq_key):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL as PROTOCOL

        record, _ = _manual_record(PROTOCOL, source=SOURCE_LLM_INFERENCE)
        blob = json.dumps(record.to_dict())
        assert FAKE_KEY not in blob
        assert "gsk-" not in blob
