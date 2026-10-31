"""The paper experiment's real-inference prompt must carry the price window.

The defect these tests pin: ``ProviderInference`` asked the model for a
directional call while telling it "given no further data". A directional question
with no data has one safe answer, ``Hold``, so a live run returned 20 HOLD out of
20 decisions and never submitted an order. The window the experiment already
builds was used only *after* the rating came back, to hash and validate the
record, so the record attested to an input the model never saw.

These tests are deterministic and offline: the provider client is a fake that
records the prompt it was handed, so nothing here calls a network or needs a key.
They assert the *input*, which is the thing that was wrong, rather than a rating
the model happens to return.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    PaperExperiment,
    StaticPriceSource,
)
from tradingagents.paper_experiment.experiment import InferenceUnavailable
from tradingagents.paper_experiment.inference import (
    PAPER_INFERENCE_PROMPT_VERSION,
    WINDOW_BARS,
    ProviderInference,
    StubInference,
    _render_window,
)


def _frame(periods: int = 40, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
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


class _RecordingLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.prompts: list[str] = []

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self._content})()

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]


class _RecordingClient:
    def __init__(self, content: str) -> None:
        self.llm = _RecordingLLM(content)

    def get_llm(self):
        return self.llm


def _factory(content: str = "**Rating**: Buy\n**Executive Summary**: go"):
    client = _RecordingClient(content)
    return (lambda **kwargs: client), client


@pytest.fixture
def credential(monkeypatch):
    """A present-but-fake credential, so the seam reaches the (fake) client."""
    monkeypatch.setenv("GROQ_API_KEY", "g" * 32)


# ---------------------------------------------------------------------------
# The request carries the window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRequestCarriesTheWindow:
    def test_the_prompt_contains_the_price_history(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame()
        inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)

        prompt = client.llm.last_prompt
        assert "date,open,high,low,close,volume" in prompt
        # A real bar is present, with its date and a price from the window.
        assert str(frame.index[-1])[:10] in prompt
        assert "139" in prompt  # the last close in the ascending fixture

    def test_the_prompt_never_says_there_is_no_data(self, credential):
        """The exact regression: the old prompt told the model to decide blind."""
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame()
        inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)
        assert "given no further data" not in client.llm.last_prompt.lower()

    def test_the_window_ends_at_the_decision_session(self, credential):
        """Point-in-time: no bar after the decision date may reach the model."""
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame()
        session = str(frame.index[10])[:10]
        bounded = frame[frame.index <= pd.Timestamp(session)]
        inference(symbol="SPY", session=session, window=bounded)

        prompt = client.llm.last_prompt
        for later in frame.index[11:]:
            assert str(later)[:10] not in prompt
        assert str(frame.index[10])[:10] in prompt

    def test_the_window_is_capped(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame(periods=200)
        inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)

        rows = [
            line
            for line in client.llm.last_prompt.splitlines()
            if line[:4].isdigit() and "," in line
        ]
        assert len(rows) == WINDOW_BARS

    def test_the_same_window_renders_the_same_prompt(self):
        frame = _frame()
        first = _render_window(frame)
        second = _render_window(frame.copy())
        assert first == second

    def test_the_prompt_version_is_bumped_past_the_blind_one(self):
        assert PAPER_INFERENCE_PROMPT_VERSION != "paper-inference-1"
        assert PAPER_INFERENCE_PROMPT_VERSION != "paper-inference-2"


# ---------------------------------------------------------------------------
# The request carries the computed indicator evidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRequestCarriesTheIndicatorEvidence:
    def test_the_prompt_states_the_indicators(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame(periods=260)
        inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)

        prompt = client.llm.last_prompt
        assert "Indicators for SPY" in prompt
        for label in ("RSI", "MACD", "Bollinger", "ATR", "trend"):
            assert label in prompt, label

    def test_the_prompt_asks_for_supporting_and_opposing_factors(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame(periods=260)
        inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)

        prompt = client.llm.last_prompt
        assert "Supporting factors" in prompt
        assert "Opposing factors" in prompt

    def test_the_indicator_window_is_the_same_bounded_window(self, credential):
        """No later bar may reach the indicator block either."""
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame(periods=260)
        session = str(frame.index[100])[:10]
        bounded = frame[frame.index <= pd.Timestamp(session)]
        inference(symbol="SPY", session=session, window=bounded)

        prompt = client.llm.last_prompt
        for later in frame.index[101:]:
            assert str(later)[:10] not in prompt

    def test_a_malformed_window_blocks_before_any_indicator_is_rendered(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame(periods=260)
        frame.iloc[-1, frame.columns.get_loc("close")] = math.nan
        with pytest.raises(InferenceUnavailable):
            inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)
        assert client.llm.calls == 0


# ---------------------------------------------------------------------------
# The record's provenance is the seam that actually ran, not the graph's
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRecordProvenanceIsThePaperSeam:
    def _experiment(self, tmp_path, inference):
        source = StaticPriceSource({"SPY": _frame(periods=60)})
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
        )
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp", inference=inference
        )
        experiment.ensure_created()
        return experiment, source

    def test_the_record_names_the_paper_prompt_not_the_graph_prompt(self, tmp_path):
        experiment, source = self._experiment(tmp_path, StubInference(rating="Buy"))
        outcome = experiment.process_session(source.sessions()[0])
        record = experiment._record_for(experiment.decisions[outcome.decision_id])

        assert record.prompt_version == PAPER_INFERENCE_PROMPT_VERSION
        assert record.prompt_version != "tradingagents-default"

    def test_the_record_does_not_claim_the_agent_graph_ran(self, tmp_path):
        experiment, source = self._experiment(tmp_path, StubInference(rating="Buy"))
        outcome = experiment.process_session(source.sessions()[0])
        record = experiment._record_for(experiment.decisions[outcome.decision_id])

        assert record.graph_version == "paper-single-call"
        assert record.graph_version != "tradingagents-graph"

    def test_the_config_carries_the_paper_protocol_identity(self):
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
        )
        from tradingagents.paper_experiment.protocol import PAPER_PROTOCOL

        assert config.protocol_version == PAPER_PROTOCOL.version
        assert config.protocol_config_hash == PAPER_PROTOCOL.config_hash()


# ---------------------------------------------------------------------------
# A malformed window is refused, not rendered
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAMalformedWindowIsRefused:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_price_is_refused_and_no_request_is_made(self, credential, bad):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame()
        frame.iloc[-1, frame.columns.get_loc("close")] = bad
        with pytest.raises(InferenceUnavailable):
            inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)
        # Fail closed: a bad window must not become a request, and so cannot
        # become a fill or a fabricated decision.
        assert client.llm.calls == 0

    def test_a_missing_column_is_refused(self, credential):
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        frame = _frame().drop(columns=["volume"])
        with pytest.raises(InferenceUnavailable):
            inference(symbol="SPY", session=str(frame.index[-1])[:10], window=frame)
        assert client.llm.calls == 0

    def test_an_absent_window_still_makes_one_request(self, credential):
        """A session with no window is a request with a stated absence, not a crash."""
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        inference(symbol="SPY", session="2024-01-05", window=None)
        assert client.llm.calls == 1
        assert "No price history" in client.llm.last_prompt


# ---------------------------------------------------------------------------
# The experiment hands the window to the seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheExperimentHandsTheWindowToTheSeam:
    def _experiment(self, tmp_path, source, inference):
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
        )
        return PaperExperiment(
            source=source, config=config, root=tmp_path / "exp", inference=inference
        )

    def test_the_stub_receives_the_point_in_time_window(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame()})
        stub = StubInference(rating="Buy")
        experiment = self._experiment(tmp_path, source, stub)
        experiment.ensure_created()

        session = source.sessions()[5]
        experiment.process_session(session)

        assert stub.calls == 1
        assert stub.last_window is not None
        assert list(stub.last_window.index) == list(
            pd.to_datetime(list(source.sessions()[:6]))
        )
        # The window the model sees ends at the decision session.
        assert str(stub.last_window.index[-1])[:10] == session

    def test_the_window_the_model_sees_matches_the_recorded_snapshot(self, tmp_path):
        """The input shown to the model is the window the record attests to."""
        source = StaticPriceSource({"SPY": _frame()})
        stub = StubInference(rating="Buy")
        experiment = self._experiment(tmp_path, source, stub)
        experiment.ensure_created()

        session = source.sessions()[5]
        experiment.process_session(session)

        snapshot = experiment._snapshot("SPY", session)
        assert snapshot is not None
        assert list(stub.last_window.index) == list(snapshot.to_frame().index)

    def test_a_malformed_window_blocks_rather_than_fabricates(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame()})
        factory, client = _factory()
        inference = ProviderInference(
            provider="groq", model="openai/gpt-oss-20b", client_factory=factory
        )
        experiment = self._experiment(tmp_path, source, inference)
        experiment.ensure_created()

        # Force a non-finite close into the source's window for this session.
        frame = source.frame("SPY").copy()
        frame.iloc[-1, frame.columns.get_loc("close")] = math.nan
        source._frames["SPY"] = frame  # noqa: SLF001 - a deliberately corrupt source

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("GROQ_API_KEY", "g" * 32)
            outcome = experiment.process_session(source.sessions()[-1])

        assert outcome.decision_state == "blocked"
        # The model is never asked to decide on a window that cannot be anchored.
        assert client.llm.calls == 0
        assert experiment.session.ledger.cash == experiment.session.ledger.initial_cash
        assert experiment.session.ledger.open_positions() == {}
