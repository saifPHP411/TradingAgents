"""Live, credential-gated verification of the vendor providers and the LLM seam.

These tests hit the **real** APIs. They are marked ``integration`` and skipped
unless the relevant credential is present in the environment, so the default
``pytest`` run (``-m "not integration"``) never needs a secret and CI stays
hermetic. Run them explicitly, with the keys set in the environment only::

    pytest -m integration tests/test_live_providers_integration.py -v

Rules these tests obey, per the project's safety boundary:

- No key value is ever printed, logged, stored, or written to an artifact; only
  the *name* of the environment variable is referenced, and only its presence
  (non-empty) is asserted.
- The requests are the minimum needed: **one** Alpha Vantage call, **one**
  Finnhub call, and **one** LLM inference. A vendor's free-tier quota is small,
  so an extra call is a real cost.
- No order is placed. The paper experiment is simulation-only.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    PaperExperiment,
    StaticPriceSource,
)
from tradingagents.paper_experiment.inference import ProviderInference
from tradingagents.paper_experiment.providers import (
    AlphaVantageDailySource,
    FinnhubQuoteSource,
)


def _present(name: str) -> bool:
    value = os.environ.get(name, "")
    return bool(value and value.strip())


requires_alpha_vantage = pytest.mark.skipif(
    not _present("ALPHA_VANTAGE_API_KEY"),
    reason="ALPHA_VANTAGE_API_KEY not set; skipping live vendor call",
)
requires_finnhub = pytest.mark.skipif(
    not _present("FINNHUB_API_KEY"),
    reason="FINNHUB_API_KEY not set; skipping live vendor call",
)
requires_groq = pytest.mark.skipif(
    not _present("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live LLM call",
)


@pytest.mark.integration
@requires_alpha_vantage
class TestAlphaVantageLive:
    """One real TIME_SERIES_DAILY call: authentication, OHLCV, provenance."""

    def test_the_free_daily_endpoint_returns_usable_bars(self):
        # The key is read from the environment inside the source; the test never
        # touches the value.
        source = AlphaVantageDailySource("IBM")

        sessions = source.sessions()
        assert sessions, "the live response carried no sessions"
        assert source.kind == "vendor_daily"

        # The last bar must be a real, finite, positive OHLC row.
        bar = source.bars("IBM", sessions[-1])
        assert bar is not None
        for name in ("open", "high", "low", "close"):
            value = getattr(bar, name)
            assert value == value and value not in (float("inf"), float("-inf"))
            assert value > 0

        # Provenance must be honest about the source and the bar date.
        provenance = source.provenance
        assert provenance.kind == "vendor_daily"
        assert provenance.last_bar_date is not None
        # A daily-bar dataset is historical even when its last bar is recent; the
        # source reports what it is, and the report layer decides readiness.
        assert isinstance(provenance.is_current_market, bool)

    def test_the_full_frame_has_no_malformed_bars(self):
        source = AlphaVantageDailySource("IBM")
        frame = source.frame("IBM")
        ohlc = frame[["open", "high", "low", "close"]]
        # The shared validator already refuses NaN/Inf at construction; this
        # re-asserts it on the live payload.
        assert not ohlc.isna().any().any()
        assert (ohlc > 0).all().all()
        assert (frame["high"] >= frame["low"]).all()


@pytest.mark.integration
@requires_finnhub
class TestFinnhubLive:
    """One real /quote call: authentication and a current-day price."""

    def test_the_free_quote_endpoint_returns_a_current_price(self):
        source = FinnhubQuoteSource("AAPL")
        observation = source.quote()

        assert observation.symbol == "AAPL"
        assert observation.price > 0
        assert observation.session  # an ISO date
        assert source.is_current_market is True
        assert source.kind == "vendor_quote"

    def test_a_quote_source_refuses_a_historical_request(self):
        from tradingagents.paper_experiment.providers import ProviderUnsupportedError

        source = FinnhubQuoteSource("AAPL")
        with pytest.raises(ProviderUnsupportedError):
            source.sessions()


@pytest.mark.integration
@requires_groq
class TestGroqLive:
    """One real LLM inference through the production path, then a paper cycle."""

    def test_one_genuine_inference_produces_a_decision_record(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame()})
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",),
            decision_frequency_sessions=1,
            evaluation_lag_sessions=2,
            provider="groq",
            model="openai/gpt-oss-20b",
        )
        inference = ProviderInference(provider="groq", model="openai/gpt-oss-20b")
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp", inference=inference
        )
        experiment.ensure_created()

        session = source.sessions()[0]
        outcome = experiment.process_session(session)

        # A genuine inference must have produced a decision, not a blocked one.
        assert outcome.decision_state != "blocked", outcome.blocked_reason
        assert outcome.decision_id is not None

        state = experiment.decisions[outcome.decision_id]
        assert state.source == "llm_inference"
        assert state.snapshot_id is not None
        assert state.data_window_hash

        # The record the production path rebuilt from the journal is a genuine
        # inference, and no order field was invented on its behalf.
        record = experiment._record_for(state)
        assert record.source == "llm_inference"
        assert record.model_provider == "groq"
        assert record.order_intent == "MARKET"
        assert record.position_size is None
        assert record.stop_loss is None
        assert record.limit_price is None

    def test_a_full_paper_cycle_places_no_real_order(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame(periods=6)})
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",),
            decision_frequency_sessions=1,
            evaluation_lag_sessions=2,
            provider="groq",
            model="openai/gpt-oss-20b",
        )
        inference = ProviderInference(provider="groq", model="openai/gpt-oss-20b")
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp", inference=inference
        )
        experiment.ensure_created()
        for session in source.sessions():
            experiment.process_session(session)

        # Simulation only: every execution is paper, and the ledger reconciles.
        assert experiment.session.mode.value == "PAPER"
        ledger = experiment.session.ledger
        assert ledger.cash >= 0
        assert ledger.equity > 0


def _frame(periods: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [100.0 + i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 1 for o in opens],
            "volume": [1_000_000.0] * periods,
        },
        index=index,
    )
