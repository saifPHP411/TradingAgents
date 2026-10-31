"""Prospective paper experiment: record LLM decisions, run them in paper, evaluate.

This package lets an LLM strategy make decisions *forward in time* and be
evaluated as the data arrives — without real money and without a live trading
path. It is the Phase 10 layer, and it is built entirely from pieces that already
exist and are already tested:

- the book, risk check, simulated fill and ledger come from
  :mod:`tradingagents.execution.paper`;
- the decision comes from :mod:`tradingagents.llm_evaluation.records` and
  :mod:`tradingagents.llm_evaluation.capture`;
- the reproducible input window comes from
  :mod:`tradingagents.llm_evaluation.snapshot`;
- the metrics come from :mod:`tradingagents.performance`.

What this package adds is only what was missing: a durable, hash-chained
**journal** of the experiment's history, a **config** frozen before results, a
**cycle** that separates decision, execution and evaluation time, and an honest
**report**.

What it deliberately does not add:

- no live broker execution, no order routing, no credentials in any artifact;
- no *silent* current-market claim: the default source is still the frozen
  historical dataset, and a vendor source is opt-in and declares its own
  provenance. A vendor that fails stops the run rather than silently substituting
  the frozen dataset;
- no fabricated decision — a missing credential records a *blocked* decision,
  and a test stub is labelled ``synthetic_fixture`` and can never be counted as
  genuine inference.
"""

from __future__ import annotations

from .config import ExperimentConfig, ExperimentConfigError
from .experiment import (
    PAPER_EXPERIMENT_SCHEMA_VERSION,
    STATE_BLOCKED,
    STATE_EVALUATED,
    STATE_EXECUTED,
    STATE_PENDING,
    STATE_UNEVALUABLE,
    DecisionState,
    ExperimentError,
    InferenceFn,
    InferenceUnavailable,
    PaperExperiment,
    SessionOutcome,
)
from .indicators import IndicatorError, IndicatorSet, compute_indicators, render_indicator_context
from .journal import (
    JOURNAL_SCHEMA_VERSION,
    ExperimentJournal,
    JournalEntry,
    JournalError,
)
from .protocol import PAPER_GRAPH_VERSION, PAPER_INFERENCE_PROMPT_VERSION, PAPER_PROTOCOL
from .providers import (
    AlphaVantageDailySource,
    FinnhubQuoteSource,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderStaleDataError,
    ProviderUnsupportedError,
    SourceSelection,
    SourceSelectionError,
    provider_registry,
    select_source,
)
from .report import REPORT_SCHEMA_VERSION, ExperimentReport, build_report
from .sources import (
    FrozenPriceSource,
    PriceSource,
    PriceSourceError,
    SessionBars,
    SourceProvenance,
    StaticPriceSource,
    current_market_source_available,
)

__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "PAPER_EXPERIMENT_SCHEMA_VERSION",
    "PAPER_GRAPH_VERSION",
    "PAPER_INFERENCE_PROMPT_VERSION",
    "PAPER_PROTOCOL",
    "REPORT_SCHEMA_VERSION",
    "STATE_BLOCKED",
    "STATE_EVALUATED",
    "STATE_EXECUTED",
    "STATE_PENDING",
    "STATE_UNEVALUABLE",
    "AlphaVantageDailySource",
    "DecisionState",
    "ExperimentConfig",
    "ExperimentConfigError",
    "ExperimentError",
    "ExperimentJournal",
    "ExperimentReport",
    "FinnhubQuoteSource",
    "FrozenPriceSource",
    "InferenceFn",
    "InferenceUnavailable",
    "IndicatorError",
    "IndicatorSet",
    "JournalEntry",
    "JournalError",
    "PaperExperiment",
    "PriceSource",
    "PriceSourceError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderRateLimitedError",
    "ProviderStaleDataError",
    "ProviderUnsupportedError",
    "SessionBars",
    "SessionOutcome",
    "SourceProvenance",
    "SourceSelection",
    "SourceSelectionError",
    "StaticPriceSource",
    "build_report",
    "compute_indicators",
    "current_market_source_available",
    "provider_registry",
    "render_indicator_context",
    "select_source",
]
