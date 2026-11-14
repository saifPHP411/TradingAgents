"""The frozen LLM-evaluation protocol: declared before any result is seen.

The reference baseline (``tradingagents/baseline/protocol.py``) already fixes the
universe, period, costs, risk limits and execution assumptions for the simulated
system. The LLM strategy is evaluated on **exactly those**, so the two are
directly comparable: same instruments, same dates, same capital, same costs,
same fill timing. This module adds only what is specific to evaluating a language
model, and it is frozen for the same reason -- so the evaluation cannot be tuned
after seeing the test window.

What this protocol can and cannot compare is stated here, not discovered later.
The frozen dataset carries daily OHLCV only. The LLM graph is built to read news,
sentiment, fundamentals and macro as well as prices, and those sources are not in
the frozen dataset. A price-only LLM run is therefore a *restricted* evaluation:
it measures the pipeline on the one input the baseline and the LLM share, and it
does not measure the news/sentiment agents at all. That gap is recorded, not
hidden, and it is why a price-only result is reported as a lower bound on what
the full pipeline could do rather than as the full pipeline's result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tradingagents.baseline.protocol import BaselineProtocol

PROTOCOL_VERSION = "1.0.0"

# The evaluation inherits the baseline's universe, period, costs, risk and
# execution timing verbatim. Anything that differs would break comparability.
_REFERENCE = BaselineProtocol()


@dataclass(frozen=True)
class LLMEvaluationProtocol:
    """The frozen specification of the LLM evaluation."""

    version: str = PROTOCOL_VERSION

    # --- Inherited from the reference baseline ------------------------------
    universe: tuple[str, ...] = _REFERENCE.universe
    benchmark: str = _REFERENCE.benchmark
    start: str = _REFERENCE.start
    end: str = _REFERENCE.end
    timeframe: str = _REFERENCE.timeframe
    initial_cash: float = _REFERENCE.initial_cash
    currency: str = _REFERENCE.currency
    holding_period_days: int = _REFERENCE.holding_period_days

    # --- Splits, inherited so the windows are identical ---------------------
    dev_end: str = _REFERENCE.dev_end
    validation_start: str = _REFERENCE.validation_start
    validation_end: str = _REFERENCE.validation_end
    test_start: str = _REFERENCE.test_start
    test_end: str = _REFERENCE.test_end

    # --- Decision cadence ---------------------------------------------------
    decision_frequency_sessions: int = 5
    """How often the graph is asked, in sessions. The graph is expensive per call,
    so a pilot does not ask every session. A fixed cadence, declared here, so the
    LLM and a same-cadence reference are asked at identical moments."""
    max_decisions: int = 120
    """A hard ceiling on model calls for a pilot, so a misconfiguration cannot
    spend without bound. Declared, not discovered."""

    # --- What the model is asked, and how it is configured ------------------
    input_scope: str = "price_only"
    """``price_only`` (the frozen dataset) or ``full`` (news/sentiment/fundamentals,
    which the frozen dataset does not carry and which are not PIT-safe for a
    historical run; see ``dataflows/pit.py``). A ``full`` run is out of scope for
    this phase and is refused rather than faked."""
    prompt_version: str = "tradingagents-default"
    """The graph's shipped prompts, unmodified. No prompt tuning in this phase."""
    graph_version: str = "tradingagents-graph"

    # --- Sizing when a record carries no size -------------------------------
    default_position_size: float = 0.20
    """The baseline's per-entry size, so a BUY with no size is comparable."""
    stop_loss_pct: float | None = 0.20
    """The baseline's protective stop, applied identically to a replayed BUY."""

    # --- Determinism --------------------------------------------------------
    temperature: float | None = None
    """Left at the provider default. Setting 0.0 does not guarantee identical
    output across calls, and this protocol does not claim it does."""

    def __post_init__(self) -> None:
        if self.input_scope not in ("price_only", "full"):
            raise ValueError("input_scope must be 'price_only' or 'full'")
        if self.decision_frequency_sessions < 1:
            raise ValueError("decision_frequency_sessions must be at least 1")
        if self.max_decisions < 1:
            raise ValueError("max_decisions must be at least 1")
        if self.input_scope == "full":
            raise ValueError(
                "input_scope='full' needs news/sentiment/fundamentals, which the frozen "
                "dataset does not carry and which are not point-in-time safe for a "
                "historical run (dataflows/pit.py); it is out of scope for this phase"
            )
        if not (self.dev_end < self.validation_start <= self.validation_end < self.test_start):
            raise ValueError("the dev/validation/test periods must be ordered and disjoint")

    def split_periods(self) -> tuple[tuple[str, str, str], ...]:
        return (
            ("development", self.start, self.dev_end),
            ("validation", self.validation_start, self.validation_end),
            ("test", self.test_start, self.test_end),
        )

    # --- Delegated to the reference baseline --------------------------------
    # The evaluation reuses the baseline's cost and risk *objects* rather than
    # copying their values, so the two runs cannot drift apart on costs or
    # limits: there is one definition, and both protocols read it.
    @property
    def limits(self):
        return _REFERENCE.limits

    @property
    def costs(self):
        return _REFERENCE.costs

    @property
    def max_volume_participation(self) -> float:
        return _REFERENCE.max_volume_participation

    @property
    def use_stops(self) -> bool:
        return _REFERENCE.use_stops

    @property
    def execute_on_next_open(self) -> bool:
        return _REFERENCE.execute_on_next_open

    @property
    def position_size(self) -> float:
        return self.default_position_size

    def config_hash(self) -> str:
        """A stable hash of everything that shapes a decision's inputs and costs.

        Recorded on every decision so a replay can prove it ran under the same
        configuration the model did.
        """
        payload = {
            "version": self.version,
            "universe": list(self.universe),
            "benchmark": self.benchmark,
            "start": self.start,
            "end": self.end,
            "input_scope": self.input_scope,
            "prompt_version": self.prompt_version,
            "graph_version": self.graph_version,
            "holding_period_days": self.holding_period_days,
            "default_position_size": self.default_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "initial_cash": self.initial_cash,
            "costs": _REFERENCE.costs.model_dump(),
            "risk_limits": _REFERENCE.limits.model_dump(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "universe": list(self.universe),
            "benchmark": self.benchmark,
            "start": self.start,
            "end": self.end,
            "timeframe": self.timeframe,
            "initial_cash": self.initial_cash,
            "currency": self.currency,
            "holding_period_days": self.holding_period_days,
            "splits": {
                "development": [self.start, self.dev_end],
                "validation": [self.validation_start, self.validation_end],
                "test": [self.test_start, self.test_end],
            },
            "decision_frequency_sessions": self.decision_frequency_sessions,
            "max_decisions": self.max_decisions,
            "input_scope": self.input_scope,
            "prompt_version": self.prompt_version,
            "graph_version": self.graph_version,
            "default_position_size": self.default_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "temperature": self.temperature,
            "config_hash": self.config_hash(),
            "inherited_from_baseline": _REFERENCE.version,
            "dataset_checksums": _REFERENCE.dataset_checksums(),
        }


DEFAULT_PROTOCOL = LLMEvaluationProtocol()
