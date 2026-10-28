"""The declared configuration of a prospective paper experiment.

Everything that shapes a decision, its execution and its evaluation is fixed
here, before any session is processed, and hashed into a ``config_hash`` recorded
on the journal. A result can therefore be tied to the exact configuration that
produced it, and a re-run under a different configuration is visible as a
different experiment rather than a silent change.

The configuration inherits the frozen evaluation protocol's universe, costs, risk
limits and initial capital, so the paper experiment and the reference baseline
are comparable. It adds only the experiment-specific timing:

- ``execute_on_next_open`` is forced ``True``. A decision made from bars through
  session T's close is executed at session T+1's open. Allowing ``False`` would
  let a decision fill at the very close it was made from, which is the look-ahead
  this whole layer exists to prevent, so it is refused rather than defaulted.
- ``evaluation_lag_sessions`` is how many sessions after execution a decision's
  outcome is measured. It is declared, not chosen after seeing the result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL, LLMEvaluationProtocol


class ExperimentConfigError(ValueError):
    """The experiment configuration is invalid or internally inconsistent."""


@dataclass(frozen=True)
class ExperimentConfig:
    """A frozen experiment specification, with a stable content hash."""

    strategy_id: str = "llm-prospective-paper"
    symbols: tuple[str, ...] = ("SPY",)
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    decision_frequency_sessions: int = 5
    max_decisions: int = 120
    execute_on_next_open: bool = True
    evaluation_lag_sessions: int = 5
    initial_cash: float = 100_000.0
    holding_period_days: int = 21
    protocol_version: str = DEFAULT_PROTOCOL.version
    protocol_config_hash: str = ""
    costs: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    min_trades_for_metrics: int = 20
    """The sample size below which win rate, Sharpe and drawdown are not reported.

    A ratio computed from a handful of trades is noise presented as a number. The
    report shows the sample size and marks the metric N/A until this is met.
    """

    max_volume_participation: float = 0.0
    """The most of a bar's volume one order may take, as a fraction.

    ``0.0`` means no cap (the protocol default). A positive value models a thin
    book, so a large order fills partially across sessions rather than whole —
    the behaviour a partial-fill recovery test needs.
    """

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ExperimentConfigError("strategy_id must not be empty")
        if not self.symbols:
            raise ExperimentConfigError("at least one symbol is required")
        if not self.execute_on_next_open:
            raise ExperimentConfigError(
                "execute_on_next_open must be True: filling a decision at the close "
                "it was made from is look-ahead, not a supported configuration"
            )
        if self.decision_frequency_sessions < 1:
            raise ExperimentConfigError("decision_frequency_sessions must be at least 1")
        if self.evaluation_lag_sessions < 1:
            raise ExperimentConfigError("evaluation_lag_sessions must be at least 1")
        if self.max_decisions < 1:
            raise ExperimentConfigError("max_decisions must be at least 1")
        if self.initial_cash <= 0:
            raise ExperimentConfigError("initial_cash must be positive")
        if self.min_trades_for_metrics < 1:
            raise ExperimentConfigError("min_trades_for_metrics must be at least 1")
        if not 0.0 <= self.max_volume_participation <= 1.0:
            raise ExperimentConfigError(
                "max_volume_participation must be between 0 and 1, got "
                f"{self.max_volume_participation}"
            )

    @classmethod
    def from_protocol(
        cls,
        protocol: LLMEvaluationProtocol | None = None,
        **overrides,
    ) -> ExperimentConfig:
        """Build a config from the frozen protocol, with explicit overrides."""
        protocol = protocol or DEFAULT_PROTOCOL
        base = {
            "symbols": (protocol.benchmark,),
            "decision_frequency_sessions": protocol.decision_frequency_sessions,
            "max_decisions": protocol.max_decisions,
            "initial_cash": protocol.initial_cash,
            "holding_period_days": protocol.holding_period_days,
            "protocol_version": protocol.version,
            "protocol_config_hash": protocol.config_hash(),
            "costs": protocol.costs.model_dump(),
            "limits": protocol.limits.model_dump(),
        }
        base.update(overrides)
        return cls(**base)

    def config_hash(self) -> str:
        """A stable hash of everything that shapes the experiment's behaviour."""
        payload = {
            "strategy_id": self.strategy_id,
            "symbols": list(self.symbols),
            "provider": self.provider,
            "model": self.model,
            "decision_frequency_sessions": self.decision_frequency_sessions,
            "max_decisions": self.max_decisions,
            "execute_on_next_open": self.execute_on_next_open,
            "evaluation_lag_sessions": self.evaluation_lag_sessions,
            "initial_cash": self.initial_cash,
            "holding_period_days": self.holding_period_days,
            "protocol_version": self.protocol_version,
            "protocol_config_hash": self.protocol_config_hash,
            "costs": self.costs,
            "limits": self.limits,
            "min_trades_for_metrics": self.min_trades_for_metrics,
            "max_volume_participation": self.max_volume_participation,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "strategy_id": self.strategy_id,
            "symbols": list(self.symbols),
            "provider": self.provider,
            "model": self.model,
            "decision_frequency_sessions": self.decision_frequency_sessions,
            "max_decisions": self.max_decisions,
            "execute_on_next_open": self.execute_on_next_open,
            "evaluation_lag_sessions": self.evaluation_lag_sessions,
            "initial_cash": self.initial_cash,
            "holding_period_days": self.holding_period_days,
            "protocol_version": self.protocol_version,
            "protocol_config_hash": self.protocol_config_hash,
            "costs": self.costs,
            "limits": self.limits,
            "min_trades_for_metrics": self.min_trades_for_metrics,
            "max_volume_participation": self.max_volume_participation,
            "config_hash": self.config_hash(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ExperimentConfig:
        if not isinstance(payload, dict):
            raise ExperimentConfigError("an experiment config must be a mapping")
        version = payload.get("schema_version", 1)
        if version != 1:
            raise ExperimentConfigError(f"unsupported experiment config schema_version {version!r}")
        fields = {k: v for k, v in payload.items() if k not in ("schema_version", "config_hash")}
        if "symbols" in fields:
            fields["symbols"] = tuple(fields["symbols"])
        try:
            config = cls(**fields)
        except TypeError as exc:
            raise ExperimentConfigError(f"malformed experiment config: {exc}") from exc
        expected = payload.get("config_hash")
        if expected is not None and expected != config.config_hash():
            raise ExperimentConfigError(
                "experiment config hash does not match its contents; the file has been altered"
            )
        return config
