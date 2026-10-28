"""LLM strategy evaluation: record, replay and audit the agent decision pipeline.

The reference baseline (``tradingagents/baseline/``) measures a deterministic SMA
rule. This package measures the *actual* LLM-agent pipeline, which is a different
strategy: it is credential-gated, its inputs include news and sentiment, and its
outputs are not guaranteed reproducible. The two are never conflated.

The evaluation is honest by construction:

- A decision is captured in a :class:`~.records.DecisionRecord` that carries the
  hash of the data window the model saw, so a replay can prove it ran on the same
  data.
- A :class:`~.replay.ReplayStrategy` feeds recorded decisions into the frozen
  engine through the production signal contract, and calls no model.
- :func:`~.evaluate.build_evaluation` reports ``LLM BASELINE BLOCKED`` with the
  exact missing prerequisite when no genuine run is possible, and never invents a
  number.

No live trading path is added. Replay travels the same ``BACKTEST``/``PAPER``
engine the audited stack uses.
"""

from __future__ import annotations

from .capture import (
    MODE_MOCK_TEST,
    MODE_REAL_INFERENCE_CAPTURE,
    MODE_REPLAY,
    VALID_MODES,
    CaptureError,
    CaptureResult,
    build_record,
    capture_decision,
    graph_observed_real_inference,
    observed_decision,
)
from .evaluate import (
    ARTIFACT_SCHEMA_VERSION,
    STATUS_BLOCKED,
    STATUS_ESTABLISHED,
    STATUS_ESTABLISHED_WITH_FINDINGS,
    EvaluationOutcome,
    build_evaluation,
    credential_status,
    write_evaluation,
)
from .point_in_time import (
    Classification,
    ReadinessVerdict,
    agent_dependency_matrix,
    graph_readiness,
    point_in_time_report,
    source_facts,
)
from .protocol import DEFAULT_PROTOCOL, LLMEvaluationProtocol
from .provider_audit import (
    ProviderReadiness,
    looks_like_a_real_credential,
    provider_readiness,
    provider_report,
)
from .records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_OPERATOR_SUPPLIED,
    SOURCE_SYNTHETIC_FIXTURE,
    CredentialLeakError,
    DecisionRecord,
    DecisionRecordError,
    DecisionRecordSet,
    redact_reasoning,
    sha256_text,
)
from .replay import ReplayError, ReplayStrategy, no_llm_calls, signal_from_record, window_hash

# ``smoke`` is imported lazily: it is run as ``python -m ...smoke``, and a
# package-level import of it would make the ``-m`` loader warn about a
# double-import. Accessing the names below imports it on first use.
_SMOKE_NAMES = {"SmokeBlocked", "SmokeReport", "run_smoke"}


def __getattr__(name: str):
    if name in _SMOKE_NAMES:
        from . import smoke

        return getattr(smoke, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_PROTOCOL",
    "MODE_MOCK_TEST",
    "MODE_REAL_INFERENCE_CAPTURE",
    "MODE_REPLAY",
    "SOURCE_LLM_INFERENCE",
    "SOURCE_OPERATOR_SUPPLIED",
    "SOURCE_SYNTHETIC_FIXTURE",
    "STATUS_BLOCKED",
    "STATUS_ESTABLISHED",
    "STATUS_ESTABLISHED_WITH_FINDINGS",
    "VALID_MODES",
    "CaptureError",
    "CaptureResult",
    "Classification",
    "CredentialLeakError",
    "DecisionRecord",
    "DecisionRecordError",
    "DecisionRecordSet",
    "EvaluationOutcome",
    "LLMEvaluationProtocol",
    "ProviderReadiness",
    "ReadinessVerdict",
    "ReplayError",
    "ReplayStrategy",
    "SmokeBlocked",
    "SmokeReport",
    "agent_dependency_matrix",
    "build_evaluation",
    "build_record",
    "capture_decision",
    "credential_status",
    "graph_observed_real_inference",
    "graph_readiness",
    "looks_like_a_real_credential",
    "no_llm_calls",
    "observed_decision",
    "point_in_time_report",
    "provider_readiness",
    "provider_report",
    "redact_reasoning",
    "run_smoke",
    "sha256_text",
    "signal_from_record",
    "source_facts",
    "window_hash",
    "write_evaluation",
]
