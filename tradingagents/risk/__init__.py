"""The deterministic risk engine: code, not a prompt, decides what is allowed.

``engine`` holds :class:`RiskLimits`, :class:`RiskEngine` and the verdict/reason
vocabularies. It reads the ledger and the validated signal, and nothing else —
no model text, no network, no clock beyond what the caller supplies.
"""

from tradingagents.risk.engine import (
    RiskDecision,
    RiskEngine,
    RiskLimits,
    RiskReason,
    RiskVerdict,
    drawdown_from_peak,
)

__all__ = [
    "RiskDecision",
    "RiskEngine",
    "RiskLimits",
    "RiskReason",
    "RiskVerdict",
    "drawdown_from_peak",
]
