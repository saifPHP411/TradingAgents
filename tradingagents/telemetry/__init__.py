"""Telemetry: LLM cost accounting, budgets, and the run record for reproducibility.

- ``pricing``: a caller-supplied price table. No built-in rates, so no stale ones.
- ``llm``: :class:`LLMTelemetry`, which records each call and enforces a budget
  *before* the call is made.
- ``run``: :class:`RunMetadata`, which records what produced a result and flags
  the ways it may not be reproducible.

Nothing here stores a secret: there is no field for a credential, and the records
are written to be safe to show a user.
"""

from tradingagents.telemetry.llm import (
    Budget,
    BudgetExceededError,
    CallRecord,
    LLMTelemetry,
    TelemetrySummary,
)
from tradingagents.telemetry.pricing import EMPTY_PRICE_TABLE, ModelPrice, PriceTable
from tradingagents.telemetry.run import ModelConfig, RunMetadata

__all__ = [
    "EMPTY_PRICE_TABLE",
    "Budget",
    "BudgetExceededError",
    "CallRecord",
    "LLMTelemetry",
    "ModelConfig",
    "ModelPrice",
    "PriceTable",
    "RunMetadata",
    "TelemetrySummary",
]
