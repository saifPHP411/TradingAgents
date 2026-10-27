"""LLM telemetry: what each call cost, and a budget that stops a run before it overruns.

Two jobs, and the second matters more:

1. **Accounting.** Every call records its model, provider, token counts, latency,
   retries and estimated cost, so a run's expense is inspectable after the fact
   rather than guessed at.

2. **A hard budget.** ``max_run_cost``, ``max_llm_calls`` and ``max_retries`` are
   checked *before* a call is made. A budget that is only checked afterwards is
   not a budget: it reports the overspend instead of preventing it. When a limit
   is reached the recorder raises :class:`BudgetExceededError`, which the caller
   is expected to let end the run.

No secret ever reaches this module. There is no field for an API key, and the
recorder stores a provider name and a model name, both of which are public. A
telemetry record is written to be shown to a user, so it must be safe to show.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.telemetry.pricing import EMPTY_PRICE_TABLE, PriceTable


class BudgetExceededError(RuntimeError):
    """Raised before a call that would take a run past a configured budget.

    Raised ahead of the call, so the money is not spent. A caller should let this
    propagate and end the run: continuing past it would make the budget advisory,
    which is the same as not having one.
    """

    def __init__(self, limit: str, current: float, maximum: float) -> None:
        self.limit = limit
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"LLM budget exceeded: {limit} is {current:,.4f}, at or above the limit of "
            f"{maximum:,.4f}. The run was stopped before the call was made."
        )


@dataclass
class CallRecord:
    """One LLM call: what it was, what it used, and what it cost.

    Deliberately free of prompt and response text. A telemetry log that stores
    prompts becomes a second copy of potentially sensitive research content, and
    a cost report does not need it.
    """

    sequence: int
    provider: str
    model: str
    started_at: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    retries: int = 0
    error: str = ""
    estimated_cost: float | None = None
    purpose: str = ""
    """What the call was for (an agent role, say). Never the prompt itself."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def succeeded(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "error": self.error,
            "estimated_cost": self.estimated_cost,
            "purpose": self.purpose,
        }


@dataclass
class Budget:
    """Optional limits on a run's LLM usage. ``None`` disables a limit.

    A budget is not applied by accident: the defaults disable every limit, so a
    run without one configured is unbounded and says so.
    """

    max_run_cost: float | None = None
    max_llm_calls: int | None = None
    max_retries: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_run_cost", self.max_run_cost),
            ("max_llm_calls", self.max_llm_calls),
            ("max_retries", self.max_retries),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")


@dataclass
class TelemetrySummary:
    """The totals for a run, ready to embed in its record or its report."""

    calls: int
    failures: int
    retries: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float | None
    """``None`` when any call's cost was unknown, because a partial total would
    understate the run."""
    unpriced_calls: int
    mean_latency_seconds: float | None
    by_model: dict[str, dict] = field(default_factory=dict)
    budget: dict | None = None

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "unpriced_calls": self.unpriced_calls,
            "mean_latency_seconds": self.mean_latency_seconds,
            "by_model": self.by_model,
            "budget": self.budget,
        }


class LLMTelemetry:
    """Collects call records and enforces a :class:`Budget`.

    Thread-safety is not attempted: a run drives one agent at a time, and a lock
    would suggest a concurrency the rest of the system does not have.
    """

    def __init__(
        self,
        *,
        price_table: PriceTable | None = None,
        budget: Budget | None = None,
        clock=time.monotonic,
    ) -> None:
        self.price_table = price_table or EMPTY_PRICE_TABLE
        self.budget = budget or Budget()
        self.records: list[CallRecord] = []
        self._clock = clock
        self._next_sequence = 0

    # --- Budget checks ------------------------------------------------------

    def check_budget(self) -> None:
        """Raise if any configured limit has already been reached.

        Called before a call is made, so the budget prevents the spend rather
        than reporting it. The cost limit is checked against the *recorded* cost
        so far; an unpriced call contributes nothing to it, which is a limitation
        the summary reports rather than hides.
        """
        if self.budget.max_llm_calls is not None and len(self.records) >= self.budget.max_llm_calls:
            raise BudgetExceededError("max_llm_calls", len(self.records), self.budget.max_llm_calls)
        cost = self.estimated_cost()
        if (
            self.budget.max_run_cost is not None
            and cost is not None
            and cost >= self.budget.max_run_cost
        ):
            raise BudgetExceededError("max_run_cost", cost, self.budget.max_run_cost)

    def check_retries(self, retries: int) -> None:
        """Raise if a call has already been retried to the configured limit."""
        if self.budget.max_retries is not None and retries > self.budget.max_retries:
            raise BudgetExceededError("max_retries", retries, self.budget.max_retries)

    # --- Recording ----------------------------------------------------------

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float = 0.0,
        retries: int = 0,
        error: str = "",
        purpose: str = "",
        started_at: str | None = None,
    ) -> CallRecord:
        """Append a call record and return it."""
        cost = self.price_table.cost(
            provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        record = CallRecord(
            sequence=self._next_sequence,
            provider=provider,
            model=model,
            started_at=started_at or datetime.now(UTC).isoformat(),
            latency_seconds=float(latency_seconds),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            retries=int(retries),
            error=error,
            estimated_cost=cost,
            purpose=purpose,
        )
        self._next_sequence += 1
        self.records.append(record)
        return record

    @contextmanager
    def timed_call(
        self, *, provider: str, model: str, purpose: str = "", retries: int = 0
    ) -> Iterator[dict]:
        """Time a call and record it, enforcing the budget before it starts.

        Usage::

            with telemetry.timed_call(provider="openai", model="gpt-5") as call:
                response = llm.invoke(prompt)
                call["input_tokens"] = ...   # from the response's usage

        The budget is checked on entry, so a call that would breach it raises
        before the request is sent. On an exception the record is still written,
        with the error recorded, because a failed call still cost latency and
        possibly tokens.
        """
        self.check_budget()
        self.check_retries(retries)
        started_wall = datetime.now(UTC).isoformat()
        start = self._clock()
        outcome: dict = {"input_tokens": 0, "output_tokens": 0}
        error = ""
        try:
            yield outcome
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = self._clock() - start
            self.record(
                provider=provider,
                model=model,
                input_tokens=int(outcome.get("input_tokens", 0) or 0),
                output_tokens=int(outcome.get("output_tokens", 0) or 0),
                latency_seconds=elapsed,
                retries=retries,
                error=error,
                purpose=purpose,
                started_at=started_wall,
            )

    # --- Aggregates ---------------------------------------------------------

    def estimated_cost(self) -> float | None:
        """The total cost so far, or ``None`` if any call's price was unknown.

        A partial total would understate the run, which is the dangerous
        direction for a budget, so an unpriced call makes the whole total
        unknown rather than merely incomplete.
        """
        if not self.records:
            return 0.0
        if any(r.estimated_cost is None for r in self.records):
            return None
        return sum(r.estimated_cost for r in self.records)

    def summary(self) -> TelemetrySummary:
        """The run's totals, including a per-model breakdown."""
        by_model: dict[str, dict] = {}
        for record in self.records:
            key = f"{record.provider}/{record.model}"
            entry = by_model.setdefault(
                key,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                 "unpriced_calls": 0, "failures": 0},
            )
            entry["calls"] += 1
            entry["input_tokens"] += record.input_tokens
            entry["output_tokens"] += record.output_tokens
            if record.estimated_cost is None:
                entry["unpriced_calls"] += 1
            else:
                entry["cost"] += record.estimated_cost
            if not record.succeeded:
                entry["failures"] += 1
        latencies = [r.latency_seconds for r in self.records]
        budget_dict = None
        if any(v is not None for v in (self.budget.max_run_cost, self.budget.max_llm_calls,
                                       self.budget.max_retries)):
            budget_dict = {
                "max_run_cost": self.budget.max_run_cost,
                "max_llm_calls": self.budget.max_llm_calls,
                "max_retries": self.budget.max_retries,
            }
        return TelemetrySummary(
            calls=len(self.records),
            failures=sum(1 for r in self.records if not r.succeeded),
            retries=sum(r.retries for r in self.records),
            input_tokens=sum(r.input_tokens for r in self.records),
            output_tokens=sum(r.output_tokens for r in self.records),
            total_tokens=sum(r.total_tokens for r in self.records),
            estimated_cost=self.estimated_cost(),
            unpriced_calls=sum(1 for r in self.records if r.estimated_cost is None),
            mean_latency_seconds=(sum(latencies) / len(latencies)) if latencies else None,
            by_model=by_model,
            budget=budget_dict,
        )

    def write_json(self, path: str | Path) -> Path:
        """Write the records and the summary to ``path`` as JSON.

        Records only; there is no prompt or response text in them, so the file is
        safe to attach to a report.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary().to_dict(),
            "calls": [r.to_dict() for r in self.records],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
