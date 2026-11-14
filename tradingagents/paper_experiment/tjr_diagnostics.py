"""Sanitized diagnostics for a TJR evaluation failure.

A rule chain that raises must not be reported as a bare exception name. A
``ValidationError`` says *that* a contract was broken and nothing about
*which* contract or *which* field, so the operator is left with a name to
search for and no evidence to act on.

This module turns the failure into a short, **sanitized** description: the
**stage** of the evaluation that failed and the **field** whose value broke a
constraint. It deliberately reports *no market data* — the failing value may
be a price, and a price is not needed to identify a broken constraint — and no
secret, because nothing here reads a credential.

The diagnostics never change the outcome. A failure stays a failure: the
source still returns ``STRATEGY_UNAVAILABLE`` and still refuses to produce a
decision. The point is to make the failure **diagnosable**, not to turn it
into a trade or into a silent no-signal.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_DETAIL_CHARS = 300
"""The ceiling on a reported constraint message.

A pydantic message carries the offending value, so it can carry a price. The
message is therefore truncated hard and the *value* is never reported
separately."""

STAGE_RULE_CHAIN = "rule_chain"
"""The rule chain itself raised — a frame, a level, or a pure computation."""

STAGE_SIGNAL_CONTRACT = "signal_contract"
"""The evaluation completed and the *signal* it produced failed its contract.

This is the stage that says "the chain worked and the answer was rejected",
which is a very different fault from "the chain could not run"."""

STAGE_UNKNOWN = "unknown"
"""The failure happened before the stage could be identified."""


def classify_evaluation_failure(exc: BaseException) -> dict[str, Any]:
    """Describe where an evaluation failed and which field broke a constraint.

    Returns a record with:

    - ``stage`` — :data:`STAGE_RULE_CHAIN`, :data:`STAGE_SIGNAL_CONTRACT` or
      :data:`STAGE_UNKNOWN`;
    - ``field`` — the contract field that was refused, when the failure names
      one, else ``None``;
    - ``constraint`` — a truncated, sanitized description of the constraint
      that was broken, or ``None``;
    - ``error_type`` — the exception's class name.

    The record never contains the offending **value**. A pydantic
    ``ValidationError`` carries the value it refused, and that value is market
    data, so only the field path and the constraint text are kept.
    """
    error_type = type(exc).__name__
    errors = _validation_errors(exc)
    if errors:
        first = errors[0]
        field = _field_path(first.get("loc"))
        constraint = _constraint_text(first.get("msg"))
        return {
            "stage": STAGE_SIGNAL_CONTRACT,
            "field": field,
            "constraint": constraint,
            "error_type": error_type,
        }
    return {
        "stage": STAGE_RULE_CHAIN,
        "field": None,
        "constraint": _constraint_text(str(exc)),
        "error_type": error_type,
    }


def format_evaluation_failure(symbol: str, diagnostics: dict[str, Any]) -> str:
    """Render a failure record as the detail line the worker persists.

    The line names the symbol, the stage and the field, and carries the
    constraint in a truncated form. It is short enough to read in a dashboard
    and long enough to identify the fault without the raw exception.
    """
    stage = str(diagnostics.get("stage") or STAGE_UNKNOWN)
    field = diagnostics.get("field")
    constraint = diagnostics.get("constraint")
    parts = [f"the TJR chain could not evaluate {symbol}", f"stage={stage}"]
    if field:
        parts.append(f"field={field}")
    if constraint:
        parts.append(f"constraint={constraint}")
    parts.append(f"error={diagnostics.get('error_type') or 'unknown'}")
    return ": ".join(parts)


def _validation_errors(exc: BaseException) -> list[dict[str, Any]]:
    """The structured errors of a pydantic ``ValidationError``, if it has any.

    A non-pydantic exception has no ``errors()``, and one that has it but
    returns something else is simply not a validation failure — both cases
    return an empty list, which reads as "the chain itself failed".
    """
    getter = getattr(exc, "errors", None)
    if not callable(getter):
        return []
    try:
        errors = getter()
    except Exception:  # noqa: BLE001 - a broken error accessor is not a fault
        return []
    if not isinstance(errors, list) or not errors:
        return []
    return [item for item in errors if isinstance(item, dict)]


def _field_path(location: Any) -> str | None:
    """The contract field that was refused, as a dotted path.

    A model-level validator reports an **empty** location, because it refuses
    the model as a whole rather than one field. In that case the field is
    unknown and reported as such, rather than guessed at.
    """
    if not isinstance(location, (list, tuple)) or not location:
        return None
    parts = [str(part) for part in location]
    return ".".join(parts)


def _constraint_text(message: Any) -> str | None:
    """The constraint text, with its numbers removed and its length bounded.

    A pydantic message is usually ``"Value error, <constraint>"``, and the
    project's own constraints quote the value they refused — ``"a BUY
    stop_loss (11.0) must be below its entry_price (10.0)"``. Those numbers
    are market data, and this module promises not to report market data, so
    they are replaced by a placeholder before the message is kept.

    The **wording** is what identifies the broken constraint, so it is kept
    intact; only the values are dropped.
    """
    if message is None:
        return None
    text = str(message).strip()
    if not text:
        return None
    text = _redact_numbers(text)
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[: _MAX_DETAIL_CHARS - 1].rstrip() + "…"
    return text


_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
"""A numeric literal, including a leading sign and a decimal part.

The lookbehind keeps a digit that is part of a word (``RSI14``) from being
redacted, so a constraint that names a parameter is still readable."""


def _redact_numbers(text: str) -> str:
    """Replace every numeric literal in ``text`` with a placeholder."""
    return _NUMBER.sub("<value>", text)
