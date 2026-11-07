"""Decision records: capture an LLM decision, serialise it, and validate it.

A decision record is the bridge between the (non-deterministic, credential-gated)
LLM pipeline and the (deterministic, frozen) simulation. It holds everything the
signal path needs to rebuild a decision without calling a model, plus the
provenance needed to audit it.

Three sources are kept strictly apart, because conflating them is how a mock
result gets reported as strategy performance:

- ``llm_inference`` — a real provider/model call produced this decision.
- ``operator_supplied`` — a human captured decisions from a real run elsewhere.
- ``synthetic_fixture`` — a test fixture. **Never** strategy performance.

This module also refuses to let a credential reach an artifact. Every record set
is scanned for credential-shaped values and for the values of credential-bearing
environment variables before it is serialised; a hit raises rather than writing
the file, and the error names the field, never the secret.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DECISION_RECORD_SCHEMA_VERSION = 1

SOURCE_LLM_INFERENCE = "llm_inference"
SOURCE_OPERATOR_SUPPLIED = "operator_supplied"
SOURCE_SYNTHETIC_FIXTURE = "synthetic_fixture"
VALID_SOURCES = frozenset(
    {SOURCE_LLM_INFERENCE, SOURCE_OPERATOR_SUPPLIED, SOURCE_SYNTHETIC_FIXTURE}
)

VALID_ACTIONS = frozenset({"BUY", "HOLD", "SELL"})
# The rationale is recorded as a bounded excerpt, not the full transcript, and
# never chain-of-thought. Long enough to audit a call, short enough to keep.
MAX_REASONING_CHARS = 2000

# A credential-shaped literal, independent of any environment: an OpenAI-style
# key, an AWS access key id, a GitHub token, a Google API key, a Slack token.
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AIza[0-9A-Za-z_\-]{30,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,})"
)

_SECRET_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

_REQUIRED_RECORD_KEYS = frozenset(
    {
        "decision_id",
        "ticker",
        "as_of_date",
        "action",
        "rating",
        "confidence",
        "signal_timestamp",
        "window_start",
        "window_end",
        "bar_count",
        "data_snapshot_id",
        "data_window_hash",
        "config_hash",
        "prompt_version",
        "graph_version",
        "model_provider",
        "model_name",
        "source",
        "reasoning_excerpt",
        "position_size",
        "stop_loss",
        "take_profit",
        "order_intent",
        "limit_price",
        "time_in_force",
        "holding_period_days",
    }
)


class DecisionRecordError(ValueError):
    """A decision record is missing, malformed, or internally inconsistent."""


class CredentialLeakError(DecisionRecordError):
    """A credential-shaped value was about to be written to an artifact."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact_reasoning(text: str | None) -> str:
    """A bounded, credential-scrubbed excerpt of a decision's stated rationale."""
    if not text:
        return ""
    cleaned = _SECRET_VALUE_RE.sub("<redacted>", text).strip()
    if len(cleaned) > MAX_REASONING_CHARS:
        cleaned = cleaned[:MAX_REASONING_CHARS] + "…"
    return cleaned


def _secret_env_values() -> dict[str, str]:
    """Values of environment variables whose *name* marks them as a credential."""
    found: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if value and len(value) >= 8 and any(marker in upper for marker in _SECRET_ENV_MARKERS):
            found[name] = value
    return found


def assert_secret_free(payload: Any, *, path: str = "$") -> Any:
    """Raise if any string in a JSON-able payload looks like a credential.

    Two checks: a credential-shaped literal, and the exact value of a
    credential-bearing environment variable. The error names the offending field
    and never echoes the secret.
    """
    secrets = _secret_env_values()

    def walk(node: Any, where: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")
        elif isinstance(node, str):
            if _SECRET_VALUE_RE.search(node):
                raise CredentialLeakError(
                    f"a credential-shaped value would be written at {where}"
                )
            for name, value in secrets.items():
                if value in node:
                    raise CredentialLeakError(
                        f"the value of {name} would be written at {where}"
                    )

    walk(payload, path)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionRecordError(message)


def _finite(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a number",
    )
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite, got {value!r}")
    return number


def _parse_date(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value, f"{field} must be a non-empty date string")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise DecisionRecordError(f"{field} must be yyyy-mm-dd, got {value!r}") from exc
    return value


@dataclass(frozen=True)
class DecisionRecord:
    """One LLM decision, with enough context to replay it and audit it."""

    decision_id: str
    ticker: str
    as_of_date: str
    action: str
    rating: str
    confidence: float
    signal_timestamp: str
    window_start: str
    window_end: str
    bar_count: int
    data_snapshot_id: str
    data_window_hash: str
    config_hash: str
    prompt_version: str
    graph_version: str
    source: str = SOURCE_OPERATOR_SUPPLIED
    model_provider: str | None = None
    model_name: str | None = None
    reasoning_excerpt: str = ""
    position_size: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_intent: str = "MARKET"
    limit_price: float | None = None
    time_in_force: str = "DAY"
    holding_period_days: int | None = None

    def __post_init__(self) -> None:
        _require(bool(self.decision_id), "decision_id must not be empty")
        _require(bool(self.ticker), "ticker must not be empty")
        _parse_date(self.as_of_date, "as_of_date")
        _parse_date(self.window_start, "window_start")
        _parse_date(self.window_end, "window_end")
        _require(
            self.window_start <= self.as_of_date,
            "window_start must not be later than as_of_date",
        )
        # The window the model saw must end at the decision date. A window that
        # ended later would mean the model saw bars after the decision, which is
        # the look-ahead this whole layer exists to prevent.
        _require(
            self.window_end == self.as_of_date,
            "window_end must equal as_of_date: the model may not see a later bar",
        )
        _require(
            self.action in VALID_ACTIONS,
            f"action must be one of {sorted(VALID_ACTIONS)}, got {self.action!r}",
        )
        _require(bool(self.rating), "rating must not be empty")
        _require(self.source in VALID_SOURCES, f"unknown source {self.source!r}")
        confidence = _finite(self.confidence, "confidence")
        _require(0.0 <= confidence <= 1.0, "confidence must be in [0, 1]")
        _require(
            isinstance(self.bar_count, int)
            and not isinstance(self.bar_count, bool)
            and self.bar_count >= 0,
            "bar_count must be a non-negative integer",
        )
        # A record must carry the hash of the window it saw, or a replay cannot
        # prove it is feeding the same data the model saw.
        for field in ("data_window_hash", "config_hash"):
            _require(
                isinstance(getattr(self, field), str) and len(getattr(self, field)) == 64,
                f"{field} must be a 64-character SHA-256 hex digest",
            )
        if self.position_size is not None:
            size = _finite(self.position_size, "position_size")
            _require(0.0 < size <= 1.0, "position_size must be in (0, 1]")
        for field in ("stop_loss", "take_profit", "limit_price"):
            value = getattr(self, field)
            if value is not None:
                _require(_finite(value, field) > 0.0, f"{field} must be positive when set")
        if self.holding_period_days is not None:
            _require(
                isinstance(self.holding_period_days, int)
                and not isinstance(self.holding_period_days, bool)
                and self.holding_period_days > 0,
                "holding_period_days must be a positive integer",
            )
        _require(
            self.order_intent in ("MARKET", "LIMIT"),
            f"order_intent must be MARKET or LIMIT, got {self.order_intent!r}",
        )
        _require(
            self.time_in_force in ("DAY", "GTC"),
            f"time_in_force must be DAY or GTC, got {self.time_in_force!r}",
        )
        if self.order_intent == "MARKET":
            _require(
                self.limit_price is None, "a MARKET record must not carry a limit_price"
            )
        if self.time_in_force == "GTC":
            _require(self.order_intent == "LIMIT", "only a LIMIT record may be GTC")

        # A record states both an action and the rating it came from. If they
        # disagree, one of them is wrong; the signal path follows the rating, so
        # a mismatch would replay a decision nobody made.
        from tradingagents.signals import action_for_rating

        derived = action_for_rating(self.rating)
        if derived is None:
            # A rating with no action (REVIEW, or an unknown one) is a decision
            # nobody can read. It is not a trade, so it must carry no action.
            _require(
                self.action == "HOLD" and self.position_size is None,
                f"rating {self.rating!r} has no tradeable action, so the record must "
                "carry action HOLD and no position_size",
            )
        else:
            _require(
                derived.value == self.action,
                f"action {self.action!r} does not match rating {self.rating!r}, "
                f"which maps to {derived.value!r}",
            )

        # Mirror the signal contract: a HOLD does not request size or levels.
        if self.action == "HOLD":
            _require(
                self.position_size is None, "a HOLD record must not request a position_size"
            )
            _require(
                self.stop_loss is None and self.take_profit is None,
                "a HOLD record must not carry stop_loss or take_profit",
            )
            _require(self.limit_price is None, "a HOLD record must not request a LIMIT order")

        # Scrub at construction, so a credential in the rationale cannot reach a
        # serialiser even if the caller never calls to_dict().
        assert_secret_free(self.reasoning_excerpt, path="reasoning_excerpt")

    @property
    def is_genuine_inference(self) -> bool:
        return self.source == SOURCE_LLM_INFERENCE

    def to_dict(self) -> dict:
        payload = {"schema_version": DECISION_RECORD_SCHEMA_VERSION, **asdict(self)}
        return assert_secret_free(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> DecisionRecord:
        _require(isinstance(payload, dict), "a decision record must be a mapping")
        version = payload.get("schema_version", DECISION_RECORD_SCHEMA_VERSION)
        _require(
            version == DECISION_RECORD_SCHEMA_VERSION,
            f"unsupported decision record schema_version {version!r}; "
            f"this build reads version {DECISION_RECORD_SCHEMA_VERSION}",
        )
        fields = {k: v for k, v in payload.items() if k != "schema_version"}
        missing = _REQUIRED_RECORD_KEYS - set(fields)
        _require(not missing, f"decision record is missing {sorted(missing)}")
        unknown = set(fields) - _REQUIRED_RECORD_KEYS
        _require(not unknown, f"decision record has unknown fields {sorted(unknown)}")
        try:
            return cls(**fields)
        except TypeError as exc:
            raise DecisionRecordError(f"malformed decision record: {exc}") from exc


@dataclass(frozen=True)
class DecisionRecordSet:
    """A set of decisions, with the provenance of the whole set."""

    strategy_id: str
    graph_version: str
    protocol_version: str
    mode: str
    created_at: str
    decisions: tuple[DecisionRecord, ...]
    schema_version: int = DECISION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.mode in VALID_SOURCES, f"unknown record-set mode {self.mode!r}")
        _require(
            all(isinstance(d, DecisionRecord) for d in self.decisions),
            "every element of decisions must be a DecisionRecord",
        )
        _require(
            all(d.source == self.mode for d in self.decisions),
            "every decision's source must match the record set's mode",
        )
        seen: set[str] = set()
        for decision in self.decisions:
            _require(
                decision.decision_id not in seen,
                f"duplicate decision_id {decision.decision_id!r}",
            )
            seen.add(decision.decision_id)

    @property
    def is_genuine_inference(self) -> bool:
        return self.mode == SOURCE_LLM_INFERENCE

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({d.as_of_date for d in self.decisions}))

    def to_dict(self) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "graph_version": self.graph_version,
            "protocol_version": self.protocol_version,
            "mode": self.mode,
            "created_at": self.created_at,
            "decisions": [d.to_dict() for d in self.decisions],
        }
        return assert_secret_free(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> DecisionRecordSet:
        _require(isinstance(payload, dict), "a decision record set must be a mapping")
        version = payload.get("schema_version", DECISION_RECORD_SCHEMA_VERSION)
        _require(
            version == DECISION_RECORD_SCHEMA_VERSION,
            f"unsupported record set schema_version {version!r}",
        )
        required = {
            "strategy_id",
            "graph_version",
            "protocol_version",
            "mode",
            "created_at",
            "decisions",
        }
        missing = required - set(payload)
        _require(not missing, f"record set is missing {sorted(missing)}")
        raw = payload["decisions"]
        _require(isinstance(raw, list), "decisions must be a list")
        return cls(
            strategy_id=payload["strategy_id"],
            graph_version=payload["graph_version"],
            protocol_version=payload["protocol_version"],
            mode=payload["mode"],
            created_at=payload["created_at"],
            decisions=tuple(DecisionRecord.from_dict(d) for d in raw),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Serialise through to_dict(), which scans for credentials first, so a
        # leak is refused before anything is written to disk.
        text = json.dumps(self.to_dict(), indent=2, sort_keys=False)
        assert_secret_free(text)
        target.write_text(text + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> DecisionRecordSet:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DecisionRecordError(f"no decision record file at {source}") from exc
        except json.JSONDecodeError as exc:
            raise DecisionRecordError(f"{source} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)
