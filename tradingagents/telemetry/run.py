"""Run metadata: everything needed to say what produced a result.

A backtest figure without its configuration is not evidence. Two runs of "the
same" strategy differ if the model, the temperature, the risk limits or the data
range differ, and the difference is invisible unless it is recorded. This module
records it, and — just as importantly — records what it *could not* determine.

Three kinds of field, kept distinct on purpose:

- **Determined**: the git commit, the code version, the timestamps. Read from the
  environment.
- **Supplied**: the model, the temperature, the limits. The caller knows these;
  the system must not guess them.
- **Unknown, and marked as such**: a field with no value is ``None``, never a
  plausible default. A fabricated configuration is worse than a missing one,
  because it looks like evidence.

Nondeterminism is recorded too. A run whose LLM calls were made at a non-zero
temperature is flagged, because its decisions are not reproducible from the
config alone even when the config matches.
"""

from __future__ import annotations

import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def _git_commit() -> str | None:
    """The current commit hash, or ``None`` when this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes, or ``None`` if unknown.

    A run made from a dirty tree is not reproducible from its commit, so this is
    recorded rather than ignored.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _code_version() -> str | None:
    """The installed TradingAgents version, or ``None`` if it cannot be read."""
    try:
        from tradingagents.version import __version__  # type: ignore[attr-defined]
    except Exception:
        try:
            from importlib.metadata import version

            return version("tradingagents")
        except Exception:
            return None
    return __version__


@dataclass
class ModelConfig:
    """Which model, and how it was configured. All fields optional and explicit."""

    provider: str | None = None
    model: str | None = None
    model_version: str | None = None
    temperature: float | None = None
    base_url: str | None = None
    """The endpoint, when it is not the provider's default. Recorded because a
    local Ollama endpoint and a hosted one are different experiments."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunMetadata:
    """The record of one run: what it was, what produced it, and what was uncertain."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    code_version: str | None = field(default_factory=_code_version)
    git_commit: str | None = field(default_factory=_git_commit)
    git_dirty: bool | None = field(default_factory=_git_dirty)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)

    model: ModelConfig = field(default_factory=ModelConfig)
    research_depth: int | None = None
    risk_rounds: int | None = None
    debate_rounds: int | None = None
    data_sources: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    random_seed: int | None = None
    configuration: dict = field(default_factory=dict)
    """The run's own settings (risk limits, costs, holding period), as given."""

    telemetry: dict | None = None
    """The LLM usage summary, attached once the run finishes."""

    nondeterministic: bool = False
    """Whether the run's decisions can vary between identical runs."""

    nondeterminism_notes: list[str] = field(default_factory=list)
    """Why it is nondeterministic, in plain terms, so the flag is actionable."""

    def flag_nondeterminism(self, reason: str) -> None:
        """Record a reason the run cannot be exactly reproduced."""
        self.nondeterministic = True
        if reason not in self.nondeterminism_notes:
            self.nondeterminism_notes.append(reason)

    def assess_determinism(self) -> None:
        """Set the determinism flag from the recorded model settings.

        A non-zero temperature makes the model's output vary between runs, so the
        decisions are not reproducible even from an identical config. A missing
        temperature is treated as unknown and flagged, because a run that did not
        record it cannot be shown to be deterministic.
        """
        if self.model.temperature is None:
            self.flag_nondeterminism(
                "the model temperature was not recorded, so determinism cannot be asserted"
            )
        elif self.model.temperature > 0:
            self.flag_nondeterminism(
                f"the model runs at temperature {self.model.temperature}, so its output "
                "varies between identical runs"
            )
        if self.git_dirty:
            self.flag_nondeterminism(
                "the working tree had uncommitted changes, so this run is not reproducible "
                "from its commit alone"
            )
        if self.git_commit is None:
            self.flag_nondeterminism("the git commit could not be determined")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "model": self.model.to_dict(),
            "research_depth": self.research_depth,
            "risk_rounds": self.risk_rounds,
            "debate_rounds": self.debate_rounds,
            "data_sources": list(self.data_sources),
            "symbols": list(self.symbols),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "random_seed": self.random_seed,
            "configuration": self.configuration,
            "telemetry": self.telemetry,
            "nondeterministic": self.nondeterministic,
            "nondeterminism_notes": list(self.nondeterminism_notes),
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the record to ``path`` as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> RunMetadata:
        """Read a record written by :meth:`write_json`.

        Only the known fields are used, so a record written by a newer version
        does not break an older one.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model_payload = payload.pop("model", {}) or {}
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in payload.items() if k in known}
        metadata = cls(**payload)
        metadata.model = ModelConfig(**{
            k: v for k, v in model_payload.items() if k in ModelConfig.__dataclass_fields__
        })
        return metadata
