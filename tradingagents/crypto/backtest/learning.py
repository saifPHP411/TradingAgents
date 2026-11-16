"""Leakage-safe online expert learning for chronological crypto research.

This policy learns only from closed, realized, after-cost trades. It is a
research overlay: it can abstain from weak experts, but cannot change sizing,
risk limits, execution costs, or submit orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from statistics import mean, stdev


def _timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamps must be ISO-8601") from exc
    else:
        raise ValueError("timestamps must be ISO-8601 strings or datetimes")
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return stamp


@dataclass(frozen=True)
class ExpertObservation:
    strategy: str
    exit_time: str
    net_return_pct: float


@dataclass
class OnlineExpertLearner:
    """Conservative per-expert gate updated strictly after each OOS fold."""

    min_observations: int = 20
    max_observations: int = 100
    z_score: float = 1.96
    minimum_mean_return_pct: float = 0.0
    observations: dict[str, list[ExpertObservation]] = field(default_factory=dict)
    _last_update: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.min_observations, bool) or not isinstance(self.min_observations, int) or self.min_observations < 2:
            raise ValueError("min_observations must be an integer >= 2")
        if isinstance(self.max_observations, bool) or not isinstance(self.max_observations, int) or self.max_observations < self.min_observations:
            raise ValueError("max_observations must be an integer >= min_observations")
        if not math.isfinite(self.z_score) or self.z_score <= 0:
            raise ValueError("z_score must be finite and positive")
        if not math.isfinite(self.minimum_mean_return_pct):
            raise ValueError("minimum_mean_return_pct must be finite")

    def update(self, strategy: str, exit_time: str, net_return_pct: float) -> None:
        if not strategy or strategy == "none":
            raise ValueError("a named strategy is required")
        stamp = _timestamp(exit_time)
        if not math.isfinite(net_return_pct):
            raise ValueError("net_return_pct must be finite")
        if self._last_update is not None and stamp < self._last_update:
            raise ValueError("learning observations must arrive chronologically")
        rows = self.observations.setdefault(strategy, [])
        rows.append(ExpertObservation(strategy, stamp.isoformat(), float(net_return_pct)))
        if len(rows) > self.max_observations:
            del rows[:-self.max_observations]
        self._last_update = stamp

    def allows(self, strategy: str, decision_time: str | datetime) -> bool:
        decision_stamp = _timestamp(decision_time)
        matured = [
            row.net_return_pct
            for row in self.observations.get(strategy, [])
            if _timestamp(row.exit_time) < decision_stamp
        ]
        if len(matured) < self.min_observations:
            return True
        avg = mean(matured)
        standard_error = stdev(matured) / math.sqrt(len(matured))
        lower_bound = avg - self.z_score * standard_error
        return lower_bound > self.minimum_mean_return_pct

    def snapshot(self, decision_time: str | datetime) -> dict:
        decision_stamp = _timestamp(decision_time)
        result = {}
        for strategy in sorted(self.observations):
            values = [
                row.net_return_pct
                for row in self.observations[strategy]
                if _timestamp(row.exit_time) < decision_stamp
            ]
            if len(values) < self.min_observations:
                result[strategy] = {
                    "observations": len(values),
                    "status": "learning",
                    "mean_net_return_pct": mean(values) if values else None,
                    "lower_95pct_bound_pct": None,
                    "allowed": True,
                }
                continue
            avg = mean(values)
            lower = avg - self.z_score * stdev(values) / math.sqrt(len(values))
            result[strategy] = {
                "observations": len(values),
                "status": "validated" if lower > self.minimum_mean_return_pct else "blocked",
                "mean_net_return_pct": avg,
                "lower_95pct_bound_pct": lower,
                "allowed": lower > self.minimum_mean_return_pct,
            }
        return result
