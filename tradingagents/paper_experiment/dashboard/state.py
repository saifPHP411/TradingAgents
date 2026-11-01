"""Read-only views of a paper experiment for the dashboard.

Everything here reads the experiment's persisted files (``session_state.json``,
``journal.json``, ``snapshots/``) and its report. Nothing here mutates a book:
the one mutating action the dashboard offers — advancing the simulation — lives
in :mod:`tradingagents.paper_experiment.dashboard.server` and is gated there.

The views are JSON-safe and secret-free. The experiment's own ``build_report``
already refuses to emit a credential-shaped value, so the dashboard reuses it
rather than re-deriving figures from raw state and risking a divergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..report import build_report


class DashboardStateError(RuntimeError):
    """The experiment directory is missing or unreadable."""


@dataclass
class ExperimentView:
    """A loaded experiment, plus the helpers the dashboard needs to read it."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.state_path = self.root / "session_state.json"
        if not self.state_path.exists():
            raise DashboardStateError(
                f"no experiment state at {self.state_path}; run `start` first"
            )

    # -- raw state --------------------------------------------------------

    def _raw_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DashboardStateError(
                f"the experiment state {self.state_path} is not usable: {exc}"
            ) from exc

    def load(self, *, inference=None):
        """Load the experiment, restoring its price source without a network call.

        ``inference`` is the decision seam. A *read* leaves it ``None`` (nothing is
        decided, so nothing is needed). A caller that advances the simulation must
        supply one — the CLI's ``run`` builds it from the arguments, and the
        dashboard builds it from the persisted config — otherwise every decision
        is recorded as blocked.
        """
        from ..__main__ import _restored_source
        from ..experiment import PaperExperiment

        source = _restored_source(self.root)
        return PaperExperiment.load(self.root, source=source, inference=inference)

    # -- summary ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The headline view: identity, counts, book and readiness.

        Built from the report so the dashboard and the CLI cannot disagree about
        a figure. Read-only: no session is processed and no file is written.
        """
        experiment = self.load()
        report = build_report(experiment).to_dict()
        raw = self._raw_state()
        refresh = _refresh_view(self.root)
        return {
            "experiment_id": report["experiment_id"],
            "strategy_id": report["strategy_id"],
            "created_at": report["created_at"],
            "generated_at": report["generated_at"],
            "last_session_processed": report["last_session_processed"],
            "counts": report["counts"],
            "orders": report["orders"],
            "book": report["book"],
            "performance": report["performance"],
            "costs": report["costs"],
            "data": report["data"],
            "refresh": refresh,
            "freshness": report.get("freshness"),
            "cadence": report.get("cadence"),
            "warnings": _warnings(report, refresh),
            "baseline": report["baseline"],
            "readiness": report["readiness"],
            "limitations": report["limitations"],
            "journal": report["journal"],
            "resting_orders": [
                _resting_view(order) for order in raw.get("session", {}).get("resting", [])
            ],
            "processed_sessions": raw.get("processed_sessions", []),
            "schema_version": raw.get("schema_version"),
        }

    def decisions(self) -> list[dict[str, Any]]:
        """Every decision, newest first, as JSON-safe records."""
        raw = self._raw_state()
        decisions = list(raw.get("decisions", {}).values())
        decisions.sort(key=lambda d: (d.get("decision_session", ""), d.get("symbol", "")))
        return list(reversed(decisions))

    def journal_tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """The last ``limit`` journal entries, newest first."""
        from ..journal import ExperimentJournal

        path = self.root / "journal.json"
        if not path.exists():
            return []
        journal = ExperimentJournal.load(path)
        entries = list(journal.entries)[-max(1, limit):]
        entries.reverse()
        return [e.to_dict() for e in entries]

    def snapshots(self) -> list[str]:
        snap_dir = self.root / "snapshots"
        if not snap_dir.exists():
            return []
        return sorted(p.name for p in snap_dir.glob("*.json"))


def _warnings(report: dict[str, Any], refresh: dict[str, Any]) -> list[str]:
    """Operator-facing warnings derived from the report, never fabricated.

    These are conditions that are correct but need a human's attention — a stale
    source, a reached decision cap, an idle engine. Each is a plain sentence, so
    the dashboard can show them without interpreting a status code.
    """
    warnings: list[str] = []
    freshness = report.get("freshness") or {}
    if freshness.get("is_stale"):
        warnings.append(f"stale market data: {freshness.get('reason')}")
    if freshness.get("market_status") == "data_source_unavailable" or (
        freshness.get("source_status") == "failed"
    ):
        kind = freshness.get("source_failure_kind") or "unknown"
        warnings.append(
            f"market data source is failing ({kind}): the source could not serve new "
            f"data, so the data is not current — {freshness.get('source_status_detail') or freshness.get('reason')}"
        )
    cadence = report.get("cadence") or {}
    if cadence.get("max_decisions_reached"):
        warnings.append(
            f"max_decisions ({cadence.get('max_decisions')}) reached; "
            "no further decisions will be taken"
        )
    elif cadence.get("max_decisions") and cadence.get("decisions_taken") is not None:
        remaining = int(cadence["max_decisions"]) - int(cadence["decisions_taken"])
        if 0 < remaining <= 10:
            warnings.append(
                f"decision cap nearly reached: {remaining} of "
                f"{cadence['max_decisions']} decisions remaining"
            )
    if refresh.get("last_error"):
        warnings.append(f"last refresh error: {refresh['last_error']}")
    return warnings


def _refresh_view(root: Path) -> dict[str, Any]:
    """The last refresh's status, read from the sidecar; never a live fetch.

    Reports the newest data date and the refresh outcome so an operator can see,
    from the dashboard alone, why the engine is or is not advancing. Read-only and
    secret-free: the sidecar holds no credential.
    """
    from ..refresh import SOURCE_FAILURE_STATUSES, read_refresh_state

    state = read_refresh_state(root)
    source_failed = state.last_status in SOURCE_FAILURE_STATUSES
    return {
        "last_status": state.last_status,
        "last_attempt_at": state.last_attempt_at,
        "last_success_at": state.last_success_at,
        "source_last_bar": state.last_bar_date,
        "last_new_session": state.last_new_session,
        "last_new_sessions": state.last_new_sessions,
        "fetches_today": state.fetches_today,
        "day": state.day,
        "last_error": state.last_error,
        # Source health, kept explicit so a bare "throttled" tick still shows the
        # standing failure behind it rather than hiding it.
        "source_status": "failed" if source_failed else "ok",
        "failure_kind": state.last_failure_kind,
        "last_skip": state.last_skip,
        "last_skip_at": state.last_skip_at,
    }


def _resting_view(order: dict[str, Any]) -> dict[str, Any]:
    """One resting order, reduced to the fields a phone needs to show."""
    return {
        "order_id": order.get("order_id"),
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "order_type": order.get("order_type"),
        "quantity": order.get("quantity"),
        "filled_quantity": order.get("filled_quantity"),
        "remaining_quantity": max(
            0.0, float(order.get("quantity", 0.0)) - float(order.get("filled_quantity", 0.0))
        ),
        "limit_price": order.get("limit_price"),
        "average_fill_price": order.get("average_fill_price"),
        "time_in_force": order.get("time_in_force"),
        "expires_session": order.get("expires_session"),
        "status": order.get("status"),
    }
