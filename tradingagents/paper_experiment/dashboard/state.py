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

    def load(self):
        """Load the experiment, restoring its price source without a network call."""
        from ..__main__ import _restored_source
        from ..experiment import PaperExperiment

        source = _restored_source(self.root)
        return PaperExperiment.load(self.root, source=source)

    # -- summary ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The headline view: identity, counts, book and readiness.

        Built from the report so the dashboard and the CLI cannot disagree about
        a figure. Read-only: no session is processed and no file is written.
        """
        experiment = self.load()
        report = build_report(experiment).to_dict()
        raw = self._raw_state()
        return {
            "experiment_id": report["experiment_id"],
            "strategy_id": report["strategy_id"],
            "created_at": report["created_at"],
            "generated_at": report["generated_at"],
            "last_session_processed": report["last_session_processed"],
            "counts": report["counts"],
            "book": report["book"],
            "performance": report["performance"],
            "costs": report["costs"],
            "data": report["data"],
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
