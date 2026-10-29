"""Driving a paper experiment forward, shared by the CLI and the dashboard.

``process_pending`` is the one place sessions are advanced. The CLI's ``run``
command and the dashboard's control action both call it, so a session driven
from the dashboard is processed by exactly the same code — and persisted by the
same ``process_session`` — as one driven from the terminal. There is no second
implementation to drift.

The experiment is simulation-only by construction: ``PaperExperiment`` never
touches a broker, and this module adds no order path of its own.
"""

from __future__ import annotations

from .experiment import PaperExperiment, SessionOutcome


def pending_sessions(experiment: PaperExperiment) -> list[str]:
    """The source's sessions the experiment has not processed yet, in order."""
    return [s for s in experiment.source.sessions() if s not in experiment.processed_sessions]


def process_pending(
    experiment: PaperExperiment, *, sessions: int | None = None
) -> list[SessionOutcome]:
    """Process up to ``sessions`` unprocessed sessions (all of them when ``None``).

    Each session is processed through :meth:`PaperExperiment.process_session`,
    which settles due decisions, works resting orders and persists state after
    the session. ``sessions`` bounds a single call so a caller (the dashboard, a
    cron tick) can make progress without a long, uninterruptible loop.
    """
    remaining = pending_sessions(experiment)
    if sessions is not None:
        if sessions < 0:
            raise ValueError("sessions must not be negative")
        remaining = remaining[:sessions]
    return [experiment.process_session(session) for session in remaining]
