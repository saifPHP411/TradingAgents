"""A phone-friendly monitoring and control surface for a paper experiment.

Simulation only. This package adds no broker path, no credential, and no live
trading mode; the only action it can take is advancing
``PaperExperiment.process_session``, which never reaches a venue.

``Dashboard`` and ``serve`` are re-exported lazily (PEP 562) rather than imported
here. Importing ``.server`` at package-import time put it in ``sys.modules``
before ``python -m ...dashboard.server`` could execute it, which made runpy warn
that the module was "found in sys.modules ... prior to execution"; the lazy
attribute keeps ``from ...dashboard import Dashboard, serve`` working without
that eager import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .auth import (
    TOKEN_ENV,
    DashboardAuthError,
    check_bearer,
    is_loopback,
    require_bindable,
    resolve_token,
)
from .state import DashboardStateError, ExperimentView

if TYPE_CHECKING:  # for type checkers only; no runtime import of .server
    from .server import Dashboard, serve

__all__ = [
    "TOKEN_ENV",
    "Dashboard",
    "DashboardAuthError",
    "DashboardStateError",
    "ExperimentView",
    "check_bearer",
    "is_loopback",
    "require_bindable",
    "resolve_token",
    "serve",
]


def __getattr__(name: str) -> Any:
    """Resolve ``Dashboard``/``serve`` from ``.server`` on first access."""
    if name in ("Dashboard", "serve"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
