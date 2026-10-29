"""A phone-friendly monitoring and control surface for a paper experiment.

Simulation only. This package adds no broker path, no credential, and no live
trading mode; the only action it can take is advancing
``PaperExperiment.process_session``, which never reaches a venue.
"""

from .auth import (
    TOKEN_ENV,
    DashboardAuthError,
    check_bearer,
    is_loopback,
    require_bindable,
    resolve_token,
)
from .server import Dashboard, serve
from .state import DashboardStateError, ExperimentView

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
