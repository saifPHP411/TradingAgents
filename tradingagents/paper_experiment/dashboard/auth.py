"""Bearer-token auth for the paper-experiment dashboard.

The dashboard can read an experiment's state and (when explicitly enabled) drive
more simulated sessions. Either is worth protecting, so a token is required
whenever the server is reachable from outside the loopback interface. Binding to
a non-loopback host with no token is refused rather than served open: an
unauthenticated control surface on a public host is the failure this module
exists to prevent.

The token comes from ``TRADINGAGENTS_DASHBOARD_TOKEN``. It is compared in
constant time and never logged, echoed, or written to a report.
"""

from __future__ import annotations

import hmac
import os

TOKEN_ENV = "TRADINGAGENTS_DASHBOARD_TOKEN"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class DashboardAuthError(RuntimeError):
    """The dashboard was asked to serve in a configuration it must refuse."""


def is_loopback(host: str) -> bool:
    """Whether ``host`` is the loopback interface (no external reach)."""
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def resolve_token(token: str | None = None) -> str:
    """The dashboard token, from the argument or the environment.

    A blank value is treated as absent, so an exported-but-empty variable does
    not silently disable auth.
    """
    if token is not None:
        return token.strip()
    return os.environ.get(TOKEN_ENV, "").strip()


def require_bindable(host: str, token: str) -> None:
    """Refuse a non-loopback bind that has no token.

    Loopback is allowed without a token because it is not reachable from another
    host. Anything else must be authenticated.
    """
    if is_loopback(host):
        return
    if not token:
        raise DashboardAuthError(
            f"refusing to bind {host!r} without a token: set {TOKEN_ENV} (or pass "
            "--token) so the dashboard is not served unauthenticated on a public "
            "interface. Use --host 127.0.0.1 for a local-only dashboard."
        )


def check_bearer(header_value: str | None, token: str) -> bool:
    """Whether a request's ``Authorization`` header carries the expected token.

    Accepts ``Bearer <token>``. Compared with :func:`hmac.compare_digest` so a
    wrong token cannot be recovered by timing. A missing/blank expected token
    denies every request — an unset token must never mean "allow all".
    """
    if not token:
        return False
    if not header_value:
        return False
    parts = header_value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return hmac.compare_digest(parts[1].strip(), token)
