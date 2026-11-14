"""The one place the TJR strategy's crypto adaptation is written down.

TJR was written for equities. Exactly **one** of its settings does not transfer
to a 24/7 crypto market: the entry-window cut. On equities TJR stops taking new
entries at 15:30 ET, leaving the last 30 minutes of the session alone. A crypto
session has no bell — its "close" is the end of the UTC day — so the equivalent
cut is **23:30 UTC**, the last 30 minutes before the session rolls.

Every other rule, default and limit is unchanged. This module is the single
source of that adaptation so the **backtest** and the **live paper path** cannot
drift apart: a live path that traded on a different rule than the one it was
backtested on would make the backtest's numbers describe a system that does not
exist.

The session *policy* (which labels a bar's day) is a separate decision and lives
in :mod:`tradingagents.tjr.sessions`; both crypto paths use
:func:`~tradingagents.tjr.sessions.utc_day_session`.
"""

from __future__ import annotations

from tradingagents.tjr.rules import TjrSettings
from tradingagents.tjr.sessions import SessionPolicy, utc_day_session

# The last minute at which a new crypto entry may be taken, in the session's own
# timezone (UTC for the crypto policy). 23:30 leaves the final 30 minutes of the
# UTC day alone, mirroring the equity rule's 15:30 ET cut.
CRYPTO_ENTRY_CUT: tuple[int, int] = (23, 30)


def crypto_tjr_settings() -> TjrSettings:
    """TJR's settings adapted for a 24/7 market.

    The only adaptation is the **entry-window cut**; every other rule and default
    is the equity one, unchanged.
    """
    return TjrSettings(latest_entry_time=CRYPTO_ENTRY_CUT)


def crypto_tjr_policy() -> SessionPolicy:
    """The session policy both crypto paths use: the whole UTC calendar day."""
    return utc_day_session()


__all__ = [
    "CRYPTO_ENTRY_CUT",
    "crypto_tjr_policy",
    "crypto_tjr_settings",
]
