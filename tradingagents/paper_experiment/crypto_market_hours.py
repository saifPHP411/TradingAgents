"""The market-hours gate for a 24/7 crypto market.

The equity path (:mod:`.market_hours`) asks Alpaca's clock whether the US market is
open, because equities close overnight, on weekends and on holidays. Crypto does
not: an exchange is always open, so a "closed" gate would be wrong and an
"assume open" gate would be the defect the equity module exists to avoid.

This gate therefore answers ``open`` — but it *derives* that answer from a clock it
can read, and fails **closed** (``unknown``) when the clock cannot be read, exactly
like the equity gate. A gate that returned ``open`` unconditionally would report a
market state it never verified; this one reports the state of a market that is
structurally always open, and still refuses to guess when it cannot tell the time.

The **session** it reports is the UTC calendar day, which is the one boundary every
exchange and every vendor agrees on. It is not presented as a market open — it is
the period a day's structure is measured over, matching
:func:`tradingagents.tjr.sessions.utc_day_session`.

The clock is injectable, so the gate is exercised offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .market_hours import MARKET_OPEN, MARKET_UNKNOWN, MarketHoursStatus

__all__ = ["CryptoMarketHours", "StaticCryptoMarketHours"]


def _utc_session(stamp: datetime) -> str:
    return stamp.astimezone(UTC).strftime("%Y-%m-%d")


@dataclass
class CryptoMarketHours:
    """Crypto's always-open market clock, with a fail-closed session date.

    ``clock`` returns the current instant; if it raises, the state is ``unknown``
    (never ``open``), so a broken clock cannot be read as a tradable market.
    """

    clock: Callable[[], datetime] | None = None
    provider_name: str = "crypto_utc_clock"
    kind: str = "market_hours"

    def __post_init__(self) -> None:
        self._clock = self.clock or (lambda: datetime.now(UTC))

    def status(self) -> MarketHoursStatus:
        try:
            now = self._clock()
        except Exception as exc:  # noqa: BLE001 - any clock failure is unknown
            return MarketHoursStatus(
                state=MARKET_UNKNOWN,
                session=None,
                is_open=None,
                next_open=None,
                next_close=None,
                observed_at=datetime.now(UTC).isoformat(),
                detail=f"the clock could not be read: {type(exc).__name__}",
            )
        if now.tzinfo is None:
            return MarketHoursStatus(
                state=MARKET_UNKNOWN,
                session=None,
                is_open=None,
                next_open=None,
                next_close=None,
                observed_at=datetime.now(UTC).isoformat(),
                detail="the clock returned a naive timestamp; the session cannot be dated",
            )
        return MarketHoursStatus(
            state=MARKET_OPEN,
            session=_utc_session(now),
            is_open=True,
            next_open=None,
            next_close=None,
            observed_at=now.astimezone(UTC).isoformat(),
            detail="crypto trades continuously; the session is the UTC calendar day",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "market_hours_provider": self.provider_name,
            "market_hours_kind": self.kind,
            "market_hours_always_open": True,
            "market_hours_session_basis": "utc_day",
        }


class StaticCryptoMarketHours:
    """A fixed crypto clock, for tests and offline demonstration.

    It is *not* a live calendar: it answers with whatever it was built with, so a
    test can pin the session and the unknown branch deterministically.
    """

    provider_name = "static_crypto_clock"
    kind = "market_hours"

    def __init__(self, *, session: str, error: str | None = None) -> None:
        self._session = session
        self._error = error

    def status(self) -> MarketHoursStatus:
        if self._error is not None:
            from .market_hours import MarketHoursError

            raise MarketHoursError(self._error)
        return MarketHoursStatus(
            state=MARKET_OPEN,
            session=self._session,
            is_open=True,
            next_open=None,
            next_close=None,
            observed_at=datetime.now(UTC).isoformat(),
            detail=None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "market_hours_provider": self.provider_name,
            "market_hours_kind": self.kind,
            "market_hours_always_open": True,
            "market_hours_session": self._session,
        }
