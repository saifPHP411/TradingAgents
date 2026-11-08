"""Whether the US equity market is open, from Alpaca's own clock.

A current-market decision must not be taken when the market is closed, and it
must not assume the market is open. The exchange calendar (weekends, holidays,
half-days) is not something this code can compute reliably, so it asks Alpaca:
``GET https://paper-api.alpaca.markets/v2/clock`` returns the market's own
``timestamp``, ``is_open``, ``next_open`` and ``next_close``.

Safety properties, each deliberate:

- the clock is read from the **paper** host only (``assert_paper_endpoint``), the
  same host the broker uses, with the same paper credentials. It never touches a
  live host and never places an order — it is a single read-only ``GET``;
- an unreachable clock is **fail-closed**: the market state is *unknown*, which
  is reported as such rather than assumed open. Deciding on an unknown calendar
  is exactly the "assume the market is open" defect this module exists to avoid;
- the session date is the clock's own local (US/Eastern) calendar day, which is
  the same day a daily bar is stamped with, so the session the gate reports and
  the session a bar carries cannot disagree by a timezone offset.

The transport is injectable, so every branch is exercised with a mocked response
and no socket. The default transport is ``requests``, through the shared
:func:`~.providers.base.fetch_json` helper (bounded timeout, credential
redaction).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

from .broker.alpaca_paper import (
    API_KEY_ENV,
    API_SECRET_ENV,
    PAPER_BASE_URL,
    assert_paper_endpoint,
    assert_paper_request_url,
)
from .providers.base import ProviderError, fetch_json

MARKET_OPEN = "open"
MARKET_CLOSED = "closed"
MARKET_UNKNOWN = "unknown"
"""The closed vocabulary of market-hours states. ``unknown`` is the fail-closed
result of an unreachable clock — never conflated with ``closed`` or ``open``."""

MARKET_HOURS_STATES = (MARKET_OPEN, MARKET_CLOSED, MARKET_UNKNOWN)

REQUEST_TIMEOUT = 15.0


class MarketHoursError(RuntimeError):
    """The market clock could not be read, so the market state is unknown."""


@dataclass(frozen=True)
class MarketHoursStatus:
    """The market's state at one instant, as Alpaca reports it."""

    state: str
    session: str | None
    """The US/Eastern calendar day the clock reports, ``YYYY-MM-DD``, or ``None``.

    This is the session a current decision is dated to, and the session a daily
    bar for the in-progress day is stamped with, so the two agree.
    """
    is_open: bool | None
    next_open: str | None
    next_close: str | None
    observed_at: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "session": self.session,
            "is_open": self.is_open,
            "next_open": self.next_open,
            "next_close": self.next_close,
            "observed_at": self.observed_at,
            "detail": self.detail,
        }


@runtime_checkable
class MarketHoursGate(Protocol):
    """Reports whether the market is open, or that it cannot be determined."""

    def status(self) -> MarketHoursStatus:  # pragma: no cover - protocol
        ...

    def describe(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


def _parse_clock_timestamp(value: Any) -> datetime | None:
    """Parse Alpaca's ISO-8601 clock timestamp to an aware datetime, or ``None``.

    The offset is preserved, so the *local* (US/Eastern) calendar day — the day a
    daily bar is stamped with — can be read from it. A naive value is refused
    rather than guessed at UTC, because guessing would shift the session.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    # Alpaca's nanosecond precision exceeds datetime's microseconds; trim it.
    if "." in text:
        head, _, tail = text.partition(".")
        offset = ""
        for index, char in enumerate(tail):
            if char in "+-":
                offset = tail[index:]
                break
        digits = tail[: index] if offset else tail
        text = f"{head}.{digits[:6]}{offset}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


class AlpacaMarketClock:
    """The market clock, read from Alpaca's paper host. Never places an order."""

    provider_name = "alpaca_clock"
    kind = "market_hours"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = PAPER_BASE_URL,
        transport: Callable[..., Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        # Endpoint first: refuse a live host before any credential is read.
        self._base_url = assert_paper_endpoint(base_url)
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        self._key = key.strip()
        self._secret = secret.strip()
        self._timeout = timeout
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }

    def _classify_status(self, status_code: int, body: str) -> None:
        if status_code in (401, 403):
            raise MarketHoursError(
                f"Alpaca refused the market clock (HTTP {status_code}): {body}"
            )

    def status(self) -> MarketHoursStatus:
        """Read the clock once. Raises :class:`MarketHoursError` when unreadable."""
        if not self._key or not self._secret:
            raise MarketHoursError(
                f"no Alpaca paper credentials: set {API_KEY_ENV} and {API_SECRET_ENV} "
                "so the market clock can be read"
            )
        # The full request URL is re-checked against the approved paper host
        # before it is sent, so a request can never leave for another host.
        url = assert_paper_request_url(f"{self._base_url}/v2/clock")
        try:
            payload = fetch_json(
                url=url,
                params={},
                timeout=self._timeout,
                secrets=(self._key, self._secret),
                headers=self._headers(),
                transport=self._transport,
                classify_status=self._classify_status,
            )
        except MarketHoursError:
            raise
        except ProviderError as exc:
            raise MarketHoursError(f"the market clock could not be read: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - any failure is an unknown state
            raise MarketHoursError(
                f"the market clock could not be read: {type(exc).__name__}"
            ) from None

        observed_at = datetime.now(UTC).isoformat()
        is_open = payload.get("is_open")
        if not isinstance(is_open, bool):
            raise MarketHoursError("Alpaca's clock did not report a boolean 'is_open'")
        stamp = _parse_clock_timestamp(payload.get("timestamp"))
        if stamp is None:
            # Without an offset-aware timestamp the US/Eastern session day cannot
            # be read, so the decision cannot be dated. Fail closed rather than
            # guess the session.
            raise MarketHoursError(
                "Alpaca's clock did not report an offset-aware 'timestamp', so the "
                "trading session cannot be determined"
            )
        session = stamp.date().isoformat()
        return MarketHoursStatus(
            state=MARKET_OPEN if is_open else MARKET_CLOSED,
            session=session,
            is_open=is_open,
            next_open=_iso_or_none(payload.get("next_open")),
            next_close=_iso_or_none(payload.get("next_close")),
            observed_at=observed_at,
            detail=None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "market_hours_provider": self.provider_name,
            "market_hours_kind": self.kind,
            "market_hours_endpoint": self._base_url,
        }


def _iso_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class StaticMarketHours:
    """A fixed market-hours gate, for tests and offline demonstration.

    Explicitly *not* a live calendar: it answers with whatever it was built with,
    so a test can pin the closed, open and unknown branches deterministically.
    """

    provider_name = "static_clock"
    kind = "market_hours"

    def __init__(
        self,
        *,
        is_open: bool,
        session: str,
        next_open: str | None = None,
        next_close: str | None = None,
        error: str | None = None,
    ) -> None:
        self._is_open = is_open
        self._session = session
        self._next_open = next_open
        self._next_close = next_close
        self._error = error

    def status(self) -> MarketHoursStatus:
        if self._error is not None:
            raise MarketHoursError(self._error)
        return MarketHoursStatus(
            state=MARKET_OPEN if self._is_open else MARKET_CLOSED,
            session=self._session,
            is_open=self._is_open,
            next_open=self._next_open,
            next_close=self._next_close,
            observed_at=datetime.now(UTC).isoformat(),
            detail=None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "market_hours_provider": self.provider_name,
            "market_hours_kind": self.kind,
            "market_hours_session": self._session,
            "market_hours_is_open": self._is_open,
        }


def session_date(value: str | date | None) -> date | None:
    """A session string as a date, or ``None`` when it is absent/unparseable."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


__all__ = [
    "MARKET_CLOSED",
    "MARKET_HOURS_STATES",
    "MARKET_OPEN",
    "MARKET_UNKNOWN",
    "REQUEST_TIMEOUT",
    "AlpacaMarketClock",
    "MarketHoursError",
    "MarketHoursGate",
    "MarketHoursStatus",
    "StaticMarketHours",
    "session_date",
]
