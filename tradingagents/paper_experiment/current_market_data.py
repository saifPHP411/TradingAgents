"""Real current-market *decision* data: the OHLCV window the strategy reasons on.

The current-market path and the historical path differ in exactly one place —
where the decision's input window comes from. The historical path slices the
experiment's point-in-time dataset; this module fetches the *current* daily bars
from Alpaca and validates them before they can reach the strategy.

Two prices are deliberately kept apart, and this module supplies only the first:

- **decision data** (here): the daily OHLCV window the strategy/indicators need,
  fetched from ``https://data.alpaca.markets`` on the IEX feed. It is *not* the
  execution price.
- **execution price**: the current price the worker sizes and submits an order
  from. That stays on the existing ``MarketData`` seam
  (``AlpacaSnapshotMarketData``), untouched by this module.

Safety properties, each pinned by a test:

- the host is fixed to Alpaca's **market-data** host; the paper trading host and
  every live host are refused, so decision data can never be pointed at orders;
- the feed is ``iex``, the only feed the Basic plan can read;
- a frame is validated through the shared validator, so a malformed, non-finite,
  duplicate, out-of-order or non-positive payload is refused rather than fed to
  the strategy;
- a **stale** newest bar (older than the freshness limit relative to the decision
  session) is refused, so the strategy never reasons on a bar that is not current;
- a **future-dated** bar (dated after the decision session) is refused;
- the fetch is bounded (one page, a timeout) and never retries silently.

The transport is injectable, so every branch is exercised with a mocked response
and no socket.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .providers.alpaca_iex import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DATA_BASE_URL,
    FEED,
    REQUEST_TIMEOUT,
    _auth_headers,
    _classify_status,
    assert_data_host,
    build_request_params,
)
from .providers.base import ProviderError, fetch_json
from .sources import BAR_COLUMNS, _validate_frame

DEFAULT_LOOKBACK_DAYS = 400
"""Calendar days of history to request, so SMA200 and the indicator block have a
window to compute from. Mirrors the historical Alpaca source's own lookback."""

DEFAULT_MAX_STALE_DAYS = 4
"""How many calendar days a newest bar may lag the decision session before it is
refused as stale. Four covers a normal weekend plus a market holiday; beyond it
the data is not current and the strategy must not reason on it."""


class CurrentMarketDataError(RuntimeError):
    """The current decision data could not be served. Never returns partial data."""

    def __init__(self, detail: str, *, kind: str = "unavailable") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(detail)

    @property
    def code(self) -> str:
        """The worker's no-trade code for this failure.

        Every current-decision-data failure — stale, malformed, missing,
        future-dated, unavailable — is a *market data* failure, so the code is
        the same closed-vocabulary value; ``kind`` carries the finer reason for a
        dashboard's detail line.
        """
        from .broker.trading import NO_TRADE_MARKET_DATA_UNAVAILABLE

        return NO_TRADE_MARKET_DATA_UNAVAILABLE


@dataclass(frozen=True)
class CurrentWindow:
    """A validated current-market decision window, with its provenance.

    ``session`` is the decision session (the market clock's own trading day),
    which is also the ``as_of_date`` the resulting decision is dated to.
    ``content_hash`` fingerprints the window so a decision_id can carry a stable
    provenance without depending on the (intraday-varying) in-progress bar.
    """

    symbol: str
    session: str
    frame: pd.DataFrame
    last_bar_date: str
    fetched_at: str
    provider: str
    content_hash: str

    @property
    def bar_count(self) -> int:
        return int(len(self.frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session": self.session,
            "last_bar_date": self.last_bar_date,
            "bar_count": self.bar_count,
            "fetched_at": self.fetched_at,
            "provider": self.provider,
            "content_hash": self.content_hash,
        }


@runtime_checkable
class CurrentMarketData(Protocol):
    """Fetches the current decision window for a symbol at a decision session."""

    def fetch(self, symbol: str, *, session: str) -> CurrentWindow:  # pragma: no cover
        ...

    def describe(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


def _content_hash(frame: pd.DataFrame) -> str:
    """A stable fingerprint of the window's OHLC, independent of pandas internals."""
    digest = hashlib.sha256()
    for index, row in frame.iterrows():
        digest.update(str(index)[:10].encode())
        for column in BAR_COLUMNS[:4]:
            digest.update(f"{column}={float(row[column]):.10f}".encode())
    return digest.hexdigest()


class AlpacaCurrentBars:
    """Current daily OHLCV bars from Alpaca market data, validated and freshness-checked."""

    provider_name = "alpaca_current_bars"
    kind = "vendor_daily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DATA_BASE_URL,
        transport: Callable[..., Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._base_url = assert_data_host(base_url)
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        self._key = key.strip()
        self._secret = secret.strip()
        self._timeout = timeout
        self._transport = transport
        self._lookback_days = max(int(lookback_days), 1)
        self._max_stale_days = max_stale_days
        self._clock = clock or (lambda: datetime.now(UTC))
        # The last successful window and the last failure, so a status view can
        # report what was actually fetched without a second request.
        self.last_window: CurrentWindow | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_attempt_at: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        window = self.last_window
        age_days = None
        if window is not None:
            session = _parse_session(window.session)
            bar_date = _parse_session(window.last_bar_date)
            if session is not None and bar_date is not None:
                age_days = (session - bar_date).days
        return {
            "decision_data_provider": self.provider_name,
            "decision_data_kind": self.kind,
            "decision_data_timeframe": "1Day",
            "decision_data_endpoint": self._base_url,
            "decision_data_feed": FEED,
            "decision_data_lookback_days": self._lookback_days,
            "decision_data_max_stale_days": self._max_stale_days,
            "decision_data_last_fetch_at": self.last_attempt_at,
            "decision_data_last_bar_date": window.last_bar_date if window else None,
            "decision_data_last_session": window.session if window else None,
            "decision_data_age_days": age_days,
            "decision_data_bar_count": window.bar_count if window else None,
            "decision_data_last_error": self.last_error,
        }

    # -- fetch ------------------------------------------------------------

    def fetch(self, symbol: str, *, session: str) -> CurrentWindow:
        """Fetch and validate the current window for ``symbol`` at ``session``.

        ``session`` is the decision session (the market clock's trading day). A
        newest bar older than the freshness limit, or dated after ``session``, is
        refused. Raises :class:`CurrentMarketDataError` on any failure; it never
        returns a partial or stale frame.
        """
        symbol = symbol.upper()
        reference = _parse_session(session)
        if reference is None:
            raise CurrentMarketDataError(
                f"the decision session {session!r} is not a usable date",
                kind="malformed",
            )
        self.last_attempt_at = self._clock().isoformat()
        if not self._key or not self._secret:
            raise self._fail(
                f"no Alpaca market-data credentials: set {API_KEY_ENV} and "
                f"{API_SECRET_ENV} so current decision data can be fetched",
                kind="not_configured",
            )

        start = date.fromordinal(reference.toordinal() - self._lookback_days)
        try:
            payload = fetch_json(
                url=f"{self._base_url}/v2/stocks/bars",
                params=build_request_params(symbol=symbol, start=start, end=reference),
                timeout=self._timeout,
                secrets=(self._key, self._secret),
                headers=_auth_headers(self._key, self._secret),
                transport=self._transport,
                classify_status=_classify_status,
            )
        except ProviderError as exc:
            raise self._fail(
                f"Alpaca current bars could not be read: {exc}", kind="transport"
            ) from None
        except Exception as exc:  # noqa: BLE001 - any failure is unavailable data
            raise self._fail(
                f"Alpaca current bars could not be read: {type(exc).__name__}",
                kind="transport",
            ) from None

        frame, last_bar = self._parse(payload, symbol)
        self._assert_fresh(last_bar, reference)
        window = CurrentWindow(
            symbol=symbol,
            session=session,
            frame=frame,
            last_bar_date=last_bar,
            fetched_at=self._clock().isoformat(),
            provider=self.provider_name,
            content_hash=_content_hash(frame),
        )
        self.last_window = window
        self.last_error = None
        self.last_error_kind = None
        return window

    # -- internals --------------------------------------------------------

    def _fail(self, detail: str, *, kind: str) -> CurrentMarketDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return CurrentMarketDataError(detail, kind=kind)

    def _parse(self, payload: dict, symbol: str) -> tuple[pd.DataFrame, str]:
        bars_by_symbol = payload.get("bars")
        if not isinstance(bars_by_symbol, dict):
            raise self._fail("Alpaca returned no 'bars' object", kind="malformed")
        raw = bars_by_symbol.get(symbol)
        if raw is None:
            raise self._fail(
                f"Alpaca returned no bars for {symbol!r}; the symbol may be "
                "unsupported or the window empty",
                kind="unsupported",
            )
        if not isinstance(raw, list) or not raw:
            raise self._fail(
                f"Alpaca returned an empty bar series for {symbol!r}", kind="no_data"
            )
        rows: dict[str, dict] = {}
        for bar in raw:
            if not isinstance(bar, dict):
                raise self._fail("an Alpaca bar is not an object", kind="malformed")
            stamp = bar.get("t")
            if not isinstance(stamp, str) or not stamp:
                raise self._fail("an Alpaca bar is missing its timestamp", kind="malformed")
            day = stamp[:10]
            if day in rows:
                raise self._fail(
                    f"Alpaca returned the session {day} more than once for {symbol!r}",
                    kind="malformed",
                )
            try:
                rows[day] = {
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                }
            except (KeyError, TypeError, ValueError):
                raise self._fail(
                    f"Alpaca's bar for {day} is malformed", kind="malformed"
                ) from None

        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index = pd.to_datetime(frame.index, format="%Y-%m-%d", errors="coerce")
        frame = frame.sort_index(kind="stable")
        try:
            frame = _validate_frame(frame[list(BAR_COLUMNS)], symbol, self.kind)
        except CurrentMarketDataError:
            raise
        except Exception as exc:  # a malformed vendor payload is refused, not served
            raise self._fail(
                f"Alpaca's current bars for {symbol!r} failed validation: {exc}",
                kind="malformed",
            ) from None
        return frame, str(frame.index[-1])[:10]

    def _assert_fresh(self, last_bar: str, reference: date) -> None:
        """Refuse a newest bar that lags the session or is dated in the future."""
        bar_date = _parse_session(last_bar)
        if bar_date is None:
            raise self._fail(
                f"Alpaca's newest bar date {last_bar!r} is unparseable", kind="malformed"
            )
        if bar_date > reference:
            raise self._fail(
                f"Alpaca's newest bar {last_bar} is dated after the decision session "
                f"{reference.isoformat()}; refusing future-dated data",
                kind="future",
            )
        if self._max_stale_days is not None:
            age = (reference - bar_date).days
            if age > self._max_stale_days:
                raise self._fail(
                    f"Alpaca's newest bar {last_bar} is {age} day(s) before the "
                    f"decision session {reference.isoformat()}, beyond the "
                    f"{self._max_stale_days}-day freshness limit",
                    kind="stale",
                )


def _parse_session(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


class StaticCurrentBars:
    """A fixed current-window provider, for tests and offline demonstration.

    Explicitly *not* a live vendor: it returns whatever frame it was built with,
    so a test can pin the valid, stale, malformed and future branches without a
    socket. It still validates and freshness-checks through the same rules, so a
    test exercises the real code path rather than a bypass.
    """

    provider_name = "static_current_bars"
    kind = "test"

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._frames = {k.upper(): v for k, v in frames.items()}
        self._max_stale_days = max_stale_days
        self._clock = clock or (lambda: datetime.now(UTC))
        self.last_window: CurrentWindow | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_attempt_at: str | None = None

    def describe(self) -> dict[str, Any]:
        window = self.last_window
        age_days = None
        if window is not None:
            session = _parse_session(window.session)
            bar_date = _parse_session(window.last_bar_date)
            if session is not None and bar_date is not None:
                age_days = (session - bar_date).days
        return {
            "decision_data_provider": self.provider_name,
            "decision_data_kind": self.kind,
            "decision_data_timeframe": "1Day",
            "decision_data_max_stale_days": self._max_stale_days,
            "decision_data_last_fetch_at": self.last_attempt_at,
            "decision_data_last_bar_date": window.last_bar_date if window else None,
            "decision_data_last_session": window.session if window else None,
            "decision_data_age_days": age_days,
            "decision_data_bar_count": window.bar_count if window else None,
            "decision_data_last_error": self.last_error,
        }

    def fetch(self, symbol: str, *, session: str) -> CurrentWindow:
        symbol = symbol.upper()
        reference = _parse_session(session)
        self.last_attempt_at = self._clock().isoformat()
        if reference is None:
            raise self._error(f"the decision session {session!r} is not usable", "malformed")
        frame = self._frames.get(symbol)
        if frame is None or len(frame) == 0:
            raise self._error(f"no current bars for {symbol!r}", "no_data")
        try:
            validated = _validate_frame(frame.copy(), symbol, self.kind)
        except Exception as exc:
            raise self._error(
                f"current bars for {symbol!r} failed validation: {exc}", "malformed"
            ) from None
        last_bar = str(validated.index[-1])[:10]
        bar_date = _parse_session(last_bar)
        if bar_date is None:
            raise self._error(f"the newest bar date {last_bar!r} is unparseable", "malformed")
        if bar_date > reference:
            raise self._error(
                f"the newest bar {last_bar} is after the decision session "
                f"{reference.isoformat()}; refusing future-dated data",
                "future",
            )
        if self._max_stale_days is not None and (reference - bar_date).days > self._max_stale_days:
            raise self._error(
                f"the newest bar {last_bar} is {(reference - bar_date).days} day(s) "
                f"before the session {reference.isoformat()}, beyond the "
                f"{self._max_stale_days}-day limit",
                "stale",
            )
        window = CurrentWindow(
            symbol=symbol,
            session=session,
            frame=validated,
            last_bar_date=last_bar,
            fetched_at=self._clock().isoformat(),
            provider=self.provider_name,
            content_hash=_content_hash(validated),
        )
        self.last_window = window
        self.last_error = None
        self.last_error_kind = None
        return window

    def _error(self, detail: str, kind: str) -> CurrentMarketDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return CurrentMarketDataError(detail, kind=kind)


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MAX_STALE_DAYS",
    "AlpacaCurrentBars",
    "CurrentMarketData",
    "CurrentMarketDataError",
    "CurrentWindow",
    "StaticCurrentBars",
]
