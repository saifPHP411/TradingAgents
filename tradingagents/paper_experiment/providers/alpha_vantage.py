"""Alpha Vantage daily OHLCV, for the multi-session window a paper run needs.

Facts this module is built on, taken from Alpha Vantage's own documentation and
support page (not assumed):

- ``TIME_SERIES_DAILY`` returns raw (as-traded) daily OHLCV and is available on
  the free plan with ``outputsize=compact`` (the latest 100 data points).
  ``outputsize=full`` (25+ years) requires a premium plan, so ``full`` is refused
  unless the caller passes ``allow_premium_outputsize=True`` and thereby accepts
  a paid call.
- ``TIME_SERIES_DAILY_ADJUSTED`` is a **premium** endpoint, so it is not used.
  Because ``TIME_SERIES_DAILY`` is unadjusted, this source is unadjusted and says
  so (``adjusted=False``); it never mixes adjusted and unadjusted bars.
- The free plan allows 25 requests per day. Alpha Vantage reports a quota or a bad
  key inside an **HTTP 200** body (``"Information"`` / ``"Note"``), and a rejected
  call as ``"Error Message"``; each is classified and raised, never parsed as data.

The provider fetches once, at construction, reorders the vendor's newest-first
series to oldest-first (the price-source contract), validates the frame through
the shared validator, and freezes it. It therefore never sees a bar after the
fetch, and a requested session the frame does not cover is refused rather than
answered with a neighbouring day's price.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pandas as pd

from ..sources import (
    BAR_COLUMNS,
    FramePriceSource,
    PriceSourceError,
    SourceProvenance,
    _validate_frame,
    bar_is_fresh,
)
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderStaleDataError,
    ProviderUnsupportedError,
    Transport,
    fetch_json,
)

API_BASE_URL = "https://www.alphavantage.co/query"
API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
REQUEST_TIMEOUT = 30.0

# Free-plan function. The adjusted variant is premium and is not used.
_FUNCTION = "TIME_SERIES_DAILY"

# Alpha Vantage's daily series carries US exchange session dates.
_SERIES_TIMEZONE = "US/Eastern"

# A newest bar older than this many calendar days (from the fetch date) is not
# "recent data"; the fetch is refused rather than served as if current.
DEFAULT_MAX_STALE_DAYS = 10


def _field(row: dict, *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise ProviderError(f"a daily bar is missing field {names[0]!r}")


def build_request_params(*, symbol: str, api_key: str, outputsize: str = "compact") -> dict:
    """The exact query parameters sent to Alpha Vantage, credential included.

    One source of truth for the outgoing request, so a diagnostic or a test can
    assert that the credential reaches the ``apikey`` parameter without a second
    copy of the mapping drifting from the real call. The returned mapping holds
    the secret; callers must not log or print it.
    """
    return {
        "function": _FUNCTION,
        "symbol": symbol.upper(),
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": api_key,
    }


def _redact(text: str, secret: str) -> str:
    """Remove the credential from a vendor message before it can be raised.

    Only a plausibly key-length secret is replaced: a very short secret would
    match ordinary words (a one-character key would turn "apikey" into "api***ey"),
    mangling the message without protecting anything real.
    """
    if secret and len(secret) >= 8 and secret in text:
        return text.replace(secret, "***")
    return text


def _classify_notice(notice: str, *, secret: str) -> None:
    """Raise the right error for an Alpha Vantage 200-body notice.

    Classification reads the *raw* notice, so redaction cannot hide a marker; the
    message that is raised carries the redacted text.
    """
    low = notice.lower()
    message = _redact(notice, secret)
    if any(
        marker in low
        for marker in (
            "rate limit",
            "requests per day",
            "call frequency",
            "per minute",
            "premium",
            "thank you for using",
        )
    ):
        raise ProviderRateLimitedError(f"Alpha Vantage quota/limit reached: {message}")
    if "api key" in low or "apikey" in low:
        raise ProviderAuthError(f"Alpha Vantage rejected the credential: {message}")
    raise ProviderUnsupportedError(f"Alpha Vantage could not serve the request: {message}")


class AlphaVantageDailySource(FramePriceSource):
    """Daily OHLCV bars from Alpha Vantage's free ``TIME_SERIES_DAILY`` function."""

    kind = "vendor_daily"

    def __init__(
        self,
        symbol: str,
        *,
        api_key: str | None = None,
        outputsize: str = "compact",
        allow_premium_outputsize: bool = False,
        timeout: float = REQUEST_TIMEOUT,
        transport: Transport | None = None,
        max_stale_days: int | None = DEFAULT_MAX_STALE_DAYS,
        today: date | None = None,
        name: str = "alpha_vantage_daily",
    ) -> None:
        if outputsize not in ("compact", "full"):
            raise ProviderError(f"outputsize must be 'compact' or 'full', got {outputsize!r}")
        if outputsize == "full" and not allow_premium_outputsize:
            raise ProviderUnsupportedError(
                "outputsize='full' is a premium Alpha Vantage feature; pass "
                "allow_premium_outputsize=True to accept a paid call, or keep the "
                "free 'compact' size (latest 100 data points)"
            )
        key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise ProviderNotConfiguredError(
                f"no Alpha Vantage credential: set the {API_KEY_ENV} environment variable"
            )
        self.symbol = symbol.upper()

        payload = fetch_json(
            url=API_BASE_URL,
            params=build_request_params(
                symbol=self.symbol, api_key=key, outputsize=outputsize
            ),
            timeout=timeout,
            secrets=(key,),
            transport=transport,
        )
        frame, last_bar, meta = self._parse(payload, key=key, outputsize=outputsize)

        fetched_at = datetime.now(UTC)
        self._assert_recent(last_bar, fetched_at, max_stale_days=max_stale_days, today=today)

        provenance = SourceProvenance(
            name=name,
            kind=self.kind,
            is_current_market=_is_recent(last_bar, today or fetched_at.date(), max_stale_days),
            symbol=self.symbol,
            fetched_at=fetched_at.isoformat(),
            last_bar_date=last_bar,
            timezone=meta.get("timezone") or _SERIES_TIMEZONE,
            adjusted=False,
            dataset_start=str(frame.index[0])[:10],
            dataset_end=str(frame.index[-1])[:10],
            max_stale_days=max_stale_days,
            notes=(
                "Alpha Vantage TIME_SERIES_DAILY (free plan, unadjusted, as-traded)",
                "quota: 25 requests/day on the free plan",
            ),
        )
        super().__init__({self.symbol: frame}, name=name, provenance=provenance)

    # -- parsing ----------------------------------------------------------

    def _parse(
        self, payload: dict, *, key: str, outputsize: str
    ) -> tuple[pd.DataFrame, str, dict]:
        notice = payload.get("Information") or payload.get("Note")
        if notice:
            _classify_notice(str(notice), secret=key)
        rejection = payload.get("Error Message")
        if rejection:
            raise ProviderUnsupportedError(
                f"Alpha Vantage rejected the request for {self.symbol!r}: "
                f"{_redact(str(rejection), key)}"
            )

        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            raise ProviderError(
                f"Alpha Vantage returned no daily series for {self.symbol!r}"
            )

        meta = payload.get("Meta Data") or {}
        rows: dict[str, dict] = {}
        for session, bar in series.items():
            if not isinstance(bar, dict):
                raise ProviderError(f"Alpha Vantage bar for {session} is not an object")
            try:
                rows[session] = {
                    "open": float(_field(bar, "1. open")),
                    "high": float(_field(bar, "2. high")),
                    "low": float(_field(bar, "3. low")),
                    "close": float(_field(bar, "4. close")),
                    "volume": float(_field(bar, "5. volume")),
                }
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    f"Alpha Vantage bar for {session} is malformed: {exc}"
                ) from None

        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index = pd.to_datetime(frame.index, format="%Y-%m-%d", errors="coerce")
        # Alpha Vantage returns its daily series newest-first, but the price-source
        # contract is oldest-to-newest. Reorder to ascending before validation,
        # which is what ``_validate_frame`` demands. ``sort_index`` is stable, so
        # equal labels keep their insertion order and a genuine duplicate stays
        # adjacent for the validator to reject rather than being silently hidden;
        # the bar values themselves are untouched.
        frame = frame.sort_index(kind="stable")
        # The shared validator raises on NaN, infinity, a non-positive price, a
        # high below low, an open/close outside the range, a duplicate or an
        # out-of-order session, and a non-numeric column. A malformed *vendor*
        # payload is a provider error, so it is re-raised as one.
        try:
            frame = _validate_frame(frame[list(BAR_COLUMNS)], self.symbol, self.kind)
        except PriceSourceError as exc:
            raise ProviderError(str(exc)) from None
        last_bar = str(frame.index[-1])[:10]
        meta_out = {
            "timezone": meta.get("5. Time Zone"),
            "last_refreshed": meta.get("3. Last Refreshed"),
            "output_size": meta.get("4. Output Size"),
        }
        return frame, last_bar, meta_out

    def _assert_recent(
        self,
        last_bar: str,
        fetched_at: datetime,
        *,
        max_stale_days: int | None,
        today: date | None,
    ) -> None:
        if max_stale_days is None:
            return
        reference = today or fetched_at.date()
        age = (reference - date.fromisoformat(last_bar)).days
        if age > max_stale_days:
            raise ProviderStaleDataError(
                f"Alpha Vantage's newest bar for {self.symbol} is {last_bar}, "
                f"{age} days before {reference.isoformat()}; that is older than the "
                f"{max_stale_days}-day limit, so it is not current data"
            )


def _is_recent(last_bar: str, reference: date, max_stale_days: int | None) -> bool:
    """Delegate to the shared freshness rule, so a fetch and a restore agree."""
    return bar_is_fresh(last_bar, reference=reference, max_stale_days=max_stale_days)
