"""Alpaca IEX daily OHLCV, for the multi-session window a paper run needs.

This is the *market-data* side of Alpaca, and it is deliberately separate from the
broker side (``broker/alpaca_paper.py``). The two use different hosts and different
purposes:

- market data comes from ``https://data.alpaca.markets`` only;
- orders go to ``https://paper-api.alpaca.markets`` only.

Neither host is ever substituted for the other. This provider never places an
order, never reads a trading credential variable, and never reaches a live host.

Facts this module is built on, taken from Alpaca's own documentation (not
assumed):

- ``GET https://data.alpaca.markets/v2/stocks/bars`` returns historical bars. The
  response is ``{"bars": {"<SYMBOL>": [ {...}, ... ]}, "next_page_token": ...}``.
- Each bar carries ``t`` (RFC-3339 start), ``o``/``h``/``l``/``c`` (prices), ``v``
  (volume), ``n`` (trade count) and ``vw`` (volume-weighted average price).
- The free/Basic plan serves the **IEX** feed for equities. A request must name
  ``feed=iex``; the default feed is SIP, which the Basic plan cannot read, so the
  feed is fixed to ``iex`` here rather than left to a default that would 403.
- Authentication is by the ``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY`` headers,
  the same credentials the paper trading account already uses.

The provider fetches once, at construction, validates the frame through the shared
validator (so it cannot diverge from the frozen dataset's behaviour), and freezes
it. It never sees a bar after the fetch.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from urllib.parse import urlsplit

import pandas as pd

from ..sources import (
    BAR_COLUMNS,
    FramePriceSource,
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

DATA_BASE_URL = "https://data.alpaca.markets"
"""The only host this provider will talk to. Alpaca's market-data host."""

_LIVE_DATA_HOSTS = frozenset(
    {"api.alpaca.markets", "alpaca.markets", "broker-api.alpaca.markets"}
)
"""Alpaca live-trading hosts, refused outright. Never a fallback.

Kept in step with the broker's own ``_LIVE_HOSTS``, so a host the broker refuses is
refused here too, with the same "this is live" message rather than a vaguer one.
"""

API_KEY_ENV = "ALPACA_PAPER_API_KEY"
API_SECRET_ENV = "ALPACA_PAPER_API_SECRET"
REQUEST_TIMEOUT = 30.0

# The Basic plan can read the IEX feed only; SIP requires a paid subscription.
FEED = "iex"

# A newest bar older than this many calendar days (from the fetch date) is not
# "recent data"; the fetch is refused rather than served as if current.
DEFAULT_MAX_STALE_DAYS = 10

# A generous page size: enough for a multi-session window in one request. The
# provider reads a single page; it does not follow ``next_page_token``.
_BAR_LIMIT = 1000


def assert_data_host(base_url: str) -> str:
    """Return ``base_url`` if it is Alpaca's market-data host, else raise.

    The single gate the request URL passes through. The paper trading host
    (``paper-api.alpaca.markets``) is *not* a market-data host and is refused, as
    is every live host, so a configuration mistake cannot point data at orders or
    at live infrastructure.
    """
    host = (urlsplit(base_url).hostname or "").lower()
    if host in _LIVE_DATA_HOSTS:
        raise ProviderUnsupportedError(
            f"{base_url!r} is an Alpaca live host; this provider is market-data "
            f"only and refuses it. Use {DATA_BASE_URL}."
        )
    if host != urlsplit(DATA_BASE_URL).hostname:
        raise ProviderUnsupportedError(
            f"{base_url!r} is not the Alpaca market-data host ({DATA_BASE_URL}); "
            "refusing to send a market-data request to an unapproved host"
        )
    return base_url


def build_request_params(*, symbol: str, start: date, end: date, feed: str = FEED) -> dict:
    """The exact query parameters sent to Alpaca, credential excluded.

    One source of truth for the outgoing request, so a diagnostic or a test can
    assert the feed and the timeframe without a second copy drifting from the real
    call. The credential travels in headers, never in this mapping, so it cannot
    appear in a URL or a log.
    """
    return {
        "symbols": symbol.upper(),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": feed,
        "adjustment": "raw",
        "limit": _BAR_LIMIT,
        "sort": "asc",
    }


def _auth_headers(key: str, secret: str) -> dict:
    """The auth headers. Never logged, never serialised."""
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    return out


def _classify_status(status_code: int, body: str) -> None:
    """Give Alpaca's status codes the specific meaning an operator needs.

    401 is an authentication failure (a missing or invalid credential); 403 is a
    forbidden request (most often a feed the plan cannot read); 429 is a rate
    limit. The body is already redacted by :func:`fetch_json` before this runs, so
    a credential cannot leak through the raised message.
    """
    if status_code == 401:
        raise ProviderAuthError(
            f"Alpaca rejected the market-data credential (HTTP 401): {body}"
        )
    if status_code == 403:
        raise ProviderUnsupportedError(
            "Alpaca refused the market-data request (HTTP 403); the IEX feed is "
            f"available on the Basic plan, so check the symbol and the plan: {body}"
        )
    if status_code == 429:
        raise ProviderRateLimitedError(
            f"Alpaca throttled the market-data request (HTTP 429): {body}"
        )


class AlpacaIexDailySource(FramePriceSource):
    """Daily OHLCV bars from Alpaca's market-data API, on the IEX feed."""

    kind = "vendor_daily"

    def __init__(
        self,
        symbol: str,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DATA_BASE_URL,
        timeout: float = REQUEST_TIMEOUT,
        transport: Transport | None = None,
        max_stale_days: int | None = DEFAULT_MAX_STALE_DAYS,
        today: date | None = None,
        lookback_days: int = 400,
        name: str = "alpaca_iex_daily",
    ) -> None:
        # Host first: refuse a non-market-data host before any credential is read.
        assert_data_host(base_url)

        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret
            if api_secret is not None
            else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        key, secret = key.strip(), secret.strip()
        if not key or not secret:
            raise ProviderNotConfiguredError(
                f"no Alpaca market-data credentials: set {API_KEY_ENV} and "
                f"{API_SECRET_ENV} (the same paper credentials the broker uses)"
            )
        self.symbol = symbol.upper()

        reference = today or datetime.now(UTC).date()
        start = date.fromordinal(reference.toordinal() - max(lookback_days, 1))
        payload = fetch_json(
            url=f"{base_url}/v2/stocks/bars",
            params=build_request_params(symbol=self.symbol, start=start, end=reference),
            timeout=timeout,
            secrets=(key, secret),
            headers=_auth_headers(key, secret),
            transport=transport,
            classify_status=_classify_status,
        )
        frame, last_bar = self._parse(payload, secrets=(key, secret))

        fetched_at = datetime.now(UTC)
        self._assert_recent(last_bar, reference, max_stale_days=max_stale_days)

        provenance = SourceProvenance(
            name=name,
            kind=self.kind,
            is_current_market=bar_is_fresh(
                last_bar, reference=reference, max_stale_days=max_stale_days
            ),
            symbol=self.symbol,
            fetched_at=fetched_at.isoformat(),
            last_bar_date=last_bar,
            timezone="UTC",
            adjusted=False,
            dataset_start=str(frame.index[0])[:10],
            dataset_end=str(frame.index[-1])[:10],
            max_stale_days=max_stale_days,
            notes=(
                "Alpaca market data v2 stocks/bars (IEX feed, unadjusted, as-traded)",
                "feed: iex (the Basic plan serves IEX only; SIP requires a paid plan)",
            ),
        )
        super().__init__({self.symbol: frame}, name=name, provenance=provenance)

    # -- parsing ----------------------------------------------------------

    def _parse(
        self, payload: dict, *, secrets: tuple[str, ...]
    ) -> tuple[pd.DataFrame, str]:
        bars_by_symbol = payload.get("bars")
        if not isinstance(bars_by_symbol, dict):
            raise ProviderError("Alpaca returned no 'bars' object in the response")
        raw = bars_by_symbol.get(self.symbol)
        if raw is None:
            # A symbol the vendor does not know is an unsupported request, not an
            # empty series dressed up as data.
            raise ProviderUnsupportedError(
                f"Alpaca returned no bars for {self.symbol!r}; the symbol may be "
                "unsupported or the window may be empty"
            )
        if not isinstance(raw, list):
            raise ProviderError(
                f"Alpaca's bar list for {self.symbol!r} is not an array"
            )
        if not raw:
            raise ProviderStaleDataError(
                f"Alpaca returned an empty bar series for {self.symbol!r}; there is "
                "no session to serve, so it is not usable as current data"
            )

        rows: dict[str, dict] = {}
        for bar in raw:
            if not isinstance(bar, dict):
                raise ProviderError("an Alpaca bar is not an object")
            stamp = bar.get("t")
            if not isinstance(stamp, str) or not stamp:
                raise ProviderError("an Alpaca bar is missing its timestamp 't'")
            session = stamp[:10]
            if session in rows:
                # A repeated session label is refused, not silently collapsed: a
                # dict assignment would keep only the last and hide a vendor
                # anomaly that could restate a bar the run already used.
                raise ProviderError(
                    f"Alpaca returned the session {session} more than once for "
                    f"{self.symbol!r}; refusing a duplicated bar"
                )
            try:
                rows[session] = {
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(
                    f"Alpaca's bar for {session} is malformed: {type(exc).__name__}"
                ) from None

        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index = pd.to_datetime(frame.index, format="%Y-%m-%d", errors="coerce")
        frame = frame.sort_index(kind="stable")
        try:
            frame = _validate_frame(frame[list(BAR_COLUMNS)], self.symbol, self.kind)
        except ProviderError:
            raise
        except Exception as exc:  # a malformed vendor payload is a provider error
            raise ProviderError(
                f"Alpaca's daily bars for {self.symbol!r} failed validation: {exc}"
            ) from None
        last_bar = str(frame.index[-1])[:10]
        return frame, last_bar

    def _assert_recent(
        self,
        last_bar: str,
        reference: date,
        *,
        max_stale_days: int | None,
    ) -> None:
        if max_stale_days is None:
            return
        age = (reference - date.fromisoformat(last_bar)).days
        if age > max_stale_days:
            raise ProviderStaleDataError(
                f"Alpaca's newest bar for {self.symbol} is {last_bar}, {age} days "
                f"before {reference.isoformat()}; that is older than the "
                f"{max_stale_days}-day limit, so it is not current data"
            )
