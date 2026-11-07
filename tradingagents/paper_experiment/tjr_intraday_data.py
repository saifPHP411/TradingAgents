"""The current *intraday* decision window the TJR rule chain reasons on.

The existing current-market path (:mod:`.current_market_data`) fetches **daily**
bars, because the strategy it feeds is the LLM, which reasons on a daily window.
TJR is different: it is a multi-timeframe day-trading rule chain built from
**1-minute and 5-minute** bars. A daily bar cannot serve it, so this module is a
separate seam with the same safety shape as its daily sibling.

Two prices are kept apart, exactly as in the daily path:

- **decision data** (here): the intraday bars TJR reads, from
  ``https://data.alpaca.markets`` on the IEX feed. It is *not* the execution price.
- **execution price**: the current price the worker sizes and submits an order
  from. That stays on the existing ``MarketData`` seam
  (``AlpacaSnapshotMarketData``), untouched by this module.

Safety properties, each deliberate:

- the host is fixed to Alpaca's **market-data** host; the paper trading host and
  every live host are refused, so decision data can never be pointed at orders;
- the feed is ``iex``, the only feed the Basic plan can read;
- only bars whose interval is **1Min** or **5Min** are requested, and every bar is
  validated (finite, positive, high ≥ low, no duplicate start) before the strategy
  sees it;
- a **stale** newest bar (older than the freshness limit relative to the decision
  session) is refused, so TJR never reasons on a bar that is not current;
- a **future-dated** bar (dated after the decision session) is refused;
- an **in-progress** bar — one whose close is after the decision instant — is
  dropped, so a half-formed bar can never enter a decision. The closed-bar rule
  lives in :mod:`tradingagents.tjr.frames`; this seam only ever hands back bars
  the caller could already have seen;
- the fetch is bounded (a timeout) and never retries silently.

The transport is injectable, so every branch is exercised with a mocked response
and no socket.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from tradingagents.tjr.frames import IntradaySeries, resample, session_of

from .providers.alpaca_iex import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DATA_BASE_URL,
    FEED,
    REQUEST_TIMEOUT,
    _auth_headers,
    _classify_status,
    assert_data_host,
)
from .providers.base import ProviderError, fetch_json

DEFAULT_LOOKBACK_DAYS = 5
"""Calendar days of intraday history to request. TJR reads the previous session's
high/low as a draw on liquidity, so a few sessions are needed for structure; the
sweep/confirmation window itself is the current session. Kept small because
intraday bars are voluminous and the Basic IEX plan serves a limited history."""

DEFAULT_MAX_STALE_MINUTES = 20
"""How many minutes the newest *closed* 5m bar may lag the decision instant before
it is refused as stale. A 5m bar is at most 5 minutes old the moment it closes, so
20 minutes tolerates a few missed bars (thin pre/post-open prints) without
accepting a genuinely frozen feed. A stale feed yields no decision rather than a
decision on an old bar."""

_INTRADAY_TIMEFRAME = {"1m": "1Min", "5m": "5Min"}
_BAR_LIMIT = 10000
"""Bars per request. A single symbol's 1m bars for one regular session are 390, so
this comfortably covers the lookback window in one page."""


class TjrIntradayDataError(RuntimeError):
    """The current intraday decision window could not be served. Never partial."""

    def __init__(self, detail: str, *, kind: str = "unavailable") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(detail)

    @property
    def code(self) -> str:
        """The worker's no-trade code for this failure.

        Every intraday-decision-data failure is a *market data* failure, so the
        code is the closed-vocabulary value; ``kind`` carries the finer reason.
        """
        from .broker.trading import NO_TRADE_MARKET_DATA_UNAVAILABLE

        return NO_TRADE_MARKET_DATA_UNAVAILABLE


@runtime_checkable
class TjrIntradayData(Protocol):
    """Serves the 1m/5m decision series for a symbol at a decision session."""

    def series(
        self, symbol: str, *, session: str, as_of: datetime
    ) -> tuple[IntradaySeries, IntradaySeries]:  # pragma: no cover - protocol
        ...

    def describe(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


def _parse_session(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _parse_stamp(value: Any) -> pd.Timestamp | None:
    """Parse an RFC-3339 bar timestamp to a UTC timestamp, or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = pd.Timestamp(text)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.tz_convert("UTC")


def _validate_intraday_frame(frame: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """The 1m/5m bars as an :class:`IntradaySeries` will accept them, or raise.

    :class:`~tradingagents.tjr.frames.IntradaySeries` re-validates on construction;
    this pre-check gives a *specific* vendor-facing error (naming the symbol and
    interval) rather than a generic frame error, and lets the seam drop an
    in-progress bar before the closed-bar guard is applied downstream.
    """
    if frame is None or frame.empty:
        raise TjrIntradayDataError(
            f"Alpaca returned no {interval} bars for {symbol}", kind="no_data"
        )
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    for column in ("open", "high", "low", "close"):
        if column not in data.columns:
            raise TjrIntradayDataError(
                f"Alpaca's {interval} bars for {symbol} are missing {column!r}",
                kind="malformed",
            )
    if "volume" not in data.columns:
        data["volume"] = 0.0
    index = pd.to_datetime(data.index, errors="coerce", utc=True)
    if index.isna().any():
        raise TjrIntradayDataError(
            f"Alpaca's {interval} bars for {symbol} carry an unparseable timestamp",
            kind="malformed",
        )
    data.index = index
    if data.index.has_duplicates:
        raise TjrIntradayDataError(
            f"Alpaca returned a duplicate {interval} bar start for {symbol}",
            kind="malformed",
        )
    data = data.sort_index()
    prices = data[["open", "high", "low", "close"]].astype(float)
    if not prices.notna().all().all():
        raise TjrIntradayDataError(
            f"Alpaca's {interval} bars for {symbol} carry a missing price", kind="malformed"
        )
    if (prices <= 0).any().any():
        raise TjrIntradayDataError(
            f"Alpaca's {interval} bars for {symbol} carry a non-positive price",
            kind="malformed",
        )
    if (prices["high"] < prices["low"]).any():
        raise TjrIntradayDataError(
            f"Alpaca's {interval} bars for {symbol} have a high below its low",
            kind="malformed",
        )
    data[["open", "high", "low", "close"]] = prices
    return data[["open", "high", "low", "close", "volume"]]


def _build_series(frame: pd.DataFrame, symbol: str, interval: str) -> IntradaySeries:
    try:
        return IntradaySeries(symbol, interval, frame)
    except Exception as exc:  # a frame the TJR reader refuses is a data failure
        raise TjrIntradayDataError(
            f"Alpaca's {interval} bars for {symbol} failed validation: {exc}",
            kind="malformed",
        ) from None


class AlpacaIntradayBars:
    """Current 1m/5m bars from Alpaca market data, validated and freshness-checked."""

    provider_name = "alpaca_intraday_bars"
    kind = "vendor_intraday"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DATA_BASE_URL,
        transport: Callable[..., Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_stale_minutes: float | None = DEFAULT_MAX_STALE_MINUTES,
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
        self._max_stale_minutes = max_stale_minutes
        self._clock = clock or (lambda: datetime.now(UTC))
        self.last_series: tuple[str, str] | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_attempt_at: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "tjr_data_provider": self.provider_name,
            "tjr_data_kind": self.kind,
            "tjr_data_endpoint": self._base_url,
            "tjr_data_feed": FEED,
            "tjr_data_lookback_days": self._lookback_days,
            "tjr_data_max_stale_minutes": self._max_stale_minutes,
            "tjr_data_last_fetch_at": self.last_attempt_at,
            "tjr_data_last_error": self.last_error,
        }

    # -- fetch ------------------------------------------------------------

    def series(
        self, symbol: str, *, session: str, as_of: datetime
    ) -> tuple[IntradaySeries, IntradaySeries]:
        """The (5m, 1m) decision series for ``symbol`` at ``session`` and ``as_of``.

        Both series are sliced to bars that had *closed* at ``as_of``. A stale or
        future-dated newest bar is refused. Raises
        :class:`TjrIntradayDataError` on any failure; it never returns a partial
        or stale frame.
        """
        symbol = symbol.upper()
        reference = _parse_session(session)
        if reference is None:
            raise TjrIntradayDataError(
                f"the decision session {session!r} is not a usable date", kind="malformed"
            )
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise TjrIntradayDataError(
                "the decision instant must be timezone-aware", kind="malformed"
            )
        stamp = stamp.tz_convert("UTC")
        self.last_attempt_at = self._clock().isoformat()
        if not self._key or not self._secret:
            raise self._fail(
                f"no Alpaca market-data credentials: set {API_KEY_ENV} and "
                f"{API_SECRET_ENV} so current intraday decision data can be fetched",
                kind="not_configured",
            )

        one = self._fetch_interval(symbol, "1m", reference)
        five = self._fetch_interval(symbol, "5m", reference)

        one_series = _build_series(one, symbol, "1m")
        five_series = _build_series(five, symbol, "5m")

        # Only closed bars: a bar that closes after ``as_of`` cannot be used.
        one_closed = one_series.frame[one_series.frame.index <= stamp - pd.Timedelta(minutes=1)]
        five_closed = five_series.frame[five_series.frame.index <= stamp - pd.Timedelta(minutes=5)]
        if one_closed.empty or five_closed.empty:
            raise self._fail(
                f"no closed {symbol} bars by {stamp.isoformat()}; the session has not "
                "opened far enough for a decision",
                kind="not_started",
            )

        self._assert_fresh(symbol, five_closed.index[-1], one_closed.index[-1], reference)

        result = (
            _build_series(five_closed, symbol, "5m"),
            _build_series(one_closed, symbol, "1m"),
        )
        self.last_series = (str(one_closed.index[0]), str(one_closed.index[-1]))
        self.last_error = None
        self.last_error_kind = None
        return result

    # -- internals --------------------------------------------------------

    def _fail(self, detail: str, *, kind: str) -> TjrIntradayDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return TjrIntradayDataError(detail, kind=kind)

    def _fetch_interval(self, symbol: str, interval: str, reference: date) -> pd.DataFrame:
        timeframe = _INTRADAY_TIMEFRAME[interval]
        start = date.fromordinal(reference.toordinal() - self._lookback_days)
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": reference.isoformat(),
            "feed": FEED,
            "adjustment": "raw",
            "limit": _BAR_LIMIT,
            "sort": "asc",
        }
        try:
            payload = fetch_json(
                url=f"{self._base_url}/v2/stocks/bars",
                params=params,
                timeout=self._timeout,
                secrets=(self._key, self._secret),
                headers=_auth_headers(self._key, self._secret),
                transport=self._transport,
                classify_status=_classify_status,
            )
        except ProviderError as exc:
            raise self._fail(
                f"Alpaca {interval} bars for {symbol} could not be read: {exc}",
                kind="transport",
            ) from None
        except Exception as exc:  # noqa: BLE001 - any failure is unavailable data
            raise self._fail(
                f"Alpaca {interval} bars for {symbol} could not be read: {type(exc).__name__}",
                kind="transport",
            ) from None
        return self._parse(payload, symbol, interval)

    def _parse(self, payload: dict, symbol: str, interval: str) -> pd.DataFrame:
        bars_by_symbol = payload.get("bars")
        if not isinstance(bars_by_symbol, dict):
            raise self._fail(
                f"Alpaca returned no 'bars' object for {symbol} {interval}", kind="malformed"
            )
        raw = bars_by_symbol.get(symbol)
        if raw is None:
            raise self._fail(
                f"Alpaca returned no {interval} bars for {symbol!r}; the symbol may be "
                "unsupported or the window empty",
                kind="unsupported",
            )
        if not isinstance(raw, list) or not raw:
            raise self._fail(
                f"Alpaca returned an empty {interval} bar series for {symbol!r}",
                kind="no_data",
            )
        rows: dict[pd.Timestamp, dict[str, float]] = {}
        for bar in raw:
            if not isinstance(bar, dict):
                raise self._fail("an Alpaca bar is not an object", kind="malformed")
            start = _parse_stamp(bar.get("t"))
            if start is None:
                raise self._fail(
                    f"an Alpaca {interval} bar is missing a usable timestamp",
                    kind="malformed",
                )
            if start in rows:
                raise self._fail(
                    f"Alpaca returned the {interval} bar {start.isoformat()} more than "
                    f"once for {symbol!r}",
                    kind="malformed",
                )
            try:
                rows[start] = {
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar.get("v") or 0.0),
                }
            except (KeyError, TypeError, ValueError):
                raise self._fail(
                    f"Alpaca's {interval} bar at {start.isoformat()} is malformed",
                    kind="malformed",
                ) from None
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        return _validate_intraday_frame(frame, symbol, interval)

    def _assert_fresh(
        self,
        symbol: str,
        newest_five: pd.Timestamp,
        newest_one: pd.Timestamp,
        reference: date,
    ) -> None:
        """Refuse a newest bar that lags the decision session or is future-dated."""
        for interval, newest in (("5m", newest_five), ("1m", newest_one)):
            session = session_of(newest)
            bar_date = _parse_session(session)
            if bar_date is None:
                raise self._fail(
                    f"Alpaca's newest {interval} bar date {session!r} is unparseable",
                    kind="malformed",
                )
            if bar_date > reference:
                raise self._fail(
                    f"Alpaca's newest {interval} bar for {symbol} is dated "
                    f"{bar_date.isoformat()}, after the decision session "
                    f"{reference.isoformat()}; refusing future-dated data",
                    kind="future",
                )
        if self._max_stale_minutes is None:
            return
        # The newest closed 5m bar's *close* instant, compared to the decision
        # instant, is the real staleness: a 5m bar labelled 15:55 closes at 16:00.
        newest_close = newest_five + pd.Timedelta(minutes=5)
        now = self._clock()
        age_minutes = (now - newest_close.tz_convert("UTC").to_pydatetime()).total_seconds() / 60.0
        if age_minutes > self._max_stale_minutes:
            raise self._fail(
                f"Alpaca's newest 5m bar for {symbol} closed at "
                f"{newest_close.isoformat()}, {age_minutes:.1f} minutes before now, "
                f"beyond the {self._max_stale_minutes:.0f}-minute freshness limit",
                kind="stale",
            )


class StaticIntradayBars:
    """A fixed 1m/5m provider, for tests and offline demonstration.

    Explicitly *not* a live vendor: it returns whatever frames it was built with,
    so a test can pin the valid, stale, malformed and future branches without a
    socket. It still validates and freshness-checks through the same rules, so a
    test exercises the real code path rather than a bypass.
    """

    provider_name = "static_intraday_bars"
    kind = "test"

    def __init__(
        self,
        frames: dict[str, dict[str, pd.DataFrame]],
        *,
        max_stale_minutes: float | None = DEFAULT_MAX_STALE_MINUTES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._frames = {k.upper(): v for k, v in frames.items()}
        self._max_stale_minutes = max_stale_minutes
        self._clock = clock or (lambda: datetime.now(UTC))
        self.last_series: tuple[str, str] | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_attempt_at: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "tjr_data_provider": self.provider_name,
            "tjr_data_kind": self.kind,
            "tjr_data_max_stale_minutes": self._max_stale_minutes,
            "tjr_data_last_fetch_at": self.last_attempt_at,
            "tjr_data_last_error": self.last_error,
        }

    def series(
        self, symbol: str, *, session: str, as_of: datetime
    ) -> tuple[IntradaySeries, IntradaySeries]:
        symbol = symbol.upper()
        reference = _parse_session(session)
        self.last_attempt_at = self._clock().isoformat()
        if reference is None:
            raise self._error(f"the decision session {session!r} is not usable", "malformed")
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise self._error("the decision instant must be timezone-aware", "malformed")
        stamp = stamp.tz_convert("UTC")
        bundle = self._frames.get(symbol)
        if not bundle:
            raise self._error(f"no intraday bars for {symbol!r}", "no_data")
        one = _validate_intraday_frame(bundle.get("1m"), symbol, "1m")
        five = _validate_intraday_frame(bundle.get("5m"), symbol, "5m")
        one_series = _build_series(one, symbol, "1m")
        five_series = _build_series(five, symbol, "5m")
        one_closed = one_series.frame[one_series.frame.index <= stamp - pd.Timedelta(minutes=1)]
        five_closed = five_series.frame[five_series.frame.index <= stamp - pd.Timedelta(minutes=5)]
        if one_closed.empty or five_closed.empty:
            raise self._error(
                f"no closed {symbol} bars by {stamp.isoformat()}", "not_started"
            )
        for interval, newest in (("5m", five_closed.index[-1]), ("1m", one_closed.index[-1])):
            bar_date = _parse_session(session_of(newest))
            if bar_date is None:
                raise self._error(
                    f"the newest {interval} bar date is unparseable", "malformed"
                )
            if bar_date > reference:
                raise self._error(
                    f"the newest {interval} bar is after the decision session "
                    f"{reference.isoformat()}; refusing future-dated data",
                    "future",
                )
        if self._max_stale_minutes is not None:
            newest_close = five_closed.index[-1] + pd.Timedelta(minutes=5)
            age_minutes = (
                self._clock() - newest_close.tz_convert("UTC").to_pydatetime()
            ).total_seconds() / 60.0
            if age_minutes > self._max_stale_minutes:
                raise self._error(
                    f"the newest 5m bar closed {age_minutes:.1f} minutes ago, beyond "
                    f"the {self._max_stale_minutes:.0f}-minute limit",
                    "stale",
                )
        self.last_series = (str(one_closed.index[0]), str(one_closed.index[-1]))
        self.last_error = None
        self.last_error_kind = None
        return (
            _build_series(five_closed, symbol, "5m"),
            _build_series(one_closed, symbol, "1m"),
        )

    def _error(self, detail: str, kind: str) -> TjrIntradayDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return TjrIntradayDataError(detail, kind=kind)


def resample_series(series: IntradaySeries, interval: str) -> IntradaySeries | None:
    """Resample ``series`` to ``interval``, or ``None`` when no full bucket exists.

    A short intraday window may not fill a single 60m or 4h bucket; that is an
    absence of a level source, not an error, so it is reported as ``None`` rather
    than raising. Mirrors ``tjr.run._resample_if_possible`` so the live path builds
    the same hourly/4h levels the backtest does.
    """
    try:
        return resample(series, interval)
    except Exception:  # noqa: BLE001 - an unfilled bucket is an absence, not a failure
        return None


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MAX_STALE_MINUTES",
    "AlpacaIntradayBars",
    "StaticIntradayBars",
    "TjrIntradayData",
    "TjrIntradayDataError",
    "resample_series",
]
