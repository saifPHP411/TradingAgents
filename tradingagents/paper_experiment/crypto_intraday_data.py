"""The current intraday decision window for a *crypto* TJR run.

The equity seam (:mod:`.tjr_intraday_data`) fetches 1m/5m bars from Alpaca's IEX
feed and labels the session with the US/Eastern regular-session clock. Crypto is a
different market with a different source and a different session clock, so this is
its own seam with the same *safety* shape:

- the source is Binance's **public** market data (no key, no registration), on the
  data-only host — it can never be pointed at an order endpoint;
- only bars that had **closed** at the decision instant are returned; an
  in-progress bar is dropped, so a half-formed bar never enters a decision;
- a **stale** newest bar (its close older than the freshness limit relative to the
  decision instant) is refused, so TJR never reasons on a frozen feed;
- a **future-dated** bar (dated after the decision session) is refused;
- every bar is validated (finite, positive, high >= low, no duplicate start)
  before the strategy sees it;
- the session is the **UTC calendar day** — the one boundary every exchange agrees
  on — carried by the series' own :class:`~tradingagents.tjr.sessions.SessionPolicy`,
  so the unchanged rule chain reads a crypto session without a second clock.

The Binance client is injectable, so every branch is exercised offline with a
mocked client and no socket. The decision data is *separate* from the worker's
execution price: this seam never prices an order, and the worker's own market-data
seam never reads these bars.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from tradingagents.crypto.binance import BinancePublicClient
from tradingagents.crypto.symbols import parse_pair
from tradingagents.crypto.transport import CryptoDataError
from tradingagents.crypto.types import Candle, candles_to_frame
from tradingagents.tjr.frames import IntradaySeries, normalize_frame
from tradingagents.tjr.sessions import SessionPolicy, utc_day_session

from .tjr_intraday_data import TjrIntradayDataError

DEFAULT_MAX_STALE_MINUTES = 20.0
"""How many minutes the newest *closed* 5m bar may lag the decision instant before
it is refused. A 5m bar is at most 5 minutes old the moment it closes, so 20 minutes
tolerates a few missed bars without accepting a genuinely frozen feed. A stale feed
yields no decision rather than a decision on an old bar."""

_FETCH_BARS = 500
"""5m bars to request: enough for the previous session's high/low plus the current
session's structure. A 1m window of the same span is fetched separately."""

_ONE_MINUTE_BARS = 2000
"""1m bars to request. The 1m trigger only needs the current session, so this is a
generous but bounded window (the venue caps a page at 1000 and pages internally)."""


class CryptoIntradayDataError(TjrIntradayDataError):
    """The crypto intraday decision window could not be served.

    It subclasses the equity seam's error so the shared ``TjrDecisionSource``
    catches it as a *data* failure and records the worker's closed no-trade code
    (``MARKET_DATA_UNAVAILABLE``) with this detail, instead of degrading to a
    generic message. The crypto-specific ``kind`` (``stale``, ``future``,
    ``malformed``, ``no_data``, ``transport``, ``not_started``) carries the finer
    reason for the audit trail.
    """

    def __init__(self, detail: str, *, kind: str = "unavailable") -> None:
        super().__init__(detail, kind=kind)


def _candles_to_closed_frame(
    candles: list[Candle], symbol: str, interval: str
) -> pd.DataFrame:
    """A validated frame of the bars, indexed by bar start in UTC."""
    if not candles:
        raise CryptoIntradayDataError(
            f"the venue returned no {interval} bars for {symbol}", kind="no_data"
        )
    try:
        frame = candles_to_frame(candles)
    except ValueError as exc:
        raise CryptoIntradayDataError(
            f"the venue's {interval} bars for {symbol} failed validation: {exc}",
            kind="malformed",
        ) from None
    return normalize_frame(frame, symbol)


def _build_series(
    frame: pd.DataFrame, symbol: str, interval: str, policy: SessionPolicy
) -> IntradaySeries:
    try:
        return IntradaySeries(symbol, interval, frame, session=policy)
    except Exception as exc:  # noqa: BLE001 - a frame TJR refuses is a data failure
        raise CryptoIntradayDataError(
            f"the {interval} bars for {symbol} failed the TJR reader: {exc}",
            kind="malformed",
        ) from None


@dataclass
class CryptoIntradayBars:
    """Current 1m/5m Binance bars, validated, closed-bar and freshness checked."""

    provider_name: str = "binance_intraday_bars"
    kind: str = "vendor_intraday"
    client: BinancePublicClient | None = None
    policy: SessionPolicy | None = None
    max_stale_minutes: float | None = DEFAULT_MAX_STALE_MINUTES
    clock: Callable[[], datetime] | None = None

    def __post_init__(self) -> None:
        self._client = self.client or BinancePublicClient()
        self._policy = self.policy or utc_day_session()
        self._clock = self.clock or (lambda: datetime.now(UTC))
        self.last_series: tuple[str, str] | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_attempt_at: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            **self._client.describe(),
            "tjr_data_provider": self.provider_name,
            "tjr_data_kind": self.kind,
            "tjr_data_session_policy": self._policy.label,
            "tjr_data_max_stale_minutes": self.max_stale_minutes,
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
        :class:`CryptoIntradayDataError` on any failure; it never returns a partial
        or stale frame.
        """
        pair = parse_pair(symbol)
        canonical = pair.canonical
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise self._fail(
                "the decision instant must be timezone-aware", kind="malformed"
            )
        stamp = stamp.tz_convert("UTC")
        self.last_attempt_at = self._clock().isoformat()

        try:
            one_candles = self._client.klines(canonical, "1m", limit=_ONE_MINUTE_BARS)
            five_candles = self._client.klines(canonical, "5m", limit=_FETCH_BARS)
        except CryptoDataError as exc:
            raise self._fail(
                f"current {canonical} bars could not be read: {exc}", kind="transport"
            ) from None

        one = _candles_to_closed_frame(one_candles, canonical, "1m")
        five = _candles_to_closed_frame(five_candles, canonical, "5m")

        one_closed = one[one.index <= stamp - pd.Timedelta(minutes=1)]
        five_closed = five[five.index <= stamp - pd.Timedelta(minutes=5)]
        if one_closed.empty or five_closed.empty:
            raise self._fail(
                f"no closed {canonical} bars by {stamp.isoformat()}", kind="not_started"
            )

        self._assert_not_future(canonical, five_closed.index[-1], one_closed.index[-1], session)
        self._assert_fresh(canonical, five_closed.index[-1])

        self.last_series = (str(one_closed.index[0]), str(one_closed.index[-1]))
        self.last_error = None
        self.last_error_kind = None
        return (
            _build_series(five_closed, canonical, "5m", self._policy),
            _build_series(one_closed, canonical, "1m", self._policy),
        )

    # -- internals --------------------------------------------------------

    def _fail(self, detail: str, *, kind: str) -> CryptoIntradayDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return CryptoIntradayDataError(detail, kind=kind)

    def _assert_not_future(
        self,
        symbol: str,
        newest_five: pd.Timestamp,
        newest_one: pd.Timestamp,
        session: str,
    ) -> None:
        for interval, newest in (("5m", newest_five), ("1m", newest_one)):
            bar_session = self._policy.session_of(newest)
            if bar_session > session:
                raise self._fail(
                    f"the newest {interval} bar for {symbol} is dated {bar_session}, "
                    f"after the decision session {session}; refusing future-dated data",
                    kind="future",
                )

    def _assert_fresh(self, symbol: str, newest_five: pd.Timestamp) -> None:
        if self.max_stale_minutes is None:
            return
        # The newest closed 5m bar's *close* instant, compared to the decision
        # instant, is the real staleness: a 5m bar labelled 15:55 closes at 16:00.
        newest_close = (newest_five + pd.Timedelta(minutes=5)).tz_convert("UTC").to_pydatetime()
        age_minutes = (self._clock() - newest_close).total_seconds() / 60.0
        if age_minutes > self.max_stale_minutes:
            raise self._fail(
                f"the newest 5m bar for {symbol} closed at "
                f"{newest_close.isoformat()}, {age_minutes:.1f} minutes before now, "
                f"beyond the {self.max_stale_minutes:.0f}-minute freshness limit",
                kind="stale",
            )


class StaticCryptoIntradayBars:
    """A fixed crypto 1m/5m provider, for tests and offline demonstration.

    Explicitly *not* a live vendor: it returns whatever frames it was built with,
    so a test can pin the valid, stale, malformed and future branches without a
    socket. It still validates and freshness-checks through the same rules, so a
    test exercises the real code path rather than a bypass.
    """

    provider_name = "static_crypto_intraday_bars"
    kind = "test"

    def __init__(
        self,
        frames: dict[str, dict[str, pd.DataFrame]],
        *,
        policy: SessionPolicy | None = None,
        max_stale_minutes: float | None = DEFAULT_MAX_STALE_MINUTES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._frames = {parse_pair(k).canonical: v for k, v in frames.items()}
        self._policy = policy or utc_day_session()
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
            "tjr_data_session_policy": self._policy.label,
            "tjr_data_max_stale_minutes": self._max_stale_minutes,
            "tjr_data_last_fetch_at": self.last_attempt_at,
            "tjr_data_last_error": self.last_error,
        }

    def series(
        self, symbol: str, *, session: str, as_of: datetime
    ) -> tuple[IntradaySeries, IntradaySeries]:
        canonical = parse_pair(symbol).canonical
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise self._error("the decision instant must be timezone-aware", "malformed")
        stamp = stamp.tz_convert("UTC")
        self.last_attempt_at = self._clock().isoformat()
        bundle = self._frames.get(canonical)
        if not bundle:
            raise self._error(f"no intraday bars for {canonical!r}", "no_data")
        one = normalize_frame(bundle["1m"], canonical)
        five = normalize_frame(bundle["5m"], canonical)
        one_closed = one[one.index <= stamp - pd.Timedelta(minutes=1)]
        five_closed = five[five.index <= stamp - pd.Timedelta(minutes=5)]
        if one_closed.empty or five_closed.empty:
            raise self._error(f"no closed {canonical} bars by {stamp.isoformat()}", "not_started")
        for interval, newest in (("5m", five_closed.index[-1]), ("1m", one_closed.index[-1])):
            bar_session = self._policy.session_of(newest)
            if bar_session > session:
                raise self._error(
                    f"the newest {interval} bar is after the decision session {session}; "
                    "refusing future-dated data",
                    "future",
                )
        if self._max_stale_minutes is not None:
            newest_close = (
                five_closed.index[-1] + pd.Timedelta(minutes=5)
            ).tz_convert("UTC").to_pydatetime()
            age_minutes = (self._clock() - newest_close).total_seconds() / 60.0
            if age_minutes > self._max_stale_minutes:
                raise self._error(
                    f"the newest 5m bar closed {age_minutes:.1f} minutes ago, beyond the "
                    f"{self._max_stale_minutes:.0f}-minute limit",
                    "stale",
                )
        self.last_series = (str(one_closed.index[0]), str(one_closed.index[-1]))
        self.last_error = None
        self.last_error_kind = None
        return (
            _build_series(five_closed, canonical, "5m", self._policy),
            _build_series(one_closed, canonical, "1m", self._policy),
        )

    def _error(self, detail: str, kind: str) -> CryptoIntradayDataError:
        self.last_error = detail
        self.last_error_kind = kind
        return CryptoIntradayDataError(detail, kind=kind)


__all__ = [
    "DEFAULT_MAX_STALE_MINUTES",
    "CryptoIntradayBars",
    "CryptoIntradayDataError",
    "StaticCryptoIntradayBars",
]
