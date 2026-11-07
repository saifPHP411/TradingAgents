"""Finnhub quotes: a current-day price, never a stand-in for historical OHLCV.

Facts this module is built on, taken from Finnhub's own documentation and pricing
page (not assumed):

- ``/stock/candle`` (OHLCV) is marked **"Premium Access Required"** and the
  pricing table leaves the Free column's OHLC row blank, so the free plan does
  **not** provide historical candles. This provider therefore does **not** call
  the candle endpoint and does not pretend to have a bar history.
- ``/quote`` is a free endpoint ("High Usage") that returns a real-time quote:
  the current price (``c``), the day's open/high/low (``o``/``h``/``l``), the
  previous close (``pc``) and a Unix ``t`` timestamp.
- The free plan allows 60 API calls/minute.

Because ``/quote`` is a single current-day observation, this source is a
:class:`~tradingagents.paper_experiment.providers.quote_source.QuoteSource`, not a
``FramePriceSource``: it can answer "what is the price now", and it *refuses* to
answer a request for a historical session or a historical window. That refusal is
the point — a quote must never be quietly served as a past bar.

The provider fetches at construction, validates that the quote is finite and
positive, checks that the quote's own date matches the expected session, and
freezes the result. It never sees a later price, and a quote for the wrong day is
refused rather than served.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from ..sources import (
    SessionBars,
    SourceProvenance,
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

API_BASE_URL = "https://finnhub.io/api/v1/quote"
API_KEY_ENV = "FINNHUB_API_KEY"
REQUEST_TIMEOUT = 30.0

# Finnhub's US quotes carry an exchange session date.
_QUOTE_TIMEZONE = "US/Eastern"


@dataclass(frozen=True)
class QuoteObservation:
    """One validated current-day observation from a vendor."""

    symbol: str
    price: float
    open: float
    high: float
    low: float
    previous_close: float
    observed_at: datetime
    session: str

    def __post_init__(self) -> None:
        for name in ("price", "open", "high", "low", "previous_close"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ProviderError(
                    f"{self.symbol}: {name} must be finite and positive, got {value}"
                )
        if self.high < self.low:
            raise ProviderError(
                f"{self.symbol}: quote high {self.high} is below low {self.low}"
            )
        if not (self.low - 1e-9 <= self.price <= self.high + 1e-9):
            raise ProviderError(
                f"{self.symbol}: quote price {self.price} is outside "
                f"[{self.low}, {self.high}]"
            )


class FinnhubQuoteSource:
    """A current-day quote from Finnhub's free ``/quote`` endpoint.

    It is **not** a bar history. :meth:`sessions`, :meth:`covers`, :meth:`bars`
    and :meth:`window` raise :class:`ProviderUnsupportedError` for a historical
    request, so a caller cannot accidentally drive a multi-session paper
    experiment on a single quote. Only :meth:`quote` — the current price — is
    served.
    """

    kind = "vendor_quote"

    def __init__(
        self,
        symbol: str,
        *,
        api_key: str | None = None,
        expected_session: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        transport: Transport | None = None,
        name: str = "finnhub_quote",
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV, "")
        if not key or not key.strip():
            raise ProviderNotConfiguredError(
                f"no Finnhub credential: set the {API_KEY_ENV} environment variable"
            )
        self.symbol = symbol.upper()
        self.name = name

        payload = fetch_json(
            url=API_BASE_URL,
            params={"symbol": self.symbol, "token": key},
            timeout=timeout,
            secrets=(key,),
            transport=transport,
        )
        self._observation = self._parse(payload, key=key)

        if expected_session is not None and self._observation.session != expected_session:
            raise ProviderStaleDataError(
                f"Finnhub's quote for {self.symbol} is dated {self._observation.session}, "
                f"not the requested session {expected_session}; a quote for another day "
                "is not a price for this one"
            )

        self.provenance = SourceProvenance(
            name=name,
            kind=self.kind,
            is_current_market=True,
            symbol=self.symbol,
            fetched_at=datetime.now(UTC).isoformat(),
            last_bar_date=self._observation.session,
            timezone=_QUOTE_TIMEZONE,
            adjusted=False,
            dataset_start=self._observation.session,
            dataset_end=self._observation.session,
            notes=(
                "Finnhub /quote (free plan): a current-day quote, NOT historical OHLCV",
                "Finnhub's historical candle endpoint is premium and is not used",
            ),
        )

    # -- parsing ----------------------------------------------------------

    def _parse(self, payload: dict, *, key: str) -> QuoteObservation:
        # Finnhub signals a bad token or an unsupported symbol by a body that
        # carries no usable quote, and sometimes an "error" field.
        error = payload.get("error")
        if error:
            message = str(error)
            low = message.lower()
            if "token" in low or "api key" in low or "unauthor" in low or "forbidden" in low:
                raise ProviderAuthError(f"Finnhub rejected the credential: {message}")
            if "limit" in low or "429" in low:
                raise ProviderRateLimitedError(f"Finnhub throttled the request: {message}")
            raise ProviderUnsupportedError(f"Finnhub could not serve {self.symbol!r}: {message}")

        required = ("c", "o", "h", "l", "pc", "t")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ProviderUnsupportedError(
                f"Finnhub returned no usable quote for {self.symbol!r} "
                f"(missing {missing}); the symbol may be unsupported"
            )
        try:
            price = float(payload["c"])
            open_ = float(payload["o"])
            high = float(payload["h"])
            low = float(payload["l"])
            previous_close = float(payload["pc"])
            epoch = int(payload["t"])
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"Finnhub's quote for {self.symbol!r} is malformed: {exc}") from None

        # A zero price is Finnhub's "no data for this symbol" sentinel, not a price.
        if price <= 0 or open_ <= 0 or high <= 0 or low <= 0:
            raise ProviderUnsupportedError(
                f"Finnhub returned a zero price for {self.symbol!r}; the symbol is "
                "unsupported or has no data"
            )
        if epoch <= 0:
            raise ProviderError(
                f"Finnhub's quote for {self.symbol!r} carries no usable timestamp"
            )

        observed_at = datetime.fromtimestamp(epoch, tz=UTC)
        return QuoteObservation(
            symbol=self.symbol,
            price=price,
            open=open_,
            high=high,
            low=low,
            previous_close=previous_close,
            observed_at=observed_at,
            session=observed_at.date().isoformat(),
        )

    # -- the quote surface ------------------------------------------------

    @property
    def is_current_market(self) -> bool:
        return True

    def quote(self) -> QuoteObservation:
        """The validated current-day observation."""
        return self._observation

    def bars(self, symbol: str, session: str) -> SessionBars | None:
        raise self._refuse("a session's OHLC")

    def window(self, symbol: str, session: str):
        raise self._refuse("a historical window")

    def sessions(self) -> tuple[str, ...]:
        raise self._refuse("a session list")

    def covers(self, session: str) -> bool:
        raise self._refuse("session coverage")

    def _refuse(self, what: str) -> ProviderUnsupportedError:
        return ProviderUnsupportedError(
            f"{self.name} serves a current-day quote only; it cannot provide {what}. "
            "Finnhub's historical candle endpoint is a premium feature, so use a "
            "daily-bar source for a multi-session paper experiment."
        )
