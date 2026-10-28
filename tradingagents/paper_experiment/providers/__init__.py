"""Real market-data providers for the paper experiment, kept strictly separate.

This package is the only place in the paper experiment that talks to a network.
It exists so a paper experiment *can* be pointed at a live vendor, and it is
written so that doing so cannot quietly become live trading: a provider fetches
data, never an order, and every provider declares its own provenance.

Two providers ship:

- :class:`~.alpha_vantage.AlphaVantageDailySource` — daily OHLCV bars, for the
  multi-session window a paper experiment actually needs. It uses
  ``TIME_SERIES_DAILY`` (the free function). ``TIME_SERIES_DAILY_ADJUSTED`` is a
  premium endpoint and is deliberately *not* used.
- :class:`~.finnhub.FinnhubQuoteSource` — a same-day quote. Finnhub's historical
  candle endpoint is premium, so this provider serves a *current-day price only*
  and refuses to stand in for historical OHLCV. It is labelled ``vendor_quote``
  and never masquerades as a bar history.

Neither provider is the default, neither is contacted unless a caller constructs
it, and a provider that is not configured raises rather than returning invented
data.
"""

from __future__ import annotations

from .alpha_vantage import AlphaVantageDailySource
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderStaleDataError,
    ProviderUnsupportedError,
)
from .finnhub import FinnhubQuoteSource
from .selection import (
    KNOWN_SOURCES,
    SourceSelection,
    SourceSelectionError,
    select_source,
)

# The providers that represent the *current* market, as opposed to a historical
# dataset. Finnhub's quote endpoint is a same-day price; Alpha Vantage's daily
# series is historical, so it is not listed here.
_CURRENT_MARKET_PROVIDERS = (FinnhubQuoteSource,)


def current_market_sources() -> tuple[type, ...]:
    """The provider classes that serve the current market."""
    return _CURRENT_MARKET_PROVIDERS


def provider_registry() -> dict[str, type]:
    """The selectable source names, for the CLI and the report.

    ``frozen`` and ``static`` are always available; the vendor providers are
    listed so a caller can select them, and constructing one without a credential
    raises rather than silently falling back.
    """
    from ..sources import FrozenPriceSource, StaticPriceSource

    return {
        "frozen": FrozenPriceSource,
        "static": StaticPriceSource,
        "alpha_vantage": AlphaVantageDailySource,
        "finnhub": FinnhubQuoteSource,
    }


__all__ = [
    "KNOWN_SOURCES",
    "AlphaVantageDailySource",
    "FinnhubQuoteSource",
    "ProviderAuthError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderRateLimitedError",
    "ProviderStaleDataError",
    "ProviderUnsupportedError",
    "SourceSelection",
    "SourceSelectionError",
    "current_market_sources",
    "provider_registry",
    "select_source",
]
