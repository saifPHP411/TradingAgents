"""Explicit price-source selection, with a logged, non-mixing fallback.

Choosing a price source is a decision that changes what an experiment *is*, so it
is made in one place and recorded, not guessed per session:

- :func:`select_source` builds the requested source. A vendor source is only built
  when it is asked for; nothing is fetched by default.
- A fallback is **opt-in**. When a primary fails to build — a missing credential,
  a quota, a stale feed, an unsupported symbol — the caller must have named a
  fallback explicitly. Otherwise the failure is raised, so a run stops instead of
  silently changing its data.
- A fallback switch is recorded (:class:`SourceSelection`) and logged, and it is
  **all-or-nothing**: the experiment runs on exactly one source's bars, so a
  snapshot never mixes two vendors. There is no per-session, silent fallback.

The last rule matters most: the frozen 2024 dataset must never be silently
substituted for a vendor that failed and then read as current data. If a vendor
fails, the run stops (or switches, loudly) — it does not fall back to 2024 and
carry on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ..sources import PriceSource, PriceSourceError
from .base import ProviderError

logger = logging.getLogger(__name__)

# Selectable source names, so a typo is refused rather than silently defaulted.
KNOWN_SOURCES = ("frozen", "static", "alpha_vantage", "finnhub")


class SourceSelectionError(ProviderError):
    """A price source could not be selected as requested."""


@dataclass
class SourceSelection:
    """The source an experiment will run on, and how it was chosen."""

    source: PriceSource
    requested: str
    switched: bool = False
    reason: str | None = None
    primary_name: str | None = None
    fallback_name: str | None = None
    attempted: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved_name": self.source.name,
            "resolved_kind": getattr(self.source, "kind", "unknown"),
            "is_current_market": self.source.is_current_market,
            "switched": self.switched,
            "reason": self.reason,
            "primary_name": self.primary_name,
            "fallback_name": self.fallback_name,
            "attempted": list(self.attempted),
        }


def _build(
    spec: str,
    *,
    symbols: tuple[str, ...],
    frozen_feed_factory,
    api_keys: dict[str, str] | None = None,
    provider_kwargs: dict | None = None,
    expected_session: str | None = None,
    today: date | None = None,
) -> PriceSource:
    """Build one source by name, or raise :class:`SourceSelectionError`."""
    keys = api_keys or {}
    kwargs = dict(provider_kwargs or {})
    if spec == "frozen":
        if frozen_feed_factory is None:
            raise SourceSelectionError("the frozen dataset source is not available here")
        return frozen_feed_factory()
    if spec == "static":
        raise SourceSelectionError(
            "the static source is a test fixture and cannot be selected for a run"
        )
    if spec == "alpha_vantage":
        from .alpha_vantage import API_KEY_ENV, AlphaVantageDailySource

        if len(symbols) != 1:
            raise SourceSelectionError(
                "the Alpha Vantage daily source fetches one symbol per source; "
                f"got {len(symbols)}"
            )
        return AlphaVantageDailySource(
            symbols[0],
            api_key=keys.get(API_KEY_ENV),
            today=today,
            **kwargs,
        )
    if spec == "finnhub":
        from .finnhub import API_KEY_ENV, FinnhubQuoteSource

        if len(symbols) != 1:
            raise SourceSelectionError(
                "the Finnhub quote source fetches one symbol per source; "
                f"got {len(symbols)}"
            )
        return FinnhubQuoteSource(
            symbols[0],
            api_key=keys.get(API_KEY_ENV),
            expected_session=expected_session,
            **kwargs,
        )
    raise SourceSelectionError(
        f"unknown price source {spec!r}; known sources are {', '.join(KNOWN_SOURCES)}"
    )


def select_source(
    requested: str,
    *,
    symbols: tuple[str, ...],
    frozen_feed_factory=None,
    fallback: str | None = None,
    api_keys: dict[str, str] | None = None,
    provider_kwargs: dict | None = None,
    expected_session: str | None = None,
    today: date | None = None,
) -> SourceSelection:
    """Build ``requested``, falling back to ``fallback`` only if one is named.

    The fallback is explicit and logged. If the primary fails and no fallback was
    named, the failure is raised — a run does not silently change its data source.
    """
    if requested not in KNOWN_SOURCES:
        raise SourceSelectionError(
            f"unknown price source {requested!r}; known sources are "
            f"{', '.join(KNOWN_SOURCES)}"
        )
    if fallback is not None and fallback not in KNOWN_SOURCES:
        raise SourceSelectionError(
            f"unknown fallback source {fallback!r}; known sources are "
            f"{', '.join(KNOWN_SOURCES)}"
        )

    attempted: list[dict] = []
    try:
        source = _build(
            requested,
            symbols=symbols,
            frozen_feed_factory=frozen_feed_factory,
            api_keys=api_keys,
            provider_kwargs=provider_kwargs,
            expected_session=expected_session,
            today=today,
        )
        return SourceSelection(source=source, requested=requested, attempted=attempted)
    except (ProviderError, PriceSourceError) as exc:
        attempted.append({"source": requested, "error": f"{type(exc).__name__}: {exc}"})
        if fallback is None:
            raise SourceSelectionError(
                f"the price source {requested!r} could not be built ({exc}), and no "
                "fallback was named; refusing to change the data source silently"
            ) from exc
        logger.warning(
            "price source %r unavailable (%s); switching explicitly to the named "
            "fallback %r",
            requested,
            exc,
            fallback,
        )

    try:
        source = _build(
            fallback,
            symbols=symbols,
            frozen_feed_factory=frozen_feed_factory,
            api_keys=api_keys,
            provider_kwargs=provider_kwargs,
            expected_session=expected_session,
            today=today,
        )
    except (ProviderError, PriceSourceError) as exc:
        attempted.append({"source": fallback, "error": f"{type(exc).__name__}: {exc}"})
        raise SourceSelectionError(
            f"neither the requested source {requested!r} nor the fallback {fallback!r} "
            f"could be built; last error: {exc}"
        ) from exc

    return SourceSelection(
        source=source,
        requested=requested,
        switched=True,
        reason=f"{requested} was unavailable; switched explicitly to {fallback}",
        primary_name=requested,
        fallback_name=fallback,
        attempted=attempted,
    )
