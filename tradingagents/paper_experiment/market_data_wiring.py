"""One place that decides *which* price seam a paper worker uses.

Both execution entrypoints — the ``serve`` subcommand and the module-form
dashboard server — select their market-data source through the functions here, so
they cannot drift: a provider choice means the same seam, the same environment
variable, and the same freshness default whichever way the process is started.

The choice is explicit and has no silent fallback. ``bar_open`` is the historical
default (the experiment's latest daily-bar open) and is a *different purpose* from
``alpaca_snapshot`` (a current price, validated for freshness). Selecting
``alpaca_snapshot`` never degrades to historical prices: if the credential is
missing or the snapshot is stale, the worker refuses to size an order rather than
quietly trade a stale number.
"""

from __future__ import annotations

import os

from .broker import (
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    MARKET_DATA_CHOICES,
    PROVIDER_ALPACA_SNAPSHOT,
    PROVIDER_BAR_OPEN,
)

MARKET_DATA_ENV = "TRADINGAGENTS_PAPER_MARKET_DATA"
"""Environment variable the entrypoints read the provider choice from."""

FRACTIONAL_SHARES_ENV = "TRADINGAGENTS_PAPER_FRACTIONAL_SHARES"
"""Environment variable that opts into fractional sizing (default: whole shares)."""

SNAPSHOT_MAX_AGE_ENV = "TRADINGAGENTS_PAPER_SNAPSHOT_MAX_AGE_SECONDS"
"""Environment variable for the maximum age of a snapshot price, in seconds."""

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off", "")


class MarketDataWiringError(ValueError):
    """A provider or freshness setting could not be resolved."""


def resolve_provider(choice: str | None) -> str:
    """Validate an explicit provider choice, or raise.

    No default is applied here: the caller supplies the default, so a bad value
    from a flag or an environment variable is a loud error rather than a silent
    fall-back to historical prices.
    """
    if choice is None:
        raise MarketDataWiringError("no market-data provider was given")
    if choice not in MARKET_DATA_CHOICES:
        raise MarketDataWiringError(
            f"unknown market-data provider {choice!r}; expected one of "
            f"{', '.join(MARKET_DATA_CHOICES)}"
        )
    return choice


def parse_bool(value: str | None, *, default: bool) -> bool:
    """Parse a permissive boolean flag value, or return ``default`` when unset."""
    if value is None:
        return default
    text = value.strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise MarketDataWiringError(
        f"expected a boolean (one of {', '.join(_TRUTHY + _FALSY)}), got {value!r}"
    )


def resolve_max_age(value: str | float | None) -> float | None:
    """Resolve the snapshot maximum age in seconds, or ``None`` to disable it."""
    if value is None:
        return DEFAULT_MAX_SNAPSHOT_AGE_SECONDS
    if isinstance(value, str) and value.strip().lower() in ("none", "off", "disabled"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MarketDataWiringError(
            f"the snapshot maximum age must be a number of seconds or 'none', got {value!r}"
        ) from exc
    if not (number > 0):
        raise MarketDataWiringError(
            f"the snapshot maximum age must be positive, got {number!r}"
        )
    return number


def env_provider(default: str = PROVIDER_BAR_OPEN) -> str:
    """The provider named by :data:`MARKET_DATA_ENV`, or ``default`` when unset."""
    return resolve_provider(os.environ.get(MARKET_DATA_ENV) or default)


def env_fractional(default: bool = False) -> bool:
    """Whether fractional sizing is opted into via :data:`FRACTIONAL_SHARES_ENV`."""
    return parse_bool(os.environ.get(FRACTIONAL_SHARES_ENV), default=default)


def env_max_age(default: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS) -> float | None:
    """The snapshot maximum age from :data:`SNAPSHOT_MAX_AGE_ENV`, or ``default``."""
    raw = os.environ.get(SNAPSHOT_MAX_AGE_ENV)
    if raw is None or raw == "":
        return default
    return resolve_max_age(raw)


def resolve_settings(
    *,
    provider: str | None = None,
    max_age: str | float | None = None,
    fractional: str | bool | None = None,
) -> tuple[str, float | None, bool]:
    """Resolve the three paper market-data settings with CLI-over-env precedence.

    Each argument is the CLI value, or ``None`` when the flag was not given. The
    environment is consulted **only** for a setting the CLI did not provide, so an
    explicit flag always wins and an unset flag falls back to the environment and
    then to the safe default. Returns ``(provider, max_age_seconds, whole_shares)``.

    A bad value — from either source — raises :class:`MarketDataWiringError`
    rather than falling back, so a typo can never silently change which price a
    paper order is sized from.
    """
    resolved_provider = env_provider() if provider is None else resolve_provider(provider)
    resolved_max_age = env_max_age() if max_age is None else resolve_max_age(max_age)
    if fractional is None:
        resolved_fractional = env_fractional()
    elif isinstance(fractional, str):
        resolved_fractional = parse_bool(fractional, default=False)
    else:
        resolved_fractional = bool(fractional)
    return resolved_provider, resolved_max_age, not resolved_fractional


def market_data_factory_for(choice: str, *, max_age_seconds: float | None = None):
    """The market-data factory the chosen provider selects.

    ``bar_open`` returns ``None``, which keeps the controller's historical default
    (the experiment's latest daily-bar open). ``alpaca_snapshot`` returns a factory
    that builds a current-price source from Alpaca market data, carrying the
    configured freshness limit. A factory that cannot be built (a missing
    credential) raises when the worker is constructed, so a misconfiguration is
    reported rather than silently replaced by a stale price.
    """
    provider = resolve_provider(choice)
    if provider == PROVIDER_BAR_OPEN:
        return None
    if max_age_seconds is None:
        max_age_seconds = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS

    def factory(symbols):
        from .broker import AlpacaSnapshotMarketData

        return AlpacaSnapshotMarketData(symbols, max_age_seconds=max_age_seconds)

    return factory


def describe_provider(choice: str) -> str:
    """A one-line, operator-facing description of a provider choice."""
    provider = resolve_provider(choice)
    if provider == PROVIDER_ALPACA_SNAPSHOT:
        return (
            "alpaca_snapshot: a CURRENT price from Alpaca market data (feed=iex), "
            "validated for freshness; requires ALPACA_PAPER_API_KEY / "
            "ALPACA_PAPER_API_SECRET and IEX feed access"
        )
    return (
        "bar_open: the HISTORICAL daily-bar open for the experiment's latest "
        "session (a past-session price, not a live quote)"
    )


__all__ = [
    "FRACTIONAL_SHARES_ENV",
    "MARKET_DATA_ENV",
    "SNAPSHOT_MAX_AGE_ENV",
    "MarketDataWiringError",
    "describe_provider",
    "env_fractional",
    "env_max_age",
    "env_provider",
    "market_data_factory_for",
    "parse_bool",
    "resolve_max_age",
    "resolve_provider",
    "resolve_settings",
]
