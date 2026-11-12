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

from tradingagents.risk.paper_parity import FRACTIONAL_SHARES

from .broker import (
    DECISION_SOURCE_CHOICES,
    DECISION_SOURCE_CRYPTO,
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_HISTORY,
    DECISION_SOURCE_TJR,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_BYBIT_DEMO,
    EXECUTION_PROVIDER_CHOICES,
    MARKET_DATA_CHOICES,
    PROVIDER_ALPACA_SNAPSHOT,
    PROVIDER_BAR_OPEN,
    PROVIDER_BINANCE_SNAPSHOT,
)

MARKET_DATA_ENV = "TRADINGAGENTS_PAPER_MARKET_DATA"
"""Environment variable the entrypoints read the provider choice from."""

DECISION_SOURCE_ENV = "TRADINGAGENTS_PAPER_DECISION_SOURCE"
"""Environment variable the entrypoints read the decision-source choice from."""

EXECUTION_PROVIDER_ENV = "TRADINGAGENTS_PAPER_EXECUTION_PROVIDER"
"""Environment variable the entrypoints read the execution-broker choice from."""

FRACTIONAL_SHARES_ENV = "TRADINGAGENTS_PAPER_FRACTIONAL_SHARES"
"""Environment variable that opts into fractional sizing (default: fractional, the
one parity constant)."""

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


def resolve_decision_source(choice: str | None) -> str:
    """Validate an explicit decision-source choice, or raise.

    No default is applied here: the caller supplies the default (``history``), so
    a bad value from a flag or an environment variable is a loud error rather than
    a silent fall-back to a different decision source.
    """
    if choice is None:
        raise MarketDataWiringError("no decision source was given")
    if choice not in DECISION_SOURCE_CHOICES:
        raise MarketDataWiringError(
            f"unknown decision source {choice!r}; expected one of "
            f"{', '.join(DECISION_SOURCE_CHOICES)}"
        )
    return choice


def env_decision_source(default: str = DECISION_SOURCE_HISTORY) -> str:
    """The decision source named by :data:`DECISION_SOURCE_ENV`, or ``default``."""
    return resolve_decision_source(os.environ.get(DECISION_SOURCE_ENV) or default)


def resolve_execution_provider(choice: str | None) -> str:
    """Validate an explicit execution-broker choice, or raise.

    No default is applied here: the caller supplies the default
    (``alpaca_paper``), so a bad value from a flag or an environment variable
    is a loud error rather than a silent fall-back to a different broker.

    There is deliberately **no** fall-back from ``bybit_demo`` to
    ``alpaca_paper``: a Bybit selection whose credentials are missing stops the
    run rather than trading somewhere the operator did not choose.
    """
    if choice is None:
        raise MarketDataWiringError("no execution provider was given")
    if choice not in EXECUTION_PROVIDER_CHOICES:
        raise MarketDataWiringError(
            f"unknown execution provider {choice!r}; expected one of "
            f"{', '.join(EXECUTION_PROVIDER_CHOICES)}"
        )
    return choice


def env_execution_provider(default: str = EXECUTION_PROVIDER_ALPACA_PAPER) -> str:
    """The execution broker named by :data:`EXECUTION_PROVIDER_ENV`, or ``default``."""
    return resolve_execution_provider(os.environ.get(EXECUTION_PROVIDER_ENV) or default)


def describe_execution_provider(provider: str) -> str:
    """A one-line description of an execution broker, for a start log."""
    if provider == EXECUTION_PROVIDER_BYBIT_DEMO:
        return (
            "bybit_demo (Bybit Demo Spot only: crypto Spot, demo trading, long-only; "
            "needs BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET)"
        )
    return "alpaca_paper (Alpaca Paper; the default)"


def env_fractional(default: bool = FRACTIONAL_SHARES) -> bool:
    """Whether fractional sizing is opted into via :data:`FRACTIONAL_SHARES_ENV`.

    The default is the one parity constant, so an unset environment variable and
    an unset flag both leave sizing on the same footing as the TJR backtest.
    """
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
        resolved_fractional = parse_bool(fractional, default=FRACTIONAL_SHARES)
    else:
        resolved_fractional = bool(fractional)
    return resolved_provider, resolved_max_age, not resolved_fractional


def resolve_paper_settings(
    *,
    provider: str | None = None,
    max_age: str | float | None = None,
    fractional: str | bool | None = None,
    decision_source: str | None = None,
) -> tuple[str, float | None, bool, str]:
    """Resolve the paper market-data settings *and* the decision source.

    The same CLI-over-environment precedence as :func:`resolve_settings`, extended
    with the decision-source choice. Kept as a separate function so existing
    callers of :func:`resolve_settings` are unchanged. Returns
    ``(provider, max_age_seconds, whole_shares, decision_source)``.
    """
    resolved_provider, resolved_max_age, whole_shares = resolve_settings(
        provider=provider, max_age=max_age, fractional=fractional
    )
    resolved_source = (
        env_decision_source()
        if decision_source is None
        else resolve_decision_source(decision_source)
    )
    return resolved_provider, resolved_max_age, whole_shares, resolved_source


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
        if provider == PROVIDER_BINANCE_SNAPSHOT:
            from .broker import BinanceSnapshotMarketData

            return BinanceSnapshotMarketData(symbols, max_age_seconds=max_age_seconds)
        from .broker import AlpacaSnapshotMarketData

        return AlpacaSnapshotMarketData(symbols, max_age_seconds=max_age_seconds)

    return factory


def describe_provider(choice: str) -> str:
    """A one-line, operator-facing description of a provider choice."""
    provider = resolve_provider(choice)
    if provider == PROVIDER_BINANCE_SNAPSHOT:
        return (
            "binance_snapshot: a CURRENT crypto price from Binance public market data "
            "(no API key, no registration); validated for freshness. Pair with "
            "--paper-decision-source crypto."
        )
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


def describe_decision_source(choice: str) -> str:
    """A one-line, operator-facing description of a decision-source choice."""
    source = resolve_decision_source(choice)
    if source == DECISION_SOURCE_CURRENT_MARKET:
        return (
            "current_market: an INDEPENDENT decision taken live over current Alpaca "
            "market data, gated on Alpaca's market clock; requires "
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET and an LLM credential. "
            "Not blocked by a consumed historical experiment."
        )
    if source == DECISION_SOURCE_TJR:
        return (
            "tjr: the deterministic TJR rule chain (no model) over current Alpaca "
            "INTRADAY 1m/5m market data, gated on Alpaca's market clock and the "
            "RiskEngine; requires ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET. "
            "One decision per symbol per session. Pair with --paper-market-data "
            "alpaca_snapshot so orders size at a current price."
        )
    if source == DECISION_SOURCE_CRYPTO:
        return (
            "crypto: the deterministic TJR rule chain (no model) over current Binance "
            "INTRADAY 1m/5m market data (no API key), gated on a 24/7 crypto clock and "
            "the RiskEngine. One decision per symbol per UTC session, id "
            "tjr-<symbol>-<session>. Pair with --paper-market-data binance_snapshot so "
            "orders size at a current crypto price. Requires a crypto-capable paper "
            "broker; paper-only."
        )
    return (
        "history: the paper experiment's recorded, HISTORICAL decisions (the "
        "default); bounded by the experiment's sessions, so it reports "
        "HISTORY_EXHAUSTED once they are consumed"
    )


__all__ = [
    "DECISION_SOURCE_ENV",
    "FRACTIONAL_SHARES_ENV",
    "MARKET_DATA_ENV",
    "SNAPSHOT_MAX_AGE_ENV",
    "MarketDataWiringError",
    "describe_decision_source",
    "describe_provider",
    "env_decision_source",
    "env_fractional",
    "env_max_age",
    "env_provider",
    "market_data_factory_for",
    "parse_bool",
    "resolve_decision_source",
    "resolve_max_age",
    "resolve_paper_settings",
    "resolve_provider",
    "resolve_settings",
]
