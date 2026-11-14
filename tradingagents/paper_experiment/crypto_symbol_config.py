"""The crypto symbol configuration: one request, two venue registries, one answer.

The project reads crypto from **two venues that do not share a symbol list**:

- **Binance** (market data) serves USDT-quoted spot pairs — ``BTC/USDT``;
- **Alpaca** (execution) serves USD-quoted pairs — ``BTC/USD``.

A pair that one venue lists is not necessarily listed by the other, and a pair's
quote asset is part of its identity: ``BTC/USDT`` and ``BTC/USD`` are *different
instruments* with different prices. A single shared list of symbols is therefore
not a safe configuration, and this module exists to make the two explicit.

What it does:

- reads the **requested** symbols from one place (a flag, an environment variable
  or the default list) and parses them into canonical pairs;
- validates every one of them against the **market-data venue's** registry — a pair
  Binance does not list cannot be analysed;
- validates every one of them against the **execution venue's** registry — a pair
  Alpaca does not list cannot be traded;
- reports **accepted** and **rejected** symbols separately, with the venue's own
  reason, so a dashboard can show exactly which pair was dropped and why;
- never silently substitutes a different pair, and never falls back to a different
  symbol list.

The result is a :class:`CryptoSymbolConfiguration`: the accepted canonical
symbols, the rejected ones with their reasons, and the per-symbol mapping from the
canonical decision symbol to each venue's own spelling.

The two environment variables this project already has keep their distinct jobs:

- ``TRADINGAGENTS_CRYPTO_SYMBOLS`` — the pairs the **dashboard displays** (the
  live market section). It is a *request*; a pair the venue does not serve is
  reported, not hidden.
- ``TRADINGAGENTS_EXPERIMENT_SYMBOLS`` — the pairs the **experiment** is created
  with (the persisted session state, the decision source and the risk book). It is
  the list that ends up in the experiment, after validation.

Neither variable is a substitute for the other, and neither is allowed to change
what is already persisted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules
from tradingagents.crypto.symbols import parse_pair

from .symbol_mapping import SymbolMapping

CRYPTO_SYMBOLS_ENV = "TRADINGAGENTS_CRYPTO_SYMBOLS"
"""Environment variable naming the crypto pairs the live dashboard section shows.

Space- or comma-separated, e.g. ``"BTC/USDT ETH/USDT"``. It selects what the
dashboard *displays*; it never decides what can be traded — the venue's own
listing and the risk engine do that.
"""

EXPERIMENT_SYMBOLS_ENV = "TRADINGAGENTS_EXPERIMENT_SYMBOLS"
"""Environment variable naming the experiment symbols (canonical crypto pairs).

Space- or comma-separated, e.g. ``"BTC/USDT ETH/USDT"``. Used to create the
experiment idempotently on first startup, and it is the list that is *persisted*
into the experiment once it has been validated against both venue registries.
"""

DEFAULT_MARKET_DATA_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "BCH/USDT",
)
"""The default market-data request: the liquid majors in Binance's USDT pairs.

These are the pairs Binance actually lists (verified against its public
``exchangeInfo``), and they are a *request*: each one is still validated against
both venue registries before it is used, so a pair one venue does not serve is
reported and excluded rather than assumed.
"""

DEFAULT_EXECUTION_SYMBOLS: tuple[str, ...] = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "BNB/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
)
"""The default execution request: the same majors in Alpaca's USD pairs.

Also a *request*. Alpaca supports "over 20 coins" and the exact list changes, so
every pair here is checked against the venue's own asset registry before an order
may name it. A pair Alpaca does not list is reported as rejected, and no order is
ever sent for it.
"""

QUOTE_PREFERENCE: tuple[str, ...] = ("USD", "USDT", "USDC")
"""Quote assets considered when resolving an execution pair, in order.

The *canonical* symbol keeps its own quote — the market-data venue's — because
that is the instrument the strategy analysed. The execution venue's quote is
chosen separately, from the venue's own registry, so no USDT-to-USD rate is ever
assumed.

**USD comes first.** The account is funded in USD and every figure the order
carries — the reference price, the entry, the stop, the target, the quantity and
the resulting P&L — is denominated in the execution venue's quote. A USD-quoted
pair therefore keeps all of them in the account's own currency, with no
conversion step and no risk of a USDT figure being read as a USD one.

A venue that lists only a USDT pair is still usable: the quote is then taken from
the venue's own registry and the conversion is done explicitly, from a named,
recent rate. It is never *assumed* — the venue's listing decides which pair
exists, and this order only decides which of the venue's own pairs is preferred.
"""


class SymbolConfigurationError(ValueError):
    """The requested symbol configuration could not be resolved."""


REJECTION_MARKET_DATA_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
"""The market-data venue does not list the requested pair."""

REJECTION_MARKET_DATA_NOT_TRADABLE = "MARKET_DATA_NOT_TRADABLE"
"""The market-data venue lists the pair but does not trade it."""

REJECTION_EXECUTION_UNAVAILABLE = "EXECUTION_UNAVAILABLE"
"""The execution venue does not list the pair (in any preferred quote)."""

REJECTION_EXECUTION_NOT_TRADABLE = "EXECUTION_NOT_TRADABLE"
"""The execution venue lists the pair but does not trade it."""

REJECTION_EXECUTION_NOT_FRACTIONABLE = "EXECUTION_NOT_FRACTIONABLE"
"""The execution venue lists the pair but refuses fractional quantities.

This is only a rejection when fractional execution is actually required. A
whole-unit order is still possible for such an instrument, so the check is
made against the requirement, not assumed.
"""

REJECTION_EXECUTION_REGISTRY_UNAVAILABLE = "EXECUTION_REGISTRY_UNAVAILABLE"
"""The execution venue's own listing could not be read.

Unknown is not tradable: without the venue's listing, tradability on the
execution side cannot be established, so the symbol is rejected rather than
assumed to be fine.
"""


@dataclass(frozen=True)
class SymbolRejection:
    """One requested symbol that a venue will not accept, and why.

    ``code`` is a stable, machine-readable reason; ``market_data_reason`` and
    ``execution_reason`` are the venues' own words. A dashboard shows the code
    and the detail, so an operator sees *why* a pair was dropped without having
    to read prose to tell two failure modes apart.
    """

    symbol: str
    market_data_reason: str | None = None
    execution_reason: str | None = None
    code: str = REJECTION_MARKET_DATA_UNAVAILABLE

    @property
    def reason(self) -> str:
        """The single reason the symbol was rejected, preferring the venue first."""
        if self.market_data_reason and self.execution_reason:
            return (
                f"market-data venue: {self.market_data_reason}; "
                f"execution venue: {self.execution_reason}"
            )
        return self.market_data_reason or self.execution_reason or "unsupported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "code": self.code,
            "market_data_reason": self.market_data_reason,
            "execution_reason": self.execution_reason,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CryptoSymbolConfiguration:
    """The resolved symbol configuration: what is accepted and what is not.

    ``accepted`` holds the canonical decision symbols — the pair the strategy
    analysed and the pair the experiment persists. ``rejected`` holds the rest,
    with the venue's own reason. ``mappings`` holds the per-symbol translation to
    each venue's spelling.
    """

    requested: tuple[str, ...] = ()
    accepted: tuple[str, ...] = ()
    rejected: tuple[SymbolRejection, ...] = ()
    mappings: tuple[SymbolMapping, ...] = ()
    market_data_registry: InstrumentRegistry | None = None
    execution_registry: InstrumentRegistry | None = None

    @property
    def symbols(self) -> tuple[str, ...]:
        """The accepted canonical symbols — the list to persist."""
        return self.accepted

    def mapping_for(self, canonical: str) -> SymbolMapping | None:
        """The mapping for a canonical decision symbol, or ``None``."""
        key = parse_pair(canonical).canonical
        for mapping in self.mappings:
            if mapping.canonical == key:
                return mapping
        return None

    def rejected_for(self, canonical: str) -> SymbolRejection | None:
        """The rejection for a canonical symbol, or ``None`` when it was accepted."""
        key = parse_pair(canonical).canonical
        for rejection in self.rejected:
            if rejection.symbol == key:
                return rejection
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": list(self.requested),
            "accepted": list(self.accepted),
            "rejected": [r.to_dict() for r in self.rejected],
            "mappings": [
                {
                    "canonical": m.canonical,
                    "market_data_symbol": m.market_data_symbol,
                    "execution_symbol": m.execution_symbol,
                    "base_asset": m.base_asset,
                    "market_data_quote": m.market_data_quote,
                    "execution_quote": m.execution_quote,
                    "market_data_tradable": m.market_data_tradable,
                    "execution_tradable": m.execution_tradable,
                }
                for m in self.mappings
            ],
        }

    def describe(self) -> str:
        """A one-line, operator-facing summary of what was accepted and dropped."""
        parts = [f"{len(self.accepted)} accepted"]
        if self.rejected:
            dropped = ", ".join(f"{r.symbol} ({r.reason})" for r in self.rejected)
            parts.append(f"rejected: {dropped}")
        return "; ".join(parts)


def parse_symbol_list(raw: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Parse a symbol list from a flag, an environment string, or a sequence.

    Space- and comma-separated strings are split; a sequence is taken as it is.
    Every entry is canonicalised, so ``btc-usdt`` and ``BTC/USDT`` are the same
    symbol. A malformed entry is a **loud** error (:class:`SymbolConfigurationError`),
    never a silently dropped symbol.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [part for part in raw.replace(",", " ").split() if part]
    else:
        parts = [str(part) for part in raw if str(part).strip()]
    canonical: list[str] = []
    for part in parts:
        try:
            canonical.append(parse_pair(part).canonical)
        except Exception as exc:  # noqa: BLE001 - a bad symbol is reported, not dropped
            raise SymbolConfigurationError(
                f"{part!r} is not a valid crypto pair: {exc}"
            ) from exc
    # Preserve order, drop duplicates: the first mention wins.
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in canonical:
        if symbol not in seen:
            seen.add(symbol)
            ordered.append(symbol)
    return tuple(ordered)


def env_symbols(name: str) -> tuple[str, ...]:
    """The symbols named by an environment variable, or an empty tuple."""
    return parse_symbol_list(os.environ.get(name, ""))


def _execution_symbol_for(
    base: str,
    execution_registry: InstrumentRegistry,
) -> tuple[str | None, str | None, str]:
    """The execution venue's own symbol for ``base``, and the reason if there is none.

    The quote is chosen from the execution registry's own listings, in a fixed
    preference order, so no quote is assumed and no USDT-to-USD conversion is
    implied. The venue's symbol is returned verbatim.

    Returns the symbol, the prose reason and the stable rejection code.
    """
    for quote in QUOTE_PREFERENCE:
        candidate = f"{base}/{quote}"
        rules = execution_registry.get(candidate)
        if rules is None:
            continue
        if not rules.tradable:
            return None, f"status={rules.status} on execution venue", (
                REJECTION_EXECUTION_NOT_TRADABLE
            )
        return candidate, None, ""
    return None, "not listed on execution venue", REJECTION_EXECUTION_UNAVAILABLE


def resolve_symbol_configuration(
    requested: list[str] | tuple[str, ...],
    *,
    market_data_registry: InstrumentRegistry,
    execution_registry: InstrumentRegistry | None = None,
    require_fractional: bool = True,
) -> CryptoSymbolConfiguration:
    """Validate every requested symbol against **both** venue registries.

    A symbol is accepted only when the market-data venue lists it as trading
    *and* the execution venue lists it as trading. It is otherwise rejected with
    the venue's own reason, and it is never substituted for a different pair.

    ``execution_registry`` may be ``None``, which rejects every symbol with an
    explicit "execution venue registry not available" reason: without the venue's
    own listing, tradability on the execution side is unknown, and unknown is not
    tradable.

    ``require_fractional`` says whether a fractional quantity is actually
    needed. It is a rejection only when it is: a venue that refuses fractional
    quantities can still take a whole-unit order, so the requirement — not an
    assumption — decides. A venue that does not *state* fractionability is not
    refused for it; an unstated answer is unknown, not a refusal.
    """
    symbols = parse_symbol_list(tuple(requested))
    accepted: list[str] = []
    rejections: list[SymbolRejection] = []
    mappings: list[SymbolMapping] = []

    for canonical in symbols:
        md_rules = market_data_registry.get(canonical)
        md_reason: str | None = None
        md_code = ""
        if md_rules is None:
            md_reason = "not listed on market-data venue"
            md_code = REJECTION_MARKET_DATA_UNAVAILABLE
        elif not md_rules.tradable:
            md_reason = f"status={md_rules.status} on market-data venue"
            md_code = REJECTION_MARKET_DATA_NOT_TRADABLE

        if execution_registry is None:
            exec_reason: str | None = "execution venue registry not available"
            exec_code = REJECTION_EXECUTION_REGISTRY_UNAVAILABLE
            exec_symbol: str | None = None
            exec_rules: TradingRules | None = None
        else:
            base = parse_pair(canonical).base
            exec_symbol, exec_reason, exec_code = _execution_symbol_for(
                base, execution_registry
            )
            exec_rules = execution_registry.get(exec_symbol) if exec_symbol else None
            # Only a venue that *states* it refuses fractional quantities is
            # refused. An unstated answer is unknown, not a refusal.
            if (
                exec_symbol
                and require_fractional
                and exec_rules is not None
                and exec_rules.fractionable is False
            ):
                exec_reason = "execution venue does not accept fractional quantities"
                exec_code = REJECTION_EXECUTION_NOT_FRACTIONABLE

        if md_reason or exec_reason:
            rejections.append(
                SymbolRejection(
                    symbol=canonical,
                    market_data_reason=md_reason,
                    execution_reason=exec_reason,
                    code=md_code or exec_code or REJECTION_MARKET_DATA_UNAVAILABLE,
                )
            )
            continue

        assert md_rules is not None  # md_reason is None, so the venue listed it
        assert exec_rules is not None  # exec_reason is None, so the venue listed it
        accepted.append(canonical)
        mappings.append(
            SymbolMapping(
                canonical=canonical,
                market_data_symbol=canonical.replace("/", ""),
                execution_symbol=exec_symbol or "",
                base_asset=parse_pair(canonical).base,
                market_data_quote=parse_pair(canonical).quote,
                execution_quote=(exec_symbol or "").split("/")[-1],
                market_data_tradable=True,
                execution_tradable=True,
                market_data_rules=md_rules,
                execution_rules=exec_rules,
            )
        )

    return CryptoSymbolConfiguration(
        requested=symbols,
        accepted=tuple(accepted),
        rejected=tuple(rejections),
        mappings=tuple(mappings),
        market_data_registry=market_data_registry,
        execution_registry=execution_registry,
    )


__all__ = [
    "CRYPTO_SYMBOLS_ENV",
    "DEFAULT_EXECUTION_SYMBOLS",
    "DEFAULT_MARKET_DATA_SYMBOLS",
    "EXPERIMENT_SYMBOLS_ENV",
    "QUOTE_PREFERENCE",
    "REJECTION_EXECUTION_NOT_FRACTIONABLE",
    "REJECTION_EXECUTION_NOT_TRADABLE",
    "REJECTION_EXECUTION_REGISTRY_UNAVAILABLE",
    "REJECTION_EXECUTION_UNAVAILABLE",
    "REJECTION_MARKET_DATA_NOT_TRADABLE",
    "REJECTION_MARKET_DATA_UNAVAILABLE",
    "CryptoSymbolConfiguration",
    "SymbolConfigurationError",
    "SymbolRejection",
    "env_symbols",
    "parse_symbol_list",
    "resolve_symbol_configuration",
]
