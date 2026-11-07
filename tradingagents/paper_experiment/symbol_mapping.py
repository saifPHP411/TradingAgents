"""Symbol mapping between market-data venues and execution venues.

This module provides explicit, validated mapping between Binance market-data symbols
(USDT-quoted) and Alpaca Paper execution symbols (USD-quoted). The mapping is
based on the base asset only, and must be validated against each venue's actual
tradable instruments before use.

Key principles:
- Market-data symbols (Binance) and execution symbols (Alpaca) are kept separate
- No silent USDT->USD conversion; prices from different venues are not identical
- Mapping is validated against both venues' instrument registries
- Unsupported pairs are rejected with clear reasons, never silently dropped
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules
from tradingagents.crypto.symbols import (
    binance_symbol,
    parse_pair,
)


@dataclass(frozen=True)
class SymbolMapping:
    """A validated mapping between market-data and execution symbols.

    Attributes:
        canonical: Canonical BASE/QUOTE symbol (e.g., BTC/USDT or BTC/USD)
        market_data_symbol: Symbol for market-data venue (Binance: BTCUSDT)
        execution_symbol: Symbol for execution venue (Alpaca: BTC/USD)
        base_asset: Base asset (e.g., BTC)
        market_data_quote: Quote asset on market-data venue (e.g., USDT)
        execution_quote: Quote asset on execution venue (e.g., USD)
        market_data_tradable: Whether the symbol is tradable on market-data venue
        execution_tradable: Whether the symbol is tradable on execution venue
    """
    canonical: str
    market_data_symbol: str
    execution_symbol: str
    base_asset: str
    market_data_quote: str
    execution_quote: str
    market_data_tradable: bool
    execution_tradable: bool
    market_data_rules: TradingRules | None = None
    execution_rules: TradingRules | None = None

    @property
    def fully_tradable(self) -> bool:
        """Whether the symbol is tradable on both venues."""
        return self.market_data_tradable and self.execution_tradable


def build_symbol_mapping(
    *,
    canonical_symbols: list[str] | tuple[str, ...],
    market_data_registry: InstrumentRegistry,
    execution_registry: InstrumentRegistry | None = None,
) -> tuple[list[SymbolMapping], dict[str, str]]:
    """Build validated symbol mappings from canonical symbols.

    Args:
        canonical_symbols: List of canonical BASE/QUOTE symbols (e.g., BTC/USDT)
        market_data_registry: InstrumentRegistry from market-data venue (Binance)
        execution_registry: Optional InstrumentRegistry from execution venue (Alpaca).
                           If None, execution tradability is marked as unknown.

    Returns:
        Tuple of (validated_mappings, rejected_symbols) where:
        - validated_mappings: List of SymbolMapping for symbols supported on market-data venue
        - rejected_symbols: Dict mapping canonical symbol to rejection reason

    Raises:
        ValueError: If a symbol cannot be parsed as a valid crypto pair.
    """
    mappings: list[SymbolMapping] = []
    rejected: dict[str, str] = {}

    for canonical in canonical_symbols:
        try:
            pair = parse_pair(canonical)
        except Exception as e:
            rejected[canonical] = f"invalid symbol format: {e}"
            continue

        # Check market-data venue (Binance)
        md_rules = market_data_registry.get(canonical)
        md_tradable = md_rules is not None and md_rules.tradable

        if not md_tradable:
            reason = "not tradable on market-data venue"
            if md_rules is None:
                reason = "not listed on market-data venue"
            elif md_rules.status != "TRADING":
                reason = f"status={md_rules.status} on market-data venue"
            rejected[canonical] = reason
            continue

        # Build market-data venue symbol (Binance uses base+quote without separator)
        md_symbol = binance_symbol(canonical)

        # Build execution venue symbol (Alpaca uses base/quote with /)
        # We map by BASE asset, keeping the execution venue's quote asset
        # The canonical symbol's quote may differ from execution venue's quote
        # (e.g., BTC/USDT canonical -> BTC/USD on Alpaca)
        base = pair.base
        # For execution, we need to determine the correct quote asset
        # For Alpaca, crypto pairs are typically USD-quoted
        exec_symbol = f"{base}/USD"

        # Check execution venue if registry provided
        exec_tradable = True
        exec_rules = None
        if execution_registry is not None:
            # Try the USD-quoted symbol first
            exec_rules = execution_registry.get(exec_symbol)
            if exec_rules is not None:
                exec_tradable = exec_rules.tradable
            else:
                # Try other common quote assets
                for quote in ("USDT", "USDC", "USD"):
                    test_symbol = f"{base}/{quote}"
                    exec_rules = execution_registry.get(test_symbol)
                    if exec_rules is not None:
                        exec_symbol = test_symbol
                        exec_tradable = exec_rules.tradable
                        break
                else:
                    exec_tradable = False
                    exec_rules = None

        if not exec_tradable:
            rejected[canonical] = "not tradable on execution venue"
            continue

        mappings.append(SymbolMapping(
            canonical=canonical,
            market_data_symbol=md_symbol,
            execution_symbol=exec_symbol,
            base_asset=base,
            market_data_quote=pair.quote,
            execution_quote=exec_symbol.split("/")[1] if "/" in exec_symbol else "USD",
            market_data_tradable=True,
            execution_tradable=exec_tradable,
            market_data_rules=md_rules,
            execution_rules=exec_rules,
        ))

    return mappings, rejected


def validate_execution_symbols(
    symbols: list[str] | tuple[str, ...],
    execution_registry: InstrumentRegistry,
) -> tuple[list[str], dict[str, str]]:
    """Validate symbols against execution venue's tradable instruments.

    Args:
        symbols: List of canonical BASE/QUOTE symbols to validate
        execution_registry: InstrumentRegistry from execution venue

    Returns:
        Tuple of (valid_symbols, rejected_symbols) where:
        - valid_symbols: List of canonical symbols that are tradable on execution venue
        - rejected_symbols: Dict mapping canonical symbol to rejection reason
    """
    valid: list[str] = []
    rejected: dict[str, str] = {}

    for canonical in symbols:
        try:
            pair = parse_pair(canonical)
        except Exception as e:
            rejected[canonical] = f"invalid symbol format: {e}"
            continue

        # Try to find matching symbol on execution venue
        # Alpaca uses BASE/QUOTE format with /
        base = pair.base
        found = False
        for quote in ("USD", "USDT", "USDC"):
            test_symbol = f"{base}/{quote}"
            rules = execution_registry.get(test_symbol)
            if rules is not None and rules.tradable:
                valid.append(canonical)
                found = True
                break
            elif rules is not None:
                rejected[canonical] = f"status={rules.status} on execution venue"
                found = True
                break

        if not found:
            rejected[canonical] = "not listed on execution venue"

    return valid, rejected


def get_allowed_symbols_for_broker(
    mappings: list[SymbolMapping],
) -> tuple[str, ...]:
    """Extract the execution symbols for broker's allowed_symbols config.

    Args:
        mappings: Validated SymbolMapping objects

    Returns:
        Tuple of execution symbols suitable for PaperRiskLimits.allowed_symbols
    """
    return tuple(m.execution_symbol for m in mappings if m.execution_tradable)


# Convenience function for the common Binance->Alpaca mapping
def map_binance_to_alpaca(
    canonical_symbols: list[str] | tuple[str, ...],
    binance_registry: InstrumentRegistry,
    alpaca_registry: InstrumentRegistry | None = None,
) -> tuple[list[SymbolMapping], dict[str, str]]:
    """Map canonical crypto symbols from Binance market-data to Alpaca execution.

    This is the standard mapping for the crypto TJR paper trading path:
    - Market data: Binance public 1m/5m bars (USDT-quoted)
    - Execution: Alpaca Paper (USD-quoted)

    Args:
        canonical_symbols: Canonical BASE/QUOTE symbols (e.g., BTC/USDT)
        binance_registry: Binance InstrumentRegistry from exchangeInfo
        alpaca_registry: Optional Alpaca InstrumentRegistry (if available)

    Returns:
        Tuple of (validated_mappings, rejected_symbols)
    """
    return build_symbol_mapping(
        canonical_symbols=canonical_symbols,
        market_data_registry=binance_registry,
        execution_registry=alpaca_registry,
    )


def get_alpaca_execution_symbol(canonical: str) -> str:
    """Convert a canonical symbol to Alpaca execution format.

    Maps by BASE asset, using USD as the quote asset (Alpaca standard).

    Args:
        canonical: Canonical BASE/QUOTE symbol (e.g., BTC/USDT)

    Returns:
        Alpaca execution symbol (e.g., BTC/USD)

    Raises:
        ValueError: If canonical symbol is invalid
    """
    pair = parse_pair(canonical)
    return f"{pair.base}/USD"


def get_binance_market_data_symbol(canonical: str) -> str:
    """Convert a canonical symbol to Binance market-data format.

    Args:
        canonical: Canonical BASE/QUOTE symbol (e.g., BTC/USDT)

    Returns:
        Binance market-data symbol (e.g., BTCUSDT)

    Raises:
        ValueError: If canonical symbol is invalid
    """
    return binance_symbol(canonical)
