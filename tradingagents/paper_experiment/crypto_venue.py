"""The crypto venues' own registries, read and translated into one shape.

The crypto paper path needs two registries before it may place an order:

- the **market-data** venue's (Binance) — which pairs exist for analysis, and
  their filters;
- the **execution** venue's (Alpaca) — which pairs exist to *trade*, and their
  lot size, price increment and minimum order size.

Both are read from the venue itself, never from a hard-coded list, because
"Alpaca supports over 20 coins" is not a list and "Binance lists 2500 symbols" is
not a list either. This module reads each venue's own answer and turns it into
the project's :class:`~tradingagents.crypto.instruments.InstrumentRegistry`, so
the rest of the system asks one question in one vocabulary.

The execution registry is read from the **paper** host only. The live trading
host is refused by the broker's own endpoint gate, and this module never issues
an order — it reads an asset list.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules

from .providers.alpaca_crypto import AlpacaCryptoClient


def _as_float(value: Any) -> float | None:
    """A finite float from a venue's decimal string, or ``None``."""
    import math

    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def build_execution_registry(
    assets: list[dict[str, Any]],
    *,
    venue: str = "alpaca_paper",
    fetched_at: str | None = None,
) -> InstrumentRegistry:
    """An :class:`InstrumentRegistry` from Alpaca's own crypto asset list.

    Only an asset the venue marks ``tradable`` **and** whose status is active
    becomes a tradable rule. An asset the venue lists but refuses to trade is
    recorded with its status, so the caller can say *why* a pair was rejected
    instead of only that it was.

    The venue's own numbers are kept: ``min_order_size``,
    ``min_trade_increment`` (the quantity step) and ``price_increment`` (the
    price tick). ``min_notional`` is not stated by the Alpaca assets API, so it
    is left ``None`` — unstated, not zero — and the project's own risk limits do
    the notional bounding.
    """
    rules: dict[str, TradingRules] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if asset.get("class") not in (None, "crypto"):
            continue
        symbol = str(asset.get("symbol") or "").strip().upper()
        if "/" not in symbol:
            # A crypto asset is a pair (``BTC/USD``); a bare symbol is not one.
            continue
        base, _, quote = symbol.partition("/")
        status = str(asset.get("status") or "active").upper()
        tradable = bool(asset.get("tradable")) and status == "ACTIVE"
        step = _as_float(asset.get("min_trade_increment"))
        tick = _as_float(asset.get("price_increment"))
        rules[symbol] = TradingRules(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            status="TRADING" if tradable else status,
            min_quantity=_as_float(asset.get("min_order_size")),
            quantity_step=step,
            price_tick=tick,
            order_types=("market", "limit", "stop_limit"),
            base_precision=_decimals(step),
            quote_precision=_decimals(tick),
            # Alpaca's assets API states fractionability directly. A missing
            # field is *unstated* (None), which is not the same as "not
            # fractionable" — the caller must not read it as a refusal.
            fractionable=_as_bool(asset.get("fractionable")),
        )
    return InstrumentRegistry(
        rules=rules,
        venue=venue,
        fetched_at=fetched_at or datetime.now(UTC).isoformat(),
    )


def _as_bool(value: Any) -> bool | None:
    """A venue's boolean field, or ``None`` when the venue did not state it.

    A missing field is unknown, not ``False``: reading an unstated value as a
    refusal would reject an instrument the venue may well accept.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _decimals(step: float | None) -> int:
    """The number of decimal places a step implies."""
    if step is None or step <= 0:
        return 0
    text = f"{step:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def read_alpaca_execution_registry(
    *,
    client: AlpacaCryptoClient | None = None,
    base_url: str,
    transport: Any | None = None,
) -> InstrumentRegistry:
    """Read Alpaca's crypto assets from the **paper** host and build the registry.

    ``base_url`` is the paper host; the caller has already validated it with the
    broker's own ``assert_paper_endpoint``, so a live host cannot be reached
    through here. The read raises on failure — a registry that could not be read
    is not an empty registry, and an empty one would look like "this venue trades
    nothing", which is a different and wrong claim.
    """
    from .broker.alpaca_paper import assert_paper_endpoint

    paper_url = assert_paper_endpoint(base_url)
    client = client or AlpacaCryptoClient()
    assets = client.assets(base_url=paper_url, transport=transport)
    return build_execution_registry(assets)


def build_bybit_execution_registry(
    instruments: dict[str, Any],
    *,
    fetched_at: str | None = None,
) -> InstrumentRegistry:
    """An :class:`InstrumentRegistry` from Bybit Demo's own Spot instrument list.

    Bybit's ``instruments-info`` reports each pair's own numbers:
    ``lotSizeFilter.minOrderQty``, ``lotSizeFilter.minOrderAmt``,
    ``lotSizeFilter.qtyStep`` and ``priceFilter.tickSize``. They are carried
    through as the venue stated them, so a pair's minimum is the venue's minimum
    and never a project-wide default.

    The symbol is written canonically as ``BASE/QUOTE`` so the registry speaks
    the same vocabulary as every other registry in the project. Bybit's own
    spelling (``BTCUSDT``) is kept in the rule's status-free fields only where
    it is the venue's; the key is canonical because that is what
    :meth:`InstrumentRegistry.get` looks up.
    """
    rules: dict[str, TradingRules] = {}
    for info in instruments.values():
        base = getattr(info, "base_coin", "") or ""
        quote = getattr(info, "quote_coin", "") or ""
        if not base or not quote:
            continue
        canonical = f"{base.upper()}/{quote.upper()}"
        status = str(getattr(info, "status", "") or "unknown")
        tradable = bool(getattr(info, "tradable", False))
        rules[canonical] = TradingRules(
            symbol=canonical,
            base_asset=base.upper(),
            quote_asset=quote.upper(),
            status="TRADING" if tradable else status.upper(),
            min_quantity=_as_float(getattr(info, "min_order_qty", None)),
            max_quantity=_as_float(getattr(info, "max_order_qty", None)),
            quantity_step=_as_float(getattr(info, "qty_step", None)),
            price_tick=_as_float(getattr(info, "tick_size", None)),
            min_notional=_as_float(getattr(info, "min_order_amt", None)),
            # Bybit Spot states no fractionability field, so the answer is
            # unknown. An unstated answer is not a refusal.
            fractionable=None,
        )
    return InstrumentRegistry(
        rules=rules,
        venue="bybit_demo",
        fetched_at=fetched_at or datetime.now(UTC).isoformat(),
    )


def read_bybit_execution_registry(
    *,
    base_url: str,
    transport: Any | None = None,
) -> InstrumentRegistry:
    """Read Bybit Demo's own Spot instrument list and build the registry.

    ``base_url`` is checked by the adapter's own demo-endpoint gate before any
    request, so a live host cannot be reached through here. The read raises on
    failure: a registry that could not be read is not an empty registry, and an
    empty one would look like "this venue trades nothing", which is a different
    and wrong claim.
    """
    from .broker.bybit_demo import DEMO_BASE_URL, BybitDemoBroker

    demo_url = base_url or DEMO_BASE_URL
    broker = BybitDemoBroker(base_url=demo_url, transport=transport)
    return build_bybit_execution_registry(broker.load_instruments())


__all__ = [
    "build_bybit_execution_registry",
    "build_execution_registry",
    "read_alpaca_execution_registry",
    "read_bybit_execution_registry",
]
