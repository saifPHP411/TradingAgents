"""Research helpers for Bybit PostOnly limit orders.

Pure payload construction only: this module performs no HTTP calls, reads no
credentials, and never submits or cancels an exchange order.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Literal


def _decimal(value: str | int | float | Decimal, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def build_post_only_limit_payload(
    *,
    symbol: str,
    side: Literal["Buy", "Sell"],
    quantity: str | int | float | Decimal,
    bid: str | int | float | Decimal,
    ask: str | int | float | Decimal,
    tick_size: str | int | float | Decimal,
    category: Literal["spot", "linear"] = "spot",
    order_link_id: str | None = None,
) -> dict[str, str]:
    """Build a passive Bybit V5 limit-order payload; never sends it.

    Buy quotes are rounded down from best bid; sell quotes are rounded up
    from best ask. A quote that would cross the opposite best price is rejected.
    """
    if not symbol or side not in ("Buy", "Sell"):
        raise ValueError("symbol and a valid side are required")
    if category not in ("spot", "linear"):
        raise ValueError("category must be spot or linear")
    qty = _decimal(quantity, "quantity")
    best_bid = _decimal(bid, "bid")
    best_ask = _decimal(ask, "ask")
    tick = _decimal(tick_size, "tick_size")
    if best_bid >= best_ask:
        raise ValueError("order book must have bid below ask")

    if side == "Buy":
        price = (best_bid / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        if price >= best_ask:
            raise ValueError("buy quote would be marketable; PostOnly must not take liquidity")
    else:
        units = (best_ask / tick).to_integral_value(rounding=ROUND_DOWN)
        price = units * tick
        if price < best_ask:
            price += tick
        if price <= best_bid:
            raise ValueError("sell quote would be marketable; PostOnly must not take liquidity")

    payload = {
        "category": category,
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": format(qty, "f"),
        "price": format(price, "f"),
        "timeInForce": "PostOnly",
    }
    if order_link_id:
        payload["orderLinkId"] = order_link_id
    return payload
