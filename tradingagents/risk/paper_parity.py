"""The paper account's risk limits, in one place, so two paths cannot disagree.

The paper worker and the TJR backtest must enforce the *same* caps: a backtest
that sizes and counts differently from the worker measures a strategy nobody will
run, and the difference stays invisible until real money is at stake. These
constants are that single source. Every consumer reads them rather than repeating
a number that can drift.

The numbers are the worker's, unchanged: a 25 USD ceiling on one entry order, at
most 5 open positions, at most 10 entries a session, no shorts, fractional shares
allowed. Nothing here is a new policy; it is the existing paper policy made
importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tradingagents.risk.engine import RiskLimits

MAX_ORDER_NOTIONAL: float = 25.0
"""Largest notional one entry order may carry, in the account currency."""

MAX_OPEN_POSITIONS: int = 5
"""Most distinct positions the book may hold at once."""

MAX_DAILY_ORDERS: int = 10
"""Most entry orders that may be placed in one trading session."""

ALLOW_SHORT: bool = False
"""Whether a short may be opened. Kept False: the book is long-only."""

FRACTIONAL_SHARES: bool = True
"""Whether a sized quantity may be fractional. ``True`` keeps the size exact, so a
$25 order on a $200+ name still buys a positive quantity instead of flooring to
zero. The paper worker's ``whole_shares`` is its negation, and both read this one
constant (see :data:`WHOLE_SHARES`)."""

WHOLE_SHARES: bool = not FRACTIONAL_SHARES
"""The paper worker's ``whole_shares`` setting, derived from
:data:`FRACTIONAL_SHARES` so the two paths cannot disagree.

``PaperTradingConfig.whole_shares`` and the entrypoints default to this, and the
TJR backtest reads :func:`paper_parity_limits` (whose ``fractional_shares`` is
:data:`FRACTIONAL_SHARES`). One constant therefore decides both. Flipping it
flips sizing on both sides together; there is no second place to edit."""


def paper_parity_limits() -> RiskLimits:
    """A :class:`RiskLimits` carrying exactly the paper account's caps.

    The position and exposure fractions keep their engine defaults: they are
    stricter than the 25 USD order cap could ever bind on a funded book, so
    leaving them on cannot weaken anything, and turning them off would.
    """
    from tradingagents.risk.engine import RiskLimits

    return RiskLimits(
        max_open_positions=MAX_OPEN_POSITIONS,
        max_daily_orders=MAX_DAILY_ORDERS,
        max_order_value=MAX_ORDER_NOTIONAL,
        target_order_value=MAX_ORDER_NOTIONAL,
        long_only=not ALLOW_SHORT,
        fractional_shares=FRACTIONAL_SHARES,
    )


__all__ = [
    "ALLOW_SHORT",
    "FRACTIONAL_SHARES",
    "MAX_DAILY_ORDERS",
    "MAX_OPEN_POSITIONS",
    "MAX_ORDER_NOTIONAL",
    "WHOLE_SHARES",
    "paper_parity_limits",
]
