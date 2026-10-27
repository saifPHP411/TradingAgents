"""Execution: orders, simulated fills, and the costs they carry.

``orders`` defines the order and its lifecycle, ``costs`` the configurable cost
schedule, ``interface`` the mode guard that makes live trading unreachable, and
``simulator`` the engine that turns an order into a fill or a rejection.

There is no live path here and no way to add one by configuration.
"""

from tradingagents.execution.costs import NO_COSTS, CostBreakdown, CostModel, Side
from tradingagents.execution.interface import (
    ExecutionEngine,
    ExecutionMode,
    LiveTradingNotSupportedError,
    MarketContext,
    assert_simulation_only,
    resolve_mode,
)
from tradingagents.execution.orders import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
)
from tradingagents.execution.simulator import ExecutionSimulator

__all__ = [
    "NO_COSTS",
    "CostBreakdown",
    "CostModel",
    "ExecutionEngine",
    "ExecutionMode",
    "ExecutionSimulator",
    "LiveTradingNotSupportedError",
    "MarketContext",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "RejectReason",
    "Side",
    "assert_simulation_only",
    "resolve_mode",
]
