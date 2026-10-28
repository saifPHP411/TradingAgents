"""Execution: orders, simulated fills, the costs they carry, and paper sessions.

``orders`` defines the order and its lifecycle, ``costs`` the configurable cost
schedule, ``interface`` the mode guard that makes live trading unreachable,
``simulator`` the engine that turns an order into a fill or a rejection, and
``paper`` the session driver that runs signals through the same pipeline a
backtest uses, against live quotes supplied by its caller.

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
    TimeInForce,
)
from tradingagents.execution.paper import (
    PAPER_STATE_SCHEMA_VERSION,
    PaperSession,
    PaperSessionError,
    PaperStateError,
    Quote,
    QuoteSource,
    SessionStep,
    dict_quote_source,
    new_session,
)
from tradingagents.execution.simulator import ExecutionSimulator

__all__ = [
    "NO_COSTS",
    "PAPER_STATE_SCHEMA_VERSION",
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
    "PaperSession",
    "PaperSessionError",
    "PaperStateError",
    "Quote",
    "QuoteSource",
    "RejectReason",
    "TimeInForce",
    "SessionStep",
    "Side",
    "assert_simulation_only",
    "dict_quote_source",
    "new_session",
    "resolve_mode",
]
