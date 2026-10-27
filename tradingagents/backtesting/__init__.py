"""The portfolio backtester: an event loop, not a rating score.

``data`` holds the point-in-time feed that makes look-ahead impossible by
construction. ``engine`` holds the loop: decision -> risk -> order -> execution
-> ledger -> PnL.
"""

from tradingagents.backtesting.data import (
    Bar,
    BarDataError,
    MarketDataFeed,
    SessionClock,
    as_date,
    trading_days_between,
)
from tradingagents.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    EquityPoint,
    OpenPosition,
    PortfolioBacktester,
    Strategy,
    StrategyContext,
    Trade,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Bar",
    "BarDataError",
    "EquityPoint",
    "MarketDataFeed",
    "OpenPosition",
    "PortfolioBacktester",
    "SessionClock",
    "Strategy",
    "StrategyContext",
    "Trade",
    "as_date",
    "trading_days_between",
]
