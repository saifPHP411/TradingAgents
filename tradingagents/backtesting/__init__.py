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
from tradingagents.backtesting.distribution import (
    METHOD_SESSION_BLOCK,
    METHOD_TRADE_IID,
    BootstrapResult,
    DistributionError,
    InOutComparison,
    OutcomeDistribution,
    bootstrap_mean_ci,
    effective_sample_size,
    in_sample_vs_out_of_sample,
    outcome_distribution,
    session_returns,
    trade_net_pnl,
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
from tradingagents.backtesting.walkforward import (
    Fold,
    WalkForwardResult,
    Window,
    WindowError,
    assert_no_leakage,
    build_folds,
    run_walk_forward,
    select_best,
    slice_feed,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Bar",
    "BarDataError",
    "BootstrapResult",
    "DistributionError",
    "EquityPoint",
    "Fold",
    "InOutComparison",
    "METHOD_SESSION_BLOCK",
    "METHOD_TRADE_IID",
    "MarketDataFeed",
    "OpenPosition",
    "OutcomeDistribution",
    "PortfolioBacktester",
    "SessionClock",
    "Strategy",
    "StrategyContext",
    "Trade",
    "WalkForwardResult",
    "Window",
    "WindowError",
    "as_date",
    "assert_no_leakage",
    "bootstrap_mean_ci",
    "build_folds",
    "effective_sample_size",
    "in_sample_vs_out_of_sample",
    "outcome_distribution",
    "run_walk_forward",
    "select_best",
    "session_returns",
    "slice_feed",
    "trade_net_pnl",
    "trading_days_between",
]
