"""Performance analytics: what a run produced, measured with defined conventions.

``metrics`` holds :func:`compute_metrics` and the individual estimators. Every
metric that can be undefined for a degenerate run returns ``None`` rather than a
flattering number.
"""

from tradingagents.performance.metrics import (
    TRADING_DAYS_PER_YEAR,
    DrawdownDetail,
    PerformanceMetrics,
    annualize_return,
    buy_and_hold_equity,
    cagr,
    calmar_ratio,
    compute_metrics,
    drawdown_series,
    format_metrics,
    max_drawdown,
    period_returns,
    sharpe_ratio,
    sortino_ratio,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "DrawdownDetail",
    "PerformanceMetrics",
    "annualize_return",
    "buy_and_hold_equity",
    "cagr",
    "calmar_ratio",
    "compute_metrics",
    "drawdown_series",
    "format_metrics",
    "max_drawdown",
    "period_returns",
    "sharpe_ratio",
    "sortino_ratio",
]
