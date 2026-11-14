"""Turn a backtest result into metrics, without the backtester importing analytics.

The dependency runs one way: ``performance`` knows about ``backtesting``, and
``backtesting`` knows nothing about ``performance``. That keeps the loop free of
reporting concerns and lets the analytics be tested against a hand-built curve
without running an engine.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.performance.metrics import (
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    buy_and_hold_equity,
    compute_metrics,
)


def metrics_from_result(
    result,
    *,
    risk_free_rate: float = 0.0,
    benchmark_prices: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    """Metrics for a :class:`~tradingagents.backtesting.engine.BacktestResult`.

    Reads the equity curve, the closed trades and the ledger's own totals, so the
    reported fees and P&L are the ledger's figures rather than a second
    computation that could disagree with them.
    """
    equity = result.equity_series()
    exposure = pd.Series(
        [point.exposure for point in result.equity_curve],
        index=pd.Index([point.session for point in result.equity_curve], name="session"),
        dtype=float,
    )
    ledger = result.ledger
    traded_notional = 0.0
    if ledger is not None:
        traded_notional = sum(
            abs(float(event.detail.get("quantity", 0.0)) * float(event.detail.get("price", 0.0)))
            for event in ledger.events
            if event.type.value == "ORDER_FILLED"
        )

    benchmark = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark = buy_and_hold_equity(
            benchmark_prices.reindex(equity.index).ffill().dropna(),
            initial_capital=result.initial_cash,
        )

    return compute_metrics(
        equity,
        initial_capital=result.initial_cash,
        trades=result.trades,
        exposure=exposure,
        traded_notional=traded_notional,
        realized_pnl=ledger.realized_pnl if ledger is not None else 0.0,
        unrealized_pnl=ledger.unrealized_pnl if ledger is not None else 0.0,
        total_fees=ledger.total_fees if ledger is not None else 0.0,
        risk_free_rate=risk_free_rate,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
    )
