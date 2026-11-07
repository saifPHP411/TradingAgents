"""Crypto TJR backtesting: real historical bars, many periods, honest metrics.

The package has two parts:

- :mod:`.history` fetches and caches real Binance bars, and is explicit that
  historical data and the live tape are different things;
- :mod:`.study` runs the **unchanged** TJR rule chain over those bars across
  periods, symbols, walk-forward folds, market regimes, cost scenarios and
  sensitivity variants, reusing the project's own ledger, risk engine and metrics.

Nothing here changes a rule or a risk limit; a study measures the strategy the
worker would actually run, and reports the losses as readily as the wins.
"""

from __future__ import annotations

from .history import (
    DEFAULT_CACHE_DIR,
    Coverage,
    HistoryCache,
    fetch_history,
)
from .study import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    REGIMES,
    STANDARD_PERIODS,
    CryptoBacktestConfig,
    CryptoStudyResult,
    PeriodResult,
    build_series,
    cost_scenarios,
    determinism_check,
    equal_weight_benchmark,
    metrics_for,
    per_symbol_metrics,
    run_cost_scenarios,
    run_periods,
    run_regimes,
    run_sensitivity,
    run_study,
    run_walk_forward,
    run_window,
    series_from_minutes,
    sessions_for,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "REGIMES",
    "REGIME_BEAR",
    "REGIME_BULL",
    "REGIME_SIDEWAYS",
    "STANDARD_PERIODS",
    "Coverage",
    "CryptoBacktestConfig",
    "CryptoStudyResult",
    "HistoryCache",
    "PeriodResult",
    "build_series",
    "cost_scenarios",
    "determinism_check",
    "equal_weight_benchmark",
    "fetch_history",
    "metrics_for",
    "per_symbol_metrics",
    "run_cost_scenarios",
    "run_periods",
    "run_regimes",
    "run_sensitivity",
    "run_study",
    "run_walk_forward",
    "run_window",
    "series_from_minutes",
    "sessions_for",
]
