"""Performance analytics: the metrics, and the cases where they must be undefined.

The tests that matter most here are the degenerate ones. A metric that returns a
large number for a flat curve, or annualises a two-week sample into a yearly
rate, makes a run look good for reasons that have nothing to do with trading. So
each such case is pinned to ``None`` or to an explicit convention.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tradingagents.performance import (
    TRADING_DAYS_PER_YEAR,
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


def _curve(values, *, start="2026-01-05") -> pd.Series:
    index = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=pd.Index([d.strftime("%Y-%m-%d") for d in index], name="session"),
                     dtype=float)


class _Trade:
    """Minimal stand-in for the backtester's Trade: only net_pnl is read."""

    def __init__(self, net_pnl: float) -> None:
        self.net_pnl = net_pnl


# ---------------------------------------------------------------------------
# Return conventions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReturns:
    def test_total_return_is_not_annualised(self):
        """A short run's headline return must not be extrapolated."""
        equity = _curve([100.0, 110.0])
        metrics = compute_metrics(equity, initial_capital=100.0)
        assert metrics.total_return == pytest.approx(0.10)

    def test_annualization_is_explicit_and_compounded(self):
        # 10% over 126 sessions (half a year) -> ~21% annualised
        value = annualize_return(0.10, 126)
        assert value == pytest.approx(0.21, abs=0.01)

    def test_annualization_of_a_full_year_is_the_return_itself(self):
        assert annualize_return(0.10, TRADING_DAYS_PER_YEAR) == pytest.approx(0.10)

    def test_annualization_of_a_total_loss_is_undefined(self):
        assert annualize_return(-1.0, 100) is None

    def test_annualization_of_an_empty_run_is_undefined(self):
        assert annualize_return(0.1, 0) is None

    def test_cagr_uses_calendar_time(self):
        value = cagr(100.0, 121.0, "2026-01-01", "2027-12-31")
        assert value is not None
        assert 0.09 < value < 0.11  # ~2 years, sqrt(1.21) = 1.1

    def test_cagr_without_elapsed_time_is_undefined(self):
        assert cagr(100.0, 110.0, "2026-01-05", "2026-01-05") is None

    def test_cagr_with_a_non_positive_endpoint_is_undefined(self):
        assert cagr(100.0, 0.0, "2026-01-01", "2027-01-01") is None

    def test_period_returns_drop_the_first_observation(self):
        returns = period_returns(_curve([100.0, 110.0, 121.0]))
        assert len(returns) == 2
        assert returns.iloc[0] == pytest.approx(0.10)

    def test_period_returns_of_a_single_point_are_empty(self):
        assert period_returns(_curve([100.0])).empty


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrawdown:
    def test_a_flat_curve_has_no_drawdown(self):
        detail = max_drawdown(_curve([100.0, 100.0, 100.0]))
        assert detail.max_drawdown == 0.0
        assert detail.start is None

    def test_a_simple_decline(self):
        detail = max_drawdown(_curve([100.0, 120.0, 90.0, 95.0]))
        # from 120 to 90 is -25%
        assert detail.max_drawdown == pytest.approx(0.25)

    def test_the_peak_and_trough_sessions_are_reported(self):
        equity = _curve([100.0, 120.0, 90.0, 95.0])
        detail = max_drawdown(equity)
        assert detail.start == equity.index[1]
        assert detail.end == equity.index[2]

    def test_recovery_is_detected(self):
        equity = _curve([100.0, 120.0, 90.0, 125.0])
        detail = max_drawdown(equity)
        assert detail.recovery == equity.index[3]

    def test_no_recovery_is_reported_as_none(self):
        equity = _curve([100.0, 120.0, 90.0, 95.0])
        detail = max_drawdown(equity)
        assert detail.recovery is None

    def test_the_deepest_drawdown_wins_not_the_first(self):
        equity = _curve([100.0, 95.0, 100.0, 80.0])
        assert max_drawdown(equity).max_drawdown == pytest.approx(0.20)

    def test_the_drawdown_series_is_non_negative(self):
        series = drawdown_series(_curve([100.0, 120.0, 90.0, 95.0]))
        assert (series >= 0).all()
        assert series.max() == pytest.approx(0.25)

    def test_an_empty_curve_has_no_drawdown(self):
        assert max_drawdown(pd.Series(dtype=float)).max_drawdown == 0.0


# ---------------------------------------------------------------------------
# Ratios: the undefined cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRatios:
    def test_sharpe_of_a_flat_curve_is_undefined_not_infinite(self):
        """A do-nothing run must not look spectacular."""
        assert sharpe_ratio(period_returns(_curve([100.0] * 20))) is None

    def test_sharpe_needs_at_least_two_returns(self):
        assert sharpe_ratio(period_returns(_curve([100.0, 101.0]))) is None

    def test_sharpe_is_positive_for_a_rising_curve(self):
        equity = _curve([100.0, 101.0, 102.5, 103.0, 105.0, 106.0])
        assert sharpe_ratio(period_returns(equity)) > 0

    def test_sortino_is_undefined_without_downside(self):
        """A curve that never fell has no downside deviation to divide by."""
        equity = _curve([100.0, 101.0, 102.0, 103.0])
        assert sortino_ratio(period_returns(equity)) is None

    def test_sortino_is_defined_with_downside(self):
        equity = _curve([100.0, 101.0, 99.0, 102.0, 98.0, 103.0])
        value = sortino_ratio(period_returns(equity))
        assert value is not None and math.isfinite(value)

    def test_sortino_exceeds_sharpe_when_downside_is_milder_than_upside(self):
        equity = _curve([100.0, 105.0, 104.0, 110.0, 109.0, 115.0])
        returns = period_returns(equity)
        assert sortino_ratio(returns) > sharpe_ratio(returns)

    def test_calmar_is_undefined_without_drawdown(self):
        assert calmar_ratio(0.20, 0.0) is None

    def test_calmar_is_undefined_without_an_annualized_return(self):
        assert calmar_ratio(None, 0.10) is None

    def test_calmar_divides_return_by_drawdown(self):
        assert calmar_ratio(0.20, 0.10) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTradeStatistics:
    def test_win_rate_counts_only_closed_trades(self):
        trades = [_Trade(10.0), _Trade(-5.0), _Trade(20.0)]
        metrics = compute_metrics(_curve([100.0, 125.0]), initial_capital=100.0, trades=trades)
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(2 / 3)

    def test_a_flat_trade_is_neither_a_win_nor_a_loss(self):
        trades = [_Trade(0.0)]
        metrics = compute_metrics(_curve([100.0, 100.0]), initial_capital=100.0, trades=trades)
        assert metrics.num_trades == 1
        assert metrics.win_rate is None

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        trades = [_Trade(30.0), _Trade(-10.0)]
        metrics = compute_metrics(_curve([100.0, 120.0]), initial_capital=100.0, trades=trades)
        assert metrics.profit_factor == pytest.approx(3.0)

    def test_profit_factor_is_undefined_without_losses(self):
        """An infinite profit factor is not a number a report can use."""
        metrics = compute_metrics(_curve([100.0, 120.0]), initial_capital=100.0,
                                  trades=[_Trade(10.0)])
        assert metrics.profit_factor is None

    def test_average_win_and_loss(self):
        trades = [_Trade(30.0), _Trade(10.0), _Trade(-20.0)]
        metrics = compute_metrics(_curve([100.0, 120.0]), initial_capital=100.0, trades=trades)
        assert metrics.average_win == pytest.approx(20.0)
        assert metrics.average_loss == pytest.approx(-20.0)

    def test_expectancy_is_the_mean_net_pnl(self):
        trades = [_Trade(30.0), _Trade(-10.0)]
        metrics = compute_metrics(_curve([100.0, 120.0]), initial_capital=100.0, trades=trades)
        assert metrics.expectancy == pytest.approx(10.0)

    def test_expectancy_is_undefined_without_trades(self):
        metrics = compute_metrics(_curve([100.0, 120.0]), initial_capital=100.0)
        assert metrics.expectancy is None
        assert metrics.win_rate is None

    def test_the_loss_sign_convention_is_negative(self):
        """average_loss is negative, matching the P&L sign, not a magnitude."""
        metrics = compute_metrics(_curve([100.0, 90.0]), initial_capital=100.0,
                                  trades=[_Trade(-10.0)])
        assert metrics.average_loss == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Exposure, turnover, volatility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExposureAndTurnover:
    def test_mean_exposure_is_averaged_over_the_curve(self):
        exposure = pd.Series([0.0, 0.5, 1.0], dtype=float)
        metrics = compute_metrics(_curve([100.0, 101.0, 102.0]), initial_capital=100.0,
                                  exposure=exposure)
        assert metrics.exposure == pytest.approx(0.5)

    def test_exposure_is_zero_without_a_series(self):
        metrics = compute_metrics(_curve([100.0, 101.0]), initial_capital=100.0)
        assert metrics.exposure == 0.0

    def test_turnover_scales_by_mean_equity(self):
        metrics = compute_metrics(_curve([100.0, 100.0]), initial_capital=100.0,
                                  traded_notional=300.0)
        assert metrics.turnover == pytest.approx(3.0)

    def test_volatility_is_annualised(self):
        equity = _curve([100.0, 101.0, 100.0, 102.0, 101.0, 103.0])
        metrics = compute_metrics(equity, initial_capital=100.0)
        expected = period_returns(equity).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert metrics.volatility == pytest.approx(expected)

    def test_volatility_of_a_flat_curve_is_zero_not_undefined(self):
        """Zero volatility is a fact about the curve; a Sharpe on it is undefined."""
        metrics = compute_metrics(_curve([100.0] * 5), initial_capital=100.0)
        assert metrics.volatility == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmark:
    def test_buy_and_hold_scales_to_the_initial_capital(self):
        prices = pd.Series([100.0, 110.0], index=["2026-01-05", "2026-01-06"])
        curve = buy_and_hold_equity(prices, initial_capital=10_000.0)
        assert curve.iloc[0] == pytest.approx(10_000.0)
        assert curve.iloc[-1] == pytest.approx(11_000.0)

    def test_benchmark_return_and_excess_are_reported(self):
        equity = _curve([100.0, 105.0])  # +5%
        benchmark = pd.Series([100.0, 110.0], index=equity.index)  # +10%
        metrics = compute_metrics(equity, initial_capital=100.0, benchmark=benchmark)
        assert metrics.benchmark_return == pytest.approx(0.10)
        assert metrics.excess_return == pytest.approx(-0.05)

    def test_no_benchmark_means_no_benchmark_metrics(self):
        metrics = compute_metrics(_curve([100.0, 105.0]), initial_capital=100.0)
        assert metrics.benchmark_return is None
        assert metrics.excess_return is None

    def test_a_single_point_benchmark_is_not_usable(self):
        equity = _curve([100.0, 105.0])
        benchmark = pd.Series([100.0], index=equity.index[:1])
        metrics = compute_metrics(equity, initial_capital=100.0, benchmark=benchmark)
        assert metrics.benchmark_return is None


# ---------------------------------------------------------------------------
# Whole-run integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeMetrics:
    def test_a_flat_run_reports_zero_return_and_no_ratios(self):
        metrics = compute_metrics(_curve([100.0] * 10), initial_capital=100.0)
        assert metrics.total_return == pytest.approx(0.0)
        assert metrics.sharpe_ratio is None
        assert metrics.sortino_ratio is None
        assert metrics.calmar_ratio is None

    def test_an_empty_curve_does_not_raise(self):
        metrics = compute_metrics(pd.Series(dtype=float), initial_capital=1_000.0)
        assert metrics.sessions == 0
        assert metrics.final_equity == 1_000.0

    def test_the_session_count_and_span_are_reported(self):
        metrics = compute_metrics(_curve([100.0, 101.0, 102.0]), initial_capital=100.0)
        assert metrics.sessions == 3
        assert metrics.start is not None and metrics.end is not None

    def test_the_dict_is_json_safe(self):
        import json

        metrics = compute_metrics(_curve([100.0, 101.0, 102.0]), initial_capital=100.0)
        payload = json.dumps(metrics.to_dict())
        assert "total_return" in payload

    def test_a_losing_run_reports_a_negative_return(self):
        metrics = compute_metrics(_curve([100.0, 90.0]), initial_capital=100.0)
        assert metrics.total_return == pytest.approx(-0.10)
        assert metrics.max_drawdown == pytest.approx(0.10)

    def test_the_same_curve_gives_the_same_metrics(self):
        equity = _curve([100.0, 102.0, 101.0, 104.0, 103.0])
        first = compute_metrics(equity, initial_capital=100.0).to_dict()
        for _ in range(10):
            assert compute_metrics(equity, initial_capital=100.0).to_dict() == first

    def test_format_metrics_renders_undefined_values_as_na(self):
        text = format_metrics(compute_metrics(_curve([100.0] * 5), initial_capital=100.0))
        assert "n/a" in text
        assert "Sharpe" in text

    def test_format_metrics_includes_the_benchmark_when_present(self):
        equity = _curve([100.0, 105.0])
        benchmark = pd.Series([100.0, 110.0], index=equity.index)
        text = format_metrics(compute_metrics(equity, initial_capital=100.0, benchmark=benchmark))
        assert "benchmark return" in text


@pytest.mark.unit
class TestAnnualizationDays:
    def test_the_default_is_us_equity_sessions(self):
        assert TRADING_DAYS_PER_YEAR == 252

    def test_a_different_session_count_changes_the_annualized_return(self):
        equity = _curve([100.0, 110.0])
        us = compute_metrics(equity, initial_capital=100.0, periods_per_year=252)
        crypto = compute_metrics(equity, initial_capital=100.0, periods_per_year=365)
        assert us.annualized_return != crypto.annualized_return

    def test_the_convention_is_reported(self):
        metrics = compute_metrics(_curve([100.0, 110.0]), initial_capital=100.0,
                                  periods_per_year=365)
        assert metrics.annualization_days == 365


@pytest.mark.unit
def test_metrics_dataclass_is_frozen():
    metrics = PerformanceMetrics(
        initial_capital=1.0, final_equity=1.0, total_return=0.0, annualized_return=None,
        cagr=None, realized_pnl=0.0, unrealized_pnl=0.0, total_fees=0.0, num_trades=0,
        winning_trades=0, losing_trades=0, win_rate=None, profit_factor=None, average_win=None,
        average_loss=None, expectancy=None, max_drawdown=0.0, max_drawdown_start=None,
        max_drawdown_end=None, sharpe_ratio=None, sortino_ratio=None, calmar_ratio=None,
        exposure=0.0, turnover=0.0, volatility=None,
    )
    with pytest.raises(FrozenInstanceError):
        metrics.total_return = 0.5


# ---------------------------------------------------------------------------
# Integration with the backtester
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromBacktestResult:
    """The analytics read the ledger's own totals, so the report cannot disagree."""

    @staticmethod
    def _result(*, costs=None, holding_period_days=2):
        from datetime import UTC, datetime

        from tradingagents.backtesting import (
            BacktestConfig,
            MarketDataFeed,
            PortfolioBacktester,
        )
        from tradingagents.execution import CostModel
        from tradingagents.risk import RiskLimits
        from tradingagents.signals import SignalAction, TradingSignal

        dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        closes = [100.0, 101.0, 103.0, 104.0, 106.0]
        frame = pd.DataFrame(
            {"open": closes, "high": [c * 1.005 for c in closes],
             "low": [c * 0.995 for c in closes], "close": closes,
             "volume": [1_000_000] * len(dates)},
            index=pd.to_datetime(dates),
        )
        feed = MarketDataFeed({"AAPL": frame})

        def strategy(context):
            if context.session != "2026-01-05":
                return None
            return TradingSignal(
                symbol="AAPL", action=SignalAction.BUY, confidence=0.8,
                timestamp=datetime.strptime("2026-01-05", "%Y-%m-%d").replace(hour=21, tzinfo=UTC),
                as_of_date="2026-01-05", position_size=0.5,
            )

        config = BacktestConfig(
            initial_cash=100_000.0,
            holding_period_days=holding_period_days,
            costs=costs or CostModel(),
            risk_limits=RiskLimits(max_position_size=0.5, max_portfolio_exposure=None),
        )
        engine = PortfolioBacktester(feed, strategy, config=config)
        return engine.run(), engine

    def test_metrics_are_computed_from_a_real_run(self):
        from tradingagents.performance.report import metrics_from_result

        result, _ = self._result()
        metrics = metrics_from_result(result)
        assert metrics.sessions == 5
        assert metrics.initial_capital == 100_000.0
        assert metrics.final_equity == pytest.approx(result.final_equity)
        # The price rose from 101 (entry) to 106 (exit), so the run made money.
        assert metrics.total_return > 0
        assert metrics.num_trades == 1

    def test_the_reported_fees_are_the_ledgers_fees(self):
        from tradingagents.execution import CostModel
        from tradingagents.performance.report import metrics_from_result

        result, engine = self._result(costs=CostModel(commission_rate=0.001))
        metrics = metrics_from_result(result)
        assert metrics.total_fees == pytest.approx(engine.ledger.total_fees)
        assert metrics.total_fees > 0

    def test_turnover_is_computed_from_fill_events(self):
        from tradingagents.execution import CostModel
        from tradingagents.performance.report import metrics_from_result

        result, _ = self._result(costs=CostModel(commission_rate=0.001))
        metrics = metrics_from_result(result)
        # A buy and a sell, so two fills and a positive turnover.
        assert metrics.turnover > 0

    def test_a_benchmark_can_be_supplied(self):
        from tradingagents.performance.report import metrics_from_result

        result, _ = self._result()
        benchmark = pd.Series([100.0, 101.0, 103.0, 104.0, 106.0], index=result.equity_series().index)
        metrics = metrics_from_result(result, benchmark_prices=benchmark)
        assert metrics.benchmark_return is not None
        assert metrics.excess_return is not None

    def test_the_metrics_are_json_serializable(self):
        import json

        from tradingagents.performance.report import metrics_from_result

        result, _ = self._result()
        payload = json.dumps(metrics_from_result(result).to_dict())
        assert "final_equity" in payload

    def test_the_run_is_deterministic(self):
        from tradingagents.performance.report import metrics_from_result

        first = metrics_from_result(self._result()[0]).to_dict()
        second = metrics_from_result(self._result()[0]).to_dict()
        assert first == second
