"""The outcome distribution: a resample of a run, and the limits of that resample.

These tests are written around two questions a reader will ask. First, is the
number reproducible — same seed, same answer? Second, does the number mean what it
says — is a sample count kept apart from an iteration count, is an empty run
reported as unavailable rather than zero, and are in-sample and out-of-sample
results impossible to merge?

The bootstrap is a statement about one sample, so the tests pin its arithmetic
directly: the interval must come from the resampled distribution, the point
estimates must be the sample's own, and a wider block must not be reported as
though the observations were independent.
"""

from __future__ import annotations

import pytest

from tradingagents.backtesting.distribution import (
    DEFAULT_ITERATIONS,
    METHOD_SESSION_BLOCK,
    METHOD_TRADE_IID,
    DistributionError,
    bootstrap_mean_ci,
    effective_sample_size,
    in_sample_vs_out_of_sample,
    outcome_distribution,
    session_returns,
    trade_net_pnl,
)
from tradingagents.backtesting.engine import BacktestResult, EquityPoint, Trade

# ---------------------------------------------------------------------------
# Fixtures: a BacktestResult built by hand, so the arithmetic is pinned exactly.
# ---------------------------------------------------------------------------


def _equity_result(equities, *, start_cash=100_000.0, sessions=None) -> BacktestResult:
    sessions = sessions or [f"2026-01-{day:02d}" for day in range(1, len(equities) + 1)]
    points = [
        EquityPoint(session=s, equity=e, cash=start_cash, market_value=e - start_cash, exposure=0.0)
        for s, e in zip(sessions, equities, strict=True)
    ]
    return BacktestResult(run_id="test", sessions=list(sessions), equity_curve=points)


def _trade(pnl, *, quantity=10.0, entry=100.0, reason="holding_period") -> Trade:
    """A round trip whose net P&L is exactly ``pnl`` (fees folded into the exit)."""
    exit_price = entry + pnl / quantity
    return Trade(
        symbol="AAPL", entry_session="2026-01-05", exit_session="2026-01-09",
        quantity=quantity, entry_price=entry, exit_price=exit_price, fees=0.0, reason=reason,
    )


def _trade_result(pnls) -> BacktestResult:
    return BacktestResult(run_id="test", trades=[_trade(p) for p in pnls])


# ---------------------------------------------------------------------------
# Numeric guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNumericGuards:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_sample_is_refused(self, bad):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, bad, 2.0], iterations=10, seed=1)

    def test_a_non_numeric_sample_is_refused(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, "2.0"], iterations=10, seed=1)

    def test_a_bool_is_not_accepted_as_a_number(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([True, 1.0], iterations=10, seed=1)

    def test_an_empty_sample_is_refused(self):
        with pytest.raises(DistributionError, match="empty"):
            bootstrap_mean_ci([], iterations=10, seed=1)

    def test_a_non_positive_iteration_count_is_refused(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, 2.0], iterations=0, seed=1)

    def test_a_confidence_outside_zero_and_one_is_refused(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, 2.0], iterations=10, seed=1, confidence=1.0)

    def test_a_non_positive_block_size_is_refused(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, 2.0], iterations=10, seed=1, block_size=0)

    def test_a_non_integer_seed_is_refused(self):
        with pytest.raises(DistributionError):
            bootstrap_mean_ci([1.0, 2.0], iterations=10, seed="x")

    def test_a_non_positive_equity_point_is_refused_when_forming_returns(self):
        result = _equity_result([100.0, 0.0, 50.0])
        with pytest.raises(DistributionError, match="not positive"):
            session_returns(result)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_the_same_seed_gives_an_identical_interval(self):
        samples = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, -0.005]
        first = bootstrap_mean_ci(samples, iterations=500, seed=42)
        second = bootstrap_mean_ci(samples, iterations=500, seed=42)
        assert first == second

    def test_a_different_seed_can_give_a_different_resample(self):
        samples = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, -0.005]
        a = bootstrap_mean_ci(samples, iterations=500, seed=1)
        b = bootstrap_mean_ci(samples, iterations=500, seed=2)
        # The sample statistics are identical; only the resampled interval moves.
        assert a.sample_mean == b.sample_mean
        assert a.sample_median == b.sample_median
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_the_whole_distribution_is_deterministic_for_a_seed(self):
        result = _equity_result([100_000, 101_000, 100_500, 102_000, 101_500, 103_000])
        first = outcome_distribution(result, iterations=300, seed=7)
        second = outcome_distribution(result, iterations=300, seed=7)
        assert first == second

    def test_the_distribution_has_no_seed_by_default_and_still_runs(self):
        result = _equity_result([100_000, 101_000, 100_500, 102_000])
        outcome = outcome_distribution(result, iterations=100)
        assert outcome.available
        assert outcome.seed is None


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInternalConsistency:
    def test_point_estimates_are_the_samples_own(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        boot = bootstrap_mean_ci(samples, iterations=200, seed=3)
        assert boot.sample_mean == pytest.approx(3.0)
        assert boot.sample_median == pytest.approx(3.0)
        assert boot.sample_count == 5

    def test_the_interval_brackets_the_point_estimate(self):
        samples = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, -0.005, 0.015]
        boot = bootstrap_mean_ci(samples, iterations=2_000, seed=11)
        assert boot.ci_low <= boot.sample_mean <= boot.ci_high

    def test_the_interval_low_never_exceeds_its_high(self):
        samples = [-0.05, 0.02, 0.01, -0.01, 0.03]
        boot = bootstrap_mean_ci(samples, iterations=1_000, seed=5)
        assert boot.ci_low <= boot.ci_high

    def test_the_bootstrap_mean_tracks_the_sample_mean(self):
        samples = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, -0.005, 0.015, 0.007]
        boot = bootstrap_mean_ci(samples, iterations=5_000, seed=13)
        assert boot.bootstrap_mean == pytest.approx(boot.sample_mean, abs=0.005)

    def test_the_probability_of_a_positive_outcome_is_a_share(self):
        boot = bootstrap_mean_ci([0.01, 0.02, 0.03, 0.04], iterations=500, seed=9)
        assert 0.0 <= boot.probability_positive <= 1.0
        # Every resample mean of strictly positive numbers is positive.
        assert boot.probability_positive == pytest.approx(1.0)

    def test_the_sample_count_is_not_the_iteration_count(self):
        """The two numbers are reported apart and must never be conflated."""
        boot = bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=10_000, seed=1)
        assert boot.sample_count == 3
        assert boot.iterations == 10_000
        assert boot.sample_count != boot.iterations

    def test_the_block_size_never_exceeds_the_sample(self):
        boot = bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=100, seed=1, block_size=99)
        assert boot.block_size == 3

    def test_a_single_observation_has_a_degenerate_interval(self):
        boot = bootstrap_mean_ci([0.02], iterations=100, seed=1)
        assert boot.sample_count == 1
        assert boot.ci_low == pytest.approx(0.02)
        assert boot.ci_high == pytest.approx(0.02)
        assert boot.sample_stdev == 0.0


# ---------------------------------------------------------------------------
# Reading a result
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadingAResult:
    def test_session_returns_are_simple_returns_in_order(self):
        result = _equity_result([100.0, 110.0, 99.0])
        assert session_returns(result) == pytest.approx([0.10, -0.10])

    def test_a_single_point_has_no_returns(self):
        assert session_returns(_equity_result([100.0])) == []

    def test_trade_net_pnl_reads_the_ledgers_own_figure(self):
        result = _trade_result([10.0, -5.0, 2.5])
        assert trade_net_pnl(result) == pytest.approx([10.0, -5.0, 2.5])

    def test_fees_are_not_applied_a_second_time(self):
        """``Trade.net_pnl`` is already net; the reader must not subtract again."""
        trade = Trade(
            symbol="AAPL", entry_session="2026-01-05", exit_session="2026-01-09",
            quantity=10.0, entry_price=100.0, exit_price=110.0, fees=5.0, reason="take_profit",
        )
        result = BacktestResult(run_id="t", trades=[trade])
        # gross 100, fees 5 -> net 95, which is what the reader must return.
        assert trade.net_pnl == pytest.approx(95.0)
        assert trade_net_pnl(result) == pytest.approx([95.0])


# ---------------------------------------------------------------------------
# The outcome distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutcomeDistribution:
    def test_an_empty_run_is_unavailable_not_zero(self):
        outcome = outcome_distribution(BacktestResult(run_id="empty"), iterations=100, seed=1)
        assert outcome.available is False
        assert outcome.sample_mean is None
        assert outcome.ci_low is None
        assert outcome.ci_high is None
        assert outcome.probability_positive is None
        assert outcome.sample_count == 0
        assert any("unavailable" in w for w in outcome.warnings)

    def test_a_single_equity_point_is_unavailable(self):
        """One point has no return, so there is nothing to resample."""
        outcome = outcome_distribution(_equity_result([100_000.0]), iterations=100, seed=1)
        assert outcome.available is False

    def test_one_trade_is_available_and_flagged(self):
        outcome = outcome_distribution(
            _trade_result([42.0]), method=METHOD_TRADE_IID, iterations=200, seed=1
        )
        assert outcome.available
        assert outcome.sample_count == 1
        assert outcome.sample_mean == pytest.approx(42.0)
        assert any("only 1 observations" in w for w in outcome.warnings)

    def test_a_small_sample_carries_a_warning(self):
        outcome = outcome_distribution(
            _trade_result([10.0, -5.0, 3.0]), method=METHOD_TRADE_IID, iterations=200, seed=1
        )
        assert any("only 3 observations" in w for w in outcome.warnings)

    def test_a_large_sample_does_not_carry_the_small_sample_warning(self):
        pnls = [float(i % 7) - 3.0 for i in range(30)]
        outcome = outcome_distribution(
            _trade_result(pnls), method=METHOD_TRADE_IID, iterations=200, seed=1
        )
        assert not any("observations: the interval is wide" in w for w in outcome.warnings)

    def test_all_positive_returns_have_a_positive_interval(self):
        result = _equity_result([100_000, 101_000, 102_000, 103_000, 104_000])
        outcome = outcome_distribution(result, iterations=500, seed=1)
        assert outcome.available
        assert outcome.ci_low > 0.0
        assert outcome.probability_positive == pytest.approx(1.0)

    def test_all_negative_returns_have_a_negative_interval(self):
        result = _equity_result([100_000, 99_000, 98_000, 97_000, 96_000])
        outcome = outcome_distribution(result, iterations=500, seed=1)
        assert outcome.available
        assert outcome.ci_high < 0.0
        assert outcome.probability_positive == pytest.approx(0.0)

    def test_mixed_returns_span_zero(self):
        result = _equity_result([100_000, 102_000, 99_000, 103_000, 100_000, 104_000])
        outcome = outcome_distribution(result, iterations=2_000, seed=1)
        assert outcome.available
        assert outcome.ci_low < 0.0 < outcome.ci_high

    def test_the_method_names_what_is_measured(self):
        block = outcome_distribution(
            _equity_result([100_000, 101_000, 100_500, 102_000]), iterations=100, seed=1
        )
        assert block.method == METHOD_SESSION_BLOCK
        assert block.block_size is not None and block.block_size >= 1
        trade = outcome_distribution(
            _trade_result([1.0, 2.0, 3.0]), method=METHOD_TRADE_IID, iterations=100, seed=1
        )
        assert trade.method == METHOD_TRADE_IID
        assert trade.block_size == 1

    def test_the_trade_method_always_warns_about_dependence(self):
        outcome = outcome_distribution(
            _trade_result([float(i) for i in range(40)]),
            method=METHOD_TRADE_IID, iterations=100, seed=1,
        )
        assert any("not" in w and "independent" in w for w in outcome.warnings)

    def test_the_session_method_does_not_claim_independence(self):
        outcome = outcome_distribution(
            _equity_result([100_000 + i * 100 for i in range(10)]), iterations=100, seed=1
        )
        assert not any("as if independent" in w for w in outcome.warnings)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(DistributionError):
            outcome_distribution(_trade_result([1.0]), method="magic", iterations=10, seed=1)

    def test_the_dict_form_is_json_safe(self):
        import json

        outcome = outcome_distribution(
            _equity_result([100_000, 101_000, 100_500, 102_000, 101_000]), iterations=200, seed=1
        )
        payload = json.dumps(outcome.to_dict())
        assert "sample_count" in payload
        assert "iterations" in payload


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEffectiveSampleSize:
    def test_an_uncorrelated_series_keeps_its_length(self):
        # Alternating signs: lag-1 autocorrelation is negative, so no shrinkage.
        series = [1.0 if i % 2 == 0 else -1.0 for i in range(30)]
        assert effective_sample_size(series) == pytest.approx(30.0)

    def test_a_persistent_series_shrinks(self):
        """A slow, one-directional drift carries less independent information."""
        series = [float(i) for i in range(40)]
        ess = effective_sample_size(series)
        assert ess is not None
        assert ess < len(series)

    def test_too_few_points_is_unavailable(self):
        assert effective_sample_size([1.0, 2.0]) is None

    def test_a_constant_series_is_not_penalised_to_zero(self):
        ess = effective_sample_size([5.0, 5.0, 5.0, 5.0])
        assert ess is not None
        assert ess >= 1.0

    def test_the_estimate_never_exceeds_the_sample(self):
        series = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        ess = effective_sample_size(series)
        assert ess is not None
        assert 1.0 <= ess <= len(series)


# ---------------------------------------------------------------------------
# Block behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBlockBootstrap:
    def test_a_block_of_one_is_the_iid_bootstrap(self):
        samples = [0.01, 0.02, 0.03, 0.04, 0.05]
        a = bootstrap_mean_ci(samples, iterations=500, seed=1, block_size=1)
        b = bootstrap_mean_ci(samples, iterations=500, seed=1, block_size=1)
        assert a == b
        assert a.block_size == 1

    def test_a_larger_block_changes_the_resample(self):
        samples = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, -0.005]
        iid = bootstrap_mean_ci(samples, iterations=2_000, seed=1, block_size=1)
        blocked = bootstrap_mean_ci(samples, iterations=2_000, seed=1, block_size=4)
        assert blocked.block_size == 4
        assert (iid.ci_low, iid.ci_high) != (blocked.ci_low, blocked.ci_high)

    def test_block_resampling_preserves_consecutive_runs(self):
        """The default session method keeps a run of sessions together."""
        # A run of large gains followed by a run of losses: a block resample must
        # be able to return a whole run, which an i.i.d. draw would smear out.
        equities = [100_000, 110_000, 121_000, 133_100, 120_000, 108_000, 97_200]
        outcome = outcome_distribution(
            _equity_result(equities), method=METHOD_SESSION_BLOCK, iterations=2_000, seed=1,
        )
        assert outcome.available
        assert outcome.block_size >= 2
        assert outcome.ci_low <= outcome.sample_mean <= outcome.ci_high


# ---------------------------------------------------------------------------
# In-sample vs out-of-sample separation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInOutOfSampleSeparation:
    def test_the_two_windows_are_reported_separately(self):
        in_sample = _equity_result([100_000, 101_000, 102_000, 103_000])
        out_of_sample = _equity_result([100_000, 99_000, 98_000, 97_000])
        comparison = in_sample_vs_out_of_sample(
            in_sample, out_of_sample, iterations=200, seed=1
        )
        assert comparison.in_sample.sample_mean > 0
        assert comparison.out_of_sample.sample_mean < 0

    def test_the_comparison_has_no_pooled_field(self):
        """There is deliberately no combined statistic to read by accident."""
        comparison = in_sample_vs_out_of_sample(
            _equity_result([100_000, 101_000]), _equity_result([100_000, 99_000]),
            iterations=100, seed=1,
        )
        payload = comparison.to_dict()
        assert set(payload) == {"in_sample", "out_of_sample", "note"}
        assert "mean" not in payload

    def test_each_side_keeps_its_own_sample_count(self):
        in_sample = _equity_result([100_000, 101_000, 102_000, 103_000])
        out_of_sample = _equity_result([100_000, 99_000])
        comparison = in_sample_vs_out_of_sample(
            in_sample, out_of_sample, iterations=100, seed=1
        )
        assert comparison.in_sample.sample_count == 3
        assert comparison.out_of_sample.sample_count == 1
        assert comparison.in_sample.sample_count != comparison.out_of_sample.sample_count

    def test_an_empty_side_is_unavailable_not_dropped(self):
        comparison = in_sample_vs_out_of_sample(
            BacktestResult(run_id="empty"), _equity_result([100_000, 99_000]),
            iterations=100, seed=1,
        )
        assert comparison.in_sample.available is False
        assert comparison.out_of_sample.available is True


# ---------------------------------------------------------------------------
# Integration: a real backtest result feeds the distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgainstARealRun:
    def test_a_real_backtest_result_is_consumable(self):
        import pandas as pd

        from tradingagents.backtesting import (
            BacktestConfig,
            MarketDataFeed,
            PortfolioBacktester,
        )
        from tradingagents.risk import RiskLimits
        from tradingagents.signals import SignalAction, TradingSignal

        dates = [f"2026-01-{d:02d}" for d in range(5, 16)]
        closes = [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0]
        frame = pd.DataFrame(
            {"open": closes, "high": [c * 1.01 for c in closes],
             "low": [c * 0.99 for c in closes], "close": closes,
             "volume": [1_000_000] * len(dates)},
            index=pd.to_datetime(dates),
        )
        feed = MarketDataFeed({"AAPL": frame})

        def strategy(context):
            if context.session == dates[0]:
                return TradingSignal(
                    symbol="AAPL", action=SignalAction.BUY, confidence=0.8,
                    timestamp=context.clock.at,
                    as_of_date=context.session,
                )
            return None

        config = BacktestConfig(
            holding_period_days=3,
            risk_limits=RiskLimits(max_position_size=0.5, max_portfolio_exposure=None),
        )
        result = PortfolioBacktester(feed, strategy, config=config).run()

        outcome = outcome_distribution(result, iterations=500, seed=1)
        assert outcome.available
        assert outcome.sample_count == len(result.equity_curve) - 1
        assert outcome.ci_low <= outcome.sample_mean <= outcome.ci_high
        # The default iteration count is documented and distinct from the sample.
        assert outcome.iterations == 500
        assert outcome.sample_count != outcome.iterations

    def test_the_default_iteration_count_is_documented(self):
        assert DEFAULT_ITERATIONS == 2_000

    def test_the_module_never_imports_a_broker_or_network(self):
        """A guard on the module's own boundaries: it must stay analytics only."""
        import tradingagents.backtesting.distribution as module

        source = module.__doc__ or ""
        assert "not a second portfolio simulator" in source
        # No execution, broker or network symbols are reachable from the module.
        names = set(dir(module))
        for forbidden in ("ExecutionSimulator", "Order", "CostModel", "requests", "yfinance"):
            assert forbidden not in names
