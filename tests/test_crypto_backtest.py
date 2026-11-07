"""The crypto TJR backtest study: the properties that make a result trustworthy.

The frames here are **hand-built**, not random: a test frame's expected outcome is
known from its construction, so a failure points at the harness rather than at the
market. A study over *real* bars is verified by running it (see
``docs/crypto/CRYPTO_STUDY.md``); what a unit test proves is that the harness is
honest:

1. a period the data does not cover is reported short, never padded;
2. the walk-forward's folds are disjoint sessions;
3. costs reduce the result, and a frictionless run is labelled as such;
4. the same bars and the same config produce the same numbers;
5. no decision reads a bar that had not closed;
6. the study's sizing is the *study's* sizing, not the paper account's 25-USD cap;
7. every metric the report promises is present, and an undefined one is ``None``.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from tests.tjr_helpers import BULLISH_BARS, SESSION, one_minute_from_five
from tradingagents.crypto.backtest.study import (
    CryptoBacktestConfig,
    cost_scenarios,
    determinism_check,
    equal_weight_benchmark,
    intervals_for_window,
    metrics_for,
    per_symbol_metrics,
    run_cost_scenarios,
    run_periods,
    run_regimes,
    run_sensitivity,
    run_walk_forward,
    run_window,
    sensitivity_variants,
    sessions_for,
    study_risk_limits,
)
from tradingagents.execution.costs import CostModel
from tradingagents.tjr.frames import IntradaySeries
from tradingagents.tjr.sessions import utc_day_session

# A bullish session extended so the entry's target is reached before the bell.
WIN = BULLISH_BARS[:15] + [
    (100.6, 110.0, 100.55, 109.0),
    (109.0, 110.0, 108.0, 109.5),
]


def _crypto_day_bars(session: str, bars=None):
    """A 5-minute frame that starts at the UTC day's open, not an exchange bell.

    Crypto has no opening bell, so a crypto session is the UTC day. The equity
    helper starts at 09:30 ET, which would place a crypto session's bars in the
    middle of the UTC day; this one starts at 00:00 UTC, so the session the rule
    chain labels is the whole day.
    """
    bars = WIN if bars is None else bars
    index = pd.date_range(
        f"{session}T00:00:00+00:00", periods=len(bars), freq="5min", tz="UTC"
    )
    frame = pd.DataFrame(bars, index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


def _crypto_series(symbol: str = "BTC/USD", bars=None, *, session: str = SESSION):
    """A crypto series bundle under the 24/7 UTC-day policy.

    The bars are the same hand-built session the equity TJR tests use, but the
    series carry the crypto session policy, so the rule chain labels them under
    the UTC day rather than an exchange session.
    """
    policy = utc_day_session()
    five = _crypto_day_bars(session, bars)
    one = one_minute_from_five(five)
    return {
        symbol: {
            "1m": IntradaySeries(symbol, "1m", one, session=policy),
            "5m": IntradaySeries(symbol, "5m", five, session=policy),
        }
    }


def _multi_day_series(symbol: str = "BTC/USD", days: int = 3):
    """Several UTC-day sessions, each the same hand-built bullish session."""
    policy = utc_day_session()
    first = datetime.fromisoformat(SESSION)
    frames = []
    for offset in range(days):
        day = (first + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        frames.append(_crypto_day_bars(day))
    five = pd.concat(frames)
    one = one_minute_from_five(five)
    return {
        symbol: {
            "1m": IntradaySeries(symbol, "1m", one, session=policy),
            "5m": IntradaySeries(symbol, "5m", five, session=policy),
        }
    }


def _config(symbols=("BTC/USD",), **kwargs) -> CryptoBacktestConfig:
    return CryptoBacktestConfig(symbols=tuple(symbols), initial_cash=10_000.0, **kwargs)


@pytest.mark.unit
class TestStudyConfig:
    def test_the_study_sizing_is_not_the_paper_order_cap(self):
        config = _config()
        limits = config.risk_limits
        assert limits is not None
        # The paper account caps an order at 25 USD; the study sizes to the book.
        assert limits.max_order_value is None
        assert limits.max_position_size == pytest.approx(0.10)
        assert limits.max_open_positions == 5
        assert limits.long_only is True
        assert limits.fractional_shares is True

    def test_the_study_limits_keep_the_paper_shape(self):
        limits = study_risk_limits(initial_cash=100_000.0)
        assert limits.max_portfolio_exposure == pytest.approx(0.95)
        assert limits.max_daily_orders is None

    def test_an_explicit_limit_is_kept(self):
        from tradingagents.risk import RiskLimits

        mine = RiskLimits(max_order_value=1_000.0)
        config = _config(risk_limits=mine)
        assert config.risk_limits is mine

    def test_a_config_needs_a_symbol_and_positive_cash(self):
        with pytest.raises(ValueError):
            CryptoBacktestConfig(symbols=(), initial_cash=1_000.0)
        with pytest.raises(ValueError):
            CryptoBacktestConfig(symbols=("BTC/USD",), initial_cash=0.0)

    def test_the_crypto_settings_cut_the_entry_window_at_the_utc_day_end(self):
        config = _config()
        assert config.settings.latest_entry_time == (23, 30)
        assert config.session_policy.label


@pytest.mark.unit
class TestPeriods:
    def test_a_period_reports_the_sessions_it_actually_covered(self):
        series = _multi_day_series(days=3)
        periods = run_periods(series, _config(periods=(1, 2)))
        by_label = {p.label: p for p in periods}
        assert by_label["1d"].sessions == 1
        assert by_label["2d"].sessions == 2
        assert by_label["max"].sessions == 3

    def test_a_period_the_data_does_not_cover_is_short_not_padded(self):
        series = _multi_day_series(days=2)
        periods = run_periods(series, _config(periods=(90,)))
        # The 90-day request is reported as a 90-day request over 2 sessions, and
        # the short label never claims the venue served 90 days.
        assert [p.label for p in periods] == ["90d_requested_2d", "max"]
        assert periods[0].sessions == 2
        assert periods[0].days == 90

    def test_every_period_reports_the_metrics_the_report_promises(self):
        series = _crypto_series()
        periods = run_periods(series, _config(periods=(1,)))
        metrics = periods[0].metrics
        for key in (
            "initial_capital",
            "final_equity",
            "num_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "average_r",
            "average_planned_reward_risk",
            "fees",
            "slippage_cost",
            "exposure",
            "turnover",
            "rejections",
            "benchmark_return",
        ):
            assert key in metrics, key

    def test_a_flat_run_has_no_sharpe_ratio(self):
        # A bearish session produces no trade at all in a long-only book.
        from tests.tjr_helpers import BEARISH_BARS

        series = _crypto_series(bars=BEARISH_BARS)
        periods = run_periods(series, _config(periods=(1,)))
        metrics = periods[0].metrics
        assert metrics["num_trades"] == 0
        # An undefined metric is None, never 0.0.
        assert metrics["sharpe_ratio"] is None
        assert metrics["sortino_ratio"] is None
        assert metrics["win_rate"] is None

    def test_the_period_names_its_intervals(self):
        series = _crypto_series()
        periods = run_periods(series, _config(periods=(1,)))
        assert "5m" in periods[0].intervals

    def test_a_window_the_data_does_not_reach_drops_the_one_minute_trigger(self):
        series = _multi_day_series(days=2)
        first_day = min(s for s in sessions_for(series, utc_day_session()))
        # A window entirely after the data: no session of it is covered at 1m, so
        # the 1m entry trigger cannot be evaluated for it.
        later = (datetime.fromisoformat(first_day) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        intervals = intervals_for_window(series, [later])
        assert "1m" not in intervals
        assert "5m" in intervals

    def test_a_window_the_data_covers_keeps_the_one_minute_trigger(self):
        series = _multi_day_series(days=2)
        sessions = sessions_for(series, utc_day_session())
        assert "1m" in intervals_for_window(series, sessions)


@pytest.mark.unit
class TestWalkForward:
    def test_the_folds_are_disjoint_sessions(self):
        series = _multi_day_series(days=6)
        folds = run_walk_forward(series, _config(walk_forward_folds=3))
        assert len(folds) == 3
        windows = [(f.start, f.end) for f in folds]
        assert len(windows) == len(set(windows))

    def test_the_folds_cover_the_data_without_reusing_a_session(self):
        series = _multi_day_series(days=6)
        folds = run_walk_forward(series, _config(walk_forward_folds=3))
        assert sum(f.sessions for f in folds) == 6

    def test_a_fold_is_a_view_of_the_shared_book(self):
        series = _multi_day_series(days=4)
        folds = run_walk_forward(series, _config(walk_forward_folds=2))
        for fold in folds:
            assert fold.start is not None
            assert fold.end is not None
            assert fold.start <= fold.end


@pytest.mark.unit
class TestRegimes:
    def test_the_regimes_bucket_the_sessions(self):
        series = _multi_day_series(days=4)
        config = _config(regime_trend_sessions=1)
        regimes = run_regimes(series, config)
        assert regimes
        assert sum(r.sessions for r in regimes) <= 4


@pytest.mark.unit
class TestCosts:
    def test_the_scenarios_are_explicit(self):
        scenarios = cost_scenarios(_config())
        assert "frictionless" in scenarios
        assert "realistic" in scenarios
        assert "conservative" in scenarios
        assert scenarios["frictionless"].commission_rate == 0.0
        assert scenarios["realistic"].commission_rate > 0.0
        assert scenarios["conservative"].commission_rate >= scenarios["realistic"].commission_rate

    def test_a_costlier_scenario_earns_less(self):
        series = _crypto_series()
        results = run_cost_scenarios(series, _config())
        by_label = {r.label: r.metrics for r in results}
        assert by_label["conservative"]["final_equity"] <= by_label["realistic"]["final_equity"]
        assert by_label["realistic"]["final_equity"] <= by_label["frictionless"]["final_equity"]

    def test_the_fees_are_reported_and_not_zero_when_a_scenario_charges_them(self):
        series = _crypto_series()
        results = run_cost_scenarios(series, _config())
        by_label = {r.label: r.metrics for r in results}
        realistic = by_label["realistic"]
        assert realistic["num_trades"] >= 1
        assert realistic["fees"] > 0.0
        assert realistic["slippage_cost"] >= 0.0


@pytest.mark.unit
class TestSensitivity:
    def test_the_variants_are_named(self):
        variants = sensitivity_variants(_config())
        assert variants

    def test_the_sensitivity_runs(self):
        series = _crypto_series()
        results = run_sensitivity(series, _config())
        assert len(results) == len(sensitivity_variants(_config()))


@pytest.mark.unit
class TestDeterminism:
    def test_the_same_bars_give_the_same_numbers(self):
        series = _crypto_series()
        config = _config()
        check = determinism_check(series, config)
        assert check["identical"] is True
        assert check["final_equity"][0] == check["final_equity"][1]
        assert check["num_trades"][0] == check["num_trades"][1]

    def test_two_runs_agree_trade_for_trade(self):
        series = _crypto_series()
        config = _config()
        first = run_window(series, config)
        second = run_window(series, config)
        assert len(first.trades) == len(second.trades)
        for a, b in zip(first.trades, second.trades, strict=True):
            assert a.symbol == b.symbol
            assert a.entry_price == pytest.approx(b.entry_price)
            assert a.exit_price == pytest.approx(b.exit_price)


@pytest.mark.unit
class TestNoLookahead:
    def test_truncating_the_feed_does_not_change_the_earlier_trades(self):
        """The executable form of the closed-bar rule.

        The session is truncated after the entry and re-run. The entry must be
        identical: if a decision had read a later bar, the truncated run would
        differ.
        """
        full = run_window(_crypto_series(bars=WIN), _config())
        truncated = run_window(_crypto_series(bars=WIN[:16]), _config())
        full_entry = next(t for t in full.trades)
        truncated_entry = next(t for t in truncated.trades)
        assert full_entry.entry_price == truncated_entry.entry_price
        assert full_entry.entry_time == truncated_entry.entry_time


@pytest.mark.unit
class TestPerSymbolAndBenchmark:
    def test_each_symbol_is_measured_on_its_own(self):
        series = _crypto_series("BTC/USD")
        series["ETH/USD"] = _crypto_series("ETH/USD")["ETH/USD"]
        metrics = per_symbol_metrics(series, _config(symbols=("BTC/USD", "ETH/USD")))
        assert set(metrics) == {"BTC/USD", "ETH/USD"}
        for values in metrics.values():
            assert values["initial_capital"] == 10_000.0

    def test_the_benchmark_is_an_equal_weight_hold(self):
        series = _crypto_series()
        equity_index = series["BTC/USD"]["1m"].frame.index
        benchmark = equal_weight_benchmark(
            series, initial_capital=10_000.0, equity_index=equity_index
        )
        assert benchmark is not None
        assert len(benchmark) == len(equity_index)
        assert benchmark.iloc[0] == pytest.approx(10_000.0)

    def test_the_benchmark_is_none_when_there_is_no_price(self):
        assert (
            equal_weight_benchmark({}, initial_capital=10_000.0, equity_index=pd.Index([]))
            is None
        )


@pytest.mark.unit
class TestAccountancy:
    def test_the_final_equity_is_the_initial_plus_realized_pnl_less_fees(self):
        series = _crypto_series()
        config = _config()
        result = run_window(series, config)
        metrics = metrics_for(result)
        expected = metrics["initial_capital"] + metrics["realized_pnl"] - metrics["fees"]
        assert metrics["final_equity"] == pytest.approx(expected, abs=1e-6)

    def test_the_winning_and_losing_counts_add_up_to_the_trade_count(self):
        series = _crypto_series()
        periods = run_periods(series, _config(periods=(1,)))
        metrics = periods[0].metrics
        assert metrics["winning_trades"] + metrics["losing_trades"] <= metrics["num_trades"]

    def test_the_run_is_flat_at_the_session_close(self):
        result = run_window(_crypto_series(), _config())
        for trade in result.trades:
            assert trade.exit_reason
        assert result.ledger.position_quantity("BTC/USD") == 0


@pytest.mark.unit
class TestCostModelIsExplicit:
    def test_a_cost_model_with_no_rates_charges_nothing(self):
        model = CostModel()
        assert model.commission_rate == 0.0
        assert model.slippage_bps == 0.0
        assert model.half_spread_bps == 0.0
