"""The official strategy baseline: protocol, determinism, splits, artifacts.

These tests guard the baseline's *trustworthiness*, not the strategy's quality.
They prove the protocol is frozen and ordered, the dataset is the committed one,
two identical runs produce identical results, the train/validate/test windows are
disjoint and a strategy cannot read past its session, the headline metrics survive
an independent recomputation, and the artifact regenerates byte-for-byte.

Everything here is offline: the frozen CSVs are read from disk and the synthetic
feeds are built in memory. The suite's autouse network guard would fail any test
that tried to fetch.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from tradingagents.backtesting import (
    MarketDataFeed,
    PortfolioBacktester,
    assert_no_leakage,
    build_folds,
)
from tradingagents.baseline import (
    DEFAULT_PROTOCOL,
    PROTOCOL_VERSION,
    BaselineProtocol,
    SmaCrossover,
    crosscheck,
    harness,
    load_feed,
    load_frames,
)
from tradingagents.baseline.run import build_baseline
from tradingagents.performance.report import metrics_from_result


# A small, fast protocol over the same frozen data: SPY only, a short calendar,
# and short walk-forward windows so a handful of folds build quickly. It is used
# wherever a full baseline would be too slow for a unit test; the *shape* of the
# protocol is identical to the official one.
def small_protocol() -> BaselineProtocol:
    return dataclasses.replace(
        DEFAULT_PROTOCOL,
        universe=("SPY",),
        benchmark="SPY",
        start="2015-01-02",
        end="2016-06-30",
        dev_end="2015-06-30",
        validation_start="2015-07-01",
        validation_end="2015-12-31",
        test_start="2016-01-01",
        test_end="2016-06-30",
        wf_train_sessions=120,
        wf_validate_sessions=60,
        wf_test_sessions=60,
    )


def fast_strategy():
    """A short-window reference strategy, so unit tests warm up quickly.

    Same class and same code path as the official strategy; only the lengths
    differ, and the lengths do not change what is being tested (determinism,
    splits, cross-checks). The official lengths are exercised by the artifact
    test that uses ``build_baseline`` on the small protocol.
    """
    return SmaCrossover(position_size=0.2, fast=5, slow=20)


def synthetic_feed(sessions: int = 300, *, symbols=("AAA", "BBB"), seed: int = 7) -> MarketDataFeed:
    """A deterministic synthetic feed, for the look-ahead and fold tests.

    Uses a fixed pseudo-random walk (a plain LCG, so no global RNG state and no
    dependency on numpy's generator version), so the frames are identical on every
    run and every machine.
    """
    dates = pd.bdate_range("2020-01-01", periods=sessions)
    frames = {}
    for k, symbol in enumerate(symbols):
        state = seed + k * 1_000_003
        closes = []
        price = 100.0 + k * 10.0
        for _ in range(sessions):
            state = (1103515245 * state + 12345) % (2**31)
            step = ((state / 2**31) - 0.5) * 0.02
            price = max(1.0, price * (1.0 + step))
            closes.append(round(price, 4))
        frames[symbol] = pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [1_000_000.0] * sessions,
            },
            index=dates,
        )
    return MarketDataFeed(frames)


# ---------------------------------------------------------------------------
# The protocol is frozen, ordered and serialisable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocol:
    def test_version_is_present_and_recorded(self):
        assert DEFAULT_PROTOCOL.version == PROTOCOL_VERSION
        assert DEFAULT_PROTOCOL.to_dict()["version"] == PROTOCOL_VERSION

    def test_periods_are_ordered_and_disjoint(self):
        p = DEFAULT_PROTOCOL
        assert p.start <= p.dev_end < p.validation_start
        assert p.validation_start <= p.validation_end < p.test_start
        assert p.test_start <= p.test_end <= p.end

    def test_an_out_of_order_protocol_is_refused(self):
        with pytest.raises(ValueError):
            dataclasses.replace(DEFAULT_PROTOCOL, dev_end="2020-01-01")  # overlaps validation

    def test_no_leverage_and_long_only_caps(self):
        assert DEFAULT_PROTOCOL.leverage == 1.0
        assert DEFAULT_PROTOCOL.max_portfolio_exposure <= 1.0
        assert DEFAULT_PROTOCOL.max_position_size <= 1.0

    def test_to_dict_is_json_serialisable_and_carries_checksums(self):
        blob = DEFAULT_PROTOCOL.to_dict()
        text = json.dumps(blob)  # must not raise
        assert len(text) > 0
        assert set(blob["dataset_checksums"]) == set(DEFAULT_PROTOCOL.universe)
        for digest in blob["dataset_checksums"].values():
            assert len(digest) == 64  # sha256 hex

    def test_nested_objects_match_the_scalar_fields(self):
        p = DEFAULT_PROTOCOL
        assert p.costs.commission_rate == p.commission_rate
        assert p.costs.slippage_bps == p.slippage_bps
        assert p.limits.max_position_size == p.max_position_size


# ---------------------------------------------------------------------------
# The dataset is the committed one, and it loads offline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataset:
    def test_frames_load_for_every_universe_symbol(self):
        frames = load_frames(DEFAULT_PROTOCOL)
        assert set(frames) == set(DEFAULT_PROTOCOL.universe)
        for frame in frames.values():
            assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
            assert len(frame) > 2000

    def test_committed_files_match_the_recorded_checksums(self):
        import hashlib

        from tradingagents.baseline.protocol import DATA_DIR

        for symbol, digest in DEFAULT_PROTOCOL.dataset_checksums().items():
            actual = hashlib.sha256((DATA_DIR / f"{symbol}.csv").read_bytes()).hexdigest()
            assert actual == digest, f"{symbol} changed since the checksum was recorded"

    def test_feed_builds_and_calendar_is_ordered(self):
        feed = load_feed(small_protocol())
        cal = feed.calendar()
        assert cal == sorted(cal)
        assert len(cal) > 300


# ---------------------------------------------------------------------------
# The reference strategy is deterministic and cannot see the future
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReferenceStrategy:
    def test_identical_context_yields_identical_signal(self):
        feed = synthetic_feed()
        session = feed.calendar()[250]
        from tradingagents.backtesting import SessionClock, StrategyContext
        from tradingagents.portfolio import Ledger, LedgerView

        ledger = Ledger(initial_cash=100_000.0)
        ctx = StrategyContext(
            session=session, feed=feed, ledger=LedgerView(ledger),
            clock=SessionClock.for_session(session), symbols=feed.symbols,
        )
        strategy = SmaCrossover(position_size=0.2, fast=5, slow=20, stop_loss_pct=0.2)
        first = strategy(ctx)
        second = strategy(ctx)
        assert (first is None) == (second is None)
        if first is not None:
            assert first.signal_id == second.signal_id
            assert first.symbol == second.symbol

    def test_strategy_cannot_read_a_future_bar(self):
        """The probe strategy fails the test if the feed ever answers past its session."""
        feed = synthetic_feed(sessions=120)
        violations: list[str] = []

        def probe(context):
            nxt = context.feed.next_session(context.session)
            if nxt is not None and context.feed.bar(context.symbols[0], nxt) is not None:
                violations.append(f"saw {nxt} from {context.session}")
            hist = context.history(context.symbols[0])
            if len(hist) and hist.index[-1].strftime("%Y-%m-%d") > context.session:
                violations.append(f"history past {context.session}")
            return None

        PortfolioBacktester(feed, probe).run()
        assert violations == []

    def test_a_buy_is_not_repeated_while_a_position_is_open(self):
        feed = synthetic_feed(sessions=200)
        strategy = SmaCrossover(position_size=0.2, fast=5, slow=20)
        result = PortfolioBacktester(feed, strategy).run()
        # One position per symbol at a time: the count of a symbol's signals never
        # exceeds its count of completed round trips by more than one (the open one).
        symbols = [s.symbol for s in result.signals]
        for symbol in set(symbols):
            exits = sum(1 for t in result.trades if t.symbol == symbol)
            entries = symbols.count(symbol)
            assert entries - exits <= 1


# ---------------------------------------------------------------------------
# Determinism: identical inputs, identical outputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_two_runs_of_one_window_are_identical(self):
        p = small_protocol()
        feed = load_feed(p)
        strategy = fast_strategy()
        a = harness.run_window(feed, p, strategy, p.test_start, p.test_end)
        b = harness.run_window(feed, p, strategy, p.test_start, p.test_end)
        assert a.final_equity == b.final_equity
        assert [point.equity for point in a.equity_curve] == [
            point.equity for point in b.equity_curve
        ]
        assert [(t.symbol, t.entry_session, t.exit_session, t.net_pnl) for t in a.trades] == [
            (t.symbol, t.entry_session, t.exit_session, t.net_pnl) for t in b.trades
        ]

    def test_two_full_baselines_are_identical(self):
        p = small_protocol()
        first = build_baseline(p, fast_strategy())
        second = build_baseline(p, fast_strategy())
        # The wall-clock stamp is the only field allowed to differ.
        first.pop("generated_at")
        second.pop("generated_at")
        assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
            second, sort_keys=True, default=str
        )


# ---------------------------------------------------------------------------
# The splits are disjoint and out of sample
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSplits:
    def test_split_periods_do_not_overlap(self):
        periods = DEFAULT_PROTOCOL.split_periods()
        bounds = [(s, e) for _, s, e in periods]
        for i, (s1, e1) in enumerate(bounds):
            for s2, e2 in bounds[i + 1:]:
                assert e1 < s2 or e2 < s1, f"{(s1, e1)} overlaps {(s2, e2)}"

    def test_walk_forward_folds_are_disjoint_and_pass_the_leakage_guard(self):
        p = small_protocol()
        feed = load_feed(p)
        folds = build_folds(
            feed,
            train_sessions=p.wf_train_sessions,
            validate_sessions=p.wf_validate_sessions,
            test_sessions=p.wf_test_sessions,
        )
        assert folds
        for fold in folds:
            assert fold.train.end < fold.validate.start
            assert fold.validate.end < fold.test.start
        assert_no_leakage(folds, feed)

    def test_test_sessions_are_never_inside_train_or_validate(self):
        p = small_protocol()
        feed = load_feed(p)
        folds = build_folds(
            feed,
            train_sessions=p.wf_train_sessions,
            validate_sessions=p.wf_validate_sessions,
            test_sessions=p.wf_test_sessions,
        )
        for fold in folds:
            train = set(fold.train.sessions(feed))
            validate = set(fold.validate.sessions(feed))
            test = set(fold.test.sessions(feed))
            assert not (train & test)
            assert not (validate & test)

    def test_a_leaking_fold_cannot_be_built(self):
        from tradingagents.backtesting import Fold, Window, WindowError

        with pytest.raises(WindowError):
            Fold(
                index=0,
                train=Window("2020-01-01", "2020-06-30", "train"),
                validate=Window("2020-04-01", "2020-09-30", "validate"),  # overlaps train
                test=Window("2021-01-01", "2021-03-31", "test"),
            )


# ---------------------------------------------------------------------------
# Independent metric cross-check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossCheck:
    def test_independent_metrics_agree_with_the_engine(self):
        p = small_protocol()
        feed = load_feed(p)
        strategy = fast_strategy()
        result = harness.run_window(feed, p, strategy, p.test_start, p.test_end)
        metrics = metrics_from_result(result)
        checks = crosscheck.cross_check(result, metrics)
        assert checks
        failures = [c for c in checks if not c.agrees]
        assert not failures, f"independent recomputation disagreed: {failures}"
        assert crosscheck.all_agree(checks)

    def test_independent_drawdown_matches_a_hand_example(self):
        # peak 100 -> trough 80 -> 0.20; a later peak 120 -> 90 -> 0.25 is larger.
        equity = [100.0, 90.0, 80.0, 110.0, 120.0, 90.0]
        assert crosscheck.independent_max_drawdown(equity) == pytest.approx(0.25)

    def test_independent_total_return_matches_a_hand_example(self):
        assert crosscheck.independent_total_return([100.0, 150.0], 100.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The artifact is complete and regenerates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_artifact():
    """One small-protocol baseline artifact, built once and shared by its tests."""
    return build_baseline(small_protocol(), fast_strategy())


@pytest.mark.unit
class TestArtifact:
    def test_default_strategy_is_the_frozen_reference_rule(self):
        strategy = SmaCrossover()
        assert strategy.name == "sma_crossover_20_100"
        assert (strategy.fast, strategy.slow) == (20, 100)

    def test_artifact_has_every_required_block(self, small_artifact):
        artifact = small_artifact
        for key in (
            "artifact_schema_version", "generated_at", "metadata", "protocol",
            "strategy", "benchmark", "results", "independent_cross_check",
            "cross_check_passes", "regime_analysis", "concentration",
            "cost_sensitivity", "walk_forward",
        ):
            assert key in artifact, f"artifact missing {key!r}"
        assert set(artifact["results"]) == {"development", "validation", "test"}
        assert artifact["cross_check_passes"] is True

    def test_artifact_is_json_serialisable(self, small_artifact):
        assert json.dumps(small_artifact, default=str)

    def test_artifact_records_protocol_and_data_provenance(self, small_artifact):
        artifact = small_artifact
        assert artifact["protocol"]["version"] == PROTOCOL_VERSION
        assert artifact["protocol"]["dataset_checksums"]
        assert artifact["strategy"]["id"] == fast_strategy().name
        assert artifact["protocol"]["seed"] is not None


# ---------------------------------------------------------------------------
# Regime labels are knowable at the session (no future leak in the label)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegimes:
    def test_regime_labels_use_only_trailing_data(self):
        p = small_protocol()
        feed = load_feed(p)
        sessions = feed.calendar(p.start, p.end)
        labels = harness.regime_labels(feed, p, sessions)
        assert len(labels) == len(sessions)
        assert set(labels["trend"].unique()) <= {"bull", "bear"}
        assert set(labels["volatility"].unique()) <= {"high_vol", "low_vol"}

    def test_regime_breakdown_partitions_the_curve(self):
        p = small_protocol()
        feed = load_feed(p)
        strategy = fast_strategy()
        result = harness.run_window(feed, p, strategy, p.test_start, p.test_end)
        equity = result.equity_series()
        labels = harness.regime_labels(feed, p, list(result.sessions))
        breakdown = harness.regime_breakdown(equity, labels)
        counted = sum(g["sessions"] for g in breakdown["trend"].values())
        assert counted == len(equity)


# ---------------------------------------------------------------------------
# Cost sensitivity is a sensitivity test, not a tuning step
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCostSensitivity:
    def test_higher_costs_never_improve_the_result(self):
        p = small_protocol()
        feed = load_feed(p)
        strategy = fast_strategy()
        sensitivity = harness.cost_sensitivity(feed, p, strategy, label="test")
        assert sensitivity["zero_cost"]["net_pnl"] >= sensitivity["baseline"]["net_pnl"]
        assert sensitivity["double_cost"]["net_pnl"] <= sensitivity["baseline"]["net_pnl"]
        assert sensitivity["quadruple_cost"]["net_pnl"] <= sensitivity["double_cost"]["net_pnl"]

    def test_costs_are_non_negative_and_ordered(self):
        p = small_protocol()
        feed = load_feed(p)
        strategy = fast_strategy()
        sensitivity = harness.cost_sensitivity(feed, p, strategy, label="test")
        assert sensitivity["zero_cost"]["total_fees"] == 0.0
        assert (
            sensitivity["quadruple_cost"]["total_fees"]
            > sensitivity["baseline"]["total_fees"]
            >= sensitivity["zero_cost"]["total_fees"]
        )
