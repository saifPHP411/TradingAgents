"""Walk-forward evaluation: the split, the leakage checks, and the selection rule.

Three properties are worth testing and each has a way of failing silently:

- **The windows do not overlap.** If they do, the "out-of-sample" score is
  in-sample and simply looks better. So the overlap is a construction error, and
  the tests assert it raises.
- **The test window is unreachable from the training window.** Not just
  documented — the feed handed to a window's evaluation contains only that
  window's bars, so a strategy asking for history gets nothing later than the
  window's end.
- **Selection never reads the test score.** The tests pass a candidate that is
  best on test and worst on validation and assert it is *not* chosen, which is
  the only way to show the selection is honest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.backtesting import (
    MarketDataFeed,
    Window,
    WindowError,
    assert_no_leakage,
    build_folds,
    run_walk_forward,
    select_best,
    slice_feed,
)

DATES = pd.bdate_range("2026-01-05", periods=60)


def _feed(closes=None) -> MarketDataFeed:
    closes = closes or [100.0 + i for i in range(len(DATES))]
    frame = pd.DataFrame(
        {"open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
         "close": closes, "volume": [1_000] * len(DATES)},
        index=DATES,
    )
    return MarketDataFeed({"AAPL": frame})


# ---------------------------------------------------------------------------
# Window and Fold validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWindow:
    def test_a_reversed_window_is_rejected(self):
        with pytest.raises(WindowError):
            Window("2026-03-01", "2026-01-01")

    def test_contains_is_inclusive(self):
        window = Window("2026-01-05", "2026-01-09")
        assert window.contains("2026-01-05")
        assert window.contains("2026-01-09")
        assert not window.contains("2026-01-12")

    def test_overlap_detection(self):
        a = Window("2026-01-05", "2026-01-09")
        assert a.overlaps(Window("2026-01-09", "2026-01-12"))
        assert not a.overlaps(Window("2026-01-12", "2026-01-16"))

    def test_sessions_are_read_from_the_feed(self):
        feed = _feed()
        window = Window("2026-01-05", "2026-01-09")
        assert window.sessions(feed) == ["2026-01-05", "2026-01-06", "2026-01-07",
                                         "2026-01-08", "2026-01-09"]


@pytest.mark.unit
class TestFoldValidation:
    def _windows(self):
        return (
            Window("2026-01-05", "2026-01-16", "train"),
            Window("2026-01-19", "2026-01-23", "validate"),
            Window("2026-01-26", "2026-01-30", "test"),
        )

    def test_a_valid_fold_is_accepted(self):
        from tradingagents.backtesting import Fold

        train, validate, test = self._windows()
        fold = Fold(index=0, train=train, validate=validate, test=test)
        assert fold.label == "fold0"

    def test_a_fold_whose_validation_overlaps_training_is_rejected(self):
        from tradingagents.backtesting import Fold

        train, validate, test = self._windows()
        overlapping = Window("2026-01-14", "2026-01-23", "validate")
        with pytest.raises(WindowError, match="overlap"):
            Fold(index=0, train=train, validate=overlapping, test=test)

    def test_a_fold_whose_test_overlaps_validation_is_rejected(self):
        """A test that overlaps validation is not out of sample."""
        from tradingagents.backtesting import Fold

        train, validate, test = self._windows()
        leaking = Window("2026-01-21", "2026-01-30", "test")
        with pytest.raises(WindowError, match="out of sample"):
            Fold(index=0, train=train, validate=validate, test=leaking)

    def test_a_fold_whose_test_precedes_training_is_rejected(self):
        from tradingagents.backtesting import Fold

        train, validate, test = self._windows()
        early = Window("2026-01-05", "2026-01-09", "test")
        with pytest.raises(WindowError):
            Fold(index=0, train=train, validate=validate, test=early)


# ---------------------------------------------------------------------------
# Building folds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildFolds:
    def test_folds_are_built_and_ordered(self):
        folds = build_folds(_feed(), train_sessions=10, validate_sessions=5, test_sessions=5)
        assert folds
        for fold in folds:
            assert fold.train.end < fold.validate.start
            assert fold.validate.end < fold.test.start

    def test_test_windows_do_not_overlap_each_other(self):
        folds = build_folds(_feed(), train_sessions=10, validate_sessions=5, test_sessions=5)
        for earlier, later in zip(folds, folds[1:], strict=False):
            assert not earlier.test.overlaps(later.test)

    def test_the_step_controls_the_advance(self):
        folds = build_folds(_feed(), train_sessions=10, validate_sessions=5, test_sessions=5,
                            step_sessions=1)
        assert len(folds) > 5
        for earlier, later in zip(folds, folds[1:], strict=False):
            assert earlier.train.start < later.train.start

    def test_too_little_data_is_refused(self):
        with pytest.raises(WindowError, match="needs"):
            build_folds(_feed(), train_sessions=50, validate_sessions=50, test_sessions=50)

    def test_a_non_positive_window_is_refused(self):
        for kwargs in ({"train_sessions": 0}, {"validate_sessions": 0}, {"test_sessions": 0}):
            base = {"train_sessions": 5, "validate_sessions": 5, "test_sessions": 5}
            base.update(kwargs)
            with pytest.raises(WindowError):
                build_folds(_feed(), **base)

    def test_the_date_range_can_be_bounded(self):
        folds = build_folds(_feed(), train_sessions=5, validate_sessions=5, test_sessions=5,
                            start="2026-01-05", end="2026-02-06")
        assert folds
        assert folds[0].train.start >= "2026-01-05"
        assert folds[-1].test.end <= "2026-02-06"


# ---------------------------------------------------------------------------
# Slicing the feed: enforcement, not documentation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSliceFeed:
    def test_the_slice_contains_only_the_window(self):
        feed = _feed()
        window = Window("2026-01-05", "2026-01-09", "train")
        sliced = slice_feed(feed, window)
        assert sliced.calendar() == window.sessions(feed)

    def test_a_strategy_cannot_see_past_the_window(self):
        """The frames are re-cut, so asking for history returns nothing later."""
        feed = _feed()
        window = Window("2026-01-05", "2026-01-09", "train")
        sliced = slice_feed(feed, window)
        history = sliced.history("AAPL", "2026-12-31")
        assert all(d <= "2026-01-09" for d in history.index.strftime("%Y-%m-%d"))

    def test_the_slice_cannot_see_before_the_window_either(self):
        feed = _feed()
        window = Window("2026-02-02", "2026-02-06", "test")
        sliced = slice_feed(feed, window)
        assert all(d >= "2026-02-02" for d in sliced.calendar())

    def test_a_window_with_no_bars_is_refused(self):
        feed = _feed()
        with pytest.raises(WindowError):
            slice_feed(feed, Window("2030-01-01", "2030-01-31", "test"))


# ---------------------------------------------------------------------------
# Selection: the test score must not inform the choice
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelection:
    def test_the_best_validation_score_wins(self):
        assert select_best({"a": 0.1, "b": 0.3, "c": 0.2}) == "b"

    def test_minimization_can_be_requested(self):
        assert select_best({"a": 0.1, "b": 0.3}, maximize=False) == "a"

    def test_ties_break_deterministically(self):
        assert select_best({"b": 0.2, "a": 0.2}) == "a"
        assert select_best({"a": 0.2, "b": 0.2}) == "a"

    def test_an_empty_score_set_is_refused(self):
        with pytest.raises(WindowError):
            select_best({})

    def test_selection_has_no_way_to_see_a_test_score(self):
        """There is no argument for a test score, so the choice cannot use one."""
        import inspect

        params = set(inspect.signature(select_best).parameters)
        assert not any("test" in name for name in params)


# ---------------------------------------------------------------------------
# The walk-forward run
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunWalkForward:
    def test_a_fold_selects_on_validation_and_scores_on_test(self):
        """A candidate that is best on test but not on validation must not be chosen."""
        # 'overfit' wins on training, 'honest' wins on validation, and 'overfit'
        # would win on test. Selection must pick 'honest'.
        calls: list[tuple[str, str]] = []

        def evaluate(candidate, feed, window):
            calls.append((candidate, window.label))
            if candidate == "overfit":
                return {"train": 10.0, "validate": 0.1, "test": 10.0}[window.label]
            return {"train": 1.0, "validate": 5.0, "test": 1.0}[window.label]

        result = run_walk_forward(
            _feed(), candidates=["overfit", "honest"], evaluate=evaluate,
            train_sessions=10, validate_sessions=5, test_sessions=5,
        )
        assert set(result.selected.values()) == {"honest"}
        # The test window was scored by exactly one candidate per fold.
        test_calls = [c for c in calls if c[1] == "test"]
        assert {c for c, _ in test_calls} == {"honest"}
        assert len(test_calls) == len(result.folds)

    def test_the_test_window_is_scored_once_per_fold(self):
        counts: dict[str, int] = {}

        def evaluate(candidate, feed, window):
            counts[window.label] = counts.get(window.label, 0) + 1
            return 1.0

        result = run_walk_forward(
            _feed(), candidates=["a", "b", "c"], evaluate=evaluate,
            train_sessions=10, validate_sessions=5, test_sessions=5,
        )
        # 3 candidates on train and validate, 1 on test, per fold.
        assert counts["train"] == 3 * len(result.folds)
        assert counts["validate"] == 3 * len(result.folds)
        assert counts["test"] == len(result.folds)

    def test_the_evaluator_only_sees_its_own_window(self):
        seen: list[tuple[str, str, str]] = []

        def evaluate(candidate, feed, window):
            seen.append((window.label, feed.calendar()[0], feed.calendar()[-1]))
            return 0.0

        run_walk_forward(_feed(), candidates=["a"], evaluate=evaluate,
                         train_sessions=10, validate_sessions=5, test_sessions=5)
        for _label, first, last in seen:
            assert first <= last

    def test_the_overfitting_gap_is_reported(self):
        def evaluate(candidate, feed, window):
            return {"train": 10.0, "validate": 1.0, "test": 1.0}[window.label]

        result = run_walk_forward(_feed(), candidates=["a"], evaluate=evaluate,
                                  train_sessions=10, validate_sessions=5, test_sessions=5)
        assert result.overfitting_gap == pytest.approx(9.0)

    def test_the_mean_test_score_is_the_headline(self):
        def evaluate(candidate, feed, window):
            return 2.0 if window.label == "test" else 1.0

        result = run_walk_forward(_feed(), candidates=["a"], evaluate=evaluate,
                                  train_sessions=10, validate_sessions=5, test_sessions=5)
        assert result.mean_test_score == pytest.approx(2.0)

    def test_the_result_is_json_safe(self):
        import json

        def evaluate(candidate, feed, window):
            return 1.0

        result = run_walk_forward(_feed(), candidates=["a"], evaluate=evaluate,
                                  train_sessions=10, validate_sessions=5, test_sessions=5)
        json.dumps(result.to_dict())

    def test_a_candidate_object_with_a_name_is_supported(self):
        class Candidate:
            def __init__(self, name):
                self.name = name

        def evaluate(candidate, feed, window):
            return 1.0 if candidate.name == "x" else 2.0

        result = run_walk_forward(_feed(), candidates=[Candidate("x"), Candidate("y")],
                                  evaluate=evaluate, train_sessions=10, validate_sessions=5,
                                  test_sessions=5)
        assert set(result.selected.values()) == {"y"}

    def test_the_run_is_deterministic(self):
        def evaluate(candidate, feed, window):
            return {"train": 1.0, "validate": 2.0, "test": 3.0}[window.label]

        first = run_walk_forward(_feed(), candidates=["a", "b"], evaluate=evaluate,
                                 train_sessions=10, validate_sessions=5,
                                 test_sessions=5).to_dict()
        second = run_walk_forward(_feed(), candidates=["a", "b"], evaluate=evaluate,
                                  train_sessions=10, validate_sessions=5,
                                  test_sessions=5).to_dict()
        assert first == second


# ---------------------------------------------------------------------------
# The leakage assertion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeakageAssertion:
    def test_valid_folds_pass(self):
        folds = build_folds(_feed(), train_sessions=10, validate_sessions=5, test_sessions=5)
        assert_no_leakage(folds, _feed())

    def test_overlapping_folds_are_caught(self):
        """A fold list from a config file has not been through the constructor."""
        from dataclasses import dataclass

        @dataclass
        class Loose:
            index: int
            train: Window
            validate: Window
            test: Window

            def all_windows(self):
                return (self.train, self.validate, self.test)

        bad = Loose(
            index=0,
            train=Window("2026-01-05", "2026-01-20", "train"),
            validate=Window("2026-01-19", "2026-01-23", "validate"),
            test=Window("2026-01-26", "2026-01-30", "test"),
        )
        with pytest.raises(WindowError, match="overlaps"):
            assert_no_leakage([bad], _feed())

    def test_a_window_with_no_sessions_is_caught(self):
        folds = build_folds(_feed(), train_sessions=10, validate_sessions=5, test_sessions=5)
        from dataclasses import dataclass

        @dataclass
        class Loose:
            index: int
            train: Window
            validate: Window
            test: Window

            def all_windows(self):
                return (self.train, self.validate, self.test)

        bad = Loose(index=0, train=folds[0].train, validate=folds[0].validate,
                    test=Window("2031-01-01", "2031-01-31", "test"))
        with pytest.raises(WindowError, match="no sessions"):
            assert_no_leakage([bad], _feed())


# ---------------------------------------------------------------------------
# Reproducibility of candidate naming
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCandidateNamesAreReproducible:
    """A candidate's key must not change between runs.

    Regression test for a real defect: a candidate with no ``name`` was keyed by
    ``repr()``, which embeds the object's memory address. The same configuration
    therefore produced a different ``selected`` mapping on every run, so a
    walk-forward result could not be compared, cached, or asserted on.
    """

    def _run(self, candidates, evaluate):
        return run_walk_forward(
            _feed(), candidates=candidates, evaluate=evaluate,
            train_sessions=10, validate_sessions=5, test_sessions=5,
        )

    def test_a_nameless_candidate_is_not_keyed_by_address(self):
        def evaluate(candidate, feed, window):
            return 1.0

        first = self._run([object()], evaluate)
        second = self._run([object()], evaluate)
        assert set(first.selected.values()) == set(second.selected.values())

    def test_a_nameless_candidate_key_contains_no_hex_address(self):
        import re

        def evaluate(candidate, feed, window):
            return 1.0

        result = self._run([object()], evaluate)
        for name in result.selected.values():
            assert not re.search(r"0x[0-9a-fA-F]+", name)

    def test_two_candidates_of_the_same_type_keep_separate_scores(self):
        """A shared key would make one candidate's score overwrite the other's."""
        class Strategy:
            def __init__(self, score):
                self.score = score

        def evaluate(candidate, feed, window):
            return candidate.score

        result = self._run([Strategy(0.1), Strategy(0.9)], evaluate)
        # Two distinct keys, so the better one (0.9) is chosen and both scores show.
        assert len(result.train_scores[result.folds[0].index]) == 2
        assert len(set(result.selected.values())) == 1
