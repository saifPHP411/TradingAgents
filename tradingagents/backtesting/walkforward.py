"""Walk-forward evaluation: train, validate, test, with the future kept out.

The failure this module exists to prevent is choosing a strategy because it
worked on the data it was chosen from. A single backtest cannot tell you that:
the same numbers come back whether the parameters were reasoned about or fitted.
Splitting the data does, but only if the split is enforced rather than documented.

So the split is enforced by construction, three ways:

1. **Windows are disjoint in the direction that matters.** A fold's training and
   validation windows end strictly before its test window begins, and the check
   is a hard error rather than a warning. There is no way to build a fold whose
   test data overlaps its own training data.
2. **The feed is re-cut per window.** :meth:`WalkForward.split_feed` returns a
   feed containing only the window's sessions, so a strategy cannot reach a bar
   outside it even by asking — the frames are sliced, not just the date range.
3. **The selection happens on validation, never on test.** :func:`select_best`
   takes validation scores and returns the chosen parameter set; the caller then
   evaluates that one set on the test window. The test score is never an input to
   the choice, which is the whole point.

What walk-forward does *not* fix is a data source that backfills. If a news
vendor inserts an article with an old timestamp, every window looks clean and the
look-ahead is real anyway. That is Phase 9's problem, and it is why the audit
records PIT per source rather than claiming it globally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tradingagents.backtesting.data import MarketDataFeed


class WindowError(ValueError):
    """A window or a fold is not a valid out-of-sample split."""


@dataclass(frozen=True)
class Window:
    """A contiguous date range: ``[start, end]``, both inclusive."""

    start: str
    end: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise WindowError(f"window start {self.start} is after end {self.end}")

    def contains(self, session: str) -> bool:
        return self.start <= session <= self.end

    def overlaps(self, other: Window) -> bool:
        return not (self.end < other.start or other.end < self.start)

    def sessions(self, feed: MarketDataFeed) -> list[str]:
        return feed.calendar(self.start, self.end)

    def __str__(self) -> str:
        return f"{self.label or 'window'}[{self.start}..{self.end}]"


@dataclass(frozen=True)
class Fold:
    """One walk-forward split: a training window, a validation window, a test window.

    The ordering is validated at construction. A fold whose training window
    reaches into its test window is rejected rather than silently producing an
    optimistic number, because that number would look like a result.
    """

    index: int
    train: Window
    validate: Window
    test: Window

    def __post_init__(self) -> None:
        # Validation is drawn from data the training window did not use, and the
        # test window comes after both. Anything else is in-sample testing.
        if self.validate.start <= self.train.end:
            raise WindowError(
                f"fold {self.index}: the validation window {self.validate} starts before the "
                f"training window {self.train} has ended, so they overlap"
            )
        if self.test.start <= self.validate.end:
            raise WindowError(
                f"fold {self.index}: the test window {self.test} starts before the validation "
                f"window {self.validate} has ended, so the test is not out of sample"
            )

    @property
    def label(self) -> str:
        return f"fold{self.index}"

    def all_windows(self) -> tuple[Window, Window, Window]:
        return (self.train, self.validate, self.test)


@dataclass
class WalkForwardResult:
    """The outcome of a walk-forward run: one result per fold, plus the selection."""

    folds: list[Fold] = field(default_factory=list)
    train_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    """Per fold, the score of each candidate parameter set on the training window."""
    validation_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    selected: dict[int, str] = field(default_factory=dict)
    """Per fold, the parameter set chosen from the validation scores."""
    test_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    """Per fold, the score of the *selected* set on the test window."""
    selected_test_scores: list[float] = field(default_factory=list)
    """The chosen set's test score, per fold, in order. The honest headline."""
    best_in_sample_scores: list[float] = field(default_factory=list)
    """The best training score per fold. Comparing this with the test score shows
    how much of the training performance was fit rather than real."""

    @property
    def mean_test_score(self) -> float | None:
        if not self.selected_test_scores:
            return None
        return sum(self.selected_test_scores) / len(self.selected_test_scores)

    @property
    def mean_best_in_sample_score(self) -> float | None:
        if not self.best_in_sample_scores:
            return None
        return sum(self.best_in_sample_scores) / len(self.best_in_sample_scores)

    @property
    def overfitting_gap(self) -> float | None:
        """Best in-sample score minus out-of-sample score.

        A large positive gap means the parameter search was fitting noise: the
        chosen set did well on the data it was chosen from and worse on data it
        had not seen. This is the number that makes walk-forward worth running.
        """
        best = self.mean_best_in_sample_score
        out = self.mean_test_score
        if best is None or out is None:
            return None
        return best - out

    def to_dict(self) -> dict:
        return {
            "folds": [
                {"index": f.index, "train": [f.train.start, f.train.end],
                 "validate": [f.validate.start, f.validate.end],
                 "test": [f.test.start, f.test.end]}
                for f in self.folds
            ],
            "train_scores": self.train_scores,
            "validation_scores": self.validation_scores,
            "selected": self.selected,
            "test_scores": self.test_scores,
            "selected_test_scores": self.selected_test_scores,
            "best_in_sample_scores": self.best_in_sample_scores,
            "mean_test_score": self.mean_test_score,
            "mean_best_in_sample_score": self.mean_best_in_sample_score,
            "overfitting_gap": self.overfitting_gap,
        }


def _advance(feed: MarketDataFeed, session: str, sessions: int) -> str | None:
    """The session ``sessions`` after ``session``, or ``None`` past the data."""
    cursor = session
    for _ in range(sessions):
        nxt = feed.next_session(cursor)
        if nxt is None:
            return None
        cursor = nxt
    return cursor


def build_folds(
    feed: MarketDataFeed,
    *,
    train_sessions: int,
    validate_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[Fold]:
    """Roll a train/validate/test split across the calendar.

    Each fold advances by ``step_sessions`` (defaulting to ``test_sessions``, so
    the test windows are contiguous and non-overlapping). The three windows within
    a fold never overlap, and a fold is only produced when all three windows have
    enough sessions, so no fold is silently short of data.

    ``train_sessions`` counts the sessions before the validation window; the
    training window is the ``train_sessions`` sessions ending where validation
    begins, so training always immediately precedes validation.
    """
    for name, value in (("train_sessions", train_sessions), ("validate_sessions", validate_sessions),
                        ("test_sessions", test_sessions)):
        if value < 1:
            raise WindowError(f"{name} must be at least 1, got {value}")
    step = step_sessions if step_sessions is not None else test_sessions
    if step < 1:
        raise WindowError(f"step_sessions must be at least 1, got {step}")

    calendar = feed.calendar(start, end)
    needed = train_sessions + validate_sessions + test_sessions
    if len(calendar) < needed:
        raise WindowError(
            f"the calendar has {len(calendar)} sessions but a fold needs {needed} "
            f"({train_sessions} train + {validate_sessions} validate + {test_sessions} test)"
        )

    positions = {session: i for i, session in enumerate(calendar)}
    folds: list[Fold] = []
    index = 0
    anchor = 0
    while anchor + needed <= len(calendar):
        train_start = calendar[anchor]
        train_end = calendar[anchor + train_sessions - 1]
        validate_start = calendar[anchor + train_sessions]
        validate_end = calendar[anchor + train_sessions + validate_sessions - 1]
        test_start = calendar[anchor + train_sessions + validate_sessions]
        test_end = calendar[anchor + needed - 1]
        folds.append(
            Fold(
                index=index,
                train=Window(train_start, train_end, label="train"),
                validate=Window(validate_start, validate_end, label="validate"),
                test=Window(test_start, test_end, label="test"),
            )
        )
        index += 1
        anchor += step
        # Guard against a step that cannot make progress, which would loop forever.
        if step < 1:
            break
    if not folds:
        raise WindowError("no fold could be built from the supplied windows and calendar")
    # `positions` documents that the calendar is the authority; unused otherwise.
    del positions
    return folds


def slice_feed(feed: MarketDataFeed, window: Window) -> MarketDataFeed:
    """A feed containing only ``window``'s bars, for every symbol.

    The frames are re-cut rather than the range merely recorded, so a strategy
    running inside the window cannot obtain a bar outside it even by asking the
    feed directly. This is the difference between a split that is documented and
    one that is enforced.
    """
    frames: dict[str, Any] = {}
    for symbol in feed.symbols:
        history = feed.history(symbol, window.end)
        if history.empty:
            continue
        cut = history[history.index.strftime("%Y-%m-%d") >= window.start]
        if not cut.empty:
            frames[symbol] = cut
    if not frames:
        raise WindowError(f"no bars fall inside {window}")
    return MarketDataFeed(frames)


def select_best(
    validation_scores: dict[str, float], *, maximize: bool = True
) -> str:
    """The parameter set with the best validation score.

    Takes validation scores only, and there is no argument for a test score: the
    selection cannot see the test window, which is what makes the test result
    meaningful. Ties break on the parameter name so the choice is deterministic.
    """
    if not validation_scores:
        raise WindowError("no validation scores to select from")
    key = (lambda item: (-item[1], item[0])) if maximize else (lambda item: (item[1], item[0]))
    return min(validation_scores.items(), key=key)[0]


def run_walk_forward(
    feed: MarketDataFeed,
    *,
    candidates: Sequence[Any],
    evaluate: Callable[[Any, MarketDataFeed, Window], float],
    train_sessions: int,
    validate_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
    maximize: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> WalkForwardResult:
    """Run every candidate on every fold, select on validation, score on test.

    ``evaluate(candidate, feed, window)`` must run the candidate over the window's
    feed and return a single score (an out-of-sample return, say). It is given a
    feed already cut to the window, so it cannot read outside it.

    The sequence per fold is: score every candidate on training, score every
    candidate on validation, choose the best on validation, then score *only the
    chosen candidate* on test. The test window is touched once, by one candidate,
    and never informs the choice.
    """
    folds = build_folds(
        feed,
        train_sessions=train_sessions,
        validate_sessions=validate_sessions,
        test_sessions=test_sessions,
        step_sessions=step_sessions,
        start=start,
        end=end,
    )
    result = WalkForwardResult(folds=folds)
    for fold in folds:
        train_feed = slice_feed(feed, fold.train)
        validate_feed = slice_feed(feed, fold.validate)
        test_feed = slice_feed(feed, fold.test)

        train_scores = {
            _candidate_name(candidate): float(evaluate(candidate, train_feed, fold.train))
            for candidate in candidates
        }
        validation_scores = {
            _candidate_name(candidate): float(evaluate(candidate, validate_feed, fold.validate))
            for candidate in candidates
        }
        chosen = select_best(validation_scores, maximize=maximize)
        # The chosen candidate is re-evaluated on test, and nothing else is.
        chosen_candidate = next(c for c in candidates if _candidate_name(c) == chosen)
        test_score = float(evaluate(chosen_candidate, test_feed, fold.test))

        result.train_scores[fold.index] = train_scores
        result.validation_scores[fold.index] = validation_scores
        result.selected[fold.index] = chosen
        result.test_scores[fold.index] = {chosen: test_score}
        result.selected_test_scores.append(test_score)
        best_in_sample = max(train_scores.values()) if maximize else min(train_scores.values())
        result.best_in_sample_scores.append(best_in_sample)
    return result


def _candidate_name(candidate: Any) -> str:
    """A stable name for a candidate, so scores can be keyed by it."""
    if isinstance(candidate, str):
        return candidate
    name = getattr(candidate, "name", None)
    if name:
        return str(name)
    return repr(candidate)


def assert_no_leakage(folds: Iterable[Fold], feed: MarketDataFeed) -> None:
    """Verify every fold's windows are ordered and disjoint, over real sessions.

    A belt-and-braces check for a caller that built its folds elsewhere: the
    construction already enforces this, but a fold list loaded from a config file
    has not been through it.
    """
    for fold in folds:
        windows = fold.all_windows()
        for i, left in enumerate(windows):
            for right in windows[i + 1:]:
                if left.overlaps(right):
                    raise WindowError(
                        f"fold {fold.index}: {left} overlaps {right}, so a score computed on "
                        "one would not be out of sample with respect to the other"
                    )
        for window in windows:
            if not window.sessions(feed):
                raise WindowError(f"fold {fold.index}: {window} contains no sessions in the feed")
