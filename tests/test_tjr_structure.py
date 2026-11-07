"""TJR structure primitives: swings, breaks, gaps, equilibrium, running extremes.

Each test builds a frame whose answer is obvious from the construction, so a
regression in a primitive shows up as a wrong level rather than a wrong trade.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.tjr.levels import running_extreme_before
from tradingagents.tjr.structure import (
    break_of_structure,
    equilibrium,
    fair_value_gaps,
    inverse_fair_value_gap,
    last_swing,
    swing_points,
)

pytestmark = pytest.mark.unit


def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-10-29 13:30", periods=len(bars), freq="5min", tz="UTC")
    frame = pd.DataFrame(bars, index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


def test_swing_high_and_low_are_confirmed_only_after_the_lookback():
    frame = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.2, 9.9, 10.1),
        (10.1, 11.0, 10.0, 10.5),   # swing high 11.0 at index 2
        (10.5, 10.6, 10.0, 10.2),
        (10.2, 10.3, 9.5, 9.8),     # swing low 9.5 at index 4
        (9.8, 10.0, 9.6, 9.9),
        (9.9, 10.1, 9.8, 10.0),
    ])
    swings = swing_points(frame, lookback=2)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    assert [round(s.price, 2) for s in highs] == [11.0]
    assert [round(s.price, 2) for s in lows] == [9.5]
    # Confirmed two bars later: index 2 -> index 4; index 4 -> index 6.
    assert highs[0].confirmed_at == frame.index[4]
    assert lows[0].confirmed_at == frame.index[6]


def test_last_swing_respects_the_confirmation_instant():
    frame = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.2, 9.9, 10.1),
        (10.1, 11.0, 10.0, 10.5),
        (10.5, 10.6, 10.0, 10.2),
        (10.2, 10.3, 9.5, 9.8),
        (9.8, 10.0, 9.6, 9.9),
        (9.9, 10.1, 9.8, 10.0),
    ])
    swings = swing_points(frame, lookback=2)
    # At index 3 the high at index 2 is not yet confirmed.
    assert last_swing(swings, "high", at=frame.index[3]) is None
    assert last_swing(swings, "high", at=frame.index[4]) is not None


def test_break_of_structure_needs_a_close_beyond_the_swing():
    frame = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.2, 9.9, 10.1),
        (10.1, 11.0, 10.0, 10.5),   # swing high 11.0, confirmed at index 4
        (10.5, 10.6, 10.0, 10.2),
        (10.2, 10.9, 10.0, 10.8),   # wick 10.9, close 10.8: no break
        (10.8, 11.1, 10.7, 11.05),  # close 11.05 > 11.0: break
        (11.05, 11.2, 11.0, 11.1),
    ])
    swings = swing_points(frame, lookback=2)
    assert [round(s.price, 2) for s in swings if s.kind == "high"] == [11.0]
    assert break_of_structure(frame, swings, direction="up") == frame.index[5]


def test_break_of_structure_can_be_restricted_to_since():
    frame = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.2, 9.9, 10.1),
        (10.1, 11.0, 10.0, 10.5),
        (10.5, 10.6, 10.0, 10.2),
        (10.2, 10.9, 10.0, 10.8),
        (10.8, 11.1, 10.7, 11.05),  # the only break, at index 5
        (11.05, 11.2, 10.8, 10.95),  # closes back below 11.0: no second break
    ])
    swings = swing_points(frame, lookback=2)
    assert break_of_structure(frame, swings, direction="up") == frame.index[5]
    assert break_of_structure(frame, swings, direction="up", since=frame.index[6]) is None


def test_fair_value_gap_is_a_three_candle_imbalance():
    frame = _frame([
        (10.0, 10.2, 9.8, 10.0),
        (10.0, 10.6, 9.9, 10.5),
        (10.7, 11.2, 10.5, 11.0),   # low 10.5 > high two back 10.2 -> bull gap
    ])
    gaps = fair_value_gaps(frame, kind="bull")
    assert len(gaps) == 1
    assert gaps[0].low == 10.2
    assert gaps[0].high == 10.5
    assert gaps[0].midpoint == pytest.approx(10.35)


def test_bearish_gap_is_the_mirror():
    frame = _frame([
        (10.0, 10.4, 9.8, 10.0),
        (10.0, 10.1, 9.5, 9.6),
        (9.3, 9.5, 8.9, 9.0),       # high 9.5 < low two back 9.8 -> bear gap
    ])
    gaps = fair_value_gaps(frame, kind="bear")
    assert len(gaps) == 1
    assert gaps[0].low == 9.5
    assert gaps[0].high == 9.8


def test_inverse_fair_value_gap_flips_when_price_closes_back_through():
    frame = _frame([
        (10.0, 10.4, 9.8, 10.0),
        (10.0, 10.1, 9.5, 9.6),
        (9.3, 9.5, 8.9, 9.0),       # bear gap band (9.5, 9.8)
        (9.0, 9.6, 8.9, 9.55),
        (9.55, 9.9, 9.5, 9.85),     # close 9.85 > 9.8: the bear gap became support
    ])
    flipped = inverse_fair_value_gap(frame, direction="up")
    assert flipped is not None
    assert flipped.kind == "bear"
    assert flipped.low == 9.5 and flipped.high == 9.8


def test_inverse_fair_value_gap_is_none_before_the_close_through():
    frame = _frame([
        (10.0, 10.4, 9.8, 10.0),
        (10.0, 10.1, 9.5, 9.6),
        (9.3, 9.5, 8.9, 9.0),
        (9.0, 9.6, 8.9, 9.55),
    ])
    assert inverse_fair_value_gap(frame, direction="up") is None


def test_equilibrium_is_the_midpoint_and_refuses_an_inverted_leg():
    assert equilibrium(98.0, 102.0) == 100.0
    with pytest.raises(ValueError):
        equilibrium(102.0, 98.0)


def test_running_extreme_excludes_the_current_bar():
    frame = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.5, 9.8, 10.2),
        (10.2, 11.0, 10.1, 10.9),   # a bar that both sweeps and prints the high
    ])
    running_high = running_extreme_before(frame, kind="high")
    running_low = running_extreme_before(frame, kind="low")
    assert pd.isna(running_high.iloc[0])
    assert running_high.iloc[1] == 10.1
    # The third bar's own high (11.0) is excluded; the prior high is 10.5.
    assert running_high.iloc[2] == 10.5
    assert running_low.iloc[2] == 9.8
