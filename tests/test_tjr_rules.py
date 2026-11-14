"""The TJR rule chain and its adapter to the engine's signal.

The two hand-built sessions in ``tjr_helpers`` are the fixtures: a bullish one whose
sweep-confirmation-retrace-entry chain completes, and its bearish mirror. Every
assertion is about *which stage* the chain reached and *why* it stopped, so a change
that quietly trades on a partial setup fails here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tjr_helpers import BEARISH_BARS, BULLISH_BARS, SESSION, series_pair
from tradingagents.signals.contract import SignalAction
from tradingagents.tjr.frames import IntradaySeries, session_close_utc, session_open_utc
from tradingagents.tjr.rules import (
    STAGE_IN_ZONE,
    STAGE_READY,
    STAGE_SWEPT,
    TJR_ENTRY_WINDOW_CLOSED,
    TJR_INVALID_GEOMETRY,
    TJR_NO_1M_ENTRY,
    TJR_NO_5M_CONFIRMATION,
    TJR_NO_SWEEP,
    TJR_OUTSIDE_SESSION,
    TJR_READY,
    TJR_SHORT_SETUP_LONG_ONLY,
    evaluate,
)
from tradingagents.tjr.strategy import TjrStrategy

pytestmark = pytest.mark.unit


def _as_of(series: IntradaySeries, index: int) -> pd.Timestamp:
    """The instant the bar at ``index`` closed."""
    return series.frame.index[index] + pd.Timedelta(minutes=series.bar_minutes)


def _first_ready(five, one, *, allow_short: bool = False) -> int | None:
    """The first bar index at which the chain completes, or ``None``.

    Computed by walking the session, which is what the engine does; the tests then
    assert the *properties* of that bar rather than hard-coding an index that a
    fixture tweak would silently invalidate.
    """
    for index in range(len(five.frame)):
        outcome = evaluate(
            symbol="X", session=SESSION, as_of=_as_of(five, index),
            series_5m=five, series_1m=one, allow_short=allow_short,
        )
        if outcome.reason_code == TJR_READY:
            return index
    return None


def test_chain_completes_only_after_the_sweep_and_a_retrace():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    ready = _first_ready(five, one)
    assert ready is not None
    # The chain cannot complete before the sweep bar, which is at index 10.
    assert ready > 10
    # Every bar before it is short of a completed setup, and every bar from it on
    # stays complete (the sweep, once found, remains the most recent one).
    for index in range(11, ready):
        outcome = evaluate(
            symbol="X", session=SESSION, as_of=_as_of(five, index), series_5m=five, series_1m=one,
        )
        assert outcome.reason_code != TJR_READY
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, ready), series_5m=five, series_1m=one,
    )
    assert outcome.stage == STAGE_READY
    assert outcome.side == "long"
    assert outcome.entry is not None and outcome.stop is not None and outcome.target is not None
    # The stop sits below the swept low; the target is above the entry.
    assert outcome.stop < 97.9
    assert outcome.target > outcome.entry


def test_chain_stops_at_the_sweep_when_there_is_no_5m_confirmation():
    # Only the sweep exists; the bars after it do nothing structural.
    bars = BULLISH_BARS[:11] + [
        (99.9, 100.0, 99.85, 99.95),
        (99.95, 100.0, 99.9, 99.95),
        (99.95, 100.0, 99.9, 99.95),
        (99.95, 100.0, 99.9, 99.95),
    ]
    five, one = series_pair("X", SESSION, bars)
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, len(bars) - 1),
        series_5m=five, series_1m=one,
    )
    assert outcome.stage == STAGE_SWEPT
    assert outcome.reason_code == TJR_NO_5M_CONFIRMATION


def test_chain_reports_no_sweep_when_nothing_is_taken_out():
    bars = [
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 100.3, 100.0, 100.2),
        (100.2, 100.4, 100.1, 100.3),
        (100.3, 100.5, 100.2, 100.4),
        (100.4, 100.6, 100.3, 100.5),
    ]
    five, one = series_pair("X", SESSION, bars)
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, len(bars) - 1),
        series_5m=five, series_1m=one,
    )
    assert outcome.reason_code == TJR_NO_SWEEP


def test_chain_stops_in_the_zone_without_a_1m_trigger():
    # The same bullish session, but the 1m series stops at the retrace bar, so no 1m
    # bar prints a fresh break inside the zone.
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    stalled = IntradaySeries("X", "1m", one.frame[one.frame.index <= _as_of(five, 12)])
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, 15),
        series_5m=five, series_1m=stalled,
    )
    assert outcome.stage == STAGE_IN_ZONE
    assert outcome.reason_code == TJR_NO_1M_ENTRY


def test_a_complete_bearish_setup_is_declined_by_a_long_only_book():
    five, one = series_pair("X", SESSION, BEARISH_BARS)
    ready = _first_ready(five, one, allow_short=True)
    assert ready is not None
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, ready),
        series_5m=five, series_1m=one, allow_short=False,
    )
    assert outcome.stage == STAGE_READY
    assert outcome.reason_code == TJR_SHORT_SETUP_LONG_ONLY
    assert outcome.side is None
    assert not outcome.is_ready


def test_the_same_chain_would_take_the_short_when_shorts_are_allowed():
    five, one = series_pair("X", SESSION, BEARISH_BARS)
    ready = _first_ready(five, one, allow_short=True)
    assert ready is not None
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, ready),
        series_5m=five, series_1m=one, allow_short=True,
    )
    assert outcome.side == "short"
    assert outcome.reason_code == TJR_READY
    assert outcome.stop > 102.1  # above the swept high
    assert outcome.target < outcome.entry


def test_entry_window_closes_before_the_bell():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    late = session_close_utc(SESSION) - pd.Timedelta(minutes=5)
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=late, series_5m=five, series_1m=one,
    )
    assert outcome.reason_code == TJR_ENTRY_WINDOW_CLOSED


def test_no_bars_yet_is_reported_as_outside_the_session():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    before_open = session_open_utc(SESSION) - pd.Timedelta(minutes=1)
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=before_open, series_5m=five, series_1m=one,
    )
    assert outcome.reason_code == TJR_OUTSIDE_SESSION


def test_evaluation_is_deterministic_for_the_same_instant():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    instant = _as_of(five, _first_ready(five, one))
    first = evaluate(symbol="X", session=SESSION, as_of=instant, series_5m=five, series_1m=one)
    second = evaluate(symbol="X", session=SESSION, as_of=instant, series_5m=five, series_1m=one)
    assert first.to_dict() == second.to_dict()


def test_strategy_maps_a_ready_setup_to_a_buy_with_levels():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    strategy = TjrStrategy(one_minute={"X": one}, five_minute={"X": five})
    signal, outcome = strategy.signal("X", SESSION, _as_of(five, _first_ready(five, one)))
    assert outcome.reason_code == TJR_READY
    assert signal.action is SignalAction.BUY
    assert signal.entry_price is not None
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    # The strategy requests no size: sizing is the risk engine's job.
    assert signal.position_size is None


def test_strategy_maps_an_incomplete_chain_to_a_hold_with_a_reason():
    bars = BULLISH_BARS[:8]
    five, one = series_pair("X", SESSION, bars)
    strategy = TjrStrategy(one_minute={"X": one}, five_minute={"X": five})
    signal, outcome = strategy.signal("X", SESSION, _as_of(five, len(bars) - 1))
    assert signal.action is SignalAction.HOLD
    assert outcome.reason_code in signal.reasoning


def test_no_look_ahead_the_decision_never_reads_a_later_bar():
    # Truncating the data at the decision instant must not change the decision.
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    instant = _as_of(five, _first_ready(five, one))
    full = evaluate(symbol="X", session=SESSION, as_of=instant, series_5m=five, series_1m=one)

    cut = five.frame[five.frame.index <= instant - pd.Timedelta(minutes=5)]
    cut_1m = one.frame[one.frame.index <= instant - pd.Timedelta(minutes=1)]
    truncated = evaluate(
        symbol="X", session=SESSION, as_of=instant,
        series_5m=IntradaySeries("X", "5m", cut),
        series_1m=IntradaySeries("X", "1m", cut_1m),
    )
    assert full.to_dict() == truncated.to_dict()


def test_a_stop_on_the_wrong_side_of_the_entry_is_refused_not_traded(monkeypatch):
    """A long whose stop is above the entry must never be reported READY.

    Real bars do this: price makes a new low *after* the sweep, so the sweep's own
    extreme ends up above the entry the 1m trigger closed at. ``risk = abs(entry -
    stop)`` would hide the sign and report a clean R:R, and the contract would then
    refuse the BUY at construction. The strategy has to refuse it first, with a
    reason code, so a run never dies on a signal it should have declined.

    The condition is forced at the boundary (a real deep-retrace session reaches it,
    as the live ten-symbol run showed) so the guard is pinned without a fragile
    hand-built chain.
    """
    import tradingagents.tjr.rules as rules

    five, one = series_pair("X", SESSION, BULLISH_BARS)
    ready_index = _first_ready(five, one)
    assert ready_index is not None  # the chain does complete under the real rules

    def flipped_target(side, *, entry, stop, levels, settings, formed_by, impulse):
        # Force the geometry the guard exists for: a "long" whose stop sits above
        # its entry. The target is a valid reward beyond the entry, so only the
        # stop's side can make the setup invalid.
        return entry + 2.0 * abs(entry - stop) + 1.0

    monkeypatch.setattr(
        rules, "_levels_and_stop", lambda frame, sweep, settings: (10_000.0, 10_000.0)
    )
    monkeypatch.setattr(rules, "_target", flipped_target)
    outcome = evaluate(
        symbol="X", session=SESSION, as_of=_as_of(five, ready_index),
        series_5m=five, series_1m=one,
    )
    assert outcome.reason_code == TJR_INVALID_GEOMETRY
    assert outcome.side is None
    assert not outcome.is_ready
