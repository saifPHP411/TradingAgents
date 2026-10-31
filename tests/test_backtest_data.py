"""The point-in-time feed: no path to a future bar, and no invented prices.

The look-ahead tests here are the point of the module. A backtest that can see
tomorrow is not conservative, it is fiction, and the failure mode is silent — the
numbers simply look better. So the feed is tested by trying to get future data
and asserting there is no way to.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from tradingagents.backtesting import Bar, BarDataError, MarketDataFeed, as_date


def _frame(dates, closes, *, opens=None, volume=None) -> pd.DataFrame:
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {
            "open": opens,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": volume if volume is not None else [1_000] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
CLOSES = [100.0, 101.0, 102.0, 103.0, 104.0]


@pytest.fixture
def feed() -> MarketDataFeed:
    return MarketDataFeed({"AAPL": _frame(DATES, CLOSES)})


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_an_empty_feed_is_rejected(self):
        with pytest.raises(BarDataError):
            MarketDataFeed({})

    def test_a_symbol_with_no_bars_is_rejected(self):
        with pytest.raises(BarDataError):
            MarketDataFeed({"AAPL": pd.DataFrame()})

    def test_missing_columns_are_rejected(self):
        frame = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=pd.to_datetime(["2026-01-05"]))
        with pytest.raises(BarDataError, match="missing"):
            MarketDataFeed({"AAPL": frame})

    def test_columns_are_case_insensitive(self):
        frame = pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5]},
            index=pd.to_datetime(["2026-01-05"]),
        )
        feed = MarketDataFeed({"AAPL": frame})
        bar = feed.bar("AAPL", "2026-01-05")
        assert bar is not None and bar.close == 10.5

    def test_an_unparseable_index_is_rejected(self):
        frame = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=["not-a-date"]
        )
        with pytest.raises(BarDataError):
            MarketDataFeed({"AAPL": frame})

    def test_rows_with_non_positive_prices_are_dropped(self):
        frame = _frame(DATES, [100.0, 0.0, 102.0, 103.0, 104.0])
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.bar("AAPL", "2026-01-06") is None

    def test_asking_for_an_unknown_symbol_raises(self, feed):
        with pytest.raises(BarDataError):
            feed.bar("MSFT", "2026-01-05")


@pytest.mark.unit
class TestBarValidation:
    def test_a_bar_with_high_below_low_is_rejected(self):
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", date="2026-01-05", open=10.0, high=9.0, low=11.0, close=10.0)

    def test_a_close_outside_the_range_is_rejected(self):
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", date="2026-01-05", open=10.0, high=11.0, low=9.0, close=12.0)

    def test_a_non_positive_price_is_rejected(self):
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", date="2026-01-05", open=0.0, high=11.0, low=9.0, close=10.0)


# ---------------------------------------------------------------------------
# Look-ahead
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoLookAhead:
    def test_history_stops_at_the_requested_date(self, feed):
        history = feed.history("AAPL", "2026-01-07")
        assert list(history.index.strftime("%Y-%m-%d")) == ["2026-01-05", "2026-01-06", "2026-01-07"]

    def test_history_never_contains_a_later_bar(self, feed):
        for on in DATES:
            history = feed.history("AAPL", on)
            assert all(d <= on for d in history.index.strftime("%Y-%m-%d"))

    def test_a_future_close_is_unreachable_through_the_feed(self, feed):
        """There is no method that returns a bar after the requested date."""
        assert feed.bar("AAPL", "2026-01-07").close == 102.0
        history = feed.history("AAPL", "2026-01-07")
        assert 103.0 not in set(history["close"])
        assert 104.0 not in set(history["close"])

    def test_history_before_the_first_bar_is_empty(self, feed):
        assert feed.history("AAPL", "2026-01-01").empty

    def test_the_calendar_can_be_bounded(self, feed):
        assert feed.calendar("2026-01-06", "2026-01-08") == ["2026-01-06", "2026-01-07", "2026-01-08"]

    def test_next_session_is_the_following_bar(self, feed):
        assert feed.next_session("2026-01-05") == "2026-01-06"

    def test_next_session_is_none_at_the_end(self, feed):
        assert feed.next_session("2026-01-09") is None

    def test_previous_session_is_the_preceding_bar(self, feed):
        assert feed.previous_session("2026-01-07") == "2026-01-06"

    def test_previous_session_is_none_at_the_start(self, feed):
        assert feed.previous_session("2026-01-05") is None


# ---------------------------------------------------------------------------
# Missing bars
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingBars:
    def test_a_missing_session_returns_none_not_a_neighbouring_price(self):
        """A weekend must be 'no bar', never Friday's price."""
        dates = ["2026-01-05", "2026-01-06", "2026-01-08"]
        feed = MarketDataFeed({"AAPL": _frame(dates, [100.0, 101.0, 103.0])})
        assert feed.bar("AAPL", "2026-01-07") is None

    def test_a_carried_forward_price_is_available_only_when_asked_for_explicitly(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-08"]
        feed = MarketDataFeed({"AAPL": _frame(dates, [100.0, 101.0, 103.0])})
        carried = feed.latest_bar_at_or_before("AAPL", "2026-01-07")
        assert carried is not None
        assert carried.date == "2026-01-06"
        assert carried.close == 101.0

    def test_the_calendar_is_the_union_of_symbols_bars(self):
        feed = MarketDataFeed({
            "AAPL": _frame(["2026-01-05", "2026-01-06"], [100.0, 101.0]),
            "MSFT": _frame(["2026-01-06", "2026-01-07"], [200.0, 201.0]),
        })
        assert feed.calendar() == ["2026-01-05", "2026-01-06", "2026-01-07"]

    def test_a_symbol_that_did_not_trade_that_session_has_no_bar(self):
        feed = MarketDataFeed({
            "AAPL": _frame(["2026-01-05", "2026-01-06"], [100.0, 101.0]),
            "MSFT": _frame(["2026-01-06", "2026-01-07"], [200.0, 201.0]),
        })
        assert feed.bar("AAPL", "2026-01-07") is None
        assert feed.bar("MSFT", "2026-01-05") is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalization:
    def test_a_timezone_aware_index_keeps_its_local_date(self):
        index = pd.to_datetime(["2026-01-05 00:00:00+09:00"])
        frame = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10]},
            index=index,
        )
        feed = MarketDataFeed({"7203.T": frame})
        assert feed.bar("7203.T", "2026-01-05") is not None

    def test_duplicate_dates_keep_the_last(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-05"])
        frame = pd.DataFrame(
            {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0]},
            index=index,
        )
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.bar("AAPL", "2026-01-05").close == 2.0

    def test_rows_are_sorted(self):
        index = pd.to_datetime(["2026-01-07", "2026-01-05", "2026-01-06"])
        frame = pd.DataFrame(
            {"open": [3.0, 1.0, 2.0], "high": [3.0, 1.0, 2.0], "low": [3.0, 1.0, 2.0],
             "close": [3.0, 1.0, 2.0]},
            index=index,
        )
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05", "2026-01-06", "2026-01-07"]

    def test_a_missing_volume_column_is_tolerated(self):
        frame = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.to_datetime(["2026-01-05"]),
        )
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.bar("AAPL", "2026-01-05").volume is None


@pytest.mark.unit
class TestHelpers:
    def test_as_date_normalizes(self):
        assert as_date("2026-1-5") == "2026-01-05"

    def test_as_date_rejects_nonsense(self):
        with pytest.raises(BarDataError):
            as_date("banana")


# ---------------------------------------------------------------------------
# Non-finite bars and the session-bounded view
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonFiniteBarsAreDroppedAtConstruction:
    """A non-finite price must be dropped when the feed is built, not raise later.

    Regression test for a real defect: ``_normalize`` kept a row whose price was
    ``inf`` because ``inf > 0`` is true, and ``Bar`` only rejected it when
    ``bar()`` was called — mid-backtest. A vendor that emits one inf would abort
    a run on an arbitrary later session rather than simply leaving that date with
    no bar. Dropping it at construction keeps the failure at the boundary and the
    run alive.
    """

    def test_an_infinite_price_row_is_dropped(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame.loc[pd.Timestamp("2026-01-06"), "close"] = float("inf")
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05"]
        assert feed.bar("AAPL", "2026-01-06") is None

    def test_a_nan_price_row_is_dropped(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame.loc[pd.Timestamp("2026-01-06"), "open"] = float("nan")
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05"]


@pytest.mark.unit
class TestInconsistentOhlcBarsAreDroppedAtConstruction:
    """A bar whose high is below its low must be dropped when the feed is built.

    Regression test for a real defect: ``_normalize`` dropped non-finite and
    non-positive rows but not OHLC-inconsistent ones, so a row with ``high < low``
    (or an open/close outside the range) survived construction and then made
    ``Bar`` raise when ``bar()`` was called — mid-backtest, on an arbitrary later
    session, aborting a run that was holding a position. A vendor emitting one bad
    row must cost the run that one date, not the whole run.
    """

    def test_a_high_below_low_row_is_dropped(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame.loc[pd.Timestamp("2026-01-06"), "high"] = 99.0
        frame.loc[pd.Timestamp("2026-01-06"), "low"] = 100.0
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05"]
        assert feed.bar("AAPL", "2026-01-06") is None

    def test_an_open_above_the_high_is_dropped(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame.loc[pd.Timestamp("2026-01-06"), "open"] = 200.0
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05"]

    def test_a_close_below_the_low_is_dropped(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame.loc[pd.Timestamp("2026-01-06"), "close"] = 1.0
        feed = MarketDataFeed({"AAPL": frame})
        assert feed.calendar() == ["2026-01-05"]

    def test_a_served_bar_never_raises_for_a_consistent_frame(self):
        """Every date the calendar reports must serve a valid ``Bar``."""
        feed = MarketDataFeed({"AAPL": _frame(DATES, CLOSES)})
        for session in feed.calendar():
            assert feed.bar("AAPL", session) is not None

    def test_a_frame_of_only_bad_rows_is_refused(self):
        frame = _frame(DATES[:2], [100.0, 101.0])
        frame["high"] = [1.0, 1.0]
        frame["low"] = [2.0, 2.0]
        with pytest.raises(BarDataError):
            MarketDataFeed({"AAPL": frame})


@pytest.mark.unit
class TestSessionBoundedView:
    """``as_of`` must hide every later bar, on every accessor.

    Regression test for a real defect: the strategy context handed over the raw
    feed, so ``context.feed.bar(symbol, tomorrow)`` returned tomorrow's bar even
    though ``context.history`` was safe. The guarantee has to hold for the object
    the strategy holds, not only for the helper methods it is expected to use.
    """

    def test_as_of_hides_later_bars(self, feed):
        view = feed.as_of("2026-01-07")
        assert view.calendar() == ["2026-01-05", "2026-01-06", "2026-01-07"]
        assert view.bar("AAPL", "2026-01-08") is None
        assert view.bar("AAPL", "2026-01-07") is not None

    def test_as_of_history_cannot_reach_past_the_boundary(self, feed):
        view = feed.as_of("2026-01-06")
        history = view.history("AAPL", "2026-01-09")
        assert [str(d.date()) for d in history.index] == ["2026-01-05", "2026-01-06"]

    def test_as_of_latest_bar_at_or_before_is_bounded(self, feed):
        view = feed.as_of("2026-01-06")
        bar = view.latest_bar_at_or_before("AAPL", "2026-01-09")
        assert bar.date == "2026-01-06"

    def test_as_of_the_last_session_returns_the_same_feed(self, feed):
        assert feed.as_of("2026-01-09") is feed

    def test_as_of_before_any_bar_is_an_error(self, feed):
        with pytest.raises(BarDataError):
            feed.as_of("2026-01-01")
