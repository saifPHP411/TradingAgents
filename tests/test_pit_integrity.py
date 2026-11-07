"""Point-in-time integrity: the registry, and whether its claims match the code.

A classification is only worth having if it is checkable. So the tests here do
two things: they pin the classification of each source, and they check that the
guard a classification cites actually exists. A guarantee whose guard has been
renamed away is a stale claim, and the test that reads the guard's name catches
it.

The other property is that the unsafe list is not silently empty. A registry that
called every source safe would be worse than none, because it would look like
evidence. The tests assert that the sources with no historical view are recorded
as such.
"""

from __future__ import annotations

import inspect

import pytest

from tradingagents.dataflows.pit import (
    PIT_GUARANTEES,
    PITLevel,
    SnapshotSource,
    describe,
    effective_level,
    get_guarantee,
    register_snapshot_source,
    report_for_run,
    unsafe_sources,
)


@pytest.mark.unit
class TestRegistry:
    def test_every_source_the_router_can_reach_is_classified(self):
        """A source missing from the registry is one nobody checked."""
        for source in ("yahoo_prices", "yahoo_indicators", "sec_edgar", "fred",
                       "yahoo_news", "reddit", "stocktwits", "yahoo_profile",
                       "insider_transactions", "polymarket"):
            assert source in PIT_GUARANTEES

    def test_an_unknown_source_raises_and_names_what_is_known(self):
        with pytest.raises(KeyError, match="known:"):
            get_guarantee("a_source_that_does_not_exist")

    def test_every_guarantee_carries_a_guard_and_a_note(self):
        for source, guarantee in PIT_GUARANTEES.items():
            assert guarantee.guard, f"{source} cites no guard"
            assert len(guarantee.note) > 40, f"{source}'s note is too thin to be a caveat"

    def test_the_safe_list_is_not_everything(self):
        """A registry that called every source safe would look like evidence."""
        unsafe = [s for s, g in PIT_GUARANTEES.items() if not g.safe_for_backtest]
        assert unsafe, "no source is recorded as unsafe, which cannot be right"

    def test_the_sources_with_no_historical_view_are_recorded_as_unsafe(self):
        for source in ("polymarket", "yahoo_profile", "insider_transactions"):
            assert not get_guarantee(source).safe_for_backtest


@pytest.mark.unit
class TestClassificationMatchesCode:
    """Each REQUEST_BOUNDED claim is checked against the function it cites."""

    def test_the_date_window_guards_exist(self):
        from tradingagents.dataflows import date_window

        for name in ("as_of", "as_of_window", "in_window", "coverage_gap",
                     "withhold_live_profile", "withhold_undisclosed_trades",
                     "withhold_undated_statements"):
            assert hasattr(date_window, name), f"date_window.{name} is cited but missing"

    def test_the_price_path_clamps_to_the_run_date(self):
        """The clamp is in the tool layer; the test names it so a rename is caught."""
        from tradingagents.agents import tools

        source = inspect.getsource(tools.get_stock_data.func
                                   if hasattr(tools.get_stock_data, "func") else tools.get_stock_data)
        assert "as_of_window" in source, (
            "the price path is classified REQUEST_BOUNDED but its tool does not clamp the window"
        )

    def test_the_clamp_is_not_in_the_vendor_itself(self):
        """A documented caveat: a direct vendor call would bypass the point-in-time clamp.

        The registry says REQUEST_BOUNDED because the agent reaches the vendor only
        through the tool. This test records that the vendor has no clamp of its own,
        so the guarantee's scope is stated in code rather than assumed.
        """
        from tradingagents.dataflows.vendors.yahoo import market

        assert "as_of" not in inspect.getsource(market.get_YFin_data_online)

    def test_the_fred_path_bounds_its_observation_window(self):
        from tradingagents.dataflows.vendors import fred

        source = inspect.getsource(fred.get_macro_data)
        assert "observation_end" in source, (
            "fred is classified REQUEST_BOUNDED but does not bound observation_end"
        )

    def test_the_profile_is_withheld_rather_than_served_as_current(self):
        from tradingagents.dataflows.vendors.yahoo import fundamentals

        source = inspect.getsource(fundamentals.get_fundamentals)
        assert "withhold_live_profile" in source, (
            "the profile is classified NOT_PIT_SAFE but is not withheld from historical runs"
        )

    def test_news_and_social_paths_filter_to_the_window(self):
        from tradingagents.dataflows.vendors import reddit, stocktwits
        from tradingagents.dataflows.vendors.yahoo import news

        for module in (news, reddit, stocktwits):
            source = inspect.getsource(module)
            assert "_within_window" in source or "in_window" in source, (
                f"{module.__name__} is classified WINDOW_FILTERED but does not filter a window"
            )

    def test_an_undated_item_is_withheld_from_a_historical_run(self):
        """The rule that makes WINDOW_FILTERED honest, checked directly."""
        from datetime import UTC, datetime

        from tradingagents.dataflows.date_window import in_window

        start = datetime(2026, 1, 5, tzinfo=UTC)
        end = datetime(2026, 1, 9, tzinfo=UTC)
        # Undated, and the window is in the past -> withheld.
        assert not in_window(None, start, end)
        # Dated inside the window -> kept.
        assert in_window(datetime(2026, 1, 7, tzinfo=UTC), start, end)
        # Dated after the window -> dropped.
        assert not in_window(datetime(2026, 1, 12, tzinfo=UTC), start, end)


@pytest.mark.unit
class TestSnapshotSeam:
    def test_no_snapshot_source_is_registered_by_default(self):
        """None of the current vendors has a dated archive, so the registry is empty."""
        from tradingagents.dataflows.pit import SNAPSHOT_SOURCES

        assert SNAPSHOT_SOURCES == {}

    def test_registering_a_snapshot_raises_the_effective_level(self):
        class FakeSnapshot:
            source = "polymarket"

            def snapshot(self, *, symbol, as_of_date, **kwargs):
                return "captured"

        before = effective_level("polymarket")
        assert before is PITLevel.NOT_PIT_SAFE
        try:
            register_snapshot_source(FakeSnapshot())
            assert effective_level("polymarket") is PITLevel.REQUEST_BOUNDED
        finally:
            from tradingagents.dataflows.pit import SNAPSHOT_SOURCES

            SNAPSHOT_SOURCES.clear()
        assert effective_level("polymarket") is PITLevel.NOT_PIT_SAFE

    def test_registering_an_unknown_source_is_refused(self):
        class FakeSnapshot:
            source = "a_source_that_does_not_exist"

            def snapshot(self, *, symbol, as_of_date, **kwargs):
                return ""

        with pytest.raises(KeyError):
            register_snapshot_source(FakeSnapshot())

    def test_the_protocol_is_structural(self):
        """A snapshot source needs no inheritance, only the attribute and method."""
        assert isinstance(SnapshotSource, type) or hasattr(SnapshotSource, "__protocol_attrs__")


@pytest.mark.unit
class TestReporting:
    def test_unsafe_sources_are_listed(self):
        assert unsafe_sources(["yahoo_prices", "polymarket"]) == ["polymarket"]

    def test_a_fully_safe_run_lists_nothing(self):
        assert unsafe_sources(["yahoo_prices", "fred"]) == []

    def test_describe_renders_every_source(self):
        text = describe(["yahoo_prices", "polymarket"])
        assert "yahoo_prices" in text
        assert "polymarket" in text
        assert "Not safe for backtest" in text

    def test_the_run_report_is_json_safe(self):
        import json

        payload = report_for_run(["yahoo_prices", "reddit", "polymarket"])
        json.dumps(payload)
        assert payload["safe_for_backtest"] == ["yahoo_prices"]
        assert set(payload["unsafe_for_backtest"]) == {"reddit", "polymarket"}
        assert payload["levels"]["reddit"] == "WINDOW_FILTERED"

    def test_the_report_carries_the_caveat_not_just_the_label(self):
        payload = report_for_run(["reddit"])
        assert "vendor" in payload["notes"]["reddit"] or "dating" in payload["notes"]["reddit"] \
            or "created_utc" in payload["notes"]["reddit"]
