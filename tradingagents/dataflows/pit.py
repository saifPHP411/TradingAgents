"""Point-in-time integrity: what each data source can and cannot guarantee.

The guards in :mod:`tradingagents.dataflows.date_window` clamp the dates a tool
asks a vendor for. That is necessary and not sufficient: it prevents a tool from
*requesting* the future, but it cannot stop a vendor from *returning* it. A news
search that ranks by relevance can hand back an article written after the
analysis date, and a fundamentals endpoint that serves "as reported today"
already contains every later restatement.

So this module states, per source, which of the two halves is guaranteed:

- ``REQUEST_BOUNDED`` — the tool will not ask for data later than the run's date,
  and the source is served as it was at that date. This is what "point-in-time"
  should mean, and it is the only level that makes a backtest trustworthy.
- ``WINDOW_FILTERED`` — the tool bounds the request, and dated items returned are
  then filtered to the window. Items the source cannot date are withheld from a
  historical run rather than assumed to be old. This is honest but weaker: it
  depends on the vendor dating its items correctly.
- ``NOT_PIT_SAFE`` — the source has no way to serve a historical view, so a run
  before today sees present-day content. Documented, not claimed.

Nothing here is asserted without evidence: each level cites the guard that
implements it, and the tests check that the guard exists and that the
classification matches the code. A source that gains a snapshot endpoint can be
promoted in one place.

The intended consumer is a report: a backtest that used a ``NOT_PIT_SAFE`` source
should say so, so its results can be discounted accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PITLevel(Enum):
    """How far a source's point-in-time integrity goes."""

    REQUEST_BOUNDED = "REQUEST_BOUNDED"
    WINDOW_FILTERED = "WINDOW_FILTERED"
    NOT_PIT_SAFE = "NOT_PIT_SAFE"


@dataclass(frozen=True)
class PITGuarantee:
    """One source's guarantee, with the guard that implements it and its caveat."""

    source: str
    level: PITLevel
    guard: str
    """The function or mechanism enforcing it, so the claim is checkable."""
    note: str
    """What the level does not cover, in plain terms."""

    @property
    def safe_for_backtest(self) -> bool:
        """Whether a historical run can treat this source's data as it stood then.

        Only ``REQUEST_BOUNDED`` qualifies. ``WINDOW_FILTERED`` data is filtered
        but its dating is the vendor's word, so a backtest using it should report
        the caveat rather than assume the result is clean.
        """
        return self.level is PITLevel.REQUEST_BOUNDED


def _guarantees() -> dict[str, PITGuarantee]:
    """The registry. One entry per source the router can reach."""
    return {
        "yahoo_prices": PITGuarantee(
            source="yahoo_prices",
            level=PITLevel.REQUEST_BOUNDED,
            guard="agents.tools.get_stock_data -> date_window.as_of_window",
            note=(
                "The agent tool clamps the requested window's end to the run's date before the "
                "vendor is called, so a bar after it is never returned. The clamp is in the tool "
                "layer, not the vendor: a direct call to the vendor or the router bypasses it, "
                "which is why the backtester reads bars from MarketDataFeed rather than calling "
                "the vendor itself. Adjusted prices reflect later splits and dividends, so a "
                "historical price is the split-adjusted value rather than the price on the tape "
                "that day."
            ),
        ),
        "yahoo_indicators": PITGuarantee(
            source="yahoo_indicators",
            level=PITLevel.REQUEST_BOUNDED,
            guard="agents.tools.get_indicators -> date_window.as_of_window",
            note=(
                "An indicator is computed from the price window, which is itself clamped to the "
                "run's date, so no future bar enters the calculation."
            ),
        ),
        "sec_edgar": PITGuarantee(
            source="sec_edgar",
            level=PITLevel.WINDOW_FILTERED,
            guard="date_window.withhold_undated_statements",
            note=(
                "EDGAR serves US filings as filed, which is the right shape for a backtest. A "
                "statement whose filing date the vendor does not expose is withheld from a "
                "historical run rather than assumed to predate it, so a historical run may see "
                "less than was available rather than more. Restatements are served as filed, "
                "which is the point, but a filing dated by acceptance time can be a day late."
            ),
        ),
        "fred": PITGuarantee(
            source="fred",
            level=PITLevel.REQUEST_BOUNDED,
            guard="fred.get_macro_data (observation_end + vintage pinning)",
            note=(
                "Observations are requested up to the run's date, clamped to FRED's own today, "
                "and a series with a vintage is pinned to it, so a revision published later does "
                "not appear in an earlier run. Publication lags mean a run may see a release "
                "before its figures were public in the ordinary case; the clamp bounds the "
                "window but does not model each series' release calendar."
            ),
        ),
        "yahoo_news": PITGuarantee(
            source="yahoo_news",
            level=PITLevel.WINDOW_FILTERED,
            guard="agents.tools.get_news -> date_window.as_of_window + in_window + coverage_gap",
            note=(
                "Yahoo news search returns its latest items whatever window is asked for. Dated "
                "items are filtered to the window and an undated item is withheld from a "
                "historical run. A window the feed never observed is reported as unavailable "
                "rather than empty, so an absence is not invented."
            ),
        ),
        "reddit": PITGuarantee(
            source="reddit",
            level=PITLevel.WINDOW_FILTERED,
            guard="date_window.in_window + reddit._within_window + date_window.coverage_gap",
            note=(
                "The search API serves recent posts, so a window older than its lookback is "
                "reported as unobserved. Posts are filtered on their own created_utc, and an "
                "undated post is withheld from a historical run."
            ),
        ),
        "stocktwits": PITGuarantee(
            source="stocktwits",
            level=PITLevel.WINDOW_FILTERED,
            guard="date_window.in_window + stocktwits._within_window + date_window.coverage_gap",
            note=(
                "The public feed serves the latest messages, so only recent windows are "
                "observed and an older one is reported as unavailable. Messages are filtered on "
                "their created_at."
            ),
        ),
        "yahoo_profile": PITGuarantee(
            source="yahoo_profile",
            level=PITLevel.NOT_PIT_SAFE,
            guard="date_window.withhold_live_profile",
            note=(
                "Company profile fields (sector, industry, name) are present-day snapshots with "
                "no historical view, so they are withheld entirely from a historical run rather "
                "than served as if they applied then. Companies are reclassified over time, so "
                "serving them would put post-decision information into the analysis. The "
                "withholding is the mitigation, not a point-in-time guarantee."
            ),
        ),
        "insider_transactions": PITGuarantee(
            source="insider_transactions",
            level=PITLevel.NOT_PIT_SAFE,
            guard="date_window.withhold_undisclosed_trades",
            note=(
                "The vendor does not date insider transactions by public disclosure, so a "
                "historical run cannot tell which trades were known on its date. They are "
                "withheld from a historical run rather than served as if disclosed."
            ),
        ),
        "polymarket": PITGuarantee(
            source="polymarket",
            level=PITLevel.NOT_PIT_SAFE,
            guard="none",
            note=(
                "Prediction-market prices are served as of now with no historical endpoint, so a "
                "backtest must not use them. There is no guard because there is nothing to "
                "clamp: the source has no date parameter to bound."
            ),
        ),
    }


PIT_GUARANTEES: dict[str, PITGuarantee] = _guarantees()


def get_guarantee(source: str) -> PITGuarantee:
    """The guarantee for ``source``, or raise ``KeyError`` naming what is known."""
    try:
        return PIT_GUARANTEES[source]
    except KeyError:
        known = ", ".join(sorted(PIT_GUARANTEES))
        raise KeyError(f"no point-in-time guarantee is recorded for {source!r}; known: {known}") from None


def unsafe_sources(sources) -> list[str]:
    """Those of ``sources`` whose data a historical run must not treat as it stood then.

    Used by a report: a backtest that touched one of these should say so, so its
    result is discounted rather than trusted.
    """
    return [s for s in sources if not get_guarantee(s).safe_for_backtest]


def describe(sources=None) -> str:
    """A plain-text table of the guarantees, for a report or a CLI."""
    chosen = sources if sources is not None else sorted(PIT_GUARANTEES)
    lines = ["Point-in-time integrity by source:", ""]
    for source in chosen:
        guarantee = get_guarantee(source)
        flag = "ok  " if guarantee.safe_for_backtest else "weak"
        lines.append(f"  [{flag}] {source:<24} {guarantee.level.value}")
    unsafe = unsafe_sources(chosen)
    if unsafe:
        lines += ["", f"Not safe for backtest: {', '.join(unsafe)}"]
    return "\n".join(lines)


def report_for_run(sources) -> dict:
    """A JSON-safe summary of a run's sources, for its record."""
    chosen = list(sources)
    return {
        "sources": chosen,
        "safe_for_backtest": [s for s in chosen if get_guarantee(s).safe_for_backtest],
        "unsafe_for_backtest": unsafe_sources(chosen),
        "levels": {s: get_guarantee(s).level.value for s in chosen},
        "notes": {s: get_guarantee(s).note for s in chosen},
    }


class SnapshotSource(Protocol):
    """A source that can serve data exactly as it stood at a past date.

    The seam for promoting a ``WINDOW_FILTERED`` or ``NOT_PIT_SAFE`` source
    without changing its callers. A vendor that adds a dated archive, or a local
    store of daily captures, implements this and is registered in
    :data:`SNAPSHOT_SOURCES`; the guarantee is then raised to
    ``REQUEST_BOUNDED`` and nothing else has to move.

    ``snapshot`` must not consult the live API: a snapshot is only a snapshot if
    it was captured at the time, and a method that fetches now and relabels the
    result would silently restore the look-ahead it is meant to remove.
    """

    source: str

    def snapshot(self, *, symbol: str, as_of_date: str, **kwargs) -> str:
        """Data for ``symbol`` as it was available on ``as_of_date``."""
        ...


SNAPSHOT_SOURCES: dict[str, SnapshotSource] = {}
"""Registered snapshot-capable sources, keyed by the name used in the registry.

Empty, because none of the current vendors offers a dated archive. A caller that
has one registers it here and the guarantee below is raised for that source.
"""


def register_snapshot_source(source: SnapshotSource) -> None:
    """Register a snapshot-capable implementation, promoting its guarantee.

    Raises ``KeyError`` for a source the registry does not know, so a typo cannot
    quietly register a capability nothing consults.
    """
    if source.source not in PIT_GUARANTEES:
        raise KeyError(f"{source.source!r} is not a known source; add it to the registry first")
    SNAPSHOT_SOURCES[source.source] = source


def effective_level(source: str) -> PITLevel:
    """The source's level, raised to ``REQUEST_BOUNDED`` when a snapshot exists.

    This is what a report should read, so registering a snapshot source takes
    effect everywhere without a second edit.
    """
    guarantee = get_guarantee(source)
    if source in SNAPSHOT_SOURCES:
        return PITLevel.REQUEST_BOUNDED
    return guarantee.level
