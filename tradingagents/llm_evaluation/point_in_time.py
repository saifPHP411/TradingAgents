"""Point-in-time readiness of the agent graph: what a historical run may trust.

The reference baseline and the decision-replay layer both answer "what did the
strategy decide". This module answers a prior question: *for a run dated in the
past, is every input the agents would read actually available as of that date?*

Nothing here is asserted by hand. The dependency matrix is derived from the code
that builds the graph (:mod:`tradingagents.graph.analyst_execution`), the code
that maps a tool to a vendor (:mod:`tradingagents.dataflows.router`) and the
per-source guarantees already recorded with their guards
(:mod:`tradingagents.dataflows.pit`). A source whose classification changes is
changed in one place — ``pit.py`` — and this module follows. If the code and the
matrix ever disagree, a test fails rather than the matrix quietly going stale.

The output is a *readiness verdict*, not a result. It states which agents can be
run over a historical window with trustworthy inputs and which cannot, so a
restricted experiment is labelled as restricted and a full-graph historical run
is refused rather than faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingagents.dataflows.pit import PIT_GUARANTEES, PITLevel, get_guarantee
from tradingagents.dataflows.router import (
    OPTIONAL_CATEGORIES,
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
)
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS


class Classification(Enum):
    """The five-way vocabulary the audit reports, from the registry's three levels.

    The registry answers "what does the guard enforce"; this answers "may a
    historical decision use it". The mapping is conservative: only a source
    served as it stood at the run's date is ``PIT_VERIFIED``; a filtered source
    depends on the vendor dating its items correctly, so it is only
    ``PIT_PARTIALLY_VERIFIED``; a live-only source is ``CURRENT_ONLY``.
    """

    PIT_VERIFIED = "PIT_VERIFIED"
    PIT_PARTIALLY_VERIFIED = "PIT_PARTIALLY_VERIFIED"
    PIT_UNVERIFIED = "PIT_UNVERIFIED"
    CURRENT_ONLY = "CURRENT_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


# The registry level -> the audit classification. ``PIT_UNVERIFIED`` and
# ``UNAVAILABLE`` are not produced by the current registry; they are the buckets a
# future source falls into (historical data whose availability cannot be
# established; data that cannot be obtained at all), kept here so the vocabulary
# is closed and a caller never has to invent a sixth label.
_LEVEL_TO_CLASSIFICATION = {
    PITLevel.REQUEST_BOUNDED: Classification.PIT_VERIFIED,
    PITLevel.WINDOW_FILTERED: Classification.PIT_PARTIALLY_VERIFIED,
    PITLevel.NOT_PIT_SAFE: Classification.CURRENT_ONLY,
}

# Tool name -> the registry sources it can reach. Derived from the router's own
# tool -> vendor map (one entry per vendor the tool can be routed to), so adding a
# vendor to a tool is caught by the consistency test, not by memory.
_TOOL_SOURCES: dict[str, tuple[str, ...]] = {
    "get_stock_data": ("yahoo_prices",),
    "get_indicators": ("yahoo_indicators",),
    "get_verified_market_snapshot": ("yahoo_prices", "yahoo_indicators"),
    "get_fundamentals": ("yahoo_profile",),
    "get_balance_sheet": ("sec_edgar",),
    "get_cashflow": ("sec_edgar",),
    "get_income_statement": ("sec_edgar",),
    "get_news": ("yahoo_news",),
    "get_global_news": ("yahoo_news",),
    "get_insider_transactions": ("insider_transactions",),
    "get_macro_indicators": ("fred",),
    "get_prediction_markets": ("polymarket",),
}

# The sentiment analyst fetches its sources before calling the model, so its data
# dependencies are not visible in ``ANALYST_NODE_SPECS[*].tools`` (which is empty).
# They are declared here, mirroring ``agents/analysts/sentiment_analyst.py``.
_SENTIMENT_PREFETCHED_SOURCES: tuple[str, ...] = ("yahoo_news", "stocktwits", "reddit")

# A source reachable only through an optional enrichment category. A vendor
# failure here degrades to a sentinel rather than aborting the run, so the source
# is not required for the graph to function.
_OPTIONAL_SOURCES: frozenset[str] = frozenset(
    source
    for category in OPTIONAL_CATEGORIES
    for tool in TOOLS_CATEGORIES[category]["tools"]
    for source in _TOOL_SOURCES.get(tool, ())
)


@dataclass(frozen=True)
class SourceFacts:
    """What a historical run needs to know about one data source."""

    source: str
    classification: Classification
    guard: str
    """The mechanism enforcing the guarantee, so the claim is checkable."""
    historical: bool
    """Whether the source serves a historical view at all."""
    has_availability_timestamp: bool
    """Whether each returned item carries a time it became public."""
    revisions_possible: bool
    """Whether later data can restate an earlier value."""
    required_for_graph: bool
    """Whether the graph cannot run without it (a core category)."""
    disables_cleanly: bool
    """Whether it can be dropped without changing what the experiment measures."""
    internet_access: bool
    """Whether the source reaches the network / current-date information."""
    note: str


def tool_vendors() -> dict[str, tuple[str, ...]]:
    """Tool name -> the vendors the router can route it to, from ``VENDOR_METHODS``."""
    return {tool: tuple(sorted(vendors)) for tool, vendors in VENDOR_METHODS.items()}


def _facts(source: str) -> SourceFacts:
    guarantee = get_guarantee(source)
    classification = _LEVEL_TO_CLASSIFICATION[guarantee.level]
    return SourceFacts(
        source=source,
        classification=classification,
        guard=guarantee.guard,
        historical=classification is not Classification.CURRENT_ONLY,
        has_availability_timestamp=classification is not Classification.CURRENT_ONLY,
        revisions_possible=source in ("sec_edgar", "fred"),
        required_for_graph=source not in _OPTIONAL_SOURCES,
        disables_cleanly=source in _OPTIONAL_SOURCES,
        internet_access=True,
        note=guarantee.note,
    )


def source_facts() -> dict[str, SourceFacts]:
    """Every registry source with its audit facts, derived from the registry."""
    return {source: _facts(source) for source in PIT_GUARANTEES}


def agent_sources() -> dict[str, tuple[str, ...]]:
    """Analyst key -> the sources it consumes, derived from the graph spec.

    Tools come from ``ANALYST_NODE_SPECS``; the sentiment analyst's pre-fetched
    feeds are added because its tool tuple is deliberately empty.
    """
    out: dict[str, tuple[str, ...]] = {}
    for key, spec in ANALYST_NODE_SPECS.items():
        sources: list[str] = []
        for tool in spec.tools:
            for source in _TOOL_SOURCES.get(tool.name, ()):
                if source not in sources:
                    sources.append(source)
        if key == "social":
            for source in _SENTIMENT_PREFETCHED_SOURCES:
                if source not in sources:
                    sources.append(source)
        out[key] = tuple(sources)
    return out


def agent_dependency_matrix() -> dict[str, dict]:
    """The agent-by-agent dependency matrix, JSON-safe and code-derived."""
    facts = source_facts()
    vendors = tool_vendors()
    matrix: dict[str, dict] = {}
    for key, sources in agent_sources().items():
        spec = ANALYST_NODE_SPECS[key]
        matrix[key] = {
            "agent": spec.agent_node,
            "report_key": spec.report_key,
            "tools": [tool.name for tool in spec.tools],
            "tool_vendors": {
                tool.name: list(vendors.get(tool.name, ())) for tool in spec.tools
            },
            "sources": list(sources),
            "source_details": {
                source: {
                    "classification": facts[source].classification.value,
                    "historical": facts[source].historical,
                    "has_availability_timestamp": facts[source].has_availability_timestamp,
                    "revisions_possible": facts[source].revisions_possible,
                    "required_for_graph": facts[source].required_for_graph,
                    "disables_cleanly": facts[source].disables_cleanly,
                    "internet_access": facts[source].internet_access,
                }
                for source in sources
            },
        }
    return matrix


@dataclass(frozen=True)
class ReadinessVerdict:
    """Whether a declared experiment scope can run on trustworthy inputs."""

    scope: str
    ready: bool
    verified_sources: tuple[str, ...]
    weak_sources: tuple[str, ...]
    current_only_sources: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "ready": self.ready,
            "verified_sources": list(self.verified_sources),
            "weak_sources": list(self.weak_sources),
            "current_only_sources": list(self.current_only_sources),
            "blockers": list(self.blockers),
        }


def graph_readiness(scope: str = "full") -> ReadinessVerdict:
    """Whether the graph can run over a historical window in the given scope.

    ``full`` is every agent the graph is built from. ``price_only`` is the
    market analyst alone, which reads prices and indicators — the two sources the
    frozen dataset and the request-bounded vendor both serve as they stood at the
    run's date.

    ``full`` is never ready for a historical run today: the news, sentiment and
    fundamentals analysts read sources the registry records as window-filtered or
    current-only. That is a property of the sources, not a bug to work around.
    """
    if scope not in ("full", "price_only"):
        raise ValueError("scope must be 'full' or 'price_only'")

    facts = source_facts()
    if scope == "price_only":
        chosen = agent_sources()["market"]
    else:
        chosen = tuple(
            dict.fromkeys(
                source for sources in agent_sources().values() for source in sources
            )
        )

    verified = tuple(s for s in chosen if facts[s].classification is Classification.PIT_VERIFIED)
    weak = tuple(
        s for s in chosen if facts[s].classification is Classification.PIT_PARTIALLY_VERIFIED
    )
    current_only = tuple(
        s for s in chosen if facts[s].classification is Classification.CURRENT_ONLY
    )

    blockers: list[str] = []
    if scope == "full":
        for source in weak:
            blockers.append(
                f"{source} is PIT_PARTIALLY_VERIFIED: its dating is the vendor's word "
                f"(guard: {facts[source].guard}), so a historical run depends on it "
                "being correct"
            )
        for source in current_only:
            blockers.append(
                f"{source} is CURRENT_ONLY: it serves present-day values with no "
                "historical view, so it must be withheld from a historical run"
            )
        blockers.append(
            "a full-graph historical run cannot be point-in-time safe while the "
            "news/sentiment/fundamentals sources above are weak or current-only; "
            "run price_only (restricted) or supply verified snapshots"
        )

    return ReadinessVerdict(
        scope=scope,
        ready=not blockers,
        verified_sources=verified,
        weak_sources=weak,
        current_only_sources=current_only,
        blockers=tuple(blockers),
    )


def point_in_time_report(scope: str = "full") -> dict:
    """The JSON-safe audit block: matrix, classifications and readiness verdict."""
    facts = source_facts()
    return {
        "scope": scope,
        "agent_dependency_matrix": agent_dependency_matrix(),
        "source_classifications": {
            source: {
                "classification": fact.classification.value,
                "guard": fact.guard,
                "historical": fact.historical,
                "has_availability_timestamp": fact.has_availability_timestamp,
                "revisions_possible": fact.revisions_possible,
                "required_for_graph": fact.required_for_graph,
                "disables_cleanly": fact.disables_cleanly,
                "internet_access": fact.internet_access,
                "note": fact.note,
            }
            for source, fact in facts.items()
        },
        "readiness": graph_readiness(scope).to_dict(),
        "verified_source_count": sum(
            1 for f in facts.values() if f.classification is Classification.PIT_VERIFIED
        ),
        "weak_source_count": sum(
            1
            for f in facts.values()
            if f.classification is Classification.PIT_PARTIALLY_VERIFIED
        ),
        "current_only_source_count": sum(
            1 for f in facts.values() if f.classification is Classification.CURRENT_ONLY
        ),
    }


__all__ = [
    "Classification",
    "ReadinessVerdict",
    "SourceFacts",
    "agent_dependency_matrix",
    "agent_sources",
    "graph_readiness",
    "point_in_time_report",
    "source_facts",
    "tool_vendors",
]
