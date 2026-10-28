"""The official strategy baseline: a reproducible reference point, not a strategy.

This package establishes a fixed, versioned protocol and measures one
pre-registered deterministic reference strategy against a buy-and-hold benchmark
over a frozen dataset. It exists so a future strategy change can be compared
against a number that was produced the same way every time.

It is not the TradingAgents LLM strategy. That strategy is a non-deterministic
multi-agent workflow that needs model credentials; it cannot be replayed to a
single number, so it cannot be a baseline. See ``docs/STRATEGY_BASELINE.md``.

Nothing here reimplements the engine: it builds the existing ``MarketDataFeed``,
runs the existing ``PortfolioBacktester``, measures with the existing
``performance`` engine, and re-derives the headline numbers independently to
cross-check them.
"""

from tradingagents.baseline.data import BaselineDataError, load_feed, load_frames
from tradingagents.baseline.protocol import (
    DEFAULT_PROTOCOL,
    PROTOCOL_VERSION,
    BaselineProtocol,
)
from tradingagents.baseline.reference import SmaCrossover

__all__ = [
    "BaselineDataError",
    "BaselineProtocol",
    "DEFAULT_PROTOCOL",
    "PROTOCOL_VERSION",
    "SmaCrossover",
    "load_feed",
    "load_frames",
]
