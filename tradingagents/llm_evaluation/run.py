"""Assemble the LLM-strategy evaluation artifact. Run offline.

    python -m tradingagents.llm_evaluation.run                 # report or evaluate
    python -m tradingagents.llm_evaluation.run --records FILE   # replay captured decisions
    python -m tradingagents.llm_evaluation.run --out PATH       # choose the output path

With no credentials and no captured decisions, this reports ``LLM BASELINE
BLOCKED`` and the exact missing prerequisite. It never invents a number for the
LLM strategy. With a captured decision record set it replays those decisions
through the frozen engine and measures them.

The reference-context windows (the SMA baseline and buy-and-hold SPY over the
same periods) are also computed, so a future LLM result has a fixed comparison
produced by the same protocol. They are labelled as context, not LLM results.
"""

from __future__ import annotations

from .evaluate import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
