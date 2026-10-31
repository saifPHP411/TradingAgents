"""The paper experiment's own protocol provenance.

The paper experiment is *not* the agent graph. It makes one bounded model call
over a point-in-time window, so labelling its records with the graph's
``prompt_version``/``graph_version`` would attest to prompts and a pipeline the
decision never used. That is the same class of defect as the data-blind prompt:
the record claiming an input that did not happen.

This module derives a paper-specific protocol from the frozen
:data:`~tradingagents.llm_evaluation.protocol.DEFAULT_PROTOCOL` — inheriting its
universe, costs, risk limits, holding period and capital so the two remain
directly comparable — and overrides only the two provenance fields that describe
*how* the decision was produced. The prompt version tracks the request shape in
``inference.py``; the graph version names the single-call seam.

It is a leaf module (it imports only the frozen protocol) so both ``config`` and
``inference`` can read it without a circular import.
"""

from __future__ import annotations

import dataclasses

from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL

PAPER_INFERENCE_PROMPT_VERSION = "paper-inference-3"
"""Bumped when the request shape changes.

``paper-inference-1`` was the data-blind prompt that could only elicit ``Hold``
and saw no bars. ``paper-inference-2`` rendered the raw window but left the model
to recompute trend, momentum and volatility itself. This version adds the
deterministic indicator block computed in code and asks for the factors behind
the call."""

PAPER_GRAPH_VERSION = "paper-single-call"
"""The paper seam is one model call, not the multi-agent graph. Naming it honestly
keeps a paper record from being read as a graph result."""

PAPER_PROTOCOL = dataclasses.replace(
    DEFAULT_PROTOCOL,
    prompt_version=PAPER_INFERENCE_PROMPT_VERSION,
    graph_version=PAPER_GRAPH_VERSION,
)
"""The frozen protocol as the paper experiment actually runs it."""

__all__ = [
    "PAPER_GRAPH_VERSION",
    "PAPER_INFERENCE_PROMPT_VERSION",
    "PAPER_PROTOCOL",
]
