"""Capture a real graph decision into a validated :class:`DecisionRecord`.

This is the missing producer. Until now ``DecisionRecord`` was built only in
tests, so the replay/evaluate layer had nothing genuine to consume. This module
turns a completed graph run's final state into a record, refusing rather than
guessing whenever the inputs cannot be identified faithfully.

The one rule that shapes everything here: **a record is only labelled
``llm_inference`` when a real model call produced the decision.** The label is
not taken from the caller's word. It is derived from what the graph actually
did:

- ``graph.inference_observed`` is set by the LLM client layer the moment a real
  provider call returns. A scripted/stubbed model (a test, a fixture, a mock)
  never sets it, so a mocked run cannot be mislabelled as genuine inference even
  by a caller who passes ``SOURCE_LLM_INFERENCE`` by mistake.
- ``capture_decision(..., mode=MOCK_TEST)`` records a fixture, and the record's
  ``source`` becomes ``synthetic_fixture`` regardless of the model.

What is captured, and from where:

- the decision (rating, and the PM's own summary/thesis) from ``final_state``;
- the model and provider from the graph's own ``run_settings()`` allowlist;
- the input window (dates, bar count and a stable hash) from the same
  session-bounded price frame the market analyst would see, so a replay can
  prove it fed the model the same bars;
- the config hash from the frozen evaluation protocol, so a replay can prove it
  ran under the same configuration.

What is deliberately *not* captured, and why:

- **No order, no size.** The graph decides a rating, not an order. This module
  never invents a ``position_size``, ``stop_loss`` or ``limit_price``; the
  decision's ``order_intent`` is ``MARKET`` with no levels, and sizing is left to
  the risk engine exactly as the production signal path leaves it.
- **No silent repair.** A rating with no tradeable action (``REVIEW``) is refused
  as a record, not quietly turned into a Hold. A decision that cannot be read is
  a fact to report, not a trade to fabricate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from tradingagents.agents.rating import RATINGS_5_TIER
from tradingagents.signals import action_for_rating

from .protocol import DEFAULT_PROTOCOL, LLMEvaluationProtocol
from .records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_SYNTHETIC_FIXTURE,
    DecisionRecord,
    DecisionRecordError,
    assert_secret_free,
    redact_reasoning,
)
from .replay import window_hash

# The evaluation modes this phase distinguishes. They are kept strictly apart so
# a report can never conflate a genuine capture with a replay or a test fixture.
MODE_REAL_INFERENCE_CAPTURE = "REAL_INFERENCE_CAPTURE"
MODE_REPLAY = "REPLAY"
MODE_MOCK_TEST = "MOCK_TEST"
VALID_MODES = (MODE_REAL_INFERENCE_CAPTURE, MODE_REPLAY, MODE_MOCK_TEST)

# The graph attribute that carries a genuine-inference observation. It is set by
# the InferenceObserver on the real provider clients; a scripted model sets
# nothing, so the read fails closed.
_INFERENCE_MARKER = "inference_observed"


class CaptureError(DecisionRecordError):
    """A graph decision could not be captured faithfully."""


def graph_observed_real_inference(graph) -> bool:
    """Whether the graph reports that a real model call produced its output.

    Read from the graph object, never from a caller's argument. An absent marker
    => False, so an unknown or stubbed graph fails closed to "not genuine".
    """
    return bool(getattr(graph, _INFERENCE_MARKER, False))


def _session_frame(feed, ticker: str, session: str):
    """The session-bounded price frame for ``ticker``, or ``None`` if absent.

    Uses the feed's own point-in-time slice, so the window hashed here is the
    window the market analyst would see: never a later bar.
    """
    if feed is None:
        return None
    try:
        if ticker not in feed.symbols:
            return None
        return feed.history(ticker, session)
    except Exception as exc:  # a feed that cannot answer is a capture failure
        raise CaptureError(f"could not read the price window for {ticker} at {session}: {exc}") from exc


def _window_bounds(frame) -> tuple[str, str, int]:
    """``(start, end, bar_count)`` for a frame, or the empty shape when absent."""
    if frame is None or len(frame) == 0:
        return "", "", 0
    start = str(frame.index[0])[:10]
    end = str(frame.index[-1])[:10]
    return start, end, int(len(frame))


@dataclass(frozen=True)
class CaptureResult:
    """A captured record plus what the capture could and could not establish."""

    record: DecisionRecord
    mode: str
    inference_observed: bool
    window_reproducible: bool
    """Whether the input window could be read from a feed and hashed."""
    data_snapshot_id: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "inference_observed": self.inference_observed,
            "window_reproducible": self.window_reproducible,
            "data_snapshot_id": self.data_snapshot_id,
            "decision_id": self.record.decision_id,
            "notes": list(self.notes),
        }


def _decision_summary(final_state: dict) -> str:
    """The PM's own summary/thesis text, bounded and scrubbed, for the record.

    Prefers the structured ``executive_summary`` the PM produces; falls back to
    the rendered decision text. This is context for an auditor, never parsed for
    an action: the rating is read separately.
    """
    for key in ("pm_executive_summary", "executive_summary"):
        value = final_state.get(key)
        if isinstance(value, str) and value.strip():
            return redact_reasoning(value)
    decision = final_state.get("final_trade_decision")
    if isinstance(decision, str):
        return redact_reasoning(decision)
    return ""


def build_record(
    *,
    ticker: str,
    session: str,
    rating: str,
    action,
    frame,
    protocol: LLMEvaluationProtocol,
    source: str,
    provider: str | None,
    model_name: str | None,
    reasoning_excerpt: str = "",
    confidence: float | None = None,
    snapshot_prefix: str = "feed",
    captured_at: str | None = None,
) -> tuple[DecisionRecord, bool]:
    """Build a validated record from an already-resolved decision and window.

    Returns ``(record, window_reproducible)``. Shared by the graph-capture and
    single-call smoke paths so both produce the same format and validation. No
    size or levels are ever invented here.
    """
    window_start, window_end, bar_count = _window_bounds(frame)
    window_reproducible = bar_count > 0
    if window_reproducible:
        data_window_hash = window_hash(frame)
        data_snapshot_id = f"{snapshot_prefix}:{window_start}..{window_end}"
    else:
        data_window_hash = hashlib.sha256().hexdigest()
        data_snapshot_id = f"{snapshot_prefix}:unavailable"
    if window_end and window_end != session:
        raise CaptureError(
            f"price window ends {window_end} but the decision is dated {session}; "
            "the window must end at the decision date"
        )
    stamp = captured_at or datetime.now(UTC).isoformat()
    record = DecisionRecord(
        decision_id=f"{ticker}-{session}",
        ticker=ticker,
        as_of_date=session,
        action=action.value,
        rating=rating,
        confidence=confidence if confidence is not None else _confidence_for({}, rating),
        signal_timestamp=stamp,
        window_start=window_start or session,
        window_end=window_end or session,
        bar_count=bar_count,
        data_snapshot_id=data_snapshot_id,
        data_window_hash=data_window_hash,
        config_hash=protocol.config_hash(),
        prompt_version=protocol.prompt_version,
        graph_version=protocol.graph_version,
        source=source,
        model_provider=provider,
        model_name=model_name,
        reasoning_excerpt=reasoning_excerpt,
        position_size=None,
        stop_loss=None,
        take_profit=None,
        order_intent="MARKET",
        limit_price=None,
        time_in_force="DAY",
        holding_period_days=protocol.holding_period_days,
    )
    assert_secret_free(record.to_dict())
    return record, window_reproducible


def capture_decision(
    *,
    final_state: dict,
    graph,
    feed,
    protocol: LLMEvaluationProtocol | None = None,
    mode: str = MODE_REAL_INFERENCE_CAPTURE,
    captured_at: str | None = None,
) -> CaptureResult:
    """Turn a completed graph run into a validated, provenance-labelled record.

    ``final_state`` is the value ``TradingAgentsGraph.propagate`` returns.
    ``graph`` supplies the model/provider metadata and the genuine-inference
    marker. ``feed`` is the point-in-time price source the window is hashed
    from. ``mode`` declares the caller's intent; the *effective* source is
    derived from the graph, so ``mode=MOCK_TEST`` or a graph with no observed
    inference yields a ``synthetic_fixture`` record and never ``llm_inference``.
    """
    protocol = protocol or DEFAULT_PROTOCOL
    if mode not in VALID_MODES:
        raise CaptureError(f"unknown capture mode {mode!r}; expected one of {VALID_MODES}")

    ticker = final_state.get("company_of_interest")
    if not isinstance(ticker, str) or not ticker.strip():
        raise CaptureError("final_state has no company_of_interest; cannot identify the symbol")
    session = final_state.get("trade_date")
    if not isinstance(session, str) or not session.strip():
        raise CaptureError("final_state has no trade_date; cannot identify the decision date")
    session = session.strip()[:10]

    rating = final_state.get("final_rating")
    if not isinstance(rating, str) or not rating.strip():
        raise CaptureError("final_state has no final_rating; the decision cannot be read")
    rating = rating.strip()

    action = action_for_rating(rating)
    if action is None:
        # REVIEW or an unknown rating is a decision nobody could read. Refuse it
        # rather than degrade to Hold: a fabricated Hold would be replayed as a
        # call the model never made.
        raise CaptureError(
            f"rating {rating!r} has no tradeable action (valid: {', '.join(RATINGS_5_TIER)}); "
            "a decision that cannot be read is not captured as a trade"
        )

    frame = _session_frame(feed, ticker, session)

    settings = graph.run_settings() if hasattr(graph, "run_settings") else {}
    provider = settings.get("deep_think_provider") or settings.get("llm_provider")
    model_name = settings.get("deep_think_llm") or settings.get("quick_think_llm")

    inference_observed = graph_observed_real_inference(graph)
    effective_source = (
        SOURCE_LLM_INFERENCE
        if (mode == MODE_REAL_INFERENCE_CAPTURE and inference_observed)
        else SOURCE_SYNTHETIC_FIXTURE
    )
    effective_mode = (
        MODE_REAL_INFERENCE_CAPTURE
        if effective_source == SOURCE_LLM_INFERENCE
        else MODE_MOCK_TEST
    )

    notes: list[str] = []
    if mode == MODE_REAL_INFERENCE_CAPTURE and not inference_observed:
        notes.append(
            "the graph did not report a real model call, so the record is labelled a "
            "fixture rather than genuine inference"
        )

    record, window_reproducible = build_record(
        ticker=ticker,
        session=session,
        rating=rating,
        action=action,
        frame=frame,
        protocol=protocol,
        source=effective_source,
        provider=provider,
        model_name=model_name,
        reasoning_excerpt=_decision_summary(final_state),
        confidence=_confidence_for(final_state, rating),
        captured_at=captured_at,
    )
    if not window_reproducible:
        notes.append(
            "the price window could not be read from the feed, so the record's input "
            "is not reproducible from this capture"
        )
    return CaptureResult(
        record=record,
        mode=effective_mode,
        inference_observed=inference_observed,
        window_reproducible=window_reproducible,
        data_snapshot_id=record.data_snapshot_id,
        notes=tuple(notes),
    )


def _confidence_for(final_state: dict, rating: str) -> float:
    """A record's confidence: the run's own if it carries one, else the tier's.

    The production signal path uses the same per-rating confidence when a caller
    supplies none, so a captured record and a live signal agree.
    """
    value = final_state.get("final_confidence")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    ):
        return float(value)
    from tradingagents.signals import confidence_for_rating

    return confidence_for_rating(rating)


def observed_decision(response_text: str) -> tuple[str, str]:
    """``(rating, reasoning)`` read from an observed model response, or raise.

    Uses the production rating parser, so a response whose rating cannot be read
    is refused rather than degraded to Hold. The reasoning is the scrubbed,
    bounded response text, kept for an auditor and never parsed for an action.
    """
    from tradingagents.agents.rating import extract_rating

    if not isinstance(response_text, str) or not response_text.strip():
        raise CaptureError("the model returned no text; nothing to capture")
    rating = extract_rating(response_text)
    if rating is None:
        raise CaptureError(
            "the model's response carries no readable rating; refusing to capture a "
            "decision nobody could read"
        )
    return rating, redact_reasoning(response_text)


__all__ = [
    "MODE_MOCK_TEST",
    "MODE_REAL_INFERENCE_CAPTURE",
    "MODE_REPLAY",
    "VALID_MODES",
    "CaptureError",
    "CaptureResult",
    "build_record",
    "capture_decision",
    "graph_observed_real_inference",
    "observed_decision",
]
