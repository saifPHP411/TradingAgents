"""Evaluate the LLM strategy if it can run; otherwise report the exact blocker.

This module never invents a result. It has three outcomes:

- **blocked** — no genuine LLM decisions are available (no credential, or no
  captured decisions). The artifact says ``LLM BASELINE BLOCKED`` and lists the
  exact missing prerequisites. The reference SMA and SPY numbers are included as
  *context only*, clearly labelled, so a reader can see what the comparison would
  be against without mistaking them for LLM performance.
- **replay** — a set of recorded decisions (operator-supplied, captured from a
  real run elsewhere) is replayed through the frozen engine. The artifact records
  the result and marks the mode ``replay``: genuine model output, no model call.
- **inference** — decisions were produced by a real provider call in this run.
  Only the presence of a working credential and a captured set can reach this.

The credential check reports *which* environment variables are set, never their
values. Nothing here prints, logs or serialises a secret.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tradingagents import __version__ as package_version
from tradingagents.baseline import harness
from tradingagents.baseline.data import load_feed
from tradingagents.baseline.reference import SmaCrossover
from tradingagents.llm_clients.api_key_env import get_api_key_env

from .point_in_time import point_in_time_report
from .protocol import DEFAULT_PROTOCOL, LLMEvaluationProtocol
from .provider_audit import provider_report
from .records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_OPERATOR_SUPPLIED,
    SOURCE_SYNTHETIC_FIXTURE,
    DecisionRecordSet,
    assert_secret_free,
)
from .replay import ReplayStrategy, no_llm_calls

ARTIFACT_SCHEMA_VERSION = "1.0.0"

STATUS_ESTABLISHED = "LLM BASELINE ESTABLISHED"
STATUS_ESTABLISHED_WITH_FINDINGS = "LLM BASELINE ESTABLISHED WITH FINDINGS"
STATUS_BLOCKED = "LLM BASELINE BLOCKED"

# Providers the factory can build, and the environment variable each needs.
# ``chatgpt`` and ``bedrock`` are not key-in-env providers, so they are listed
# separately rather than as a key.
_KEYED_PROVIDERS = (
    "openai",
    "google",
    "anthropic",
    "xai",
    "deepseek",
    "minimax",
    "openrouter",
    "mistral",
    "groq",
    "nvidia",
    "azure",
    "openai_compatible",
)
_NON_KEY_PROVIDERS = ("chatgpt", "bedrock")


# Values that look like a credential to ``os.environ`` but are not one. The test
# suite's autouse fixture sets every provider key to ``"placeholder"`` so that
# unit tests can construct clients; if those counted as credentials, the
# evaluation would report a runnable LLM strategy that cannot actually run. A
# genuine key is neither an obvious placeholder nor a documentation example.
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "changeme",
    "change-me",
    "your-api-key",
    "your_api_key",
    "dummy",
    "example",
    "fake",
    "todo",
    "xxx",
    "sk-test",
    "test-key",
    "testkey",
)
# Real provider keys are long random tokens. A very short value is a stand-in.
_MIN_PLAUSIBLE_KEY_LENGTH = 16


def _looks_like_a_real_credential(value: str | None) -> bool:
    """Whether an environment value is plausibly a usable credential.

    Deliberately conservative: it errs toward "not usable". Reporting a blocked
    LLM baseline when a key is present is a small, visible cost; reporting an
    established one from a placeholder is a false result, which is worse.
    """
    if not value:
        return False
    lowered = value.strip().lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    return len(lowered) >= _MIN_PLAUSIBLE_KEY_LENGTH


def credential_status() -> dict:
    """Which providers have a credential available, by variable *name* only.

    Never returns a credential value: the caller sees variable names and
    booleans, so the artifact cannot leak a key even if it is serialised.
    """
    available: list[str] = []
    missing: list[str] = []
    details: dict[str, str] = {}
    for provider in _KEYED_PROVIDERS:
        env_name = get_api_key_env(provider)
        if env_name is None:
            continue
        is_set = _looks_like_a_real_credential(os.environ.get(env_name))
        details[provider] = env_name
        (available if is_set else missing).append(provider)
    local_endpoint = os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL")
    return {
        "keyed_providers": list(_KEYED_PROVIDERS),
        "non_key_providers": list(_NON_KEY_PROVIDERS),
        "env_var_by_provider": details,
        "providers_with_credentials": available,
        "providers_without_credentials": missing,
        "any_credential_available": bool(available) or bool(local_endpoint),
        "local_endpoint_configured": bool(local_endpoint),
    }


_AGENTS_BY_INPUT_SCOPE = {
    # What each declared scope can actually evaluate, by analyst key. The frozen
    # dataset carries prices only, so a price_only run measures the market
    # analyst alone: news, sentiment and fundamentals need sources the dataset
    # does not carry and which are not point-in-time safe for a historical run.
    "price_only": ("market",),
}


def _scope_agents(scope: str) -> tuple[str, ...]:
    from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

    if scope in _AGENTS_BY_INPUT_SCOPE:
        return _AGENTS_BY_INPUT_SCOPE[scope]
    return tuple(ANALYST_NODE_SPECS)


@dataclass
class EvaluationOutcome:
    """Everything a caller needs, plus the artifact-ready dict."""

    status: str
    mode: str
    artifact: dict
    blockers: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.status == STATUS_BLOCKED


def _reference_context(feed, protocol: LLMEvaluationProtocol) -> dict:
    """SMA and SPY over the protocol's own windows, as context only.

    Computed with the same engine, the same costs and the same windows the LLM
    run would use, so the numbers are comparable if and when a genuine LLM run
    exists. They are *not* LLM performance and the artifact labels them so.
    """
    strategy = SmaCrossover(position_size=protocol.default_position_size)
    out: dict[str, dict] = {}
    for label, start, end in protocol.split_periods():
        window = harness.evaluate_window(feed, protocol, strategy, label, start, end)
        summary = harness.summarize_window(window)
        summary["cross_check_passes"] = window.cross_check_passes
        out[label] = summary
    return out


def _spy_context(feed, protocol: LLMEvaluationProtocol) -> dict:
    """Buy-and-hold SPY over the same windows, for the benchmark column."""
    from tradingagents.baseline.reference import align_benchmark, benchmark_equity

    out: dict[str, dict] = {}
    for label, start, end in protocol.split_periods():
        sessions = [s for s in feed.calendar() if start <= s <= end]
        if not sessions:
            continue
        curve = benchmark_equity(
            feed, protocol.benchmark, initial_cash=protocol.initial_cash, sessions=sessions
        )
        curve = align_benchmark(curve, sessions)
        total = (float(curve.iloc[-1]) - protocol.initial_cash) / protocol.initial_cash
        out[label] = {
            "total_return": total,
            "final_equity": float(curve.iloc[-1]),
            "sessions": len(sessions),
        }
    return out


def _run_replay(feed, protocol: LLMEvaluationProtocol, record_set: DecisionRecordSet) -> dict:
    """Replay recorded decisions over the protocol's windows and measure them.

    A window with no sessions in the feed is recorded as empty rather than
    raising: a protocol window that falls outside the available data is a fact to
    report, not a crash. The empty marker keeps it visible in the artifact.
    """
    results: dict[str, dict] = {}
    for label, start, end in protocol.split_periods():
        if not feed.calendar(start, end):
            results[label] = {
                "label": label,
                "start": start,
                "end": end,
                "sessions": 0,
                "empty": True,
                "note": "no sessions in the feed for this window",
            }
            continue
        strategy = ReplayStrategy(record_set)
        # no_llm_calls() turns a stray model call into an error rather than a
        # silent paid request.
        with no_llm_calls():
            window = harness.evaluate_window(feed, protocol, strategy, label, start, end)
        summary = harness.summarize_window(window)
        summary["cross_check_passes"] = window.cross_check_passes
        summary["replay_stats"] = {
            "sessions_seen": strategy.stats.sessions_seen,
            "sessions_with_record": strategy.stats.sessions_with_record,
            "signals_emitted": strategy.stats.signals_emitted,
            "signals_skipped_review": strategy.stats.signals_skipped_review,
        }
        results[label] = summary
    return results


def _capture_status(
    record_set: DecisionRecordSet | None, creds: dict, providers: dict
) -> dict:
    """The honest, field-by-field answer to "how far did we actually get?".

    Kept separate from ``credentials`` (which provider keys are set) and from
    ``provider_readiness`` (which providers are usable at all). Every field is a
    fact about *this* environment, not a plan.
    """
    inference_succeeded = bool(
        record_set is not None and record_set.mode == SOURCE_LLM_INFERENCE
    )
    record_captured = record_set is not None
    inputs_reproducible = bool(
        record_captured
        and all(r.bar_count > 0 and len(r.data_window_hash) == 64 for r in record_set.decisions)
    )
    # Input reproducibility is not point-in-time safety: a hash proves the replay
    # saw the same bars, not that the bars were available at the decision time.
    pit = point_in_time_report(DEFAULT_PROTOCOL.input_scope)
    inputs_pit_safe = bool(pit["readiness"]["ready"])
    # Historical evaluation of the *declared scope*: a genuine capture whose
    # window is reproducible and PIT-safe for that scope. The full graph always
    # stays blocked, because news/sentiment/fundamentals are not PIT-safe.
    historical_allowed = bool(inference_succeeded and inputs_reproducible and inputs_pit_safe)
    full_graph_pit = point_in_time_report("full")
    full_graph_allowed = bool(
        inference_succeeded
        and inputs_reproducible
        and bool(full_graph_pit["readiness"]["ready"])
    )

    mode = "blocked"
    if record_set is not None:
        mode = (
            "REAL_INFERENCE_CAPTURE"
            if record_set.mode == SOURCE_LLM_INFERENCE
            else "REPLAY"
        )

    blocked: list[str] = []
    if not creds["any_credential_available"]:
        blocked.append(
            "no provider credential in this environment; set one of "
            + ", ".join(
                creds["env_var_by_provider"][p]
                for p in creds["providers_without_credentials"]
            )
            + " (or a local OpenAI-compatible endpoint) to run genuine inference"
        )
    if not record_captured:
        blocked.append(
            "no genuine decision record set was captured here; run the opt-in smoke "
            "test (python -m tradingagents.llm_evaluation.smoke) with a credential, "
            "or supply a captured set to build_evaluation(record_set=...)"
        )
    if not inputs_reproducible:
        blocked.append(
            "the captured record has no reproducible price window (bar_count 0 or an "
            "unavailable snapshot), so it cannot anchor a prospective paper "
            "experiment; re-capture against a session the frozen dataset covers, e.g. "
            "python -m tradingagents.llm_evaluation.smoke --snapshot-out <path>"
        )
    if not inputs_pit_safe:
        blocked.append(
            "the news/sentiment/fundamentals inputs are not point-in-time safe for a "
            "historical run, so a full historical evaluation stays blocked; see "
            "docs/LLM_POINT_IN_TIME_AUDIT.md"
        )

    return {
        "evaluation_mode": mode,
        "provider_configured": bool(creds["any_credential_available"]),
        "genuine_inference_request_succeeded": inference_succeeded,
        "decision_record_captured": record_captured,
        "inputs_reproducible": inputs_reproducible,
        "inputs_point_in_time_safe": inputs_pit_safe,
        "historical_evaluation_allowed": historical_allowed,
        "historical_evaluation_scope": DEFAULT_PROTOCOL.input_scope,
        "full_graph_historical_evaluation_allowed": full_graph_allowed,
        "prospective_paper_experiment_possible": bool(record_captured and inputs_reproducible),
        "blocked_prerequisites": blocked,
        "note": (
            "A captured current inference is not automatically a valid historical "
            "backtest observation. Input reproducibility (the replay sees the same "
            "bars) is distinct from point-in-time safety (those bars were available "
            "at the decision time)."
        ),
    }


def build_evaluation(
    *,
    protocol: LLMEvaluationProtocol | None = None,
    record_set: DecisionRecordSet | None = None,
    feed=None,
    include_reference_context: bool = True,
) -> EvaluationOutcome:
    """Assemble the evaluation artifact. Never fabricates a result.

    ``include_reference_context`` runs the SMA and SPY comparison over the
    protocol's windows. It is the expensive part and does not depend on the
    record set, so a caller that only needs the blocker or the replay result can
    turn it off.
    """
    protocol = protocol or DEFAULT_PROTOCOL
    creds = credential_status()
    feed = feed if feed is not None else load_feed(protocol)

    pit = point_in_time_report(protocol.input_scope)
    providers = provider_report()
    evaluated = _scope_agents(protocol.input_scope)
    omitted = tuple(a for a in _scope_agents("full") if a not in evaluated)
    restricted = protocol.input_scope != "full"

    if include_reference_context:
        reference = _reference_context(feed, protocol)
        spy = _spy_context(feed, protocol)
        reference_block: dict = {
            "note": (
                "SMA reference and buy-and-hold SPY over the same windows, for context "
                "only. These are NOT LLM-strategy results."
            ),
            "sma_reference": reference,
            "spy_buy_and_hold": spy,
        }
    else:
        reference_block = {
            "note": "reference context not computed for this build",
        }

    mode = "blocked"
    status = STATUS_BLOCKED
    blockers: list[str] = []
    llm_results: dict | None = None
    provenance: dict = {}

    if record_set is not None:
        mode = record_set.mode
        provenance = {
            "strategy_id": record_set.strategy_id,
            "graph_version": record_set.graph_version,
            "created_at": record_set.created_at,
            "decision_count": len(record_set.decisions),
            "sessions": list(record_set.sessions),
        }
        llm_results = _run_replay(feed, protocol, record_set)
        if record_set.mode == SOURCE_LLM_INFERENCE:
            status = STATUS_ESTABLISHED
        else:
            # Operator-supplied or fixture decisions are genuine model output
            # replayed, or a fixture; neither is a fresh inference run, so the
            # result carries findings rather than a clean "established".
            status = STATUS_ESTABLISHED_WITH_FINDINGS
            if record_set.mode == SOURCE_SYNTHETIC_FIXTURE:
                blockers.append(
                    "decisions are a synthetic fixture: the numbers below exercise the "
                    "replay path and are NOT strategy performance"
                )
            else:
                blockers.append(
                    "decisions were captured outside this environment and replayed; the "
                    "model's own inference was not repeated here"
                )
    else:
        if not creds["any_credential_available"]:
            blockers.append(
                "no LLM provider credential is available in this environment; set one of "
                + ", ".join(creds["env_var_by_provider"][p] for p in creds["providers_without_credentials"])
                + " (or configure a local OpenAI-compatible endpoint) to run genuine inference"
            )
            if providers["free_tier_options"]:
                blockers.append(
                    "documented free-tier options to obtain a key: "
                    + ", ".join(providers["free_tier_options"])
                    + "; see docs/LLM_PROVIDER_AUDIT.md for the official source of each claim"
                )
        blockers.append(
            "no captured decision record set was supplied; pass one to build_evaluation() "
            "to evaluate real model output via replay"
        )

    # A restricted scope omits agents; the artifact must say so rather than let a
    # price-only replay be read as a full-graph evaluation.
    if restricted:
        blockers.append(
            "input_scope='price_only' is a RESTRICTED evaluation: only the market "
            f"analyst runs. Omitted agents: {', '.join(omitted)}. It does not measure "
            "the news/sentiment/fundamentals pipeline."
        )

    capture_status = _capture_status(record_set, creds, providers)

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "mode": mode,
        "evaluation_mode": capture_status["evaluation_mode"],
        "metadata": {
            "package_version": package_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": protocol.to_dict(),
        "credentials": creds,
        "provider_readiness": providers,
        "capture_status": capture_status,
        "dataset_checksums": protocol.to_dict()["dataset_checksums"],
        "point_in_time": pit,
        "evaluation_scope": {
            "input_scope": protocol.input_scope,
            "restricted": restricted,
            "agents_evaluated": list(evaluated),
            "agents_omitted": list(omitted),
            "is_full_graph": not restricted,
        },
        "reference_context": reference_block,
        "llm_results": llm_results,
        "provenance": provenance,
        "blockers": blockers,
        "genuine_inference": bool(record_set is not None and record_set.mode == SOURCE_LLM_INFERENCE),
        "replay_tested": record_set is not None,
        "reproducibility": {
            "execution_determinism": "verified by tests (same records -> identical fills/ledger)",
            "replay_determinism": "verified by tests (two replays -> identical artifact)",
            "inference_repeatability": (
                "not tested: requires a working credential"
                if not capture_status["genuine_inference_request_succeeded"]
                else "a single genuine request succeeded; repeatability across calls is not "
                "established by one observation"
            ),
            "inference_determinism_claimed": False,
        },
        "limitations": {
            "point_in_time_scope": pit["readiness"]["scope"],
            "point_in_time_ready": pit["readiness"]["ready"],
            "news_sentiment_fundamentals_evaluated": False,
            "note": (
                "The frozen dataset is price-only and the news/sentiment/fundamentals "
                "sources are not point-in-time safe for a historical run; see "
                "docs/LLM_POINT_IN_TIME_AUDIT.md."
            ),
        },
    }
    # The artifact is scanned for credentials before it can leave this function.
    assert_secret_free(artifact)
    return EvaluationOutcome(status=status, mode=mode, artifact=artifact, blockers=blockers)


def write_evaluation(outcome: EvaluationOutcome, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(outcome.artifact, indent=2, sort_keys=False)
    assert_secret_free(text)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None, *, feed=None) -> int:
    """Run the evaluation from the command line.

    ``feed`` is a seam for tests: it defaults to the frozen dataset, and a caller
    that passes one gets that feed instead. The production path never passes it.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the LLM strategy, or report the blocker.")
    parser.add_argument(
        "--records",
        default=None,
        help="path to a captured decision record set (JSON) to replay",
    )
    parser.add_argument(
        "--out",
        default="docs/baseline/llm_evaluation.json",
        help="where to write the artifact",
    )
    parser.add_argument(
        "--no-reference-context",
        action="store_true",
        help=(
            "skip the SMA and buy-and-hold comparison windows. They do not depend "
            "on the records and are the slow part of a build; skipping them is for "
            "quick iteration, and the artifact records that they are absent."
        ),
    )
    args = parser.parse_args(argv)

    record_set = DecisionRecordSet.load(args.records) if args.records else None
    outcome = build_evaluation(
        record_set=record_set,
        feed=feed,
        include_reference_context=not args.no_reference_context,
    )
    path = write_evaluation(outcome, args.out)
    print(f"{outcome.status} (mode: {outcome.mode})")
    for blocker in outcome.blockers:
        print(f"  - {blocker}")
    print(f"wrote {path}")
    return 0


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SOURCE_LLM_INFERENCE",
    "SOURCE_OPERATOR_SUPPLIED",
    "SOURCE_SYNTHETIC_FIXTURE",
    "STATUS_BLOCKED",
    "STATUS_ESTABLISHED",
    "STATUS_ESTABLISHED_WITH_FINDINGS",
    "EvaluationOutcome",
    "build_evaluation",
    "credential_status",
    "main",
    "write_evaluation",
]
