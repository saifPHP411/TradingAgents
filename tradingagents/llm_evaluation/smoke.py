"""An opt-in, bounded, single-request real-provider smoke test.

This is the only path in the phase that may make a real model call, and it is
deliberately hard to run by accident:

- it is invoked explicitly (``python -m tradingagents.llm_evaluation.smoke``),
  never from the offline test suite;
- it checks configuration *before* any request and stops with the exact missing
  environment-variable name if there is no credential;
- it makes **at most one** small request, with a timeout;
- it validates the returned decision with the production rating parser and
  refuses to write a record for a response that has no readable rating;
- it writes a record only when the decision and its metadata are valid;
- it never places an order and never touches the execution or risk layers;
- it reports whether the request genuinely succeeded.

Cost and credential safety:

- the credential value is never printed, logged or written; the report names the
  environment variable only;
- the default provider is Groq, whose Free Plan and OpenAI-compatible API are
  documented at https://console.groq.com/docs/rate-limits and
  https://console.groq.com/docs/models (the provider audit carries the quotes).
  If the credential is absent the command stops; it does not prompt for one or
  create an account.

A successful smoke test proves one thing: a real model call returned a decision
that validated and was captured. It is a current-date observation, **not** a
historical backtest point, and it says nothing about strategy quality.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.llm_clients.api_key_env import get_api_key_env

from .capture import (
    MODE_MOCK_TEST,
    MODE_REAL_INFERENCE_CAPTURE,
    CaptureError,
    build_record,
    observed_decision,
)
from .protocol import DEFAULT_PROTOCOL
from .records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_SYNTHETIC_FIXTURE,
    DecisionRecordSet,
    assert_secret_free,
)

# Groq: OpenAI-compatible, a documented Free Plan, and its production models are
# listed with per-token prices in the official docs. The provider audit records
# the exact quotes; this constant only names the default.
DEFAULT_PROVIDER = "groq"
# A small, fast, documented production model. Overridable, but the default must
# be one the official docs list.
DEFAULT_MODEL = "openai/gpt-oss-20b"

# A short, single-turn prompt that asks for the decision in the shape the
# production rating parser reads ("**Rating**: X"). Deliberately tiny: the smoke
# test exercises the request/response/validate/capture path, not the full graph.
_PROMPT = (
    "You are a portfolio manager. For the instrument SPY on {session}, given no "
    "further data, reply in exactly this format and nothing else:\n"
    "**Rating**: <one of Buy, Overweight, Hold, Underweight, Sell>\n"
    "**Executive Summary**: <one short sentence>"
)


class SmokeBlocked(RuntimeError):
    """The smoke test cannot run: a prerequisite is missing. No request was made."""


@dataclass
class SmokeReport:
    """The outcome of a smoke attempt, artifact-ready and secret-free."""

    status: str
    provider: str
    model: str
    api_key_env: str
    attempted: bool = False
    succeeded: bool = False
    genuine: bool = False
    requests_made: int = 0
    record_path: str | None = None
    mode: str = MODE_REAL_INFERENCE_CAPTURE
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    failure_category: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "genuine": self.genuine,
            "requests_made": self.requests_made,
            "mode": self.mode,
            "record_path": self.record_path,
            "notes": list(self.notes),
            "error": self.error,
            "failure_category": self.failure_category,
        }
        return assert_secret_free(payload)


def _classify_failure(exc: BaseException) -> str:
    """A closed-vocabulary category for a failed provider call.

    Phase 9 must report the *category* of a failure, not just its text. The
    mapping is by exception type and status, so it stays stable across providers
    and never echoes a raw response body or a credential.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is not None:
        if status == 401:
            return "authentication_failure"
        if status == 403:
            return "authentication_failure"
        if status == 404:
            return "model_unavailable"
        if status == 408:
            return "timeout"
        if status == 429:
            return "rate_limit"
        if isinstance(status, int) and 500 <= status < 600:
            return "provider_error"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "auth" in name or "permission" in name:
        return "authentication_failure"
    if "connection" in name:
        return "provider_unreachable"
    if "value" in name or "config" in name or "key" in name:
        return "configuration_error"
    if "json" in name or "decode" in name or "validation" in name:
        return "malformed_response"
    return "other"


def _credential_present(provider: str) -> tuple[bool, str]:
    """Whether a plausible credential is set, and the env-var name to name.

    Reads the variable *name* and presence only, never the value.
    """
    env_name = get_api_key_env(provider)
    if env_name is None:
        return False, ""
    value = os.environ.get(env_name)
    return bool(value and value.strip()), env_name


def run_smoke(
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    session: str | None = None,
    feed=None,
    timeout: float = 30.0,
    client_factory=None,
    record_out: str | Path | None = None,
) -> SmokeReport:
    """Run one bounded real request and capture the decision, or stop before it.

    ``client_factory`` and ``feed`` are seams for tests. In production both are
    left unset: the factory is the real provider client and the feed is the
    frozen dataset. An injected factory exercises the mechanics but is never
    counted as genuine: the report's ``genuine`` is ``True`` only when the real
    provider client made the call, and the record is labelled a fixture
    otherwise. Tests assert on the *behaviour* (config check, one request,
    validation), never on a real provider.
    """
    session = session or datetime.now(UTC).strftime("%Y-%m-%d")
    present, env_name = _credential_present(provider)
    if not present:
        # Fail before any request, naming the variable and nothing else.
        raise SmokeBlocked(
            f"no credential for provider {provider!r}: set the {env_name} environment "
            "variable to a real key. The smoke test makes no request without it."
        )

    from tradingagents.llm_clients import factory

    make_client = client_factory or factory.create_llm_client

    report = SmokeReport(
        status="SMOKE BLOCKED",
        provider=provider,
        model=model,
        api_key_env=env_name,
    )
    report.attempted = True
    try:
        # max_retries=0 is a cost guard, not a tuning knob: the SDK default is 3
        # attempts on a 5xx, which would turn one "smoke test" into up to three
        # billable requests. One request means one request.
        client = make_client(
            provider=provider, model=model, timeout=timeout, max_retries=0
        )
        llm = client.get_llm()
        response = llm.invoke(_PROMPT.format(session=session))
        report.requests_made += 1
    except Exception as exc:  # a provider error is reported, not swallowed
        report.status = "SMOKE FAILED"
        report.error = f"{type(exc).__name__}: {exc}"
        report.failure_category = _classify_failure(exc)
        return report

    text = getattr(response, "content", response)
    if isinstance(text, list):  # some providers return content blocks
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in text
        )
    try:
        rating, reasoning = observed_decision(text)
    except CaptureError as exc:
        report.status = "SMOKE FAILED"
        report.error = str(exc)
        return report

    from tradingagents.signals import action_for_rating

    action = action_for_rating(rating)
    if action is None:  # pragma: no cover - observed_decision already refuses this
        report.status = "SMOKE FAILED"
        report.error = f"rating {rating!r} has no tradeable action"
        return report

    frame = None
    if feed is not None:
        try:
            if "SPY" in feed.symbols:
                frame = feed.history("SPY", session)
        except Exception as exc:
            report.notes.append(f"price window unavailable: {exc}")

    # A record is genuine inference only when the *real* provider client made the
    # call. An injected client (the test seam) can exercise the mechanics but can
    # never be counted as a real observation, so its record is labelled a fixture
    # and the report says so.
    genuine = client_factory is None
    source = SOURCE_LLM_INFERENCE if genuine else SOURCE_SYNTHETIC_FIXTURE
    report.genuine = genuine
    report.mode = MODE_REAL_INFERENCE_CAPTURE if genuine else MODE_MOCK_TEST
    if not genuine:
        report.notes.append(
            "an injected client was used (test path); the record is labelled a fixture "
            "and is NOT a real inference observation"
        )

    try:
        record, window_reproducible = build_record(
            ticker="SPY",
            session=session,
            rating=rating,
            action=action,
            frame=frame,
            protocol=DEFAULT_PROTOCOL,
            source=source,
            provider=provider,
            model_name=model,
            reasoning_excerpt=reasoning,
        )
    except CaptureError as exc:
        report.status = "SMOKE FAILED"
        report.error = str(exc)
        return report

    report.succeeded = True
    report.status = "SMOKE OK"
    if not window_reproducible:
        report.notes.append(
            "no price window was available for this session, so the record's input is "
            "not reproducible; a prospective observation only"
        )
    report.notes.append(
        "one genuine model call returned a decision that validated; this is a "
        "current observation, not a historical backtest point, and says nothing "
        "about strategy quality"
    )

    if record_out is not None:
        record_set = DecisionRecordSet(
            strategy_id=f"smoke-{provider}",
            graph_version=DEFAULT_PROTOCOL.graph_version,
            protocol_version=DEFAULT_PROTOCOL.version,
            mode=source,
            created_at=datetime.now(UTC).isoformat(),
            decisions=(record,),
        )
        path = record_set.save(record_out)
        report.record_path = str(path)
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One bounded, opt-in real-provider smoke test. Makes at most one request "
            "and only when a credential is configured; never places an order."
        )
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--session", default=None, help="decision date YYYY-MM-DD (default: today)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--record-out", default=None, help="path to write the captured record set")
    args = parser.parse_args(argv)

    try:
        report = run_smoke(
            provider=args.provider,
            model=args.model,
            session=args.session,
            timeout=args.timeout,
            record_out=args.record_out,
        )
    except SmokeBlocked as exc:
        print(f"SMOKE BLOCKED: {exc}")
        return 2
    print(json.dumps(report.to_dict(), indent=2))
    if report.record_path:
        print(f"wrote {report.record_path}")
    if report.failure_category:
        print(f"failure_category: {report.failure_category}")
    return 0 if report.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = ["DEFAULT_MODEL", "DEFAULT_PROVIDER", "SmokeBlocked", "SmokeReport", "run_smoke"]
