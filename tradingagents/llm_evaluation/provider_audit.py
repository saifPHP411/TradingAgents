"""Provider readiness: which LLM providers this environment can actually reach.

The evaluation is gated on a working model credential. This module reports, from
the code's own registries and the *names* of the environment variables they
declare, whether that gate is open — and, when it is not, the exact variable to
set and where to obtain a key. It is the provider-side companion to
:mod:`.point_in_time`, which covers the data side.

Two things it deliberately does not do:

- It never reads, stores, prints or serialises a credential *value*. A provider
  is reported ready only when its declared env var holds a value that is neither
  empty nor an obvious placeholder, and the value itself never leaves
  :func:`os.environ`.
- It makes no network request. Reachability and a documented free tier are
  recorded as static facts with a source, not probed — so this runs offline and
  cannot spend a request budget.

The facts are derived from ``llm_clients`` (``api_key_env``, the OpenAI-compatible
provider registry, the capability table and the model catalog) rather than
hand-written, so adding a provider to the client registers it here too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tradingagents.llm_clients.api_key_env import get_api_key_env

# The providers the factory can build. Mirrors ``llm_clients.factory``: the
# native-API clients plus the OpenAI-compatible registry.
_NATIVE_PROVIDERS = ("anthropic", "google", "azure", "bedrock")

# Values that look like a credential to ``os.environ`` but are not one. Kept in
# step with ``evaluate._PLACEHOLDER_MARKERS``; the same conservative rule is
# applied here so the two readiness reports cannot disagree.
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
_MIN_PLAUSIBLE_KEY_LENGTH = 16


def looks_like_a_real_credential(value: str | None) -> bool:
    """Whether an environment value is plausibly a usable credential.

    Deliberately conservative: it errs toward "not usable". Reporting a blocked
    provider when a key is present is a small, visible cost; reporting a ready
    one from a placeholder is a false result, which is worse. Never returns or
    logs the value.
    """
    if not value:
        return False
    lowered = value.strip().lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    return len(lowered) >= _MIN_PLAUSIBLE_KEY_LENGTH


def supported_providers() -> tuple[str, ...]:
    """Every provider the LLM client factory can build, in registry order."""
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    ordered = list(_NATIVE_PROVIDERS)
    ordered.extend(p for p in OPENAI_COMPATIBLE_PROVIDERS if p not in ordered)
    return tuple(ordered)


def _needs_structured_output(provider: str) -> bool:
    """Whether the graph needs tool/structured output for this provider.

    The graph's agents ask for structured output. Providers that the capability
    table marks as having no structured-output method would force a free-text
    fallback; every provider shipped today offers function calling, so this is a
    property to *check* rather than assume, and it is reported per provider.
    """
    from tradingagents.llm_clients.capabilities import get_capabilities
    from tradingagents.llm_clients.model_catalog import get_known_models

    models = [m for m in get_known_models().get(provider, []) if m != "custom"]
    if not models:
        # Providers with no fixed list (local servers, relays) are user-defined;
        # we cannot assert a capability, so we report the default table entry.
        models = ["__default__"]
    return all(get_capabilities(m).preferred_structured_method != "none" for m in models)


def _transport(provider: str) -> str:
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    if provider == "anthropic":
        return "anthropic"
    if provider == "google":
        return "google"
    if provider == "azure":
        return "azure_openai"
    if provider == "bedrock":
        return "aws_bedrock"
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if spec is None:
        return "unknown"
    if provider == "openai" and spec.use_responses_api:
        return "openai_responses"
    if provider == "chatgpt":
        return "chatgpt_signin"
    if provider == "ollama":
        return "openai_compatible_local"
    if provider == "openai_compatible":
        return "openai_compatible_generic"
    return "openai_compatible"


def _credential_requirement(provider: str) -> str:
    if provider == "bedrock":
        return "aws_credential_chain"
    if provider == "chatgpt":
        return "chatgpt_signin"
    if provider in ("ollama", "openai_compatible"):
        return "none_or_optional"
    return "api_key"


@dataclass(frozen=True)
class ProviderReadiness:
    """One provider's readiness, with no credential value in sight."""

    provider: str
    transport: str
    credential_requirement: str
    api_key_env: str | None
    api_key_present: bool
    needs_structured_output: bool
    free_tier_documented: str
    """``yes`` / ``no`` / ``unverified`` — a claim about a documented free tier."""
    free_tier_source: str
    """The official URL the free-tier claim comes from, or empty when unverified."""
    free_tier_note: str
    base_url: str | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "transport": self.transport,
            "credential_requirement": self.credential_requirement,
            "api_key_env": self.api_key_env,
            "api_key_present": self.api_key_present,
            "needs_structured_output": self.needs_structured_output,
            "free_tier_documented": self.free_tier_documented,
            "free_tier_source": self.free_tier_source,
            "free_tier_note": self.free_tier_note,
            "base_url": self.base_url,
            "notes": self.notes,
        }


# Documented free-tier facts, verified against official pages on the date noted.
# A provider absent from this table is reported ``unverified``: this module does
# not assume an API is free, and it does not guess a URL.
_FREE_TIER_FACTS: dict[str, dict[str, str]] = {
    "groq": {
        "documented": "yes",
        "source": "https://console.groq.com/docs/rate-limits",
        "note": (
            "Groq documents a 'Free Plan Limits' table alongside 'Developer Plan "
            "Limits' (verified 2026-10-28). A free-plan key is required; the plan "
            "is free of charge but needs a Groq account."
        ),
    },
    "google": {
        "documented": "yes",
        "source": "https://ai.google.dev/gemini-api/docs/rate-limits",
        "note": (
            "Gemini documents a Free tier with per-model RPM/TPM/RPD limits; "
            "moving to Tier 1 requires setting up billing (verified 2026-10-28)."
        ),
    },
    "openrouter": {
        "documented": "yes",
        "source": "https://openrouter.ai/docs/api-reference/limits",
        "note": (
            "OpenRouter documents free model variants (IDs ending in ':free') with "
            "their own per-minute/per-day caps; the caps depend on credits "
            "purchased all-time (verified 2026-10-28)."
        ),
    },
    "ollama": {
        "documented": "yes",
        "source": "https://ollama.com",
        "note": (
            "A local Ollama runtime serves models with no account and no per-token "
            "cost; it needs a machine able to run the model (verified 2026-10-28)."
        ),
    },
}

# Providers we checked and could not confirm a documented free API tier for, with
# the reason, so 'unverified' is an explicit finding rather than an omission.
_UNVERIFIED_NOTES: dict[str, str] = {
    "deepseek": "no free API tier found on the official pricing page (checked 2026-10-28)",
    "openai": "pay-as-you-go API; no free API tier documented (checked 2026-10-28)",
    "anthropic": "pay-as-you-go API; no free API tier documented (checked 2026-10-28)",
}


def provider_readiness(provider: str) -> ProviderReadiness:
    """Readiness for one provider, derived from the client registries."""
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    api_key_env = get_api_key_env(provider)
    requirement = _credential_requirement(provider)
    if requirement == "none_or_optional":
        # No key is required, but the credential is not the blocker: a live
        # endpoint must be running. Reported as "key not required", not "usable".
        present = False
    else:
        present = bool(api_key_env and looks_like_a_real_credential(os.environ.get(api_key_env)))

    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    facts = _FREE_TIER_FACTS.get(provider)
    return ProviderReadiness(
        provider=provider,
        transport=_transport(provider),
        credential_requirement=requirement,
        api_key_env=api_key_env,
        api_key_present=present,
        needs_structured_output=_needs_structured_output(provider),
        free_tier_documented=facts["documented"] if facts else "unverified",
        free_tier_source=facts["source"] if facts else "",
        free_tier_note=facts["note"] if facts else _UNVERIFIED_NOTES.get(provider, ""),
        base_url=(spec.base_url if spec else None),
    )


def provider_report() -> dict[str, Any]:
    """The JSON-safe provider audit block, with no credential value in it."""
    providers = {p: provider_readiness(p).to_dict() for p in supported_providers()}
    usable = sorted(p for p, r in providers.items() if r["api_key_present"])
    local_endpoint = os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL") or None
    free_options = sorted(
        p for p, r in providers.items() if r["free_tier_documented"] == "yes"
    )
    return {
        "providers": providers,
        "usable_now": usable,
        "free_tier_options": free_options,
        "local_endpoint_configured": bool(local_endpoint),
        "any_provider_usable": bool(usable) or bool(local_endpoint),
        "checked_on": "2026-10-28",
        "network_probed": False,
        "note": (
            "Readiness is derived from environment-variable *names* and the client "
            "registries; no credential value is read into this report and no "
            "provider was contacted. A documented free tier is not proof a key is "
            "held; a key is not proof the tier is free."
        ),
    }


__all__ = [
    "ProviderReadiness",
    "looks_like_a_real_credential",
    "provider_readiness",
    "provider_report",
    "supported_providers",
]
