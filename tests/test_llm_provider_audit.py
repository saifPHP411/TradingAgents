"""Provider-readiness audit tests. Offline; no provider is contacted.

These tests cover the *provider* side of the evaluation gate: which providers
exist, which need a key, whether a key looks real, whether a key value can leak,
and whether the artifact records provider readiness honestly. They never read a
real credential and never make a request. A test that mocks a provider proves the
code path, not that the provider is reachable; that distinction is stated in the
module and repeated here so a green run is not misread.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.llm_evaluation.provider_audit import (
    looks_like_a_real_credential,
    provider_readiness,
    provider_report,
    supported_providers,
)


class TestSupportedProviders:
    def test_the_registry_matches_the_client_factory(self):
        from tradingagents.llm_clients.factory import create_llm_client
        from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

        # Every provider this module reports must be one the factory can build.
        for provider in supported_providers():
            assert provider in OPENAI_COMPATIBLE_PROVIDERS or provider in (
                "anthropic", "google", "azure", "bedrock",
            ), f"{provider} is reported but not buildable"
        # And the native clients must be present in the report.
        for native in ("anthropic", "google", "azure", "bedrock"):
            assert native in supported_providers()
        assert create_llm_client  # imported to prove the factory import path works

    def test_every_provider_has_a_readiness_row(self):
        report = provider_report()
        assert set(report["providers"]) == set(supported_providers())

    def test_no_provider_name_appears_twice(self):
        names = supported_providers()
        assert len(names) == len(set(names))


class TestCredentialRequirement:
    @pytest.mark.parametrize("provider,env", [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ])
    def test_keyed_providers_name_their_env_var(self, provider, env):
        row = provider_readiness(provider)
        assert row.credential_requirement == "api_key"
        assert row.api_key_env == env

    def test_local_providers_do_not_require_a_key(self):
        for provider in ("ollama", "openai_compatible"):
            row = provider_readiness(provider)
            assert row.credential_requirement == "none_or_optional"
            assert row.api_key_present is False

    def test_bedrock_uses_the_aws_chain_not_a_key(self):
        row = provider_readiness("bedrock")
        assert row.credential_requirement == "aws_credential_chain"
        assert row.api_key_env is None

    def test_chatgpt_uses_signin_not_a_key(self):
        row = provider_readiness("chatgpt")
        assert row.credential_requirement == "chatgpt_signin"


class TestCredentialPlausibility:
    def test_empty_is_not_a_credential(self):
        assert looks_like_a_real_credential("") is False
        assert looks_like_a_real_credential(None) is False

    @pytest.mark.parametrize("value", [
        "placeholder", "changeme", "your-api-key", "dummy", "example",
        "fake", "todo", "xxx", "sk-test", "test-key",
    ])
    def test_placeholders_are_not_credentials(self, value):
        assert looks_like_a_real_credential(value) is False

    def test_a_short_value_is_not_a_credential(self):
        assert looks_like_a_real_credential("short") is False

    def test_a_long_random_value_is_a_credential(self):
        assert looks_like_a_real_credential("sk-" + "a" * 40) is True


class TestReadinessReflectsEnvironment:
    def test_a_placeholder_key_does_not_open_the_gate(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "placeholder")
        assert provider_readiness("groq").api_key_present is False

    def test_a_plausible_key_opens_the_gate_for_that_provider_only(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-" + "a" * 40)
        assert provider_readiness("groq").api_key_present is True
        assert provider_readiness("openai").api_key_present is False

    def test_a_local_endpoint_is_reported_separately(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_LLM_BACKEND_URL", "http://localhost:8000/v1")
        report = provider_report()
        assert report["local_endpoint_configured"] is True
        assert report["any_provider_usable"] is True


class TestNoSecretLeak:
    def test_the_report_never_carries_a_key_value(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-abcdefghijklmnopqrstuvwx")
        report = provider_report()
        assert report["providers"]["openai"]["api_key_present"] is True
        assert "sk-live" not in json.dumps(report)

    def test_the_report_is_json_serialisable(self):
        json.dumps(provider_report())


class TestFreeTierClaims:
    def test_a_free_tier_claim_carries_an_official_source(self):
        report = provider_report()
        for provider, row in report["providers"].items():
            if row["free_tier_documented"] == "yes":
                assert row["free_tier_source"].startswith("http"), (
                    f"{provider} claims a free tier without an official source URL"
                )

    def test_an_unverified_provider_is_not_claimed_free(self):
        # OpenAI has no documented free API tier; it must not be reported as free.
        assert provider_readiness("openai").free_tier_documented != "yes"

    def test_groq_free_plan_is_recorded_from_official_docs(self):
        row = provider_readiness("groq")
        assert row.free_tier_documented == "yes"
        assert "console.groq.com" in row.free_tier_source

    def test_the_report_says_it_did_not_probe_the_network(self):
        assert provider_report()["network_probed"] is False


class TestArtifactRecordsProviderReadiness:
    def test_the_artifact_carries_the_provider_block(self, feed):
        from tradingagents.llm_evaluation import build_evaluation

        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert "provider_readiness" in outcome.artifact
        block = outcome.artifact["provider_readiness"]
        assert "providers" in block and "free_tier_options" in block

    def test_no_credential_leaves_the_baseline_blocked(self, feed, monkeypatch):
        from tradingagents.llm_evaluation import build_evaluation

        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            monkeypatch.setenv(env, "placeholder")
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert outcome.status == "LLM BASELINE BLOCKED"
        assert outcome.artifact["genuine_inference"] is False

    def test_a_key_in_the_environment_still_does_not_leak_into_the_artifact(
        self, feed, monkeypatch
    ):
        from tradingagents.llm_evaluation import build_evaluation

        monkeypatch.setenv("GROQ_API_KEY", "gsk-" + "b" * 40)
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert "gsk-" + "b" * 40 not in json.dumps(outcome.artifact)


@pytest.fixture
def feed():
    from tradingagents.baseline.data import load_feed
    from tradingagents.baseline.protocol import DEFAULT_PROTOCOL as BASE

    return load_feed(BASE)
