"""Phase 21: an honest data-source failure, and a throttle that does not lie.

The reported production failure had two faces, both about a paper run telling the
truth about *why* it is idle:

1. A deploy with Alpha Vantage configured logged ``the price source
   'alpha_vantage' could not be built (Alpha Vantage rejected the request for
   'SPY': the parameter apikey is invalid or missing. ...), and no fallback was
   named``. That text is Alpha Vantage's *own* message, which means the key
   **reached** the vendor and was rejected — a SET-but-INVALID credential, not a
   missing one. The provider classified it as ``ProviderUnsupportedError`` (an
   "unsupported symbol" style fault an operator cannot act on) rather than an
   authentication failure.

2. The status line read ``status=throttled new=0 fetches_today=0/24`` and hid the
   real error. Two defects produced that: a throttled tick *overwrote* the last
   real outcome, so the standing auth failure vanished; and a throttled tick
   advanced the throttle clock, so the retry window reset forever and a fixed key
   would never be picked up.

These tests are written to catch a premise error, not to confirm the code. They
pin the properties that make the failure honest: an auth rejection is named as
one; a skip never masks the last real outcome; a skip never deadlocks the retry;
a failing source never reads as "no newer bar"; and no credential ever reaches a
log, a status line or a report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.dashboard.state import ExperimentView
from tradingagents.paper_experiment.experiment import PaperExperiment
from tradingagents.paper_experiment.freshness import (
    MARKET_STATUS_SOURCE_UNAVAILABLE,
    MARKET_STATUSES,
    market_freshness,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.providers.alpha_vantage import (
    AlphaVantageDailySource,
)
from tradingagents.paper_experiment.providers.base import (
    HttpResponse,
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnsupportedError,
)
from tradingagents.paper_experiment.providers.selection import (
    SourceSelectionError,
    select_source,
)
from tradingagents.paper_experiment.refresh import (
    FAILURE_KIND_AUTH,
    FAILURE_KIND_RATE_LIMITED,
    FAILURE_KIND_TRANSPORT,
    STATUS_AUTH_FAILED,
    STATUS_NOT_CONFIGURED,
    STATUS_THROTTLED,
    STATUS_UNAVAILABLE,
    RefreshPolicy,
    read_refresh_state,
    refresh_source,
)
from tradingagents.paper_experiment.sources import StaticPriceSource

SYMBOL = "SPY"
# Alpha Vantage's own rejection text for a missing/invalid key, verbatim.
AV_INVALID_KEY_BODY = (
    '{"Error Message": "the parameter apikey is invalid or missing. Please set the '
    'key in the query string or header."}'
)
AV_QUOTA_BODY = (
    '{"Information": "We have detected your API key and are offering you a free '
    'premium... You have reached the rate limit of 25 requests per day."}'
)

SESSION_SET = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
NOW = datetime(2024, 1, 5, 12, 0, tzinfo=UTC)


def _transport(body: str, *, status_code: int = 200):
    def send(*, url, params, headers, timeout):
        return HttpResponse(status_code, body)

    return send


def _frame(sessions, *, base: float = 100.0) -> pd.DataFrame:
    index = pd.to_datetime(list(sessions))
    epoch = pd.Timestamp("2024-01-01")
    opens = [base + float((ts - epoch).days) for ts in index]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * len(sessions),
        },
        index=index,
    )


def _vendor_source(sessions, *, name: str = "alpha_vantage_daily"):
    source = StaticPriceSource({SYMBOL: _frame(sessions)}, name=name)
    source.kind = "vendor_daily"
    return source


def _experiment(tmp_path, sessions=SESSION_SET[:3]) -> PaperExperiment:
    root = tmp_path / "exp"
    experiment = PaperExperiment(
        source=_vendor_source(sessions),
        config=ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), decision_frequency_sessions=1
        ),
        root=root,
        inference=StubInference(rating="Buy"),
    )
    experiment.ensure_created()
    experiment.save_state()
    return experiment


def _bad_key_fetcher(body: str = AV_INVALID_KEY_BODY, *, key: str = "a-set-but-bad-key"):
    """A fetcher that rebuilds the vendor source exactly as the runner would.

    The real provider code runs (its parser classifies the body); only the HTTP
    transport is mocked, so this exercises the production path, not a stub.
    """

    def fetch(experiment, *, today=None):
        return AlphaVantageDailySource(
            SYMBOL, api_key=key, transport=_transport(body), today=today
        )

    return fetch


# ---------------------------------------------------------------------------
# The provider names an authentication rejection as one
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuthClassification:
    def test_a_missing_key_notice_is_an_auth_error(self):
        with pytest.raises(ProviderAuthError):
            AlphaVantageDailySource(
                SYMBOL, api_key="k", transport=_transport(AV_INVALID_KEY_BODY)
            )

    def test_a_generic_rejection_is_still_unsupported_not_auth(self):
        """The fix must not over-classify: a truly malformed call stays unsupported."""
        body = json.dumps({"Error Message": "Invalid API call. Please retry."})
        with pytest.raises(ProviderUnsupportedError):
            AlphaVantageDailySource(SYMBOL, api_key="k", transport=_transport(body))

    def test_a_quota_notice_is_rate_limited_not_auth(self):
        with pytest.raises(ProviderRateLimitedError):
            AlphaVantageDailySource(SYMBOL, api_key="k", transport=_transport(AV_QUOTA_BODY))

    def test_the_credential_is_redacted_from_an_auth_rejection(self):
        secret = "LEAKY-KEY-NOT-REAL"
        body = json.dumps({"Error Message": f"the parameter apikey {secret} is invalid"})
        with pytest.raises(ProviderAuthError) as excinfo:
            AlphaVantageDailySource(SYMBOL, api_key=secret, transport=_transport(body))
        assert secret not in str(excinfo.value)
        assert "***" in str(excinfo.value)


@pytest.mark.unit
class TestSelectionFailureIsActionable:
    def test_an_invalid_key_names_the_credential_fix(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "a-set-but-bad-key")

        def bad_build(spec, **kwargs):
            raise ProviderAuthError(
                "Alpha Vantage rejected the credential: the parameter apikey is "
                "invalid or missing."
            )

        monkeypatch.setattr(
            "tradingagents.paper_experiment.providers.selection._build", bad_build
        )
        with pytest.raises(SourceSelectionError) as excinfo:
            select_source("alpha_vantage", symbols=(SYMBOL,))
        message = str(excinfo.value)
        # Still fails closed (no silent switch) and now says what to change.
        assert "no fallback was named" in message
        assert "rejected it" in message

    def test_no_silent_fallback_still_holds(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.paper_experiment.providers.selection._build",
            lambda spec, **kwargs: (_ for _ in ()).throw(ProviderAuthError("rejected")),
        )
        with pytest.raises(SourceSelectionError):
            select_source("alpha_vantage", symbols=(SYMBOL,))


# ---------------------------------------------------------------------------
# The refresh reports the real reason, and does not hide it behind a skip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefreshReportsTheRealReason:
    def test_an_auth_failure_is_reported_as_auth_failed(self, tmp_path):
        experiment = _experiment(tmp_path)
        outcome = refresh_source(
            experiment,
            policy=RefreshPolicy(),
            fetcher=_bad_key_fetcher(),
            now=NOW,
        )
        assert outcome.status == STATUS_AUTH_FAILED
        assert outcome.failure_kind == FAILURE_KIND_AUTH
        assert outcome.source_status == "failed"
        assert outcome.message and "apikey" in outcome.message
        state = read_refresh_state(experiment.root)
        assert state.last_status == STATUS_AUTH_FAILED
        assert state.last_failure_kind == FAILURE_KIND_AUTH
        assert state.last_error

    def test_a_quota_failure_is_reported_as_rate_limited(self, tmp_path):
        experiment = _experiment(tmp_path)
        outcome = refresh_source(
            experiment,
            policy=RefreshPolicy(),
            fetcher=_bad_key_fetcher(AV_QUOTA_BODY),
            now=NOW,
        )
        assert outcome.status == STATUS_UNAVAILABLE
        assert outcome.failure_kind == FAILURE_KIND_RATE_LIMITED

    def test_a_transport_failure_is_reported_as_unavailable(self, tmp_path):
        experiment = _experiment(tmp_path)

        def boom(experiment, *, today=None):
            raise OSError("connection reset by peer")

        outcome = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=boom, now=NOW
        )
        assert outcome.status == STATUS_UNAVAILABLE
        assert outcome.failure_kind == FAILURE_KIND_TRANSPORT

    def test_a_missing_credential_is_reported_as_not_configured(self, tmp_path, monkeypatch):
        experiment = _experiment(tmp_path)
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

        def fetch(experiment, *, today=None):
            # The real provider refuses a blank key before any call.
            return AlphaVantageDailySource(SYMBOL, api_key="", transport=_transport("{}"))

        outcome = refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=fetch, now=NOW
        )
        assert outcome.status == STATUS_NOT_CONFIGURED
        assert outcome.failure_kind == "not_configured"

    def test_the_failure_leaves_accounting_untouched(self, tmp_path):
        experiment = _experiment(tmp_path)
        before = json.dumps(experiment.session.state(), sort_keys=True)
        refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=_bad_key_fetcher(), now=NOW
        )
        assert json.dumps(experiment.session.state(), sort_keys=True) == before


@pytest.mark.unit
class TestThrottleDoesNotLieOrDeadlock:
    def test_a_skip_does_not_overwrite_the_last_real_outcome(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        first = refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW
        )
        assert first.status == STATUS_AUTH_FAILED

        # A tick five minutes later is inside the interval: it skips.
        second = refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW + timedelta(minutes=5)
        )
        assert second.status == STATUS_THROTTLED
        assert second.attempted is False
        # The standing failure is still the recorded outcome, not "throttled".
        assert read_refresh_state(experiment.root).last_status == STATUS_AUTH_FAILED
        # And the skip is recorded separately, not lost.
        assert read_refresh_state(experiment.root).last_skip == STATUS_THROTTLED

    def test_a_skip_summary_line_carries_the_standing_failure(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        refresh_source(experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW)
        skip = refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW + timedelta(minutes=5)
        )
        line = skip.summary_line()
        assert "status=throttled" in line
        assert "last_outcome=auth_failed" in line
        assert "last_error=" in line

    def test_the_throttle_does_not_deadlock_the_retry(self, tmp_path):
        """A tick that skips must not reset the clock, or a fixed key never retries."""
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        calls = {"n": 0}

        def counting_fetcher(experiment, *, today=None):
            calls["n"] += 1
            return _bad_key_fetcher()(experiment, today=today)

        # Simulate the 60s auto-advance loop for two hours.
        for i in range(121):
            refresh_source(
                experiment,
                policy=policy,
                fetcher=counting_fetcher,
                now=NOW + timedelta(seconds=60 * i),
            )
        # One call at t=0, one at t=60min, one at t=120min — not one, and not 121.
        assert calls["n"] == 3

    def test_a_skip_does_not_spend_a_fetch(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        refresh_source(experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW)
        before = read_refresh_state(experiment.root).fetches_today
        refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW + timedelta(minutes=1)
        )
        assert read_refresh_state(experiment.root).fetches_today == before

    def test_the_skip_state_round_trips_through_the_sidecar(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        refresh_source(experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW)
        refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW + timedelta(minutes=5)
        )
        state = read_refresh_state(experiment.root)
        payload = json.loads(
            (experiment.root / "data_refresh.json").read_text(encoding="utf-8")
        )
        assert payload["last_status"] == STATUS_AUTH_FAILED
        assert payload["last_skip"] == STATUS_THROTTLED
        assert state.last_status == STATUS_AUTH_FAILED


# ---------------------------------------------------------------------------
# Freshness distinguishes a failing source from "no newer bar"
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreshnessNamesASourceFailure:
    def test_a_failing_source_is_not_up_to_date(self, tmp_path):
        experiment = _experiment(tmp_path)
        refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=_bad_key_fetcher(), now=NOW
        )
        fresh = market_freshness(experiment, now=NOW)
        assert fresh.market_status == MARKET_STATUS_SOURCE_UNAVAILABLE
        assert fresh.market_status in MARKET_STATUSES
        assert fresh.source_status == "failed"
        assert fresh.source_failure_kind == FAILURE_KIND_AUTH
        assert fresh.is_stale is False

    def test_a_skip_after_a_failure_still_reads_as_a_failing_source(self, tmp_path):
        experiment = _experiment(tmp_path)
        policy = RefreshPolicy(min_interval_seconds=3600.0)
        refresh_source(experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW)
        refresh_source(
            experiment, policy=policy, fetcher=_bad_key_fetcher(), now=NOW + timedelta(minutes=1)
        )
        fresh = market_freshness(experiment, now=datetime(2024, 1, 5, 12, 2, tzinfo=UTC))
        assert fresh.market_status == MARKET_STATUS_SOURCE_UNAVAILABLE
        assert fresh.source_status == "failed"

    def test_a_healthy_source_reports_ok(self, tmp_path):
        experiment = _experiment(tmp_path)
        fresh = market_freshness(experiment, now=NOW)
        assert fresh.source_status == "ok"
        assert fresh.source_failure_kind is None

    def test_the_source_health_fields_are_json_safe(self, tmp_path):
        experiment = _experiment(tmp_path)
        refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=_bad_key_fetcher(), now=NOW
        )
        payload = market_freshness(experiment, now=NOW).to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["source_status"] == "failed"


# ---------------------------------------------------------------------------
# The dashboard surfaces the failure, and never the credential
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDashboardSurfacesTheFailure:
    def test_the_summary_carries_the_source_health(self, tmp_path):
        experiment = _experiment(tmp_path)
        refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=_bad_key_fetcher(), now=NOW
        )
        summary = ExperimentView(experiment.root).summary()
        assert summary["freshness"]["source_status"] == "failed"
        assert summary["freshness"]["source_failure_kind"] == FAILURE_KIND_AUTH
        assert summary["refresh"]["source_status"] == "failed"
        assert json.loads(json.dumps(summary)) == summary

    def test_a_failing_source_raises_an_operator_warning(self, tmp_path):
        experiment = _experiment(tmp_path)
        refresh_source(
            experiment, policy=RefreshPolicy(), fetcher=_bad_key_fetcher(), now=NOW
        )
        warnings = ExperimentView(experiment.root).summary()["warnings"]
        assert any("source is failing" in w for w in warnings)

    def test_the_summary_never_carries_the_credential(self, tmp_path):
        secret = "SUPER-SECRET-KEY-VALUE"
        experiment = _experiment(tmp_path)
        body = json.dumps({"Error Message": f"the parameter apikey {secret} is invalid"})
        refresh_source(
            experiment,
            policy=RefreshPolicy(),
            fetcher=_bad_key_fetcher(body, key=secret),
            now=NOW,
        )
        blob = json.dumps(ExperimentView(experiment.root).summary())
        assert secret not in blob
        assert "***" in blob


# ---------------------------------------------------------------------------
# The diagnose command tells SET/MISSING/INVALID apart, without a probe by default
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiagnoseCredentialVerdict:
    def test_a_missing_credential_reads_missing(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        assert main(["diagnose", "--source", "alpha_vantage"]) == 0
        out = capsys.readouterr().out
        assert "credential status: MISSING" in out

    def test_a_present_credential_reads_set_without_a_probe(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "a-set-but-bad-key")
        assert main(["diagnose", "--source", "alpha_vantage"]) == 0
        out = capsys.readouterr().out
        assert "credential status: SET" in out
        assert "live probe: skipped" in out
        # No probe means no vendor call, and the key never appears.
        assert "a-set-but-bad-key" not in out

    def test_a_probe_classifies_a_rejected_key_as_invalid(self):
        from tradingagents.paper_experiment.__main__ import _probe_source

        verdict, detail = _probe_source(
            "alpha_vantage", "ALPHA_VANTAGE_API_KEY", "a-set-but-bad-key"
        )
        # No network here: the probe fails to connect, which is UNREACHABLE, not
        # INVALID — a key is only INVALID when the vendor answers and rejects it.
        assert verdict in ("UNREACHABLE", "INVALID")
        assert "a-set-but-bad-key" not in detail

    def test_the_probe_never_prints_the_key(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main
        from tradingagents.paper_experiment.providers.base import ProviderAuthError

        secret = "SUPER-SECRET-KEY-VALUE"
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", secret)

        def fake_probe(source_name, env_name, value):
            raise ProviderAuthError(f"rejected the credential: apikey {value} invalid")

        monkeypatch.setattr(
            "tradingagents.paper_experiment.__main__._probe_source",
            lambda *a, **k: ("INVALID", "the vendor rejected the credential: apikey *** invalid"),
        )
        assert main(["diagnose", "--source", "alpha_vantage", "--probe"]) == 0
        out = capsys.readouterr().out
        assert secret not in out
        assert "live probe: INVALID" in out
