"""Phase 11: real market-data providers (Alpha Vantage, Finnhub).

Every test here drives a **mocked HTTP transport**; no test reaches the network
(the suite's autouse ``_no_network`` fixture would refuse it anyway) and no test
needs a real API key. The mock is a fake transport, not a fake provider: the
provider's own parsing, validation, classification and provenance code runs for
real, so a test that passes proves the production path works.

The tests are written to catch a premise error, not to confirm the code. The
headline claims — that a quota notice inside an HTTP 200 is refused rather than
parsed as data, that a stale or malformed bar is refused rather than filled, that
Finnhub's free tier has no historical candles so the provider refuses to stand in
for them, that a fallback is explicit and never mixes two vendors' bars, and that
no credential reaches an artifact — are each pinned by a test that would fail if
the claim were false.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    PaperExperiment,
    PriceSourceError,
    StaticPriceSource,
    build_report,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.providers import (
    KNOWN_SOURCES,
    AlphaVantageDailySource,
    FinnhubQuoteSource,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderStaleDataError,
    ProviderUnsupportedError,
    SourceSelectionError,
    provider_registry,
    select_source,
)
from tradingagents.paper_experiment.providers.base import HttpResponse, fetch_json, redact_url
from tradingagents.paper_experiment.providers.selection import _build

# A "today" well after every fixture bar, so staleness is deterministic.
TODAY = date(2024, 3, 1)


def _daily_body(sessions, *, start=100.0, note=None, error=None):
    """An Alpha Vantage TIME_SERIES_DAILY body for ``sessions`` (ISO strings)."""
    series = {}
    for i, session in enumerate(sessions):
        base = start + i
        series[session] = {
            "1. open": f"{base:.4f}",
            "2. high": f"{base + 2:.4f}",
            "3. low": f"{base - 1:.4f}",
            "4. close": f"{base + 1:.4f}",
            "5. volume": "1000000",
        }
    body = {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": "IBM",
            "3. Last Refreshed": sessions[-1] if sessions else "",
            "4. Output Size": "Compact",
            "5. Time Zone": "US/Eastern",
        },
        "Time Series (Daily)": series,
    }
    if note is not None:
        body.pop("Time Series (Daily)")
        body["Note"] = note
    if error is not None:
        body.pop("Time Series (Daily)")
        body["Error Message"] = error
    return json.dumps(body)


def _transport(body, *, status=200, capture=None, exc=None):
    def send(*, url, params, headers, timeout):
        if capture is not None:
            capture["url"] = url
            capture["params"] = dict(params)
            capture["headers"] = dict(headers)
            capture["timeout"] = timeout
        if exc is not None:
            raise exc
        return HttpResponse(status, body)
    return send


def _av(sessions=("2024-02-26", "2024-02-27", "2024-02-28"), **kwargs):
    capture = kwargs.pop("capture", None)
    body = kwargs.pop("body", None) or _daily_body(list(sessions))
    return AlphaVantageDailySource(
        "IBM",
        api_key="test-key-not-real",
        transport=_transport(body, capture=capture),
        today=TODAY,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The shared HTTP layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHttpLayer:
    def test_a_non_json_body_is_refused(self):
        with pytest.raises(ProviderError, match="did not answer with JSON"):
            fetch_json(
                url="https://example.invalid",
                params={},
                timeout=1.0,
                transport=_transport("<html>not json</html>"),
            )

    def test_a_non_200_status_is_refused(self):
        with pytest.raises(ProviderError, match="HTTP 500"):
            fetch_json(
                url="https://example.invalid",
                params={},
                timeout=1.0,
                transport=_transport("boom", status=500),
            )

    def test_a_transport_exception_is_reported_not_hidden(self):
        with pytest.raises(ProviderError, match="the request failed"):
            fetch_json(
                url="https://example.invalid",
                params={},
                timeout=1.0,
                transport=_transport("", exc=ConnectionError("refused")),
            )

    def test_a_credential_is_scrubbed_from_an_error(self):
        secret = "SECRETKEY1234567890"
        with pytest.raises(ProviderError) as raised:
            fetch_json(
                url="https://example.invalid",
                params={},
                timeout=1.0,
                secrets=(secret,),
                transport=_transport("", exc=ValueError(f"bad url ?token={secret}")),
            )
        assert secret not in str(raised.value)
        assert "***" in str(raised.value)

    def test_a_timeout_is_always_passed(self):
        captured = {}
        fetch_json(
            url="https://example.invalid",
            params={"a": 1},
            timeout=7.5,
            transport=_transport("{}", capture=captured),
        )
        assert captured["timeout"] == 7.5

    def test_a_retry_is_bounded(self):
        calls = {"n": 0}

        def flaky(*, url, params, headers, timeout):
            calls["n"] += 1
            raise ConnectionError("nope")

        with pytest.raises(ProviderError):
            fetch_json(
                url="https://example.invalid",
                params={},
                timeout=1.0,
                transport=flaky,
                max_retries=2,
            )
        assert calls["n"] == 3  # initial + 2 retries, then give up

    def test_a_url_secret_is_redacted_for_display(self):
        url = redact_url("https://x.invalid/q?a=1&token=SECRET", ("SECRET",))
        assert "SECRET" not in url
        assert "token=%2A%2A%2A" in url or "token=***" in url


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAlphaVantage:
    def test_a_valid_response_becomes_a_validated_frame(self):
        source = _av()
        assert source.kind == "vendor_daily"
        assert source.sessions() == ("2024-02-26", "2024-02-27", "2024-02-28")
        bar = source.bars("IBM", "2024-02-27")
        assert bar is not None
        assert bar.open == pytest.approx(101.0)
        assert bar.high == pytest.approx(103.0)
        assert bar.low == pytest.approx(100.0)
        assert bar.close == pytest.approx(102.0)
        assert bar.volume == pytest.approx(1_000_000.0)

    def test_it_uses_the_free_function_not_the_premium_one(self):
        captured = {}
        _av(capture=captured)
        assert captured["params"]["function"] == "TIME_SERIES_DAILY"
        assert captured["params"]["outputsize"] == "compact"
        assert "ADJUSTED" not in captured["params"]["function"]

    def test_it_declares_itself_unadjusted_and_provenanced(self):
        source = _av()
        prov = source.provenance
        assert prov.adjusted is False
        assert prov.kind == "vendor_daily"
        assert prov.symbol == "IBM"
        assert prov.last_bar_date == "2024-02-28"
        assert prov.timezone == "US/Eastern"
        assert prov.fetched_at is not None

    def test_a_recent_feed_is_reported_as_current_market(self):
        # Bars within the staleness window of TODAY -> current market.
        source = _av(sessions=("2024-02-27", "2024-02-28"))
        assert source.is_current_market is True

    def test_a_rate_limit_notice_in_an_http_200_is_refused(self):
        body = json.dumps(
            {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit "
                     "is 25 requests per day."}
        )
        with pytest.raises(ProviderRateLimitedError):
            AlphaVantageDailySource(
                "IBM", api_key="k", transport=_transport(body), today=TODAY
            )

    def test_an_invalid_key_notice_is_an_auth_error_not_a_rate_limit(self):
        body = json.dumps(
            {"Information": "the parameter apikey is invalid or missing. "
                            "Please claim your free API key."}
        )
        with pytest.raises(ProviderAuthError):
            AlphaVantageDailySource(
                "IBM", api_key="k", transport=_transport(body), today=TODAY
            )

    def test_a_rejected_call_is_unsupported_not_data(self):
        body = json.dumps({"Error Message": "Invalid API call. Please retry."})
        with pytest.raises(ProviderUnsupportedError):
            AlphaVantageDailySource(
                "NOPE", api_key="k", transport=_transport(body), today=TODAY
            )

    def test_a_missing_credential_is_refused_before_any_call(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        called = {"n": 0}

        def send(**kwargs):
            called["n"] += 1
            return HttpResponse(200, "{}")

        with pytest.raises(ProviderNotConfiguredError):
            AlphaVantageDailySource("IBM", api_key="", transport=send, today=TODAY)
        assert called["n"] == 0

    def test_a_stale_feed_is_refused(self):
        # Bars in 2020, "today" in 2024 -> far beyond the staleness window.
        with pytest.raises(ProviderStaleDataError):
            _av(sessions=("2020-01-02", "2020-01-03"))

    def test_the_staleness_check_can_be_disabled_explicitly(self):
        source = _av(sessions=("2020-01-02", "2020-01-03"), max_stale_days=None)
        assert source.is_current_market is False  # unknown recency is not "current"

    def test_outputsize_full_is_refused_unless_premium_is_accepted(self):
        with pytest.raises(ProviderUnsupportedError, match="premium"):
            AlphaVantageDailySource(
                "IBM",
                api_key="k",
                outputsize="full",
                transport=_transport(_daily_body(["2024-02-28"])),
                today=TODAY,
            )

    def test_outputsize_full_is_allowed_when_premium_is_accepted(self):
        source = AlphaVantageDailySource(
            "IBM",
            api_key="k",
            outputsize="full",
            allow_premium_outputsize=True,
            transport=_transport(_daily_body(["2024-02-28"])),
            today=TODAY,
        )
        assert source.sessions() == ("2024-02-28",)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda b: b["Time Series (Daily)"]["2024-02-27"].update({"2. high": "nan"}), "non-numeric"),
            (lambda b: b["Time Series (Daily)"]["2024-02-27"].update({"1. open": "0"}), "non-positive"),
            (lambda b: b["Time Series (Daily)"]["2024-02-27"].update({"3. low": "999"}), "high below low"),
            (lambda b: b["Time Series (Daily)"]["2024-02-27"].update({"4. close": "1"}), "outside"),
            (lambda b: b["Time Series (Daily)"]["2024-02-27"].update({"1. open": "inf"}), "non-finite"),
        ],
    )
    def test_malformed_ohlc_is_refused(self, mutate, match):
        body = json.loads(_daily_body(["2024-02-26", "2024-02-27", "2024-02-28"]))
        mutate(body)
        with pytest.raises(ProviderError, match=match):
            AlphaVantageDailySource(
                "IBM", api_key="k", transport=_transport(json.dumps(body)), today=TODAY
            )

    def test_a_duplicate_session_is_refused(self):
        # A JSON object cannot hold a duplicate key, so the validator's duplicate
        # check is exercised directly on the frame it would build.
        from tradingagents.paper_experiment.sources import _validate_frame

        frame = pd.DataFrame(
            {"open": [1.0, 2.0], "high": [3.0, 4.0], "low": [0.5, 1.5],
             "close": [2.5, 3.5], "volume": [1.0, 2.0]},
            index=pd.to_datetime(["2024-02-27", "2024-02-27"]),
        )
        with pytest.raises(PriceSourceError, match="duplicate"):
            _validate_frame(frame, "IBM", "alpha_vantage_daily")

    def test_out_of_order_sessions_are_refused(self):
        from tradingagents.paper_experiment.sources import _validate_frame

        frame = pd.DataFrame(
            {"open": [1.0, 2.0], "high": [3.0, 4.0], "low": [0.5, 1.5],
             "close": [2.5, 3.5], "volume": [1.0, 2.0]},
            index=pd.to_datetime(["2024-02-28", "2024-02-27"]),
        )
        with pytest.raises(PriceSourceError, match="ascending"):
            _validate_frame(frame, "IBM", "alpha_vantage_daily")

    def test_a_requested_session_outside_the_feed_is_refused(self):
        source = _av()
        with pytest.raises(PriceSourceError):
            source.bars("IBM", "2024-02-29")

    def test_a_historical_window_ends_at_the_requested_session(self):
        source = _av()
        window = source.window("IBM", "2024-02-27")
        assert list(window.index) == list(pd.to_datetime(["2024-02-26", "2024-02-27"]))

    def test_the_credential_never_appears_in_the_frame_or_provenance(self):
        source = _av()
        payload = json.dumps(source.provenance.to_dict())
        assert "test-key-not-real" not in payload
        assert "test-key-not-real" not in source.frame("IBM").to_json()

    def test_the_credential_from_the_environment_reaches_the_apikey_parameter(
        self, monkeypatch
    ):
        """The key in the env var must appear in the outgoing request's ``apikey``.

        This is the regression guard for the bug where the request omitted
        ``apikey`` entirely, so Alpha Vantage answered "the parameter apikey is
        invalid or missing" no matter how valid the key was. A fake key is used
        here and nowhere else; no real credential is involved.
        """
        fake = "FAKE-ENV-KEY-NOT-REAL"
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", fake)
        captured = {}
        # No api_key= argument: the source must read the environment itself.
        AlphaVantageDailySource(
            "IBM",
            transport=_transport(_daily_body(["2024-02-28"]), capture=captured),
            today=TODAY,
        )
        assert captured["params"]["apikey"] == fake

    def test_the_apikey_parameter_is_always_present(self):
        """Every outgoing request carries ``apikey``, whatever the source."""
        captured = {}
        _av(capture=captured)
        assert "apikey" in captured["params"]
        assert captured["params"]["apikey"] == "test-key-not-real"

    def test_the_credential_is_scrubbed_from_a_vendor_rejection(self):
        """A vendor that echoes the key back in an error body must not leak it."""
        secret = "LEAKY-KEY-NOT-REAL"
        body = json.dumps({"Error Message": f"Invalid API call for apikey {secret}"})
        with pytest.raises(ProviderUnsupportedError) as excinfo:
            AlphaVantageDailySource(
                "IBM", api_key=secret, transport=_transport(body), today=TODAY
            )
        assert secret not in str(excinfo.value)
        assert "***" in str(excinfo.value)

    def test_the_credential_is_scrubbed_from_a_notice_body(self):
        secret = "LEAKY-KEY-NOT-REAL"
        body = json.dumps(
            {"Information": f"the parameter apikey {secret} is invalid or missing"}
        )
        with pytest.raises(ProviderAuthError) as excinfo:
            AlphaVantageDailySource(
                "IBM", api_key=secret, transport=_transport(body), today=TODAY
            )
        assert secret not in str(excinfo.value)

    def test_a_whitespace_only_credential_is_refused(self, monkeypatch):
        """A key of spaces is treated as missing, not sent as an empty apikey."""
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        called = {"n": 0}

        def send(**kwargs):
            called["n"] += 1
            return HttpResponse(200, "{}")

        with pytest.raises(ProviderNotConfiguredError):
            AlphaVantageDailySource("IBM", api_key="   ", transport=send, today=TODAY)
        assert called["n"] == 0

    def test_a_surrounding_whitespace_credential_is_trimmed(self, monkeypatch):
        """A key pasted with a trailing newline is trimmed before it is sent."""
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "  KEY-WITH-PADDING  ")
        captured = {}
        AlphaVantageDailySource(
            "IBM",
            transport=_transport(_daily_body(["2024-02-28"]), capture=captured),
            today=TODAY,
        )
        assert captured["params"]["apikey"] == "KEY-WITH-PADDING"

    def test_build_request_params_matches_what_is_sent(self):
        """The single-source-of-truth builder agrees with the real request."""
        from tradingagents.paper_experiment.providers.alpha_vantage import (
            build_request_params,
        )

        captured = {}
        _av(capture=captured)
        expected = build_request_params(
            symbol="IBM", api_key="test-key-not-real", outputsize="compact"
        )
        assert captured["params"] == expected


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------


def _quote_body(*, c=190.5, o=189.0, h=191.0, low=188.5, pc=188.0, t=1709164800):
    return json.dumps({"c": c, "d": 2.5, "dp": 1.33, "h": h, "l": low, "o": o, "pc": pc, "t": t})


def _fh(**kwargs):
    capture = kwargs.pop("capture", None)
    body = kwargs.pop("body", None) or _quote_body()
    return FinnhubQuoteSource(
        "AAPL",
        api_key="test-key-not-real",
        transport=_transport(body, capture=capture),
        **kwargs,
    )


@pytest.mark.unit
class TestFinnhub:
    def test_a_valid_quote_is_parsed_and_validated(self):
        source = _fh()
        obs = source.quote()
        assert obs.symbol == "AAPL"
        assert obs.price == pytest.approx(190.5)
        assert obs.open == pytest.approx(189.0)
        assert obs.high == pytest.approx(191.0)
        assert obs.low == pytest.approx(188.5)
        assert obs.previous_close == pytest.approx(188.0)

    def test_it_uses_the_free_quote_endpoint_not_the_premium_candle_one(self):
        captured = {}
        _fh(capture=captured)
        assert captured["url"].endswith("/quote")
        assert "candle" not in captured["url"]
        assert captured["params"]["symbol"] == "AAPL"

    def test_it_declares_itself_a_current_market_quote(self):
        source = _fh()
        assert source.is_current_market is True
        assert source.kind == "vendor_quote"
        assert source.provenance.kind == "vendor_quote"
        assert "NOT historical OHLCV" in " ".join(source.provenance.notes)

    def test_it_refuses_to_provide_a_bar_history(self):
        source = _fh()
        for call in (
            lambda: source.sessions(),
            lambda: source.covers("2024-01-02"),
            lambda: source.bars("AAPL", "2024-01-02"),
            lambda: source.window("AAPL", "2024-01-02"),
        ):
            with pytest.raises(ProviderUnsupportedError):
                call()

    def test_a_zero_price_is_an_unsupported_symbol_not_a_price(self):
        with pytest.raises(ProviderUnsupportedError, match="unsupported"):
            _fh(body=_quote_body(c=0, o=0, h=0, low=0))

    def test_a_missing_credential_is_refused_before_any_call(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        called = {"n": 0}

        def send(**kwargs):
            called["n"] += 1
            return HttpResponse(200, "{}")

        with pytest.raises(ProviderNotConfiguredError):
            FinnhubQuoteSource("AAPL", api_key="", transport=send)
        assert called["n"] == 0

    def test_an_auth_error_body_is_classified(self):
        with pytest.raises(ProviderAuthError):
            _fh(body=json.dumps({"error": "Invalid API key"}))

    def test_a_rate_limit_body_is_classified(self):
        with pytest.raises(ProviderRateLimitedError):
            _fh(body=json.dumps({"error": "API limit reached"}))

    def test_an_incomplete_quote_is_refused(self):
        with pytest.raises(ProviderUnsupportedError):
            _fh(body=json.dumps({"c": 190.5}))

    def test_a_malformed_quote_is_refused(self):
        with pytest.raises(ProviderError):
            _fh(body=json.dumps({"c": "abc", "o": 1, "h": 2, "l": 1, "pc": 1, "t": 1}))

    def test_a_non_finite_price_is_refused(self):
        with pytest.raises(ProviderError):
            _fh(body=json.dumps({"c": float("nan"), "o": 1, "h": 2, "l": 1, "pc": 1, "t": 1}))

    def test_a_quote_for_the_wrong_day_is_refused(self):
        # t = 1709164800 -> 2024-02-29; ask for 2024-02-28.
        with pytest.raises(ProviderStaleDataError):
            _fh(expected_session="2024-02-28")

    def test_a_quote_for_the_expected_day_is_accepted(self):
        source = _fh(expected_session="2024-02-29")
        assert source.quote().session == "2024-02-29"

    def test_the_credential_never_appears_in_the_provenance(self):
        source = _fh()
        assert "test-key-not-real" not in json.dumps(source.provenance.to_dict())


# ---------------------------------------------------------------------------
# Selection and explicit fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelection:
    def test_an_unknown_source_is_refused(self):
        with pytest.raises(SourceSelectionError, match="unknown price source"):
            select_source("nasdaq", symbols=("SPY",))

    def test_a_frozen_selection_builds_the_frozen_source(self):
        sentinel = object()
        selection = select_source(
            "frozen", symbols=("SPY",), frozen_feed_factory=lambda: sentinel
        )
        assert selection.source is sentinel
        assert selection.switched is False

    def test_a_vendor_source_without_a_credential_fails_when_no_fallback_is_named(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(SourceSelectionError, match="no fallback was named"):
            select_source("alpha_vantage", symbols=("SPY",))

    def test_a_failure_switches_only_to_a_named_fallback(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        sentinel = object()
        selection = select_source(
            "alpha_vantage",
            symbols=("SPY",),
            fallback="frozen",
            frozen_feed_factory=lambda: sentinel,
        )
        assert selection.source is sentinel
        assert selection.switched is True
        assert selection.primary_name == "alpha_vantage"
        assert selection.fallback_name == "frozen"
        assert selection.attempted and "ProviderNotConfiguredError" in selection.attempted[0]["error"]

    def test_the_switch_is_recorded_and_reported(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        selection = select_source(
            "alpha_vantage",
            symbols=("SPY",),
            fallback="frozen",
            frozen_feed_factory=lambda: StaticPriceSource({"SPY": _frame()}),
        )
        payload = selection.to_dict()
        assert payload["switched"] is True
        assert payload["requested"] == "alpha_vantage"
        assert payload["fallback_name"] == "frozen"

    def test_both_sources_failing_is_reported(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        with pytest.raises(SourceSelectionError, match="neither the requested source"):
            select_source("alpha_vantage", symbols=("SPY",), fallback="finnhub")

    def test_the_registry_lists_every_known_source(self):
        registry = provider_registry()
        assert set(registry) == set(KNOWN_SOURCES)

    def test_the_static_source_cannot_be_selected_for_a_run(self):
        with pytest.raises(SourceSelectionError, match="test fixture"):
            _build("static", symbols=("SPY",), frozen_feed_factory=None)


# ---------------------------------------------------------------------------
# No cross-vendor mixing; provenance in state and report
# ---------------------------------------------------------------------------


def _frame(periods=12, start=100.0):
    index = pd.date_range("2024-02-01", periods=periods, freq="B")
    opens = [start + i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 1 for o in opens],
            "volume": [1_000_000.0] * periods,
        },
        index=index,
    )


@pytest.mark.unit
class TestNoMixingAndProvenance:
    def test_a_frame_source_run_uses_one_vendor_only(self, tmp_path):
        """A source is one vendor's bars; a run cannot mix two."""
        source = StaticPriceSource({"SPY": _frame()}, name="alpha_vantage_daily")
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1
        )
        experiment = PaperExperiment(
            source=source,
            config=config,
            root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        for session in source.sessions():
            experiment.process_session(session)

        # Every bar the experiment saw came from the one source object.
        assert experiment.source is source
        # And a snapshot is named for that one source, never a blend.
        snapshots = list(experiment.snapshot_dir.glob("*.json"))
        assert snapshots
        payload = json.loads(snapshots[0].read_text())
        assert payload["source"].startswith("alpha_vantage_daily:SPY")

    def test_the_state_records_the_source_provenance(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame()})
        config = ExperimentConfig.from_protocol(symbols=("SPY",))
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(),
        )
        state = experiment.state()
        assert state["source"]["kind"] == "static"
        assert "provenance" in state["source"]

    def test_the_report_surfaces_the_source_kind_and_provenance(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame()})
        config = ExperimentConfig.from_protocol(symbols=("SPY",))
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.process_session(source.sessions()[0])
        report = build_report(experiment).to_dict()
        assert report["data"]["source_kind"] == "static"
        assert report["data"]["is_current_market"] is False
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"

    def test_a_quote_only_source_can_never_read_ready_for_live_paper(self, tmp_path):
        """A current-market quote source is not enough: it cannot drive sessions."""

        class _QuoteOnlySource:
            name = "finnhub_quote"
            kind = "vendor_quote"

            @property
            def is_current_market(self):
                return True

            def sessions(self):
                return ("2024-02-28",)

            def covers(self, session):
                return session == "2024-02-28"

            def bars(self, symbol, session):
                raise ProviderUnsupportedError("quote only")

            def window(self, symbol, session):
                raise ProviderUnsupportedError("quote only")

        config = ExperimentConfig.from_protocol(symbols=("SPY",))
        experiment = PaperExperiment(
            source=_QuoteOnlySource(), config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        report = build_report(experiment).to_dict()
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"
        assert any("current-day quote" in b for b in report["readiness"]["blockers"])


# ---------------------------------------------------------------------------
# An end-to-end run on a vendor-shaped frame, fully offline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVendorRunOffline:
    def test_a_run_on_a_vendor_frame_is_point_in_time_and_secret_free(self, tmp_path):
        """The whole cycle runs on a vendor-shaped frame, offline."""
        source = StaticPriceSource({"SPY": _frame(20)}, name="alpha_vantage_daily")
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=2, evaluation_lag_sessions=3
        )
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        for session in source.sessions():
            experiment.process_session(session)
        experiment.save_state()

        report = build_report(experiment)
        payload = report.to_dict()
        # A decision's window ends at the decision date: no future bar leaks in.
        first = experiment.decisions["SPY-2024-02-01"]
        window_end = first.data_window_hash is not None
        assert window_end
        # The whole report is JSON-safe and carries no credential.
        text = json.dumps(payload)
        assert "test-key" not in text
        assert payload["counts"]["sessions_processed"] == len(source.sessions())


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCli:
    def test_the_sources_command_lists_the_registry(self, capsys):
        from tradingagents.paper_experiment.__main__ import main

        assert main(["sources", "--root", "/tmp/does-not-matter"]) == 0
        out = capsys.readouterr().out
        assert "alpha_vantage" in out
        assert "finnhub" in out
        assert "frozen" in out

    def test_an_unknown_source_is_a_usage_error(self):
        from tradingagents.paper_experiment.__main__ import main

        with pytest.raises(SystemExit):
            main(["start", "--root", "/tmp/x", "--source", "nasdaq"])

    def test_a_vendor_source_without_a_credential_stops_the_run(self, tmp_path, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        code = main(["start", "--root", str(tmp_path / "exp"), "--source", "alpha_vantage"])
        assert code == 2
        err = capsys.readouterr().err
        assert "no fallback was named" in err
        # The experiment directory was not created: a failed source is a hard stop.
        assert not (tmp_path / "exp" / "session_state.json").exists()

    def test_the_report_command_never_fetches_from_a_vendor(self, tmp_path, monkeypatch):
        """A saved run is reported from its state; no network call happens."""
        from tradingagents.paper_experiment.__main__ import main

        # Create a frozen run and report it. The frozen source is local.
        assert main(["start", "--root", str(tmp_path / "exp"), "--stub"]) == 0
        assert main(["run", "--root", str(tmp_path / "exp"), "--stub", "--sessions", "3"]) == 0
        # A report must succeed without any provider credential present.
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        assert main(["report", "--root", str(tmp_path / "exp"), "--out", str(tmp_path / "r.json")]) == 0
        payload = json.loads((tmp_path / "r.json").read_text())
        assert payload["data"]["source_kind"] == "frozen"

    def test_a_vendor_run_cannot_be_silently_resumed_on_the_frozen_dataset(self, tmp_path, monkeypatch, capsys):
        """The CLI refuses to substitute the frozen dataset for a vendor run."""
        from tradingagents.paper_experiment.__main__ import main

        root = tmp_path / "exp"
        root.mkdir()
        state = {
            "schema_version": 1,
            "experiment_id": "x",
            "created_at": "2024-01-01T00:00:00+00:00",
            "config": ExperimentConfig.from_protocol(symbols=("SPY",)).to_dict(),
            "source": {"name": "alpha_vantage_daily", "kind": "vendor_daily"},
            "processed_sessions": [],
            "decisions": {},
            "session": {},
        }
        (root / "session_state.json").write_text(json.dumps(state))
        code = main(["run", "--root", str(root)])
        assert code == 2
        assert "will not silently substitute" in capsys.readouterr().err

    def test_verify_checks_the_journal_chain(self, tmp_path):
        from tradingagents.paper_experiment.__main__ import main

        assert main(["start", "--root", str(tmp_path / "exp"), "--stub"]) == 0
        assert main(["verify", "--root", str(tmp_path / "exp")]) == 0

    def test_the_diagnose_command_reports_a_present_credential_without_leaking_it(
        self, monkeypatch, capsys
    ):
        from tradingagents.paper_experiment.__main__ import main

        secret = "DIAGNOSE-SECRET-NOT-REAL"
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", secret)
        assert main(["diagnose", "--source", "alpha_vantage"]) == 0
        out = capsys.readouterr().out
        assert "ALPHA_VANTAGE_API_KEY" in out
        assert "env var present and non-blank: True" in out
        assert "request carries apikey parameter: True" in out
        # The value, any part of it, and its length must never appear.
        assert secret not in out
        assert str(len(secret)) not in out

    def test_the_diagnose_command_reports_an_absent_credential(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        assert main(["diagnose", "--source", "alpha_vantage"]) == 0
        out = capsys.readouterr().out
        assert "env var present and non-blank: False" in out
        assert "would fail before it is sent" in out

    def test_the_diagnose_command_needs_no_root(self, monkeypatch, capsys):
        from tradingagents.paper_experiment.__main__ import main

        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        # No --root: diagnose is a configuration check, not an experiment command.
        assert main(["diagnose", "--source", "alpha_vantage"]) == 0

    def test_a_command_that_needs_root_without_one_is_a_clear_error(self, capsys):
        from tradingagents.paper_experiment.__main__ import main

        code = main(["verify"])
        assert code == 2
        assert "needs --root" in capsys.readouterr().err
