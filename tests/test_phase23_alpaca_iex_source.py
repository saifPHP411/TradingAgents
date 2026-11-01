"""Phase 23: the Alpaca IEX daily market-data source.

The market-data half of Alpaca, kept apart from the broker half. These tests pin
the properties that make pointing a paper run at Alpaca *data* safe:

- the host is fixed to ``https://data.alpaca.markets``; a live host and the paper
  *trading* host are both refused, so data can never be pointed at orders;
- the feed is ``iex``, the only feed the Basic plan can read, so the request cannot
  403 on a SIP default;
- credentials come from the environment only, are never logged, and a missing one
  is a clear ``not_configured`` rather than an invented series;
- a 401 is an authentication failure, a 403 is an unsupported/forbidden request;
- a successful response is parsed into the shared frame shape, and a malformed,
  empty, or duplicated response is refused rather than served as data.

Every test drives a mocked transport; no test reaches the network.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents.paper_experiment.providers.alpaca_iex import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DATA_BASE_URL,
    FEED,
    AlpacaIexDailySource,
    assert_data_host,
    build_request_params,
)
from tradingagents.paper_experiment.providers.base import (
    HttpResponse,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderStaleDataError,
    ProviderUnsupportedError,
)
from tradingagents.paper_experiment.providers.selection import select_source

SYMBOL = "SPY"
TODAY = date(2026, 11, 2)


def _transport(body, *, status_code: int = 200, recorder: list | None = None):
    def send(*, url, params, headers, timeout):
        if recorder is not None:
            recorder.append({"url": url, "params": dict(params), "headers": dict(headers)})
        text = body if isinstance(body, str) else json.dumps(body)
        return HttpResponse(status_code, text)

    return send


def _bars_body(sessions, *, symbol: str = SYMBOL):
    bars = [
        {
            "t": f"{session}T04:00:00Z",
            "o": 100.0 + i,
            "h": 102.0 + i,
            "l": 99.0 + i,
            "c": 101.0 + i,
            "v": 1_000_000 + i,
            "n": 5000 + i,
            "vw": 100.5 + i,
        }
        for i, session in enumerate(sessions)
    ]
    return {"bars": {symbol: bars}, "next_page_token": None}


def _weekdays(end: date, count: int) -> list[str]:
    out: list[str] = []
    cursor = end
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return sorted(out)


def _source(body, *, today: date = TODAY, **kwargs):
    return AlpacaIexDailySource(
        SYMBOL,
        api_key="test-key-not-real",
        api_secret="test-secret-not-real",
        transport=_transport(body),
        today=today,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Host and feed are fixed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHostValidation:
    def test_the_data_host_is_accepted(self):
        assert assert_data_host(DATA_BASE_URL) == DATA_BASE_URL

    def test_the_paper_trading_host_is_not_a_market_data_host(self):
        with pytest.raises(ProviderUnsupportedError):
            assert_data_host("https://paper-api.alpaca.markets")

    def test_a_live_host_is_refused(self):
        with pytest.raises(ProviderUnsupportedError):
            assert_data_host("https://api.alpaca.markets")

    def test_an_unapproved_host_is_refused(self):
        with pytest.raises(ProviderUnsupportedError):
            assert_data_host("https://evil.example.com")

    def test_a_source_with_a_live_base_url_is_refused(self):
        with pytest.raises(ProviderUnsupportedError):
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret="s",
                base_url="https://api.alpaca.markets", transport=_transport({}),
            )


@pytest.mark.unit
class TestFeedAndRequest:
    def test_the_request_names_the_iex_feed(self):
        params = build_request_params(symbol=SYMBOL, start=date(2026, 1, 1), end=TODAY)
        assert params["feed"] == "iex"
        assert FEED == "iex"

    def test_the_request_asks_for_daily_bars(self):
        params = build_request_params(symbol=SYMBOL, start=date(2026, 1, 1), end=TODAY)
        assert params["timeframe"] == "1Day"
        assert params["symbols"] == SYMBOL
        assert params["sort"] == "asc"

    def test_the_credential_is_not_in_the_query_parameters(self):
        """Alpaca auth is by header; the key must never reach a URL or a log."""
        params = build_request_params(symbol=SYMBOL, start=date(2026, 1, 1), end=TODAY)
        assert not any("key" in k.lower() or "secret" in k.lower() for k in params)

    def test_the_source_calls_the_data_host_with_iex(self):
        recorder: list = []
        AlpacaIexDailySource(
            SYMBOL, api_key="k", api_secret="s", today=TODAY,
            transport=_transport(_bars_body(_weekdays(TODAY, 3)), recorder=recorder),
        )
        assert recorder and recorder[0]["url"] == f"{DATA_BASE_URL}/v2/stocks/bars"
        assert recorder[0]["params"]["feed"] == "iex"
        assert recorder[0]["headers"]["APCA-API-KEY-ID"] == "k"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCredentials:
    def test_missing_credentials_are_not_configured(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(ProviderNotConfiguredError):
            AlpacaIexDailySource(SYMBOL, transport=_transport(_bars_body([])))

    def test_a_missing_secret_is_not_configured(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "k")
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(ProviderNotConfiguredError):
            AlpacaIexDailySource(SYMBOL, transport=_transport(_bars_body([])))

    def test_credentials_are_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "env-key")
        monkeypatch.setenv(API_SECRET_ENV, "env-secret")
        source = AlpacaIexDailySource(
            SYMBOL, transport=_transport(_bars_body(_weekdays(TODAY, 3))), today=TODAY
        )
        assert source.provenance.symbol == SYMBOL

    def test_the_credential_never_reaches_the_provenance(self):
        source = _source(_bars_body(_weekdays(TODAY, 3)))
        blob = json.dumps(source.provenance.to_dict())
        assert "test-key-not-real" not in blob
        assert "test-secret-not-real" not in blob


# ---------------------------------------------------------------------------
# HTTP status classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatusClassification:
    def test_a_401_is_an_authentication_failure(self):
        with pytest.raises(ProviderAuthError):
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret="s", today=TODAY,
                transport=_transport("unauthorized", status_code=401),
            )

    def test_a_401_does_not_leak_the_credential(self):
        secret = "LEAKY-SECRET-NOT-REAL"
        with pytest.raises(ProviderAuthError) as excinfo:
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret=secret, today=TODAY,
                transport=_transport(f"bad secret {secret}", status_code=401),
            )
        assert secret not in str(excinfo.value)
        assert "***" in str(excinfo.value)

    def test_a_403_is_an_unsupported_request(self):
        with pytest.raises(ProviderUnsupportedError):
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret="s", today=TODAY,
                transport=_transport("forbidden", status_code=403),
            )

    def test_a_500_is_a_provider_error(self):
        with pytest.raises(ProviderError):
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret="s", today=TODAY,
                transport=_transport("boom", status_code=500),
            )

    def test_a_transport_failure_is_a_provider_error(self):
        def boom(*, url, params, headers, timeout):
            raise OSError("connection reset")

        with pytest.raises(ProviderError):
            AlpacaIexDailySource(
                SYMBOL, api_key="k", api_secret="s", today=TODAY, transport=boom
            )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParsing:
    def test_a_successful_response_is_parsed(self):
        sessions = _weekdays(TODAY, 5)
        source = _source(_bars_body(sessions))
        assert source.sessions() == tuple(sessions)
        bar = source.bars(SYMBOL, sessions[-1])
        assert bar is not None
        assert bar.close == pytest.approx(101.0 + 4)

    def test_the_frame_is_oldest_first(self):
        sessions = _weekdays(TODAY, 4)
        source = _source(_bars_body(list(reversed(sessions))))
        assert source.sessions() == tuple(sessions)

    def test_a_recent_bar_is_current_market(self):
        sessions = _weekdays(TODAY, 3)
        source = _source(_bars_body(sessions))
        assert source.provenance.last_bar_date == sessions[-1]
        assert source.is_current_market is True

    def test_an_empty_bar_list_is_refused(self):
        with pytest.raises(ProviderStaleDataError):
            _source({"bars": {SYMBOL: []}, "next_page_token": None})

    def test_an_unknown_symbol_is_refused(self):
        with pytest.raises(ProviderUnsupportedError):
            _source({"bars": {}, "next_page_token": None})

    def test_a_missing_bars_object_is_refused(self):
        with pytest.raises(ProviderError):
            _source({"next_page_token": None})

    def test_a_malformed_bar_is_refused(self):
        body = {"bars": {SYMBOL: [{"t": "2026-10-30T04:00:00Z", "o": "abc",
                                   "h": 2, "l": 1, "c": 1, "v": 1}]}}
        with pytest.raises(ProviderError):
            _source(body)

    def test_a_bar_missing_its_timestamp_is_refused(self):
        body = {"bars": {SYMBOL: [{"o": 1, "h": 2, "l": 1, "c": 1, "v": 1}]}}
        with pytest.raises(ProviderError):
            _source(body)

    def test_a_non_finite_price_is_refused(self):
        body = {"bars": {SYMBOL: [{"t": "2026-10-30T04:00:00Z", "o": float("nan"),
                                   "h": 2, "l": 1, "c": 1, "v": 1}]}}
        with pytest.raises(ProviderError):
            _source(body)

    def test_a_high_below_low_is_refused(self):
        body = {"bars": {SYMBOL: [{"t": "2026-10-30T04:00:00Z", "o": 1.0,
                                   "h": 0.5, "l": 1.0, "c": 1.0, "v": 1}]}}
        with pytest.raises(ProviderError):
            _source(body)

    def test_a_duplicated_session_is_refused(self):
        """A repeated session label must not be silently collapsed to the last one."""
        dup = [
            {"t": "2026-10-30T04:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10},
            {"t": "2026-10-30T04:00:00Z", "o": 9.0, "h": 9.5, "l": 8.5, "c": 9.2, "v": 20},
        ]
        with pytest.raises(ProviderError):
            _source({"bars": {SYMBOL: dup}, "next_page_token": None})

    def test_a_stale_feed_is_refused(self):
        old = _weekdays(TODAY - timedelta(days=40), 3)
        with pytest.raises(ProviderStaleDataError):
            _source(_bars_body(old))

    def test_the_provenance_records_the_iex_feed(self):
        source = _source(_bars_body(_weekdays(TODAY, 3)))
        notes = " ".join(source.provenance.notes or ())
        assert "iex" in notes.lower()
        assert source.provenance.kind == "vendor_daily"
        assert source.provenance.adjusted is False


# ---------------------------------------------------------------------------
# Selection wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelection:
    def test_alpaca_iex_is_a_known_source(self):
        from tradingagents.paper_experiment.providers.selection import KNOWN_SOURCES

        assert "alpaca_iex" in KNOWN_SOURCES

    def test_selecting_alpaca_iex_builds_the_source(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "k")
        monkeypatch.setenv(API_SECRET_ENV, "s")
        selection = select_source(
            "alpaca_iex",
            symbols=(SYMBOL,),
            provider_kwargs={"transport": _transport(_bars_body(_weekdays(TODAY, 3)))},
            today=TODAY,
        )
        assert selection.source.name == "alpaca_iex_daily"
        assert selection.source.kind == "vendor_daily"
        assert selection.switched is False

    def test_selecting_alpaca_iex_without_credentials_raises(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        from tradingagents.paper_experiment.providers.selection import SourceSelectionError

        with pytest.raises(SourceSelectionError):
            select_source("alpaca_iex", symbols=(SYMBOL,))

    def test_the_market_data_clock_is_utc(self):
        source = _source(_bars_body(_weekdays(TODAY, 3)))
        assert source.provenance.timezone == "UTC"
        assert datetime.now(UTC)  # sanity: UTC is importable and used
