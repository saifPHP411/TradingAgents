"""The current-price path for the crypto paper worker: ticker lookup, failures,
bounded retries, and the structured diagnostics a production failure is read from.

These tests exist because of a real production failure. The worker asked Binance's
ticker map for ``DOGEUSDT`` (the venue's spelling, which the symbol mapping
carries) while the map was keyed only by the canonical ``DOGE/USDT``. The lookup
missed, and a perfectly good price was reported as "no ticker for 'DOGEUSDT'".
The affected symbol was never the point: *every* symbol failed the same way.

The tests are written to catch that premise again, so they pin the behaviour that
was missing rather than the behaviour that happened to be there:

- a ticker is found by either spelling, and by the canonical pair;
- one symbol's failure never hides another symbol's price;
- a throttled or briefly unreachable venue is retried a *bounded* number of
  times, and a considered refusal (400, 404) is not retried;
- a stale, malformed or non-positive price is refused rather than used;
- when no current price can be read, no order can be sized.

Every test runs offline through the injected transport; the suite's autouse guard
refuses a live socket.
"""

from __future__ import annotations

import json
import time

import pytest

from tradingagents.crypto import BinancePublicClient, CryptoDataError, HttpResponse
from tradingagents.crypto.transport import (
    MAX_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    fetch_json,
)
from tradingagents.paper_experiment.broker.trading import (
    MARKET_DATA_STATUS_ERROR,
    MARKET_DATA_STATUS_NO_DATA,
    MARKET_DATA_STATUS_OK,
    MARKET_DATA_STATUS_STALE,
    BinanceSnapshotMarketData,
    MarketDataError,
)

# A real Binance 24-hour ticker row, as the venue sends it.
DOGE_ROW = {
    "symbol": "DOGEUSDT",
    "priceChange": "-0.00030000",
    "priceChangePercent": "-0.418",
    "weightedAvgPrice": "0.07120235",
    "prevClosePrice": "0.07169000",
    "lastPrice": "0.07137000",
    "lastQty": "17.00000000",
    "bidPrice": "0.07139000",
    "bidQty": "202084.00000000",
    "askPrice": "0.07140000",
    "askQty": "154726.00000000",
    "openPrice": "0.07169000",
    "highPrice": "0.07210000",
    "lowPrice": "0.07093000",
    "volume": "145766256.10000000",
    "quoteVolume": "10383928.52894400",
    "openTime": 1794064789156,
    "closeTime": 1794151189559,
    "firstId": 49686558233,
    "lastId": 49687111275,
    "count": 553043,
}

ACCEPTED_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "BCH/USDT",
)


def _now_ms() -> int:
    """The venue's clock, in milliseconds, so a fresh row really is fresh."""
    return int(time.time() * 1000)


def _row(symbol: str, last: str = "1.5", close_time: int | None = None) -> dict:
    """A ticker row for ``symbol`` in the venue's own spelling."""
    return {
        "symbol": symbol.replace("/", ""),
        "lastPrice": last,
        "bidPrice": last,
        "askPrice": last,
        "volume": "1000",
        "quoteVolume": "100500",
        "highPrice": last,
        "lowPrice": last,
        "priceChange": "0.1",
        "priceChangePercent": "0.1",
        "closeTime": _now_ms() if close_time is None else close_time,
    }


# ---------------------------------------------------------------------------
# The ticker map is reachable by either spelling
# ---------------------------------------------------------------------------


class TestTickerLookup:
    """The lookup must find a ticker however the symbol was written."""

    def test_ticker_is_found_by_the_venue_spelling(self):
        """The worker's mapping holds ``DOGEUSDT``; the map must answer for it."""
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        tickers = client.ticker_24h(["DOGEUSDT"])
        assert tickers["DOGEUSDT"].last == pytest.approx(1.5)

    def test_ticker_is_found_by_the_canonical_spelling(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        tickers = client.ticker_24h(["DOGE/USDT"])
        assert tickers["DOGE/USDT"].last == pytest.approx(1.5)

    def test_both_spellings_name_the_same_ticker(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        tickers = client.ticker_24h(["DOGE/USDT"])
        assert tickers["DOGEUSDT"] is tickers["DOGE/USDT"]

    def test_seam_prices_the_venue_spelling(self):
        """The seam's ``price()`` accepts the spelling the worker actually holds."""
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        assert seam.price("DOGEUSDT") == pytest.approx(1.5)
        assert seam.last_status == MARKET_DATA_STATUS_OK

    def test_seam_prices_the_canonical_spelling(self):
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        assert seam.price("DOGE/USDT") == pytest.approx(1.5)

    def test_a_missing_doge_is_reported_as_no_data(self):
        """A genuinely absent symbol is refused; it is not a spelling mismatch."""
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("BTC/USDT")]))
        )
        with pytest.raises(MarketDataError, match="no ticker"):
            seam.price("DOGEUSDT")
        assert seam.last_status == MARKET_DATA_STATUS_NO_DATA

    def test_a_missing_symbol_does_not_hide_the_others(self):
        """One absent symbol costs only its own price."""
        rows = [_row(s.replace("/", "")) for s in ACCEPTED_SYMBOLS if s != "DOGE/USDT"]

        def transport(*, url, params, headers, timeout):
            if params.get("symbols"):
                # The batch is refused because one symbol is unknown.
                return HttpResponse(400, json.dumps({"code": -1121, "msg": "Invalid symbol."}))
            symbol = params.get("symbol")
            if symbol == "DOGEUSDT":
                return HttpResponse(400, json.dumps({"code": -1121, "msg": "Invalid symbol."}))
            row = next((r for r in rows if r["symbol"] == symbol), None)
            if row is None:
                return HttpResponse(404, json.dumps({"code": -1121, "msg": "Invalid symbol."}))
            return HttpResponse(200, json.dumps(row))

        seam = BinanceSnapshotMarketData(ACCEPTED_SYMBOLS)
        seam._client = BinancePublicClient(transport=transport)

        for symbol in ACCEPTED_SYMBOLS:
            if symbol == "DOGE/USDT":
                continue
            assert seam.price(symbol.replace("/", "")) == pytest.approx(1.5)

        with pytest.raises(MarketDataError, match="no ticker"):
            seam.price("DOGEUSDT")

    def test_every_accepted_symbol_is_priced(self):
        """All nine accepted symbols price, each under its venue spelling."""
        rows = [_row(s.replace("/", "")) for s in ACCEPTED_SYMBOLS]

        def transport(*, url, params, headers, timeout):
            if params.get("symbols"):
                return HttpResponse(200, json.dumps(rows))
            symbol = params.get("symbol")
            row = next((r for r in rows if r["symbol"] == symbol), None)
            return HttpResponse(200, json.dumps(row)) if row else HttpResponse(404, "{}")

        seam = BinanceSnapshotMarketData(ACCEPTED_SYMBOLS)
        seam._client = BinancePublicClient(transport=transport)
        for symbol in ACCEPTED_SYMBOLS:
            assert seam.price(symbol.replace("/", "")) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The response shape the venue actually sends
# ---------------------------------------------------------------------------


class TestResponseShapes:
    def test_a_real_doge_row_is_parsed(self):
        """The venue's own DOGE row, shape and all, is read correctly."""
        row = dict(DOGE_ROW, closeTime=_now_ms())
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([row]))
        )
        ticker = client.ticker_24h(["DOGE/USDT"])["DOGE/USDT"]
        assert ticker.symbol == "DOGEUSDT"
        assert ticker.last == pytest.approx(0.07137)
        assert ticker.bid == pytest.approx(0.07139)
        assert ticker.ask == pytest.approx(0.0714)
        assert ticker.volume_24h == pytest.approx(145766256.1)
        assert ticker.observed_at is not None

    def test_a_single_object_is_accepted_as_well_as_a_list(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps(_row("DOGE/USDT")))
        )
        assert client.ticker_24h(["DOGE/USDT"])["DOGE/USDT"].last == pytest.approx(1.5)

    def test_an_empty_response_yields_no_ticker(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([]))
        )
        assert client.ticker_24h(["DOGE/USDT"]) == {}

    def test_malformed_json_is_refused(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, "<html>maintenance</html>")
        )
        with pytest.raises(CryptoDataError):
            client.ticker_24h(["DOGE/USDT"])

    def test_a_non_positive_price_is_refused_by_the_seam(self):
        """A zero last price is not an observation; it must not size an order."""
        row = _row("DOGE/USDT", last="0")
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([row]))
        )
        with pytest.raises(MarketDataError):
            seam.price("DOGEUSDT")
        assert seam.last_status == MARKET_DATA_STATUS_NO_DATA

    def test_a_nan_price_is_refused_by_the_seam(self):
        row = _row("DOGE/USDT", last="NaN")
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([row]))
        )
        with pytest.raises(MarketDataError):
            seam.price("DOGEUSDT")

    def test_a_stale_price_is_refused(self):
        """An old ticker is not a current price, so no order is sized from it."""
        row = _row("DOGE/USDT", close_time=1_700_000_000_000)
        seam = BinanceSnapshotMarketData(["DOGE/USDT"], max_age_seconds=60)
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([row]))
        )
        with pytest.raises(MarketDataError, match="stale"):
            seam.price("DOGEUSDT")
        assert seam.last_status == MARKET_DATA_STATUS_STALE


# ---------------------------------------------------------------------------
# Venue failures
# ---------------------------------------------------------------------------


class TestVenueFailures:
    def test_http_400_is_reported_with_the_venue_code(self):
        """A symbol the venue refuses is that symbol's answer, not a crash."""
        body = json.dumps({"code": -1121, "msg": "Invalid symbol."})
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(400, body)
        )
        assert client.ticker_24h(["NOPE/USDT"]) == {}
        assert client.last_error is not None
        assert "400" in client.last_error
        assert "-1121" in client.last_error

    def test_http_404_is_reported(self):
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(404, json.dumps({"code": -1121}))
        )
        assert client.ticker_24h(["NOPE/USDT"]) == {}
        assert client.last_error is not None
        assert "404" in client.last_error

    def test_http_429_is_rate_limited(self):
        from tradingagents.crypto import CryptoRateLimitedError

        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(429, json.dumps({"code": -1003}))
        )
        with pytest.raises(CryptoRateLimitedError):
            client.ticker_24h(["BTC/USDT"])

    def test_a_connection_failure_is_reported_not_hidden(self):
        """A fault that is not the venue's answer is never turned into "no ticker"."""

        def transport(**_):
            raise ConnectionError("connection refused")

        client = BinancePublicClient(transport=transport)
        with pytest.raises(CryptoDataError, match="connection refused"):
            client.ticker_24h(["BTC/USDT"])

    def test_no_order_can_be_sized_when_every_price_source_fails(self):
        """When no current price exists, the seam refuses rather than guessing."""

        def transport(**_):
            raise ConnectionError("connection refused")

        seam = BinanceSnapshotMarketData(ACCEPTED_SYMBOLS)
        seam._client = BinancePublicClient(transport=transport)
        for symbol in ACCEPTED_SYMBOLS:
            with pytest.raises(MarketDataError):
                seam.price(symbol.replace("/", ""))
        assert seam.last_status == MARKET_DATA_STATUS_ERROR
        assert seam.last_observation is None

    def test_available_is_false_when_the_source_is_unreachable(self):
        def transport(**_):
            raise ConnectionError("connection refused")

        seam = BinanceSnapshotMarketData(ACCEPTED_SYMBOLS)
        seam._client = BinancePublicClient(transport=transport)
        assert seam.available() is False
        assert seam.last_status == MARKET_DATA_STATUS_ERROR


# ---------------------------------------------------------------------------
# Bounded retries and backoff
# ---------------------------------------------------------------------------


class TestBoundedRetries:
    def test_a_throttled_request_is_retried_and_then_succeeds(self):
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            if len(calls) == 1:
                return HttpResponse(429, json.dumps({"code": -1003, "msg": "Too many requests."}))
            return HttpResponse(200, json.dumps([_row("DOGE/USDT")]))

        client = BinancePublicClient(transport=transport)
        tickers = client.ticker_24h(["DOGE/USDT"])
        assert tickers["DOGEUSDT"].last == pytest.approx(1.5)
        assert len(calls) == 2

    def test_retries_are_bounded(self):
        """A permanently throttled venue is tried a bounded number of times."""
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            return HttpResponse(429, json.dumps({"code": -1003}))

        client = BinancePublicClient(transport=transport)
        with pytest.raises(CryptoDataError):
            client.ticker_24h(["DOGE/USDT"])
        assert len(calls) == MAX_ATTEMPTS

    def test_backoff_grows_between_attempts(self):
        delays: list[float] = []

        def transport(*, url, params, headers, timeout):
            return HttpResponse(429, json.dumps({"code": -1003}))

        with pytest.raises(CryptoDataError):
            fetch_json(
                url="https://x/y",
                transport=transport,
                sleep=delays.append,
            )
        assert delays == [RETRY_BACKOFF_SECONDS, RETRY_BACKOFF_SECONDS * 2]

    def test_a_transient_network_failure_is_retried(self):
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("connection reset by peer")
            return HttpResponse(200, json.dumps([_row("DOGE/USDT")]))

        client = BinancePublicClient(transport=transport)
        assert client.ticker_24h(["DOGE/USDT"])["DOGEUSDT"].last == pytest.approx(1.5)
        assert len(calls) == 2

    def test_a_considered_refusal_is_not_retried(self):
        """HTTP 400 is the venue's answer; repeating it would only repeat it."""
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            return HttpResponse(400, json.dumps({"code": -1121, "msg": "Invalid symbol."}))

        with pytest.raises(CryptoDataError):
            fetch_json(url="https://x/y", transport=transport, sleep=lambda _: None)
        assert len(calls) == 1

    def test_a_404_is_not_retried(self):
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            return HttpResponse(404, json.dumps({"code": -1121}))

        with pytest.raises(CryptoDataError):
            fetch_json(url="https://x/y", transport=transport, sleep=lambda _: None)
        assert len(calls) == 1

    def test_a_malformed_body_is_not_retried(self):
        calls: list[int] = []

        def transport(*, url, params, headers, timeout):
            calls.append(1)
            return HttpResponse(200, "<html>not json</html>")

        with pytest.raises(CryptoDataError, match="JSON"):
            fetch_json(url="https://x/y", transport=transport, sleep=lambda _: None)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# The structured diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_a_successful_read_records_the_status_and_shape(self):
        records: list[dict] = []
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")])),
            diagnose=records.append,
        )
        client.ticker_24h(["DOGE/USDT"])
        assert records
        record = records[-1]
        assert record["path"] == "/api/v3/ticker/24hr"
        assert record["status"] == 200
        assert record["outcome"] == "ok"
        assert record["response_type"] == "list"
        assert record["symbol_present"] is True

    def test_an_http_error_records_the_venue_code_and_message(self):
        records: list[dict] = []
        body = json.dumps({"code": -1121, "msg": "Invalid symbol."})
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(400, body),
            diagnose=records.append,
        )
        client.ticker_24h(["NOPE/USDT"])
        record = records[-1]
        assert record["status"] == 400
        assert record["outcome"] == "http_error"
        assert record["venue_code"] == -1121
        assert record["venue_message"] == "Invalid symbol."

    def test_a_transport_failure_records_the_outcome(self):
        records: list[dict] = []

        def transport(**_):
            raise ConnectionError("connection refused")

        client = BinancePublicClient(transport=transport, diagnose=records.append)
        with pytest.raises(CryptoDataError):
            client.ticker_24h(["DOGE/USDT"])
        assert records[-1]["outcome"] == "transport_failure"
        assert records[-1]["status"] is None

    def test_a_malformed_body_records_the_outcome(self):
        records: list[dict] = []
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, "<html>nope</html>"),
            diagnose=records.append,
        )
        with pytest.raises(CryptoDataError):
            client.ticker_24h(["DOGE/USDT"])
        assert records[-1]["outcome"] == "malformed_json"

    def test_the_record_carries_no_query_string_or_header(self):
        """The record names the path, never the host's query string."""
        records: list[dict] = []
        client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")])),
            diagnose=records.append,
        )
        client.ticker_24h(["DOGE/USDT"])
        record = records[-1]
        assert "?" not in record["path"]
        assert "headers" not in record
        assert "params" not in record

    def test_a_secret_echoed_by_the_venue_is_scrubbed_from_the_record(self):
        """A diagnostic record is shown in a dashboard, so it carries no secret.

        The venue's own message is kept — it is what explains the failure — but
        a credential the venue echoed back must be replaced, not displayed.
        """
        secret = "SUPERSECRETKEY12345"
        records: list[dict] = []

        def transport(**_):
            return HttpResponse(
                400, json.dumps({"code": -1121, "msg": f"bad key {secret}"})
            )

        with pytest.raises(CryptoDataError):
            fetch_json(
                url="https://x/y",
                transport=transport,
                secrets=(secret,),
                diagnose=records.append,
            )

        blob = json.dumps(records, default=str)
        assert secret not in blob
        assert records[-1]["venue_code"] == -1121
        # The message still explains the failure; only the secret is gone.
        assert "bad key" in records[-1]["venue_message"]

    def test_the_record_carries_only_the_venue_code_and_message(self):
        """The record is a summary, not the body the venue sent."""
        records: list[dict] = []

        def transport(**_):
            return HttpResponse(
                400,
                json.dumps(
                    {
                        "code": -1121,
                        "msg": "Invalid symbol.",
                        "echo": "a field the venue should not be echoing",
                    }
                ),
            )

        client = BinancePublicClient(transport=transport, diagnose=records.append)
        client.ticker_24h(["NOPE/USDT"])

        record = records[-1]
        assert record["venue_code"] == -1121
        assert record["venue_message"] == "Invalid symbol."
        assert "echo" not in record

    def test_the_seam_exposes_the_diagnostics(self):
        seam = BinanceSnapshotMarketData(["DOGE/USDT"])
        seam._client = BinancePublicClient(
            transport=lambda **_: HttpResponse(200, json.dumps([_row("DOGE/USDT")]))
        )
        seam.price("DOGEUSDT")
        diagnostics = seam.describe()["market_data_diagnostics"]
        assert diagnostics
        assert diagnostics[-1]["path"] == "/api/v3/ticker/24hr"
        assert diagnostics[-1]["status"] == 200


# ---------------------------------------------------------------------------
# The batch request and its fallback
# ---------------------------------------------------------------------------


class TestBatchRequest:
    def test_the_batch_is_asked_for_with_compact_json(self):
        """Binance's batch parameter rejects spaces, so it must be compact."""
        seen: list[str] = []

        def transport(*, url, params, headers, timeout):
            if params.get("symbols"):
                seen.append(params["symbols"])
                return HttpResponse(200, json.dumps([_row("BTCUSDT")]))
            return HttpResponse(200, json.dumps(_row("BTCUSDT")))

        BinancePublicClient(transport=transport).ticker_24h(["BTC/USDT"])
        assert seen == ['["BTCUSDT"]']
        assert " " not in seen[0]

    def test_a_refused_batch_falls_back_to_one_request_per_symbol(self):
        calls: list[dict] = []

        def transport(*, url, params, headers, timeout):
            calls.append(dict(params))
            if params.get("symbols"):
                return HttpResponse(400, json.dumps({"code": -1100, "msg": "Illegal characters"}))
            row = _row(params["symbol"])
            return HttpResponse(200, json.dumps(row))

        tickers = BinancePublicClient(transport=transport).ticker_24h(["BTC/USDT", "ETH/USDT"])
        assert tickers["BTC/USDT"].last == pytest.approx(1.5)
        assert tickers["ETH/USDT"].last == pytest.approx(1.5)
        # One batch attempt, then one request per symbol.
        assert len(calls) == 3
        assert calls[1]["symbol"] == "BTCUSDT"
        assert calls[2]["symbol"] == "ETHUSDT"

    def test_the_fallback_records_which_symbol_was_refused(self):
        def transport(*, url, params, headers, timeout):
            if params.get("symbols"):
                return HttpResponse(400, json.dumps({"code": -1100}))
            if params.get("symbol") == "NOPEUSDT":
                return HttpResponse(400, json.dumps({"code": -1121, "msg": "Invalid symbol."}))
            return HttpResponse(200, json.dumps(_row(params["symbol"])))

        client = BinancePublicClient(transport=transport)
        tickers = client.ticker_24h(["BTC/USDT", "NOPE/USDT"])
        assert "BTC/USDT" in tickers
        assert "NOPE/USDT" not in tickers
        assert client.last_error is not None
        assert "NOPEUSDT" in client.last_error


__all__ = ["ACCEPTED_SYMBOLS", "DOGE_ROW"]
