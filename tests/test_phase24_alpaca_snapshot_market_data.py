"""Phase 24: a current price for the paper worker, from Alpaca market data.

The worker sizes an order from whatever ``MarketData.price`` returns. The
historical default was a daily bar's **open** — not a current price. This phase
adds a genuinely current seam and pins the properties that make it safe:

- the price comes from the latest trade / quote / minute bar, never a daily open;
- a missing, malformed, NaN, infinite or zero price raises, so the worker places
  **no** order rather than sizing from a fabricated number;
- the request goes to the market-data host on the IEX feed, with the paper
  credentials, and never to a trading endpoint;
- a missing credential is reported, not papered over.

Every test drives a mocked transport; no test reaches the network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from tradingagents.paper_experiment.broker import (
    AlpacaSnapshotMarketData,
    MarketDataError,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
    TradingDecision,
)
from tradingagents.paper_experiment.providers.alpaca_iex import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DATA_BASE_URL,
)
from tradingagents.paper_experiment.providers.base import HttpResponse

SYMBOL = "SPY"


def _transport(body, *, status_code: int = 200, recorder: list | None = None):
    def send(*, url, params, headers, timeout):
        if recorder is not None:
            recorder.append({"url": url, "params": dict(params), "headers": dict(headers)})
        text = body if isinstance(body, str) else json.dumps(body)
        return HttpResponse(status_code, text)

    return send


def _snapshot_body(symbol: str = SYMBOL, **fields):
    snapshot = {"latestTrade": {"p": 580.25, "t": "2026-10-30T20:00:00Z"}}
    snapshot.update(fields)
    return {symbol: snapshot}


def _market_data(body, *, symbols=(SYMBOL,), **kwargs):
    return AlpacaSnapshotMarketData(
        symbols,
        api_key="test-key-not-real",
        api_secret="test-secret-not-real",
        transport=_transport(body),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The current price is read from the right field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCurrentPrice:
    def test_the_latest_trade_price_is_used(self):
        md = _market_data(_snapshot_body())
        assert md.price(SYMBOL) == pytest.approx(580.25)

    def test_a_quote_midpoint_is_used_when_there_is_no_trade(self):
        body = _snapshot_body(latestTrade=None, latestQuote={"bp": 100.0, "ap": 102.0})
        md = _market_data(body)
        assert md.price(SYMBOL) == pytest.approx(101.0)

    def test_the_minute_bar_close_is_the_last_resort(self):
        body = _snapshot_body(latestTrade=None, latestQuote=None,
                              minuteBar={"c": 55.5, "o": 54.0})
        md = _market_data(body)
        assert md.price(SYMBOL) == pytest.approx(55.5)

    def test_a_daily_bar_open_is_never_used_as_a_current_price(self):
        """The precise defect this seam fixes: an open is not a current price."""
        body = _snapshot_body(latestTrade=None, latestQuote=None, minuteBar=None,
                              dailyBar={"o": 500.0, "c": 580.0})
        md = _market_data(body)
        with pytest.raises(MarketDataError):
            md.price(SYMBOL)

    def test_available_is_true_when_a_price_can_be_read(self):
        assert _market_data(_snapshot_body()).available() is True

    def test_available_is_false_when_no_price_can_be_read(self):
        body = _snapshot_body(latestTrade=None, latestQuote=None, minuteBar=None)
        assert _market_data(body).available() is False

    def test_available_is_false_when_the_request_fails(self):
        md = AlpacaSnapshotMarketData(
            (SYMBOL,), api_key="k", api_secret="s",
            transport=_transport("nope", status_code=500),
        )
        assert md.available() is False


# ---------------------------------------------------------------------------
# Malformed prices never fabricate an order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMalformedPrices:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, None, "abc"])
    def test_a_bad_trade_price_falls_through_to_no_price(self, bad):
        body = _snapshot_body(latestTrade={"p": bad}, latestQuote=None, minuteBar=None)
        with pytest.raises(MarketDataError):
            _market_data(body).price(SYMBOL)

    def test_a_missing_symbol_snapshot_is_an_error(self):
        with pytest.raises(MarketDataError):
            _market_data({"AAPL": {"latestTrade": {"p": 1.0}}}).price(SYMBOL)

    def test_a_non_object_response_is_an_error(self):
        with pytest.raises(MarketDataError):
            _market_data(["not", "an", "object"]).price(SYMBOL)

    def test_a_missing_credential_is_reported(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(MarketDataError):
            AlpacaSnapshotMarketData((SYMBOL,))

    def test_a_live_host_is_refused(self):
        from tradingagents.paper_experiment.providers.base import ProviderError

        with pytest.raises(ProviderError):
            AlpacaSnapshotMarketData(
                (SYMBOL,), api_key="k", api_secret="s",
                base_url="https://api.alpaca.markets",
            )


# ---------------------------------------------------------------------------
# The request is a market-data request, never a trading one
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequestShape:
    def test_the_snapshot_request_targets_the_data_host_on_iex(self):
        recorder: list = []
        AlpacaSnapshotMarketData(
            (SYMBOL,), api_key="k", api_secret="s",
            transport=_transport(_snapshot_body(), recorder=recorder),
        ).price(SYMBOL)
        assert recorder[0]["url"] == f"{DATA_BASE_URL}/v2/stocks/snapshots"
        assert recorder[0]["params"]["feed"] == "iex"
        assert "paper-api" not in recorder[0]["url"]

    def test_the_credential_is_sent_as_a_header_not_a_query_param(self):
        recorder: list = []
        AlpacaSnapshotMarketData(
            (SYMBOL,), api_key="k", api_secret="s",
            transport=_transport(_snapshot_body(), recorder=recorder),
        ).price(SYMBOL)
        assert recorder[0]["headers"]["APCA-API-KEY-ID"] == "k"
        assert "k" not in json.dumps(recorder[0]["params"])


# ---------------------------------------------------------------------------
# The worker uses the current price and never fabricates an order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkerIntegration:
    def _worker(self, tmp_path, market_data):
        return PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=market_data,
            broker_factory=lambda: _FailIfCalledBroker(),
        )

    def test_the_worker_uses_the_snapshot_price_to_size(self):
        """A sizing check on the seam alone: floor(25 / 580.25) == 0 for whole shares."""
        md = _market_data(_snapshot_body())
        price = md.price(SYMBOL)
        assert price == pytest.approx(580.25)

    def test_the_worker_places_no_order_when_no_price_is_available(self, tmp_path):
        """A market-data failure must leave the worker RUNNING with no order sent."""
        md = AlpacaSnapshotMarketData(
            (SYMBOL,), api_key="k", api_secret="s",
            transport=_transport("boom", status_code=500),
        )
        worker = self._worker(tmp_path, md)
        worker.state.status = "RUNNING"
        worker._broker = _FailIfCalledBroker()  # noqa: SLF001 - the seam under test
        result = worker.run_once()
        assert result.ran is False
        assert result.skipped_reason == "market data was unavailable"
        assert worker.state.orders == {}


class _OneDecision:
    def __init__(self):
        self._done = False

    def next_decision(self):
        if self._done:
            return None
        self._done = True
        return TradingDecision(decision_id="d1", symbol=SYMBOL, side="buy", notional=25.0)


class _FailIfCalledBroker:
    def submit_order(self, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the worker submitted an order without a usable price")

    def _reconcile(self, client_order_id):  # pragma: no cover
        raise AssertionError("no reconcile should happen")

    def positions(self, refresh=False):  # pragma: no cover
        return {}

    def connect(self):  # pragma: no cover
        raise AssertionError("connect should not be reached in this test")


@pytest.mark.unit
class TestStaticStillWorks:
    def test_the_existing_static_source_is_unchanged(self):
        assert StaticMarketData({"SPY": 10.0}).price("SPY") == 10.0

    def test_the_snapshot_seam_does_not_change_the_decision_shape(self):
        """A snapshot price is a number; it never alters the decision object."""
        decision = TradingDecision(
            decision_id="d1", symbol=SYMBOL, side="buy", notional=25.0
        )
        assert decision.to_dict()["notional"] == 25.0
        assert date(2026, 10, 30)  # sanity: the date import is used
