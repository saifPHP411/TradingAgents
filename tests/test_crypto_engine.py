"""Tests for the unified crypto market-data engine.

The engine is driven through a fake REST client and a fake stream, so every branch
— the newest-of-two-sources price, the closed/forming split, the freshness verdict
and the stale-not-live rule — is exercised with no socket and no venue.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.crypto.engine import CryptoMarketDataEngine
from tradingagents.crypto.stream import STALE, StreamStatus
from tradingagents.crypto.types import Candle, Ticker


class _FakeClient:
    """A REST client returning canned bars and tickers, keyed by canonical symbol."""

    provider_name = "fake"
    kind = "vendor_rest"

    def __init__(self, *, bars=None, tickers=None, error=None):
        self._bars = bars or {}
        self._tickers = tickers or {}
        self._error = error
        self.calls = 0

    def describe(self):
        return {"crypto_data_provider": self.provider_name, "crypto_data_requires_key": False}

    def ticker_24h(self, symbols):
        self.calls += 1
        if self._error is not None:
            from tradingagents.crypto.transport import CryptoDataError

            raise CryptoDataError(self._error)
        return {s: self._tickers[s] for s in symbols if s in self._tickers}

    def klines(self, symbol, interval, *, limit=1000, start=None, end=None):
        return list(self._bars.get((symbol, interval), []))

    def depth(self, symbol, *, limit=20):
        from tradingagents.crypto.types import OrderBook

        return OrderBook(symbol=symbol, bids=[], asks=[])


def _candle(symbol: str, *, minute: int, close: float, interval: str = "1m", now=None) -> Candle:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    step = timedelta(minutes=1 if interval == "1m" else 5)
    open_time = base + step * minute
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=open_time + step,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _ticker(symbol: str, *, last: float, at: datetime) -> Ticker:
    return Ticker(
        symbol=symbol,
        last=last,
        bid=last - 0.1,
        ask=last + 0.1,
        volume_24h=100.0,
        quote_volume_24h=1000.0,
        high_24h=last + 5,
        low_24h=last - 5,
        change_24h=1.0,
        change_percent_24h=1.0,
        observed_at=at,
    )


class TestSnapshotFreshness:
    def test_an_empty_engine_is_not_fresh(self):
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient())
        snap = engine.snapshot("BTC/USDT")
        assert snap["fresh"] is False
        assert snap["price"] is None
        assert snap["age_seconds"] is None

    def test_a_fresh_ticker_is_reported_fresh(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        client = _FakeClient(tickers={"BTC/USDT": _ticker("BTC/USDT", last=100.0, at=now)})
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=client, clock=lambda: now)
        engine.refresh_tickers()
        snap = engine.snapshot("BTC/USDT")
        assert snap["fresh"] is True
        assert snap["price"] == 100.0

    def test_a_stale_ticker_is_not_fresh(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = now - timedelta(minutes=10)
        client = _FakeClient(tickers={"BTC/USDT": _ticker("BTC/USDT", last=100.0, at=old)})
        engine = CryptoMarketDataEngine(
            symbols=["BTC/USDT"], client=client, fresh_seconds=60.0, clock=lambda: now
        )
        engine.refresh_tickers()
        snap = engine.snapshot("BTC/USDT")
        assert snap["fresh"] is False
        assert snap["age_seconds"] == pytest.approx(600.0)
        # A stale price is still reported, but never *as* fresh.
        assert snap["price"] == 100.0

    def test_unknown_symbol_reports_not_fresh(self):
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient())
        assert engine.snapshot("ETH/USDT")["fresh"] is False


class TestNewestPriceWins:
    def test_the_stream_price_wins_when_newer(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        ticker_at = now - timedelta(seconds=30)
        client = _FakeClient(tickers={"BTC/USDT": _ticker("BTC/USDT", last=100.0, at=ticker_at)})
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=client, clock=lambda: now)
        engine.refresh_tickers()
        # A stream tick stamped *now* is newer than the REST ticker, so it wins.
        engine._on_stream_ticker(_ticker("BTC/USDT", last=101.0, at=now))
        assert engine.snapshot("BTC/USDT")["price"] == 101.0

    def test_the_rest_ticker_wins_when_newer(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        client = _FakeClient(tickers={"BTC/USDT": _ticker("BTC/USDT", last=100.0, at=now)})
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=client, clock=lambda: now)
        engine.refresh_tickers()
        # An older stream tick must not overwrite the newer REST ticker.
        engine._on_stream_ticker(_ticker("BTC/USDT", last=99.0, at=now - timedelta(minutes=5)))
        assert engine.snapshot("BTC/USDT")["price"] == 100.0


class TestStreamTickerMerge:
    """A streamed ticker carries the bid, the ask and the day's statistics.

    The trade stream carries only a price, so the live view's bid, ask, spread
    and 24-hour change have to come from the ticker's own stream event. These
    tests pin that a stream ticker reaches the view's ticker, and that a
    price-only trade tick does not blank the fields it does not carry.
    """

    def test_a_streamed_ticker_reaches_the_view(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient(), clock=lambda: now)
        engine._on_stream_ticker(_ticker("BTC/USDT", last=101.0, at=now))
        ticker = engine.snapshot("BTC/USDT")["ticker"]
        assert ticker is not None
        assert ticker["last"] == 101.0
        assert ticker["bid"] == 100.9
        assert ticker["ask"] == 101.1

    def test_a_price_only_tick_keeps_the_bid_and_the_ask(self):
        """A trade tick has a price but no book; it must not blank the book."""
        now = datetime(2026, 1, 1, tzinfo=UTC)
        client = _FakeClient(tickers={"BTC/USDT": _ticker("BTC/USDT", last=100.0, at=now)})
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=client, clock=lambda: now)
        engine.refresh_tickers()
        engine._on_stream_ticker(
            Ticker(
                symbol="BTC/USDT",
                last=100.5,
                bid=None,
                ask=None,
                volume_24h=None,
                quote_volume_24h=None,
                high_24h=None,
                low_24h=None,
                change_24h=None,
                change_percent_24h=None,
                observed_at=now + timedelta(seconds=1),
            )
        )
        ticker = engine.snapshot("BTC/USDT")["ticker"]
        assert ticker["last"] == 100.5
        assert ticker["bid"] == 99.9
        assert ticker["ask"] == 100.1
        assert ticker["volume_24h"] == 100.0


class TestClosedVersusForming:
    def test_only_a_closed_bar_is_kept_as_closed(self):
        now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        engine = CryptoMarketDataEngine(
            symbols=["BTC/USDT"], client=_FakeClient(), clock=lambda: now
        )
        engine._on_stream_candle(_candle("BTC/USDT", minute=0, close=100.0, now=now))
        assert engine.last_closed_bar("BTC/USDT", "1m") is not None
        assert engine.forming_bar("BTC/USDT", "1m") is None

    def test_a_forming_bar_is_kept_apart_from_closed(self):
        now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        engine = CryptoMarketDataEngine(
            symbols=["BTC/USDT"], client=_FakeClient(), clock=lambda: now
        )
        # A bar that opens at 00:05 closes at 00:06, which is after "now".
        engine._on_stream_candle(_candle("BTC/USDT", minute=5, close=100.0, now=now))
        assert engine.last_closed_bar("BTC/USDT", "1m") is None
        assert engine.forming_bar("BTC/USDT", "1m") is not None

    def test_a_closed_bar_clears_the_forming_bar(self):
        base = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        clock = {"now": base}
        engine = CryptoMarketDataEngine(
            symbols=["BTC/USDT"], client=_FakeClient(), clock=lambda: clock["now"]
        )
        # At 00:05 the 00:05 bar is still forming (it closes at 00:06).
        engine._on_stream_candle(_candle("BTC/USDT", minute=5, close=100.0))
        assert engine.forming_bar("BTC/USDT", "1m") is not None
        # Once the clock passes 00:06 the same bar is closed, so it moves out of
        # the forming slot and into the closed slot.
        clock["now"] = base + timedelta(minutes=1)
        engine._on_stream_candle(_candle("BTC/USDT", minute=5, close=100.0))
        assert engine.last_closed_bar("BTC/USDT", "1m") is not None
        assert engine.forming_bar("BTC/USDT", "1m") is None


class TestStreamStatus:
    def test_no_stream_reports_stopped(self):
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient())
        assert engine.stream_status()["state"] == "stopped"

    def test_a_stream_status_is_forwarded(self):
        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient())
        engine._stream = _FakeStream(StreamStatus(state=STALE, subscriptions=("btcusdt@trade",)))
        assert engine.stream_status()["state"] == STALE


class TestRESTErrors:
    def test_a_ticker_error_is_raised_and_recorded(self):
        from tradingagents.crypto.transport import CryptoDataError

        engine = CryptoMarketDataEngine(symbols=["BTC/USDT"], client=_FakeClient(error="boom"))
        with pytest.raises(CryptoDataError):
            engine.refresh_tickers()
        assert engine.last_error is not None


class _FakeStream:
    def __init__(self, status):
        self.status = status

    def start(self):
        pass

    def stop(self):
        pass
