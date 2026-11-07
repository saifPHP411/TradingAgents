"""Tests for the live crypto tape: reconnection, staleness, dedup, timestamps.

The consumer is driven through a fake ``connector``, so the whole reconnect /
backoff / stale / duplicate / skew path is exercised with no socket. The tests
assert the properties a dashboard depends on — a dead socket reads ``stale`` or
``reconnecting``, never ``connected``; a replayed trade is not counted twice; an
out-of-order or future-dated event is refused.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

from tradingagents.crypto.stream import (
    CONNECTED,
    DISCONNECTED,
    RECONNECTING,
    STALE,
    STOPPED,
    MarketStream,
    StreamCallbacks,
)


class _FakeConnection:
    """A connection that replays ``messages`` then either hangs or drops.

    ``hang=True`` models a socket that is open but silent (what a stale check is
    for); ``hang=False`` models a socket that has closed, so the consumer
    reconnects.
    """

    def __init__(self, messages, *, hang: bool = True):
        self._messages = list(messages)
        self._hang = hang
        self.closed = False

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        if self._hang:
            await asyncio.sleep(3600)
        raise ConnectionError("socket closed")

    async def close(self):
        self.closed = True


def _trade(symbol: str, *, price: float, trade_id: int, at: datetime) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@trade",
            "data": {
                "e": "trade",
                "s": symbol,
                "p": str(price),
                "q": "1.0",
                "T": int(at.timestamp() * 1000),
                "t": trade_id,
                "m": False,
            },
        }
    )


def _kline(symbol: str, *, closed: bool, open_ms: int, close: float, close_ms: int) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@kline_1m",
            "data": {
                "e": "kline",
                "s": symbol,
                "k": {
                    "i": "1m",
                    "t": open_ms,
                    "T": close_ms,
                    "o": "1",
                    "h": "2",
                    "l": "1",
                    "c": str(close),
                    "v": "10",
                    "x": closed,
                },
            },
        }
    )


def _connector(connection: _FakeConnection):
    """An async factory that always returns the same fake connection."""

    async def _connect(_url):
        return connection

    return _connect


def _ticker(symbol: str, *, close: float, at: datetime, bid: float = 1.0, ask: float = 2.0) -> str:
    """A rolling 24-hour ticker event, in the venue's WebSocket spelling.

    The WebSocket payload uses single-letter keys (``c``, ``b``, ``a``, ``C``),
    not the REST endpoint's ``lastPrice``/``bidPrice`` spellings. The two are
    different shapes of the same object, and both have to normalize.
    """
    ms = int(at.timestamp() * 1000)
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@ticker",
            "data": {
                "e": "24hrTicker",
                "E": ms,
                "s": symbol,
                "c": str(close),
                "b": str(bid),
                "a": str(ask),
                "v": "12.5",
                "q": "900.25",
                "h": "3.0",
                "l": "0.5",
                "p": "-0.25",
                "P": "-0.5",
                "C": ms,
            },
        }
    )


def _drive(stream: MarketStream, connector, *, seconds: float = 0.4) -> None:
    """Run the real consumer loop on its thread for ``seconds``, then stop it.

    A dropped socket makes the consumer reconnect forever (by design), so a test
    that only wants to observe a state must bound the run rather than await it.
    """
    stream._connector = connector
    stream.start()
    time.sleep(seconds)
    stream.stop()


def _pump_for(stream: MarketStream, connection: _FakeConnection, seconds: float, **_) -> None:
    """Run the consumer against one fixed connection for ``seconds``, then stop.

    Unlike :func:`_drive`, the connector always returns the *same* connection, so
    the consumer stays on it and the test observes the state a single socket
    reaches (connected, then stale) rather than a reconnect cycle.
    """
    _drive(stream, _connector(connection), seconds=seconds)


class TestStreamStatus:
    def test_starts_disconnected(self):
        assert MarketStream(symbols=["BTC/USDT"]).status.state == DISCONNECTED

    def test_subscription_names_cover_symbols_and_intervals(self):
        names = MarketStream(symbols=["BTC/USDT"], intervals=("1m", "5m")).status.subscriptions
        assert "btcusdt@trade" in names
        assert "btcusdt@kline_1m" in names
        assert "btcusdt@kline_5m" in names

    def test_stream_url_is_a_combined_stream(self):
        assert "/stream?streams=" in MarketStream(symbols=["BTC/USDT"]).stream_url()

    def test_describe_never_requires_a_key(self):
        assert MarketStream(symbols=["BTC/USDT"]).describe()["crypto_stream_requires_key"] is False

    def test_status_age_is_none_before_any_message(self):
        assert MarketStream(symbols=["BTC/USDT"]).status.age_seconds() is None


class TestMessageHandling:
    def _stream(self, now: datetime, **hooks):
        stream = MarketStream(
            symbols=["BTC/USDT"],
            callbacks=StreamCallbacks(**hooks),
            clock=lambda: now,
        )
        return stream

    def test_a_trade_reaches_the_callback(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = self._stream(now, on_trade=events.append)
        stream._handle_message(_trade("BTCUSDT", price=100.0, trade_id=1, at=now))
        assert len(events) == 1
        assert events[0].price == 100.0
        assert events[0].symbol == "BTC/USDT"

    def test_a_trade_also_drives_the_live_price(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        tickers: list = []
        stream = self._stream(now, on_ticker=tickers.append)
        stream._handle_message(_trade("BTCUSDT", price=100.0, trade_id=1, at=now))
        assert tickers and tickers[0].last == 100.0

    def test_a_duplicate_trade_is_not_counted_twice(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = self._stream(now, on_trade=events.append)
        message = _trade("BTCUSDT", price=100.0, trade_id=7, at=now)
        stream._handle_message(message)
        stream._handle_message(message)
        assert len(events) == 1
        assert stream.status.duplicates == 1

    def test_a_future_dated_event_is_refused(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = self._stream(now, on_trade=events.append)
        stream._handle_message(
            _trade("BTCUSDT", price=100.0, trade_id=1, at=now + timedelta(minutes=5))
        )
        assert events == []
        assert stream.status.rejected == 1

    def test_an_out_of_order_event_is_refused(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = self._stream(now, on_trade=events.append)
        stream._handle_message(_trade("BTCUSDT", price=100.0, trade_id=1, at=now))
        stream._handle_message(
            _trade("BTCUSDT", price=101.0, trade_id=2, at=now - timedelta(seconds=30))
        )
        assert len(events) == 1
        assert stream.status.rejected == 1

    def test_a_malformed_message_is_rejected_not_raised(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = self._stream(now, on_trade=events.append)
        stream._handle_message("not json at all")
        stream._handle_message(json.dumps({"data": "a string, not an object"}))
        assert events == []
        assert stream.status.rejected == 2

    def test_a_closed_kline_reaches_the_callback(self):
        now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        candles: list = []
        stream = self._stream(now, on_candle=candles.append)
        open_ms = int(datetime(2026, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
        close_ms = int(datetime(2026, 1, 1, 0, 1, tzinfo=UTC).timestamp() * 1000)
        stream._handle_message(
            _kline("BTCUSDT", closed=True, open_ms=open_ms, close=100.0, close_ms=close_ms)
        )
        assert len(candles) == 1
        assert candles[0].symbol == "BTC/USDT"

    def test_a_replayed_closed_kline_is_deduplicated(self):
        now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        candles: list = []
        stream = self._stream(now, on_candle=candles.append)
        open_ms = int(datetime(2026, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
        close_ms = int(datetime(2026, 1, 1, 0, 1, tzinfo=UTC).timestamp() * 1000)
        message = _kline("BTCUSDT", closed=True, open_ms=open_ms, close=100.0, close_ms=close_ms)
        stream._handle_message(message)
        stream._handle_message(message)
        assert len(candles) == 1
        assert stream.status.duplicates == 1

    def test_a_forming_bar_is_forwarded_for_display(self):
        now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        candles: list = []
        stream = self._stream(now, on_candle=candles.append)
        open_ms = int(datetime(2026, 1, 1, 0, 4, tzinfo=UTC).timestamp() * 1000)
        close_ms = int(datetime(2026, 1, 1, 0, 6, tzinfo=UTC).timestamp() * 1000)
        stream._handle_message(
            _kline("BTCUSDT", closed=False, open_ms=open_ms, close=100.0, close_ms=close_ms)
        )
        assert len(candles) == 1
        # The consumer forwards it; it is the engine (whose clock is the decision
        # instant) that must refuse to treat an unclosed bar as decision data.
        assert candles[0].close_time > now


class TestLifecycle:
    def test_a_silent_socket_reads_stale_never_connected(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        states: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            stale_seconds=0.05,
            clock=lambda: now,
            callbacks=StreamCallbacks(on_status=lambda s: states.append(s.state)),
        )
        _pump_for(stream, _FakeConnection([], hang=True), 0.3)
        # The socket connected, then went silent: the tape must report stale, not
        # keep claiming connected with a frozen price.
        assert CONNECTED in states
        assert STALE in states

    def test_a_live_feed_reads_connected(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            stale_seconds=30.0,
            callbacks=StreamCallbacks(on_trade=events.append),
            clock=lambda: now,
        )
        message = _trade("BTCUSDT", price=100.0, trade_id=1, at=now)
        _pump_for(stream, _FakeConnection([message, message], hang=True), 0.2)
        assert stream.status.messages == 2
        assert len(events) == 1

    def test_a_dropped_socket_is_reported_reconnecting(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        states: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            backoff_base=0.01,
            clock=lambda: now,
            callbacks=StreamCallbacks(on_status=lambda s: states.append(s.state)),
        )
        _drive(stream, _connector(_FakeConnection([], hang=False)), seconds=0.3)
        assert stream.status.reconnects >= 1
        assert stream.status.errors >= 1
        assert RECONNECTING in states
        assert stream.status.state == STOPPED

    def test_a_dropped_socket_never_reports_connected_after_the_drop(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        stream = MarketStream(symbols=["BTC/USDT"], backoff_base=0.01, clock=lambda: now)
        _drive(stream, _connector(_FakeConnection([], hang=False)), seconds=0.3)
        assert stream.status.state != CONNECTED
        assert stream.status.state in (RECONNECTING, STOPPED)

    def test_stop_is_idempotent_and_sets_stopped(self):
        stream = MarketStream(symbols=["BTC/USDT"])
        stream.stop()
        stream.stop()
        assert stream.status.state == STOPPED

    def test_start_then_stop_returns_to_stopped(self):
        stream = MarketStream(symbols=["BTC/USDT"], stale_seconds=0.05)
        _drive(stream, _connector(_FakeConnection([], hang=True)), seconds=0.2)
        assert stream.status.state == STOPPED


class TestTickerStream:
    """The rolling-ticker stream: the bid, the ask and the 24-hour statistics.

    The trade stream carries only a price, so the live view's bid, ask, spread
    and daily statistics have to come from the ticker stream. These tests pin
    that the ticker event is recognized (the venue spells it ``24hrTicker``) and
    that its single-letter payload normalizes to the same shape as the REST
    ticker.
    """

    def test_the_stream_subscribes_to_the_ticker_stream(self):
        stream = MarketStream(symbols=["BTC/USDT"])
        assert "btcusdt@ticker" in stream.status.subscriptions

    def test_a_ticker_event_reaches_the_callback(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        seen: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            clock=lambda: now,
            callbacks=StreamCallbacks(on_ticker=seen.append),
        )
        _drive(
            stream,
            _connector(_FakeConnection([_ticker("BTCUSDT", close=100.0, at=now)])),
            seconds=0.3,
        )
        assert len(seen) == 1
        assert seen[0].symbol == "BTC/USDT"

    def test_a_ticker_carries_the_bid_the_ask_and_the_daily_stats(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        seen: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            clock=lambda: now,
            callbacks=StreamCallbacks(on_ticker=seen.append),
        )
        _drive(
            stream,
            _connector(_FakeConnection([_ticker("BTCUSDT", close=100.0, at=now)])),
            seconds=0.3,
        )
        ticker = seen[0]
        assert ticker.last == 100.0
        assert ticker.bid == 1.0
        assert ticker.ask == 2.0
        assert ticker.spread == 1.0
        assert ticker.volume_24h == 12.5
        assert ticker.quote_volume_24h == 900.25
        assert ticker.high_24h == 3.0
        assert ticker.low_24h == 0.5
        assert ticker.change_24h == -0.25
        assert ticker.change_percent_24h == -0.5

    def test_a_ticker_is_not_counted_as_rejected(self):
        """A ticker the consumer cannot parse would vanish silently.

        The event type is ``24hrTicker``, not ``ticker``: an unrecognized event
        is dropped without an error, so a typo here would leave the live view
        with no ticker at all and no complaint. The rejection counter is what
        makes that visible.
        """
        now = datetime(2026, 1, 1, tzinfo=UTC)
        stream = MarketStream(symbols=["BTC/USDT"], clock=lambda: now)
        _drive(
            stream,
            _connector(_FakeConnection([_ticker("BTCUSDT", close=100.0, at=now)])),
            seconds=0.3,
        )
        assert stream.status.messages >= 1
        assert stream.status.rejected == 0


class TestPerStreamOrdering:
    """The out-of-order guard is per stream, not global.

    The venue interleaves the 1m and 5m kline streams, and a 5-minute bar's open
    time is five minutes behind a 1-minute bar's. A single global watermark
    would therefore reject every 5m bar that arrived after a 1m one — which is
    the normal case, not an anomaly — and the strategy would silently lose its
    5m confirmation.
    """

    def test_a_5m_bar_after_a_1m_bar_is_accepted(self):
        now = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)
        seen: list = []
        stream = MarketStream(
            symbols=["BTC/USDT"],
            intervals=("1m", "5m"),
            clock=lambda: now,
            callbacks=StreamCallbacks(on_candle=seen.append),
        )
        one_minute = _kline(
            "BTCUSDT",
            closed=True,
            open_ms=int(now.timestamp() * 1000) - 60_000,
            close=100.0,
            close_ms=int(now.timestamp() * 1000) - 1,
        )
        five_minute = json.dumps(
            {
                "stream": "btcusdt@kline_5m",
                "data": {
                    "e": "kline",
                    "s": "BTCUSDT",
                    "k": {
                        "i": "5m",
                        "t": int(now.timestamp() * 1000) - 300_000,
                        "T": int(now.timestamp() * 1000) - 1,
                        "o": "1",
                        "h": "2",
                        "l": "1",
                        "c": "101",
                        "v": "10",
                        "x": True,
                    },
                },
            }
        )
        _drive(
            stream,
            _connector(_FakeConnection([one_minute, five_minute])),
            seconds=0.3,
        )
        assert stream.status.rejected == 0
        assert {c.interval for c in seen} == {"1m", "5m"}

    def test_an_out_of_order_bar_within_one_stream_is_refused(self):
        """The guard still bites on the stream it belongs to."""
        now = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)
        stream = MarketStream(symbols=["BTC/USDT"], intervals=("1m",), clock=lambda: now)
        base = int(now.timestamp() * 1000)
        newer = _kline(
            "BTCUSDT", closed=True, open_ms=base - 60_000, close=100.0, close_ms=base - 1
        )
        older = _kline(
            "BTCUSDT", closed=True, open_ms=base - 120_000, close=99.0, close_ms=base - 61_000
        )
        _drive(stream, _connector(_FakeConnection([newer, older])), seconds=0.3)
        assert stream.status.rejected == 1
