"""The live crypto tape: a reconnecting, validated WebSocket consumer.

The dashboard must show a *current* price, which means a streaming feed, not a
poll. A stream is also where a trading system is most easily lied to: a socket
that has silently died, a bar that arrives twice, a timestamp from a clock that
has drifted, or an event older than the one before it will each make a stale
number look live. This module exists to make those failures *visible* rather than
to paper over them.

What it guarantees:

- **Reconnection** with exponential backoff and a bounded jitter, so a dropped
  socket reconnects without hammering the venue and without ever silently giving
  up: the status is ``reconnecting``/``disconnected`` until it is ``connected``
  again.
- **Heartbeat** — the venue's own ping/pong keeps the socket warm; the consumer
  also tracks the time of the last *data* message, so a socket that is open but
  silent is reported ``stale``, never ``live``.
- **Staleness** — :class:`StreamStatus` carries ``stale`` and ``age_seconds``, so
  a view can say "last update 40s ago" instead of showing a frozen price as live.
- **Duplicate protection** — trades are de-duplicated by the venue's trade id and
  bars by their open time, so a replay after a reconnect cannot double-count.
- **Timestamp validation** — an event dated far in the future (clock skew) or
  before the last accepted event is dropped and counted, not accepted.

The consumer is transport-agnostic: it is handed an async *connector* that yields
a connection with ``recv``/``send``/``close``. The default connector is
``websockets.connect``; a test supplies a fake, so the whole reconnect/backoff/
stale/dedup path is exercised offline with no socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from .binance import WS_HOST, kline_stream_names, ticker_stream_names, trade_stream_names
from .symbols import parse_pair
from .types import Candle, PublicTrade, Ticker, candle_from_binance_kline

CONNECTED = "connected"
CONNECTING = "connecting"
RECONNECTING = "reconnecting"
DISCONNECTED = "disconnected"
STOPPED = "stopped"
STALE = "stale"

STREAM_STATES = (CONNECTED, CONNECTING, RECONNECTING, DISCONNECTED, STALE, STOPPED)

DEFAULT_STALE_SECONDS = 30.0
"""No data message for this long and the stream is reported ``stale``."""

DEFAULT_FUTURE_TOLERANCE_SECONDS = 10.0
"""An event dated more than this ahead of the local clock is refused as skewed."""

DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 60.0


class Connection(Protocol):
    """The slice of a WebSocket connection this consumer uses."""

    async def recv(self) -> str | bytes:  # pragma: no cover - protocol
        ...

    async def close(self) -> None:  # pragma: no cover - protocol
        ...


Connector = Callable[[str], Awaitable[Connection]]
"""An async factory: a stream URL in, a connected :class:`Connection` out."""


@dataclass
class StreamStatus:
    """A snapshot of the stream's health, for the dashboard's System Health view."""

    state: str = DISCONNECTED
    url: str | None = None
    connected_at: str | None = None
    last_message_at: str | None = None
    last_event_at: str | None = None
    reconnects: int = 0
    messages: int = 0
    duplicates: int = 0
    rejected: int = 0
    """Events dropped for a bad timestamp (future-dated or out of order)."""
    errors: int = 0
    last_error: str | None = None
    subscriptions: tuple[str, ...] = ()

    def age_seconds(self, *, now: datetime | None = None) -> float | None:
        """Seconds since the last data message, or ``None`` if none arrived."""
        if self.last_message_at is None:
            return None
        reference = now or datetime.now(UTC)
        try:
            last = datetime.fromisoformat(self.last_message_at)
        except ValueError:
            return None
        return (reference - last).total_seconds()

    def to_dict(self, *, now: datetime | None = None) -> dict[str, Any]:
        age = self.age_seconds(now=now)
        return {
            "state": self.state,
            "url": self.url,
            "connected_at": self.connected_at,
            "last_message_at": self.last_message_at,
            "last_event_at": self.last_event_at,
            "age_seconds": age,
            "reconnects": self.reconnects,
            "messages": self.messages,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "errors": self.errors,
            "last_error": self.last_error,
            "subscriptions": list(self.subscriptions),
        }


@dataclass
class StreamCallbacks:
    """What to do with each normalized event. Every hook is optional."""

    on_ticker: Callable[[Ticker], None] | None = None
    on_candle: Callable[[Candle], None] | None = None
    on_trade: Callable[[PublicTrade], None] | None = None
    on_status: Callable[[StreamStatus], None] | None = None


class MarketStream:
    """A reconnecting Binance combined-stream consumer.

    One stream carries the 24-hour ticker, the closed-bar kline events and the
    trade tape for the subscribed symbols. It runs on its own thread with its own
    asyncio loop, so it can feed a stdlib threaded dashboard without blocking it.

    The class does no work until :meth:`start`; :meth:`stop` is idempotent.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        intervals: Iterable[str] = ("1m",),
        host: str = WS_HOST,
        connector: Connector | None = None,
        callbacks: StreamCallbacks | None = None,
        stale_seconds: float = DEFAULT_STALE_SECONDS,
        future_tolerance_seconds: float = DEFAULT_FUTURE_TOLERANCE_SECONDS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.symbols = tuple(parse_pair(s).canonical for s in symbols)
        self.intervals = tuple(intervals)
        self.host = host.rstrip("/")
        self._connector = connector
        self.callbacks = callbacks or StreamCallbacks()
        self.stale_seconds = stale_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._clock = clock or (lambda: datetime.now(UTC))
        self.status = StreamStatus(subscriptions=self._subscription_names())
        self._seen_trades: set[int] = set()
        self._seen_trade_order: list[int] = []
        self._last_candle_open: dict[tuple[str, str], datetime] = {}
        self._watermarks: dict[str, datetime] = {}
        """The last accepted event time, per stream. A single global watermark
        would reject the venue's normal interleaving of the 1m and 5m streams."""
        self._last_event_at: datetime | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- URLs -------------------------------------------------------------

    def _subscription_names(self) -> tuple[str, ...]:
        names: list[str] = ticker_stream_names(self.symbols)
        names += trade_stream_names(self.symbols)
        for interval in self.intervals:
            names += kline_stream_names(self.symbols, interval)
        return tuple(names)

    def stream_url(self) -> str:
        """The combined-stream URL for every subscription."""
        return f"{self.host}/stream?streams={'/'.join(self._subscription_names())}"

    def describe(self) -> dict[str, Any]:
        return {
            "crypto_stream_host": self.host,
            "crypto_stream_symbols": list(self.symbols),
            "crypto_stream_intervals": list(self.intervals),
            "crypto_stream_requires_key": False,
        }

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the consumer on a background thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="crypto-stream", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the consumer to stop and wait briefly for the thread."""
        self._stop.set()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(lambda: None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._set_state(STOPPED)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._consume())
        except Exception as exc:  # noqa: BLE001 - the consumer must never crash the caller
            self._record_error(f"{type(exc).__name__}: {exc}")
            self._set_state(DISCONNECTED)
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    async def _consume(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            self._set_state(CONNECTING if attempt == 0 else RECONNECTING)
            try:
                connector = self._connector or _default_connector
                connection = await connector(self.stream_url())
                attempt = 0
                self._set_state(CONNECTED, connected=True)
                await self._pump(connection)
            except Exception as exc:  # noqa: BLE001 - a dropped socket is expected
                if self._stop.is_set():
                    break
                self._record_error(f"{type(exc).__name__}: {exc}")
                self.status.reconnects += 1
                self._set_state(RECONNECTING)
                delay = min(self.backoff_base * (2 ** min(attempt, 6)), self.backoff_max)
                attempt += 1
                if await self._sleep_or_stop(delay):
                    break
        self._set_state(STOPPED)

    async def _sleep_or_stop(self, delay: float) -> bool:
        """Sleep ``delay`` seconds, returning ``True`` if a stop was requested."""
        step = 0.1
        waited = 0.0
        while waited < delay:
            if self._stop.is_set():
                return True
            await asyncio.sleep(step)
            waited += step
        return self._stop.is_set()

    async def _pump(self, connection: Connection) -> None:
        while not self._stop.is_set():
            try:
                message = await asyncio.wait_for(connection.recv(), timeout=self.stale_seconds)
            except TimeoutError:
                self._set_state(STALE)
                continue
            except Exception:
                with contextlib.suppress(Exception):
                    await connection.close()
                raise
            self._handle_message(message)
            if self.status.state == STALE:
                self._set_state(CONNECTED)

    # -- message handling -------------------------------------------------

    def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            self.status.rejected += 1
            return
        # A combined stream wraps each event as {"stream": ..., "data": {...}}.
        event = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        if not isinstance(event, dict):
            self.status.rejected += 1
            return
        event_type = event.get("e")
        self.status.messages += 1
        self.status.last_message_at = self._clock().isoformat()
        # The venue spells the rolling-ticker event ``24hrTicker``; ``ticker`` is
        # accepted as the alias some venues use for the same payload.
        if event_type in ("24hrTicker", "ticker"):
            self._handle_ticker(event)
        elif event_type == "trade":
            self._handle_trade(event)
        elif event_type == "kline":
            self._handle_kline(event)

    def _valid_timestamp(self, stamp: datetime | None, *, key: str = "") -> bool:
        """Whether an event's timestamp is acceptable: not skewed, not out of order.

        The out-of-order check is **per stream**, not global. A 5-minute bar's open
        time is five minutes behind a 1-minute bar's, and the venue interleaves
        them; a single global watermark would reject every 5m kline that arrived
        after a 1m one, which is the normal case. Each stream therefore keeps its
        own watermark, and an event is refused only if it is older than the last
        event *of the same stream*.
        """
        if stamp is None:
            self.status.rejected += 1
            return False
        now = self._clock()
        if (stamp - now).total_seconds() > self.future_tolerance_seconds:
            self.status.rejected += 1
            return False
        with self._lock:
            previous = self._watermarks.get(key)
            if previous is not None and stamp < previous:
                self.status.rejected += 1
                return False
            self._watermarks[key] = stamp
            if self._last_event_at is None or stamp > self._last_event_at:
                self._last_event_at = stamp
        self.status.last_event_at = self._last_event_at.isoformat()
        return True

    def _handle_ticker(self, event: dict) -> None:
        """A rolling 24-hour ticker event: the bid, the ask and the day's stats.

        The venue's ``@ticker`` stream carries the same fields as the REST
        ``ticker/24hr`` endpoint, so it is parsed by the same normalizer. This is
        what gives the live view its bid, ask, spread and 24-hour change without
        a poll.
        """
        ticker = Ticker.from_binance(event)
        if ticker is None:
            return
        ticker = replace(ticker, symbol=parse_pair(ticker.symbol).canonical)
        if not self._valid_timestamp(ticker.observed_at, key=f"ticker:{ticker.symbol}"):
            return
        if self.callbacks.on_ticker is not None:
            self.callbacks.on_ticker(ticker)

    def _handle_trade(self, event: dict) -> None:
        trade = PublicTrade.from_binance(event)
        if trade is None:
            return
        # Normalize the venue spelling (``BTCUSDT``) to the canonical pair so a
        # trade is keyed the same as a candle and a ticker for the same instrument.
        trade = replace(trade, symbol=parse_pair(trade.symbol).canonical)
        if not self._valid_timestamp(trade.observed_at, key=f"trade:{trade.symbol}"):
            return
        if trade.trade_id is not None:
            if trade.trade_id in self._seen_trades:
                self.status.duplicates += 1
                return
            self._remember_trade(trade.trade_id)
        if self.callbacks.on_trade is not None:
            self.callbacks.on_trade(trade)
        # A trade is also a current price, so it drives the live last-price view.
        if self.callbacks.on_ticker is not None:
            self.callbacks.on_ticker(
                Ticker(
                    symbol=parse_pair(trade.symbol).canonical,
                    last=trade.price,
                    bid=None,
                    ask=None,
                    volume_24h=None,
                    quote_volume_24h=None,
                    high_24h=None,
                    low_24h=None,
                    change_24h=None,
                    change_percent_24h=None,
                    observed_at=trade.observed_at,
                )
            )

    def _handle_kline(self, event: dict) -> None:
        kline = event.get("k")
        if not isinstance(kline, dict):
            self.status.rejected += 1
            return
        interval = str(kline.get("i") or "")
        candle = _candle_from_kline_event(event.get("s"), interval, kline)
        if candle is None:
            return
        # A bar's *open* is what a clock can validate: it is always in the past for
        # a real bar. The *close* of a forming bar is by definition in the future,
        # so validating it would reject every in-progress bar — the one a display
        # needs. The closed/forming split is the venue's own ``x`` flag, carried on
        # the candle and re-checked by the engine against the decision instant.
        key = (candle.symbol, candle.interval)
        if not self._valid_timestamp(candle.open_time, key=f"kline:{key[0]}:{key[1]}"):
            return
        closed = bool(kline.get("x"))
        # Only a *closed* bar may reach the strategy; an in-progress bar is carried
        # for display and is never a decision input.
        previous = self._last_candle_open.get(key)
        if closed:
            if previous is not None and candle.open_time <= previous:
                self.status.duplicates += 1
                return
            self._last_candle_open[key] = candle.open_time
        if self.callbacks.on_candle is not None:
            self.callbacks.on_candle(candle)

    def _remember_trade(self, trade_id: int) -> None:
        self._seen_trades.add(trade_id)
        self._seen_trade_order.append(trade_id)
        if len(self._seen_trade_order) > 5000:
            oldest = self._seen_trade_order.pop(0)
            self._seen_trades.discard(oldest)

    # -- status -----------------------------------------------------------

    def _set_state(self, state: str, *, connected: bool = False) -> None:
        self.status.state = state
        self.status.url = self.stream_url()
        if connected:
            self.status.connected_at = self._clock().isoformat()
        if self.callbacks.on_status is not None:
            self.callbacks.on_status(self.status)

    def _record_error(self, detail: str) -> None:
        self.status.errors += 1
        self.status.last_error = detail


def _candle_from_kline_event(symbol: Any, interval: str, kline: dict) -> Candle | None:
    """Normalize a Binance ``kline`` stream object into a :class:`Candle`.

    The stream reports the *market* spelling (``BTCUSDT``); the candle is keyed by
    the canonical pair so it matches a REST bar for the same instrument.
    """
    market = str(symbol or kline.get("s") or "").upper()
    if not market:
        return None
    try:
        canonical = parse_pair(market).canonical
    except Exception:  # noqa: BLE001 - an unparseable symbol is a rejected event
        canonical = market
    row = [
        kline.get("t"),
        kline.get("o"),
        kline.get("h"),
        kline.get("l"),
        kline.get("c"),
        kline.get("v"),
        kline.get("T"),
    ]
    return candle_from_binance_kline(canonical, interval, row)


async def _default_connector(url: str) -> Connection:
    import websockets

    return await websockets.connect(url, open_timeout=15, ping_interval=20, ping_timeout=20)


__all__ = [
    "CONNECTED",
    "CONNECTING",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_MAX",
    "DEFAULT_FUTURE_TOLERANCE_SECONDS",
    "DEFAULT_STALE_SECONDS",
    "DISCONNECTED",
    "RECONNECTING",
    "STALE",
    "STOPPED",
    "STREAM_STATES",
    "Connection",
    "Connector",
    "MarketStream",
    "StreamCallbacks",
    "StreamStatus",
]
