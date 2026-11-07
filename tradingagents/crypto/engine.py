"""The unified crypto market-data engine the dashboard and the strategy read.

One object owns both halves of the data path — the **REST** client (history and
snapshots) and the **WebSocket** tape (the live last price, the closed bars) — and
presents a single, normalized, freshness-tracked view. Everything above it (the
dashboard, the TJR source, the backtest) reads this object and never a venue.

The engine keeps, per symbol:

- the latest :class:`~.types.Ticker` from REST and the last traded price from the
  stream, whichever is newer, each carrying its own instant;
- the most recent *closed* bar per interval, and the current *forming* bar for
  display only — a forming bar is never returned as decision data;
- a per-symbol **freshness** verdict derived from the age of the newest accepted
  event, so a silent feed shows ``stale`` rather than a frozen price dressed as
  live.

Two rules are structural, not conventional:

- **A stale price is not a live price.** :meth:`snapshot` reports ``fresh=False``
  with the age, and a consumer that needs a current price must check it; there is
  no method that returns a stale price without the flag.
- **Only closed bars feed a decision.** :meth:`closed_bars` filters out the
  forming bar; :meth:`forming_bar` exists only for the display layer.

The REST client and the connector are injectable, so the whole engine is exercised
offline.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .binance import BinancePublicClient
from .stream import (
    CONNECTED,
    STALE,
    MarketStream,
    StreamCallbacks,
    StreamStatus,
)
from .symbols import parse_pair
from .transport import CryptoDataError
from .types import (
    CANDLE_INTERVALS,
    Candle,
    OrderBook,
    Ticker,
    candles_to_frame,
)

DEFAULT_FRESH_SECONDS = 60.0
"""A REST price older than this is not fresh. The stream's own age is used when a
stream is running, with the stream's own ``stale_seconds`` as its bound."""


@dataclass
class SymbolState:
    """Everything the engine knows about one symbol, each value with its instant."""

    symbol: str
    ticker: Ticker | None = None
    stream_price: float | None = None
    stream_price_at: datetime | None = None
    closed: dict[str, Candle] = field(default_factory=dict)
    forming: dict[str, Candle] = field(default_factory=dict)

    def newest_event_at(self) -> datetime | None:
        candidates = [
            value
            for value in (self.stream_price_at, self.ticker.observed_at if self.ticker else None)
            if value is not None
        ]
        return max(candidates) if candidates else None

    def price(self) -> float | None:
        """The most recent price the engine has, preferring the newer source."""
        stream = self.stream_price_at
        rest = self.ticker.observed_at if self.ticker else None
        if (
            self.stream_price is not None
            and stream is not None
            and (rest is None or stream >= rest)
        ):
            return self.stream_price
        if self.ticker is not None and self.ticker.last is not None:
            return self.ticker.last
        return self.stream_price


class CryptoMarketDataEngine:
    """Live + historical crypto market data, normalized and freshness-tracked.

    ``start_stream`` is opt-in: the engine serves REST snapshots without a stream,
    and a stream is started only when the caller asks. The stream's callbacks write
    into the same per-symbol state the REST path writes, under one lock, so the two
    sources cannot disagree about which price is newest.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        client: BinancePublicClient | None = None,
        intervals: Iterable[str] = ("1m", "5m"),
        fresh_seconds: float = DEFAULT_FRESH_SECONDS,
        stream: MarketStream | None = None,
        clock: Any = None,
    ) -> None:
        self.symbols = tuple(parse_pair(s).canonical for s in symbols)
        self.client = client or BinancePublicClient()
        self.intervals = tuple(intervals)
        self.fresh_seconds = fresh_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self._state: dict[str, SymbolState] = {s: SymbolState(s) for s in self.symbols}
        self.last_error: str | None = None
        self.last_refresh_at: str | None = None
        self._stream = stream

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            **self.client.describe(),
            "crypto_data_symbols": list(self.symbols),
            "crypto_data_intervals": list(self.intervals),
            "crypto_data_fresh_seconds": self.fresh_seconds,
            "crypto_data_last_refresh_at": self.last_refresh_at,
            "crypto_data_last_error": self.last_error,
        }

    # -- stream -----------------------------------------------------------

    def start_stream(self) -> MarketStream:
        """Start (or return) the WebSocket tape feeding this engine."""
        if self._stream is None:
            self._stream = MarketStream(
                symbols=self.symbols,
                intervals=("1m",),
                callbacks=StreamCallbacks(
                    on_ticker=self._on_stream_ticker,
                    on_candle=self._on_stream_candle,
                ),
            )
        self._stream.start()
        return self._stream

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    @property
    def stream(self) -> MarketStream | None:
        return self._stream

    def stream_status(self) -> dict[str, Any]:
        if self._stream is None:
            return {"state": "stopped", "detail": "no stream is running"}
        return self._stream.status.to_dict(now=self._clock())

    def _on_stream_ticker(self, ticker: Ticker) -> None:
        """Merge a streamed ticker into the symbol's state.

        The stream's ``@ticker`` event carries the bid, the ask and the 24-hour
        statistics, so it replaces the REST ticker wholesale. A *trade* event also
        produces a ticker, but only with a price: that one updates the price and
        leaves the richer fields from the last full ticker in place, rather than
        blanking the bid and the ask.

        A stream ticker that is **older** than the ticker the engine already holds
        does not replace it. The engine's price is the newer of the two sources,
        so overwriting a newer ticker with an older one would make the view
        report a stale price as the current one.
        """
        with self._lock:
            state = self._state.get(ticker.symbol)
            if state is None:
                return
            previous = state.ticker
            if previous is not None and ticker.observed_at < previous.observed_at:
                # Older than what we already hold: keep the ticker, but let the
                # stream's own price field carry the observation.
                state.stream_price = ticker.last
                state.stream_price_at = ticker.observed_at
                return
            price_only = ticker.bid is None and ticker.ask is None
            if price_only and previous is not None:
                ticker = replace(
                    ticker,
                    bid=previous.bid,
                    ask=previous.ask,
                    volume_24h=previous.volume_24h,
                    quote_volume_24h=previous.quote_volume_24h,
                    high_24h=previous.high_24h,
                    low_24h=previous.low_24h,
                    change_24h=previous.change_24h,
                    change_percent_24h=previous.change_percent_24h,
                )
            state.ticker = ticker
            state.stream_price = ticker.last
            state.stream_price_at = ticker.observed_at

    def _on_stream_candle(self, candle: Candle) -> None:
        with self._lock:
            state = self._state.get(candle.symbol)
            if state is None:
                return
            closed = candle.close_time <= self._clock()
            if closed:
                state.closed[candle.interval] = candle
                state.forming.pop(candle.interval, None)
            else:
                state.forming[candle.interval] = candle

    # -- REST -------------------------------------------------------------

    def refresh_tickers(self) -> dict[str, Ticker]:
        """Re-read the rolling 24-hour ticker for every symbol."""
        try:
            tickers = self.client.ticker_24h(self.symbols)
        except CryptoDataError as exc:
            self.last_error = str(exc)
            raise
        with self._lock:
            for symbol, ticker in tickers.items():
                state = self._state.get(symbol)
                if state is not None:
                    state.ticker = ticker
            self.last_error = None
            self.last_refresh_at = self._clock().isoformat()
        return tickers

    def refresh_candles(self, interval: str, *, limit: int = 200) -> None:
        """Re-read the newest bars at ``interval`` for every symbol.

        The last bar the venue returns may be forming; it is kept apart from the
        closed bars, so :meth:`closed_bars` never includes it.
        """
        now = self._clock()
        for symbol in self.symbols:
            candles = self.client.klines(symbol, interval, limit=limit)
            with self._lock:
                state = self._state[symbol]
                for candle in candles:
                    if candle.close_time <= now:
                        state.closed[interval] = candle
                    else:
                        state.forming[interval] = candle

    def load_history(
        self, symbol: str, interval: str, *, limit: int, end: datetime | None = None
    ) -> pd.DataFrame:
        """A validated historical frame for a backtest, oldest first."""
        candles = self.client.klines(symbol, interval, limit=limit, end=end)
        return candles_to_frame(candles)

    def order_book(self, symbol: str, *, limit: int = 20) -> OrderBook:
        return self.client.depth(symbol, limit=limit)

    def exchange_registry(self):
        """The venue's instrument registry (tradability + trading rules)."""
        return self.client.exchange_info()

    # -- reads ------------------------------------------------------------

    def tickers(self) -> dict[str, Ticker]:
        with self._lock:
            return {s: st.ticker for s, st in self._state.items() if st.ticker is not None}

    def last_closed_bar(self, symbol: str, interval: str) -> Candle | None:
        """The newest *closed* bar the engine holds for ``symbol``/``interval``."""
        state = self._state.get(parse_pair(symbol).canonical)
        if state is None:
            return None
        return state.closed.get(interval)

    def forming_bar(self, symbol: str, interval: str) -> Candle | None:
        """The current *forming* bar, for display only — never a decision input."""
        state = self._state.get(parse_pair(symbol).canonical)
        return None if state is None else state.forming.get(interval)

    def snapshot(self, symbol: str) -> dict[str, Any]:
        """A per-symbol view for the dashboard, with a freshness verdict.

        ``fresh`` is ``False`` and ``age_seconds`` is reported when the newest
        event is older than the engine's freshness bound, or when nothing has
        arrived at all — a stale or absent price is never presented as current.
        """
        canonical = parse_pair(symbol).canonical
        with self._lock:
            state = self._state.get(canonical)
            if state is None:
                return {
                    "symbol": canonical,
                    "fresh": False,
                    "age_seconds": None,
                    "detail": "unknown symbol",
                }
            newest = state.newest_event_at()
            price = state.price()
            ticker = state.ticker.to_dict() if state.ticker is not None else None
            forming = {interval: candle.to_dict() for interval, candle in state.forming.items()}
            closed = {interval: candle.to_dict() for interval, candle in state.closed.items()}
        age = None if newest is None else (self._clock() - newest).total_seconds()
        fresh = age is not None and age <= self.fresh_seconds and price is not None
        return {
            "symbol": canonical,
            "price": price,
            "fresh": fresh,
            "age_seconds": age,
            "newest_event_at": None if newest is None else newest.isoformat(),
            "ticker": ticker,
            "closed": closed,
            "forming": forming,
        }

    def snapshots(self) -> dict[str, dict[str, Any]]:
        return {s: self.snapshot(s) for s in self.symbols}

    def all_fresh(self) -> bool:
        return all(self.snapshot(s)["fresh"] for s in self.symbols)


def _single_bar_frame(candle: Candle) -> pd.DataFrame:
    """A one-row OHLCV frame for a single held bar, indexed by its start in UTC."""
    index = pd.DatetimeIndex([candle.open_time], tz="UTC")
    return pd.DataFrame(
        [
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
        ],
        index=index,
    )


def build_frame(candles: list[Candle]) -> pd.DataFrame:
    """A validated OHLCV frame from a list of candles (for backtest input)."""
    return candles_to_frame(candles)


def frame_from_engine(engine: CryptoMarketDataEngine, symbol: str, interval: str) -> pd.DataFrame:
    """A validated frame of the engine's closed bars, for a single held bar.

    The engine holds one closed bar per interval from the stream; a caller that
    needs a full window asks the REST client for it (``engine.load_history``). This
    helper exists so a display path can turn the single held bar into the same frame
    shape everything else uses, and it never includes the forming bar.
    """
    candle = engine.last_closed_bar(symbol, interval)
    if candle is None:
        raise CryptoDataError(f"no closed {interval} bar is held for {symbol}")
    return _single_bar_frame(candle)


__all__ = [
    "CANDLE_INTERVALS",
    "CONNECTED",
    "DEFAULT_FRESH_SECONDS",
    "STALE",
    "CryptoMarketDataEngine",
    "StreamStatus",
    "SymbolState",
    "build_frame",
    "frame_from_engine",
]
