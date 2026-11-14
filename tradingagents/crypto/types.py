"""Normalized market-data values, so every source speaks one vocabulary.

A ticker from one venue and a ticker from another carry the same facts under
different field names and with different precision. The dashboard, the strategy
and the backtest should not each learn a venue's schema, so every source is
normalized into these small frozen records first, and everything downstream reads
them.

Every value that can be *absent* is ``None`` rather than a zero, because a zero
volume and an unknown volume are different facts and only one of them is a number
a report may show. Every record carries the instant its own fields describe
(``observed_at``), so a freshness check compares a price against the timestamp of
that same price and never against a different field's clock.

Nothing here validates against a venue's rules or decides a trade; these are
values, and the guards that matter live where the values are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

CANDLE_INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h")
"""The bar intervals the crypto engine serves, coarsest last."""

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

BINANCE_INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}
"""The same intervals in Binance's own spelling (identical for these five)."""


def interval_millis(interval: str) -> int:
    """Bar length in milliseconds for a supported interval label."""
    try:
        return _INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}; choose one of {sorted(_INTERVAL_MS)}"
        ) from None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _stamp(value: Any) -> datetime | None:
    """Parse a millisecond epoch or an ISO string to an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return datetime.fromtimestamp(int(text) / 1000.0, tz=UTC)
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    return None


@dataclass(frozen=True)
class Ticker:
    """A 24-hour rolling ticker: the last trade, the quote and the day's stats."""

    symbol: str
    last: float | None
    bid: float | None
    ask: float | None
    volume_24h: float | None
    quote_volume_24h: float | None
    high_24h: float | None
    low_24h: float | None
    change_24h: float | None
    change_percent_24h: float | None
    observed_at: datetime | None
    """The venue's own observation instant, or ``None`` when the venue stated none.

    ``None`` means *unverifiable*, not *now*: a reader must treat it as an
    absence rather than as a current instant, so a payload with no timestamp
    cannot pass a freshness check.
    """

    @property
    def spread(self) -> float | None:
        """``ask - bid`` when both are known and coherent, else ``None``."""
        if self.bid is None or self.ask is None or self.ask < self.bid:
            return None
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float | None:
        """The spread in basis points of the mid, or ``None``."""
        mid = self.mid
        if mid is None or mid <= 0 or self.spread is None:
            return None
        return self.spread / mid * 10_000.0

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last": self.last,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "volume_24h": self.volume_24h,
            "quote_volume_24h": self.quote_volume_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "change_24h": self.change_24h,
            "change_percent_24h": self.change_percent_24h,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
        }

    @classmethod
    def from_binance(cls, payload: dict) -> Ticker:
        """Normalize a Binance 24-hour ticker.

        The REST endpoint (``/api/v3/ticker/24hr``) and the WebSocket's
        ``@ticker`` event carry the same fields under different names: the REST
        object spells the close time ``closeTime``, while the stream event spells
        it ``C`` (with ``E`` as the event time). Both are accepted, so the same
        normalizer serves the snapshot and the live tape.

        **A payload with no timestamp yields a ticker with no timestamp.** The
        observation instant is the venue's own, and when the venue did not state
        one there is no honest instant to report: substituting the local clock
        made every such ticker look infinitely fresh, so a response with no
        timestamp at all passed the freshness check and could size an order. The
        absence is kept, so the seam that reads the ticker can fail closed on an
        unverifiable price instead of trusting a fabricated one.
        """
        observed = _stamp(payload.get("closeTime"))
        if observed is None:
            observed = _stamp(payload.get("C"))
        if observed is None:
            observed = _stamp(payload.get("E"))

        def first(*keys: str) -> float | None:
            """The first key that holds a finite number, or ``None``.

            A plain ``or`` would skip a legitimate zero — a last price of exactly
            zero is not a real observation, but the *volume* may legitimately be
            zero, and losing it would report a non-zero figure that the venue
            never sent.
            """
            for key in keys:
                value = _finite(payload.get(key))
                if value is not None:
                    return value
            return None

        return cls(
            symbol=str(payload.get("symbol") or payload.get("s") or "").upper(),
            last=first("lastPrice", "c"),
            bid=first("bidPrice", "b"),
            ask=first("askPrice", "a"),
            volume_24h=first("volume", "v"),
            quote_volume_24h=first("quoteVolume", "q"),
            high_24h=first("highPrice", "h"),
            low_24h=first("lowPrice", "l"),
            change_24h=first("priceChange", "p"),
            change_percent_24h=first("priceChangePercent", "P"),
            observed_at=observed,
        )


@dataclass(frozen=True)
class BookLevel:
    price: float
    quantity: float

    def to_dict(self) -> dict[str, float]:
        return {"price": self.price, "quantity": self.quantity}


@dataclass(frozen=True)
class OrderBook:
    """A depth snapshot: the top bids and asks, best first."""

    symbol: str
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    observed_at: datetime

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bids": [level.to_dict() for level in self.bids],
            "asks": [level.to_dict() for level in self.asks],
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_binance(cls, symbol: str, payload: dict) -> OrderBook:
        def levels(raw: Any) -> tuple[BookLevel, ...]:
            out: list[BookLevel] = []
            for row in raw or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                price, quantity = _finite(row[0]), _finite(row[1])
                if price is not None and quantity is not None:
                    out.append(BookLevel(price=price, quantity=quantity))
            return tuple(out)

        return cls(
            symbol=symbol.upper(),
            bids=levels(payload.get("bids")),
            asks=levels(payload.get("asks")),
            observed_at=datetime.now(UTC),
        )


@dataclass(frozen=True)
class PublicTrade:
    """One executed public trade from the exchange's trade stream."""

    symbol: str
    price: float
    quantity: float
    trade_id: int | None
    side: str | None
    """``"buy"`` when the aggressor was a buyer (taker buy), else ``"sell"``."""
    observed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "trade_id": self.trade_id,
            "side": self.side,
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_binance(cls, payload: dict) -> PublicTrade | None:
        price = _finite(payload.get("p"))
        quantity = _finite(payload.get("q"))
        if price is None or quantity is None:
            return None
        maker = payload.get("m")
        side = None if not isinstance(maker, bool) else ("sell" if maker else "buy")
        return cls(
            symbol=str(payload.get("s") or "").upper(),
            price=price,
            quantity=quantity,
            trade_id=int(payload["t"]) if str(payload.get("t", "")).isdigit() else None,
            side=side,
            observed_at=_stamp(payload.get("T")) or datetime.now(UTC),
        )


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar, with the instant it *closed*.

    ``closed`` is the venue's own flag where it has one: a bar that is still
    forming must never enter a strategy decision, so it is carried for display
    only and the ``closed`` flag is the single thing that decides which it is.
    """

    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_closed(self) -> bool:
        return self.close_time <= datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    """A validated OHLCV frame indexed by bar start, or raise on a bad bar.

    The frame is what the TJR layer reads, so the same checks it applies — no
    duplicate start, finite positive prices, high not below low — are applied here,
    at the boundary, so a malformed vendor bar is refused where it arrives rather
    than deep inside a rule.
    """
    if not candles:
        raise ValueError("no candles to build a frame from")
    index: list[datetime] = []
    rows: list[dict[str, float]] = []
    seen: set[datetime] = set()
    for candle in candles:
        if candle.open_time in seen:
            raise ValueError(f"duplicate candle start {candle.open_time.isoformat()}")
        seen.add(candle.open_time)
        for value in (candle.open, candle.high, candle.low, candle.close):
            if value != value or value in (float("inf"), float("-inf")) or value <= 0:
                raise ValueError(
                    f"candle at {candle.open_time.isoformat()} carries a non-finite or "
                    f"non-positive price"
                )
        if candle.high < candle.low:
            raise ValueError(f"candle at {candle.open_time.isoformat()} has a high below its low")
        index.append(candle.open_time)
        rows.append(
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
        )
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index, tz="UTC"))
    return frame.sort_index()


def candle_from_binance_kline(symbol: str, interval: str, row: list) -> Candle | None:
    """Parse one Binance kline array into a :class:`Candle`, or ``None``.

    Binance's kline array is positional: ``[openTime, open, high, low, close,
    volume, closeTime, ...]``. A short or non-numeric row is ``None`` rather than
    an exception, so one bad bar does not lose the whole batch.
    """
    if not isinstance(row, (list, tuple)) or len(row) < 7:
        return None
    open_time = _stamp(row[0])
    close_time = _stamp(row[6])
    open_, high, low, close, volume = (_finite(row[i]) for i in (1, 2, 3, 4, 5))
    if None in (open_time, close_time, open_, high, low, close, volume):
        return None
    return Candle(
        symbol=symbol.upper(),
        interval=interval,
        open_time=open_time,
        close_time=close_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


__all__ = [
    "BINANCE_INTERVALS",
    "CANDLE_INTERVALS",
    "BookLevel",
    "Candle",
    "OrderBook",
    "PublicTrade",
    "Ticker",
    "candle_from_binance_kline",
    "candles_to_frame",
    "interval_millis",
]
