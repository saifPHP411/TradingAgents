"""Read-only Bybit linear WebSocket observer, checks the three roles each second.

WebSocket is push-driven (ticker/level-1 book/closed 1m klines). Indicators
are recalculated on the latest *closed* one-minute candles only; a Donchian
15-minute candidate can be acted on only during its next 60-second window.
All three votes need the same symbol, completed candle and direction.

This process NEVER places orders, signs requests, starts Demo trading, changes
leverage, or claims that a live order has been submitted. It is a SHADOW desk.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from .triad_consensus import TriadInput, evaluate_triad

PERMITTED_WS = "wss://stream.bybit.com/v5/public/linear"
_SYM = re.compile(r"^[A-Z0-9]{3,25}USDT$")
MINUTE_MS = 60_000
MAX_HISTORY = 540


def utc_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


class TriadShadowWatcher:
    """Pure, injectable market-cache and 1-second decision evaluator."""

    def __init__(self, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT")):
        if not symbols or len(symbols) > 24 or len(set(symbols)) != len(symbols):
            raise ValueError("require 1–24 unique USDT perpetual symbols")
        if any(not _SYM.fullmatch(s) for s in symbols):
            raise ValueError("invalid USDT perpetual symbol")
        self.symbols = symbols
        self.history: dict[str, deque] = {
            s: deque(maxlen=MAX_HISTORY) for s in symbols
        }
        self.mark: dict[str, tuple[float, int]] = {}
        self.book: dict[str, tuple[float, float, int]] = {}
        self._emitted: set[str] = set()
        self._emitted_order: deque[str] = deque(maxlen=256)
        self.received = 0
        self.market_events = 0
        self.rejected = 0
        self.scans = 0

    def clear_live_quotes(self) -> None:
        """Fail closed after a disconnect: never reuse an old book or mark."""
        self.mark.clear()
        self.book.clear()

    def bootstrap(self, symbol: str, rows: list) -> int:
        """Inject verified closed chronological 1m OHLCV bars from REST.

        A network read isn't required in unit tests. Historical bars only
        seed indicators; the watcher never treats them as live quotes.
        """
        from .backtest.weekly_candidates import Bar

        if symbol not in self.history:
            raise ValueError("unknown symbol")
        if not rows:
            return 0
        if any(not isinstance(bar, Bar) for bar in rows):
            raise ValueError("bootstrap rows must be verified Bar records")
        if any(rows[i].ms - rows[i - 1].ms != MINUTE_MS for i in range(1, len(rows))):
            raise ValueError("historical 1m candles contain gaps or duplicates")
        self.history[symbol] = deque(rows[-MAX_HISTORY:], maxlen=MAX_HISTORY)
        return len(self.history[symbol])

    def ingest(self, message: dict, *, received_ms: int | None = None) -> None:
        """Process *public linear* ticker, L1 snapshot, or closed 1m kline.

        Rejects malformed data without altering last valid market state.
        Receive time does not overwrite actual Bybit timestamps.
        """
        self.received += 1
        now = received_ms if received_ms is not None else utc_ms()
        if not isinstance(message, dict):
            self.rejected += 1
            return
        topic = message.get("topic", "")
        if not isinstance(topic, str):
            self.rejected += 1
            return
        if topic.startswith("tickers."):
            symbol = topic.removeprefix("tickers.")
            data = message.get("data")
            if symbol not in self.history or not isinstance(data, dict):
                self.rejected += 1
                return
            try:
                mark = float(data["markPrice"])
                ts = int(message["ts"])
            except (ValueError, TypeError, KeyError):
                self.rejected += 1
                return
            if not (math.isfinite(mark) and mark > 0 and 0 <= now-ts <= 10_000):
                self.rejected += 1
                return
            if symbol in self.mark and ts < self.mark[symbol][1]:
                self.rejected += 1
                return
            self.mark[symbol] = (mark, ts)
            self.market_events += 1
        elif topic.startswith("orderbook.1."):
            symbol = topic.removeprefix("orderbook.1.")
            data = message.get("data")
            if symbol not in self.history or not isinstance(data, dict):
                self.rejected += 1
                return
            try:
                bid = float(data["b"][0][0])
                ask = float(data["a"][0][0])
                ts = int(message["ts"])
            except (IndexError, ValueError, TypeError, KeyError):
                self.rejected += 1
                return
            if (not all(math.isfinite(x) and x > 0 for x in (bid, ask))
                    or bid >= ask or not 0 <= now-ts <= 10_000):
                self.rejected += 1
                return
            if symbol in self.book and ts < self.book[symbol][2]:
                self.rejected += 1
                return
            self.book[symbol] = (bid, ask, ts)
            self.market_events += 1
        elif topic.startswith("kline.1."):
            from .backtest.weekly_candidates import Bar

            symbol = topic.removeprefix("kline.1.")
            payload = message.get("data")
            if symbol not in self.history or not isinstance(payload, list):
                self.rejected += 1
                return
            for row in payload:
                if not isinstance(row, dict) or row.get("confirm") is not True:
                    continue
                try:
                    ms = int(row["start"])
                    bar = Bar(
                        ms=ms, open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                except (ValueError, TypeError, KeyError):
                    self.rejected += 1
                    continue
                if (not all(math.isfinite(x) for x in (
                    bar.open, bar.high, bar.low, bar.close, bar.volume,
                )) or min(bar.open, bar.high, bar.low, bar.close) <= 0
                        or bar.volume < 0
                        or bar.high < max(bar.open, bar.low, bar.close)
                        or bar.low > min(bar.open, bar.high, bar.close)
                        or not 0 <= now - (bar.ms + MINUTE_MS) <= 15_000):
                    self.rejected += 1
                    continue
                history = self.history[symbol]
                if history and bar.ms <= history[-1].ms:
                    # Closed candle duplicate after reconnect: never double count.
                    continue
                if history and bar.ms - history[-1].ms != MINUTE_MS:
                    history.clear()
                    self.rejected += 1
                history.append(bar)
                self.market_events += 1
        # Unknown topics include subscription acknowledgements and pong: safe to ignore.

    def tick(self, *, now_ms: int | None = None) -> dict:
        """Review cached market state at the caller's one-second heartbeat."""
        from .backtest.weekly_candidates import indicators, signal

        now = now_ms if now_ms is not None else utc_ms()
        self.scans += 1
        approved = []
        vetoes = []
        for symbol in self.symbols:
            bars = list(self.history[symbol])
            if len(bars) < 241:
                vetoes.append({"symbol": symbol, "reason": "warmup_or_history_gap"})
                continue
            last = bars[-1]
            bar_end = last.ms + MINUTE_MS
            if not (bar_end % 900_000 == 0 and 0 <= now-bar_end < 60_000):
                vetoes.append({"symbol": symbol, "reason": "no_fresh_15m_close"})
                continue
            feat = indicators(bars)
            direction = signal(bars, feat, len(bars)-1, "donchian_15m_trend")
            if not direction:
                vetoes.append({"symbol": symbol, "reason": "sniper_no_trigger"})
                continue
            if symbol not in self.mark or symbol not in self.book:
                vetoes.append({"symbol": symbol, "reason": "unverified_live_market_data"})
                continue
            mark, mark_ts = self.mark[symbol]
            bid, ask, book_ts = self.book[symbol]
            result = evaluate_triad(TriadInput(
                symbol=symbol, candle_end_ms=bar_end, now_ms=now,
                direction=direction, close=last.close,
                close_15m_ago=bars[-16].close,
                ema60=feat["ema60"][-1], ema240=feat["ema240"][-1],
                rsi=feat["rsi"][-1], mark=mark,
                mark_ts_ms=mark_ts, best_bid=bid, best_ask=ask,
                book_ts_ms=book_ts, venue_filters_verified=False,
            ))
            if not result["all_three_approve"]:
                vetoes.append(result)
                continue
            if result["event_id"] in self._emitted:
                continue
            if len(self._emitted_order) == self._emitted_order.maxlen:
                expired = self._emitted_order.popleft()
                self._emitted.discard(expired)
            self._emitted_order.append(result["event_id"])
            self._emitted.add(result["event_id"])
            approved.append(result)
        return {
            "checked_at_ms": now,
            "symbols_scanned": len(self.symbols),
            "scans_completed": self.scans,
            "new_unanimous_shadow_opportunities": approved,
            "vetoes": vetoes,
            "market_feed": "Bybit public linear WebSocket push plus 1-second check",
            "orders_submitted": False, "execution_authorized": False,
            "research_only": True, "demo_ready": False,
        }


def load_rest_history(watcher: TriadShadowWatcher) -> dict[str, str]:
    """Seed fixed indicators with CLOSED 1m Bybit linear perpetual candles."""
    from .backtest.weekly_candidates import Bar
    from .bybit_perpetual import BybitLinearPublicClient

    client = BybitLinearPublicClient()
    now = utc_ms()
    outcomes = {}
    for symbol in watcher.symbols:
        try:
            frame = client.klines(
                symbol, start_ms=now - MAX_HISTORY*MINUTE_MS,
                end_ms=now-MINUTE_MS, interval="1m", include_current=False,
            )
            rows = [
                Bar(
                    ms=int(dt.timestamp()*1000),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for dt, row in frame.iterrows()
            ]
            count = watcher.bootstrap(symbol, rows)
            outcomes[symbol] = f"closed_1m_candles:{count}"
        except (ValueError, OSError, RuntimeError) as exc:
            # Do not invent seed candles or silently continue with stale data.
            outcomes[symbol] = f"not_ready:{type(exc).__name__}"
    return outcomes


async def observe_forever(watcher: TriadShadowWatcher) -> None:
    """Push-driven WebSocket consumer; every second assess cached events."""
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            'Read-only stream requires pip install "websockets>=14,<17"'
        ) from exc

    topics = []
    for symbol in watcher.symbols:
        topics.extend([
            f"tickers.{symbol}", f"orderbook.1.{symbol}", f"kline.1.{symbol}",
        ])
    backoff = 1
    while True:
        watcher.clear_live_quotes()
        try:
            async with websockets.connect(
                PERMITTED_WS, ping_interval=20, ping_timeout=10, max_size=1_000_000
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": topics}))
                backoff = 1
                next_second = (utc_ms() // 1000 + 1) * 1000
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
                        watcher.ingest(json.loads(raw))
                    except TimeoutError:
                        pass
                    except (TypeError, ValueError, json.JSONDecodeError):
                        watcher.rejected += 1
                    now = utc_ms()
                    if now >= next_second:
                        # Checks every wall-clock second, even with no new quote,
                        # but old quotes veto rather than trigger any trade.
                        print(json.dumps(watcher.tick(now_ms=now)), flush=True)
                        next_second = (now // 1000 + 1) * 1000
        except Exception as exc:  # noqa: BLE001 - all WS disconnects must reconnect
            print(json.dumps({
                "stream_status": "reconnecting", "error": type(exc).__name__,
                "orders_submitted": False,
            }), flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"],
    )
    parser.add_argument(
        "--max-seconds", type=int,
        help="Optional time limit (1–120 s) for read-only CI stream verification",
    )
    parser.add_argument("--out", type=Path, help="Write bounded observer proof JSON")
    args = parser.parse_args()
    if args.max_seconds is not None and not 1 <= args.max_seconds <= 120:
        parser.error("--max-seconds must be in the inclusive range 1 to 120")
    watcher = TriadShadowWatcher(tuple(args.symbols))
    bootstrap = load_rest_history(watcher)
    print(json.dumps({
        "mode": "SHADOW_ONLY_NO_ORDER_SUBMISSION",
        "strategy": "donchian_15m_trend_plus_triads",
        "roles": ["sniper", "analyst", "professor"],
        "bootstrap": bootstrap,
    }), flush=True)
    if args.max_seconds is None:
        asyncio.run(observe_forever(watcher))
        return
    try:
        asyncio.run(asyncio.wait_for(
            observe_forever(watcher), timeout=args.max_seconds
        ))
    except TimeoutError:
        pass
    report = {
        "status": (
            "public_market_data_observed" if watcher.market_events and watcher.scans
            else "unavailable_or_no_public_events"
        ),
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "symbols": list(watcher.symbols),
        "bounded_seconds": args.max_seconds,
        "bootstrap": bootstrap,
        "public_market_events_received": watcher.market_events,
        "one_second_checks_completed": watcher.scans,
        "rejected_messages": watcher.rejected,
        "notes": (
            "A public market event proves only source reachability; "
            "it does not verify 15m readiness, three-vote approval, "
            "Bybit order fills or profitable trading."
        ),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report["status"] != "public_market_data_observed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
