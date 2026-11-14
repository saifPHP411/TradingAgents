"""Replay public Bybit market data to estimate passive limit-order execution.

This is a research simulator, not an exchange adapter. Queue priority is unknowable
from public L2 data; the model therefore exposes conservative queue assumptions.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

BUY = "Buy"
SELL = "Sell"


def _d(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


@dataclass(frozen=True)
class InstrumentRules:
    category: str
    symbol: str
    tick_size: Decimal
    qty_step: Decimal
    min_order_qty: Decimal
    min_notional: Decimal
    max_order_qty: Decimal | None = None

    @classmethod
    def from_bybit_instrument(cls, item: dict[str, Any], category: str) -> InstrumentRules:
        price = item.get("priceFilter") or {}
        lot = item.get("lotSizeFilter") or {}
        # Spot V5 uses minOrderAmt; linear/inverse use minNotionalValue.
        minimum_notional = lot.get("minOrderAmt") if category == "spot" else lot.get("minNotionalValue")
        if minimum_notional is None:
            minimum_notional = "0"
        return cls(
            category=category,
            symbol=str(item["symbol"]),
            tick_size=_d(price.get("tickSize"), "tickSize"),
            qty_step=_d(lot.get("qtyStep"), "qtyStep"),
            min_order_qty=(\n                Decimal(str(lot["minOrderQty"]))\n                if category != "spot" and lot.get("minOrderQty")\n                else Decimal("0")\n            ),
            min_notional=Decimal(str(minimum_notional)),
            max_order_qty=(
                Decimal(str(lot["maxOrderQty"])) if lot.get("maxOrderQty") else None
            ),
        )


def validate_post_only_order(
    *, rules: InstrumentRules, side: str, quantity: Any, price: Any,
    best_bid: Any, best_ask: Any,
) -> dict[str, str]:
    """Validate and normalize a passive order against current public instrument/book data."""
    if side not in (BUY, SELL):
        raise ValueError("side must be Buy or Sell")
    qty, px = _d(quantity, "quantity"), _d(price, "price")
    bid, ask = _d(best_bid, "best_bid"), _d(best_ask, "best_ask")
    if bid >= ask:
        raise ValueError("book must have best_bid below best_ask")
    if px % rules.tick_size:
        raise ValueError("price does not align to tickSize")
    if qty % rules.qty_step:
        raise ValueError("quantity does not align to qtyStep")
    if qty < rules.min_order_qty:
        raise ValueError("quantity is below minOrderQty")
    if rules.max_order_qty is not None and qty > rules.max_order_qty:
        raise ValueError("quantity exceeds maxOrderQty")
    if qty * px < rules.min_notional:
        raise ValueError("order notional is below minimum")
    if side == BUY and px >= ask:
        raise ValueError("PostOnly buy would cross the ask")
    if side == SELL and px <= bid:
        raise ValueError("PostOnly sell would cross the bid")
    return {"symbol": rules.symbol, "side": side, "qty": format(qty, "f"), "price": format(px, "f")}


@dataclass
class ReplayOrder:
    order_id: str
    submitted_ts_ms: int
    side: str
    price: Decimal
    quantity: Decimal
    latency_ms: int = 100
    ttl_ms: int = 30_000
    queue_factor: Decimal = Decimal("1.5")
    queue_ahead: Decimal = Decimal("0")
    filled_qty: Decimal = Decimal("0")
    first_fill_ts_ms: int | None = None
    status: str = "pending"
    reject_reason: str | None = None

    @property
    def active_ts_ms(self) -> int:
        return self.submitted_ts_ms + self.latency_ms

    @property
    def expires_ts_ms(self) -> int:
        return self.active_ts_ms + self.ttl_ms


@dataclass
class ReplayFill:
    order_id: str
    timestamp_ms: int
    side: str
    price: float
    quantity: float
    mid_at_fill: float | None
    maker_fee_bps: float
    markout_1s: float | None = None
    markout_5s: float | None = None
    markout_30s: float | None = None
    net_markout_30s: float | None = None


@dataclass
class _Book:
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)

    def update(self, data: dict[str, Any], snapshot: bool) -> None:
        if snapshot:
            self.bids.clear()
            self.asks.clear()
        for side_name, target in (("b", self.bids), ("a", self.asks)):
            for price, size in data.get(side_name, []):
                px, qty = Decimal(str(price)), Decimal(str(size))
                if qty <= 0:
                    target.pop(px, None)
                else:
                    target[px] = qty

    @property
    def bid(self) -> Decimal | None:
        return max(self.bids) if self.bids else None

    @property
    def ask(self) -> Decimal | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return float((self.bid + self.ask) / 2)

    def displayed_at(self, side: str, price: Decimal) -> Decimal:
        return (self.bids if side == BUY else self.asks).get(price, Decimal("0"))


class BybitExecutionReplay:
    """Conservative L2 + public-trade replay with latency, queue and partial fills.

    Trade-side interpretation follows Bybit publicTrade.S: it is the taker side.
    Queue reductions from book deltas are NOT credited as fills, because public L2
    cannot distinguish cancellations from executions. Only eligible public trades
    consume queue ahead and then simulated order quantity.
    """

    def __init__(self, *, maker_fee_bps: float, queue_factor: float = 1.5):
        if maker_fee_bps < 0 or queue_factor < 1:
            raise ValueError("maker fee must be non-negative and queue_factor >= 1")
        self.maker_fee_bps = float(maker_fee_bps)
        self.queue_factor = Decimal(str(queue_factor))
        self.book = _Book()
        self.orders: dict[str, ReplayOrder] = {}
        self.fills: list[ReplayFill] = []
        self.mid_history: list[tuple[int, float]] = []
        self.rejects: dict[str, int] = defaultdict(int)
        self.events_seen = 0
        self.cancellations = 0
        self.sequence_gaps = 0
        self._last_book_update_id: dict[str, int] = {}
        self._book_valid = False
        self.symbol: str | None = None

    def submit_order(
        self, *, order_id: str, timestamp_ms: int, side: str,
        price: Any, quantity: Any, latency_ms: int = 100, ttl_ms: int = 30_000,
    ) -> None:
        if order_id in self.orders:
            raise ValueError("duplicate order_id")
        if side not in (BUY, SELL) or latency_ms < 0 or ttl_ms <= 0:
            raise ValueError("invalid side, latency, or ttl")
        self.orders[order_id] = ReplayOrder(
            order_id, int(timestamp_ms), side, _d(price, "price"),
            _d(quantity, "quantity"), int(latency_ms), int(ttl_ms), self.queue_factor,
        )

    def cancel_order(self, order_id: str) -> None:
        """Cancel a simulated resting order; already executed quantity is retained."""
        order = self.orders[order_id]
        if order.status in ("pending", "open", "partial"):
            order.status = "cancelled"
            self.cancellations += 1

    def replace_order(
        self, order_id: str, *, new_order_id: str, timestamp_ms: int,
        side: str, price: Any, quantity: Any, latency_ms: int = 100,
        ttl_ms: int = 30_000,
    ) -> None:
        """Cancel/replace with a fresh queue position; never preserves old priority."""
        self.cancel_order(order_id)
        self.submit_order(
            order_id=new_order_id, timestamp_ms=timestamp_ms, side=side,
            price=price, quantity=quantity, latency_ms=latency_ms, ttl_ms=ttl_ms,
        )

    def _expire_due(self, timestamp_ms: int) -> None:
        for order in self.orders.values():
            if order.status in ("open", "partial") and timestamp_ms >= order.expires_ts_ms:
                order.status = "expired"

    def _activate_due(self, timestamp_ms: int) -> None:
        for order in self.orders.values():
            if order.status != "pending" or order.active_ts_ms > timestamp_ms:
                continue
            bid, ask = self.book.bid, self.book.ask
            if bid is None or ask is None or bid >= ask:
                order.status, order.reject_reason = "rejected", "invalid_or_missing_book"
            elif (order.side == BUY and order.price >= ask) or (order.side == SELL and order.price <= bid):
                order.status, order.reject_reason = "rejected", "post_only_would_cross"
            else:
                displayed = self.book.displayed_at(order.side, order.price)
                order.queue_ahead = displayed * order.queue_factor
                order.status = "open"
            if order.status == "rejected":
                self.rejects[order.reject_reason] += 1

    def process(self, event: dict[str, Any]) -> None:
        """Consume normalized or raw Bybit WebSocket event JSON."""
        self.events_seen += 1
        received = int(event.get("received_ts_ms", event.get("ts", 0)))
        topic = str(event.get("topic", ""))
        if topic.startswith(("orderbook.", "publicTrade.")):
            event_symbol = topic.rsplit(".", 1)[-1]
            if self.symbol is None:
                self.symbol = event_symbol
            elif event_symbol != self.symbol:
                raise ValueError("one replay instance can process only one symbol")
        data = event.get("data") or {}
        event_type = event.get("type")
        if topic.startswith("orderbook.") and event_type in ("snapshot", "delta"):
            self._expire_due(received)
            update_id = data.get("u")
            if event_type == "snapshot":
                self._book_valid = True
                if update_id is not None:
                    self._last_book_update_id[topic] = int(update_id)
            elif update_id is not None:
                previous = self._last_book_update_id.get(topic)
                current = int(update_id)
                if current == 1 or (previous is not None and current != previous + 1):
                    self.sequence_gaps += 1
                    self._book_valid = False
                    self.book.update({}, snapshot=True)
                elif previous is not None:
                    self._last_book_update_id[topic] = current
                else:
                    self._book_valid = False
            if self._book_valid:
                self.book.update(data, snapshot=(event_type == "snapshot"))
                mid = self.book.mid()
                if mid is not None:
                    self.mid_history.append((received, mid))
            if self._book_valid:
                self._activate_due(received)
        elif topic.startswith("publicTrade."):
            trades = data if isinstance(data, list) else [data]
            for trade in trades:
                self._expire_due(received)
                if self._book_valid:
                    self._activate_due(received)
                    # Compare local receive timestamps consistently with simulated latency.
                    self._process_trade(trade, received)
        else:
            return

    def _process_trade(self, trade: dict[str, Any], timestamp_ms: int) -> None:
        try:
            price = Decimal(str(trade["p"]))
            quantity = Decimal(str(trade["v"]))
            taker_side = trade["S"]
        except (KeyError, InvalidOperation):
            return
        if price <= 0 or quantity <= 0 or taker_side not in (BUY, SELL):
            return
        # A sell taker consumes bids; a buy taker consumes asks.
        passive_side = BUY if taker_side == SELL else SELL
        for order in self.orders.values():
            if order.status != "open" or order.side != passive_side:
                continue
            if timestamp_ms >= order.expires_ts_ms:
                order.status = "expired"
                continue
            eligible = price <= order.price if order.side == BUY else price >= order.price
            if not eligible:
                continue
            remaining_trade = quantity
            if order.queue_ahead > 0:
                consumed = min(order.queue_ahead, remaining_trade)
                order.queue_ahead -= consumed
                remaining_trade -= consumed
            if remaining_trade <= 0:
                continue
            remaining_order = order.quantity - order.filled_qty
            fill_qty = min(remaining_order, remaining_trade)
            if fill_qty <= 0:
                continue
            if order.first_fill_ts_ms is None:
                order.first_fill_ts_ms = timestamp_ms
            order.filled_qty += fill_qty
            mid = self.book.mid()
            self.fills.append(ReplayFill(
                order.order_id, timestamp_ms, order.side, float(order.price),
                float(fill_qty), mid, self.maker_fee_bps,
            ))
            order.status = "filled" if order.filled_qty >= order.quantity else "partial"

    def finalize_markouts(self) -> None:
        for fill in self.fills:
            signed = 1.0 if fill.side == BUY else -1.0
            for horizon, attr in ((1_000, "markout_1s"), (5_000, "markout_5s"), (30_000, "markout_30s")):
                target = fill.timestamp_ms + horizon
                mid = next((m for t, m in self.mid_history if t >= target), None)
                if mid is not None and fill.mid_at_fill is not None:
                    setattr(fill, attr, signed * (mid - fill.price))
            if fill.markout_30s is not None:
                fee = fill.price * self.maker_fee_bps / 10_000
                fill.net_markout_30s = fill.markout_30s - fee

    def report(self) -> dict[str, Any]:
        self.finalize_markouts()
        total = len(self.orders)
        filled_orders = sum(o.filled_qty > 0 for o in self.orders.values())
        complete = sum(o.status == "filled" for o in self.orders.values())
        markout_values = [\n            f.net_markout_30s / f.price * 10_000\n            for f in self.fills if f.net_markout_30s is not None and f.price > 0\n        ]
        markouts = [f.net_markout_30s * f.quantity for f in self.fills if f.net_markout_30s is not None]
        mean_markout = _mean(markout_values)
        variance = (
            sum((value - mean_markout) ** 2 for value in markout_values) / (len(markout_values) - 1)
            if len(markout_values) > 1 and mean_markout is not None else None
        )
        cumulative = peak = max_drawdown = 0.0
        for value in markouts:
            cumulative += value
            peak = max(peak, cumulative)
            max_drawdown = max(max_drawdown, peak - cumulative)
        total_fees = sum(f.price * f.quantity * f.maker_fee_bps / 10_000 for f in self.fills)
        gains = sum(x for x in markouts if x > 0)
        losses = abs(sum(x for x in markouts if x < 0))
        fill_delays = [
            o.first_fill_ts_ms - o.active_ts_ms for o in self.orders.values()
            if o.first_fill_ts_ms is not None
        ]
        return {
            "orders": total, "orders_with_any_fill": filled_orders,
            "fully_filled_orders": complete,
            "fill_rate_pct": filled_orders / total * 100 if total else 0.0,
            "full_fill_rate_pct": complete / total * 100 if total else 0.0,
            "rejected_orders": sum(self.rejects.values()),
            "reject_reasons": dict(self.rejects),
            "unfilled_or_expired_orders": sum(o.filled_qty == 0 for o in self.orders.values()),
            "partial_orders": sum(0 < o.filled_qty < o.quantity for o in self.orders.values()),
            "mean_time_to_first_fill_ms": sum(fill_delays) / len(fill_delays) if fill_delays else None,
            "fills": len(self.fills), "filled_quantity": sum(f.quantity for f in self.fills),
            "mean_markout_1s": _mean([f.markout_1s for f in self.fills]),
            "mean_markout_5s": _mean([f.markout_5s for f in self.fills]),
            "mean_markout_30s": _mean([f.markout_30s for f in self.fills]),
            "fill_results": [
                {
                    "order_id": fill.order_id, "timestamp_ms": fill.timestamp_ms,
                    "side": fill.side, "price": fill.price, "quantity": fill.quantity,
                    "mid_at_fill": fill.mid_at_fill,
                    "markout_1s": fill.markout_1s, "markout_5s": fill.markout_5s,
                    "markout_30s": fill.markout_30s,
                    "estimated_maker_fee": fill.price * fill.quantity * fill.maker_fee_bps / 10_000,
                    "net_markout_30s": (
                        fill.net_markout_30s * fill.quantity
                        if fill.net_markout_30s is not None else None
                    ),
                    "net_markout_30s_bps": (
                        fill.net_markout_30s / fill.price * 10_000
                        if fill.net_markout_30s is not None and fill.price > 0 else None
                    ),
                }
                for fill in self.fills
            ],
            "mean_net_markout_30s_bps": mean_markout,
            "per_fill_markout_sharpe_unannualized": (
                mean_markout / variance ** 0.5 if variance and variance > 0 else None
            ),
            "markout_max_drawdown_quote_units": max_drawdown,
            "estimated_maker_fees_quote": total_fees,
            "cancellations": self.cancellations,
            "markout_profit_factor": gains / losses if losses else (float("inf") if gains else 0.0),
            "maker_fee_bps": self.maker_fee_bps,
            "queue_factor": float(self.queue_factor),
            "queue_cancellation_credit": 0.0,
            "events_seen": self.events_seen, "orderbook_sequence_gaps": self.sequence_gaps,
            "symbol": self.symbol,
            "funding_model": "not modeled; do not interpret linear/perpetual results as net returns",
            "warning": "Markout is a post-fill mid-price proxy, not realized round-trip PnL; queue priority is estimated.",
        }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def record_public_market_data(
    *, symbols: list[str], output_path: str, category: str = "spot",
    depth: int = 50, duration_seconds: int = 300,
) -> int:
    """Record raw public Bybit orderbook/trade WebSocket messages as JSONL."""
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install 'tradingagents[market-data]'") from exc
    if category not in ("spot", "linear") or not symbols or duration_seconds <= 0:
        raise ValueError("category, symbols, or duration is invalid")
    if depth not in (1, 50, 200, 1000):
        raise ValueError("unsupported public orderbook depth")
    endpoint = f"wss://stream.bybit.com/v5/public/{category}"
    topics = [topic for symbol in symbols for topic in (
        f"orderbook.{depth}.{symbol}", f"publicTrade.{symbol}"
    )]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    stop = threading.Event()

    with path.open("a", encoding="utf-8") as output:
        def on_open(ws):
            ws.send(json.dumps({"op": "subscribe", "args": topics}))

        def on_message(ws, message):
            nonlocal count
            received = time.time_ns() // 1_000_000
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
            if "topic" not in payload:
                return
            payload["received_ts_ms"] = received
            output.write(json.dumps(payload, separators=(",", ":")) + "\n")
            output.flush()
            count += 1

        def on_error(_ws, error):
            stop.set()
            raise RuntimeError(f"Bybit public WebSocket error: {error}")

        def on_close(_ws, _status, _message):
            stop.set()

        ws = websocket.WebSocketApp(
            endpoint, on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close,
        )
        timer = threading.Timer(duration_seconds, ws.close)
        timer.start()
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        finally:
            timer.cancel()
            stop.set()
    return count


def replay_jsonl(
    path: str, *, maker_fee_bps: float, queue_factor: float = 1.5,
    orders: list[dict[str, Any]] | None = None, symbol: str | None = None,
) -> BybitExecutionReplay:
    """Replay captured market data with simulated submissions interleaved by time.

    Each order dict uses submit_order keyword names: order_id, timestamp_ms, side,
    price, quantity, and optional latency_ms/ttl_ms.
    """
    replay = BybitExecutionReplay(maker_fee_bps=maker_fee_bps, queue_factor=queue_factor)
    with Path(path).open(encoding="utf-8") as source:
        events = [json.loads(line) for line in source if line.strip()]
    symbols = {
        str(event.get("topic", "")).rsplit(".", 1)[-1]
        for event in events
        if str(event.get("topic", "")).startswith(("orderbook.", "publicTrade."))
    }
    if symbol is None:
        if len(symbols) > 1:
            raise ValueError("multi-symbol capture requires an explicit symbol filter")
        symbol = next(iter(symbols), None)
    if symbol is not None:
        events = [
            event for event in events
            if str(event.get("topic", "")).endswith(f".{symbol}")
        ]
    timeline = [
        (int(event.get("received_ts_ms", event.get("ts", 0))), 0, event)
        for event in events
    ]
    timeline.extend(
        (int(order["timestamp_ms"]), 1, order) for order in (orders or [])
    )
    timeline.sort(key=lambda item: (item[0], item[1]))
    for _, kind, payload in timeline:
        if kind == 0:
            replay.process(payload)
        else:
            replay.submit_order(**payload)
    return replay
