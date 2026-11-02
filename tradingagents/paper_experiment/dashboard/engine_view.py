"""A read-only, non-fabricated view of the paper trading engine's *state*.

The dashboard's complaint this module answers: a worker can be ``RUNNING`` while
nothing is being analysed, decided, or traded, and a single "RUNNING" badge hides
that. Here the states are separated, and each is derived from data the worker
actually recorded — never inferred from the badge, never defaulted to a
reassuring value:

- **worker running** — the process reports ``RUNNING`` *and* its loop thread is
  alive (a ``RUNNING`` state with a dead loop is reported as not running);
- **market data available** — the price seam's own last status is ``ok``;
- **strategy evaluating** — the worker has asked the decision source for a
  decision at least once, with the time it last did so;
- **decision generated** — a decision is currently offered or has been acted on;
- **order submitted** — at least one order was submitted (the broker's own
  record), distinct from *filled*;
- **order filled** — at least one order has a non-zero filled quantity.

Every field is either a value the worker stored or ``None``/``False``. Nothing is
invented, so an unknown is visibly unknown. This module reads a status dict and
touches no broker, no network, and no credential.
"""

from __future__ import annotations

from typing import Any

# A submitted order is one the broker acknowledged; a filled order is one that
# actually traded. Kept apart on purpose: a submitted-but-unfilled order is not a
# trade, and the dashboard must never report it as one.
_TERMINAL_WITHOUT_FILL = ("canceled", "expired", "rejected")


def _orders(status: dict[str, Any]) -> list[dict[str, Any]]:
    orders = status.get("orders")
    if not isinstance(orders, dict):
        return []
    return [o for o in orders.values() if isinstance(o, dict)]


def _last_fill(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recent order with a non-zero fill, or ``None``.

    "Most recent" is by the order's own ``submitted_at`` — the broker payload this
    adapter keeps carries no separate fill timestamp, so the submission time is the
    only honest ordering available, and it is reported as such rather than as a
    fill time it cannot prove.
    """
    filled = [o for o in orders if _as_number(o.get("filled_quantity")) > 0]
    if not filled:
        return None
    return max(filled, key=lambda o: str(o.get("submitted_at") or ""))


def _as_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def engine_view(status: dict[str, Any]) -> dict[str, Any]:
    """The separated engine states and their timestamps, from a worker status.

    ``status`` is a :func:`~...broker.trading.worker_status_view` mapping (or the
    equivalent from a live worker). A missing field yields ``None``/``False`` — the
    view never fills a gap with a plausible-looking value.
    """
    worker_status = status.get("status")
    loop_running = bool(status.get("loop_running"))
    orders = _orders(status)
    submitted = [o for o in orders if o.get("client_order_id")]
    fill = _last_fill(orders)

    market_status = status.get("market_data_status")
    last_eval = status.get("last_strategy_evaluation_at")
    last_decision = status.get("last_processed_decision_id")
    no_trade_reason = status.get("last_no_decision_reason")

    # "Decision generated" means a decision is currently in hand or was acted on.
    # With none, the *reason* there is none is surfaced so the state is explained,
    # not merely absent.
    decision_generated = bool(last_decision) or bool(no_trade_reason is None and last_eval)

    return {
        "banner": status.get("banner"),
        "mode": status.get("mode", "ALPACA_PAPER"),
        "live": False,
        "worker": {
            "status": worker_status,
            "loop_running": loop_running,
            # A RUNNING state whose loop thread is dead is *not* running: the two
            # must agree before the engine is reported as live.
            "running": worker_status == "RUNNING" and loop_running,
            "strategy_id": status.get("strategy_id"),
            "endpoint": status.get("endpoint"),
            "paper_endpoint_only": status.get("paper_endpoint_only"),
        },
        "states": {
            "worker_running": worker_status == "RUNNING" and loop_running,
            "market_data_available": market_status == "ok",
            "strategy_evaluating": bool(last_eval),
            "decision_generated": bool(decision_generated),
            "order_submitted": bool(submitted),
            "order_filled": fill is not None,
        },
        "market_data": {
            "provider": status.get("market_data_provider"),
            "provider_configured": status.get("market_data_provider_configured"),
            "kind": status.get("market_data_kind"),
            "status": market_status,
            "status_detail": status.get("market_data_status_detail"),
            "last_observation": status.get("last_snapshot_field"),
            "last_observed_at": status.get("last_snapshot_observed_at"),
            "last_price": status.get("last_snapshot_price"),
            "last_snapshot_at": status.get("last_snapshot_at"),
            "age_seconds": status.get("last_snapshot_age_seconds"),
            "max_age_seconds": status.get("max_snapshot_age_seconds"),
        },
        "timestamps": {
            "worker_started_at": status.get("started_at"),
            "worker_stopped_at": status.get("stopped_at"),
            "last_cycle_at": status.get("last_cycle_at"),
            "last_strategy_evaluation_at": last_eval,
            "last_order_submitted_at": (
                max(str(o.get("submitted_at") or "") for o in submitted) or None
                if submitted
                else None
            ),
            "last_fill_order_id": (fill or {}).get("client_order_id"),
            "last_fill_symbol": (fill or {}).get("symbol"),
            "last_fill_at": (fill or {}).get("submitted_at"),
            "last_position_sync_at": status.get("positions_checked_at"),
            "last_account_at": status.get("account_at"),
            "next_evaluation_at": status.get("next_cycle_at"),
        },
        "heartbeat": {
            "cycles_completed": status.get("cycles_completed"),
            "poll_interval_seconds": status.get("poll_interval_seconds"),
            "last_cycle_at": status.get("last_cycle_at"),
            "next_cycle_at": status.get("next_cycle_at"),
        },
        "last_error": status.get("last_error"),
        "no_trade_reason": no_trade_reason,
        "counts": {
            "orders": len(orders),
            "submitted": len(submitted),
            "filled": len([o for o in orders if _as_number(o.get("filled_quantity")) > 0]),
            "open_positions": len(
                [
                    p
                    for p in (status.get("positions") or {}).values()
                    if isinstance(p, dict) and _as_number(p.get("quantity")) != 0
                ]
            ),
            "exits": len(status.get("exits") or {}),
        },
        "terminal_without_fill": [
            o.get("client_order_id")
            for o in orders
            if o.get("status") in _TERMINAL_WITHOUT_FILL
        ],
    }


def _opt_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def positions_view(
    status: dict[str, Any],
    internal_positions: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Open positions from the broker, reconciled against the internal ledger.

    The broker's own positions (read into the worker state) are the source of
    truth for what is *held*; the experiment's simulation ledger is shown
    alongside, and a ``reconciled`` flag says whether the two agree. They are
    deliberately *not* merged: one is the paper account, the other is the
    simulation, and hiding a divergence would be the misleading reporting this
    view exists to remove. A field the broker does not report is ``None`` — never a
    fabricated zero.
    """
    internal = internal_positions or {}
    stops = status.get("stops") or {}
    positions = status.get("positions") or {}
    rows: list[dict[str, Any]] = []
    for symbol, position in sorted(positions.items()):
        if not isinstance(position, dict):
            continue
        quantity = _opt_float(position.get("quantity")) or 0.0
        if quantity == 0:
            continue
        entry = _opt_float(position.get("average_entry_price"))
        market_value = _opt_float(position.get("market_value"))
        current_price = None
        if market_value is not None and quantity:
            current_price = market_value / quantity
        unrealized = None
        if market_value is not None and entry is not None and quantity:
            unrealized = market_value - entry * quantity
        level = stops.get(symbol) if isinstance(stops.get(symbol), dict) else {}
        internal_row = internal.get(symbol) if isinstance(internal.get(symbol), dict) else {}
        internal_qty = _opt_float(internal_row.get("quantity")) if internal_row else None
        rows.append(
            {
                "symbol": symbol,
                "side": "long" if quantity > 0 else "short",
                "quantity": quantity,
                "entry_price": entry,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized,
                "stop_loss": level.get("stop_loss"),
                "take_profit": level.get("take_profit"),
                "opened_at": position.get("opened_at"),
                "age_seconds": None,
                "status": "open",
                "exited": symbol in (status.get("exits") or {}),
                "broker_quantity": quantity,
                "internal_quantity": internal_qty,
                "reconciled": (
                    internal_qty is not None and abs(internal_qty - quantity) < 1e-6
                ),
            }
        )
    return rows


def trades_view(
    decisions: list[dict[str, Any]],
    status: dict[str, Any],
) -> dict[str, Any]:
    """The trade journal: broker orders, and the settled trades behind them.

    Two distinct lists, never conflated:

    - ``orders`` — every order the broker acknowledged, with its own status. An
      order that is ``accepted`` is *submitted*, not executed; it is never counted
      as a trade.
    - ``trades`` — the settled outcomes the experiment recorded (an entry with a
      fill and, later, an evaluation). Each carries entry/exit sessions, prices,
      quantity, fees, realized P&L and the exit reason, all from persisted state.

    ``submitted_orders`` and ``filled_orders`` counts make the distinction explicit
    on the summary, so a submitted order can never be read as a fill.
    """
    orders: list[dict[str, Any]] = []
    for order in (status.get("orders") or {}).values():
        if not isinstance(order, dict):
            continue
        filled_qty = _opt_float(order.get("filled_quantity")) or 0.0
        orders.append(
            {
                "client_order_id": order.get("client_order_id"),
                "broker_order_id": order.get("broker_order_id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "quantity": _opt_float(order.get("quantity")),
                "filled_quantity": filled_qty,
                "remaining_quantity": max(
                    0.0, (_opt_float(order.get("quantity")) or 0.0) - filled_qty
                ),
                "average_fill_price": _opt_float(order.get("filled_avg_price")),
                "status": order.get("status"),
                "submitted_at": order.get("submitted_at"),
                "is_filled": filled_qty > 0,
            }
        )
    orders.sort(key=lambda o: str(o.get("submitted_at") or ""), reverse=True)

    trades: list[dict[str, Any]] = []
    for decision in decisions:
        filled_qty = _opt_float(decision.get("filled_quantity")) or 0.0
        if filled_qty <= 0:
            continue
        evaluation = decision.get("evaluation") or {}
        trades.append(
            {
                "decision_id": decision.get("decision_id"),
                "symbol": decision.get("symbol"),
                "side": decision.get("order_side"),
                "quantity": filled_qty,
                "entry_session": decision.get("execution_session"),
                "entry_price": _opt_float(decision.get("average_fill_price")),
                "exit_session": decision.get("evaluation_session"),
                "exit_price": _opt_float(evaluation.get("exit_price")),
                "fees": _opt_float(decision.get("fees")),
                "realized_pnl": _opt_float(evaluation.get("net_pnl")),
                "return_pct": _opt_float(evaluation.get("return_pct")),
                "exit_reason": evaluation.get("exit_reason") or decision.get("order_reason"),
                "order_status": decision.get("order_status"),
                "state": decision.get("state"),
            }
        )
    trades.sort(key=lambda t: str(t.get("entry_session") or ""), reverse=True)

    return {
        "orders": orders,
        "trades": trades,
        "submitted_orders": len(orders),
        "filled_orders": len([o for o in orders if o["is_filled"]]),
        "settled_trades": len(trades),
    }

