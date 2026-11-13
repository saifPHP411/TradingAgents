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

from ..broker.trading import MARKET_DATA_STATUS_OK

# A submitted order is one the broker acknowledged; a filled order is one that
# actually traded. Kept apart on purpose: a submitted-but-unfilled order is not a
# trade, and the dashboard must never report it as one.
_TERMINAL_WITHOUT_FILL = ("canceled", "expired", "rejected")


def _has_priced(status: dict[str, Any]) -> bool:
    """Whether the worker has actually priced something, from its own record.

    Evidence that the price seam worked: a cycle that reached a price (a
    ``reference_price`` on the last cycle) or an order that was submitted (which
    required a price). Used only to resolve a seam that reports no status of its
    own — never to override an explicit non-ok status.
    """
    last_cycle = status.get("last_cycle")
    if isinstance(last_cycle, dict) and last_cycle.get("reference_price") is not None:
        return True
    return any(o.get("client_order_id") for o in _orders(status))


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


def _blocker(
    *,
    worker_status: Any,
    loop_running: bool,
    market_data_available: bool,
    market_status: Any,
    last_eval: Any,
    no_trade_code: Any,
    no_trade_reason: Any,
) -> dict[str, Any] | None:
    """The single thing preventing a trade right now, or ``None`` when nothing is.

    Ordered by what stops the pipeline earliest, so the answer is the *first* gate
    rather than every gate at once. Returns ``None`` when the worker is not running
    (a stopped worker is not "blocked"; it is off), when no cycle has run yet (no
    evidence either way), or when the last cycle placed an order. Otherwise it names
    a stable ``code``, a human ``detail``, and whether the operator must act.
    """
    if worker_status != "RUNNING" or not loop_running:
        return None
    if last_eval is None:
        # The loop is up but has not reached its first evaluation yet — honest
        # "not yet", not a fabricated blocker.
        return {
            "code": "AWAITING_FIRST_EVALUATION",
            "detail": "the loop is running but has not evaluated the strategy yet",
            "operator_action_required": False,
        }
    if no_trade_code is None and no_trade_reason is None:
        # The last evaluation produced an order; nothing is blocking.
        return None
    if not market_data_available:
        return {
            "code": "MARKET_DATA_UNAVAILABLE",
            "detail": no_trade_reason
            or f"the price seam reported {market_status or 'no status'}",
            "operator_action_required": True,
        }
    if no_trade_code == "MARKET_DATA_UNAVAILABLE":
        return {
            "code": "MARKET_DATA_UNAVAILABLE",
            "detail": no_trade_reason or "the price source could not price the symbol",
            "operator_action_required": True,
        }
    return {
        "code": no_trade_code or "NO_DECISION_DUE",
        "detail": no_trade_reason,
        # A cadence gap clears itself; a size or credential problem needs the
        # operator. Anything else is reported as needing attention rather than
        # assumed benign.
        "operator_action_required": no_trade_code
        not in ("NO_DECISION_DUE", "ALREADY_PROCESSED", "ORDER_EXISTS"),
    }


def _statistics(status: dict[str, Any]) -> dict[str, Any]:
    """The decision statistics the dashboard shows, from data the worker stored.

    Two tallies, deliberately kept apart because they answer different questions:

    - **strategy ratings** (``strategy``): how often the *source* chose BUY / SELL /
      HOLD, with the model's own last HOLD reason. Only the source sees a rating the
      worker then refused to trade, so this is the honest "what did the strategy
      want" figure.
    - **worker outcomes** (``worker``): how many decisions produced an order and how
      many did not, split by the :data:`NO_TRADE_REASONS` code — including
      ``RISK_REJECTED``. This is "what the worker actually did".

    Both come from persisted state, so a restart does not reset them. A source that
    does not describe itself (the historical source) contributes no strategy tally,
    and it is reported as ``None`` rather than a fabricated zero.
    """
    strategy = status.get("decision_source_statistics")
    worker = status.get("decision_statistics")
    return {
        "strategy": strategy if isinstance(strategy, dict) else None,
        "worker": worker if isinstance(worker, dict) else None,
        "last_evaluated_at": status.get("last_strategy_evaluation_at"),
    }


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
    no_trade_code = status.get("last_no_trade_code")

    # A seam that does not report a status is not proof the market data is broken:
    # the historical/test seams price an order without recording one. Treat "no
    # status" as unknown, and only an explicit non-ok status as unavailable, so the
    # engine page never claims the market data is down while the worker is pricing
    # orders from it.
    market_data_available = market_status == MARKET_DATA_STATUS_OK or (
        market_status is None and _has_priced(status)
    )

    # "Decision generated" means a decision is currently in hand or was acted on.
    # With none, the *reason* there is none is surfaced so the state is explained,
    # not merely absent.
    decision_generated = bool(last_decision) or bool(no_trade_reason is None and last_eval)

    return {
        "banner": status.get("banner"),
        "mode": status.get("mode") or "UNKNOWN",
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
            "market_data_available": market_data_available,
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
        # Which decision source the worker draws on, and — for the current-market
        # source — its own last market state and last-fetched decision data. A
        # historical source contributes no extra fields, so the *configured* label
        # is what the view reports rather than a fabricated current state.
        # The symbol validation state, straight from the controller's status.
        # It reports whether the venues' own listings have been read and which
        # symbols they refused, so an operator sees a rejected pair instead of a
        # pair that silently never appears.
        "symbols": {
            "validation": status.get("symbol_validation"),
        },
        "decision_source": {
            "configured": status.get("decision_source_configured"),
            "kind": status.get("decision_source_kind"),
            "symbols": status.get("decision_source_symbols"),
            "strategy": status.get("decision_source_strategy"),
            "market_state": status.get("decision_source_last_market_state"),
            "market_session": status.get("decision_source_last_market_session"),
            "market_hours_provider": status.get("market_hours_provider"),
            "decision_data_provider": status.get("decision_data_provider"),
            "decision_data_kind": status.get("decision_data_kind"),
            "decision_data_timeframe": status.get("decision_data_timeframe"),
            "decision_data_endpoint": status.get("decision_data_endpoint"),
            "decision_data_feed": status.get("decision_data_feed"),
            "decision_data_last_fetch_at": status.get("decision_data_last_fetch_at"),
            "decision_data_last_bar_date": status.get("decision_data_last_bar_date"),
            "decision_data_last_session": status.get("decision_data_last_session"),
            "decision_data_age_days": status.get("decision_data_age_days"),
            "decision_data_bar_count": status.get("decision_data_bar_count"),
            "decision_data_last_error": status.get("decision_data_last_error"),
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
        "no_trade_code": no_trade_code,
        "statistics": _statistics(status),
        "blocker": _blocker(
            worker_status=worker_status,
            loop_running=loop_running,
            market_data_available=market_data_available,
            market_status=market_status,
            last_eval=last_eval,
            no_trade_code=no_trade_code,
            no_trade_reason=no_trade_reason,
        ),
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
    protection = status.get("position_protection") or {}
    positions = status.get("positions") or {}
    rows: list[dict[str, Any]] = []
    for symbol, position in sorted(positions.items()):
        if not isinstance(position, dict):
            continue
        quantity = _opt_float(position.get("quantity")) or 0.0
        if quantity == 0:
            continue
        # Bybit Spot positions use BrokerPosition's native field names
        # (avg_price / mark_price / unrealised_pnl); Alpaca-style positions use
        # average_entry_price / market_value. Read both without inventing data.
        entry = _opt_float(
            position.get("average_entry_price", position.get("avg_price"))
        )
        if entry is None:
            # The worker records the actual average fill beside the protective
            # levels when it accepts the entry. Prefer this exact fill over an
            # unavailable wallet-level Spot cost basis.
            level_for_entry = stops.get(symbol) if isinstance(stops.get(symbol), dict) else {}
            entry = _opt_float(level_for_entry.get("entry_price"))
        current_price = _opt_float(
            position.get("current_price", position.get("mark_price"))
        )
        market_value = _opt_float(position.get("market_value"))
        if market_value is None and current_price is not None:
            market_value = round(abs(quantity) * current_price, 12)

        # Bybit Spot currently has no wallet-level average cost field. A
        # defensible entry estimate is possible only when this worker's recorded
        # fills account for the entire held quantity and there are no sell fills
        # for the symbol. Otherwise leave entry/uPnL unknown rather than present
        # a partial or misleading cost basis.
        if entry is None:
            symbol_orders = [
                order for order in _orders(status)
                if order.get("symbol") == symbol
                and _opt_float(order.get("filled_quantity")) is not None
                and (_opt_float(order.get("filled_quantity")) or 0.0) > 0
            ]
            buy_qty = sum(
                _opt_float(order.get("filled_quantity")) or 0.0
                for order in symbol_orders if str(order.get("side", "")).lower() == "buy"
            )
            sell_qty = sum(
                _opt_float(order.get("filled_quantity")) or 0.0
                for order in symbol_orders if str(order.get("side", "")).lower() == "sell"
            )
            buy_cost = sum(
                (_opt_float(order.get("filled_quantity")) or 0.0)
                * (_opt_float(order.get("filled_avg_price")) or 0.0)
                for order in symbol_orders if str(order.get("side", "")).lower() == "buy"
                and _opt_float(order.get("filled_avg_price")) is not None
            )
            all_buys_priced = all(
                _opt_float(order.get("filled_avg_price")) is not None
                for order in symbol_orders if str(order.get("side", "")).lower() == "buy"
            )
            if (
                symbol_orders and all_buys_priced and sell_qty == 0
                and abs(buy_qty - abs(quantity)) <= max(1e-8, abs(quantity) * 1e-6)
                and buy_qty > 0
            ):
                entry = buy_cost / buy_qty

        unrealized = _opt_float(
            position.get("unrealized_pnl", position.get("unrealised_pnl"))
        )
        if unrealized is None and market_value is not None and entry is not None and quantity:
            unrealized = round((current_price - entry) * abs(quantity), 12) if current_price is not None else round(market_value - entry * abs(quantity), 12)
        level = stops.get(symbol) if isinstance(stops.get(symbol), dict) else {}
        internal_row = internal.get(symbol) if isinstance(internal.get(symbol), dict) else {}
        internal_qty = _opt_float(internal_row.get("quantity")) if internal_row else None
        # The protection state is the worker's own verdict on whether this
        # position is guarded by a recorded exit level. It is separate from the
        # ledger reconciliation below: a position can agree with the ledger and
        # still be unguarded, because the guard comes from a strategy decision
        # and this position has none.
        guard = protection.get(symbol) if isinstance(protection.get(symbol), dict) else {}
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
                "protected": guard.get("protected", False),
                "protection_state": guard.get("protection_state", "UNPROTECTED"),
                "protection_reason": guard.get("reason", "NO_RECORDED_EXIT_LEVELS"),
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

    # A broker fill is a real execution even before its position is closed.
    # Keep it separate from settled round-trip trades: entry fills have no
    # realized P&L or exit price yet, and must never be presented as closed trades.
    executions = [
        {
            "client_order_id": order.get("client_order_id"),
            "broker_order_id": order.get("broker_order_id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "filled_quantity": order["filled_quantity"],
            "average_fill_price": order.get("average_fill_price"),
            "order_timestamp": order.get("submitted_at"),
            "order_status": order.get("status"),
        }
        for order in orders
        if order["is_filled"]
    ]

    return {
        "orders": orders,
        "executions": executions,
        "trades": trades,
        "submitted_orders": len(orders),
        "filled_orders": len(executions),
        "filled_executions": len(executions),
        "settled_trades": len(trades),
    }

