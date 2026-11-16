"""Read-only feedback analytics for Bybit Demo paper executions.

This module never changes strategy parameters, risk limits, positions, or orders.
It pairs venue-reported Spot executions FIFO, deducts the actual reported fees
when denominated in the quote asset, and withholds net-PnL claims when the
execution history is incomplete or fee data is unavailable.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any


class LearningDataError(ValueError):
    """The supplied learning data is malformed."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp(value: Any) -> str:
    try:
        milliseconds = int(value)
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _valid_execution_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    quantity = _number(row.get("execQty"))
    price = _number(row.get("execPrice"))
    return bool(
        str(row.get("execId") or "").strip()
        and str(row.get("symbol") or "").strip()
        and str(row.get("side") or "").strip().lower() in {"buy", "sell"}
        and quantity is not None
        and quantity > 0
        and price is not None
        and price > 0
        and _timestamp(row.get("execTime"))
    )


def normalize_execution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize raw Bybit V5 execution rows without guessing.

    A blank orderLinkId is retained as an unattributed execution so it can block
    performance claims; it is never silently discarded as if no trade occurred.
    """
    if not isinstance(rows, list):
        raise LearningDataError("execution rows must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not _valid_execution_row(row):
            continue
        execution_id = str(row.get("execId") or "").strip()
        if execution_id in seen:
            continue
        seen.add(execution_id)
        symbol = str(row.get("symbol") or "").strip().upper()
        normalized.append({
            "execution_id": execution_id,
            "order_id": str(row.get("orderLinkId") or "").strip(),
            "symbol": symbol,
            "side": str(row.get("side") or "").strip().lower(),
            "quantity": float(_number(row.get("execQty"))),
            "price": float(_number(row.get("execPrice"))),
            "fee": _number(row.get("execFee")),
            "fee_currency": str(row.get("feeCurrency") or "").strip().upper() or None,
            "executed_at": _timestamp(row.get("execTime")),
            "raw_time": float(_number(row.get("execTime")) or 0.0),
        })
    normalized.sort(key=lambda item: (item["raw_time"], item["execution_id"]))
    return normalized

def _quote_asset(symbol: str) -> str | None:
    # Bybit V5 execution rows use the venue symbol without a slash (e.g. BTCUSDT).
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return quote
    return None


def build_learning_report(
    worker_state: dict[str, Any],
    execution_rows: list[dict[str, Any]],
    *,
    history_complete: bool,
    observed_at: str | None = None,
    minimum_trades: int = 20,
) -> dict[str, Any]:
    """Create an auditable FIFO spot-trade report; never mutates input data."""
    if not isinstance(worker_state, dict):
        raise LearningDataError("worker state must be an object")
    if minimum_trades < 1:
        raise LearningDataError("minimum_trades must be positive")

    orders = worker_state.get("orders") or {}
    if not isinstance(orders, dict):
        raise LearningDataError("worker-state orders must be a mapping")
    fills = normalize_execution_rows(execution_rows)
    invalid_execution_rows = sum(1 for row in execution_rows if not _valid_execution_row(row))
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    trades: list[dict[str, Any]] = []
    unmatched_sells = 0
    unattributed_executions = 0

    for fill in fills:
        order = orders.get(fill["order_id"])
        if not isinstance(order, dict):
            # Never fold manual/untracked account activity into the bot's PnL.
            unattributed_executions += 1
            continue
        if fill["side"] == "buy":
            decision = order.get("decision_snapshot")
            lots[fill["symbol"]].append({
                "remaining": fill["quantity"],
                "entry_price": fill["price"],
                "entry_fee_remaining": fill["fee"],
                "fee_currency": fill["fee_currency"],
                "entry_fee_known": fill["fee"] is not None,
                "entry_fee_quote": (
                    fill["fee"] is not None
                    and fill["fee_currency"] == _quote_asset(fill["symbol"])
                ),
                "entry_at": fill["executed_at"],
                "entry_execution_id": fill["execution_id"],
                "entry_order_id": fill["order_id"],
                "decision_id": order.get("decision_id"),
                "decision": decision if isinstance(decision, dict) else None,
            })
            continue

        remaining = fill["quantity"]
        if not lots[fill["symbol"]]:
            unmatched_sells += 1
            continue
        while remaining > 1e-12 and lots[fill["symbol"]]:
            lot = lots[fill["symbol"]][0]
            matched = min(remaining, lot["remaining"])
            gross = (fill["price"] - lot["entry_price"]) * matched
            entry_fee = None
            if lot["entry_fee_known"]:
                entry_fee = lot["entry_fee_remaining"] * matched / lot["remaining"]
            exit_fee = None if fill["fee"] is None else fill["fee"] * matched / fill["quantity"]
            fees_known = (
                entry_fee is not None
                and exit_fee is not None
                and lot["entry_fee_quote"]
                and fill["fee_currency"] == _quote_asset(fill["symbol"])
            )
            fees = (entry_fee + exit_fee) if fees_known else None
            trades.append({
                "symbol": fill["symbol"],
                "quantity": matched,
                "entry_price": lot["entry_price"],
                "exit_price": fill["price"],
                "entry_at": lot["entry_at"],
                "exit_at": fill["executed_at"],
                "entry_execution_id": lot["entry_execution_id"],
                "exit_execution_id": fill["execution_id"],
                "decision_id": lot["decision_id"],
                "decision": lot["decision"],
                "gross_pnl": gross,
                "fees": fees,
                "net_pnl": gross - fees if fees is not None else None,
                "fee_data_complete": fees_known,
            })
            lot["remaining"] -= matched
            remaining -= matched
            if lot["entry_fee_remaining"] is not None and lot["remaining"] > 0:
                lot["entry_fee_remaining"] = max(0.0, lot["entry_fee_remaining"] - (entry_fee or 0.0))
            if lot["remaining"] <= 1e-12:
                lots[fill["symbol"]].popleft()
        if remaining > 1e-12:
            unmatched_sells += 1

    closed = len(trades)
    fee_complete = bool(trades) and all(t["fee_data_complete"] for t in trades)
    net_values = [float(t["net_pnl"]) for t in trades if t["net_pnl"] is not None]
    wins = sum(1 for value in net_values if value > 0)
    losses = sum(1 for value in net_values if value < 0)
    gross_profit = sum(value for value in net_values if value > 0)
    gross_loss = -sum(value for value in net_values if value < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    metrics_ready = (
        history_complete
        and fee_complete
        and closed >= minimum_trades
        and len(net_values) == closed
        and unattributed_executions == 0
        and invalid_execution_rows == 0
    )

    symbol_stats: dict[str, dict[str, Any]] = {}
    for symbol in sorted({t["symbol"] for t in trades}):
        group = [t for t in trades if t["symbol"] == symbol]
        valued = [float(t["net_pnl"]) for t in group if t["net_pnl"] is not None]
        symbol_stats[symbol] = {
            "closed_trades": len(group),
            "net_pnl": sum(valued) if len(valued) == len(group) else None,
            "net_profit_factor": (
                sum(v for v in valued if v > 0) / -sum(v for v in valued if v < 0)
                if any(v < 0 for v in valued) and len(valued) == len(group)
                else None
            ),
        }

    suggestions: list[str] = []
    if not history_complete:
        suggestions.append("Execution history is truncated; do not draw performance conclusions yet.")
    if not fee_complete and closed:
        suggestions.append("Some matched trades lack quote-currency fee data; net PnL is not fully verified.")
    if unattributed_executions:
        suggestions.append("Unattributed account executions were excluded; verify no manual or external orders affected this sample.")
    if invalid_execution_rows:
        suggestions.append("Malformed execution rows were found; net performance is withheld until the history is clean.")
    if closed < minimum_trades:
        suggestions.append(f"Collect at least {minimum_trades - closed} more completed trades before judging the strategy.")
    if metrics_ready and profit_factor is not None and profit_factor <= 1.0:
        suggestions.append("Net profit factor is at or below 1; keep the current strategy in observation and test changes offline only.")
    if metrics_ready and profit_factor is not None and profit_factor > 1.0:
        suggestions.append("Net profit factor is above 1 in this sample; retain the holdout gate and do not increase risk from this result alone.")

    return {
        "schema_version": 1,
        "observed_at": observed_at or datetime.now(UTC).isoformat(),
        "mode": "read_only_feedback",
        "strategy_mutation_enabled": False,
        "automatic_promotion_enabled": False,
        "history_complete": bool(history_complete),
        "execution_rows": len(fills),
        "closed_trades": closed,
        "open_lots": sum(len(queue) for queue in lots.values()),
        "unmatched_sell_executions": unmatched_sells,
        "unattributed_executions": unattributed_executions,
        "invalid_execution_rows": invalid_execution_rows,
        "fee_data_complete": fee_complete,
        "metrics_ready": metrics_ready,
        "minimum_trades_required": minimum_trades,
        "net_win_rate": wins / len(net_values) if metrics_ready and net_values else None,
        "net_profit_factor": profit_factor if metrics_ready else None,
        "net_expectancy": sum(net_values) / len(net_values) if metrics_ready and net_values else None,
        "net_pnl": sum(net_values) if metrics_ready else None,
        "wins": wins if metrics_ready else None,
        "losses": losses if metrics_ready else None,
        "by_symbol": symbol_stats,
        "suggestions": suggestions,
        "trades": trades[-100:],
        "limitations": [
            "FIFO matching assumes long-only Spot inventory and pairs buys with later sells.",
            "Net PnL is withheld unless both execution fees are known in the quote asset.",
            "A truncated execution-history page set is not a complete performance sample.",
            "This report analyzes observed outcomes; it does not train or change the trading strategy.",
        ],
    }
