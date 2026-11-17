"""Continuous, read-only shadow forward evaluation for fixed perpetual candidates.

This module consumes only completed Bybit USDT-linear 4h candles and public
funding/mark-price history. It replays every pre-registered candidate from a
persisted forward start timestamp, models 1x/2x and three cost scenarios, and
never creates, amends, or cancels an exchange order.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualEMAStrategy,
    run_backtest,
)
from tradingagents.crypto.backtest.perpetual_ema_walkforward import (
    COST_SCENARIOS,
    DEFAULT_SYMBOLS,
    _fetch_history,
)
from tradingagents.crypto.backtest.perpetual_strategy_lab import CANDIDATES
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings

STATE_FILE = "perpetual_shadow_forward.json"
STATE_VERSION = 1
BAR_MS = 4 * 60 * 60 * 1000
WARMUP_BARS = 260
INITIAL_EQUITY_PER_SYMBOL = 1_000.0
LEVERAGES = (1.0, 2.0)
RECENT_TRADES_LIMIT = 120


def _utc_iso(timestamp_ms: int) -> str:
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).isoformat()


def _load_state(root: Path, now_ms: int) -> dict[str, Any]:
    path = root / STATE_FILE
    if path.is_file():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                state.get("version") == STATE_VERSION
                and isinstance(state.get("forward_start_ms"), int)
                and isinstance(state.get("history_start_ms"), int)
            ):
                return state
        except (OSError, ValueError, TypeError):
            pass
        raise ValueError("saved shadow-forward state is invalid; refusing to reset its forward boundary")

    forward_start_ms = (now_ms // BAR_MS + 1) * BAR_MS
    history_start_ms = forward_start_ms - WARMUP_BARS * BAR_MS
    return {
        "version": STATE_VERSION,
        "status": "waiting_for_first_forward_candle",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ms / 1000)),
        "forward_start_ms": forward_start_ms,
        "forward_start": _utc_iso(forward_start_ms),
        "history_start_ms": history_start_ms,
        "warmup_bars": WARMUP_BARS,
        "last_processed_bar_ms": None,
        "last_bar_time": None,
        "last_checked_at": None,
        "last_updated_at": None,
        "cycles": 0,
        "orders_submitted": False,
        "execution_enabled": False,
        "demo_ready": False,
        "strategy_selection": "none; all fixed candidates are monitored in parallel",
        "symbols": list(DEFAULT_SYMBOLS),
        "leverage_levels": list(LEVERAGES),
        "cost_scenarios": list(COST_SCENARIOS),
        "results": [],
        "recent_trades": [],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / STATE_FILE
    payload = json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=STATE_FILE + ".", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _trade_dict(trade: Any) -> dict[str, Any]:
    return trade.to_dict() if hasattr(trade, "to_dict") else dict(trade)


def _summarize(results_by_symbol: list[tuple[str, Any]], candidate: str,
               leverage: float, scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    closed: list[dict[str, Any]] = []
    open_marks: list[dict[str, Any]] = []
    returns: list[float] = []
    drawdowns: list[float] = []
    final_equities: list[float] = []
    liquidations = 0

    for symbol, result in results_by_symbol:
        final_equity = float(result.final_equity)
        if not math.isfinite(final_equity) or final_equity <= 0:
            raise ValueError(f"{symbol} shadow equity failed validation")
        final_equities.append(final_equity)
        returns.append((final_equity / INITIAL_EQUITY_PER_SYMBOL - 1.0) * 100.0)
        metrics = result.metrics()
        drawdowns.append(float(metrics["max_drawdown_pct"]))
        liquidations += int(metrics["liquidations"])
        for raw in result.trades:
            trade = _trade_dict(raw)
            row = {
                "candidate": candidate,
                "leverage": leverage,
                "scenario": scenario,
                "symbol": symbol,
                **trade,
            }
            if trade.get("exit_reason") == "end_of_window":
                # The engine estimates the cost of flattening at the latest close.
                # Keep this separate from realized closed trades.
                row["is_open_mark_estimate"] = True
                open_marks.append(row)
            else:
                row["is_open_mark_estimate"] = False
                closed.append(row)

    pnls = [float(row["net_pnl"]) for row in closed]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    gross_loss = abs(sum(losses))
    profit_factor: float | str | None = (
        sum(wins) / gross_loss if gross_loss > 0 else ("infinite" if wins else None)
    )
    row = {
        "candidate": candidate,
        "leverage": leverage,
        "scenario": scenario,
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(pnls) * 100.0 if pnls else None,
        "profit_factor": profit_factor,
        "realized_net_pnl_usd": sum(pnls),
        "estimated_open_pnl_after_exit_costs_usd": sum(
            float(item["net_pnl"]) for item in open_marks
        ),
        "estimated_equal_weight_return_pct": (
            sum(returns) / len(returns) if returns else None
        ),
        "estimated_mean_equity_usd": (
            sum(final_equities) / len(final_equities) if final_equities else None
        ),
        "max_symbol_drawdown_pct": max(drawdowns) if drawdowns else None,
        "open_positions": len(open_marks),
        "liquidations": liquidations,
        "symbols_observed": len(results_by_symbol),
        "orders_submitted": False,
        "demo_ready": False,
    }
    return row, closed + open_marks


def run_shadow_cycle(
    root: str | Path,
    *,
    now_ms: int | None = None,
    client_factory=None,
    history_fetcher=None,
    backtest_runner=None,
) -> dict[str, Any]:
    """Fetch fresh public data and recompute all frozen candidates from test start."""
    current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    root_path = Path(root)
    state = _load_state(root_path, current_ms)
    state["last_checked_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_ms / 1000)
    )
    # Persist the immutable forward boundary before any network request. A Bybit
    # outage or process restart must never move the start of the untouched sample.
    _save_state(root_path, state)

    factory = client_factory or BybitLinearPublicClient
    fetch_history = history_fetcher or _fetch_history
    run = backtest_runner or run_backtest
    client = factory()
    history_start_ms = int(state["history_start_ms"])
    history = {
        symbol: fetch_history(client, symbol, history_start_ms, current_ms)
        for symbol in state["symbols"]
    }

    common = None
    for _, traded, mark, _funding in history.values():
        if not traded.index.equals(mark.index):
            raise ValueError("traded and mark-price candles are not aligned")
        common = traded.index if common is None else common.intersection(traded.index)
    if common is None or len(common) < WARMUP_BARS:
        state["status"] = "waiting_for_market_history"
        state["detail"] = "Waiting for enough aligned completed 4h candles for indicator warm-up."
        _save_state(root_path, state)
        return state

    common = common.sort_values()
    if not common.is_unique or not common.is_monotonic_increasing:
        raise ValueError("common shadow candle timestamps are not unique and ordered")
    if not (common.to_series().diff().dropna() == pd.Timedelta(hours=4)).all():
        raise ValueError("shadow market history contains missing 4-hour intervals")

    forward_start = pd.to_datetime(int(state["forward_start_ms"]), unit="ms", utc=True)
    forward_index = int(common.searchsorted(forward_start))
    if forward_index >= len(common) or common[forward_index] != forward_start:
        if common[-1] < forward_start:
            state["status"] = "waiting_for_first_forward_candle"
            state["detail"] = "The forward window starts at the next 4h boundary; historical warm-up is not scored."
            _save_state(root_path, state)
            return state
        raise ValueError("the first forward candle is missing from the aligned market history")
    if forward_index >= len(common) - 1:
        state["status"] = "collecting_forward_candles"
        state["last_bar_time"] = common[-1].isoformat()
        state["detail"] = "Forward data has started; waiting for the next completed candle to evaluate next-open entries."
        _save_state(root_path, state)
        return state

    last_bar_ms = int(common[-1].value // 1_000_000)
    if (
        state.get("last_processed_bar_ms") == last_bar_ms
        and state.get("results")
    ):
        state["status"] = "running"
        state["last_bar_time"] = common[-1].isoformat()
        _save_state(root_path, state)
        return state

    summarized: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    for candidate_name, overrides in CANDIDATES.items():
        for leverage in LEVERAGES:
            for scenario_name, costs in COST_SCENARIOS.items():
                per_symbol = []
                for symbol, (instrument, traded, mark, funding) in history.items():
                    traded_window = traded.loc[common].copy()
                    mark_window = mark.loc[common].copy()
                    funding_window = funding.loc[funding.index.isin(common)].copy()
                    strategy = PerpetualEMAStrategy(
                        enabled=True,
                        risk=PerpetualRiskSettings(leverage=leverage, max_leverage=2.0),
                        **overrides,
                    )
                    result = run(
                        traded_window,
                        mark_window,
                        funding_window,
                        instrument,
                        initial_equity=INITIAL_EQUITY_PER_SYMBOL,
                        strategy=strategy,
                        costs=costs,
                        trade_start_index=forward_index,
                    )
                    per_symbol.append((symbol, result))
                summary, trades = _summarize(
                    per_symbol, candidate_name, leverage, scenario_name
                )
                summarized.append(summary)
                all_trades.extend(trades)

    all_trades.sort(key=lambda item: str(item.get("exit_time") or ""))
    state.update({
        "status": "running",
        "detail": (
            "Collecting untouched future 4h candles. All five candidates are tracked in parallel; "
            "no candidate is selected or approved for trading."
        ),
        "last_processed_bar_ms": last_bar_ms,
        "last_bar_time": common[-1].isoformat(),
        "last_updated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_ms / 1000)
        ),
        "forward_bars": len(common) - forward_index,
        "cycles": int(state.get("cycles", 0)) + 1,
        "results": summarized,
        "recent_trades": all_trades[-RECENT_TRADES_LIMIT:],
        "orders_submitted": False,
        "execution_enabled": False,
        "demo_ready": False,
    })
    _save_state(root_path, state)
    return state


def read_shadow_status(root: str | Path) -> dict[str, Any]:
    path = Path(root) / STATE_FILE
    if not path.is_file():
        return {
            "status": "idle",
            "detail": "Shadow forward test has not been initialized.",
            "orders_submitted": False,
            "execution_enabled": False,
            "demo_ready": False,
            "results": [],
            "recent_trades": [],
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "status": "error",
            "error": "Saved shadow-forward state is unreadable.",
            "orders_submitted": False,
            "execution_enabled": False,
            "demo_ready": False,
            "results": [],
            "recent_trades": [],
        }
    if state.get("version") != STATE_VERSION:
        return {
            "status": "error",
            "error": "Saved shadow-forward state version is unsupported.",
            "orders_submitted": False,
            "execution_enabled": False,
            "demo_ready": False,
            "results": [],
            "recent_trades": [],
        }
    return state
