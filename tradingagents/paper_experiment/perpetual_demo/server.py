"""Phone-friendly, authenticated Bybit Demo linear-perpetual control panel.

The process is separate from the legacy Spot dashboard. Manual orders and
the multi-timeframe scalping worker can only use the fixed Bybit Demo linear
broker, and both require an explicit confirmation header.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tradingagents.crypto.bybit_linear_demo import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DEMO_BASE_URL,
    MAX_LEVERAGE,
    MAX_OPEN_POSITIONS,
    BybitLinearDemoAmbiguous,
    BybitLinearDemoBroker,
    BybitLinearDemoError,
    BybitLinearDemoNotConfigured,
    BybitLinearDemoRejected,
)
from tradingagents.crypto.bybit_perpetual_account import BybitPerpetualAccountError
from tradingagents.crypto.scalping import ScalpingConfig, evaluate_scalping_signal
from tradingagents.paper_experiment.dashboard.auth import (
    check_bearer,
    require_bindable,
    resolve_token,
)

DEFAULT_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
    "LTCUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT",
)
STRATEGY_ID = "bollinger_rsi_quality20_atr16_di16"
STATE_FILE = "bybit_linear_demo_state.json"
POLL_SECONDS = 15
CANDLE_MS = 4 * 60 * 60 * 1000
MAX_REQUEST_BYTES = 16_384


def _number(value: object, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'valid'}")
    return number


def _safe_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return dict(default)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return value if isinstance(value, dict) else dict(default)


class PerpetualDemoController:
    """Live account view, bounded manual execution and a multi-timeframe scalper."""

    def __init__(
        self,
        root: str | Path,
        *,
        broker: BybitLinearDemoBroker | None = None,
        public_client=None,
        poll_seconds: int = POLL_SECONDS,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    ) -> None:
        if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, int) or poll_seconds < 5:
            raise ValueError("poll_seconds must be an integer of at least five seconds")
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be a non-empty unique sequence")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / STATE_FILE
        self.broker = broker or BybitLinearDemoBroker()
        self.public = public_client or self.broker.public
        self.poll_seconds = poll_seconds
        self.symbols = tuple(str(s).strip().upper() for s in symbols)
        if any(not s.isalnum() or not 3 <= len(s) <= 30 for s in self.symbols):
            raise ValueError("invalid strategy symbol")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._strategy_status: dict[str, Any] = {
            "status": "idle",
            "strategy_id": STRATEGY_ID,
            "orders_submitted": False,
            "execution_enabled": False,
            "detail": "Strategy is stopped. No orders are submitted automatically.",
        }
        self.strategy = ScalpingConfig()
        self._state = _safe_json(
            self.state_path,
            {
                "schema_version": 1,
                "last_processed_candles": {},
                "strategy_positions": {},
                "risk": {},
                "events": [],
            },
        )

    def _save_state(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._state, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _record(self, event: dict) -> None:
        events = self._state.setdefault("events", [])
        events.append({"at": datetime.now(UTC).isoformat(), **event})
        self._state["events"] = events[-100:]
        self._save_state()

    def _require_configured(self) -> None:
        if not self.broker.configured:
            raise BybitLinearDemoNotConfigured(
                f"Add a dedicated Bybit Demo Contract Trade key using {API_KEY_ENV} and {API_SECRET_ENV}."
            )

    def status(self) -> dict[str, Any]:
        if not self.broker.configured:
            return {
                "status": "not_configured",
                "venue": "Bybit Demo",
                "endpoint": DEMO_BASE_URL,
                "category": "linear",
                "spot_enabled": False,
                "credentials_required": [API_KEY_ENV, API_SECRET_ENV],
                "wallet": None,
                "positions": [],
                "open_orders": [],
                "strategy": dict(self._strategy_status),
                "detail": "A dedicated Demo key with Account Read and Contract Trade permissions is required.",
            }
        try:
            account = self.broker.account_state()
            return {
                **account,
                "strategy": dict(self._strategy_status),
                "strategy_symbols": list(self.symbols),
                "strategy_state": {
                    "last_processed_candles": dict(self._state.get("last_processed_candles", {})),
                    "events": list(self._state.get("events", [])[-20:]),
                    "risk": dict(self._state.get("risk", {})),
                },
                "refreshed_at": datetime.now(UTC).isoformat(),
            }
        except BybitLinearDemoNotConfigured as exc:
            return {
                "status": "not_configured",
                "venue": "Bybit Demo",
                "endpoint": DEMO_BASE_URL,
                "category": "linear",
                "spot_enabled": False,
                "credentials_required": [API_KEY_ENV, API_SECRET_ENV],
                "wallet": None,
                "positions": [],
                "open_orders": [],
                "strategy": dict(self._strategy_status),
                "detail": str(exc),
            }
        except Exception as exc:
            return {
                "status": "error",
                "venue": "Bybit Demo",
                "endpoint": DEMO_BASE_URL,
                "category": "linear",
                "spot_enabled": False,
                "wallet": None,
                "positions": [],
                "open_orders": [],
                "strategy": dict(self._strategy_status),
                "error_type": type(exc).__name__,
                "detail": (
                    str(exc)[:220]
                    if isinstance(exc, (BybitLinearDemoError, BybitPerpetualAccountError))
                    else f"Account read failed safely ({type(exc).__name__})."
                ),
            }

    def _scalping_market(
        self,
        symbol: str,
        now_ms: int | None = None,
        *,
        include_instrument: bool = True,
    ) -> dict:
        market = str(symbol or "").strip().upper()
        if market not in self.symbols:
            raise ValueError("symbol is outside the configured Demo perpetual universe")
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        one_minute = self.public.klines(
            market, start_ms=now - 120 * 60_000, end_ms=now,
            include_current=True, interval="1m",
        )
        five_minute = self.public.klines(
            market, start_ms=now - 100 * 5 * 60_000, end_ms=now,
            include_current=False, interval="5m",
        )
        fifteen_minute = self.public.klines(
            market, start_ms=now - 100 * 15 * 60_000, end_ms=now,
            include_current=False, interval="15m",
        )
        if one_minute is None or len(one_minute) < 40:
            raise BybitLinearDemoError("not enough 1-minute candles for scalping indicators")
        if five_minute is None or len(five_minute) < 55:
            raise BybitLinearDemoError("not enough completed 5-minute candles for trend filter")
        if fifteen_minute is None or len(fifteen_minute) < 55:
            raise BybitLinearDemoError("not enough completed 15-minute candles for trend filter")
        forming_open_ms = int(one_minute.index[-1].timestamp() * 1000)
        if not forming_open_ms <= now < forming_open_ms + 60_000:
            raise BybitLinearDemoError("live forming 1-minute candle is unavailable; refusing stale signals")
        mark = self.broker.mark_price(market)
        current_time = one_minute.index[-1]
        one_minute.at[current_time, "close"] = mark
        one_minute.at[current_time, "high"] = max(float(one_minute.at[current_time, "high"]), mark)
        one_minute.at[current_time, "low"] = min(float(one_minute.at[current_time, "low"]), mark)
        signal = evaluate_scalping_signal(one_minute, five_minute, fifteen_minute, mark, self.strategy)
        atr = signal.get("atr")
        long_stop = long_target = short_stop = short_target = None
        if isinstance(atr, (int, float)) and math.isfinite(float(atr)) and float(atr) > 0:
            stop_distance = max(
                float(atr) * self.strategy.stop_atr_multiple,
                mark * self.strategy.min_stop_fraction,
            )
            if stop_distance / mark <= self.strategy.max_stop_fraction:
                target_distance = stop_distance * self.strategy.reward_risk
                long_stop, long_target = mark - stop_distance, mark + target_distance
                short_stop, short_target = mark + stop_distance, mark - target_distance
        instrument = self.broker.instrument_info(market) if include_instrument else None
        return {
            "status": "ok", "venue": "Bybit Demo", "category": "linear",
            "symbol": market, "timeframe": "1m", "trend_timeframes": ["5m", "15m"],
            "mark_price": mark, "forming_candle_open": current_time.isoformat(),
            "signal_basis": "live_mark_1m_reclaim_with_completed_5m_15m_trend",
            "atr": float(atr) if isinstance(atr, (int, float)) and math.isfinite(float(atr)) else None,
            "rsi": signal.get("rsi"), "ema9": signal.get("ema9"),
            "signal": int(signal.get("signal", 0)), "signal_reason": signal.get("reason"),
            "long_bias": bool(signal.get("long_bias", False)),
            "short_bias": bool(signal.get("short_bias", False)),
            "suggested_long_stop": long_stop, "suggested_long_target": long_target,
            "suggested_short_stop": short_stop, "suggested_short_target": short_target,
            "quantity_step": instrument.quantity_step if instrument else None,
            "min_quantity": instrument.min_quantity if instrument else None,
            "min_notional": instrument.min_notional if instrument else None,
            "max_leverage": min(MAX_LEVERAGE, instrument.max_leverage) if instrument else MAX_LEVERAGE,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def market(self, symbol: str) -> dict:
        return self._scalping_market(symbol)

    def open_manual(
        self,
        *,
        symbol: str,
        side: str,
        leverage: float,
        stop_loss: float,
        take_profit: float,
    ) -> dict:
        self._require_configured()
        market = str(symbol or "").strip().upper()
        if market not in self.symbols:
            raise ValueError("symbol is outside the configured Demo perpetual universe")
        direction = str(side or "").strip().lower()
        if direction not in {"long", "short"}:
            raise ValueError("side must be long or short")
        lev = _number(leverage, "leverage")
        if lev not in (1.0, 2.0):
            raise ValueError("only 1x or 2x leverage is allowed")
        wallet = self.broker.get_wallet_balance()
        if wallet.get("status") != "ok":
            raise BybitLinearDemoError("Bybit Demo wallet did not pass validation")
        equity = _number(wallet.get("total_equity_usd"), "account equity")
        order_link_id = f"MNL-{market}-{uuid.uuid4().hex[:18]}"
        result = self.broker.open_market_position(
            symbol=market,
            side=direction,
            leverage=lev,
            equity=equity,
            stop_loss=_number(stop_loss, "stop_loss"),
            take_profit=_number(take_profit, "take_profit"),
            order_link_id=order_link_id,
        )
        self._record(
            {
                "type": "manual_entry",
                "symbol": market,
                "side": direction,
                "leverage": lev,
                "order_link_id": order_link_id,
                "venue_order_id": result.get("order_id"),
            }
        )
        return result

    def close_manual(self, symbol: str) -> dict:
        self._require_configured()
        market = str(symbol or "").strip().upper()
        if market not in self.symbols:
            raise ValueError("symbol is outside the configured Demo perpetual universe")
        result = self.broker.close_market_position(
            market,
            order_link_id=f"CLOSE-{market}-{uuid.uuid4().hex[:16]}",
        )
        self._record(
            {
                "type": "manual_close",
                "symbol": market,
                "order_link_id": result.get("order_link_id"),
                "venue_order_id": result.get("order_id"),
            }
        )
        return result

    def start_strategy(self) -> dict:
        self._require_configured()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return dict(self._strategy_status)
            risk_state = self._state.setdefault("risk", {})
            if risk_state.get("halted"):
                raise BybitLinearDemoRejected(
                    "strategy safety halt is latched; inspect the risk event before resetting state"
                )
            wallet = self.broker.get_wallet_balance()
            if wallet.get("status") != "ok":
                raise BybitLinearDemoError("Bybit Demo wallet preflight failed")
            _number(wallet.get("total_equity_usd"), "account equity")
            # Resolve current exchange contract filters before the worker can be armed.
            for symbol in self.symbols:
                self.broker.instrument_info(symbol)
            self.broker.get_positions()  # also refuses hedge-mode positions
            self._stop.clear()
            self._strategy_status = {
                "status": "starting",
                "strategy_id": STRATEGY_ID,
                "name": "EMA Multi-Timeframe Scalper · 1m / 5m / 15m",
                "timeframe": "1m_live_mark_5m_15m_filter",
                "symbols": list(self.symbols),
                "leverage": 2.0,
                "max_leverage": MAX_LEVERAGE,
                "max_open_positions": MAX_OPEN_POSITIONS,
                "orders_submitted": False,
                "execution_enabled": True,
                "started_at": datetime.now(UTC).isoformat(),
                "detail": f"Preflight passed; scanning 12 USDT perpetuals on live 1-minute Mark Price with completed 5m/15m trend filters every {self.poll_seconds}s.",
            }
            self._thread = threading.Thread(
                target=self._strategy_loop,
                name="bybit-demo-scalper-linear",
                daemon=True,
            )
            self._thread.start()
            return dict(self._strategy_status)

    def stop_strategy(self) -> dict:
        self._stop.set()
        with self._lock:
            self._strategy_status = {
                **self._strategy_status,
                "status": "stopping",
                "execution_enabled": False,
                "detail": "New entries are disabled. Existing Demo positions remain protected by exchange-side SL/TP.",
            }
        return dict(self._strategy_status)

    def _strategy_loop(self) -> None:
        self._strategy_status = {
            **self._strategy_status,
            "status": "running",
            "detail": f"EMA MTF scalper is checking live 1-minute Mark Price with completed 5m/15m trend filters every {self.poll_seconds}s; max hold {self.strategy.max_hold_minutes}m, 2x max leverage, 0.5% modeled risk.",
        }
        while not self._stop.is_set():
            try:
                self._strategy_cycle()
                self._strategy_status = {
                    **self._strategy_status,
                    "status": "running",
                    "last_cycle_at": datetime.now(UTC).isoformat(),
                    "last_error": None,
                    "orders_submitted": bool(self._strategy_status.get("orders_submitted")),
                }
            except Exception as exc:
                # Do not log request bodies, keys, secrets or raw exchange payloads.
                self._strategy_status = {
                    **self._strategy_status,
                    "status": "error",
                    "last_error": f"{type(exc).__name__}: {str(exc)[:180]}",
                    "last_cycle_at": datetime.now(UTC).isoformat(),
                }
            self._stop.wait(self.poll_seconds)
        self._strategy_status = {
            **self._strategy_status,
            "status": "stopped",
            "execution_enabled": False,
            "detail": "Strategy stopped. Existing positions are not force-closed.",
            "stopped_at": datetime.now(UTC).isoformat(),
        }

    def _risk_guard(self, equity: float, positions: list[dict]) -> bool:
        risk = self._state.setdefault("risk", {})
        today = datetime.now(UTC).date().isoformat()
        if risk.get("day") != today:
            risk["day"] = today
            risk["day_start_equity"] = equity
        day_start = _number(risk.get("day_start_equity", equity), "daily starting equity")
        peak = max(_number(risk.get("equity_peak", equity), "equity peak"), equity)
        risk["equity_peak"] = peak
        daily_loss = max(0.0, (day_start - equity) / day_start)
        drawdown = max(0.0, (peak - equity) / peak)
        if daily_loss >= 0.03 or drawdown >= 0.10:
            risk["halted"] = True
            risk["halt_reason"] = "daily_loss_limit" if daily_loss >= 0.03 else "max_drawdown_limit"
            risk["halted_at"] = datetime.now(UTC).isoformat()
            for position in positions:
                with contextlib.suppress(Exception):
                    self.broker.close_market_position(
                        position["symbol"],
                        order_link_id=f"KILL-{position['symbol']}-{uuid.uuid4().hex[:12]}",
                    )
            self._record(
                {
                    "type": "risk_halt",
                    "reason": risk["halt_reason"],
                    "daily_loss_fraction": daily_loss,
                    "drawdown_fraction": drawdown,
                }
            )
            self._strategy_status = {
                **self._strategy_status,
                "status": "halted",
                "execution_enabled": False,
                "detail": "Risk kill-switch activated. No new entries will be submitted.",
                "risk": dict(risk),
            }
            self._stop.set()
            return True
        self._save_state()
        return False

    def _strategy_cycle(self) -> None:
        wallet = self.broker.get_wallet_balance()
        if wallet.get("status") != "ok":
            raise BybitLinearDemoError("wallet balance could not be verified")
        equity = _number(wallet.get("total_equity_usd"), "account equity")
        positions = self.broker.get_positions()
        if self._risk_guard(equity, positions):
            return
        if len(positions) > MAX_OPEN_POSITIONS:
            raise BybitLinearDemoRejected("Demo account has more than two open positions; entries halted")

        active_by_symbol = {p["symbol"]: p for p in positions}
        managed = self._state.setdefault("scalper_positions", {})
        attempts = self._state.setdefault("scalper_entry_attempts", {})
        cooldowns = self._state.setdefault("scalper_cooldowns", {})
        scan_errors = self._state.setdefault("scalper_scan_errors", {})
        scanned = signals_seen = 0
        now_ms = int(time.time() * 1000)

        for symbol in self.symbols:
            if self._stop.is_set():
                break
            try:
                market = self._scalping_market(symbol, now_ms, include_instrument=False)
                scanned += 1
                scan_errors.pop(symbol, None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:120]}"
                if scan_errors.get(symbol) != error:
                    scan_errors[symbol] = error
                    self._record({"type": "strategy_scan_error", "symbol": symbol, "reason": error})
                continue

            current_minute = market["forming_candle_open"]
            current_signal = int(market.get("signal", 0))
            if current_signal in (-1, 1):
                signals_seen += 1
            position = active_by_symbol.get(symbol)
            meta = managed.get(symbol)

            if position is not None:
                # Never adopt or alter positions opened manually in the Demo desk.
                if not isinstance(meta, dict):
                    continue
                try:
                    entered_ms = int(meta.get("entry_time_ms") or position.get("created_time_ms") or now_ms)
                except (TypeError, ValueError):
                    entered_ms = now_ms
                held_ms = max(0, now_ms - entered_ms)
                opposite_signal = (
                    current_signal == -1 and position["side"] == "long"
                ) or (
                    current_signal == 1 and position["side"] == "short"
                )
                exit_reason = None
                if held_ms >= self.strategy.max_hold_minutes * 60_000:
                    exit_reason = "max_holding_time"
                elif opposite_signal and held_ms >= 60_000:
                    exit_reason = "opposite_scalping_signal"
                if exit_reason:
                    close_result = self.broker.close_market_position(
                        symbol, order_link_id=f"SC-X-{symbol}-{now_ms}"[-36:]
                    )
                    managed.pop(symbol, None)
                    active_by_symbol.pop(symbol, None)
                    cooldowns[symbol] = now_ms + self.strategy.cooldown_minutes * 60_000
                    self._record({
                        "type": "strategy_exit", "symbol": symbol, "side": position["side"],
                        "reason": exit_reason, "held_seconds": held_ms // 1000,
                        "order_link_id": close_result.get("order_link_id"),
                    })
                continue

            # The exchange may already have triggered the attached SL or TP.
            # Record disappearance without guessing which protection was hit.
            if isinstance(meta, dict):
                managed.pop(symbol, None)
                cooldowns[symbol] = now_ms + self.strategy.cooldown_minutes * 60_000
                self._record({
                    "type": "strategy_exit_observed", "symbol": symbol,
                    "side": meta.get("side"), "reason": "position_no_longer_open_at_venue",
                })
                continue

            if cooldowns.get(symbol, 0) > now_ms or attempts.get(symbol) == current_minute:
                continue
            if current_signal not in (-1, 1) or len(active_by_symbol) >= MAX_OPEN_POSITIONS:
                continue

            side = "long" if current_signal > 0 else "short"
            stop = float(market["suggested_long_stop"] if side == "long" else market["suggested_short_stop"])
            target = float(market["suggested_long_target"] if side == "long" else market["suggested_short_target"])
            minute_ms = int(datetime.fromisoformat(current_minute).timestamp() * 1000)
            order_link_id = f"SC-{symbol}-{minute_ms}"[-36:]
            # Persist before the network write to prevent blind resubmission on
            # a timeout or process restart during the same 1-minute candle.
            attempts[symbol] = current_minute
            self._save_state()
            try:
                result = self.broker.open_market_position(
                    symbol=symbol, side=side, leverage=self.strategy.leverage,
                    equity=equity, stop_loss=stop, take_profit=target,
                    order_link_id=order_link_id,
                )
            except Exception as exc:
                self._record({
                    "type": "strategy_entry_rejected", "symbol": symbol, "side": side,
                    "reason": f"{type(exc).__name__}: {str(exc)[:140]}",
                    "order_link_id": order_link_id,
                })
                continue

            managed[symbol] = {
                "side": side, "entry_time_ms": now_ms,
                "entry_signal_candle": current_minute,
                "entry_price_mark": market["mark_price"],
                "stop_loss": stop, "take_profit": target,
                "order_link_id": order_link_id,
            }
            active_by_symbol[symbol] = {
                "symbol": symbol, "side": side, "quantity": result["quantity"],
                "entry_price": market["mark_price"], "leverage": self.strategy.leverage,
                "created_time_ms": now_ms,
            }
            self._record({
                "type": "strategy_entry", "symbol": symbol, "side": side,
                "leverage": self.strategy.leverage, "order_link_id": order_link_id,
                "venue_order_id": result.get("order_id"),
                "quantity": result.get("quantity"), "notional_usdt": result.get("notional_usdt"),
                "stop_loss": stop, "take_profit": target,
            })
            self._strategy_status = {
                **self._strategy_status, "orders_submitted": True,
                "last_order_at": datetime.now(UTC).isoformat(),
                "last_order": {
                    "symbol": symbol, "side": side, "order_link_id": order_link_id,
                    "venue_order_id": result.get("order_id"),
                },
            }

        self._state["scalper_scan_errors"] = scan_errors
        self._state["last_scalper_scan"] = {
            "at": datetime.now(UTC).isoformat(),
            "symbols_scanned": scanned,
            "signals_seen": signals_seen,
        }
        self._save_state()
        self._strategy_status = {
            **self._strategy_status,
            "last_cycle_at": datetime.now(UTC).isoformat(),
            "last_scan_symbols": scanned,
            "last_signals_seen": signals_seen,
            "scan_errors": len(scan_errors),
        }


def pd_timestamp(value: object):
    """Parse a persisted timestamp without pulling pandas into controller state."""
    if not isinstance(value, str) or not value:
        return None
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    except ValueError:
        return None


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#160b11">
<title>Perpetual Demo Desk</title>
<style>
:root{color-scheme:dark;--bg:#10080d;--panel:#1b1017;--panel2:#24151e;--line:#432638;--ink:#f7edf2;--muted:#b99eae;--pink:#d58aa6;--green:#6cdbad;--red:#ff8198;--gold:#efc47c}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#321a2a 0,#10080d 50%);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{width:min(1120px,100%);margin:auto;padding:20px 14px 50px}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}.brand{font-size:19px;font-weight:800;letter-spacing:.03em}.sub{color:var(--muted);font-size:12px}.pill{border:1px solid var(--line);border-radius:99px;padding:5px 10px;font-size:11px;letter-spacing:.08em;color:var(--muted)}.pill.safe{color:var(--green);border-color:#32634f}.pill.warn{color:var(--gold);border-color:#71562d}.pill.bad{color:var(--red);border-color:#723347}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}.metric,.panel{background:linear-gradient(150deg,rgba(36,21,30,.96),rgba(23,13,19,.96));border:1px solid var(--line);border-radius:15px}.metric{padding:13px;min-width:0}.metric label{display:block;color:var(--muted);font-size:11px}.metric b{display:block;font-size:20px;margin-top:5px;overflow-wrap:anywhere}.metric small{color:var(--muted);font-size:10px}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{padding:16px;margin-bottom:12px;min-width:0}.wide{grid-column:1/-1}.heading{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0 0 12px}.heading h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;margin:0}.muted{color:var(--muted)}.note{padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#160d13;color:var(--muted);font-size:12px;margin:10px 0}
button,input,select{font:inherit;border-radius:9px;border:1px solid var(--line);background:#120a10;color:var(--ink);padding:11px 12px;min-width:0}button{cursor:pointer;font-weight:700}button:disabled{opacity:.45;cursor:not-allowed}.actions{display:flex;gap:8px;flex-wrap:wrap}.primary{background:#71364f;border-color:#a95b7c}.long{background:#174936;border-color:#33775d}.short{background:#592338;border-color:#9f4361}.danger{background:#49202d;border-color:#83374d}.secondary{background:#2d1b27}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.field{display:flex;flex-direction:column;gap:5px}.field label{font-size:11px;color:var(--muted)}.field input,.field select{width:100%}
.rows{display:grid;gap:8px}.rowitem{border:1px solid var(--line);background:#160d13;padding:12px;border-radius:10px}.rowtop{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.rowitem strong{font-size:13px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}.stat{font-size:11px;color:var(--muted)}.stat b{display:block;color:var(--ink);font-size:12px}.empty{border:1px dashed var(--line);padding:18px;border-radius:10px;color:var(--muted);text-align:center}
#auth{max-width:440px;margin:12vh auto;padding:20px}#auth input{width:100%;margin:10px 0}#message{display:none;padding:12px;border-radius:10px;margin:12px 0;border:1px solid var(--line);white-space:pre-wrap}#message.error{display:block;color:var(--red);border-color:#753149}#message.ok{display:block;color:var(--green);border-color:#32634f}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:9px 7px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:10px;text-transform:uppercase} .tablewrap{overflow-x:auto}.positive{color:var(--green)}.negative{color:var(--red)}
@media(max-width:760px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.layout{grid-template-columns:1fr}.wide{grid-column:auto}.metric b{font-size:17px}main{padding:14px 10px 36px}.panel{padding:13px}.stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div id="auth" class="panel">
 <div class="brand">PERPETUAL DEMO DESK</div><p class="muted">Enter the dashboard access token configured in Railway.</p>
 <input id="token" type="password" autocomplete="current-password" placeholder="Dashboard token">
 <button class="primary" id="connect">Connect dashboard</button>
</div>
<main id="app" style="display:none">
 <div class="top">
  <div><div class="brand">PERPETUAL DEMO DESK</div><div class="sub">Bybit · USDT Linear Perpetual · 1m scalping / 5m + 15m trend</div></div>
  <div class="actions"><span id="venue-badge" class="pill warn">CHECKING</span><span id="refresh-time" class="pill">—</span><button id="refresh" class="secondary">Refresh</button></div>
 </div>
 <div id="message"></div>
 <div class="note"><b>DEMO ONLY:</b> this service is hard-locked to api-demo.bybit.com and category=linear. Spot and live trading are not available here. Max leverage 2x · max 2 open positions · 0.5% modeled risk per entry.</div>
 <section class="grid">
  <div class="metric"><label>Account equity</label><b id="equity">—</b><small>USDT · Bybit Unified</small></div>
  <div class="metric"><label>Available balance</label><b id="available">—</b><small>USDT</small></div>
  <div class="metric"><label>Unrealized P&amp;L</label><b id="upl">—</b><small>Linear positions</small></div>
  <div class="metric"><label>Open positions</label><b id="position-count">—</b><small>Limit 2</small></div>
 </section>
 <div class="layout">
  <section class="panel">
   <div class="heading"><h2>Scalping strategy</h2><span id="strategy-badge" class="pill">IDLE</span></div>
   <div class="rowitem">
    <strong>EMA Multi-Timeframe Scalper</strong>
    <div class="sub">1m live reclaim · 5m momentum · 15m direction · max 15m hold · 12 USDT perpetual pairs</div>
    <div class="stats">
     <div class="stat">Leverage<b>2x max</b></div><div class="stat">Risk per entry<b>0.5% equity</b></div>
     <div class="stat">Daily stop<b>3%</b></div><div class="stat">Drawdown stop<b>10%</b></div>
    </div>
   </div>
   <p id="strategy-detail" class="muted">Stopped. No automatic orders.</p>
   <div class="actions"><button id="start-strategy" class="primary">Start Scalping Auto</button><button id="stop-strategy" class="danger" disabled>Stop strategy</button></div>
   <div class="note">The scalper checks live Mark Price every 15 seconds, using completed 5m/15m trend filters and a 1m pullback-reclaim trigger. Each entry gets exchange-side Mark Price SL/TP; strategy exits on an opposite signal or after 15 minutes. 2x leverage and existing account risk limits remain enforced. Demo only; no profit is guaranteed.</div>
  </section>
  <section class="panel">
   <div class="heading"><h2>Manual Demo order</h2><span class="pill">BUY / SELL</span></div>
   <div class="fields">
    <div class="field"><label>Contract</label><select id="symbol"></select></div>
    <div class="field"><label>Direction</label><select id="side"><option value="long">Long · Buy</option><option value="short">Short · Sell</option></select></div>
    <div class="field"><label>Leverage</label><select id="leverage"><option value="1">1x</option><option value="2" selected>2x</option></select></div>
    <div class="field"><label>Mark price</label><input id="mark" readonly value="—"></div>
    <div class="field"><label>Stop loss</label><input id="stop" inputmode="decimal" placeholder="Required"></div>
    <div class="field"><label>Take profit</label><input id="target" inputmode="decimal" placeholder="Required"></div>
   </div>
   <p id="market-note" class="sub">Loading 1m scalping indicators and risk levels…</p>
   <div class="actions"><button id="open-long" class="long">Open Long</button><button id="open-short" class="short">Open Short</button></div>
   <div class="note">Quantity is calculated server-side from verified Demo equity, the exchange's live lot filters, the stop distance, and the 0.5% risk budget. Raw quantity is not accepted from the browser.</div>
  </section>
  <section class="panel wide">
   <div class="heading"><h2>Open linear positions</h2><span id="positions-badge" class="pill">—</span></div>
   <div id="positions" class="rows"><div class="empty">Waiting for authenticated Bybit Demo data.</div></div>
  </section>
  <section class="panel wide">
   <div class="heading"><h2>Open orders</h2><span class="pill">Venue state</span></div>
   <div id="orders" class="tablewrap"><div class="empty">No verified open orders.</div></div>
  </section>
  <section class="panel wide">
   <div class="heading"><h2>Execution activity</h2><span class="pill">Last 20 events</span></div>
   <div id="events" class="rows"><div class="empty">No recorded activity.</div></div>
  </section>
 </div>
</main>
<script>
(function(){
"use strict";
var $=function(id){return document.getElementById(id)},TOKEN=localStorage.getItem("bybit_linear_demo_token")||"",symbols=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","DOTUSDT","TRXUSDT"],statusTimer=null,marketTimer=null;
symbols.forEach(function(s){var o=document.createElement("option");o.value=s;o.textContent=s;$("symbol").appendChild(o);});
function esc(v){return String(v===null||v===undefined?"—":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];});}
function fmt(v,d){var n=Number(v);return Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d===undefined?3:d,minimumFractionDigits:0}):"—";}
function msg(t,ok){$("message").textContent=t;$("message").className=ok?"ok":"error";}
async function api(path,options){options=options||{};options.headers=Object.assign({"Authorization":"Bearer "+TOKEN,"Content-Type":"application/json"},options.headers||{});var r=await fetch(path,options),body=await r.json().catch(function(){return {};});if(!r.ok)throw new Error(body.detail||body.error||"HTTP "+r.status);return body;}
function connect(){TOKEN=$("token").value.trim();if(!TOKEN){msg("Dashboard token is required.",false);return;}localStorage.setItem("bybit_linear_demo_token",TOKEN);boot();}
async function refresh(){try{var d=await api("/api/status");$("venue-badge").textContent=d.status==="ok"?"BYBIT DEMO · LINEAR":String(d.status||"UNKNOWN").toUpperCase();$("venue-badge").className="pill "+(d.status==="ok"?"safe":"warn");var w=d.wallet||{};$("equity").textContent=fmt(w.total_equity_usd);$("available").textContent=fmt(w.total_available_balance_usd);$("upl").textContent=fmt(w.total_perpetual_unrealized_pnl_usd);$("position-count").textContent=String((d.positions||[]).length);$("positions-badge").textContent=String((d.positions||[]).length)+" / 2";$("refresh-time").textContent=new Date().toLocaleTimeString();
renderPositions(d.positions||[]);renderOrders(d.open_orders||[]);renderEvents((d.strategy_state||{}).events||[]);renderStrategy(d.strategy||{});
if(d.status!=="ok")msg(d.detail||"Demo account is not ready. Check dedicated Contract Trade key permissions.",false);else if(d.spot_enabled!==false)msg("Safety error: service did not confirm Spot is disabled.",false);else $("message").style.display="none";
}catch(e){msg(e.message,false);}}
function renderStrategy(s){var state=s.status||"idle";$("strategy-badge").textContent=String(state).toUpperCase();$("strategy-badge").className="pill "+(state==="running"?"safe":state==="error"||state==="halted"?"bad":"");$("strategy-detail").textContent=s.detail||s.last_error||"Stopped.";var active=["starting","running","stopping"].indexOf(state)>=0;$("start-strategy").disabled=active||state==="halted";$("stop-strategy").disabled=!active;}
function renderPositions(rows){if(!rows.length){$("positions").innerHTML='<div class="empty">No open USDT linear perpetual positions.</div>';return;}$("positions").innerHTML=rows.map(function(p){var side=String(p.side||"").toUpperCase(),pnl=Number(p.unrealized_pnl||0);return '<div class="rowitem"><div class="rowtop"><strong>'+esc(p.symbol)+' · '+esc(side)+'</strong><span class="pill">'+fmt(p.leverage,1)+'x</span></div><div class="stats"><div class="stat">Quantity<b>'+fmt(p.quantity,6)+'</b></div><div class="stat">Entry<b>'+fmt(p.entry_price,5)+'</b></div><div class="stat">Mark<b>'+fmt(p.mark_price,5)+'</b></div><div class="stat">Unrealized P&amp;L<b class="'+(pnl>0?"positive":pnl<0?"negative":"")+'">'+fmt(pnl,4)+' USDT</b></div><div class="stat">Stop loss<b>'+fmt(p.stop_loss,5)+'</b></div><div class="stat">Take profit<b>'+fmt(p.take_profit,5)+'</b></div></div><div class="actions" style="margin-top:10px"><button class="danger" data-close="'+esc(p.symbol)+'">Close position · Reduce Only</button></div></div>';}).join("");}
function renderOrders(rows){if(!rows.length){$("orders").innerHTML='<div class="empty">No active exchange orders.</div>';return;}$("orders").innerHTML='<table><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Status</th><th>Order ID</th></tr></thead><tbody>'+rows.map(function(o){return '<tr><td>'+esc(o.symbol)+'</td><td>'+esc(o.side)+'</td><td>'+esc(o.qty)+'</td><td>'+esc(o.order_status)+'</td><td>'+esc(o.order_id).slice(0,12)+'</td></tr>';}).join("")+'</tbody></table>';}
function renderEvents(rows){if(!rows.length){$("events").innerHTML='<div class="empty">No strategy or manual execution events.</div>';return;}$("events").innerHTML=rows.slice().reverse().map(function(e){return '<div class="rowitem"><div class="rowtop"><strong>'+esc(e.type||"event")+' · '+esc(e.symbol||"—")+'</strong><span class="sub">'+esc(e.at||"")+'</span></div><div class="sub">'+esc(e.side||e.reason||"")+' '+(e.leverage?esc(e.leverage)+"x":"")+' '+(e.order_link_id?"· "+esc(e.order_link_id):"")+'</div></div>';}).join("");}
async function loadMarket(){var s=$("symbol").value;try{var d=await api("/api/market?symbol="+encodeURIComponent(s));$("mark").value=fmt(d.mark_price,6);var side=$("side").value;var stop=side==="long"?d.suggested_long_stop:d.suggested_short_stop;var target=side==="long"?d.suggested_long_target:d.suggested_short_target;$("stop").value=stop==null?"":String(stop);$("target").value=target==null?"":String(target);$("market-note").textContent="Live mark "+fmt(d.mark_price,6)+" · forming 1m candle "+new Date(d.forming_candle_open).toLocaleString()+" · ATR "+fmt(d.atr,6)+" · RSI "+fmt(d.rsi,1)+" · 5m/15m bias L:"+d.long_bias+" S:"+d.short_bias+" · signal "+d.signal+" ("+d.signal_reason+") · qty step "+d.quantity_step;}catch(e){$("market-note").textContent="Market / scalping data unavailable: "+e.message;}}
async function manual(side){$("side").value=side;var symbol=$("symbol").value,stop=Number($("stop").value),target=Number($("target").value),leverage=Number($("leverage").value);if(!Number.isFinite(stop)||!Number.isFinite(target)||stop<=0||target<=0){msg("Enter a valid stop loss and take profit first.",false);return;}if(!confirm("Submit a "+side.toUpperCase()+" market order on BYBIT DEMO "+symbol+" at up to "+leverage+"x? SL and TP will be attached at the exchange."))return;try{var d=await api("/api/order",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:JSON.stringify({symbol:symbol,side:side,leverage:leverage,stop_loss:stop,take_profit:target})});msg("Demo order accepted: "+(d.order_id||d.order_link_id)+" · "+fmt(d.quantity,6)+" "+symbol+". Confirm fill in the positions panel.",true);await refresh();}catch(e){msg(e.message,false);}}
async function closePosition(symbol){if(!confirm("Close the entire "+symbol+" position using a reduce-only market order on Bybit Demo?"))return;try{var d=await api("/api/close",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:JSON.stringify({symbol:symbol})});msg(d.closed===false?"No open position was found.":"Reduce-only close accepted for "+symbol+".",true);await refresh();}catch(e){msg(e.message,false);}}
async function startStrategy(){if(!confirm("Start automatic multi-timeframe scalping on Bybit Demo USDT Perpetual? Orders may be sent after a live 1m reclaim signal."))return;try{var s=await api("/api/strategy/start",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:"{}"});renderStrategy(s);msg("Scalping strategy start requested. Automatic orders remain limited to Bybit Demo.",true);}catch(e){msg(e.message,false);}await refresh();}
async function stopStrategy(){if(!confirm("Stop new scalping entries? Existing positions keep their exchange-side SL/TP protections."))return;try{var s=await api("/api/strategy/stop",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:"{}"});renderStrategy(s);msg("Scalping strategy stop requested. Existing positions remain open and protected.",true);}catch(e){msg(e.message,false);}await refresh();}
function boot(){$("auth").style.display="none";$("app").style.display="block";refresh();loadMarket();if(statusTimer)clearInterval(statusTimer);if(marketTimer)clearInterval(marketTimer);statusTimer=setInterval(refresh,15000);marketTimer=setInterval(loadMarket,60000);}
$("connect").addEventListener("click",connect);$("token").addEventListener("keydown",function(e){if(e.key==="Enter")connect();});$("refresh").addEventListener("click",refresh);$("symbol").addEventListener("change",loadMarket);$("side").addEventListener("change",loadMarket);$("open-long").addEventListener("click",function(){manual("long");});$("open-short").addEventListener("click",function(){manual("short");});$("start-strategy").addEventListener("click",startStrategy);$("stop-strategy").addEventListener("click",stopStrategy);$("positions").addEventListener("click",function(e){var b=e.target.closest("[data-close]");if(b)closePosition(b.getAttribute("data-close"));});if(TOKEN){$("token").value=TOKEN;boot();}else $("auth").style.display="block";
})();
</script>
</body>
</html>"""


def make_handler(controller: PerpetualDemoController, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TradingAgentsPerpetualDemo/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, code: int, payload: Any, content_type: str = "application/json; charset=utf-8") -> None:
            body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _auth(self) -> bool:
            if check_bearer(self.headers.get("Authorization"), token):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _body(self) -> dict:
            try:
                size = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ValueError("invalid content length") from None
            if size < 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            raw = self.rfile.read(size) if size else b"{}"
            try:
                value = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                raise ValueError("request body must be JSON") from None
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._send(200, {"status": "ok", "mode": "bybit-demo-linear"})
                return
            if path == "/":
                self._send(200, HTML, "text/html; charset=utf-8")
                return
            if not self._auth():
                return
            try:
                if path == "/api/status":
                    self._send(200, controller.status())
                elif path == "/api/market":
                    symbol = (parse_qs(urlsplit(self.path).query).get("symbol") or ["BTCUSDT"])[0]
                    self._send(200, controller.market(symbol))
                else:
                    self._send(404, {"error": "not found"})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except BybitLinearDemoNotConfigured as exc:
                self._send(503, {"error": str(exc)})
            except BybitLinearDemoError as exc:
                self._send(502, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: request failed safely"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._auth():
                return
            if (self.headers.get("X-Demo-Confirm") or "").strip().lower() != "yes":
                self._send(403, {"error": "explicit Demo order/control confirmation required"})
                return
            path = urlsplit(self.path).path
            try:
                body = self._body()
                if path == "/api/order":
                    result = controller.open_manual(
                        symbol=body.get("symbol"),
                        side=body.get("side"),
                        leverage=body.get("leverage"),
                        stop_loss=body.get("stop_loss"),
                        take_profit=body.get("take_profit"),
                    )
                elif path == "/api/close":
                    result = controller.close_manual(body.get("symbol"))
                elif path == "/api/strategy/start":
                    result = controller.start_strategy()
                elif path == "/api/strategy/stop":
                    result = controller.stop_strategy()
                else:
                    self._send(404, {"error": "not found"})
                    return
                self._send(202, result)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except BybitLinearDemoNotConfigured as exc:
                self._send(503, {"error": str(exc)})
            except BybitLinearDemoAmbiguous as exc:
                self._send(202, {
                    "status": "unknown_reconcile_required",
                    "order_link_id": exc.order_link_id,
                    "detail": str(exc),
                })
            except BybitLinearDemoRejected as exc:
                self._send(409, {"error": str(exc)})
            except BybitLinearDemoError as exc:
                self._send(502, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: request failed safely"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("TRADINGAGENTS_PERPETUAL_DEMO_ROOT", "/data")))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--token", default=None)
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("TRADINGAGENTS_PERPETUAL_DEMO_POLL_SECONDS", str(POLL_SECONDS))))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(args.port, bool) or not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    token = resolve_token(args.token)
    require_bindable(args.host, token)
    controller = PerpetualDemoController(args.root, poll_seconds=args.poll_seconds)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller, token))
    server.daemon_threads = True
    print(
        "Bybit Demo linear-perpetual dashboard listening; "
        f"endpoint={DEMO_BASE_URL} category=linear spot_enabled=false",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_strategy()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
