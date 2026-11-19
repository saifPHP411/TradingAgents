"""Phone-friendly, authenticated Bybit Demo linear-perpetual control panel.

The process is intentionally separate from the legacy Spot paper dashboard.
Manual orders and the DI16 strategy worker can only use the fixed Bybit Demo
linear broker, and both require an explicit confirmation header.
"""
from __future__ import annotations

import argparse
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

from tradingagents.crypto.backtest.perpetual_bollinger_forward import FORWARD_CANDIDATES
from tradingagents.crypto.backtest.perpetual_ema import PerpetualEMAStrategy, _features
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
from tradingagents.crypto.perpetuals import PerpetualRiskSettings
from tradingagents.paper_experiment.dashboard.auth import (
    DashboardAuthError,
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
POLL_SECONDS = 45
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
    """Live account view, bounded manual execution and a frozen DI16 worker."""

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
        candidate = dict(FORWARD_CANDIDATES[STRATEGY_ID])
        self.strategy = PerpetualEMAStrategy(
            enabled=True,
            risk=PerpetualRiskSettings(
                leverage=2.0,
                max_leverage=MAX_LEVERAGE,
                risk_fraction=0.005,
                max_margin_fraction=0.10,
                max_notional_fraction=0.20,
                max_daily_loss_fraction=0.03,
                max_drawdown_fraction=0.10,
                max_consecutive_losses=3,
            ),
            **candidate,
        )
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
                "detail": "Account data could not be verified. Balances and positions are not inferred or cached.",
            }

    def market(self, symbol: str) -> dict:
        market = str(symbol or "").strip().upper()
        if market not in self.symbols:
            raise ValueError("symbol is outside the configured Demo perpetual universe")
        instrument = self.broker.instrument_info(market)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 300 * CANDLE_MS
        frame = self.public.klines(market, start_ms=start_ms, end_ms=now_ms)
        if frame is None or len(frame) < 230:
            raise BybitLinearDemoError("not enough completed 4-hour candles for DI16 indicators")
        features = _features(frame, self.strategy)
        latest = features.iloc[-1]
        mark = self.broker.mark_price(market)
        atr = float(latest.atr)
        target = float(latest.mean_target)
        if not math.isfinite(atr) or atr <= 0 or not math.isfinite(target) or target <= 0:
            raise BybitLinearDemoError("DI16 ATR or mean target is unavailable")
        return {
            "status": "ok",
            "venue": "Bybit Demo",
            "category": "linear",
            "symbol": market,
            "timeframe": "4h",
            "mark_price": mark,
            "last_completed_candle": frame.index[-1].isoformat(),
            "last_close": float(frame.iloc[-1].close),
            "atr": atr,
            "mean_target": target,
            "signal": int(latest.signal),
            "suggested_long_stop": mark - self.strategy.atr_multiple * atr,
            "suggested_long_target": target,
            "suggested_short_stop": mark + self.strategy.atr_multiple * atr,
            "suggested_short_target": target,
            "quantity_step": instrument.quantity_step,
            "min_quantity": instrument.min_quantity,
            "min_notional": instrument.min_notional,
            "max_leverage": min(MAX_LEVERAGE, instrument.max_leverage),
            "updated_at": datetime.now(UTC).isoformat(),
        }

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
                "name": "Bollinger RSI Mean Reversion · DI16 + ATR1.6",
                "timeframe": "4h",
                "symbols": list(self.symbols),
                "leverage": 2.0,
                "max_leverage": MAX_LEVERAGE,
                "max_open_positions": MAX_OPEN_POSITIONS,
                "orders_submitted": False,
                "execution_enabled": True,
                "started_at": datetime.now(UTC).isoformat(),
                "detail": "Preflight passed; waiting for the next newly completed 4-hour candle.",
            }
            self._thread = threading.Thread(
                target=self._strategy_loop,
                name="bybit-demo-di16-linear",
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
            "detail": "DI16 + ATR1.6 is monitoring completed 4-hour candles; entries are capped by account risk controls.",
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
                try:
                    self.broker.close_market_position(
                        position["symbol"],
                        order_link_id=f"KILL-{position['symbol']}-{uuid.uuid4().hex[:12]}",
                    )
                except Exception:
                    pass
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
        last_processed = self._state.setdefault("last_processed_candles", {})
        managed = self._state.setdefault("strategy_positions", {})
        for symbol in self.symbols:
            if self._stop.is_set():
                break
            now_ms = int(time.time() * 1000)
            frame = self.public.klines(
                symbol,
                start_ms=now_ms - 300 * CANDLE_MS,
                end_ms=now_ms,
            )
            if frame is None or len(frame) < 230:
                continue
            features = _features(frame, self.strategy)
            candle_time = frame.index[-1]
            candle_key = candle_time.isoformat()
            if last_processed.get(symbol) == candle_key:
                continue
            # Persist the candle before any order submission: restart/timeout never replays it.
            last_processed[symbol] = candle_key
            self._save_state()
            latest = features.iloc[-1]
            position = active_by_symbol.get(symbol)
            if position is not None:
                meta = managed.get(symbol)
                if not isinstance(meta, dict):
                    continue  # manual positions are never silently managed by the strategy worker
                direction = 1 if position["side"] == "long" else -1
                adx = float(latest.adx) if math.isfinite(float(latest.adx)) else 0.0
                plus_di = float(latest.plus_di) if math.isfinite(float(latest.plus_di)) else 0.0
                minus_di = float(latest.minus_di) if math.isfinite(float(latest.minus_di)) else 0.0
                adverse = (direction > 0 and minus_di > plus_di) or (direction < 0 and plus_di > minus_di)
                count = int(meta.get("trend_exit_bars", 0)) + 1 if adx > 24.0 and adverse else 0
                meta["trend_exit_bars"] = count
                entry_candle = pd_timestamp(meta.get("entry_candle"))
                held_bars = max(0, int((candle_time - entry_candle).total_seconds() // (4 * 3600))) if entry_candle else 0
                if count >= 2 or held_bars >= 72:
                    close_result = self.broker.close_market_position(
                        symbol,
                        order_link_id=f"DI16-X-{symbol}-{int(candle_time.timestamp())}"[-36:],
                    )
                    managed.pop(symbol, None)
                    active_by_symbol.pop(symbol, None)
                    self._record(
                        {
                            "type": "strategy_exit",
                            "symbol": symbol,
                            "reason": "trend_regime_exit" if count >= 2 else "max_holding_bars",
                            "order_link_id": close_result.get("order_link_id"),
                        }
                    )
                continue
            signal = int(latest.signal)
            if signal not in (-1, 1) or len(active_by_symbol) >= MAX_OPEN_POSITIONS:
                continue
            atr = float(latest.atr)
            target = float(latest.mean_target)
            mark = self.broker.mark_price(symbol)
            if not math.isfinite(atr) or atr <= 0 or not math.isfinite(target) or target <= 0:
                continue
            side = "long" if signal > 0 else "short"
            stop = mark - signal * self.strategy.atr_multiple * atr
            if (signal > 0 and not stop < mark < target) or (signal < 0 and not target < mark < stop):
                self._record(
                    {
                        "type": "strategy_rejection",
                        "symbol": symbol,
                        "reason": "mean_target_not_favorable_at_actual_mark",
                    }
                )
                continue
            order_link_id = f"DI16-{symbol}-{int(candle_time.timestamp())}"[-36:]
            result = self.broker.open_market_position(
                symbol=symbol,
                side=side,
                leverage=2.0,
                equity=equity,
                stop_loss=stop,
                take_profit=target,
                order_link_id=order_link_id,
            )
            managed[symbol] = {
                "entry_candle": candle_key,
                "trend_exit_bars": 0,
                "order_link_id": order_link_id,
            }
            active_by_symbol[symbol] = {
                "symbol": symbol,
                "side": side,
                "quantity": result["quantity"],
                "entry_price": mark,
                "leverage": 2.0,
            }
            self._record(
                {
                    "type": "strategy_entry",
                    "symbol": symbol,
                    "side": side,
                    "leverage": 2.0,
                    "order_link_id": order_link_id,
                    "venue_order_id": result.get("order_id"),
                    "quantity": result.get("quantity"),
                    "notional_usdt": result.get("notional_usdt"),
                }
            )
            self._strategy_status = {
                **self._strategy_status,
                "orders_submitted": True,
                "last_order_at": datetime.now(UTC).isoformat(),
                "last_order": {
                    "symbol": symbol,
                    "side": side,
                    "order_link_id": order_link_id,
                    "venue_order_id": result.get("order_id"),
                },
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
  <div><div class="brand">PERPETUAL DEMO DESK</div><div class="sub">Bybit · USDT Linear Perpetual · 4H</div></div>
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
   <div class="heading"><h2>DI16 strategy</h2><span id="strategy-badge" class="pill">IDLE</span></div>
   <div class="rowitem">
    <strong>Bollinger + RSI Mean Reversion</strong>
    <div class="sub">DI16 filter · ATR 1.6 · 4-hour candles · 12 USDT perpetual pairs</div>
    <div class="stats">
     <div class="stat">Leverage<b>2x max</b></div><div class="stat">Risk per entry<b>0.5% equity</b></div>
     <div class="stat">Daily stop<b>3%</b></div><div class="stat">Drawdown stop<b>10%</b></div>
    </div>
   </div>
   <p id="strategy-detail" class="muted">Stopped. No automatic orders.</p>
   <div class="actions"><button id="start-strategy" class="primary">Start DI16 Auto</button><button id="stop-strategy" class="danger" disabled>Stop strategy</button></div>
   <div class="note">The strategy uses only completed 4H candles. Exchange-side Mark Price stop-loss and take-profit are attached to each entry. Starting the strategy enables automatic Demo orders after a new signal; it does not guarantee profitability.</div>
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
   <p id="market-note" class="sub">Loading DI16 ATR and mean target suggestions…</p>
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
async function loadMarket(){var s=$("symbol").value;try{var d=await api("/api/market?symbol="+encodeURIComponent(s));$("mark").value=fmt(d.mark_price,6);var side=$("side").value;$("stop").value=String(side==="long"?d.suggested_long_stop:d.suggested_short_stop);$("target").value=String(side==="long"?d.suggested_long_target:d.suggested_short_target);$("market-note").textContent="Completed candle "+new Date(d.last_completed_candle).toLocaleString()+" · ATR "+fmt(d.atr,6)+" · BB mean "+fmt(d.mean_target,6)+" · signal "+d.signal+" · qty step "+d.quantity_step;}catch(e){$("market-note").textContent="Market / DI16 data unavailable: "+e.message;}}
async function manual(side){$("side").value=side;var symbol=$("symbol").value,stop=Number($("stop").value),target=Number($("target").value),leverage=Number($("leverage").value);if(!Number.isFinite(stop)||!Number.isFinite(target)||stop<=0||target<=0){msg("Enter a valid stop loss and take profit first.",false);return;}if(!confirm("Submit a "+side.toUpperCase()+" market order on BYBIT DEMO "+symbol+" at up to "+leverage+"x? SL and TP will be attached at the exchange."))return;try{var d=await api("/api/order",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:JSON.stringify({symbol:symbol,side:side,leverage:leverage,stop_loss:stop,take_profit:target})});msg("Demo order accepted: "+(d.order_id||d.order_link_id)+" · "+fmt(d.quantity,6)+" "+symbol+". Confirm fill in the positions panel.",true);await refresh();}catch(e){msg(e.message,false);}}
async function closePosition(symbol){if(!confirm("Close the entire "+symbol+" position using a reduce-only market order on Bybit Demo?"))return;try{var d=await api("/api/close",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:JSON.stringify({symbol:symbol})});msg(d.closed===false?"No open position was found.":"Reduce-only close accepted for "+symbol+".",true);await refresh();}catch(e){msg(e.message,false);}}
async function startStrategy(){if(!confirm("Start automatic DI16 + ATR1.6 trading on Bybit Demo USDT Perpetual? Orders may be sent when new 4H signals appear."))return;try{var s=await api("/api/strategy/start",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:"{}"});renderStrategy(s);msg("DI16 strategy start requested. Automatic orders remain limited to Bybit Demo.",true);}catch(e){msg(e.message,false);}await refresh();}
async function stopStrategy(){if(!confirm("Stop new DI16 entries? Existing positions keep their exchange-side SL/TP protections."))return;try{var s=await api("/api/strategy/stop",{method:"POST",headers:{"X-Demo-Confirm":"yes"},body:"{}"});renderStrategy(s);msg("Strategy stop requested. Existing positions remain open and protected.",true);}catch(e){msg(e.message,false);}await refresh();}
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
