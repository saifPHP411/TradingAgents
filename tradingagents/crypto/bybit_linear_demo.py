"""Strictly Demo-only order adapter for USDT linear perpetual contracts.

This module deliberately has no configurable base URL and never reads legacy
Spot or live credential names. All trading requests are routed to Bybit Demo.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from urllib.parse import quote, urlsplit

import requests

from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import (
    PerpetualRiskSettings,
    PositionSize,
    size_linear_position,
)

DEMO_BASE_URL = "https://api-demo.bybit.com"
API_KEY_ENV = "BYBIT_LINEAR_DEMO_API_KEY"
API_SECRET_ENV = "BYBIT_LINEAR_DEMO_API_SECRET"
MAX_LEVERAGE = 2.0
MAX_OPEN_POSITIONS = 2
DEFAULT_COST_RATE_PER_FILL = 0.00085
AUTO_ENTRY_MAX_ADVERSE_BPS = 10
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")
_ORDER_LINK_RE = re.compile(r"^[A-Za-z0-9_-]{1,36}$")


class BybitLinearDemoError(RuntimeError):
    """A Demo linear-perpetual operation failed closed."""


class BybitLinearDemoNotConfigured(BybitLinearDemoError):
    """A dedicated Demo trading key pair is not configured."""


class BybitLinearDemoRejected(BybitLinearDemoError):
    """Bybit rejected a request or a local safety check refused it."""


class BybitLinearDemoAmbiguous(BybitLinearDemoError):
    """An order submission timed out and its venue state needs reconciliation."""

    def __init__(self, order_link_id: str):
        super().__init__(
            "order submission outcome is unknown; reconcile the client order ID before retrying"
        )
        self.order_link_id = order_link_id


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be finite and positive") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _encode_query(params: Mapping[str, object]) -> str:
    return "&".join(
        f"{quote(str(key), safe='')}={quote(str(params[key]), safe='')}"
        for key in sorted(params, key=str)
    )


def _decimal_string(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f")


def _round_price(price: Decimal, tick: Decimal, rounding: str) -> float:
    return float((price / tick).to_integral_value(rounding=rounding) * tick)


def _order_id(value: str) -> str:
    identifier = str(value or "").strip()
    if not _ORDER_LINK_RE.fullmatch(identifier):
        raise ValueError("orderLinkId must contain 1-36 letters, digits, underscores or dashes")
    return identifier


class BybitLinearDemoBroker:
    """Bybit V5 Demo broker; linear USDT perpetuals and one-way positions only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        transport: Callable | None = None,
        clock_ms: Callable[[], int] | None = None,
        timeout: float = 12.0,
        public_client=None,
    ) -> None:
        parsed = urlsplit(DEMO_BASE_URL)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api-demo.bybit.com"
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("the fixed Bybit Demo endpoint failed its safety check")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")

        self._key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self._secret = api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        if bool(self._key) != bool(self._secret):
            raise BybitLinearDemoNotConfigured(
                f"both {API_KEY_ENV} and {API_SECRET_ENV} must be configured"
            )
        self.transport = transport or self._default_transport
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.timeout = float(timeout)
        self.public = public_client or BybitLinearPublicClient(base_url=DEMO_BASE_URL)

    @staticmethod
    def _default_transport(*, method, url, headers, params=None, data=None, timeout):
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data,
            timeout=timeout,
        )
        response.raise_for_status()
        return response

    @property
    def configured(self) -> bool:
        return bool(self._key and self._secret)

    @property
    def endpoint(self) -> str:
        return DEMO_BASE_URL

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        body: Mapping[str, object] | None = None,
        private: bool = True,
        submit_before_ms: int | None = None,
    ) -> dict:
        method = method.upper()
        if method not in {"GET", "POST"} or not path.startswith("/v5/"):
            raise ValueError("unsupported Bybit V5 request")
        url = DEMO_BASE_URL + path
        if not url.startswith(DEMO_BASE_URL + "/"):
            raise BybitLinearDemoRejected("refused a non-Demo Bybit endpoint")
        # The exact query-string order sent over HTTP must match the order
        # used in the V5 HMAC signature. Signing sorted keys while allowing
        # requests to serialize the original insertion order causes retCode
        # 10004 (signature error) on endpoints whose parameter order differs.
        query = dict(sorted((params or {}).items(), key=lambda item: str(item[0])))
        payload = dict(body or {})
        headers = {"Accept": "application/json"}
        data = None
        if private:
            if not self.configured:
                raise BybitLinearDemoNotConfigured(
                    f"dedicated Demo trading credentials are required: {API_KEY_ENV} and {API_SECRET_ENV}"
                )
            now_ms = self.clock_ms()
            timestamp = str(now_ms)
            remaining_ms = 5000 if submit_before_ms is None else submit_before_ms - now_ms
            if remaining_ms <= 0:
                raise BybitLinearDemoRejected("entry_signal_expired")
            recv_window = str(min(5000, remaining_ms))
            if method == "GET":
                signed_payload = _encode_query(query)
            else:
                signed_payload = json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                )
                data = signed_payload
                headers["Content-Type"] = "application/json"
            signature_payload = f"{timestamp}{self._key}{recv_window}{signed_payload}"
            signature = hmac.new(
                self._secret.encode("utf-8"),
                signature_payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers.update(
                {
                    "X-BAPI-API-KEY": self._key,
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": recv_window,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                }
            )
        try:
            response = self.transport(
                method=method,
                url=url,
                headers=headers,
                params=query if method == "GET" else None,
                data=data,
                timeout=self.timeout,
            )
            status = int(getattr(response, "status_code", 200))
            if status >= 400:
                raise BybitLinearDemoError(f"Bybit Demo returned HTTP {status}")
            result = response.json() if callable(getattr(response, "json", None)) else json.loads(response.text)
        except BybitLinearDemoError:
            raise
        except Exception as exc:
            if method == "POST":
                raise BybitLinearDemoAmbiguous(
                    str(payload.get("orderLinkId") or "unknown")
                ) from None
            raise BybitLinearDemoError(
                f"Bybit Demo {path.rsplit('/', 1)[-1]} request failed ({type(exc).__name__})"
            ) from None

        if not isinstance(result, dict):
            raise BybitLinearDemoError("Bybit Demo returned a malformed response")
        try:
            ret_code = int(result["retCode"])
        except (KeyError, TypeError, ValueError):
            raise BybitLinearDemoError("Bybit Demo response omitted a valid return code") from None
        if ret_code != 0:
            raise BybitLinearDemoRejected(
                f"Bybit Demo rejected {path.rsplit('/', 1)[-1]} (retCode {ret_code})"
            )
        result_data = result.get("result")
        if not isinstance(result_data, dict):
            raise BybitLinearDemoError("Bybit Demo response omitted its result object")
        return result_data

    def instrument_info(self, symbol: str):
        market = self._symbol(symbol)
        instrument = self.public.instrument_info(market)
        if (
            instrument.symbol != market
            or instrument.status != "Trading"
            or instrument.contract_type != "LinearPerpetual"
            or instrument.quote_coin != "USDT"
            or instrument.settle_coin != "USDT"
        ):
            raise BybitLinearDemoRejected("only active USDT linear perpetual contracts are permitted")
        if instrument.max_leverage < 1.0 or instrument.min_leverage > MAX_LEVERAGE:
            raise BybitLinearDemoRejected("the instrument does not support the allowed leverage range")
        return instrument

    @staticmethod
    def _symbol(symbol: str) -> str:
        market = str(symbol or "").strip().upper()
        if not _SYMBOL_RE.fullmatch(market):
            raise ValueError("invalid Bybit symbol")
        return market

    def mark_price(self, symbol: str) -> float:
        return _finite_positive(self._ticker(symbol).get("markPrice"), "markPrice")

    def _ticker(self, symbol: str) -> dict:
        market = self._symbol(symbol)
        result = self._request(
            "GET",
            "/v5/market/tickers",
            params={"category": "linear", "symbol": market},
            private=False,
        )
        rows = result.get("list")
        if (
            not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get("symbol", market) != market
        ):
            raise BybitLinearDemoError("Bybit Demo ticker did not match the requested contract")
        return rows[0]

    def get_wallet_balance(self) -> dict:
        from tradingagents.crypto.bybit_perpetual_account import (
            BybitLinearDemoPositionClient,
        )

        if not self.configured:
            raise BybitLinearDemoNotConfigured(
                f"dedicated Demo trading credentials are required: {API_KEY_ENV} and {API_SECRET_ENV}"
            )
        reader = BybitLinearDemoPositionClient(
            api_key=self._key,
            api_secret=self._secret,
            timeout=self.timeout,
        )
        return reader.get_wallet_balance()

    def get_positions(self) -> list[dict]:
        result = self._request(
            "GET",
            "/v5/position/list",
            params={"category": "linear", "settleCoin": "USDT", "limit": 200},
        )
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitLinearDemoError("Bybit Demo positions response omitted its list")
        positions = []
        for row in rows:
            if not isinstance(row, dict):
                raise BybitLinearDemoError("Bybit Demo returned a malformed position row")
            symbol = self._symbol(row.get("symbol", ""))
            try:
                size = float(row.get("size") or 0)
            except (TypeError, ValueError):
                raise BybitLinearDemoError("Bybit Demo returned an invalid position size") from None
            if not math.isfinite(size) or size < 0:
                raise BybitLinearDemoError("Bybit Demo returned an invalid position size")
            if size == 0:
                continue
            if row.get("positionIdx") not in (0, "0", None):
                raise BybitLinearDemoRejected(
                    "hedge-mode positions are unsupported; switch the Demo account to one-way mode"
                )
            side_raw = str(row.get("side") or "").lower()
            if side_raw not in {"buy", "sell"}:
                raise BybitLinearDemoError("Bybit Demo returned an unknown position side")
            positions.append(
                {
                    "symbol": symbol,
                    "side": "long" if side_raw == "buy" else "short",
                    "quantity": size,
                    "entry_price": _finite_positive(row.get("avgPrice"), "avgPrice"),
                    "mark_price": _finite_positive(row.get("markPrice"), "markPrice"),
                    "leverage": _finite_positive(row.get("leverage"), "leverage"),
                    "unrealized_pnl": float(row.get("unrealisedPnl") or 0),
                    "liquidation_price": float(row.get("liqPrice") or 0) or None,
                    "stop_loss": float(row.get("stopLoss") or 0) or None,
                    "take_profit": float(row.get("takeProfit") or 0) or None,
                    "position_idx": 0,
                    "created_time_ms": int(row.get("createdTime") or 0),
                    "updated_time_ms": int(row.get("updatedTime") or 0),
                }
            )
        if len({p["symbol"] for p in positions}) != len(positions):
            raise BybitLinearDemoRejected("multiple position legs found; one-way mode is required")
        return positions

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        params: dict[str, object] = {"category": "linear", "settleCoin": "USDT", "openOnly": 0, "limit": 50}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        result = self._request("GET", "/v5/order/realtime", params=params)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitLinearDemoError("Bybit Demo open-orders response omitted its list")
        return [
            {
                "symbol": self._symbol(row.get("symbol", "")),
                "order_id": str(row.get("orderId") or ""),
                "order_link_id": str(row.get("orderLinkId") or ""),
                "side": str(row.get("side") or ""),
                "qty": str(row.get("qty") or ""),
                "order_status": str(row.get("orderStatus") or ""),
                "order_type": str(row.get("orderType") or ""),
                "stop_order_type": str(row.get("stopOrderType") or ""),
                "trigger_price": str(row.get("triggerPrice") or "") or None,
                "trigger_by": str(row.get("triggerBy") or "") or None,
                "take_profit": str(row.get("takeProfit") or "") or None,
                "stop_loss": str(row.get("stopLoss") or "") or None,
                "tpsl_mode": str(row.get("tpslMode") or "") or None,
                "tp_trigger_by": str(row.get("tpTriggerBy") or "") or None,
                "sl_trigger_by": str(row.get("slTriggerBy") or "") or None,
                "reduce_only": bool(row.get("reduceOnly", False)),
                "close_on_trigger": bool(row.get("closeOnTrigger", False)),
                "position_idx": int(row.get("positionIdx") or 0),
                "created_time_ms": int(row.get("createdTime") or 0),
                "updated_time_ms": int(row.get("updatedTime") or 0),
            }
            for row in rows
            if isinstance(row, dict)
        ]

    @staticmethod
    def _optional_number(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _history_time(value: object) -> int:
        try:
            number = int(value or 0)
        except (TypeError, ValueError):
            return 0
        return number if number > 0 else 0

    @staticmethod
    def _validate_history_window(start_time_ms: int | None, end_time_ms: int | None) -> None:
        for name, value in (("start_time_ms", start_time_ms), ("end_time_ms", end_time_ms)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if start_time_ms is not None and end_time_ms is not None:
            if start_time_ms > end_time_ms:
                raise ValueError("start_time_ms must not exceed end_time_ms")
            if end_time_ms - start_time_ms > 7 * 24 * 60 * 60 * 1000:
                raise ValueError("Bybit history windows cannot exceed seven days")

    def get_closed_pnl(
        self,
        *,
        symbol: str | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return venue-reported closed PnL, including average fill prices and fees."""
        self._validate_history_window(start_time_ms, end_time_ms)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("closed PnL limit must be in [1, 100]")
        params: dict[str, object] = {"category": "linear", "limit": limit}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        result = self._request("GET", "/v5/position/closed-pnl", params=params)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitLinearDemoError("Bybit Demo closed-PnL response omitted its list")
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append({
                "symbol": self._symbol(row.get("symbol", "")),
                "order_id": str(row.get("orderId") or ""),
                "side": str(row.get("side") or ""),
                "qty": self._optional_number(row.get("qty")),
                "closed_size": self._optional_number(row.get("closedSize")),
                "avg_entry_price": self._optional_number(row.get("avgEntryPrice")),
                "avg_exit_price": self._optional_number(row.get("avgExitPrice")),
                "closed_pnl": self._optional_number(row.get("closedPnl")),
                "open_fee": self._optional_number(row.get("openFee")),
                "close_fee": self._optional_number(row.get("closeFee")),
                "fill_count": self._optional_number(row.get("fillCount")),
                "leverage": self._optional_number(row.get("leverage")),
                "created_time_ms": self._history_time(row.get("createdTime")),
                "updated_time_ms": self._history_time(row.get("updatedTime")),
            })
        return normalized

    def get_order_history(
        self,
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        order_link_id: str | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent venue order records for exit attribution and fill reconciliation."""
        self._validate_history_window(start_time_ms, end_time_ms)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("order history limit must be in [1, 50]")
        params: dict[str, object] = {"category": "linear", "limit": limit}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        if order_id:
            params["orderId"] = _order_id(order_id)
        if order_link_id:
            params["orderLinkId"] = _order_id(order_link_id)
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        result = self._request("GET", "/v5/order/history", params=params)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitLinearDemoError("Bybit Demo order-history response omitted its list")
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append({
                "symbol": self._symbol(row.get("symbol", "")),
                "order_id": str(row.get("orderId") or ""),
                "order_link_id": str(row.get("orderLinkId") or ""),
                "parent_order_link_id": str(row.get("parentOrderLinkId") or ""),
                "side": str(row.get("side") or ""),
                "qty": self._optional_number(row.get("qty")),
                "price": self._optional_number(row.get("price")),
                "order_status": str(row.get("orderStatus") or ""),
                "order_type": str(row.get("orderType") or ""),
                "stop_order_type": str(row.get("stopOrderType") or ""),
                "avg_price": self._optional_number(row.get("avgPrice")),
                "cum_exec_qty": self._optional_number(row.get("cumExecQty")),
                "cum_exec_value": self._optional_number(row.get("cumExecValue")),
                "cum_fee_detail": row.get("cumFeeDetail"),
                "trigger_price": self._optional_number(row.get("triggerPrice")),
                "take_profit": self._optional_number(row.get("takeProfit")),
                "stop_loss": self._optional_number(row.get("stopLoss")),
                "tpsl_mode": str(row.get("tpslMode") or "") or None,
                "tp_trigger_by": str(row.get("tpTriggerBy") or "") or None,
                "sl_trigger_by": str(row.get("slTriggerBy") or "") or None,
                "trigger_by": str(row.get("triggerBy") or "") or None,
                "reduce_only": bool(row.get("reduceOnly", False)),
                "close_on_trigger": bool(row.get("closeOnTrigger", False)),
                "reject_reason": str(row.get("rejectReason") or "") or None,
                "cancel_type": str(row.get("cancelType") or "") or None,
                "position_idx": int(row.get("positionIdx") or 0),
                "created_time_ms": self._history_time(row.get("createdTime")),
                "updated_time_ms": self._history_time(row.get("updatedTime")),
            })
        return normalized

    def get_order_by_link_id(self, symbol: str, order_link_id: str) -> dict | None:
        market = self._symbol(symbol)
        identifier = _order_id(order_link_id)
        result = self._request(
            "GET",
            "/v5/order/realtime",
            params={"category": "linear", "symbol": market, "orderLinkId": identifier},
        )
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitLinearDemoError("Bybit Demo order reconciliation response omitted its list")
        matches = [row for row in rows if isinstance(row, dict) and row.get("orderLinkId") == identifier]
        if len(matches) > 1:
            raise BybitLinearDemoRejected("orderLinkId matched more than one venue order")
        return matches[0] if matches else None

    def _set_leverage(self, symbol: str, leverage: float, instrument) -> None:
        lev = _finite_positive(leverage, "leverage")
        if lev > MAX_LEVERAGE or lev < 1.0:
            raise BybitLinearDemoRejected("leverage must be between 1x and the hard 2x ceiling")
        if lev < instrument.min_leverage or lev > instrument.max_leverage:
            raise BybitLinearDemoRejected("requested leverage is outside the current instrument limits")
        steps = (lev - instrument.min_leverage) / instrument.leverage_step
        if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
            raise BybitLinearDemoRejected("requested leverage does not match the instrument leverage step")
        try:
            self._request(
                "POST",
                "/v5/position/set-leverage",
                body={
                    "category": "linear",
                    "symbol": symbol,
                    "buyLeverage": _decimal_string(lev),
                    "sellLeverage": _decimal_string(lev),
                },
            )
        except BybitLinearDemoRejected as exc:
            # Bybit retCode 110043 means the requested leverage is already set.
            # It is a no-op success, not an order rejection; continue only for
            # this exact response and keep every other exchange rejection fatal.
            if "(retCode 110043)" not in str(exc):
                raise

    def _assert_no_position_or_entry_order(self, symbol: str) -> None:
        positions = [p for p in self.get_positions() if p["symbol"] == symbol]
        if positions:
            raise BybitLinearDemoRejected("an open position already exists for this symbol")
        active_orders = [
            o for o in self.get_open_orders(symbol)
            if o["order_status"] in {"New", "PartiallyFilled", "Untriggered"}
        ]
        if active_orders:
            raise BybitLinearDemoRejected("an active order already exists for this symbol")

    def size_position(
        self,
        *,
        symbol: str,
        equity: float,
        entry_price: float,
        stop_loss: float,
        side: str,
        leverage: float,
        notional_reference_price: float | None = None,
    ) -> PositionSize:
        market = self._symbol(symbol)
        direction = 1 if side == "long" else -1 if side == "short" else 0
        if direction == 0:
            raise ValueError("side must be long or short")
        instrument = self.instrument_info(market)
        lev = _finite_positive(leverage, "leverage")
        if not 1.0 <= lev <= MAX_LEVERAGE:
            raise BybitLinearDemoRejected("leverage must be between 1x and the hard 2x ceiling")
        risk = PerpetualRiskSettings(
            leverage=lev,
            max_leverage=MAX_LEVERAGE,
            risk_fraction=0.005,
            max_margin_fraction=0.10,
            max_notional_fraction=0.20,
            max_daily_loss_fraction=0.03,
            max_drawdown_fraction=0.10,
            max_consecutive_losses=3,
        )
        size = size_linear_position(
            equity=_finite_positive(equity, "equity"),
            entry_price=_finite_positive(entry_price, "entry_price"),
            stop_price=_finite_positive(stop_loss, "stop_loss"),
            direction=direction,
            cost_rate_per_fill=DEFAULT_COST_RATE_PER_FILL,
            instrument=instrument,
            risk=risk,
            notional_reference_price=notional_reference_price,
        )
        if not size.approved:
            raise BybitLinearDemoRejected(
                f"risk sizing rejected the order ({size.rejection_reason})"
            )
        return size

    def open_market_position(
        self,
        *,
        symbol: str,
        side: str,
        leverage: float,
        equity: float,
        stop_loss: float,
        take_profit: float,
        order_link_id: str,
    ) -> dict:
        return self.open_position(
            symbol=symbol, side=side, leverage=leverage, equity=equity,
            stop_loss=stop_loss, take_profit=take_profit, order_link_id=order_link_id,
        )

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        leverage: float,
        equity: float,
        stop_loss: float,
        take_profit: float,
        order_link_id: str,
        signal_price: float | None = None,
        signal_expires_ms: int | None = None,
    ) -> dict:
        """Use a signal-anchored Limit IOC when an automated entry supplies its reference."""
        market = self._symbol(symbol)
        direction = 1 if side == "long" else -1 if side == "short" else 0
        if direction == 0:
            raise ValueError("side must be long or short")
        identifier = _order_id(order_link_id)
        instrument = self.instrument_info(market)
        stop = _finite_positive(stop_loss, "stop_loss")
        target = _finite_positive(take_profit, "take_profit")
        limit = None
        if signal_price is not None or signal_expires_ms is not None:
            if isinstance(signal_expires_ms, bool) or not isinstance(signal_expires_ms, int):
                raise ValueError("automated entry requires a signal expiration time in milliseconds")
            reference = Decimal(str(_finite_positive(signal_price, "signal_price")))
            tick = Decimal(str(instrument.tick_size))
            raw_limit = reference * (1 + Decimal(direction * AUTO_ENTRY_MAX_ADVERSE_BPS) / 10_000)
            rounding = ROUND_FLOOR if direction > 0 else ROUND_CEILING
            limit = _round_price(raw_limit, tick, rounding)
            target = _round_price(Decimal(str(target)), tick, rounding)
            stop_rounding = ROUND_CEILING if direction > 0 else ROUND_FLOOR
            stop = _round_price(Decimal(str(stop)), tick, stop_rounding)
            if limit <= 0:
                raise BybitLinearDemoRejected("entry_price_limit_below_tick")
            if self.clock_ms() >= signal_expires_ms:
                raise BybitLinearDemoRejected("entry_signal_expired")
        positions = self.get_positions()
        if len(positions) >= MAX_OPEN_POSITIONS:
            raise BybitLinearDemoRejected("maximum of two open Demo positions reached")
        self._assert_no_position_or_entry_order(market)
        notional_reference_price = None
        if limit is None:
            mark = self.mark_price(market)
        else:
            ticker = self._ticker(market)
            mark = _finite_positive(ticker.get("markPrice"), "markPrice")
            book_side = "ask1Price" if direction > 0 else "bid1Price"
            executable_price = _finite_positive(ticker.get(book_side), book_side)
            if direction * (executable_price - limit) > 0:
                raise BybitLinearDemoRejected("entry_quote_outside_price_limit")
            if not min(stop, target) < executable_price < max(stop, target):
                raise BybitLinearDemoRejected("entry_quote_outside_protection_levels")
            if not min(stop, target) < limit < max(stop, target):
                raise BybitLinearDemoRejected("entry_limit_outside_protection_levels")
            if direction < 0:
                notional_reference_price = max(mark, executable_price)
        if direction > 0 and not (stop < mark < target):
            raise BybitLinearDemoRejected("for a Long, stop loss must be below mark and take profit above it")
        if direction < 0 and not (target < mark < stop):
            raise BybitLinearDemoRejected("for a Short, take profit must be below mark and stop loss above it")
        sizing = self.size_position(
            symbol=market,
            equity=equity,
            entry_price=limit if limit is not None else mark,
            stop_loss=stop,
            side=side,
            leverage=leverage,
            notional_reference_price=notional_reference_price,
        )
        if sizing.notional > _finite_positive(equity, "equity") * 0.20 + 1e-8:
            raise BybitLinearDemoRejected("notional exceeds the 20% equity cap")
        try:
            self._set_leverage(market, float(leverage), instrument)
        except BybitLinearDemoAmbiguous as exc:
            if limit is None:
                raise
            raise BybitLinearDemoRejected("leverage_update_unconfirmed_before_entry") from exc
        body = {
            "category": "linear",
            "symbol": market,
            "side": "Buy" if direction > 0 else "Sell",
            "orderType": "Market" if limit is None else "Limit",
            "qty": _decimal_string(sizing.quantity),
            "timeInForce": "IOC",
            "positionIdx": 0,
            "orderLinkId": identifier,
            "stopLoss": _decimal_string(stop),
            "takeProfit": _decimal_string(target),
            "slTriggerBy": "MarkPrice",
            "tpTriggerBy": "MarkPrice",
            "tpslMode": "Full",
            "slOrderType": "Market",
            "tpOrderType": "Market",
            "reduceOnly": False,
        }
        if limit is not None:
            body["price"] = _decimal_string(limit)
        try:
            result = self._request(
                "POST", "/v5/order/create", body=body, submit_before_ms=signal_expires_ms,
            )
            if not result.get("orderId") or result.get("orderLinkId") != identifier:
                raise BybitLinearDemoError("Bybit Demo accepted a response without the expected order identity")
        except (BybitLinearDemoRejected, BybitLinearDemoNotConfigured):
            raise
        except BybitLinearDemoError as exc:
            if limit is None and not isinstance(exc, BybitLinearDemoAmbiguous):
                raise
            # A timeout is not permission to resend. Look up the exact client ID.
            try:
                reconciled = self.get_order_by_link_id(market, identifier)
            except Exception:
                raise BybitLinearDemoAmbiguous(identifier) from None
            if reconciled is None:
                raise BybitLinearDemoAmbiguous(identifier) from None
            order = self._normalize_order(reconciled, sizing, side, leverage, True)
        else:
            order = self._normalize_order(
                {"orderId": result["orderId"], "orderLinkId": result["orderLinkId"], "orderStatus": "Created"},
                sizing, side, leverage, False,
            )
        if limit is not None:
            order.update(limit_price=limit, stop_loss=stop, take_profit=target)
        return order

    @staticmethod
    def _normalize_order(row: dict, sizing: PositionSize, side: str, leverage: float, reconciled: bool) -> dict:
        return {
            "status": str(row.get("orderStatus") or "Created"),
            "order_id": str(row.get("orderId") or ""),
            "order_link_id": str(row.get("orderLinkId") or ""),
            "side": side,
            "leverage": float(leverage),
            "quantity": sizing.quantity,
            "notional_usdt": sizing.notional,
            "initial_margin_usdt": sizing.initial_margin,
            "modeled_stop_risk_usdt": sizing.modeled_stop_risk,
            "reconciled_after_timeout": bool(reconciled),
            "endpoint": DEMO_BASE_URL,
            "category": "linear",
        }

    def close_market_position(self, symbol: str, *, order_link_id: str) -> dict:
        market = self._symbol(symbol)
        identifier = _order_id(order_link_id)
        matches = [p for p in self.get_positions() if p["symbol"] == market]
        if not matches:
            return {"status": "flat", "symbol": market, "closed": False}
        if len(matches) != 1:
            raise BybitLinearDemoRejected("multiple position legs found; one-way mode is required")
        position = matches[0]
        side = "Sell" if position["side"] == "long" else "Buy"
        body = {
            "category": "linear",
            "symbol": market,
            "side": side,
            "orderType": "Market",
            "qty": _decimal_string(position["quantity"]),
            "timeInForce": "IOC",
            "positionIdx": 0,
            "orderLinkId": identifier,
            "reduceOnly": True,
            "closeOnTrigger": True,
        }
        try:
            result = self._request("POST", "/v5/order/create", body=body)
        except BybitLinearDemoAmbiguous:
            try:
                reconciled = self.get_order_by_link_id(market, identifier)
            except Exception:
                raise BybitLinearDemoAmbiguous(identifier) from None
            if reconciled is None:
                raise BybitLinearDemoAmbiguous(identifier) from None
            result = reconciled
        order_id = result.get("orderId")
        link_id = result.get("orderLinkId")
        if not order_id or link_id != identifier:
            raise BybitLinearDemoError("Bybit Demo close response omitted the expected order identity")
        return {
            "status": str(result.get("orderStatus") or "Created"),
            "symbol": market,
            "side": side,
            "quantity": position["quantity"],
            "order_id": str(order_id),
            "order_link_id": str(link_id),
            "reduce_only": True,
            "category": "linear",
            "endpoint": DEMO_BASE_URL,
        }

    def account_state(self) -> dict:
        wallet = self.get_wallet_balance()
        positions = self.get_positions()
        orders = self.get_open_orders()
        return {
            "status": "ok",
            "venue": "Bybit Demo",
            "endpoint": DEMO_BASE_URL,
            "category": "linear",
            "quote_currency": "USDT",
            "wallet": wallet,
            "positions": positions,
            "open_orders": orders,
            "max_leverage": MAX_LEVERAGE,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "spot_enabled": False,
        }


__all__ = [
    "API_KEY_ENV",
    "API_SECRET_ENV",
    "DEMO_BASE_URL",
    "MAX_LEVERAGE",
    "MAX_OPEN_POSITIONS",
    "BybitLinearDemoBroker",
    "BybitLinearDemoError",
    "BybitLinearDemoNotConfigured",
    "BybitLinearDemoRejected",
    "BybitLinearDemoAmbiguous",
]
