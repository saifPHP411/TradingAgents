"""Read-only Bybit Demo position reader for USDT linear perpetuals.

This module signs only the GET /v5/position/list request. It intentionally
contains no order, leverage-setting, cancel, or account-mutation method.
Credentials are read from the environment and never returned or logged.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from urllib.parse import quote, urlsplit

import requests

DEMO_BASE_URL = "https://api-demo.bybit.com"
API_KEY_ENV = "BYBIT_DEMO_API_KEY"
API_SECRET_ENV = "BYBIT_DEMO_API_SECRET"
LIVE_HOSTS = frozenset({"api.bybit.com", "api-testnet.bybit.com", "api-test.bybit.com"})


class BybitPerpetualAccountError(RuntimeError):
    """A Demo account position read failed closed."""


class BybitPerpetualAccountNotConfigured(BybitPerpetualAccountError):
    """The optional read-only Demo credential pair is not configured."""


def _encode_query(params: Mapping[str, object]) -> str:
    return "&".join(
        f"{quote(str(key), safe='')}={quote(str(params[key]), safe='')}"
        for key in sorted(params, key=str)
    )


def _optional_positive(value: object, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise BybitPerpetualAccountError(f"Bybit returned an invalid {field}") from None
    if not math.isfinite(parsed) or parsed < 0:
        raise BybitPerpetualAccountError(f"Bybit returned an invalid {field}")
    # Bybit commonly uses "0" for an unset optional stop, target or liquidation
    # price. Treat it as unavailable rather than rejecting an otherwise valid
    # open position or drawing a fabricated zero-price line.
    if parsed == 0:
        return None
    return parsed


class BybitLinearDemoPositionClient:
    """Fetch one symbol's actual Bybit Demo linear position; GET-only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        transport: Callable | None = None,
        clock_ms: Callable[[], int] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self._secret = api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        parsed = urlsplit(DEMO_BASE_URL)
        if (
            parsed.scheme != "https"
            or parsed.hostname in LIVE_HOSTS
            or parsed.hostname != "api-demo.bybit.com"
            or parsed.port not in (None, 443)
        ):
            raise ValueError("only the fixed Bybit Demo host is permitted")
        self.base_url = DEMO_BASE_URL
        self.transport = transport or self._default_transport
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.timeout = timeout

    @staticmethod
    def _default_transport(*, method, url, headers, params, timeout):
        response = requests.request(
            method=method, url=url, headers=headers, params=params, timeout=timeout
        )
        response.raise_for_status()
        return response

    @property
    def configured(self) -> bool:
        return bool(self._key and self._secret)

    def _get_signed(self, path: str, params: Mapping[str, object]) -> dict:
        if not self.configured:
            raise BybitPerpetualAccountNotConfigured(
                f"read-only Demo position access requires {API_KEY_ENV} and {API_SECRET_ENV}"
            )
        url = self.base_url + path
        if not url.startswith(DEMO_BASE_URL + "/"):
            raise BybitPerpetualAccountError("refused a non-Demo Bybit endpoint")
        timestamp = str(self.clock_ms())
        recv_window = "5000"
        query = _encode_query(params)
        payload = f"{timestamp}{self._key}{recv_window}{query}"
        signature = hmac.new(
            self._secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self._key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "Accept": "application/json",
        }
        try:
            response = self.transport(
                method="GET", url=url, headers=headers, params=dict(params),
                timeout=self.timeout,
            )
            if getattr(response, "status_code", 200) >= 400:
                raise BybitPerpetualAccountError(
                    f"Bybit Demo position request returned HTTP {response.status_code}"
                )
            body = response.json() if callable(getattr(response, "json", None)) else json.loads(response.text)
        except BybitPerpetualAccountError:
            raise
        except Exception as exc:
            raise BybitPerpetualAccountError(
                f"Bybit Demo position read failed: {type(exc).__name__}"
            ) from None
        if (
            not isinstance(body, dict)
            or body.get("retCode") != 0
            or not isinstance(body.get("result"), dict)
        ):
            code = body.get("retCode") if isinstance(body, dict) else None
            raise BybitPerpetualAccountError(
                f"Bybit Demo rejected the read-only position request (retCode {code})"
            )
        return body["result"]

    @staticmethod
    def _finite_number(value: object, field: str, *, optional: bool = False,
                       allow_negative: bool = False) -> float | None:
        if value in (None, "") and optional:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise BybitPerpetualAccountError(f"Bybit returned an invalid {field}") from None
        if not math.isfinite(parsed) or (not allow_negative and parsed < 0):
            raise BybitPerpetualAccountError(f"Bybit returned an invalid {field}") from None
        return parsed

    def get_wallet_balance(self) -> dict:
        """Read the Unified USDT wallet and margin summary; signed GET only."""
        result = self._get_signed(
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED", "coin": "USDT"},
        )
        rows = result.get("list")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise BybitPerpetualAccountError(
                "Bybit Demo wallet response did not contain one Unified account"
            )
        account = rows[0]
        required = {
            "total_equity_usd": "totalEquity",
            "total_wallet_balance_usd": "totalWalletBalance",
        }
        metrics: dict[str, float | None] = {}
        for output, source in required.items():
            metrics[output] = self._finite_number(account.get(source), source)
        optional_fields = {
            "total_margin_balance_usd": "totalMarginBalance",
            "total_available_balance_usd": "totalAvailableBalance",
            "total_perpetual_unrealized_pnl_usd": "totalPerpUPL",
            "total_initial_margin_usd": "totalInitialMargin",
            "total_maintenance_margin_usd": "totalMaintenanceMargin",
        }
        for output, source in optional_fields.items():
            metrics[output] = self._finite_number(
                account.get(source), source, optional=True
            )
        coins = account.get("coin")
        if not isinstance(coins, list):
            raise BybitPerpetualAccountError(
                "Bybit Demo wallet response omitted its coin list"
            )
        usdt_rows = [
            row for row in coins
            if isinstance(row, dict) and str(row.get("coin", "")).upper() == "USDT"
        ]
        if len(usdt_rows) > 1:
            raise BybitPerpetualAccountError(
                "Bybit Demo wallet returned duplicate USDT balance rows"
            )
        usdt = usdt_rows[0] if usdt_rows else {}
        coin_fields = {
            "usdt_wallet_balance": "walletBalance",
            "usdt_equity": "equity",
            "usdt_unrealized_pnl": "unrealisedPnl",
            "usdt_usd_value": "usdValue",
            "usdt_locked": "locked",
            "usdt_borrow_amount": "borrowAmount",
        }
        for output, source in coin_fields.items():
            metrics[output] = self._finite_number(
                usdt.get(source, "0"), source, optional=False,
                allow_negative=source == "unrealisedPnl",
            )
        return {
            "status": "ok",
            "account_type": str(account.get("accountType", "UNIFIED")),
            **metrics,
            "updated_at_ms": self.clock_ms(),
            "source": "Bybit Demo GET /v5/account/wallet-balance",
            "quote_currency": "USDT",
        }

    def get_closed_pnl(self, *, limit: int = 100) -> dict:
        """Read recent linear-perpetual closed PnL; Bybit defaults to the last 7 days."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("closed PnL limit must be an integer between 1 and 100")
        result = self._get_signed(
            "/v5/position/closed-pnl",
            {"category": "linear", "limit": limit},
        )
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitPerpetualAccountError(
                "Bybit Demo closed-PnL response omitted its list"
            )
        records = []
        for row in rows:
            if not isinstance(row, dict):
                raise BybitPerpetualAccountError(
                    "Bybit Demo returned a malformed closed-PnL record"
                )
            symbol = str(row.get("symbol", "")).upper()
            if not symbol.isalnum() or not 3 <= len(symbol) <= 30:
                raise BybitPerpetualAccountError(
                    "Bybit Demo returned an invalid closed-PnL symbol"
                )
            side = str(row.get("side", "")).lower()
            if side not in {"buy", "sell"}:
                raise BybitPerpetualAccountError(
                    "Bybit Demo returned an invalid closed-PnL side"
                )
            try:
                created_ms = int(row.get("createdTime"))
                updated_ms = int(row.get("updatedTime"))
            except (TypeError, ValueError):
                raise BybitPerpetualAccountError(
                    "Bybit Demo returned invalid closed-PnL timestamps"
                ) from None
            if created_ms <= 0 or updated_ms <= 0:
                raise BybitPerpetualAccountError(
                    "Bybit Demo returned invalid closed-PnL timestamps"
                )
            records.append({
                "symbol": symbol,
                "side": "long" if side == "buy" else "short",
                "quantity": self._finite_number(row.get("closedSize", row.get("qty", "0")), "closedSize"),
                "entry_price": self._finite_number(row.get("avgEntryPrice"), "avgEntryPrice", optional=True),
                "exit_price": self._finite_number(row.get("avgExitPrice"), "avgExitPrice", optional=True),
                "closed_pnl": self._finite_number(row.get("closedPnl"), "closedPnl", allow_negative=True),
                "open_fee": self._finite_number(row.get("openFee", "0"), "openFee"),
                "close_fee": self._finite_number(row.get("closeFee", "0"), "closeFee"),
                "leverage": self._finite_number(row.get("leverage"), "leverage", optional=True),
                "created_time_ms": created_ms,
                "updated_time_ms": updated_ms,
                "exec_type": str(row.get("execType", "")),
            })
        records.sort(key=lambda item: (item["created_time_ms"], item["updated_time_ms"]))
        return {
            "status": "ok",
            "records": records,
            "window": "Bybit default recent 7 days",
            "updated_at_ms": self.clock_ms(),
            "source": "Bybit Demo GET /v5/position/closed-pnl",
            "next_page_available": bool(result.get("nextPageCursor")),
        }

    def get_position(self, symbol: str) -> dict:
        """Return {status: flat|open, position: ...}; never infer from Spot."""
        market = str(symbol).strip().upper()
        if not market.isalnum() or not 3 <= len(market) <= 30:
            raise ValueError("invalid linear perpetual symbol")
        result = self._get_signed(
            "/v5/position/list",
            {"category": "linear", "settleCoin": "USDT", "symbol": market},
        )
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitPerpetualAccountError("Bybit Demo position response omitted its list")
        open_rows = []
        for row in rows:
            if not isinstance(row, dict) or row.get("symbol") != market:
                raise BybitPerpetualAccountError("Bybit Demo returned an unexpected position row")
            try:
                size = float(row.get("size") or 0)
            except (TypeError, ValueError):
                raise BybitPerpetualAccountError("Bybit Demo returned an invalid position size") from None
            if not math.isfinite(size) or size < 0:
                raise BybitPerpetualAccountError("Bybit Demo returned an invalid position size")
            if size > 0:
                open_rows.append((row, size))
        if not open_rows:
            return {"status": "flat", "position": None}
        if len(open_rows) != 1:
            raise BybitPerpetualAccountError(
                "multiple open position legs found; one-way position is required for this chart"
            )
        row, quantity = open_rows[0]
        if row.get("positionIdx") not in (0, "0", None):
            raise BybitPerpetualAccountError(
                "Bybit Demo hedge-mode position index is unsupported; one-way position is required"
            )
        side_raw = str(row.get("side") or "").lower()
        if side_raw not in {"buy", "sell"}:
            raise BybitPerpetualAccountError("Bybit Demo position direction is unknown")
        try:
            position_idx = int(row.get("positionIdx", 0))
        except (TypeError, ValueError):
            raise BybitPerpetualAccountError("Bybit Demo position mode is unknown") from None
        if position_idx != 0:
            raise BybitPerpetualAccountError(
                "hedge-mode positions are not supported by this one-way research dashboard"
            )
        try:
            leverage = float(row.get("leverage"))
            unrealized = float(row.get("unrealisedPnl") or 0)
        except (TypeError, ValueError):
            raise BybitPerpetualAccountError("Bybit Demo returned invalid position metrics") from None
        if not math.isfinite(leverage) or leverage <= 0 or not math.isfinite(unrealized):
            raise BybitPerpetualAccountError("Bybit Demo returned invalid position metrics")
        position = {
            "symbol": market,
            "side": "long" if side_raw == "buy" else "short",
            "quantity": quantity,
            "leverage": leverage,
            "entry_price": _optional_positive(row.get("avgPrice"), "avgPrice"),
            "mark_price": _optional_positive(row.get("markPrice"), "markPrice"),
            "stop_loss": _optional_positive(row.get("stopLoss"), "stopLoss"),
            "take_profit": _optional_positive(row.get("takeProfit"), "takeProfit"),
            "liquidation_price": _optional_positive(row.get("liqPrice"), "liqPrice"),
            "unrealized_pnl": unrealized,
            "updated_at": row.get("updatedTime"),
            "position_idx": row.get("positionIdx"),
            "position_source": "Bybit Demo GET /v5/position/list",
        }
        if position["entry_price"] is None:
            raise BybitPerpetualAccountError("Bybit Demo open position has no average entry price")
        return {"status": "open", "position": position}


__all__ = [
    "API_KEY_ENV",
    "API_SECRET_ENV",
    "DEMO_BASE_URL",
    "BybitLinearDemoPositionClient",
    "BybitPerpetualAccountError",
    "BybitPerpetualAccountNotConfigured",
]
