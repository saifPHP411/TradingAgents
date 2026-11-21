"""Read-only Bybit V5 current public instrument/order-size feasibility probe.

This is a CURRENT market snapshot, not October/November 2026 historical
exchangeInfo, not a Bybit Demo authenticated account check, and not an order.
Uses only /v5/market/instruments-info and /v5/market/tickers (GET).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path

API_HOST = "https://api.bybit.com"
OFFICIAL_API_HOSTS = (API_HOST, "https://api.bytick.com")
SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"}
INITIAL_EQUITY = Decimal("50")
MAX_NOTIONAL_FRACTION = Decimal("0.20")
ADVERSE_PRICE_CAP = Decimal("1.001")


def positive_decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"invalid {field}")
    return number


def evaluate_public_filters(instrument: dict, ticker: dict, symbol: str) -> dict:
    """Use floor-to-step quantities, never round up to force an executable order."""
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError("unsupported linear USDT contract symbol")
    if not isinstance(instrument, dict) or not isinstance(ticker, dict):
        raise ValueError("public instrument and ticker objects required")
    if (
        instrument.get("symbol") != symbol
        or ticker.get("symbol") != symbol
        or instrument.get("status") != "Trading"
        or instrument.get("contractType") != "LinearPerpetual"
        or instrument.get("quoteCoin") != "USDT"
        or instrument.get("settleCoin") != "USDT"
    ):
        raise ValueError("symbol is not confirmed active linear USDT perpetual")
    lots = instrument.get("lotSizeFilter")
    if not isinstance(lots, dict):
        raise ValueError("missing lotSizeFilter")
    step = positive_decimal(lots.get("qtyStep"), "qtyStep")
    min_qty = positive_decimal(lots.get("minOrderQty"), "minOrderQty")
    min_notional = positive_decimal(lots.get("minNotionalValue"), "minNotionalValue")
    mark = positive_decimal(ticker.get("markPrice"), "markPrice")
    price_tick = positive_decimal(
        (instrument.get("priceFilter") or {}).get("tickSize"),
        "tickSize",
    )
    max_notional = INITIAL_EQUITY * MAX_NOTIONAL_FRACTION
    # A pessimistic 10bp adverse price move reduces the number of units
    # affordable under the unchanged $10 notional cap.
    capped_price = mark * ADVERSE_PRICE_CAP
    qty = (max_notional / capped_price / step).to_integral_value(
        rounding=ROUND_FLOOR,
    ) * step
    estimated_notional = qty * mark
    passes = qty > 0 and qty >= min_qty and estimated_notional >= min_notional
    return {
        "symbol": symbol,
        "venue": "Bybit public linear USDT market",
        "initial_equity_usdt": str(INITIAL_EQUITY),
        "max_position_notional_usdt": str(max_notional),
        "max_position_notional_fraction": str(MAX_NOTIONAL_FRACTION),
        "adverse_price_budget_multiplier": str(ADVERSE_PRICE_CAP),
        "mark_price_usdt": str(mark),
        "qty_step": str(step),
        "min_order_qty": str(min_qty),
        "min_notional_usdt": str(min_notional),
        "price_tick_usdt": str(price_tick),
        "max_qty_at_cap": str(qty),
        "estimated_notional_at_mark_usdt": str(estimated_notional),
        "passes_public_quantity_and_notional_filters": bool(passes),
        "account_permissions_and_historical_filters_verified": False,
        "market_order_executability_verified": False,
        "orders_submitted": False,
    }


def fetch_market_object(
    endpoint: str, symbol: str, *, transport=None, host: str = API_HOST,
) -> dict:
    if (
        symbol not in SUPPORTED_SYMBOLS
        or not re.fullmatch(r"/v5/market/[a-z-]+", endpoint)
        or host not in OFFICIAL_API_HOSTS
    ):
        raise ValueError("unsupported symbol, public API endpoint or source host")
    if endpoint not in ("/v5/market/instruments-info", "/v5/market/tickers"):
        raise ValueError("endpoint must be a documented public market GET route")
    url = host + endpoint + "?" + urllib.parse.urlencode({
        "category": "linear", "symbol": symbol,
    })
    if transport is None:
        req = urllib.request.Request(
            url, headers={"User-Agent": "TradingAgents-public-filter-research/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status != 200:
                raise ValueError("Bybit returned an unexpected HTTP response")
            raw = response.read(256 * 1024 + 1)
    else:
        raw = transport(url)
    if not isinstance(raw, bytes) or not raw or len(raw) > 256 * 1024:
        raise ValueError("empty or oversized Bybit market response")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("retCode") != 0:
        raise ValueError("Bybit public market API rejected request")
    entries = payload.get("result", {}).get("list")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Bybit returned no unique instrument/ticker entry")
    if entries[0].get("symbol") != symbol:
        raise ValueError("Bybit response symbol mismatch")
    return entries[0]


def probe(symbol: str, *, transport=None) -> dict:
    # Both officially documented Bybit mainnet hosts are allowed; source
    # identity is recorded. No third-party aggregators or private endpoints.
    errors = []
    for host in OFFICIAL_API_HOSTS:
        try:
            instrument = fetch_market_object(
                "/v5/market/instruments-info", symbol,
                transport=transport, host=host,
            )
            ticker = fetch_market_object(
                "/v5/market/tickers", symbol,
                transport=transport, host=host,
            )
            return {
                "status": "current_public_filters_checked",
                "checked_at_utc": datetime.now(UTC).isoformat(),
                "source_api_host": host,
                **evaluate_public_filters(instrument, ticker, symbol),
            }
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}: {exc}")
    raise ValueError("both official Bybit market hosts unavailable: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=sorted(SUPPORTED_SYMBOLS), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = probe(args.symbol)
    except Exception as exc:
        report = {
            "status": "public_filters_unavailable_or_invalid",
            "symbol": args.symbol,
            "error": f"{type(exc).__name__}: {exc}",
            "passes_public_quantity_and_notional_filters": None,
            "orders_submitted": False,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "current_public_filters_checked":
        sys.exit(2)


if __name__ == "__main__":
    main()
