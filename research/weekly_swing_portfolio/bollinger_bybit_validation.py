"""Re-evaluate archived 4h Bollinger/RSI rules on actual Bybit public data."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from bybit_check import candles, request

from tradingagents.crypto.backtest.perpetual_bollinger_forward import (
    CORE_SYMBOLS,
    _count_forward_signals,
)
from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualEMAStrategy,
    _features,
    run_backtest,
)
from tradingagents.crypto.backtest.perpetual_ema_walkforward import COST_SCENARIOS
from tradingagents.crypto.perpetuals import LinearPerpetualInstrument, PerpetualRiskSettings

FILTERS = Path(__file__).with_name("wide-bybit-filters.json")
WINDOWS = (
    ("before_archived_development", "2024-05-01", "2024-11-01"),
    ("after_archived_development", "2026-11-01", "2026-11-20"),
)
LATEST_WEEK = ("latest_7_days", "2026-11-13", "2026-11-20")
CANDIDATE = {
    "signal_mode": "bollinger_reversion",
    "fast": 14,
    "slow": 50,
    "trend": 200,
    "min_adx": None,
    "mean_reversion_period": 20,
    "mean_reversion_std": 2.0,
    "mean_reversion_rsi_period": 14,
    "mean_reversion_rsi_long": 32.0,
    "mean_reversion_rsi_short": 68.0,
    "mean_reversion_max_adx": 18.0,
    "atr_multiple": 2.0,
    "max_holding_bars": 72,
}


def millis(stamp):
    return stamp.value // 10**6


def get_instrument(symbol):
    raw = request("instruments-info", symbol=symbol)[0]
    if raw["symbol"] != symbol:
        raise ValueError("instrument response did not match symbol")
    lot, price, lev = raw["lotSizeFilter"], raw["priceFilter"], raw["leverageFilter"]
    instrument = LinearPerpetualInstrument(
        symbol,
        raw["baseCoin"],
        raw["quoteCoin"],
        raw["settleCoin"],
        raw["status"],
        raw["contractType"],
        float(lot["minOrderQty"]),
        float(lot["maxMktOrderQty"]),
        float(lot["qtyStep"]),
        float(lot["minNotionalValue"]),
        float(price["tickSize"]),
        float(lev["minLeverage"]),
        float(lev["maxLeverage"]),
        float(lev["leverageStep"]),
        int(raw["fundingInterval"]),
    )
    tiers = request("risk-limit", symbol=symbol)
    tier = min(tiers, key=lambda row: float(row["riskLimitValue"]))
    if float(tier["riskLimitValue"]) < 10 or float(tier["maxLeverage"]) < 2:
        raise ValueError("lowest Bybit risk tier cannot support tested limits")
    return instrument, float(tier["maintenanceMargin"])


def get_funding(symbol, index):
    # Request short calendar slices to avoid truncating even hourly settlement.
    first, end = index[0], index[-1] + pd.Timedelta(hours=4)
    cuts = [*pd.date_range(first, end, freq="7D", inclusive="left"), end]
    events = {}
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        rows = request(
            "funding/history", symbol=symbol, startTime=millis(a), endTime=millis(b) - 1, limit=200
        )
        for row in rows:
            stamp = pd.to_datetime(int(row["fundingRateTimestamp"]), unit="ms", utc=True)
            rate = float(row["fundingRate"])
            if not a <= stamp < b or stamp in events:
                raise ValueError("duplicate or out-of-window funding event")
            events[stamp] = rate
    rates = pd.Series(events, dtype=float).sort_index()
    gaps = rates.index.to_series().diff()
    if (
        len(rates) < 2
        or not gaps.iloc[1:].isin([pd.Timedelta(hours=4), pd.Timedelta(hours=8)]).all()
    ):
        raise ValueError("funding intervals are incompatible with the 4h replay")
    rates.attrs["funding_interval_minutes"] = gaps.bfill().dt.total_seconds() / 60
    return rates


def run_window(trade, marks, rates, instrument, cfg):
    cases = {}
    for scenario, cost in COST_SCENARIOS.items():
        result = run_backtest(
            trade,
            marks,
            rates,
            instrument,
            initial_equity=50,
            trade_start_index=260,
            strategy=cfg,
            costs=cost,
        )
        cases[scenario] = {
            **result.metrics(),
            "closed_trade_ledger": [t.to_dict() for t in result.trades],
        }
    return {
        "bars": len(trade) - 260,
        "signals": _count_forward_signals(_features(trade, cfg), 260),
        "cases": cases,
    }


def evaluate(symbol, pool):
    instrument, maintenance = get_instrument(symbol)
    risk = PerpetualRiskSettings(leverage=2, max_leverage=2, maintenance_margin_rate=maintenance)
    cfg = PerpetualEMAStrategy(enabled=True, risk=risk, **CANDIDATE)
    results = {}
    errors = {}
    for label, start, end in WINDOWS:
        start = pd.Timestamp(start, tz="UTC")
        index = pd.date_range(
            start - pd.Timedelta(hours=4 * 260),
            pd.Timestamp(end, tz="UTC"),
            freq="4h",
            inclusive="left",
        )
        try:
            trade = candles(symbol, "kline", 240, index, pool)
            marks = candles(symbol, "mark-price-kline", 240, index, pool)
            rates = get_funding(symbol, index)
            results[label] = run_window(trade, marks, rates, instrument, cfg)
            if end == LATEST_WEEK[2]:
                cutoff = pd.Timestamp(LATEST_WEEK[1], tz="UTC") - pd.Timedelta(hours=4 * 260)
                try:
                    results[LATEST_WEEK[0]] = run_window(
                        trade.loc[cutoff:],
                        marks.loc[cutoff:],
                        rates.loc[cutoff:],
                        instrument,
                        cfg,
                    )
                except ValueError as exc:
                    errors[LATEST_WEEK[0]] = str(exc)
        except (OSError, ValueError) as exc:
            errors[label] = str(exc)
    return {
        "filters": asdict(instrument),
        "maintenance_margin_rate": maintenance,
        "results": results,
        "errors": errors,
    }


def comparisons(rows, label, symbols):
    reports = {}
    tested = [s for s in symbols if label in rows[s].get("results", {})]
    for scenario in COST_SCENARIOS:
        cases = [rows[s]["results"][label]["cases"][scenario] for s in tested]
        pnls = [t["net_pnl"] for case in cases for t in case["closed_trade_ledger"]]
        wins = sum(x > 0 for x in pnls)
        gross_win = sum(max(0, x) for x in pnls)
        gross_loss = -sum(min(0, x) for x in pnls)
        rejected = Counter()
        for case in cases:
            rejected.update(case["rejections"])
        reports[scenario] = {
            "net_pnl_sum_independent_50_usdt_runs": sum(pnls),
            "trades": len(pnls),
            "wins": wins,
            "win_rate_pct": 100 * wins / len(pnls) if pnls else None,
            "profit_factor": gross_win / gross_loss
            if gross_loss
            else ("infinite" if gross_win else None),
            "worst_symbol_drawdown_pct": max((c["max_drawdown_pct"] for c in cases), default=None),
            "rejections": dict(rejected),
            "symbols_with_trades": sum(c["trades"] > 0 for c in cases),
            "highest_symbol_end_usdt": max((c["final_equity"] for c in cases), default=None),
            "liquidations": sum(c["liquidations"] for c in cases),
        }
    return {
        "tested": tested,
        "unavailable": [s for s in symbols if s not in tested],
        "cost_cases": reports,
    }


def main():
    symbols = list(json.loads(FILTERS.read_text(encoding="utf-8")))

    def one(symbol, pool):
        try:
            return evaluate(symbol, pool)
        except (OSError, ValueError) as exc:
            return {"errors": {"instrument": str(exc)}}

    with ThreadPoolExecutor(16) as markets, ThreadPoolExecutor(18) as pool:
        rows = dict(
            zip(symbols, markets.map(lambda symbol: one(symbol, pool), symbols), strict=True)
        )
    report = {
        "candidate": "selective_bollinger_rsi_range",
        "parameters": CANDIDATE,
        "leverage": 2,
        "initial_equity_per_symbol": 50,
        "source": "Bybit public USDT perpetual",
        "windows": [*WINDOWS, LATEST_WEEK],
        "per_symbol": rows,
        "comparison": {
            label: {
                "core": comparisons(rows, label, CORE_SYMBOLS),
                "expanded": comparisons(rows, label, symbols),
            }
            for label, _, _ in (*WINDOWS, LATEST_WEEK)
        },
    }
    Path("bollinger-bybit-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
