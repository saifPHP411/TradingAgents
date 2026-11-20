"""Check two fixed 4-hour entry rules with the existing isolated-risk replay."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from bollinger_bybit_validation import FILTERS, comparisons, get_funding, get_instrument
from bybit_check import candles, request

from tradingagents.crypto.backtest.perpetual_bollinger_forward import (
    CORE_SYMBOLS,
    _count_forward_signals,
)
from tradingagents.crypto.backtest.perpetual_ema import PerpetualEMAStrategy, run_backtest
from tradingagents.crypto.backtest.perpetual_ema_walkforward import COST_SCENARIOS
from tradingagents.crypto.perpetuals import PerpetualRiskSettings

WINDOWS = (
    ("prior_comparison", "2024-05-01", "2024-11-01"),
    ("holdout", "2025-05-01", "2025-08-01"),
    ("recent_check", "2026-11-01", "2026-11-20"),
)
NAMES = ("volume_breakout", "trend_pullback")


def entry_signals(bars, quote_volume):
    close = bars.close
    fast = close.ewm(span=20, min_periods=20, adjust=False).mean()
    medium = close.ewm(span=50, min_periods=50, adjust=False).mean()
    slow = close.ewm(span=200, min_periods=200, adjust=False).mean()
    upper = bars.high.shift().rolling(24).max()
    lower = bars.low.shift().rolling(24).min()
    active = quote_volume >= 1.5 * quote_volume.shift().rolling(24).median()
    volume_breakout = ((close > upper) & (close > slow) & active).astype(int) - (
        (close < lower) & (close < slow) & active
    ).astype(int)
    trend_pullback = ((close > fast) & (close.shift() <= fast.shift()) & (medium > slow)).astype(
        int
    ) - ((close < fast) & (close.shift() >= fast.shift()) & (medium < slow)).astype(int)
    return dict(zip(NAMES, (volume_breakout, trend_pullback), strict=True))


def get_turnover(symbol, index, pool):
    # Linear USDT klines report turnover in USDT at index six.

    def page(offset):
        span = index[offset : offset + 900]
        return request(
            "kline",
            symbol=symbol,
            interval=240,
            start=span[0].value // 10**6,
            end=span[-1].value // 10**6 + 14_400_000 - 1,
            limit=len(span),
        )

    rows = [row for group in pool.map(page, range(0, len(index), 900)) for row in group]
    data = pd.Series(
        [float(row[6]) for row in rows],
        index=pd.to_datetime([int(row[0]) for row in rows], unit="ms", utc=True),
    ).sort_index()
    if not data.index.equals(index):
        raise ValueError("turnover timestamps are incomplete")
    return data


def check_symbol(symbol, pool):
    instrument, maintenance = get_instrument(symbol)
    cfg = PerpetualEMAStrategy(
        enabled=True,
        signal_mode="donchian_breakout",
        min_adx=None,
        atr_multiple=2.0,
        max_holding_bars=42,
        risk=PerpetualRiskSettings(leverage=2, max_leverage=2, maintenance_margin_rate=maintenance),
    )
    result = {"results": {}, "errors": {}}
    for phase, start, end in WINDOWS:
        index = pd.date_range(
            pd.Timestamp(start, tz="UTC") - pd.Timedelta(hours=4 * 260),
            pd.Timestamp(end, tz="UTC"),
            freq="4h",
            inclusive="left",
        )
        try:
            traded = candles(symbol, "kline", 240, index, pool)
            mark = candles(symbol, "mark-price-kline", 240, index, pool)
            funding = get_funding(symbol, index)
            for rule, signals in entry_signals(traded, get_turnover(symbol, index, pool)).items():
                cases = {}
                for name, costs in COST_SCENARIOS.items():
                    run = run_backtest(
                        traded,
                        mark,
                        funding,
                        instrument,
                        initial_equity=50,
                        trade_start_index=260,
                        strategy=cfg,
                        costs=costs,
                        entry_signals=signals,
                    )
                    cases[name] = {
                        **run.metrics(),
                        "closed_trade_ledger": [t.to_dict() for t in run.trades],
                    }
                result["results"][f"{phase}/{rule}"] = {
                    "signals": _count_forward_signals(signals.to_frame("signal"), 260),
                    "cases": cases,
                }
        except (OSError, ValueError, RuntimeError) as exc:
            result["errors"][phase] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    symbols = list(json.loads(FILTERS.read_text(encoding="utf-8")))
    with ThreadPoolExecutor(18) as assets, ThreadPoolExecutor(20) as pool:
        result = dict(
            zip(symbols, assets.map(lambda s: check_symbol(s, pool), symbols), strict=True)
        )
    summary = {}
    for phase, _, _ in WINDOWS:
        for rule in NAMES:
            label = f"{phase}/{rule}"
            summary[label] = {
                "core": comparisons(result, label, CORE_SYMBOLS),
                "expanded": comparisons(result, label, symbols),
            }
    report = {"windows": WINDOWS, "comparison": summary, "per_symbol": result}
    Path("bybit-trend-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
