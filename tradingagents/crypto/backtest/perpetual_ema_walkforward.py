"""Multi-symbol, cost-aware Bybit linear-perpetual walk-forward research.

This fixed-parameter study compares 1x and 2x isolated leverage on public
Bybit data. Development folds are disjoint and chronological; the final
holdout is reserved and evaluated once. It never places exchange orders.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualCosts,
    PerpetualEMAStrategy,
    run_backtest,
)
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
COST_SCENARIOS = {
    "realistic_estimate": PerpetualCosts(commission_bps=5.5, half_spread_bps=1.0, slippage_bps=2.0),
    "conservative": PerpetualCosts(commission_bps=8.0, half_spread_bps=2.0, slippage_bps=5.0),
    "stress": PerpetualCosts(commission_bps=12.0, half_spread_bps=4.0, slippage_bps=10.0),
}
TARGET_WIN_RATE_PCT = 79.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--equity", type=float, default=1_000.0)
    parser.add_argument("--leverage", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument("--test-bars", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("results/perpetual_ema_walkforward.json"))
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if isinstance(args.days, bool) or not isinstance(args.days, int) or not 180 <= args.days <= 3_650:
        raise ValueError("days must be an integer in [180, 3650]")
    if not math.isfinite(args.equity) or args.equity <= 0:
        raise ValueError("equity must be finite and positive")
    if isinstance(args.warmup_bars, bool) or args.warmup_bars < 220:
        raise ValueError("warmup-bars must be at least 220")
    if isinstance(args.test_bars, bool) or args.test_bars < 100:
        raise ValueError("test-bars must be at least 100")
    if not math.isfinite(args.holdout_fraction) or not 0.10 <= args.holdout_fraction <= 0.35:
        raise ValueError("holdout-fraction must be between 0.10 and 0.35")
    if not args.symbols or len(set(args.symbols)) != len(args.symbols):
        raise ValueError("symbols must be a non-empty unique list")
    if not args.leverage or any(
        isinstance(value, bool) or not math.isfinite(value) or value not in (1.0, 2.0)
        for value in args.leverage
    ):
        raise ValueError("only 1x and 2x research leverage are permitted")


def _fetch_history(client: BybitLinearPublicClient, symbol: str, start_ms: int, end_ms: int):
    instrument = client.instrument_info(symbol)
    if (
        instrument.status != "Trading"
        or instrument.contract_type != "LinearPerpetual"
        or instrument.quote_coin != "USDT"
        or instrument.settle_coin != "USDT"
    ):
        raise ValueError(f"{symbol} is not an active USDT-settled linear perpetual")
    traded = client.klines(instrument.symbol, start_ms=start_ms, end_ms=end_ms)
    mark = client.mark_price_klines(instrument.symbol, start_ms=start_ms, end_ms=end_ms)
    funding = client.funding_rates(
        instrument.symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        interval_minutes=instrument.funding_interval_minutes,
    )
    if not traded.index.equals(mark.index):
        raise ValueError(f"{symbol} traded and mark-price candle timestamps differ")
    return instrument, traded, mark, funding


def _pooled_trade_metrics(results: list) -> dict:
    trades = [trade for result in results for trade in result.trades]
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else None,
        "profit_factor": sum(wins) / gross_loss if gross_loss else ("infinite" if wins else None),
        "expectancy_per_trade": sum(t.net_pnl for t in trades) / len(trades) if trades else None,
        "liquidations": sum(t.exit_reason == "liquidation" for t in trades),
        "gross_pnl": sum(t.gross_pnl for t in trades),
        "fees": sum(t.fees for t in trades),
        "funding": sum(t.funding for t in trades),
        "net_trade_pnl": sum(t.net_pnl for t in trades),
    }


def _compound(returns_pct: list[float]) -> float:
    value = 1.0
    for value_pct in returns_pct:
        value *= 1.0 + value_pct / 100.0
    return (value - 1.0) * 100.0


def _max_drawdown_from_returns(returns_pct: list[float]) -> float:
    equity = peak = 1.0
    drawdown = 0.0
    for value_pct in returns_pct:
        equity *= 1.0 + value_pct / 100.0
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return drawdown * 100.0


def _run_window(
    traded: pd.DataFrame,
    mark: pd.DataFrame,
    funding: pd.Series,
    instrument,
    *,
    start: int,
    end: int,
    warmup_bars: int,
    equity: float,
    leverage: float,
    costs: PerpetualCosts,
):
    warm_start = max(0, start - warmup_bars)
    window = traded.iloc[warm_start:end].copy()
    mark_window = mark.iloc[warm_start:end].copy()
    if len(window) <= start - warm_start + 2:
        raise ValueError("walk-forward window is too short")
    funding_window = funding.loc[
        (funding.index >= window.index[0]) & (funding.index <= window.index[-1])
    ].copy()
    if funding_window.empty:
        raise ValueError("funding window is empty")
    strategy = PerpetualEMAStrategy(
        enabled=True,
        risk=PerpetualRiskSettings(leverage=leverage, max_leverage=2.0),
    )
    return run_backtest(
        window,
        mark_window,
        funding_window,
        instrument,
        initial_equity=equity,
        strategy=strategy,
        costs=costs,
        trade_start_index=start - warm_start,
    )


def _summarize_development(records: list[dict], symbols: list[str]) -> dict:
    all_results = [row["result"] for row in records]
    pooled = _pooled_trade_metrics(all_results)
    per_symbol_returns: dict[str, list[float]] = defaultdict(list)
    fold_returns: dict[int, list[float]] = defaultdict(list)
    for row in records:
        result = row["result"]
        ret = (result.final_equity / result.initial_equity - 1.0) * 100.0
        per_symbol_returns[row["symbol"]].append(ret)
        fold_returns[row["fold"]].append(ret)
    symbol_compounded = {
        symbol: _compound(per_symbol_returns[symbol])
        for symbol in symbols
        if per_symbol_returns[symbol]
    }
    ordered_folds = [sum(fold_returns[key]) / len(fold_returns[key]) for key in sorted(fold_returns)]
    portfolio_return = _compound(ordered_folds)
    positive_symbols = sum(value > 0 for value in symbol_compounded.values())
    return {
        **pooled,
        "equal_weight_compounded_return_pct": portfolio_return,
        "mean_symbol_compounded_return_pct": (
            sum(symbol_compounded.values()) / len(symbol_compounded) if symbol_compounded else None
        ),
        "per_symbol_compounded_return_pct": symbol_compounded,
        "positive_symbols": positive_symbols,
        "symbol_count": len(symbol_compounded),
        "positive_folds": sum(value > 0 for value in ordered_folds),
        "fold_count": len(ordered_folds),
        "worst_fold_drawdown_pct": max(
            (row["result"].metrics()["max_drawdown_pct"] for row in records),
            default=0.0,
        ),
    }


def _summarize_holdout(records: list[dict], symbols: list[str]) -> dict:
    all_results = [row["result"] for row in records]
    pooled = _pooled_trade_metrics(all_results)
    per_symbol = {}
    returns = []
    for row in records:
        result = row["result"]
        metrics = result.metrics()
        returns.append(metrics["net_return_pct"])
        per_symbol[row["symbol"]] = metrics
    return {
        **pooled,
        "equal_weight_return_pct": sum(returns) / len(returns) if returns else None,
        "mean_symbol_max_drawdown_pct": (
            sum(item["max_drawdown_pct"] for item in per_symbol.values()) / len(per_symbol)
            if per_symbol else None
        ),
        "worst_symbol_max_drawdown_pct": max(
            (item["max_drawdown_pct"] for item in per_symbol.values()), default=0.0
        ),
        "positive_symbols": sum(
            item["net_return_pct"] > 0 for item in per_symbol.values()
        ),
        "symbol_count": len(per_symbol),
        "per_symbol": per_symbol,
    }


def run_study(args: argparse.Namespace) -> dict:
    _validate_args(args)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400_000
    client = BybitLinearPublicClient()
    history = {
        symbol: _fetch_history(client, symbol, start_ms, end_ms)
        for symbol in args.symbols
    }

    # Intersect exact timestamps: never forward-fill or synthesize missing candles.
    common = None
    for _, traded, mark, _ in history.values():
        common = traded.index if common is None else common.intersection(traded.index)
    if common is None or len(common) < args.warmup_bars + args.test_bars * 3:
        raise ValueError("insufficient common aligned history for walk-forward folds")
    common = common.sort_values()
    if not common.is_unique or not common.is_monotonic_increasing:
        raise ValueError("common candle timestamps are not unique and ordered")
    if not (common.to_series().diff().dropna() == pd.Timedelta(hours=4)).all():
        raise ValueError("common market history contains missing 4-hour intervals")

    aligned = {}
    for symbol, (instrument, traded, mark, funding) in history.items():
        traded = traded.loc[common].copy()
        mark = mark.loc[common].copy()
        funding = funding.loc[
            (funding.index >= common[0]) & (funding.index <= common[-1])
        ].copy()
        aligned[symbol] = (instrument, traded, mark, funding)

    holdout_start = int(len(common) * (1.0 - args.holdout_fraction))
    if holdout_start <= args.warmup_bars + args.test_bars:
        raise ValueError("history is too short after reserving the final holdout")
    fold_ranges = []
    cursor = args.warmup_bars
    fold_number = 0
    while cursor + args.test_bars <= holdout_start:
        fold_ranges.append((fold_number, cursor, cursor + args.test_bars))
        fold_number += 1
        cursor += args.test_bars
    if len(fold_ranges) < 2:
        raise ValueError("at least two disjoint development OOS folds are required")

    reports = {}
    for leverage in args.leverage:
        for scenario_name, costs in COST_SCENARIOS.items():
            development_records = []
            for symbol, (instrument, traded, mark, funding) in aligned.items():
                for fold, start, end in fold_ranges:
                    result = _run_window(
                        traded, mark, funding, instrument,
                        start=start, end=end, warmup_bars=args.warmup_bars,
                        equity=args.equity, leverage=leverage, costs=costs,
                    )
                    development_records.append({
                        "symbol": symbol, "fold": fold,
                        "start": common[start].isoformat(),
                        "end": common[end - 1].isoformat(),
                        "result": result,
                    })

            holdout_records = []
            for symbol, (instrument, traded, mark, funding) in aligned.items():
                result = _run_window(
                    traded, mark, funding, instrument,
                    start=holdout_start, end=len(common),
                    warmup_bars=args.warmup_bars, equity=args.equity,
                    leverage=leverage, costs=costs,
                )
                holdout_records.append({"symbol": symbol, "result": result})

            development = _summarize_development(development_records, list(aligned))
            holdout = _summarize_holdout(holdout_records, list(aligned))
            gate = {
                "target_win_rate_pct": TARGET_WIN_RATE_PCT,
                "win_rate_pass": (
                    holdout["win_rate_pct"] is not None
                    and holdout["win_rate_pct"] >= TARGET_WIN_RATE_PCT
                ),
                "positive_after_costs": (
                    holdout["equal_weight_return_pct"] is not None
                    and holdout["equal_weight_return_pct"] > 0
                ),
                "profit_factor_pass": (
                    isinstance(holdout["profit_factor"], (int, float))
                    and holdout["profit_factor"] >= 1.2
                ),
                "drawdown_pass": holdout["worst_symbol_max_drawdown_pct"] <= 10.0,
                "no_liquidations": holdout["liquidations"] == 0,
                "minimum_trade_count": holdout["trades"] >= 30,
                "breadth_pass": holdout["positive_symbols"] >= max(1, math.ceil(len(aligned) * 0.75)),
            }
            gate["passed"] = all(gate[key] for key in gate if key not in ("target_win_rate_pct",))
            key = f"{leverage:g}x/{scenario_name}"
            reports[key] = {
                "leverage": leverage,
                "costs": asdict(costs),
                "development_walk_forward": development,
                "untouched_final_holdout": holdout,
                "acceptance_gate": gate,
                "folds": [
                    {
                        "fold": row["fold"],
                        "symbol": row["symbol"],
                        "start": row["start"],
                        "end": row["end"],
                        **row["result"].metrics(),
                    }
                    for row in development_records
                ],
            }

    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "credentials_used": False,
        "demo_ready": False,
        "venue": "Bybit V5 public linear USDT perpetual market data",
        "symbols": list(aligned),
        "timeframe": "4h",
        "requested_days": args.days,
        "common_candles": len(common),
        "history_start": common[0].isoformat(),
        "history_end": common[-1].isoformat(),
        "warmup_bars": args.warmup_bars,
        "test_bars": args.test_bars,
        "holdout_fraction": args.holdout_fraction,
        "development_folds": len(fold_ranges),
        "final_holdout_start": common[holdout_start].isoformat(),
        "strategy": {
            "name": "EMA14/EMA50 crossover + EMA200 regime + ADX18 + ATR14 trailing stop",
            "parameters_fixed_before_oos": True,
            "risk_per_trade_fraction": 0.005,
            "max_research_leverage": 2.0,
        },
        "assumptions": [
            "completed 4-hour candles only; next-bar-open simulated entries",
            "mark-price candles used for liquidation triggers; simplified liquidation price remains an approximation",
            "historical Bybit funding settlements are applied at their event timestamps",
            "fees, spread and slippage are estimates, not account-specific measured fills",
            "each symbol is simulated independently; aggregate returns are equal-weighted",
            "development OOS folds are disjoint; the final chronological holdout is evaluated once",
            "no parameter fitting or selection is performed on the final holdout",
        ],
        "reports": reports,
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_study(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        key: {
            "development": value["development_walk_forward"],
            "final_holdout": value["untouched_final_holdout"],
            "acceptance_gate": value["acceptance_gate"],
        }
        for key, value in report["reports"].items()
    }
    print(json.dumps({
        "symbols": report["symbols"],
        "common_candles": report["common_candles"],
        "final_holdout_start": report["final_holdout_start"],
        "reports": compact,
        "orders_submitted": False,
        "demo_ready": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
