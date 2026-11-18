"""Prospective, frozen-parameter Bollinger/RSI forward test.

Reads public Binance USD-M market data and current exchange quantity filters.
It submits no orders and does not optimize parameters on the forward window.
Each symbol is tested independently with the full configured equity; results
must not be interpreted as one simultaneous multi-symbol portfolio.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from tradingagents.crypto.backtest.perpetual_ema import (
    PerpetualEMAStrategy,
    run_backtest,
)
from tradingagents.crypto.backtest.perpetual_ema_walkforward import COST_SCENARIOS
from tradingagents.crypto.binance_perpetual import BinanceUSDMPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_FORWARD_START = "2026-11-01T00:00:00+00:00"

# Frozen development candidates. This script compares them prospectively only;
# it does not search or tune parameters on the forward period.
FORWARD_CANDIDATES: dict[str, dict] = {
    "bollinger_rsi_quality20_atr16_di16": {
        "signal_mode": "bollinger_reversion",
        "fast": 14,
        "slow": 50,
        "trend": 200,
        "min_adx": None,
        "mean_reversion_period": 20,
        "mean_reversion_std": 1.9,
        "mean_reversion_rsi_period": 14,
        "mean_reversion_rsi_long": 35.0,
        "mean_reversion_rsi_short": 65.0,
        "mean_reversion_max_adx": 18.5,
        "mean_reversion_min_rsi_recovery": 5.0,
        "mean_reversion_max_adverse_di_spread": 16.0,
        "mean_reversion_exit_adx": 24.0,
        "mean_reversion_exit_confirmation_bars": 2,
        "mean_reversion_exit_requires_adverse_di": True,
        "atr_multiple": 1.6,
        "max_holding_bars": 72,
    },
    "bollinger_rsi_quality20_atr16_std18": {
        "signal_mode": "bollinger_reversion",
        "fast": 14,
        "slow": 50,
        "trend": 200,
        "min_adx": None,
        "mean_reversion_period": 20,
        "mean_reversion_std": 1.8,
        "mean_reversion_rsi_period": 14,
        "mean_reversion_rsi_long": 35.0,
        "mean_reversion_rsi_short": 65.0,
        "mean_reversion_max_adx": 18.5,
        "mean_reversion_min_rsi_recovery": 5.0,
        "mean_reversion_exit_adx": 24.0,
        "mean_reversion_exit_confirmation_bars": 2,
        "mean_reversion_exit_requires_adverse_di": True,
        "atr_multiple": 1.6,
        "max_holding_bars": 72,
    },
}


def _parse_forward_start(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("forward-start must include a timezone")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor("4h"):
        raise ValueError("forward-start must align to a 4-hour candle boundary")
    return stamp


def _validate_args(args: argparse.Namespace) -> pd.Timestamp:
    if not args.symbols or len(set(args.symbols)) != len(args.symbols):
        raise ValueError("symbols must be a non-empty unique list")
    if not math.isfinite(args.equity) or args.equity <= 0:
        raise ValueError("equity must be finite and positive")
    if isinstance(args.warmup_bars, bool) or args.warmup_bars < 220:
        raise ValueError("warmup-bars must be at least 220")
    if not args.leverage or any(
        isinstance(value, bool)
        or not math.isfinite(value)
        or value not in (1.0, 2.0)
        for value in args.leverage
    ):
        raise ValueError("only 1x and 2x research leverage are permitted")
    return _parse_forward_start(args.forward_start)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--forward-start", default=DEFAULT_FORWARD_START)
    parser.add_argument("--equity", type=float, default=50.0)
    parser.add_argument("--leverage", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/perpetual_bollinger_forward.json"),
    )
    return parser


def _load_symbol_window(
    client: BinanceUSDMPublicClient,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    forward_start: pd.Timestamp,
):
    instrument = client.instrument_info(symbol)
    traded = client.klines(symbol, start_ms=start_ms, end_ms=end_ms)
    mark = client.mark_price_klines(symbol, start_ms=start_ms, end_ms=end_ms)
    if not traded.index.equals(mark.index):
        raise ValueError(f"{symbol}: traded and mark-price timestamps are not aligned")
    if len(traded) < 223:
        raise ValueError(f"{symbol}: insufficient completed 4-hour candles")
    gaps = traded.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(hours=4)).all():
        raise ValueError(f"{symbol}: missing or irregular 4-hour candles")
    trade_start_index = int(traded.index.searchsorted(forward_start))
    if trade_start_index < 220 or trade_start_index >= len(traded):
        raise ValueError(f"{symbol}: forward window has no tradable candles after warm-up")
    last_open_ms = int(traded.index[-1].timestamp() * 1000)
    funding = client.funding_rates(
        symbol,
        start_ms=start_ms,
        end_ms=last_open_ms,
        interval_minutes=instrument.funding_interval_minutes,
    )
    filters = {
        "min_quantity": instrument.min_quantity,
        "max_market_quantity": instrument.max_market_quantity,
        "quantity_step": instrument.quantity_step,
        "min_notional": instrument.min_notional,
        "tick_size": instrument.tick_size,
        "funding_interval_minutes": instrument.funding_interval_minutes,
        "source": "current Binance USD-M exchangeInfo and fundingInfo",
    }
    return instrument, traded, mark, funding, trade_start_index, filters


def run_forward_study(args: argparse.Namespace) -> dict:
    forward_start = _validate_args(args)
    end = pd.Timestamp.now(tz="UTC").floor("4h")
    if end <= forward_start:
        raise ValueError("no completed forward candles exist after forward-start")
    start = forward_start - pd.Timedelta(hours=4 * args.warmup_bars)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    client = BinanceUSDMPublicClient()
    aligned = {}
    instruments = {}
    errors = {}
    for symbol in args.symbols:
        try:
            loaded = _load_symbol_window(
                client,
                symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                forward_start=forward_start,
            )
            instrument, traded, mark, funding, trade_start_index, filters = loaded
            aligned[symbol] = (
                instrument,
                traded,
                mark,
                funding,
                trade_start_index,
            )
            instruments[symbol] = filters
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"

    if not aligned:
        raise RuntimeError(
            "no symbol passed market-data validation: "
            + json.dumps(errors, sort_keys=True)
        )

    reports = {}
    for candidate_name, overrides in FORWARD_CANDIDATES.items():
        for leverage in args.leverage:
            for scenario_name, costs in COST_SCENARIOS.items():
                symbol_results = {}
                for symbol, (
                    instrument,
                    traded,
                    mark,
                    funding,
                    trade_start_index,
                ) in aligned.items():
                    strategy = PerpetualEMAStrategy(
                        enabled=True,
                        risk=PerpetualRiskSettings(
                            leverage=leverage,
                            max_leverage=2.0,
                        ),
                        **overrides,
                    )
                    result = run_backtest(
                        traded,
                        mark,
                        funding,
                        instrument,
                        initial_equity=args.equity,
                        strategy=strategy,
                        costs=costs,
                        trade_start_index=trade_start_index,
                    )
                    symbol_results[symbol] = {
                        **result.metrics(),
                        "rejections": dict(result.rejections),
                        "trades_ledger": [
                            trade.to_dict() for trade in result.trades
                        ],
                    }
                key = f"{candidate_name}/{leverage:g}x/{scenario_name}"
                reports[key] = {
                    "candidate": candidate_name,
                    "parameters": overrides,
                    "leverage": leverage,
                    "costs": {
                        "commission_bps": costs.commission_bps,
                        "half_spread_bps": costs.half_spread_bps,
                        "slippage_bps": costs.slippage_bps,
                    },
                    "symbols": symbol_results,
                }

    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "credentials_used": False,
        "demo_ready": False,
        "venue": client.venue_name,
        "data_source": client.base_url,
        "forward_start": forward_start.isoformat(),
        "forward_end_exclusive": end.isoformat(),
        "history_warmup_start": start.isoformat(),
        "latest_completed_candle_open": max(
            frame[1].index[-1] for frame in aligned.values()
        ).isoformat(),
        "timeframe": "4h",
        "initial_equity_per_independent_symbol_test": args.equity,
        "portfolio_warning": (
            "Each symbol is simulated independently with the full initial equity. "
            "These results are not a combined multi-symbol portfolio and must not "
            "be summed as if the same 50 USDT were deployed four times."
        ),
        "current_exchange_filters": instruments,
        "symbol_data_errors": errors,
        "parameters_locked_before_forward_run": True,
        "forward_window_used_for_optimization": False,
        "reports": reports,
        "limitations": [
            "Binance USD-M is a cross-venue proxy if execution is intended on another exchange.",
            "Current exchangeInfo quantity filters are used, but account-specific leverage brackets are not fetched.",
            "The forward window may contain too few trades for statistical confidence.",
            "Fees, spread and slippage remain scenario estimates, not measured account fills.",
            "This is a prospective research backtest, not authenticated Demo execution.",
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_forward_study(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "forward_start": report["forward_start"],
        "forward_end_exclusive": report["forward_end_exclusive"],
        "symbols": list(report["current_exchange_filters"]),
        "symbol_data_errors": report["symbol_data_errors"],
        "parameters_locked_before_forward_run": True,
        "orders_submitted": False,
        "reports": {
            key: {
                symbol: {
                    "trades": value["trades"],
                    "win_rate_pct": value["win_rate_pct"],
                    "profit_factor": value["profit_factor"],
                    "net_return_pct": value["net_return_pct"],
                    "liquidations": value["liquidations"],
                    "rejections": value["rejections"],
                }
                for symbol, value in row["symbols"].items()
            }
            for key, row in report["reports"].items()
        },
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
