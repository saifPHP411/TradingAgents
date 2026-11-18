"""Prospective, frozen-parameter Bollinger/RSI forward test.

Uses official Binance Vision daily/monthly public market-data archives.
Historical funding is used only before the prospective cutoff. Since the
current month's realized funding archive is not yet complete, forward funding
is evaluated under explicit zero and +/-1bp-per-8h assumptions. Venue quantity
filters are unavailable in Vision archives and are therefore research
placeholders, not proof of executable sizing. No orders are submitted.
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
from tradingagents.crypto.binance_vision import BinanceVisionPublicClient
from tradingagents.crypto.perpetuals import PerpetualRiskSettings


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_FORWARD_START = "2026-11-01T00:00:00+00:00"
FUNDING_SCENARIOS = {
    "zero_forward_funding": 0.0,
    "positive_1bp_per_8h": 0.0001,
    "negative_1bp_per_8h": -0.0001,
}

# Frozen development candidates. Parameters are not optimized on this window.
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
    if isinstance(args.equity, bool) or not math.isfinite(args.equity) or args.equity <= 0:
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
    client: BinanceVisionPublicClient,
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

    # Binance Vision publishes realized funding monthly. Request history only
    # through the prospective cutoff; forward payments are explicit scenarios.
    funding = client.funding_rates(
        symbol,
        start_ms=start_ms,
        end_ms=int(forward_start.timestamp() * 1000),
        interval_minutes=instrument.funding_interval_minutes,
    )
    funding = funding.loc[funding.index < forward_start].copy()
    filters = {
        "min_quantity": instrument.min_quantity,
        "max_market_quantity": instrument.max_market_quantity,
        "quantity_step": instrument.quantity_step,
        "min_notional": instrument.min_notional,
        "tick_size": instrument.tick_size,
        "funding_interval_minutes": instrument.funding_interval_minutes,
        "source": "Binance Vision research placeholders; not live exchangeInfo",
        "executable_filters_verified": False,
    }
    return instrument, traded, mark, funding, trade_start_index, filters


def _funding_with_forward_assumption(
    historical: pd.Series,
    *,
    forward_start: pd.Timestamp,
    last_candle: pd.Timestamp,
    interval_minutes: int,
    forward_rate: float,
) -> pd.Series:
    """Combine realized pre-cutoff funding with a clearly labeled forward assumption."""
    if not math.isfinite(forward_rate) or abs(forward_rate) > 0.05:
        raise ValueError("forward funding assumption must be finite and within +/-5%")
    if (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, int)
        or not 1 <= interval_minutes <= 480
    ):
        raise ValueError("funding interval must be an integer in [1, 480]")

    historical = historical.loc[historical.index < forward_start].copy()
    step = pd.Timedelta(minutes=interval_minutes)
    synthetic_index = pd.date_range(
        start=forward_start,
        end=last_candle,
        freq=step,
    )
    synthetic = pd.Series(
        forward_rate,
        index=synthetic_index,
        name="funding_rate",
        dtype=float,
    )
    combined = pd.concat([historical, synthetic]).sort_index()
    if combined.empty or combined.index.has_duplicates:
        raise ValueError("combined funding series is empty or contains duplicate events")

    historical_intervals = historical.attrs.get("funding_interval_minutes")
    if isinstance(historical_intervals, pd.Series):
        historical_intervals = historical_intervals.reindex(historical.index)
        if historical_intervals.isna().any():
            raise ValueError("historical funding interval metadata is incomplete")
        historical_intervals = historical_intervals.astype(int)
    else:
        historical_intervals = pd.Series(
            interval_minutes,
            index=historical.index,
            dtype=int,
        )
    synthetic_intervals = pd.Series(
        interval_minutes,
        index=synthetic.index,
        dtype=int,
    )
    combined.attrs["funding_interval_minutes"] = pd.concat(
        [historical_intervals, synthetic_intervals]
    ).sort_index()
    combined.attrs["forward_funding_assumption"] = forward_rate
    return combined


def run_forward_study(args: argparse.Namespace) -> dict:
    forward_start = _validate_args(args)
    # Daily Vision archives are published after the UTC day has closed.
    end = pd.Timestamp.now(tz="UTC").normalize()
    if end <= forward_start:
        raise ValueError("no completed daily archive data exists after forward-start")
    start = forward_start - pd.Timedelta(hours=4 * args.warmup_bars)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    client = BinanceVisionPublicClient()
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
            "no symbol passed Binance Vision archive validation: "
            + json.dumps(errors, sort_keys=True)
        )

    reports = {}
    for candidate_name, overrides in FORWARD_CANDIDATES.items():
        for leverage in args.leverage:
            for cost_name, costs in COST_SCENARIOS.items():
                for funding_name, funding_rate in FUNDING_SCENARIOS.items():
                    symbol_results = {}
                    for symbol, (
                        instrument,
                        traded,
                        mark,
                        historical_funding,
                        trade_start_index,
                    ) in aligned.items():
                        forward_funding = _funding_with_forward_assumption(
                            historical_funding,
                            forward_start=forward_start,
                            last_candle=traded.index[-1],
                            interval_minutes=instrument.funding_interval_minutes,
                            forward_rate=funding_rate,
                        )
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
                            forward_funding,
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
                    key = (
                        f"{candidate_name}/{leverage:g}x/{cost_name}/"
                        f"{funding_name}"
                    )
                    reports[key] = {
                        "candidate": candidate_name,
                        "parameters": overrides,
                        "leverage": leverage,
                        "costs": {
                            "commission_bps": costs.commission_bps,
                            "half_spread_bps": costs.half_spread_bps,
                            "slippage_bps": costs.slippage_bps,
                        },
                        "forward_funding_assumption_per_settlement": funding_rate,
                        "funding_settlement_interval_minutes": 480,
                        "symbols": symbol_results,
                    }

    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "credentials_used": False,
        "demo_ready": False,
        "venue": "Binance USD-M historical public archive (Binance Vision)",
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
        "research_instrument_assumptions": instruments,
        "symbol_data_errors": errors,
        "parameters_locked_before_forward_run": True,
        "forward_window_used_for_optimization": False,
        "funding_method": (
            "Realized Binance Vision funding is used before the forward cutoff. "
            "After the cutoff, funding is not yet published as a complete monthly "
            "archive and is modeled separately at 0, +1bp/8h, and -1bp/8h."
        ),
        "reports": reports,
        "limitations": [
            "Binance Vision is historical public archive data, not authenticated execution.",
            "Venue quantity, tick and minimum-notional filters are research placeholders; executable sizing is unverified.",
            "Current account-specific leverage brackets are not queried.",
            "Forward funding after the cutoff is assumed, not realized.",
            "The forward window may contain too few trades for statistical confidence.",
            "Fees, spread and slippage are scenario estimates, not measured account fills.",
            "Results are a cross-venue proxy if live execution is intended on another exchange.",
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
        "symbols": list(report["research_instrument_assumptions"]),
        "symbol_data_errors": report["symbol_data_errors"],
        "parameters_locked_before_forward_run": True,
        "orders_submitted": False,
        "funding_method": report["funding_method"],
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
