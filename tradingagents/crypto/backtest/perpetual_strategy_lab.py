"""Development-only tournament for fixed linear-perpetual strategy candidates.

The tournament deliberately excludes a previously inspected final holdout.
It compares predeclared candidates on disjoint chronological development folds
and never labels a development winner as independently validated.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from tradingagents.crypto.backtest.perpetual_ema_walkforward import (
    COST_SCENARIOS,
    DEFAULT_SYMBOLS,
    _fetch_history,
    _run_window,
    _summarize_development,
)
from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient

# Fixed before this tournament; this is a small family comparison, not a
# parameter sweep over the historical holdout.
# The original EMA14/EMA50/EMA200/ADX18 candidate is the same fixed
# specification that already failed the prior independent final holdout at
# both 1x and 2x under realistic, conservative and stress costs. Keep it in the
# development comparison for transparency, but never select it again.
PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES = {
    "ema_14_50_adx18": (
        "Same EMA14/EMA50/EMA200/ADX18 strategy failed the previously inspected "
        "independent final holdout at both 1x and 2x across all three cost models."
    )
}


def _rank_candidates(rankings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Annotate historical disqualifications before selecting any development winner."""
    annotated = []
    for item in rankings:
        row = dict(item)
        reason = PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES.get(row["candidate"])
        row["previous_holdout_rejected"] = reason is not None
        row["previous_holdout_rejection_reason"] = reason
        row["selection_eligible"] = (
            bool(row.get("development_gate_passed_all_costs")) and reason is None
        )
        annotated.append(row)
    eligible = [row for row in annotated if row["selection_eligible"]]
    eligible.sort(
        key=lambda row: (
            row.get("minimum_scenario_profit_factor") or -math.inf,
            row.get("conservative_return_pct") or -math.inf,
        ),
        reverse=True,
    )
    annotated.sort(
        key=lambda row: (
            row["selection_eligible"],
            row.get("minimum_scenario_profit_factor") or -math.inf,
            row.get("conservative_return_pct") or -math.inf,
        ),
        reverse=True,
    )
    return annotated, eligible


CANDIDATES: dict[str, dict] = {
    "ema_14_50_adx18": {
        "signal_mode": "ema_cross", "fast": 14, "slow": 50, "trend": 200,
        "min_adx": 18.0, "atr_multiple": 2.5, "max_holding_bars": 180,
    },
    "ema_9_30_adx22": {
        "signal_mode": "ema_cross", "fast": 9, "slow": 30, "trend": 200,
        "min_adx": 22.0, "atr_multiple": 3.0, "max_holding_bars": 144,
    },
    "ema_20_60_adx16": {
        "signal_mode": "ema_cross", "fast": 20, "slow": 60, "trend": 200,
        "min_adx": 16.0, "atr_multiple": 2.0, "max_holding_bars": 240,
    },
    "donchian_20_adx18": {
        "signal_mode": "donchian_breakout", "breakout_period": 20, "trend": 200,
        "min_adx": 18.0, "atr_multiple": 2.5, "max_holding_bars": 180,
    },
    "donchian_55_adx18": {
        "signal_mode": "donchian_breakout", "breakout_period": 55, "trend": 200,
        "min_adx": 18.0, "atr_multiple": 3.0, "max_holding_bars": 240,
    },
    "selective_bollinger_rsi_range": {
        "signal_mode": "bollinger_reversion",
        "fast": 14, "slow": 50, "trend": 200, "min_adx": None,
        "mean_reversion_period": 20, "mean_reversion_std": 2.0,
        "mean_reversion_rsi_period": 14, "mean_reversion_rsi_long": 32.0,
        "mean_reversion_rsi_short": 68.0, "mean_reversion_max_adx": 18.0,
        "atr_multiple": 2.0, "max_holding_bars": 72,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--equity", type=float, default=1_000.0)
    parser.add_argument("--leverage", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument("--test-bars", type=int, default=200)
    parser.add_argument("--exclude-after", required=True, help="ISO timestamp: first bar of the previously inspected holdout")
    parser.add_argument("--output", type=Path, default=Path("results/perpetual_strategy_lab.json"))
    return parser


def _profit_factor_pass(value, minimum: float) -> bool:
    return value == "infinite" or (
        isinstance(value, (float, int)) and math.isfinite(float(value)) and value >= minimum
    )


def acceptance_gate(development: dict, scenario: str) -> dict:
    """A conservative development-screen gate; not final OOS approval."""
    thresholds = {
        "realistic_estimate": 1.20,
        "conservative": 1.05,
        "stress": 1.00,
    }
    pf_min = thresholds[scenario]
    folds = int(development.get("fold_count") or 0)
    positive_folds = int(development.get("positive_folds") or 0)
    symbols = int(development.get("symbol_count") or 0)
    positive_symbols = int(development.get("positive_symbols") or 0)
    trades = int(development.get("trades") or 0)
    ret = development.get("equal_weight_compounded_return_pct")
    gate = {
        "minimum_trade_count": trades >= 30,
        "profit_factor": _profit_factor_pass(development.get("profit_factor"), pf_min),
        "positive_compounded_return": isinstance(ret, (int, float)) and ret > 0,
        "positive_fold_ratio": folds > 0 and positive_folds / folds >= 0.60,
        "positive_symbol_breadth": symbols > 0 and positive_symbols / symbols >= 0.75,
        "drawdown": float(development.get("worst_fold_drawdown_pct") or 0.0) <= 10.0,
        "no_liquidations": int(development.get("liquidations") or 0) == 0,
    }
    gate["passed"] = all(gate.values())
    return gate


def _validate(args: argparse.Namespace) -> pd.Timestamp:
    if isinstance(args.days, bool) or not isinstance(args.days, int) or not 180 <= args.days <= 3_650:
        raise ValueError("days must be an integer in [180, 3650]")
    if not math.isfinite(args.equity) or args.equity <= 0:
        raise ValueError("equity must be finite and positive")
    if isinstance(args.warmup_bars, bool) or args.warmup_bars < 220:
        raise ValueError("warmup-bars must be at least 220")
    if isinstance(args.test_bars, bool) or args.test_bars < 100:
        raise ValueError("test-bars must be at least 100")
    if not args.symbols or len(set(args.symbols)) != len(args.symbols):
        raise ValueError("symbols must be a non-empty unique list")
    if not args.leverage or any(
        isinstance(v, bool) or not math.isfinite(v) or v not in (1.0, 2.0)
        for v in args.leverage
    ):
        raise ValueError("only 1x and 2x research leverage are permitted")
    cutoff = pd.Timestamp(args.exclude_after)
    if cutoff.tzinfo is None:
        raise ValueError("exclude-after must include a timezone")
    return cutoff.tz_convert("UTC")


def run_lab(args: argparse.Namespace) -> dict:
    """Run only on development history strictly before the consumed holdout."""
    cutoff = _validate(args)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400_000
    client = BybitLinearPublicClient()
    history = {
        symbol: _fetch_history(client, symbol, start_ms, end_ms)
        for symbol in args.symbols
    }

    common = None
    for _, traded, _mark, _funding in history.values():
        common = traded.index if common is None else common.intersection(traded.index)
    if common is None:
        raise ValueError("no common Bybit history")
    common = common.sort_values()
    if not common.is_unique or not common.is_monotonic_increasing:
        raise ValueError("common timestamps are not unique and ordered")
    if not (common.to_series().diff().dropna() == pd.Timedelta(hours=4)).all():
        raise ValueError("common Bybit history contains missing 4-hour intervals")

    # Strictly exclude the already-inspected holdout and all later bars.
    development_index = common[common < cutoff]
    if len(development_index) < args.warmup_bars + 2 * args.test_bars:
        raise ValueError("insufficient history strictly before the previously inspected holdout")

    aligned = {}
    for symbol, (instrument, traded, mark, funding) in history.items():
        traded_dev = traded.loc[development_index].copy()
        mark_dev = mark.loc[development_index].copy()
        funding_dev = funding.loc[
            (funding.index >= development_index[0]) & (funding.index <= development_index[-1])
        ].copy()
        aligned[symbol] = (instrument, traded_dev, mark_dev, funding_dev)

    folds = []
    cursor = args.warmup_bars
    fold_id = 0
    while cursor + args.test_bars <= len(development_index):
        folds.append((fold_id, cursor, cursor + args.test_bars))
        fold_id += 1
        cursor += args.test_bars
    if len(folds) < 2:
        raise ValueError("at least two disjoint development folds are required")

    reports: dict[str, dict] = {}
    rankings = []
    for candidate_name, overrides in CANDIDATES.items():
        for leverage in args.leverage:
            by_scenario = {}
            for scenario_name, costs in COST_SCENARIOS.items():
                records = []
                for symbol, (instrument, traded, mark, funding) in aligned.items():
                    for fold, start, end in folds:
                        result = _run_window(
                            traded,
                            mark,
                            funding,
                            instrument,
                            start=start,
                            end=end,
                            warmup_bars=args.warmup_bars,
                            equity=args.equity,
                            leverage=leverage,
                            costs=costs,
                            strategy_overrides=overrides,
                        )
                        records.append(
                            {
                                "symbol": symbol,
                                "fold": fold,
                                "start": development_index[start].isoformat(),
                                "end": development_index[end - 1].isoformat(),
                                "result": result,
                            }
                        )
                summary = _summarize_development(records, list(aligned))
                gate = acceptance_gate(summary, scenario_name)
                by_scenario[scenario_name] = {"metrics": summary, "gate": gate}
                reports[f"{candidate_name}/{leverage:g}x/{scenario_name}"] = {
                    "candidate": candidate_name,
                    "parameters": overrides,
                    "leverage": leverage,
                    "costs": {
                        "commission_bps": costs.commission_bps,
                        "half_spread_bps": costs.half_spread_bps,
                        "slippage_bps": costs.slippage_bps,
                    },
                    "development_only": summary,
                    "acceptance_gate": gate,
                }
            passed = all(item["gate"]["passed"] for item in by_scenario.values())
            pfs = [
                item["metrics"]["profit_factor"]
                for item in by_scenario.values()
            ]
            numeric_pfs = [
                100.0 if value == "infinite" else float(value)
                for value in pfs
                if value == "infinite" or isinstance(value, (int, float))
            ]
            robust_pf = min(numeric_pfs) if numeric_pfs else None
            conservative_return = by_scenario["conservative"]["metrics"]["equal_weight_compounded_return_pct"]
            rankings.append(
                {
                    "candidate": candidate_name,
                    "leverage": leverage,
                    "development_gate_passed_all_costs": passed,
                    "minimum_scenario_profit_factor": robust_pf,
                    "conservative_return_pct": conservative_return,
                }
            )

    rankings, eligible = _rank_candidates(rankings)
    return {
        "schema_version": 1,
        "research_only": True,
        "orders_submitted": False,
        "demo_ready": False,
        "venue": "Bybit V5 public linear USDT perpetual market data",
        "timeframe": "4h",
        "symbols": list(aligned),
        "history_start": development_index[0].isoformat(),
        "development_end_exclusive": cutoff.isoformat(),
        "development_candles": len(development_index),
        "development_folds": len(folds),
        "warmup_bars": args.warmup_bars,
        "test_bars": args.test_bars,
        "excluded_previous_holdout_start": cutoff.isoformat(),
        "previously_inspected_holdout_reused": False,
        "final_holdout_evaluated": False,
        "candidate_definitions_fixed_before_run": True,
        "selection_rule": (
            "A candidate is only shortlisted when every realistic, conservative and "
            "stress development gate passes and it has not already failed a previously "
            "inspected independent final holdout. No eligible candidate means no selection."
        ),
        "previous_holdout_disqualifications": PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES,
        "best_candidate": eligible[0] if eligible else None,
        "rankings": rankings,
        "reports": reports,
        "limitations": [
            "This is a development-only candidate screen, not independent final validation.",
            "The previously inspected holdout and every bar after its start are excluded from strategy scoring.",
            "Fees, spread and slippage are estimates, not account-specific measured Bybit fills.",
            "Liquidation prices remain approximations, not Bybit risk-tier calculations.",
            "No authenticated account, leverage-setting or order API is used.",
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_lab(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "development_folds": report["development_folds"],
        "development_end_exclusive": report["development_end_exclusive"],
        "best_candidate": report["best_candidate"],
        "rankings": report["rankings"],
        "final_holdout_evaluated": False,
        "orders_submitted": False,
        "demo_ready": False,
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
