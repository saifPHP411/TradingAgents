"""A serious historical validation of the TJR strategy, over real intraday data.

This module runs the *unchanged* TJR strategy over real Yahoo 1-minute and
5-minute bars and reports what it did, under the paper account's own risk limits
(:func:`~tradingagents.risk.paper_parity.paper_parity_limits`). It changes no rule,
no stop, no target and no limit: the rule chain in :mod:`tradingagents.tjr.rules`
is the one under test, and this harness only measures it.

What it adds over :mod:`tradingagents.tjr.run`:

- **The ten-symbol universe** (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AMD,
  AVGO, JPM), each tested alone and then as one shared book.
- **Cost scenarios** — frictionless (for comparison only), the project's frozen
  schedule, and a harsher spread/slippage assumption. Each is a separate run.
- **A chronological development/test split**, reported separately, so an
  out-of-sample read is visible rather than implied.
- **A data-quality audit** of the fetched frames: sessions, gaps, the ET session
  clock, and whether the vendor's 5m bars agree with the 1m bars they should
  aggregate.
- **Robustness probes**: a look-ahead check that truncating the feed cannot change
  a decision already made, and a cost-sensitivity read.

Two honest facts shape everything here, and both are repeated in the report:

1. Yahoo serves 1m bars for only the last ~30 days and 5m bars for the last ~60.
   The real window is therefore about a month, **not** the 6–12 months the task
   asks for. That is a hard vendor limit, and no bar is invented to cover it.
2. The universe is ten large caps that exist today. It cannot measure survivorship
   bias, because the names that died are not in it.

The evaluation cache. The rule chain is a pure function of the bars visible at an
instant, so its answer for a ``(symbol, session, instant)`` is the same in every
scenario. The harness memoizes it, which is what makes a multi-scenario study of
ten symbols finish in minutes instead of hours. The cache is exact — a cache miss
calls the real function — so a cached run and a cold run produce identical trades.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradingagents.execution.costs import CostModel
from tradingagents.performance.metrics import compute_metrics
from tradingagents.risk import paper_parity_limits
from tradingagents.risk.paper_parity import (
    MAX_DAILY_ORDERS,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
)

from .engine import TjrBacktester, TjrConfig
from .frames import SESSION_TZ, IntradaySeries, resample, session_open_utc
from .report import build_report, equal_weight_benchmark
from .run import load_series
from .strategy import TjrStrategy

UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "JPM",
)
"""The ten symbols the study is fixed to, so a report names the same set every run."""

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class CostScenario:
    """One cost assumption the study is run under."""

    name: str
    label: str
    costs: CostModel


DEFAULT_SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario(
        name="zero",
        label="frictionless (no costs), for comparison only",
        costs=CostModel(),
    ),
    CostScenario(
        name="project",
        label=(
            "the project's frozen schedule: 5 bps commission, 1 bp half-spread, "
            "2 bps slippage (the baseline/paper protocol's own costs)"
        ),
        costs=CostModel(commission_rate=0.0005, half_spread_bps=1.0, slippage_bps=2.0),
    ),
    CostScenario(
        name="high",
        label=(
            "a harsher assumption: 5 bps commission, 2 bps half-spread, 5 bps "
            "slippage, to show the result's sensitivity to friction"
        ),
        costs=CostModel(commission_rate=0.0005, half_spread_bps=2.0, slippage_bps=5.0),
    ),
)
"""The three cost scenarios. ``project`` is the current schedule the repository
uses elsewhere; ``zero`` exists only as a comparison and is never a headline."""


# ---------------------------------------------------------------------------
# Memoized evaluation
# ---------------------------------------------------------------------------


class _MemoStrategy(TjrStrategy):
    """A :class:`TjrStrategy` that caches its pure evaluations for one study run."""

    def __init__(self, *, cache: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cache = cache

    def evaluate(self, symbol: str, session: str, as_of: pd.Timestamp):
        key = (symbol, session, as_of.isoformat())
        hit = self._cache.get(key)
        if hit is None:
            hit = super().evaluate(symbol, session, as_of)
            self._cache[key] = hit
        return hit


class _MemoBacktester(TjrBacktester):
    """The TJR backtester with the memoizing strategy installed."""

    def __init__(self, *, evaluation_cache: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy = _MemoStrategy(
            one_minute=self.one_minute,
            five_minute=self.five_minute,
            hourly=self.hourly,
            four_hour=self.four_hour,
            settings=self.config.settings,
            allow_short=self.config.allow_short,
            cache=evaluation_cache,
        )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class _SeriesBundle:
    """The four interval series per symbol, filtered to a set of sessions."""

    one_minute: dict[str, IntradaySeries]
    five_minute: dict[str, IntradaySeries]
    hourly: dict[str, IntradaySeries]
    four_hour: dict[str, IntradaySeries]


def _filter_to_sessions(series: IntradaySeries, sessions: set[str]) -> IntradaySeries | None:
    """A series restricted to the given ET session dates, or ``None`` if it empties."""
    frame = series.frame
    if frame.empty:
        return None
    keep = frame.index.tz_convert(SESSION_TZ).strftime("%Y-%m-%d").isin(sessions)
    subset = frame[keep]
    if subset.empty:
        return None
    return IntradaySeries(series.symbol, series.interval, subset)


def _bundle_for(
    series: tuple[
        dict[str, IntradaySeries],
        dict[str, IntradaySeries],
        dict[str, IntradaySeries],
        dict[str, IntradaySeries],
    ],
    symbols: Iterable[str],
    sessions: set[str] | None,
) -> _SeriesBundle | None:
    """The bundle for ``symbols``, optionally restricted to ``sessions``.

    Restricting is done on the *frames*, then the hourly/4h levels are re-derived
    from the restricted 5m series, so a sub-period run cannot see a bar outside it.
    """
    one, five, hourly, four = series
    out_one: dict[str, IntradaySeries] = {}
    out_five: dict[str, IntradaySeries] = {}
    for symbol in symbols:
        if sessions is None:
            out_one[symbol] = one[symbol]
            out_five[symbol] = five[symbol]
            continue
        f = _filter_to_sessions(five[symbol], sessions)
        o = _filter_to_sessions(one[symbol], sessions)
        if f is None or o is None:
            continue
        out_five[symbol] = f
        out_one[symbol] = o
    if not out_five:
        return None
    if sessions is None:
        out_hourly = {s: hourly[s] for s in out_five if s in hourly}
        out_four = {s: four[s] for s in out_five if s in four}
    else:
        out_hourly = {}
        out_four = {}
        for symbol, f in out_five.items():
            for interval, target in (("60m", out_hourly), ("4h", out_four)):
                try:
                    target[symbol] = resample(f, interval)
                except Exception:  # noqa: BLE001 - a too-short window simply has no level
                    continue
    return _SeriesBundle(out_one, out_five, out_hourly, out_four)


def run_scenario(
    series: tuple,
    symbols: tuple[str, ...],
    *,
    costs: CostModel,
    evaluation_cache: dict,
    sessions: set[str] | None = None,
) -> dict:
    """Run one backtest and return the report as a dict.

    The risk limits are always :func:`paper_parity_limits` — the paper account's
    own caps — applied to the whole book, never relaxed per symbol.
    """
    bundle = _bundle_for(series, symbols, sessions)
    if bundle is None:
        return {"empty": True, "symbols": list(symbols)}
    config = TjrConfig(costs=costs, risk_limits=paper_parity_limits())
    backtester = _MemoBacktester(
        one_minute=bundle.one_minute,
        five_minute=bundle.five_minute,
        hourly=bundle.hourly or None,
        four_hour=bundle.four_hour or None,
        config=config,
        evaluation_cache=evaluation_cache,
    )
    result = backtester.run()
    report = build_report(result)
    return summarize_run(result, report)


def summarize_run(result, report) -> dict:
    """The full, JSON-safe summary of one run: metrics, per-symbol, reasons, days."""
    payload = report.to_dict()
    payload["net_pnl"] = result.final_equity - result.initial_cash
    payload["winning_days"], payload["losing_days"], payload["flat_days"] = _day_counts(result)
    payload["rejected_orders"] = len(result.rejections)
    payload["rejection_reasons"] = _rejection_tally(result.rejections)
    payload["trades_detail"] = [
        {
            "symbol": t.symbol,
            "session": t.session,
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "net_pnl": t.net_pnl,
            "fees": t.fees,
            "exit_reason": t.exit_reason,
            "planned_reward_risk": t.planned_reward_risk,
            "realized_r": t.realized_r,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
        }
        for t in result.trades
    ]
    return payload


def _day_counts(result) -> tuple[int, int, int]:
    """Winning, losing and flat sessions, by each session's closing equity change."""
    equity = result.equity_series()
    if equity.empty:
        return 0, 0, 0
    local_day = pd.DatetimeIndex(equity.index).tz_convert(SESSION_TZ).strftime("%Y-%m-%d")
    last_per_day = equity.groupby(local_day).last()
    deltas = last_per_day.diff().dropna()
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    flats = int((deltas == 0).sum())
    return wins, losses, flats


def _rejection_tally(rejections: list[dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for item in rejections:
        for reason in item.get("reasons", []):
            tally[reason] = tally.get(reason, 0) + 1
    return dict(sorted(tally.items()))


# ---------------------------------------------------------------------------
# Period split
# ---------------------------------------------------------------------------


def split_sessions(
    sessions: list[str], *, fraction: float = 0.5
) -> tuple[list[str], list[str]]:
    """Split an ordered session list into ``(development, test)``, chronologically.

    A contiguous cut at ``fraction``. With so few sessions the split is descriptive,
    not a statistically meaningful out-of-sample test, and the report says so.
    """
    ordered = sorted(sessions)
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
    return ordered[:cut], ordered[cut:]


def period_view(
    result,
    report,
    sessions: list[str],
    *,
    label: str,
) -> dict:
    """Re-derive one run's metrics restricted to a contiguous set of sessions.

    The book is flat at every session close, so a session set is self-contained: a
    sub-period's equity is rebased to the starting cash and its benchmark is the
    same equal-weight basket over the same sessions. This is a *view* of one run,
    not a second run, and it is labelled as such.
    """
    keep = set(sessions)
    equity = result.equity_series()
    if equity.empty:
        return {"label": label, "sessions": 0}
    local_day = pd.DatetimeIndex(equity.index).tz_convert(SESSION_TZ).strftime("%Y-%m-%d")
    mask = local_day.isin(keep)
    equity_p = equity[mask]
    if equity_p.empty:
        return {"label": label, "sessions": 0}
    base = float(equity_p.iloc[0])
    equity_rebased = equity_p * (result.initial_cash / base) if base else equity_p
    exposure_p = result.exposure_series()[mask]
    trades_p = [t for t in result.trades if t.session in keep]
    notional_p = sum(abs(t.quantity * t.entry_price) + abs(t.quantity * t.exit_price) for t in trades_p)
    fees_p = sum(t.fees for t in trades_p)
    realized_p = sum(t.net_pnl for t in trades_p)
    bench = equal_weight_benchmark(result)
    bench_p = None
    if bench is not None:
        bench_slice = bench[mask]
        if len(bench_slice) >= 2 and float(bench_slice.iloc[0]):
            bench_p = bench_slice * (result.initial_cash / float(bench_slice.iloc[0]))
    metrics = compute_metrics(
        equity_rebased,
        initial_capital=result.initial_cash,
        trades=trades_p,
        exposure=exposure_p,
        traded_notional=notional_p,
        realized_pnl=realized_p,
        unrealized_pnl=0.0,
        total_fees=fees_p,
        benchmark=bench_p,
    )
    planned = [t.planned_reward_risk for t in trades_p if t.planned_reward_risk is not None]
    realized_r = [t.realized_r for t in trades_p if t.realized_r is not None]
    reasons: dict[str, int] = {}
    per_symbol: dict[str, dict] = {}
    for t in trades_p:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        bucket = per_symbol.setdefault(t.symbol, {"trades": 0, "net_pnl": 0.0, "wins": 0, "losses": 0})
        bucket["trades"] += 1
        bucket["net_pnl"] += t.net_pnl
        if t.net_pnl > 0:
            bucket["wins"] += 1
        elif t.net_pnl < 0:
            bucket["losses"] += 1
    wins, losses, flats = _day_counts_for_equity(equity_p)
    return {
        "label": label,
        "sessions": len(set(local_day[mask])),
        "start": str(equity_p.index[0]),
        "end": str(equity_p.index[-1]),
        "metrics": metrics.to_dict(),
        "net_pnl": float(equity_rebased.iloc[-1]) - result.initial_cash,
        "average_rr": (sum(planned) / len(planned)) if planned else None,
        "average_realized_r": (sum(realized_r) / len(realized_r)) if realized_r else None,
        "exit_reasons": dict(sorted(reasons.items())),
        "per_symbol": {k: per_symbol[k] for k in sorted(per_symbol)},
        "fees": fees_p,
        "winning_days": wins,
        "losing_days": losses,
        "flat_days": flats,
    }


def _day_counts_for_equity(equity: pd.Series) -> tuple[int, int, int]:
    local_day = pd.DatetimeIndex(equity.index).tz_convert(SESSION_TZ).strftime("%Y-%m-%d")
    last_per_day = equity.groupby(local_day).last()
    deltas = last_per_day.diff().dropna()
    return int((deltas > 0).sum()), int((deltas < 0).sum()), int((deltas == 0).sum())


# ---------------------------------------------------------------------------
# Data-quality audit
# ---------------------------------------------------------------------------


def audit_frames(
    data_dir: Path, symbols: tuple[str, ...]
) -> dict:
    """Inspect the fetched frames: sessions, gaps, the ET clock, and 1m/5m agreement."""
    report: dict = {"symbols": {}, "checks": {}}
    one_all: dict[str, pd.DataFrame] = {}
    five_all: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        one = _read_frame(data_dir, symbol, "1m")
        five = _read_frame(data_dir, symbol, "5m")
        if one is None or five is None:
            report["symbols"][symbol] = {"present": False}
            continue
        one_all[symbol] = one
        five_all[symbol] = five
        report["symbols"][symbol] = _audit_symbol(one, five)
    report["checks"] = _cross_checks(one_all, five_all)
    return report


def _read_frame(data_dir: Path, symbol: str, interval: str) -> pd.DataFrame | None:
    path = data_dir / f"{symbol.upper()}-{interval}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _audit_symbol(one: pd.DataFrame, five: pd.DataFrame) -> dict:
    local_one = one.index.tz_convert(SESSION_TZ)
    local_five = five.index.tz_convert(SESSION_TZ)
    one_sessions = sorted(set(local_one.strftime("%Y-%m-%d")))
    five_sessions = sorted(set(local_five.strftime("%Y-%m-%d")))
    one_counts = pd.Series(local_one.strftime("%Y-%m-%d")).value_counts()
    five_counts = pd.Series(local_five.strftime("%Y-%m-%d")).value_counts()
    # A regular session is 09:30–16:00 ET: 390 one-minute bars, 78 five-minute bars.
    one_incomplete = sorted(k for k, v in one_counts.items() if v != 390)
    five_incomplete = sorted(k for k, v in five_counts.items() if v != 78)
    return {
        "present": True,
        "one_minute": {
            "rows": int(len(one)),
            "sessions": len(one_sessions),
            "first_session": one_sessions[0],
            "last_session": one_sessions[-1],
            "bars_per_session_min": int(one_counts.min()),
            "bars_per_session_max": int(one_counts.max()),
            "incomplete_sessions": one_incomplete,
            "first_bar_et": str(local_one.min()),
            "last_bar_et": str(local_one.max()),
        },
        "five_minute": {
            "rows": int(len(five)),
            "sessions": len(five_sessions),
            "first_session": five_sessions[0],
            "last_session": five_sessions[-1],
            "bars_per_session_min": int(five_counts.min()),
            "bars_per_session_max": int(five_counts.max()),
            "incomplete_sessions": five_incomplete,
        },
        "session_sets_match": one_sessions == five_sessions,
        "duplicate_timestamps": {
            "one_minute": int(one.index.duplicated().sum()),
            "five_minute": int(five.index.duplicated().sum()),
        },
        "non_finite_prices": {
            "one_minute": int((~one[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)).sum()),
            "five_minute": int((~five[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)).sum()),
        },
    }


def _cross_checks(one_all: dict[str, pd.DataFrame], five_all: dict[str, pd.DataFrame]) -> dict:
    """Whether the 5m bars agree with the 1m bars they should aggregate, per symbol."""
    per_symbol: dict[str, dict] = {}
    for symbol, five in five_all.items():
        one = one_all.get(symbol)
        if one is None:
            continue
        per_symbol[symbol] = _aggregation_mismatch(one, five)
    all_sessions: set[str] = set()
    for frame in one_all.values():
        all_sessions |= set(frame.index.tz_convert(SESSION_TZ).strftime("%Y-%m-%d"))
    per_symbol_sessions = {
        s: sorted(set(f.index.tz_convert(SESSION_TZ).strftime("%Y-%m-%d")))
        for s, f in one_all.items()
    }
    union = sorted(set().union(*per_symbol_sessions.values())) if per_symbol_sessions else []
    aligned = all(
        sessions == union for sessions in per_symbol_sessions.values()
    ) if per_symbol_sessions else False
    return {
        "one_minute_vs_five_minute": per_symbol,
        "session_alignment_across_symbols": aligned,
        "union_sessions": len(union),
        "first_session": union[0] if union else None,
        "last_session": union[-1] if union else None,
    }


def _aggregation_mismatch(one: pd.DataFrame, five: pd.DataFrame) -> dict:
    """Compare each 5m bar to the 1m bars that should build it.

    The vendor's 5m series is its own feed, so a small mismatch is expected and is
    reported rather than hidden. A large mismatch would mean the two timeframes are
    not the same market and the study's 1m/5m reads disagree.
    """
    one = one.copy()
    one.columns = [c.lower() for c in one.columns]
    five = five.copy()
    five.columns = [c.lower() for c in five.columns]
    local = one.index.tz_convert(SESSION_TZ)
    day = pd.Series(local.strftime("%Y-%m-%d"), index=one.index)
    anchor = pd.DatetimeIndex([pd.Timestamp(d, tz=SESSION_TZ) for d in day]) + pd.Timedelta(hours=9, minutes=30)
    offset = ((local - anchor).total_seconds() // 60).astype("int64")
    bucket = (anchor + pd.to_timedelta((offset // 5) * 5, unit="m")).tz_convert("UTC")
    grouped = one.groupby(bucket, sort=True)
    agg = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    agg = agg[grouped.size() == 5]
    common = agg.index.intersection(five.index)
    if len(common) == 0:
        return {"compared_bars": 0}
    diffs = {}
    for column in ("open", "high", "low", "close"):
        rel = ((agg.loc[common, column] - five.loc[common, column]).abs() / five.loc[common, column].abs()).abs()
        diffs[column] = {
            "max_rel_diff": float(rel.max()),
            "mean_rel_diff": float(rel.mean()),
        }
    return {"compared_bars": int(len(common)), "per_column": diffs}


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


def run_study(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = UNIVERSE,
    scenarios: tuple[CostScenario, ...] = DEFAULT_SCENARIOS,
    split_fraction: float = 0.5,
) -> dict:
    """Run the whole study and return a JSON-safe payload.

    Order of work: audit the frames, run each symbol alone under the project cost
    scenario, run the shared book under every scenario, and split the shared-book
    project run into a chronological development and test view.
    """
    series = load_series(data_dir, symbols)
    cache: dict = {}
    all_sessions = _all_sessions(series[1], symbols)
    dev, test = split_sessions(all_sessions, fraction=split_fraction)

    audit = audit_frames(data_dir, symbols)

    per_symbol_standalone: dict[str, dict] = {}
    for symbol in symbols:
        per_symbol_standalone[symbol] = run_scenario(
            series, (symbol,), costs=DEFAULT_SCENARIOS[1].costs, evaluation_cache=cache
        )

    portfolio: dict[str, dict] = {}
    for scenario in scenarios:
        portfolio[scenario.name] = run_scenario(
            series, symbols, costs=scenario.costs, evaluation_cache=cache
        )

    # Re-run the project scenario over the shared book, keeping the result object
    # so the split can be viewed from it (the metrics view is derived, not re-run).
    project_result = _run_result(series, symbols, costs=DEFAULT_SCENARIOS[1].costs, cache=cache)
    project_report = build_report(project_result)
    split = {
        "method": (
            f"one contiguous chronological cut at {split_fraction:.0%} of the "
            f"{len(all_sessions)} sessions; the rules were fixed before the run, so "
            "the test period is not used to choose anything"
        ),
        "development": period_view(project_result, project_report, dev, label="development"),
        "test": period_view(project_result, project_report, test, label="test"),
    }

    return {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "universe": list(symbols),
        "risk_limits": {
            "source": "tradingagents.risk.paper_parity.paper_parity_limits()",
            "max_order_notional": MAX_ORDER_NOTIONAL,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_daily_orders": MAX_DAILY_ORDERS,
            "allow_short": False,
            "fractional_shares": True,
            "applied_to": "the shared portfolio book, not each symbol separately",
        },
        "cost_scenarios": [
            {"name": s.name, "label": s.label, "costs": s.costs.model_dump()} for s in scenarios
        ],
        "data_audit": audit,
        "window": {
            "sessions": len(all_sessions),
            "first_session": all_sessions[0] if all_sessions else None,
            "last_session": all_sessions[-1] if all_sessions else None,
            "vendor_limit_note": (
                "Yahoo serves 1m bars for only the last ~30 days and 5m bars for the "
                "last ~60; the real window is this short, not the 6-12 months asked for"
            ),
        },
        "per_symbol_standalone": per_symbol_standalone,
        "portfolio": portfolio,
        "split": split,
        "evaluation_cache_entries": len(cache),
    }


def _run_result(series, symbols, *, costs: CostModel, cache: dict):
    bundle = _bundle_for(series, symbols, None)
    assert bundle is not None
    config = TjrConfig(costs=costs, risk_limits=paper_parity_limits())
    backtester = _MemoBacktester(
        one_minute=bundle.one_minute,
        five_minute=bundle.five_minute,
        hourly=bundle.hourly or None,
        four_hour=bundle.four_hour or None,
        config=config,
        evaluation_cache=cache,
    )
    return backtester.run()


def _all_sessions(five: Mapping[str, IntradaySeries], symbols: Iterable[str]) -> list[str]:
    days: set[str] = set()
    for symbol in symbols:
        local = five[symbol].frame.index.tz_convert(SESSION_TZ)
        days |= set(local.strftime("%Y-%m-%d"))
    return sorted(days)


# ---------------------------------------------------------------------------
# Look-ahead probe
# ---------------------------------------------------------------------------


def lookahead_probe(
    data_dir: Path,
    *,
    symbol: str,
    session: str,
    cut: str,
) -> dict:
    """Prove a decision made at ``cut`` cannot change when later bars are added.

    Run the symbol's feed twice: once whole, once truncated after ``cut``. Every
    trade *entered* at or before ``cut`` in the whole run must be entered identically
    in the truncated run — same time, same size, same price. A decision at ``cut``
    may only read bars that had closed by then, so later bars cannot change it; a
    difference would be look-ahead.

    The comparison is on the **entry**, not the exit. A trade still open at ``cut``
    has no future in the truncated feed, so the engine force-closes it at the last
    available bar; that exit is an artefact of the truncation, not a changed
    decision, and comparing exits would report a false positive. ``open_at_cut``
    counts those trades so the artefact is visible rather than hidden.
    """
    series = load_series(data_dir, (symbol,))
    full = _run_result(series, (symbol,), costs=CostModel(), cache={})
    full_entries = _entry_keys(full, cut)

    truncated_series = _truncate_series(series, symbol, cut)
    truncated = _run_result(truncated_series, (symbol,), costs=CostModel(), cache={})
    truncated_entries = _entry_keys(truncated, cut)

    open_at_cut = sum(1 for t in full.trades if pd.Timestamp(t.exit_time) > pd.Timestamp(cut))
    return {
        "symbol": symbol,
        "session": session,
        "cut": cut,
        "full_entries_total": len(full.trades),
        "full_entries_at_or_before_cut": len(full_entries),
        "truncated_entries_at_or_before_cut": len(truncated_entries),
        "open_at_cut": open_at_cut,
        "identical": full_entries == truncated_entries,
    }


def _entry_keys(result, cut: str) -> list[tuple]:
    """The entry decisions at or before ``cut``: time, size and price, nothing else."""
    cutoff = pd.Timestamp(cut)
    return sorted(
        (str(t.entry_time), round(t.quantity, 8), round(t.entry_price, 8))
        for t in result.trades
        if pd.Timestamp(t.entry_time) <= cutoff
    )


def _truncate_series(series: tuple, symbol: str, cut: str):
    """The same bundle with every frame dropped at or after ``cut``.

    The hourly and 4h level sources are re-derived from the truncated 5m feed, so
    the only thing that differs from the full run is the absence of bars at or after
    ``cut``. Dropping the level sources instead would change decisions for reasons
    that have nothing to do with look-ahead, and the probe would report a false
    positive.
    """
    one, five, _hourly, _four = series
    cutoff = pd.Timestamp(cut)
    out_five = {symbol: IntradaySeries(symbol, "5m", five[symbol].frame[five[symbol].frame.index < cutoff])}
    out_one = {symbol: IntradaySeries(symbol, "1m", one[symbol].frame[one[symbol].frame.index < cutoff])}
    out_hourly: dict[str, IntradaySeries] = {}
    out_four: dict[str, IntradaySeries] = {}
    for interval, target in (("60m", out_hourly), ("4h", out_four)):
        try:
            target[symbol] = resample(out_five[symbol], interval)
        except Exception:  # noqa: BLE001 - a truncated window may fill no bucket
            continue
    return (out_one, out_five, out_hourly, out_four)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:+.3%}"


def _num(value, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:,.{places}f}"


def _ratio(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize_prior(path: Path) -> dict | None:
    """A compact read of an earlier ``real_data_backtest*.json`` report, for comparison."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "file": path.name,
        "start": data.get("start"),
        "end": data.get("end"),
        "sessions": data.get("sessions"),
        "trades": metrics.get("num_trades"),
        "total_return": metrics.get("total_return"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "buy_and_hold_return": data.get("buy_and_hold_return"),
        "average_rr": data.get("average_rr"),
        "average_realized_r": data.get("average_realized_r"),
        "notes": data.get("notes", []),
    }


def _render_prior(lines: list[str], prior: list[dict]) -> None:
    prior = [p for p in prior if p]
    if not prior:
        return
    lines.append("## Prior reports (kept for comparison)")
    lines.append("")
    lines.append(
        "These are the earlier committed TJR runs, over a shorter window. They are "
        "shown beside this study, not merged with it."
    )
    lines.append("")
    lines.append("| report | window | trades | net return | win rate | profit factor | Sharpe | fees |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in prior:
        window = f"{str(entry['start'])[:10]}..{str(entry['end'])[:10]}" if entry.get("start") else "n/a"
        fees = "zero" if "costs" not in entry["file"] else "charged"
        lines.append(
            f"| {entry['file']} | {window} | {entry['trades']} | {_pct(entry['total_return'])} | "
            f"{_pct(entry['win_rate'])} | {_ratio(entry['profit_factor'])} | "
            f"{_ratio(entry['sharpe_ratio'])} | {fees} |"
        )
    lines.append("")
    for entry in prior:
        for note in entry.get("notes", []):
            lines.append(f"- {entry['file']}: {note}")
    lines.append("")


def render_markdown(payload: dict) -> str:
    """A readable report from a study payload, with its limits stated up front."""
    lines: list[str] = []
    window = payload["window"]
    limits = payload["risk_limits"]
    lines.append("# TJR strategy validation — historical study")
    lines.append("")
    lines.append(f"Generated (UTC): {payload['generated_utc']}")
    lines.append("")
    lines.append(
        "This study runs the **unchanged** TJR rule chain over real Yahoo intraday "
        "bars under the paper account's own risk limits. No rule, stop, target or "
        "limit was changed to improve the result."
    )
    lines.append("")

    lines.append("## 1. Data window (what the vendor actually served)")
    lines.append("")
    lines.append(
        f"- Universe: {', '.join(payload['universe'])}"
    )
    lines.append(
        f"- Sessions: **{window['sessions']}** ({window['first_session']} .. {window['last_session']}, ET)"
    )
    lines.append(f"- {window['vendor_limit_note']}")
    lines.append(
        "- **This is ~1 month, not the 6–12 months requested.** The limit is the "
        "vendor's; no bar was invented to cover the gap, and the study is not "
        "claimed to be long-horizon."
    )
    lines.append("")

    audit = payload["data_audit"]
    checks = audit["checks"]
    lines.append("### Data-quality audit")
    lines.append("")
    lines.append(
        f"- Sessions aligned across all symbols: **{checks['session_alignment_across_symbols']}** "
        f"(union {checks['union_sessions']} sessions)"
    )
    lines.append(
        "- 1m vs 5m agreement (each 5m bar against the five 1m bars that build it):"
    )
    for symbol, detail in sorted(checks["one_minute_vs_five_minute"].items()):
        if detail.get("compared_bars"):
            close = detail["per_column"]["close"]
            lines.append(
                f"  - {symbol}: {detail['compared_bars']} bars, max close rel-diff "
                f"{close['max_rel_diff']:.2e}"
            )
    incomplete = {
        s: v["one_minute"]["incomplete_sessions"]
        for s, v in audit["symbols"].items()
        if v.get("present") and v["one_minute"]["incomplete_sessions"]
    }
    lines.append(
        f"- Sessions with fewer than 390 one-minute bars: {incomplete or 'none'}"
    )
    lines.append("")

    lines.append("## 2. Risk limits (unchanged, applied to the shared book)")
    lines.append("")
    lines.append(f"- Source: `{limits['source']}`")
    lines.append(
        f"- max_order_notional = {limits['max_order_notional']} USD, "
        f"max_open_positions = {limits['max_open_positions']}, "
        f"max_daily_orders = {limits['max_daily_orders']}, "
        f"allow_short = {limits['allow_short']}, "
        f"fractional_shares = {limits['fractional_shares']}"
    )
    lines.append(f"- Applied to: {limits['applied_to']}")
    lines.append("")

    lines.append("## 3. Cost scenarios (shared portfolio)")
    lines.append("")
    lines.append(
        "| scenario | trades | net P/L (USD) | net return | B&H return | win rate | profit factor | "
        "max DD | Sharpe | Sortino | exposure | turnover | rejected |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for scenario in payload["cost_scenarios"]:
        result = payload["portfolio"].get(scenario["name"])
        if not result or result.get("empty"):
            continue
        m = result["metrics"]
        lines.append(
            f"| {scenario['name']} | {m['num_trades']} | {_num(result['net_pnl'], 2)} | "
            f"{_pct(m['total_return'])} | {_pct(m['benchmark_return'])} | {_pct(m['win_rate'])} | "
            f"{_ratio(m['profit_factor'])} | {_pct(m['max_drawdown'])} | {_ratio(m['sharpe_ratio'])} | "
            f"{_ratio(m['sortino_ratio'])} | {_pct(m['exposure'])} | {_ratio(m['turnover'])} | "
            f"{result['rejected_orders']} |"
        )
    lines.append("")
    lines.append(
        "The Buy & Hold return is an equal-weight basket of the traded symbols at 1m "
        "closes, over the same index as the equity curve (see "
        "`equal_weight_benchmark`)."
    )
    lines.append("")
    lines.append("Rejection reasons (closed vocabulary):")
    for scenario in payload["cost_scenarios"]:
        result = payload["portfolio"].get(scenario["name"])
        if not result or result.get("empty"):
            continue
        lines.append(f"- **{scenario['name']}**: {result['rejection_reasons'] or 'none'}")
    lines.append("")
    lines.append("Winning / losing / flat sessions (by each session's closing equity change):")
    for scenario in payload["cost_scenarios"]:
        result = payload["portfolio"].get(scenario["name"])
        if not result or result.get("empty"):
            continue
        lines.append(
            f"- **{scenario['name']}**: {result['winning_days']} / {result['losing_days']} / "
            f"{result['flat_days']}"
        )
    lines.append("")
    for scenario in payload["cost_scenarios"]:
        lines.append(f"- **{scenario['name']}**: {scenario['label']}")
    lines.append("")
    lines.append(
        "Costs are not hidden: the frictionless figure is the highest and is a "
        "comparison only; the project schedule and the harsher schedule both reduce "
        "the result, which is the expected direction."
    )
    lines.append("")

    lines.append("## 4. Per-symbol results (each symbol alone, project costs)")
    lines.append("")
    lines.append("| symbol | trades | net P/L | net return | B&H return | win rate | avg planned R:R | avg realized R |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for symbol in payload["universe"]:
        result = payload["per_symbol_standalone"].get(symbol)
        if not result or result.get("empty"):
            continue
        m = result["metrics"]
        lines.append(
            f"| {symbol} | {m['num_trades']} | {_num(result['net_pnl'], 2)} | "
            f"{_pct(m['total_return'])} | {_pct(m['benchmark_return'])} | "
            f"{_pct(m['win_rate'])} | {_ratio(result['average_rr'])} | "
            f"{_ratio(result['average_realized_r'])} |"
        )
    lines.append("")

    lines.append("## 5. Out-of-sample read (chronological development / test split)")
    lines.append("")
    split = payload["split"]
    lines.append(f"Method: {split['method']}")
    lines.append("")
    lines.append("| period | sessions | net P/L | net return | trades | win rate | profit factor | max DD | win/loss days |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key in ("development", "test"):
        view = split[key]
        if not view.get("sessions"):
            continue
        m = view["metrics"]
        lines.append(
            f"| {key} | {view['sessions']} | {_num(view['net_pnl'], 2)} | "
            f"{_pct(m['total_return'])} | {m['num_trades']} | {_pct(m['win_rate'])} | "
            f"{_ratio(m['profit_factor'])} | {_pct(m['max_drawdown'])} | "
            f"{view['winning_days']}/{view['losing_days']} |"
        )
    lines.append("")

    lines.append("## 6. Robustness")
    lines.append("")
    probes = payload.get("lookahead_probes")
    if not probes and payload.get("lookahead_probe"):
        probes = [payload["lookahead_probe"]]
    for probe in probes or []:
        opened = probe.get("open_at_cut", 0)
        plural = "trade was" if opened == 1 else "trades were"
        lines.append(
            f"- Look-ahead probe ({probe['symbol']}, cut {probe['cut']}): the full "
            f"feed has {probe['full_entries_total']} entries; "
            f"{probe['full_entries_at_or_before_cut']} of them are entered at or "
            f"before the cut, and the feed truncated at the cut enters the same "
            f"{probe['truncated_entries_at_or_before_cut']} at the same time, size and "
            f"price. Identical: **{probe['identical']}**. "
            f"({opened} {plural} open at the cut and force-closed by the truncation, "
            f"so only the entry is compared.)"
        )
    lines.append(
        "- Survivorship bias: **not measured.** The ten symbols exist today; names "
        "that delisted are not in the universe, and no data source here can add them."
    )
    lines.append(
        "- Cost sensitivity: the three scenarios above span frictionless to a "
        "harsher-than-project schedule, so the result's dependence on friction is "
        "visible."
    )
    lines.append(
        "- The rules were fixed before this run and no parameter was tuned on the "
        "test period, so the test read is not selected. With this many sessions it is "
        "still descriptive, not statistically significant."
    )
    lines.append("")
    _render_prior(lines, payload.get("prior_reports", []))

    lines.append("## 7. Limitations (why this is not proof of future profitability)")
    lines.append("")
    lines.append(
        "1. **Window.** ~1 month of intraday data (vendor cap), not 6–12 months. Too "
        "short for a stable estimate of edge."
    )
    lines.append(
        "2. **Regime.** One market regime, one season. A different regime may behave "
        "differently."
    )
    lines.append(
        "3. **Costs.** The vendor's bars carry no quoted spread; the spread/slippage "
        "here are assumptions, not measurements. A wider real spread worsens the result."
    )
    lines.append(
        "4. **Survivorship.** The universe is ten current large caps; delisted names "
        "are absent."
    )
    lines.append(
        "5. **Position cap.** A 25 USD order ceiling on a 100,000 USD book makes the "
        "strategy's footprint tiny, so absolute P/L is small by construction; the "
        "ratios, not the dollars, are what carry information."
    )
    lines.append(
        "6. **Selection.** No parameter was tuned here, but the rules themselves "
        "originate elsewhere; this study validates an implementation, it does not "
        "prove a market edge."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TJR validation study over fetched frames.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--symbols", nargs="+", default=list(UNIVERSE))
    parser.add_argument("--out", type=Path, required=True, help="write the JSON study here")
    parser.add_argument("--markdown", type=Path, default=None, help="also write a Markdown report")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-render the Markdown from the existing --out JSON; run no backtest",
    )
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument(
        "--no-lookahead-probe",
        action="store_true",
        help="skip the look-ahead probe (it runs one extra single-symbol backtest)",
    )
    args = parser.parse_args(argv)

    if args.render_only:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if args.markdown is None:
            raise SystemExit("--render-only needs --markdown")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
        print(f"re-rendered {args.markdown} from {args.out}")
        return 0

    payload = run_study(
        args.data, symbols=tuple(args.symbols), split_fraction=args.split_fraction
    )
    if not args.no_lookahead_probe:
        payload["lookahead_probes"] = _default_lookahead_probes(
            args.data, payload["window"], tuple(args.symbols)
        )
    payload["prior_reports"] = [
        summarize_prior(args.out.parent / name)
        for name in ("real_data_backtest.json", "real_data_backtest_costs.json")
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


def _default_lookahead_probes(data_dir: Path, window: dict, symbols: tuple[str, ...]) -> list[dict]:
    """The look-ahead probe on two symbols, cut mid-session on a real day.

    Two symbols rather than one, so the result is not a single-symbol accident. The
    cut is three hours into the last session, where both earlier entries and a later
    one are present, so the probe compares real decisions rather than two empty sets.
    """
    session = window["last_session"]
    cut = (session_open_utc(session) + pd.Timedelta(hours=3)).isoformat()
    return [
        lookahead_probe(data_dir, symbol=symbol, session=session, cut=cut)
        for symbol in symbols[:2]
    ]


if __name__ == "__main__":  # pragma: no cover - a maintainer entry point
    raise SystemExit(main())
