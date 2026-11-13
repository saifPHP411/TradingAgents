"""The crypto TJR study: multiple periods, walk-forward, regimes, sensitivity.

A single backtest over one window is an anecdote. This module is the *study*
harness: it runs the **unchanged** TJR rule chain over real crypto bars across
several periods, several symbols, out-of-sample folds, market regimes and cost
scenarios, and reports honest, reproducible metrics.

What it reuses, rather than re-implements: the same :class:`~tradingagents.tjr.TjrBacktester`
(the shared-book, next-bar-open intraday loop), the same :class:`~tradingagents.portfolio.Ledger`,
the same :class:`~tradingagents.risk.RiskEngine` under the study's own limits
(``study_risk_limits()`` — the *shape* of the paper caps, sized to the book),
the same :class:`~tradingagents.execution.costs.CostModel`, and the same
:mod:`tradingagents.performance.metrics`. It changes no rule and no limit — if a
change here would alter a decision, it belongs in ``tjr/rules.py`` and its tests.

Three things it is careful about, because they are how a study lies:

- **Periods are honest about what the venue served.** A requested 180 days that the
  venue only has 90 of is reported as 90, never padded with synthetic bars.
- **Development and test are separated.** A walk-forward fold's test sessions are
  disjoint from its training sessions, and a sub-period is a *view* of the one
  shared-book run (the book is flat at every session close, so a session set is
  self-contained), labelled as a view rather than an independent run.
- **Costs are never zero by accident.** The cost scenarios are explicit; a
  frictionless figure is labelled as such and never presented as the result.

The whole study is deterministic: the same bars and the same config produce the
same numbers, and :func:`determinism_check` proves it by running twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from tradingagents.execution.costs import CostModel
from tradingagents.performance.metrics import compute_metrics
from tradingagents.risk import RiskLimits
from tradingagents.tjr.engine import TjrBacktester, TjrConfig, TjrResult
from tradingagents.tjr.frames import IntradaySeries
from tradingagents.tjr.rules import TjrSettings
from tradingagents.tjr.sessions import SessionPolicy, policy_named, utc_day_session
from tradingagents.tjr.strategy import MemoTjrStrategy

from ..symbols import parse_pair
from ..tjr_settings import CRYPTO_ENTRY_CUT, crypto_tjr_settings

# The periods a study reports, in days. "max" is whatever the data covers.
STANDARD_PERIODS: tuple[int, ...] = (7, 30, 90, 180)

# The fraction of the book's equity one study position may use. This is a
# *research* sizing rule, not the paper account's rule: the paper worker caps an
# order at 25 USD because that is what the operator's paper account is meant to
# trade, while a study over a 100,000 USD book has to size positions so the
# strategy's returns are visible above the rounding.
STUDY_POSITION_FRACTION = 0.10

# The most equity all study positions may represent together.
STUDY_PORTFOLIO_EXPOSURE = 0.95

# The most distinct study positions open at once.
STUDY_MAX_OPEN_POSITIONS = 5


def study_risk_limits(*, initial_cash: float) -> RiskLimits:
    """The risk limits a *study* runs under.

    These are deliberately **not** :func:`paper_parity_limits`. The paper caps
    (25 USD per order, 5 positions) are the operator's live paper-account
    limits; applying them to a 100,000 USD research book would make every
    position worth 0.025% of it, so the reported P/L would measure rounding
    rather than the strategy.

    The study's limits are the same *shape* as the paper ones — a per-position
    fraction, a portfolio exposure cap, a position count, long-only, fractional
    quantities — with the notional derived from the book's own equity. The
    strategy rules are untouched; only the sizing changes, and the config
    records it, so a run states the sizing it used.
    """
    return RiskLimits(
        max_position_size=STUDY_POSITION_FRACTION,
        max_portfolio_exposure=STUDY_PORTFOLIO_EXPOSURE,
        max_open_positions=STUDY_MAX_OPEN_POSITIONS,
        max_daily_orders=None,
        target_order_value=None,
        max_order_value=None,
        long_only=True,
        fractional_shares=True,
    )

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIMES: tuple[str, ...] = (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS)


@dataclass(frozen=True)
class CryptoBacktestConfig:
    """Everything that shapes a crypto TJR run, so the run is reproducible from it."""

    symbols: tuple[str, ...]
    initial_cash: float = 100_000.0
    currency: str = "USD"
    session_policy: SessionPolicy = field(default_factory=utc_day_session)
    """The 24/7 session definition. Crypto never closes, so the default is the UTC
    day; a study of London/New York/Asia hours passes ``policy_named("crypto_named")``."""
    risk_limits: RiskLimits | None = None
    """The limits the study runs under.

    ``None`` means :func:`study_risk_limits` for this book: the paper account's
    25-USD order cap is the *operator's* live limit and would make a research
    book's returns invisible. A caller may pass explicit limits instead; the
    config records whichever was used, so a result always states its sizing.
    """
    costs: CostModel = field(default_factory=CostModel)
    settings: TjrSettings = field(default_factory=lambda: _crypto_settings())
    allow_short: bool = False
    flat_by_session_close: bool = True
    periods: tuple[int, ...] = STANDARD_PERIODS
    walk_forward_folds: int = 4
    regime_trend_sessions: int = 30
    regime_threshold: float = 0.05
    """Trailing return above which a session counts as a bull (or, negatively,
    bear) regime. 5% over 30 sessions is a documented choice, not a fitted one."""

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not self.symbols:
            raise ValueError("a study needs at least one symbol")
        if self.walk_forward_folds < 2:
            raise ValueError("walk_forward_folds must be at least 2")
        if self.risk_limits is None:
            object.__setattr__(
                self,
                "risk_limits",
                study_risk_limits(initial_cash=self.initial_cash),
            )


def _crypto_settings() -> TjrSettings:
    """TJR settings adapted for a 24/7 market.

    The adaptation itself lives in :mod:`tradingagents.crypto.tjr_settings`, so
    the live paper path and this study trade on the *same* rules: a backtest
    that measured a different rule than the one deployed would describe a
    system that does not exist.
    """
    return crypto_tjr_settings()


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------


def build_series(
    frames: Mapping[str, Mapping[str, pd.DataFrame]],
    policy: SessionPolicy,
    *,
    symbols: Iterable[str] | None = None,
) -> dict[str, dict[str, IntradaySeries]]:
    """Turn raw interval frames into point-in-time series under ``policy``.

    ``frames`` is ``{symbol: {interval: frame}}``. The result is the same shape
    with :class:`~tradingagents.tjr.frames.IntradaySeries` values, each carrying the
    crypto session policy so the rule chain labels its sessions correctly.
    """
    wanted = [parse_pair(s).canonical for s in (symbols or frames.keys())]
    out: dict[str, dict[str, IntradaySeries]] = {}
    for symbol in wanted:
        bundle = frames.get(symbol) or {}
        out[symbol] = {}
        for interval, frame in bundle.items():
            if frame is None or frame.empty:
                continue
            out[symbol][interval] = IntradaySeries(symbol, interval, frame, session=policy)
    return out


def sessions_for(
    series: Mapping[str, Mapping[str, IntradaySeries]], policy: SessionPolicy
) -> list[str]:
    """Every session present in the data, ascending, under ``policy``."""
    days: set[str] = set()
    for bundle in series.values():
        for frame_series in bundle.values():
            local = frame_series.frame.index.tz_convert(policy.timezone)
            days |= set(local.strftime("%Y-%m-%d"))
    return sorted(days)


def series_from_minutes(
    symbol: str,
    minutes: pd.DataFrame,
    policy: SessionPolicy,
    *,
    intervals: Iterable[str] = ("1m", "5m", "15m", "1h", "4h"),
) -> dict[str, IntradaySeries]:
    """Build every interval series from one 1-minute frame, by resampling.

    Fetching 1m once and resampling is both cheaper and *exact*: a 5m bar built
    from five 1m bars the strategy could see is the same bar, and it keeps the
    closed-bar guarantee. The 1m series is the base; each coarser interval is
    :func:`~tradingagents.tjr.frames.resample` of it, and a window too short to fill
    a bucket simply has no series at that interval (an absence, not an error).
    """
    from tradingagents.tjr.frames import IntradayFrameError, resample

    base = IntradaySeries(symbol, "1m", minutes, session=policy)
    out: dict[str, IntradaySeries] = {"1m": base}
    for interval in intervals:
        if interval == "1m":
            continue
        try:
            out[interval] = resample(base, interval)
        except IntradayFrameError:
            continue
    return out


def _slice_to_sessions(
    bundle: Mapping[str, IntradaySeries], sessions: set[str], policy: SessionPolicy
) -> dict[str, IntradaySeries]:
    """A copy of ``bundle`` restricted to bars whose session is in ``sessions``."""
    out: dict[str, IntradaySeries] = {}
    for interval, frame_series in bundle.items():
        local = frame_series.frame.index.tz_convert(policy.timezone)
        mask = local.strftime("%Y-%m-%d").isin(sessions)
        sub = frame_series.frame[mask]
        if not sub.empty:
            out[interval] = IntradaySeries(frame_series.symbol, interval, sub, session=policy)
    return out


# ---------------------------------------------------------------------------
# Running one window
# ---------------------------------------------------------------------------


def intervals_for_window(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    window: list[str],
    *,
    base: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h"),
) -> tuple[str, ...]:
    """The intervals whose data actually covers ``window``.

    The 1m entry trigger needs 1-minute bars over the whole window; where the venue
    served only the last few days of 1m, the window's earlier sessions would have no
    1m data and the chain would report no entry for a reason that is about *data*,
    not about the market. So the 1m interval is dropped when it does not reach the
    window's first session, and the report states which intervals a view used.

    Coverage is compared by *session date*, not by the exact instant: the venue
    serves 1m bars from partway through a day, so the first covered session is
    partial. Requiring a bar at the session's midnight would drop the 1m trigger
    from a window the 1m data in fact covers almost entirely.

    The window's sessions are the ones the caller asked for, and a window that lies
    entirely before or after the data has no covered session at all — that is a
    period the data does not reach, and the run refuses it rather than reporting a
    no-trade result.
    """
    if not window:
        return base
    start_date = window[0]
    end_date = window[-1]
    has_one_minute = False
    for bundle in series.values():
        one = bundle.get("1m")
        if one is None or one.frame.empty:
            continue
        local = one.frame.index.tz_convert("UTC").strftime("%Y-%m-%d")
        covered = [d for d in local if start_date <= d <= end_date]
        if covered:
            has_one_minute = True
            break
    if has_one_minute:
        return base
    return tuple(i for i in base if i != "1m")


def run_window(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    sessions: Iterable[str] | None = None,
    intervals: Iterable[str] | None = None,
    evaluation_cache: dict | None = None,
    run_id: str = "crypto-tjr",
) -> TjrResult:
    """Run the TJR backtester over ``series`` (optionally restricted to ``sessions``).

    The engine is the same shared-book intraday loop the equity TJR study uses; the
    only crypto-specific input is the session policy, which labels each UTC day as a
    session. A ``sessions`` subset is a *view*: the book is flat at every session
    close, so a set of whole sessions is self-contained and the result is exactly
    what the full run produced over those sessions.

    ``intervals`` restricts which timeframes are used (for a long period that the
    1m data does not reach). ``evaluation_cache`` lets a caller share the memo across
    a study's scenarios; a cache is per-study, and the memo is exact.
    """
    selected = set(sessions) if sessions is not None else None
    wanted = set(intervals) if intervals is not None else None
    one_minute: dict[str, IntradaySeries] = {}
    five_minute: dict[str, IntradaySeries] = {}
    hourly: dict[str, IntradaySeries] = {}
    four_hour: dict[str, IntradaySeries] = {}
    for symbol, bundle in series.items():
        if selected is not None:
            bundle = _slice_to_sessions(bundle, selected, config.session_policy)
        if wanted is not None:
            bundle = {i: s for i, s in bundle.items() if i in wanted}
        if "1m" in bundle:
            one_minute[symbol] = bundle["1m"]
        if "5m" in bundle:
            five_minute[symbol] = bundle["5m"]
        if "1h" in bundle:
            hourly[symbol] = bundle["1h"]
        if "4h" in bundle:
            four_hour[symbol] = bundle["4h"]
    if not five_minute:
        raise ValueError("no 5-minute series to run over")
    # TJR's entry trigger is a 1-minute rule, so a window the 1m data does not
    # cover cannot be evaluated: rather than run the chain with the 1m series
    # silently absent (which would report no trade for a *data* reason), the run
    # is refused and the caller reports the period as unavailable.
    if "1m" in (wanted or {"1m"}):
        missing_one = sorted(s for s in five_minute if s not in one_minute)
        if missing_one:
            raise ValueError(
                f"no 1-minute series for {missing_one}; the window is not covered at 1m"
            )
    usable = (
        sorted(s for s in five_minute if s in one_minute) if one_minute else sorted(five_minute)
    )
    if not usable:
        raise ValueError("no symbol has a usable series set")
    tjr_config = TjrConfig(
        initial_cash=config.initial_cash,
        currency=config.currency,
        risk_limits=config.risk_limits,
        costs=config.costs,
        settings=config.settings,
        allow_short=config.allow_short,
        flat_by_session_close=config.flat_by_session_close,
        session_policy=config.session_policy,
    )
    backtester = TjrBacktester(
        one_minute={s: one_minute[s] for s in usable},
        five_minute={s: five_minute[s] for s in usable},
        hourly={s: hourly[s] for s in hourly if s in usable} or None,
        four_hour={s: four_hour[s] for s in four_hour if s in usable} or None,
        config=tjr_config,
        run_id=run_id,
    )
    # The rule chain is a pure function of the bars, so its answer per instant is
    # the same across a study's scenarios. Memoizing it is exact and is what lets a
    # ten-symbol, multi-scenario study finish in minutes; the cache is per study, so
    # no result is carried between studies.
    backtester.strategy = MemoTjrStrategy(
        one_minute=backtester.one_minute,
        five_minute=backtester.five_minute,
        hourly=backtester.hourly,
        four_hour=backtester.four_hour,
        settings=tjr_config.settings,
        allow_short=tjr_config.allow_short,
        session_policy=tjr_config.session_policy,
        cache=evaluation_cache if evaluation_cache is not None else {},
    )
    return backtester.run()


# ---------------------------------------------------------------------------
# Benchmark and metrics
# ---------------------------------------------------------------------------


def equal_weight_benchmark(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    *,
    initial_capital: float,
    equity_index: pd.Index,
) -> pd.Series | None:
    """Buy-and-hold the symbols equally, marked on ``equity_index``.

    The benchmark the strategy must beat: hold every symbol from the first bar to
    each subsequent mark, weighted equally. Built from the same 5m closes the
    strategy read, so it invents no price. ``None`` when no symbol has a 5m series.
    """
    closes: dict[str, pd.Series] = {}
    for symbol, bundle in series.items():
        frame_series = bundle.get("5m")
        if frame_series is None or frame_series.frame.empty:
            continue
        closes[symbol] = frame_series.frame["close"]
    if not closes:
        return None
    combined = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
    if combined.empty:
        return None
    reindexed = combined.reindex(equity_index, method="ffill").dropna(how="all")
    if reindexed.empty:
        return None
    normalized = reindexed.divide(reindexed.iloc[0], axis=1)
    mean = normalized.mean(axis=1)
    return (mean * initial_capital).rename("buy_and_hold")


def metrics_for(
    result: TjrResult,
    *,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 365 * 24 * 12,
) -> dict:
    """Every reported metric for one run, from its own equity curve and trades.

    ``periods_per_year`` defaults to the number of 5-minute marks in a 365-day year,
    because a crypto run is marked on a 24/7 clock rather than on equity sessions.
    """
    equity = result.equity_series()
    exposure = result.exposure_series()
    metrics = compute_metrics(
        equity,
        initial_capital=result.initial_cash,
        trades=result.trades,
        exposure=exposure,
        traded_notional=result.traded_notional,
        realized_pnl=result.ledger.realized_pnl if result.ledger else 0.0,
        unrealized_pnl=result.ledger.unrealized_pnl if result.ledger else 0.0,
        total_fees=result.ledger.total_fees if result.ledger else 0.0,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
    )
    payload = metrics.to_dict()
    payload.update(_trade_stats(result))
    payload["sessions"] = len(result.sessions)
    payload["reason_counts"] = dict(result.reason_counts)
    payload["rejections"] = len(result.rejections)
    payload["traded_notional"] = result.traded_notional
    return payload


def _trade_stats(result: TjrResult) -> dict:
    """Average R and average planned reward/risk, from the trades that carry them."""
    realized = [t.realized_r for t in result.trades if t.realized_r is not None]
    planned = [t.planned_reward_risk for t in result.trades if t.planned_reward_risk is not None]
    return {
        "average_r": (sum(realized) / len(realized)) if realized else None,
        "average_planned_reward_risk": (sum(planned) / len(planned)) if planned else None,
        "trades_with_risk": len(realized),
        "fees": result.ledger.total_fees if result.ledger else 0.0,
        "slippage_cost": _slippage_cost(result),
    }


def _slippage_cost(result: TjrResult) -> float:
    """The cost of spread and slippage the run charged, from its own config.

    ``traded_notional`` times the model's adverse bps, applied at entry and exit.
    Reported so the friction is visible next to the fees, never folded into them.
    """
    config = result.config
    if config is None:
        return 0.0
    bps = (config.costs.half_spread_bps + config.costs.slippage_bps) / 10_000.0
    return result.traded_notional * bps


# ---------------------------------------------------------------------------
# Periods, regimes, walk-forward, sensitivity
# ---------------------------------------------------------------------------


@dataclass
class PeriodResult:
    """One period's metrics, with what the venue actually served for it."""

    label: str
    days: int | None
    start: str | None
    end: str | None
    sessions: int
    metrics: dict
    intervals: tuple[str, ...] = ()
    """The timeframes this view was run on. A long period has no 1m data, so its
    ``intervals`` state that the 1m entry trigger was not evaluated."""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "days": self.days,
            "start": self.start,
            "end": self.end,
            "sessions": self.sessions,
            "intervals": list(self.intervals),
            "metrics": self.metrics,
        }


def _window_sessions(sessions: list[str], days: int | None) -> list[str]:
    """The trailing ``days`` calendar days of sessions, or every session for ``None``."""
    if days is None:
        return list(sessions)
    if not sessions:
        return []
    last = datetime.fromisoformat(sessions[-1]).date()
    cutoff = last - timedelta(days=days - 1)
    return [s for s in sessions if datetime.fromisoformat(s).date() >= cutoff]


def _period_label(days: int, sessions: int) -> str:
    """The period's label, honest about a shortfall.

    A requested 90-day window that the data only covers for 30 days is labelled
    ``90d_requested_30d`` — never plain ``90d``, which would read as though the
    venue served 90 days. A window that got everything it asked for keeps the
    short label.
    """
    if sessions >= days:
        return f"{days}d"
    return f"{days}d_requested_{sessions}d"


def run_periods(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    coverage_days: int | None = None,
    evaluation_cache: dict | None = None,
) -> list[PeriodResult]:
    """Run the standard periods, plus "max" (whatever the data covers).

    A period whose window the data does not reach is still run over the sessions it
    does have, and its ``days``/``sessions`` report the truth. ``coverage_days``
    states the venue's coverage so a shortfall is visible rather than inferred.
    """
    sessions = sessions_for(series, config.session_policy)
    results: list[PeriodResult] = []
    for days in config.periods:
        window = _window_sessions(sessions, days)
        if not window:
            continue
        intervals = intervals_for_window(series, window)
        try:
            result = run_window(
                series,
                config,
                sessions=window,
                intervals=intervals,
                evaluation_cache=evaluation_cache,
                run_id=f"crypto-tjr-{days}d",
            )
        except ValueError:
            continue
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            series, initial_capital=config.initial_cash, equity_index=equity.index
        )
        results.append(
            PeriodResult(
                label=_period_label(days, len(window)),
                days=days,
                start=window[0],
                end=window[-1],
                sessions=len(window),
                intervals=intervals,
                metrics=metrics_for(result, benchmark=benchmark),
            )
        )
    span = _span_days(sessions)
    intervals = intervals_for_window(series, sessions)
    try:
        full = run_window(
            series,
            config,
            intervals=intervals,
            evaluation_cache=evaluation_cache,
            run_id="crypto-tjr-max",
        )
    except ValueError:
        return results
    equity = full.equity_series()
    benchmark = equal_weight_benchmark(
        series, initial_capital=config.initial_cash, equity_index=equity.index
    )
    results.append(
        PeriodResult(
            label="max",
            days=coverage_days if coverage_days is not None else span,
            start=sessions[0] if sessions else None,
            end=sessions[-1] if sessions else None,
            sessions=len(sessions),
            intervals=intervals,
            metrics=metrics_for(full, benchmark=benchmark),
        )
    )
    return results


def _span_days(sessions: list[str]) -> int | None:
    """The calendar days a session list spans, or ``None`` for an empty list."""
    if not sessions:
        return None
    first = datetime.fromisoformat(sessions[0]).date()
    last = datetime.fromisoformat(sessions[-1]).date()
    return (last - first).days + 1


def run_walk_forward(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> list[PeriodResult]:
    """Sequential out-of-sample folds.

    TJR has no fitted parameters, so "walk-forward" here means what it can honestly
    mean: the sessions are split into ``walk_forward_folds`` contiguous folds, and
    each fold is run and reported as an out-of-sample window over sessions disjoint
    from every other fold. There is no training step to leak — which is stated
    rather than implied.
    """
    sessions = sessions_for(series, config.session_policy)
    folds = config.walk_forward_folds
    if len(sessions) < folds:
        return []
    size = len(sessions) // folds
    out: list[PeriodResult] = []
    for index in range(folds):
        start = index * size
        end = len(sessions) if index == folds - 1 else (index + 1) * size
        window = sessions[start:end]
        result = run_window(
            series,
            config,
            sessions=window,
            evaluation_cache=evaluation_cache,
            run_id=f"crypto-tjr-wf{index + 1}",
        )
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            series, initial_capital=config.initial_cash, equity_index=equity.index
        )
        out.append(
            PeriodResult(
                label=f"fold_{index + 1}",
                days=None,
                start=window[0],
                end=window[-1],
                sessions=len(window),
                metrics=metrics_for(result, benchmark=benchmark),
            )
        )
    return out


def _classify_regimes(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
) -> dict[str, list[str]]:
    """Bucket every session into a bull, bear or sideways regime.

    A session's regime is its benchmark's trailing return over
    ``regime_trend_sessions`` sessions: above ``regime_threshold`` is bull, below its
    negative is bear, else sideways. Computed from the equal-weight benchmark the
    strategy is compared against, so the label and the comparison agree.
    """
    sessions = sessions_for(series, config.session_policy)
    buckets: dict[str, list[str]] = {REGIME_BULL: [], REGIME_BEAR: [], REGIME_SIDEWAYS: []}
    if not sessions:
        return buckets
    # A daily benchmark series, from the 5m closes marked at each session's last bar.
    daily: dict[str, float] = {}
    for bundle in series.values():
        frame_series = bundle.get("5m")
        if frame_series is None or frame_series.frame.empty:
            continue
        local = frame_series.frame.index.tz_convert(config.session_policy.timezone)
        days = pd.Series(local.strftime("%Y-%m-%d"), index=frame_series.frame.index)
        last_close = frame_series.frame["close"].groupby(days).last()
        for day, value in last_close.items():
            daily[day] = daily.get(day, 0.0) + float(value)
    ordered = [s for s in sessions if s in daily]
    for index, session in enumerate(ordered):
        lookback = index - config.regime_trend_sessions
        if lookback < 0:
            continue
        past, now = daily[ordered[lookback]], daily[session]
        if past <= 0:
            continue
        trend = (now - past) / past
        if trend > config.regime_threshold:
            buckets[REGIME_BULL].append(session)
        elif trend < -config.regime_threshold:
            buckets[REGIME_BEAR].append(session)
        else:
            buckets[REGIME_SIDEWAYS].append(session)
    return buckets


def run_regimes(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> list[PeriodResult]:
    """Run the strategy over each regime's sessions, as a view of the shared book."""
    buckets = _classify_regimes(series, config)
    out: list[PeriodResult] = []
    for regime in REGIMES:
        window = buckets.get(regime) or []
        if not window:
            continue
        result = run_window(
            series,
            config,
            sessions=window,
            evaluation_cache=evaluation_cache,
            run_id=f"crypto-tjr-{regime}",
        )
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            series, initial_capital=config.initial_cash, equity_index=equity.index
        )
        out.append(
            PeriodResult(
                label=regime,
                days=None,
                start=window[0],
                end=window[-1],
                sessions=len(window),
                metrics=metrics_for(result, benchmark=benchmark),
            )
        )
    return out


def cost_scenarios(config: CryptoBacktestConfig) -> dict[str, CostModel]:
    """The cost schedules a study reports, from frictionless to realistic.

    A frictionless run is included *and labelled*, so a reader can see how much of
    a result is friction. The realistic scenario uses a 10 bps per-fill commission
    (matching Bybit's published VIP 0 spot rate as a reference), a 1 bp half-spread
    and 2 bps slippage. Account- and region-specific fees must still be verified.
    """
    return {
        "frictionless": CostModel(),
        "realistic": CostModel(commission_rate=0.0010, half_spread_bps=1.0, slippage_bps=2.0),
        "conservative": CostModel(commission_rate=0.0015, half_spread_bps=2.0, slippage_bps=5.0),
    }


def _config_with_costs(config: CryptoBacktestConfig, costs: CostModel) -> CryptoBacktestConfig:
    """Copy every study setting while replacing only the execution-cost schedule."""
    return CryptoBacktestConfig(
        symbols=config.symbols,
        initial_cash=config.initial_cash,
        currency=config.currency,
        session_policy=config.session_policy,
        risk_limits=config.risk_limits,
        costs=costs,
        settings=config.settings,
        allow_short=config.allow_short,
        flat_by_session_close=config.flat_by_session_close,
        periods=config.periods,
        walk_forward_folds=config.walk_forward_folds,
        regime_trend_sessions=config.regime_trend_sessions,
        regime_threshold=config.regime_threshold,
    )


def run_cost_scenarios(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> list[PeriodResult]:
    """Run the full window under each cost scenario."""
    out: list[PeriodResult] = []
    for label, costs in cost_scenarios(config).items():
        scenario_config = _config_with_costs(config, costs)
        result = run_window(
            series,
            scenario_config,
            evaluation_cache=evaluation_cache,
            run_id=f"crypto-tjr-cost-{label}",
        )
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            series, initial_capital=config.initial_cash, equity_index=equity.index
        )
        out.append(
            PeriodResult(
                label=label,
                days=None,
                start=result.sessions[0] if result.sessions else None,
                end=result.sessions[-1] if result.sessions else None,
                sessions=len(result.sessions),
                metrics=metrics_for(result, benchmark=benchmark),
            )
        )
    return out


def run_period_cost_scenarios(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> dict[str, list[PeriodResult]]:
    """Run every requested period and the max window under every cost schedule."""
    return {
        label: run_periods(
            series,
            _config_with_costs(config, costs),
            evaluation_cache=evaluation_cache,
        )
        for label, costs in cost_scenarios(config).items()
    }


def run_walk_forward_cost_scenarios(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> dict[str, list[PeriodResult]]:
    """Run every disjoint OOS fold under every cost schedule."""
    return {
        label: run_walk_forward(
            series,
            _config_with_costs(config, costs),
            evaluation_cache=evaluation_cache,
        )
        for label, costs in cost_scenarios(config).items()
    }


def sensitivity_variants(config: CryptoBacktestConfig) -> dict[str, dict]:
    """The documented variations a sensitivity study sweeps.

    Each entry changes exactly one input, so a difference in the result is
    attributable. The session policy and the entry-window cut are included because
    they are the two crypto-specific assumptions, and a reader is entitled to see
    how much they matter.
    """
    return {
        "baseline": {},
        "swing_lookback_1": {
            "settings": TjrSettings(swing_lookback=1, latest_entry_time=CRYPTO_ENTRY_CUT)
        },
        "swing_lookback_3": {
            "settings": TjrSettings(swing_lookback=3, latest_entry_time=CRYPTO_ENTRY_CUT)
        },
        "min_rr_1.5": {
            "settings": TjrSettings(min_reward_risk=1.5, latest_entry_time=CRYPTO_ENTRY_CUT)
        },
        "min_rr_0.5": {
            "settings": TjrSettings(min_reward_risk=0.5, latest_entry_time=CRYPTO_ENTRY_CUT)
        },
        "session_named": {"session_policy": policy_named("crypto_named")},
        "entry_cut_22_30": {"settings": TjrSettings(latest_entry_time=(22, 30))},
        "no_entry_cut": {"settings": TjrSettings(latest_entry_time=(24, 0))},
    }


def run_sensitivity(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> list[PeriodResult]:
    """Run the full window under each single-input variation."""
    out: list[PeriodResult] = []
    for label, overrides in sensitivity_variants(config).items():
        variant = CryptoBacktestConfig(
            symbols=config.symbols,
            initial_cash=config.initial_cash,
            currency=config.currency,
            session_policy=overrides.get("session_policy", config.session_policy),
            risk_limits=config.risk_limits,
            costs=config.costs,
            settings=overrides.get("settings", config.settings),
            allow_short=config.allow_short,
            flat_by_session_close=config.flat_by_session_close,
            periods=config.periods,
            walk_forward_folds=config.walk_forward_folds,
            regime_trend_sessions=config.regime_trend_sessions,
            regime_threshold=config.regime_threshold,
        )
        try:
            result = run_window(
                series,
                variant,
                evaluation_cache=evaluation_cache,
                run_id=f"crypto-tjr-sens-{label}",
            )
        except ValueError:
            continue
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            series, initial_capital=config.initial_cash, equity_index=equity.index
        )
        out.append(
            PeriodResult(
                label=label,
                days=None,
                start=result.sessions[0] if result.sessions else None,
                end=result.sessions[-1] if result.sessions else None,
                sessions=len(result.sessions),
                metrics=metrics_for(result, benchmark=benchmark),
            )
        )
    return out


def per_symbol_metrics(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    evaluation_cache: dict | None = None,
) -> dict[str, dict]:
    """Run each symbol alone, so a study can show a per-symbol split."""
    out: dict[str, dict] = {}
    for symbol in series:
        single = {symbol: series[symbol]}
        try:
            result = run_window(
                single,
                config,
                evaluation_cache=evaluation_cache,
                run_id=f"crypto-tjr-{symbol}",
            )
        except ValueError:
            continue
        equity = result.equity_series()
        benchmark = equal_weight_benchmark(
            single, initial_capital=config.initial_cash, equity_index=equity.index
        )
        out[symbol] = metrics_for(result, benchmark=benchmark)
    return out


def determinism_check(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    sessions: Iterable[str] | None = None,
    evaluation_cache: dict | None = None,
) -> dict:
    """Run the same window twice and report whether the results are identical.

    The strategy is a pure function of the bars, so two runs must agree on equity,
    trades and every reason count. A mismatch is a bug, and this makes it visible.
    """
    first = run_window(
        series,
        config,
        sessions=sessions,
        evaluation_cache=evaluation_cache,
        run_id="determinism-a",
    )
    # The second run uses a fresh cache on purpose: determinism means the *result* is
    # the same whether or not the memo was warm, which is the property that matters.
    second = run_window(
        series, config, sessions=sessions, evaluation_cache={}, run_id="determinism-b"
    )
    first_metrics = metrics_for(first)
    second_metrics = metrics_for(second)
    identical = (
        first_metrics["final_equity"] == second_metrics["final_equity"]
        and first_metrics["num_trades"] == second_metrics["num_trades"]
        and first.reason_counts == second.reason_counts
    )
    return {
        "identical": identical,
        "final_equity": [first_metrics["final_equity"], second_metrics["final_equity"]],
        "num_trades": [first_metrics["num_trades"], second_metrics["num_trades"]],
    }


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


@dataclass
class CryptoStudyResult:
    """Everything a crypto TJR study produced, ready to serialize."""

    config: CryptoBacktestConfig
    generated_at: str
    coverage: dict
    """What the venue actually served, per symbol/interval."""
    periods: list[PeriodResult] = field(default_factory=list)
    per_symbol: dict[str, dict] = field(default_factory=dict)
    walk_forward: list[PeriodResult] = field(default_factory=list)
    regimes: list[PeriodResult] = field(default_factory=list)
    cost_scenarios: list[PeriodResult] = field(default_factory=list)
    period_cost_scenarios: dict[str, list[PeriodResult]] = field(default_factory=dict)
    walk_forward_cost_scenarios: dict[str, list[PeriodResult]] = field(default_factory=dict)
    sensitivity: list[PeriodResult] = field(default_factory=list)
    determinism: dict = field(default_factory=dict)
    universe: dict = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "crypto_tjr_study",
            "generated_at": self.generated_at,
            "strategy": "tjr",
            "asset_class": "crypto",
            "symbols": list(self.config.symbols),
            "initial_cash": self.config.initial_cash,
            "session_policy": self.config.session_policy.to_dict(),
            "entry_cut": list(self.config.settings.latest_entry_time),
            "coverage": self.coverage,
            "universe": self.universe,
            "periods": [p.to_dict() for p in self.periods],
            "per_symbol": self.per_symbol,
            "walk_forward": [p.to_dict() for p in self.walk_forward],
            "regimes": [p.to_dict() for p in self.regimes],
            "cost_scenarios": [p.to_dict() for p in self.cost_scenarios],
            "period_cost_scenarios": {
                label: [p.to_dict() for p in rows]
                for label, rows in self.period_cost_scenarios.items()
            },
            "walk_forward_cost_scenarios": {
                label: [p.to_dict() for p in rows]
                for label, rows in self.walk_forward_cost_scenarios.items()
            },
            "cost_assumptions": {
                label: {
                    "commission_rate": model.commission_rate,
                    "half_spread_bps": model.half_spread_bps,
                    "slippage_bps": model.slippage_bps,
                    "maker_fee_bps": model.maker_fee_bps,
                    "taker_fee_bps": model.taker_fee_bps,
                }
                for label, model in cost_scenarios(self.config).items()
            },
            "sensitivity": [p.to_dict() for p in self.sensitivity],
            "determinism": self.determinism,
            "trades": self.trades,
            "notes": self.notes,
        }


def run_study(
    series: Mapping[str, Mapping[str, IntradaySeries]],
    config: CryptoBacktestConfig,
    *,
    coverage: Mapping[str, dict] | None = None,
    universe: Mapping | None = None,
    include_sensitivity: bool = True,
) -> CryptoStudyResult:
    """Run every part of the study over ``series`` and return the result.

    The full-window run's trades are included, because a report that shows only
    aggregates hides the losses. Nothing here filters a losing trade out.
    """
    cache: dict = {}
    sessions = sessions_for(series, config.session_policy)
    full_intervals = intervals_for_window(series, sessions)
    full = run_window(
        series,
        config,
        intervals=full_intervals,
        evaluation_cache=cache,
        run_id="crypto-tjr-full",
    )
    notes = [
        "The strategy is the unchanged TJR rule chain; only the session policy and "
        "the entry-window cut are adapted for a 24/7 market, and both are reported.",
        "Historical research bars are Binance public Spot data; live demo decisions "
        "are intended to use Bybit Spot candles and orders execute on Bybit Demo Spot. "
        "Treat this backtest as a cross-venue proxy, not a Bybit historical replay.",
        "Every figure comes from real bars the venue served; a period the data does "
        "not cover is reported short, never padded.",
    ]
    result = CryptoStudyResult(
        config=config,
        generated_at=datetime.now(UTC).isoformat(),
        coverage=dict(coverage or {}),
        universe=dict(universe or {}),
        periods=run_periods(series, config, evaluation_cache=cache),
        per_symbol=per_symbol_metrics(series, config, evaluation_cache=cache),
        walk_forward=run_walk_forward(series, config, evaluation_cache=cache),
        regimes=run_regimes(series, config, evaluation_cache=cache),
        cost_scenarios=run_cost_scenarios(series, config, evaluation_cache=cache),
        period_cost_scenarios=run_period_cost_scenarios(series, config, evaluation_cache=cache),
        walk_forward_cost_scenarios=run_walk_forward_cost_scenarios(
            series, config, evaluation_cache=cache
        ),
        sensitivity=(
            run_sensitivity(series, config, evaluation_cache=cache) if include_sensitivity else []
        ),
        determinism=determinism_check(series, config),
        trades=[
            {
                "symbol": t.symbol,
                "session": t.session,
                "side": t.side,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "exit_reason": t.exit_reason,
                "net_pnl": t.net_pnl,
                "fees": t.fees,
                "stop": t.stop,
                "target": t.target,
                "planned_reward_risk": t.planned_reward_risk,
                "realized_r": t.realized_r,
                "reason_code": t.reason_code,
            }
            for t in full.trades
        ],
        notes=notes,
    )
    return result


__all__ = [
    "REGIMES",
    "REGIME_BEAR",
    "REGIME_BULL",
    "REGIME_SIDEWAYS",
    "STANDARD_PERIODS",
    "CryptoBacktestConfig",
    "CryptoStudyResult",
    "PeriodResult",
    "build_series",
    "cost_scenarios",
    "determinism_check",
    "equal_weight_benchmark",
    "metrics_for",
    "per_symbol_metrics",
    "run_cost_scenarios",
    "run_period_cost_scenarios",
    "run_walk_forward_cost_scenarios",
    "run_periods",
    "run_regimes",
    "run_sensitivity",
    "run_study",
    "run_walk_forward",
    "run_window",
    "sensitivity_variants",
    "series_from_minutes",
    "sessions_for",
]
