"""Performance analytics: what a run actually produced, measured honestly.

Every metric here is computed from the equity curve and the trade list, and each
one is defined in a docstring because the definitions are where the lying
happens. Two conventions are worth stating up front, because they change the
numbers and are easy to get wrong:

**Returns are not annualised by default, and annualisation is explicit.** A
20-session backtest that made 2% did not make ~25% a year. :meth:`PerformanceMetrics.annualized_return`
scales by 252 sessions and says so, but the headline :attr:`total_return` does not
extrapolate at all. Reporting a small sample as a rate is the most common way a
backtest misleads.

**Ratios are guarded against a zero denominator.** A flat curve has no
volatility, so a Sharpe ratio is undefined rather than infinite; those cases
return ``None``, which a report renders as "n/a". Returning a large number
instead would make a do-nothing run look spectacular.

Drawdown is peak-to-trough on the equity curve, which is the only definition
that matches what an account experiences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# US equity sessions per year. Used only for the explicitly-annualised metrics.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    """The measured performance of one run.

    A ``None`` means the metric is undefined for this run (no volatility, no
    losing trade, an empty curve). It never means zero, and it never means
    infinity.
    """

    initial_capital: float
    final_equity: float
    total_return: float
    """(final - initial) / initial. Not annualised."""
    annualized_return: float | None
    """Total return scaled to a year of sessions. ``None`` for a run under a session."""
    cagr: float | None
    """Compound annual growth rate from the elapsed calendar time. ``None`` without it."""
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float | None
    """Wins / (wins + losses). ``None`` with no closed trades."""
    profit_factor: float | None
    """Gross profit / gross loss. ``None`` when nothing was lost."""
    average_win: float | None
    average_loss: float | None
    expectancy: float | None
    """Mean net P&L per trade."""
    max_drawdown: float
    """Largest peak-to-trough decline as a positive fraction of the peak."""
    max_drawdown_start: str | None
    max_drawdown_end: str | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    exposure: float
    """Mean fraction of equity invested across the curve."""
    turnover: float
    """Total traded notional / mean equity, a crude activity measure."""
    volatility: float | None
    """Annualised standard deviation of per-session returns."""
    benchmark_return: float | None = None
    benchmark_final_equity: float | None = None
    excess_return: float | None = None
    """Strategy total return minus the benchmark's, over the same sessions."""
    sessions: int = 0
    start: str | None = None
    end: str | None = None
    annualization_days: int = TRADING_DAYS_PER_YEAR

    def to_dict(self) -> dict:
        """A flat, JSON-safe mapping, for the report and the run record."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "cagr": self.cagr,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_fees": self.total_fees,
            "num_trades": self.num_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_start": self.max_drawdown_start,
            "max_drawdown_end": self.max_drawdown_end,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "exposure": self.exposure,
            "turnover": self.turnover,
            "volatility": self.volatility,
            "benchmark_return": self.benchmark_return,
            "benchmark_final_equity": self.benchmark_final_equity,
            "excess_return": self.excess_return,
            "sessions": self.sessions,
            "start": self.start,
            "end": self.end,
            "annualization_days": self.annualization_days,
        }


@dataclass(frozen=True)
class DrawdownDetail:
    """A drawdown episode: its depth, and the sessions that bracket it."""

    max_drawdown: float
    start: str | None
    """The peak the decline fell from."""
    end: str | None
    """The trough it reached."""
    recovery: str | None
    """The session the curve regained the peak, or ``None`` if it never did."""
    duration: int
    """Sessions from peak to recovery, or to the end of the curve if unrecovered."""


def drawdown_series(equity: pd.Series) -> pd.Series:
    """The running decline from the peak, as a non-negative fraction per session."""
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    # A peak of zero (or below) has no meaningful decline; guard the division.
    safe_peak = peak.where(peak > 0, other=float("nan"))
    return ((safe_peak - equity) / safe_peak).fillna(0.0)


def max_drawdown(equity: pd.Series) -> DrawdownDetail:
    """The largest peak-to-trough decline, with the sessions that bracket it.

    The trough is the session of the maximum decline; the peak is the highest
    point at or before it, which is what the account actually fell from.

    Only a strictly positive peak is measured. A peak at or below zero is not a
    peak an account can fall from — a curve that goes from 0 to -5 has no
    meaningful percentage decline, and dividing by the zero peak would report an
    infinite drawdown. Such a decline is reported as 0, matching the guard in
    :func:`drawdown_series`.
    """
    if equity.empty:
        return DrawdownDetail(0.0, None, None, None, 0)
    positive = equity.where(equity > 0)
    running_peak = positive.cummax()
    decline = running_peak - equity
    if decline.max() <= 0 or not math.isfinite(decline.max()):
        return DrawdownDetail(0.0, None, None, None, 0)
    trough_idx = decline.idxmax()
    peak_value = positive.loc[:trough_idx].max()
    if not math.isfinite(peak_value) or peak_value <= 0:
        return DrawdownDetail(0.0, None, None, None, 0)
    peak_idx = positive.loc[:trough_idx].idxmax()
    depth = float(decline.max() / peak_value)
    recovery_idx = None
    peak_equity = equity.loc[peak_idx]
    after = equity.loc[trough_idx:]
    regained = after[after >= peak_equity]
    if len(regained) > 1 or (len(regained) == 1 and regained.index[0] != trough_idx):
        recovery_idx = regained.index[0]
    if recovery_idx is not None:
        duration = _session_distance(equity.index, peak_idx, recovery_idx)
    else:
        duration = _session_distance(equity.index, peak_idx, equity.index[-1])
    return DrawdownDetail(
        max_drawdown=depth,
        start=str(peak_idx),
        end=str(trough_idx),
        recovery=str(recovery_idx) if recovery_idx is not None else None,
        duration=duration,
    )


def _session_distance(index, start, end) -> int:
    """Sessions between two index labels, inclusive of the end."""
    positions = list(index)
    try:
        return positions.index(end) - positions.index(start)
    except ValueError:
        return 0


def period_returns(equity: pd.Series) -> pd.Series:
    """Per-session simple returns, with the first session's return dropped.

    The first observation has no predecessor, so it has no return. Including it
    as zero would dilute the volatility estimate.
    """
    if len(equity) < 2:
        return pd.Series(dtype=float)
    # A non-finite equity point (a NaN a caller let through, an inf from a
    # division) makes every return after it non-finite, which then makes the
    # Sharpe, Sortino and volatility silently ``None``. Dropping the point keeps
    # the finite returns usable rather than losing the whole series.
    clean = equity[equity.notna() & (equity.abs() != float("inf"))]
    if len(clean) < 2:
        return pd.Series(dtype=float)
    returns = clean.pct_change().iloc[1:]
    return returns.replace([math.inf, -math.inf], math.nan).dropna()


def sharpe_ratio(
    returns: pd.Series, *, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float | None:
    """Annualised Sharpe ratio, or ``None`` when volatility is zero or unknown.

    ``risk_free_rate`` is an annual rate, de-annualised to the period before use.
    A single observation cannot give a volatility, so it returns ``None``.
    """
    if len(returns) < 2:
        return None
    excess = returns - risk_free_rate / periods_per_year
    deviation = float(excess.std(ddof=1))
    if not math.isfinite(deviation) or deviation == 0.0:
        return None
    return float(excess.mean() / deviation * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, *, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float | None:
    """Annualised Sortino ratio, using only downside deviation.

    ``None`` when there is no downside at all: a curve that never fell has an
    undefined downside deviation, and reporting a huge number for it would
    flatter a run that simply had not been tested yet.
    """
    if len(returns) < 2:
        return None
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return None
    deviation = float(math.sqrt((downside**2).mean()))
    if not math.isfinite(deviation) or deviation == 0.0:
        return None
    return float(excess.mean() / deviation * math.sqrt(periods_per_year))


def calmar_ratio(annualized_return: float | None, max_dd: float) -> float | None:
    """Annualised return over maximum drawdown, or ``None`` when either is undefined."""
    if annualized_return is None or max_dd <= 0:
        return None
    return annualized_return / max_dd


def annualize_return(total_return: float, sessions: int, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """Scale a total return to a year of sessions, compounded.

    Explicit about its assumption: it treats ``sessions`` as a representative
    sample of a ``periods_per_year``-session year. That is wrong for a short or
    seasonally unusual run, which is why the headline return is not annualised
    and this is a separate, named metric.
    """
    if sessions < 1:
        return None
    growth = 1.0 + total_return
    if growth <= 0:
        # A total loss (or worse) has no finite annualised equivalent.
        return None
    return growth ** (periods_per_year / sessions) - 1.0


def cagr(initial: float, final: float, start: datetime | str | None, end: datetime | str | None) -> float | None:
    """Compound annual growth rate from the elapsed calendar time, or ``None``.

    Requires a positive final equity and a start strictly before the end; without
    a real elapsed period there is no rate to compute.
    """
    if initial <= 0 or final <= 0 or start is None or end is None:
        return None
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError):
        # A curve indexed by something that is not a date has no elapsed calendar
        # time to compound over.
        return None
    years = (end_ts - start_ts).days / 365.25
    if years <= 0:
        return None
    return (final / initial) ** (1.0 / years) - 1.0


def compute_metrics(
    equity: pd.Series,
    *,
    initial_capital: float | None = None,
    trades: list | None = None,
    exposure: pd.Series | None = None,
    traded_notional: float = 0.0,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    total_fees: float = 0.0,
    risk_free_rate: float = 0.0,
    benchmark: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    """Every metric, from an equity curve and a list of closed trades.

    ``trades`` may be any objects exposing ``net_pnl`` (the backtester's
    :class:`~tradingagents.backtesting.engine.Trade` does), so this does not
    depend on the backtester's types.
    """
    trades = trades or []
    initial = initial_capital if initial_capital is not None else (float(equity.iloc[0]) if len(equity) else 0.0)
    final = float(equity.iloc[-1]) if len(equity) else initial
    total_return = (final - initial) / initial if initial else 0.0
    sessions = len(equity)

    detail = max_drawdown(equity)
    returns = period_returns(equity)
    annualized = annualize_return(total_return, sessions, periods_per_year=periods_per_year)
    volatility = (
        float(returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(returns) >= 2 else None
    )

    net_pnls = [float(t.net_pnl) for t in trades]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    closed = len(wins) + len(losses)

    mean_exposure = float(exposure.mean()) if exposure is not None and len(exposure) else 0.0
    mean_equity = float(equity.mean()) if sessions else 0.0

    benchmark_return = None
    benchmark_final = None
    excess = None
    if benchmark is not None and len(benchmark) >= 2:
        first, last = float(benchmark.iloc[0]), float(benchmark.iloc[-1])
        if first:
            benchmark_return = (last - first) / first
            benchmark_final = initial * (1.0 + benchmark_return)
            excess = total_return - benchmark_return

    return PerformanceMetrics(
        initial_capital=initial,
        final_equity=final,
        total_return=total_return,
        annualized_return=annualized,
        cagr=cagr(initial, final, equity.index[0] if sessions else None,
                  equity.index[-1] if sessions else None),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_fees=total_fees,
        num_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=(len(wins) / closed) if closed else None,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else None,
        average_win=(gross_profit / len(wins)) if wins else None,
        average_loss=(-gross_loss / len(losses)) if losses else None,
        expectancy=(sum(net_pnls) / len(net_pnls)) if net_pnls else None,
        max_drawdown=detail.max_drawdown,
        max_drawdown_start=detail.start,
        max_drawdown_end=detail.end,
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate=risk_free_rate,
                                  periods_per_year=periods_per_year),
        sortino_ratio=sortino_ratio(returns, risk_free_rate=risk_free_rate,
                                    periods_per_year=periods_per_year),
        calmar_ratio=calmar_ratio(annualized, detail.max_drawdown),
        exposure=mean_exposure,
        turnover=(traded_notional / mean_equity) if mean_equity else 0.0,
        volatility=volatility,
        benchmark_return=benchmark_return,
        benchmark_final_equity=benchmark_final,
        excess_return=excess,
        sessions=sessions,
        start=str(equity.index[0]) if sessions else None,
        end=str(equity.index[-1]) if sessions else None,
        annualization_days=periods_per_year,
    )


def buy_and_hold_equity(prices: pd.Series, *, initial_capital: float) -> pd.Series:
    """The equity curve of holding one unit of ``prices`` from the first session.

    The benchmark the strategy is compared against. It buys at the first close
    and marks to each subsequent close, so it is comparable session by session
    with the strategy's own curve.
    """
    if prices.empty:
        return pd.Series(dtype=float)
    units = initial_capital / float(prices.iloc[0])
    return (prices * units).rename("buy_and_hold")


def format_metrics(metrics: PerformanceMetrics) -> str:
    """A plain-text summary, for a CLI or a log line."""
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.2%}"

    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:,.2f}"

    lines = [
        f"sessions          {metrics.sessions} ({metrics.start} to {metrics.end})",
        f"initial capital   {metrics.initial_capital:,.2f}",
        f"final equity      {metrics.final_equity:,.2f}",
        f"total return      {pct(metrics.total_return)}",
        f"annualized return {pct(metrics.annualized_return)}",
        f"CAGR              {pct(metrics.cagr)}",
        f"realized P&L      {metrics.realized_pnl:,.2f}",
        f"unrealized P&L    {metrics.unrealized_pnl:,.2f}",
        f"total fees        {metrics.total_fees:,.2f}",
        f"trades            {metrics.num_trades} ({metrics.winning_trades}W / {metrics.losing_trades}L)",
        f"win rate          {pct(metrics.win_rate)}",
        f"profit factor     {num(metrics.profit_factor)}",
        f"average win       {num(metrics.average_win)}",
        f"average loss      {num(metrics.average_loss)}",
        f"expectancy        {num(metrics.expectancy)}",
        f"max drawdown      {metrics.max_drawdown:.2%}",
        f"Sharpe            {num(metrics.sharpe_ratio)}",
        f"Sortino           {num(metrics.sortino_ratio)}",
        f"Calmar            {num(metrics.calmar_ratio)}",
        f"volatility (ann.) {pct(metrics.volatility)}",
        f"exposure (mean)   {metrics.exposure:.2%}",
        f"turnover          {metrics.turnover:,.2f}x",
    ]
    if metrics.benchmark_return is not None:
        lines.append(f"benchmark return  {pct(metrics.benchmark_return)}")
        lines.append(f"excess return     {pct(metrics.excess_return)}")
    return "\n".join(lines)
