"""Render a crypto TJR study as a Markdown summary.

The report shows the losses as readily as the wins, and every undefined metric as
``n/a`` rather than a zero. It states the session policy, the entry cut and the
data coverage up front, so a reader can tell what was actually run before reading a
single number.
"""

from __future__ import annotations

from .study import CryptoStudyResult, PeriodResult, cost_scenarios

_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("Initial Capital", "initial_capital"),
    ("Final Equity", "final_equity"),
    ("Net Profit/Loss", "_net_pnl"),
    ("Net Return", "_net_return"),
    ("Total Trades", "num_trades"),
    ("Winning Trades", "winning_trades"),
    ("Losing Trades", "losing_trades"),
    ("Win Rate", "_win_rate"),
    ("Profit Factor", "profit_factor"),
    ("Maximum Drawdown", "_drawdown"),
    ("Sharpe Ratio", "sharpe_ratio"),
    ("Sortino Ratio", "sortino_ratio"),
    ("Average R", "average_r"),
    ("Average Risk/Reward", "average_planned_reward_risk"),
    ("Fees", "fees"),
    ("Slippage", "slippage_cost"),
    ("Exposure", "_exposure"),
    ("Turnover", "turnover"),
    ("Rejected Trades", "rejections"),
    ("Buy & Hold Benchmark", "_benchmark"),
)


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:,.2f}%"


def _fmt_num(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _value(metrics: dict, key: str) -> str:
    """One cell of the table, always a string.

    Every branch returns text: a raw number would break the table's ``join`` and
    a missing metric is ``n/a``, never a zero.
    """
    if key == "_net_pnl":
        return _fmt_num(metrics["final_equity"] - metrics["initial_capital"])
    if key == "_net_return":
        return _fmt_pct(metrics["total_return"])
    if key == "_win_rate":
        return _fmt_pct(metrics["win_rate"])
    if key == "_drawdown":
        return _fmt_pct(metrics["max_drawdown"])
    if key == "_exposure":
        return _fmt_pct(metrics["exposure"])
    if key == "_benchmark":
        return _fmt_pct(metrics["benchmark_return"])
    raw = metrics.get(key)
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, float):
        return _fmt_num(raw)
    if isinstance(raw, str):
        return raw
    return "n/a" if raw is None else str(raw)


def _table(rows: list[PeriodResult], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_No result: the data did not cover this view._\n"
    lines = [f"### {title}", "", "| Metric | " + " | ".join(r.label for r in rows) + " |"]
    lines.append("|" + "---|" * (len(rows) + 1))
    for label, key in _METRIC_ROWS:
        cells = " | ".join(_value(r.metrics, key) for r in rows)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"


def render_markdown(study: CryptoStudyResult) -> str:
    """A human-readable summary of the whole study."""
    config = study.config
    lines: list[str] = [
        "# Crypto TJR study",
        "",
        f"- generated: {study.generated_at}",
        "- strategy: TJR (unchanged rule chain)",
        f"- symbols: {', '.join(config.symbols)}",
        f"- initial cash: {config.initial_cash:,.0f} {config.currency}",
        f"- session policy: {config.session_policy.label} ({config.session_policy.timezone})",
        f"- entry-window cut: {config.settings.latest_entry_time[0]:02d}:{config.settings.latest_entry_time[1]:02d} "
        f"(local)",
        "",
    ]
    lines.append("## Cost assumptions (per fill; estimates, not verified Bybit account fees)")
    lines.append("")
    lines.append("| Scenario | Commission | Half-spread | Slippage |")
    lines.append("|---|---:|---:|---:|")
    for label, model in cost_scenarios(config).items():
        lines.append(
            f"| {label} | {model.commission_rate * 10_000:.1f} bps | "
            f"{model.half_spread_bps:.1f} bps | {model.slippage_bps:.1f} bps |"
        )
    lines.append("")
    lines.append(
        "These are explicit research assumptions applied on each fill; actual Bybit "
        "fees and realized spread/slippage depend on the account tier, order type and market."
    )
    lines.append("")
    if study.coverage:
        lines.append("## Data coverage (what the venue actually served)")
        lines.append("")
        lines.append("| Symbol | Bars | First | Last |")
        lines.append("|---|---:|---|---|")
        for symbol, cov in sorted(study.coverage.items()):
            lines.append(
                f"| {symbol} | {cov.get('bars')} | {cov.get('first')} | {cov.get('last')} |"
            )
        lines.append("")
    if study.universe.get("substituted") or study.universe.get("unsupported"):
        lines.append("## Universe resolution")
        lines.append("")
        for requested, resolved in (study.universe.get("substituted") or {}).items():
            lines.append(
                f"- {requested} -> {resolved} (the venue serves this quote, not the requested one)"
            )
        for requested, reason in (study.universe.get("unsupported") or {}).items():
            lines.append(f"- {requested}: not tradable ({reason})")
        lines.append("")

    lines.append(_table(study.periods, "Periods (configured base cost schedule)"))
    lines.append("")
    for scenario, rows in study.period_cost_scenarios.items():
        lines.append(_table(rows, f"Periods after {scenario} costs"))
        lines.append("")
    lines.append(_table(study.walk_forward, "Walk-forward (configured base cost schedule)"))
    lines.append("")
    for scenario, rows in study.walk_forward_cost_scenarios.items():
        lines.append(_table(rows, f"Walk-forward folds after {scenario} costs"))
        lines.append("")
    lines.append(_table(study.regimes, "Market regimes"))
    lines.append("")
    lines.append(_table(study.cost_scenarios, "Cost scenarios"))
    lines.append("")
    if study.sensitivity:
        lines.append(_table(study.sensitivity, "Sensitivity (one input changed at a time)"))
        lines.append("")

    if study.per_symbol:
        lines.append("## Per-symbol (full window)")
        lines.append("")
        lines.append("| Symbol | Trades | Net Return | Win Rate | Max DD | Buy & Hold |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for symbol, metrics in sorted(study.per_symbol.items()):
            lines.append(
                f"| {symbol} | {metrics['num_trades']} | {_fmt_pct(metrics['total_return'])} | "
                f"{_fmt_pct(metrics['win_rate'])} | {_fmt_pct(metrics['max_drawdown'])} | "
                f"{_fmt_pct(metrics['benchmark_return'])} |"
            )
        lines.append("")

    lines.append("## Determinism")
    lines.append("")
    det = study.determinism
    lines.append(
        f"- two runs identical: **{det.get('identical')}** "
        f"(final equity {det.get('final_equity')}, trades {det.get('num_trades')})"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in study.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_markdown"]
