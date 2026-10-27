"""Reports: one run, in HTML and JSON, showing decision through to performance.

A backtest result is a pile of objects. A report is what a person reads, so this
module answers the questions in the order they are asked:

- **What did the AI decide?** Each signal, its action, its confidence and the
  model's own reasoning, quoted rather than paraphrased.
- **What did the risk engine do with it?** Approved, clamped or refused, and the
  named reason. A refusal is as interesting as a fill.
- **What actually executed?** The order, its fill price, quantity and fees. An
  order that did not fill is shown, with why.
- **Where did the book end up?** Cash, positions, equity, exposure and P&L, read
  from the ledger so the report cannot disagree with the book.
- **How did it perform?** Return, drawdown, Sharpe, Sortino, win rate, profit
  factor, with a benchmark if one was supplied.

Two conventions carried over from the performance module, because a report is
where they matter most: an undefined metric is rendered "n/a" rather than as a
number, and the total return is not annualised.

The HTML is a single self-contained file with no external assets, so it opens
from an Android device or a Colab download with no server and no network.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.performance.metrics import PerformanceMetrics, compute_metrics


@dataclass
class ReportSection:
    """One named block of a report, rendered in order."""

    title: str
    rows: list[tuple[str, str]] = field(default_factory=list)
    body: str = ""
    """Pre-rendered HTML for a block that is not key/value (a trade table, say)."""


def _fmt_money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.3f}"


def _fmt_pct_plain(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def decision_section(result) -> ReportSection:
    """Every AI decision, with its reasoning quoted verbatim and escaped.

    The reasoning is model output and is rendered as text, never as markup: it is
    escaped on the way in, so a response containing HTML cannot inject into the
    report.
    """
    rows: list[tuple[str, str]] = []
    reasoning_blocks: list[str] = []
    for signal in result.signals:
        detail = f"confidence {signal.confidence:.0%}"
        if signal.position_size is not None:
            detail += f" · size {signal.position_size:.1%}"
        if signal.stop_loss is not None:
            detail += f" · stop {signal.stop_loss:,.2f}"
        if signal.take_profit is not None:
            detail += f" · target {signal.take_profit:,.2f}"
        rows.append((f"{signal.symbol} {signal.action.value} ({signal.as_of_date})", detail))
        if signal.reasoning:
            reasoning_blocks.append(
                f"<h3>{html.escape(signal.symbol)} {html.escape(signal.action.value)} "
                f"({html.escape(signal.as_of_date)})</h3>"
                f"<pre>{html.escape(signal.reasoning)}</pre>"
            )
    body = "".join(reasoning_blocks)
    return ReportSection(title="AI decisions", rows=rows, body=body)


def execution_section(result) -> ReportSection:
    """Every round trip, with the costs it carried."""
    if not result.trades:
        return ReportSection(title="Execution", rows=[("Trades", "none")])
    header = ("Symbol", "Entry", "Exit", "Qty", "Entry px", "Exit px", "Fees", "Net P&L", "Reason")
    body_rows = []
    for trade in result.trades:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(trade.symbol)}</td>"
            f"<td>{html.escape(trade.entry_session)}</td>"
            f"<td>{html.escape(trade.exit_session)}</td>"
            f"<td class='num'>{trade.quantity:,.4f}</td>"
            f"<td class='num'>{trade.entry_price:,.4f}</td>"
            f"<td class='num'>{trade.exit_price:,.4f}</td>"
            f"<td class='num'>{trade.fees:,.2f}</td>"
            f"<td class='num {'pos' if trade.net_pnl >= 0 else 'neg'}'>{trade.net_pnl:,.2f}</td>"
            f"<td>{html.escape(trade.reason)}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in header)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    return ReportSection(title="Execution", body=table)


def _rejection_rows(result) -> list[tuple[str, str]]:
    rows = [(f"{session} {symbol}", reason) for session, symbol, reason in result.rejections]
    rows += [
        (f"{session} {symbol} (unfilled)", reason) for session, symbol, reason in result.unfilled
    ]
    return rows


def portfolio_section(result) -> ReportSection:
    """Where the book ended up, read from the ledger."""
    ledger = result.ledger
    rows: list[tuple[str, str]] = [
        ("Initial cash", _fmt_money(result.initial_cash)),
        ("Final equity", _fmt_money(result.final_equity)),
    ]
    if ledger is not None:
        rows += [
            ("Cash", _fmt_money(ledger.cash)),
            ("Market value", _fmt_money(ledger.market_value)),
            ("Exposure", _fmt_money(ledger.total_exposure)),
            ("Realized P&L", _fmt_money(ledger.realized_pnl)),
            ("Unrealized P&L", _fmt_money(ledger.unrealized_pnl)),
            ("Total fees", _fmt_money(ledger.total_fees)),
        ]
        for symbol, holding in ledger.open_positions().items():
            rows.append((
                f"Position {symbol}",
                f"{holding.quantity:,.4f} @ {holding.average_entry_price:,.4f} "
                f"(value {holding.market_value:,.2f})",
            ))
    return ReportSection(title="Portfolio", rows=rows)


def performance_section(metrics: PerformanceMetrics) -> ReportSection:
    """The measured performance, with undefined metrics shown as n/a."""
    rows = [
        ("Total return", _fmt_pct(metrics.total_return)),
        ("Annualized return", _fmt_pct(metrics.annualized_return)),
        ("CAGR", _fmt_pct(metrics.cagr)),
        ("Max drawdown", _fmt_pct_plain(metrics.max_drawdown)),
        ("Sharpe ratio", _fmt_num(metrics.sharpe_ratio)),
        ("Sortino ratio", _fmt_num(metrics.sortino_ratio)),
        ("Calmar ratio", _fmt_num(metrics.calmar_ratio)),
        ("Volatility (annualized)", _fmt_pct(metrics.volatility)),
        ("Trades", f"{metrics.num_trades} ({metrics.winning_trades}W / {metrics.losing_trades}L)"),
        ("Win rate", _fmt_pct_plain(metrics.win_rate)),
        ("Profit factor", _fmt_num(metrics.profit_factor)),
        ("Average win", _fmt_money(metrics.average_win)),
        ("Average loss", _fmt_money(metrics.average_loss)),
        ("Expectancy", _fmt_money(metrics.expectancy)),
        ("Mean exposure", _fmt_pct_plain(metrics.exposure)),
        ("Turnover", f"{metrics.turnover:,.2f}x"),
    ]
    if metrics.benchmark_return is not None:
        rows += [
            ("Benchmark return", _fmt_pct(metrics.benchmark_return)),
            ("Excess return", _fmt_pct(metrics.excess_return)),
        ]
    return ReportSection(title="Performance", rows=rows)


def build_report(
    result,
    *,
    title: str = "Backtest report",
    metrics: PerformanceMetrics | None = None,
    metadata: dict | None = None,
) -> dict:
    """The report as a JSON-safe dict. The single source for both renderers.

    Building the structure once and rendering from it means the HTML and the JSON
    cannot drift: they are two views of one document.
    """
    metrics = metrics or compute_metrics(
        result.equity_series(),
        initial_capital=result.initial_cash,
        trades=result.trades,
        realized_pnl=result.ledger.realized_pnl if result.ledger else 0.0,
        unrealized_pnl=result.ledger.unrealized_pnl if result.ledger else 0.0,
        total_fees=result.ledger.total_fees if result.ledger else 0.0,
    )
    sections = [
        decision_section(result),
        execution_section(result),
        portfolio_section(result),
        performance_section(metrics),
    ]
    rejections = _rejection_rows(result)
    if rejections:
        sections.append(ReportSection(title="Refusals and unfilled orders", rows=rejections))

    return {
        "title": title,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": result.run_id,
        "sessions": {
            "count": len(result.sessions),
            "start": result.sessions[0] if result.sessions else None,
            "end": result.sessions[-1] if result.sessions else None,
        },
        "metrics": metrics.to_dict(),
        "metadata": metadata or {},
        "sections": [
            {"title": s.title, "rows": [[k, v] for k, v in s.rows], "body": s.body}
            for s in sections
        ],
        "trades": [
            {
                "symbol": t.symbol,
                "entry_session": t.entry_session,
                "exit_session": t.exit_session,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "fees": t.fees,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_pct,
                "reason": t.reason,
            }
            for t in result.trades
        ],
        "signals": [
            {
                "signal_id": s.signal_id,
                "symbol": s.symbol,
                "action": s.action.value,
                "confidence": s.confidence,
                "as_of_date": s.as_of_date,
                "position_size": s.position_size,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "reasoning": s.reasoning,
            }
            for s in result.signals
        ],
    }


_STYLE = """
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 60rem; padding: 1.5rem; color: #1a1a1a;
       background: #fdfdfd; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: .25rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #e0e0e0;
     padding-bottom: .25rem; }
.sub { color: #666; font-size: .85rem; margin-top: 0; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin-top: .5rem; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eee; }
th { background: #f5f5f5; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: #1a7f37; }
.neg { color: #b42318; }
.kv td:first-child { color: #555; width: 40%; }
h3 { font-size: .95rem; margin: 1rem 0 .25rem; }
pre { background: #f6f6f6; padding: .5rem; overflow-x: auto; font-size: .8rem;
      white-space: pre-wrap; }
"""


def _render_section(section: dict) -> str:
    parts = [f"<h2>{html.escape(section['title'])}</h2>"]
    if section["rows"]:
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in section["rows"]
        )
        parts.append(f"<table class='kv'><tbody>{rows}</tbody></table>")
    if section["body"]:
        parts.append(section["body"])
    return "".join(parts)


def render_html(report: dict) -> str:
    """Render the report dict as one self-contained HTML document."""
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(report['title'])}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>{html.escape(report['title'])}</h1>",
        f"<p class='sub'>run {html.escape(report['run_id'])} · "
        f"{report['sessions']['count']} sessions "
        f"({report['sessions']['start']} to {report['sessions']['end']}) · "
        f"generated {html.escape(report['generated_at'])}</p>",
    ]
    metrics = report["metrics"]
    if metrics.get("total_return") is not None:
        direction = "pos" if metrics["total_return"] >= 0 else "neg"
        parts.append(
            f"<p>Total return <span class='{direction}'>"
            f"{metrics['total_return']:+.2%}</span>, "
            f"max drawdown {metrics['max_drawdown']:.2%}. "
            "The total return is not annualised; the annualized figure is a separate "
            "metric with a stated session assumption.</p>"
        )
    for section in report["sections"]:
        parts.append(_render_section(section))
    if report.get("metadata"):
        parts.append("<h2>Run metadata</h2>")
        parts.append(f"<pre>{html.escape(json.dumps(report['metadata'], indent=2, default=str))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_report(report: dict, directory: str | Path, *, stem: str = "report") -> dict[str, Path]:
    """Write the report as ``<stem>.json`` and ``<stem>.html``; return both paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{stem}.json"
    html_path = target / f"{stem}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}
