"""Reports for simulated runs: one document, rendered to HTML and JSON.

``backtest_report`` builds a report from a backtest result and renders it. The
document is built once as a dict and both renderers read it, so the HTML and the
JSON cannot disagree about what happened.
"""

from tradingagents.trading_report.backtest_report import (
    ReportSection,
    build_report,
    decision_section,
    execution_section,
    performance_section,
    portfolio_section,
    render_html,
    write_report,
)

__all__ = [
    "ReportSection",
    "build_report",
    "decision_section",
    "execution_section",
    "performance_section",
    "portfolio_section",
    "render_html",
    "write_report",
]
