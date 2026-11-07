"""Reports: the document, its renderers, and whether it agrees with the book.

The test that matters most is agreement. A report that quietly disagreed with the
ledger would be the worst kind of bug, because it would be read and believed. So
the report's portfolio figures are checked against the ledger's own totals, and
its performance figures against the metrics object, rather than against literals
that could drift.

The other tests pin the two conventions that make a report honest: an undefined
metric renders as "n/a" and never as a number, and the total return is presented
as-is rather than annualised.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.backtesting import (
    BacktestConfig,
    MarketDataFeed,
    PortfolioBacktester,
)
from tradingagents.execution import CostModel
from tradingagents.performance import compute_metrics
from tradingagents.risk import RiskLimits
from tradingagents.signals import SignalAction, TradingSignal
from tradingagents.trading_report import build_report, render_html, write_report


def _result(*, costs=None, with_rejection=False):
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    closes = [100.0, 101.0, 103.0, 104.0, 106.0]
    frame = pd.DataFrame(
        {"open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
         "close": closes, "volume": [1_000_000] * len(dates)},
        index=pd.to_datetime(dates),
    )
    feed = MarketDataFeed({"AAPL": frame})

    def strategy(context):
        if context.session != "2026-01-05":
            return None
        return TradingSignal(
            symbol="AAPL", action=SignalAction.BUY, confidence=0.75,
            timestamp=datetime(2026, 1, 5, 21, tzinfo=UTC), as_of_date="2026-01-05",
            position_size=0.5, stop_loss=90.0, take_profit=120.0,
            reasoning="The fundamentals analyst saw improving margins.",
        )

    limits = RiskLimits(max_position_size=0.5, max_portfolio_exposure=None)
    if with_rejection:
        limits = RiskLimits(max_position_size=0.5, max_portfolio_exposure=None,
                            min_order_value=1_000_000.0)
    config = BacktestConfig(initial_cash=100_000.0, holding_period_days=2,
                            costs=costs or CostModel(), risk_limits=limits)
    engine = PortfolioBacktester(feed, strategy, config=config)
    return engine.run()


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReport:
    def test_the_report_names_the_run_and_its_sessions(self):
        result = _result()
        report = build_report(result)
        assert report["run_id"] == result.run_id
        assert report["sessions"]["count"] == 5
        assert report["sessions"]["start"] == "2026-01-05"

    def test_the_sections_are_in_the_order_a_reader_asks_the_questions(self):
        report = build_report(_result())
        titles = [s["title"] for s in report["sections"]]
        assert titles[:4] == ["AI decisions", "Execution", "Portfolio", "Performance"]

    def test_the_ai_decisions_are_reported(self):
        report = build_report(_result())
        decisions = next(s for s in report["sections"] if s["title"] == "AI decisions")
        assert decisions["rows"]
        assert any("BUY" in row[0] for row in decisions["rows"])

    def test_the_reasoning_is_kept_verbatim(self):
        """The model's own words, not a paraphrase."""
        report = build_report(_result())
        assert report["signals"][0]["reasoning"] == (
            "The fundamentals analyst saw improving margins."
        )

    def test_the_execution_section_renders_a_trade_table(self):
        report = build_report(_result())
        execution = next(s for s in report["sections"] if s["title"] == "Execution")
        assert "<table>" in execution["body"]
        assert "AAPL" in execution["body"]

    def test_a_run_with_no_trades_says_so(self):
        result = _result(with_rejection=True)
        report = build_report(result)
        execution = next(s for s in report["sections"] if s["title"] == "Execution")
        assert execution["rows"] == [["Trades", "none"]]

    def test_a_refusal_is_reported_with_its_reason(self):
        report = build_report(_result(with_rejection=True))
        refusals = next((s for s in report["sections"]
                         if s["title"] == "Refusals and unfilled orders"), None)
        assert refusals is not None
        assert refusals["rows"]

    def test_no_refusals_means_no_refusal_section(self):
        report = build_report(_result())
        assert not any(s["title"] == "Refusals and unfilled orders"
                       for s in report["sections"])

    def test_the_metadata_is_carried_through(self):
        report = build_report(_result(), metadata={"model": {"model": "gpt-5"}})
        assert report["metadata"]["model"]["model"] == "gpt-5"

    def test_the_report_is_json_safe(self):
        json.dumps(build_report(_result()))


# ---------------------------------------------------------------------------
# Agreement with the ledger and the metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgreement:
    def test_the_reported_equity_is_the_ledgers_equity(self):
        result = _result()
        report = build_report(result)
        portfolio = next(s for s in report["sections"] if s["title"] == "Portfolio")
        rows = dict(portfolio["rows"])
        assert rows["Final equity"] == f"{result.ledger.equity:,.2f}"

    def test_the_reported_fees_are_the_ledgers_fees(self):
        result = _result(costs=CostModel(commission_rate=0.001))
        report = build_report(result)
        portfolio = next(s for s in report["sections"] if s["title"] == "Portfolio")
        rows = dict(portfolio["rows"])
        assert rows["Total fees"] == f"{result.ledger.total_fees:,.2f}"

    def test_the_reported_metrics_match_compute_metrics(self):
        result = _result()
        report = build_report(result)
        expected = compute_metrics(
            result.equity_series(), initial_capital=result.initial_cash,
            trades=result.trades,
        )
        assert report["metrics"]["total_return"] == pytest.approx(expected.total_return)
        assert report["metrics"]["max_drawdown"] == pytest.approx(expected.max_drawdown)

    def test_a_supplied_metrics_object_is_used_verbatim(self):
        result = _result()
        supplied = compute_metrics(result.equity_series(), initial_capital=result.initial_cash,
                                   risk_free_rate=0.02)
        report = build_report(result, metrics=supplied)
        assert report["metrics"]["total_return"] == pytest.approx(supplied.total_return)

    def test_every_trade_appears_in_the_report(self):
        result = _result()
        report = build_report(result)
        assert len(report["trades"]) == len(result.trades)
        assert report["trades"][0]["symbol"] == "AAPL"

    def test_every_signal_appears_in_the_report(self):
        result = _result()
        report = build_report(result)
        assert len(report["signals"]) == len(result.signals)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHtml:
    def test_the_html_is_self_contained(self):
        """No external asset, so it opens on Android or a Colab download."""
        html = render_html(build_report(_result()))
        assert "<style>" in html
        assert "http://" not in html
        assert "https://" not in html
        assert "<script" not in html

    def test_the_html_names_the_run(self):
        result = _result()
        html = render_html(build_report(result))
        assert result.run_id in html

    def test_the_html_shows_every_section(self):
        report = build_report(_result())
        html = render_html(report)
        for section in report["sections"]:
            assert section["title"] in html

    def test_an_undefined_metric_renders_as_na(self):
        """A flat run has no Sharpe; the report must not print a number for it."""
        result = _result()
        report = build_report(result)
        report["metrics"]["sharpe_ratio"] = None
        report["metrics"]["sortino_ratio"] = None
        html = render_html(report)
        assert "n/a" in html

    def test_the_total_return_is_not_presented_as_annualized(self):
        report = build_report(_result())
        html = render_html(report)
        assert "not annualised" in html

    def test_model_text_is_escaped(self):
        """Reasoning is model output; it must not be able to inject markup."""
        result = _result()
        result.signals[0].reasoning = "<script>alert(1)</script>"
        report = build_report(result)
        html = render_html(report)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_the_html_is_valid_enough_to_open(self):
        html = render_html(build_report(_result()))
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFiles:
    def test_both_files_are_written(self, tmp_path):
        paths = write_report(build_report(_result()), tmp_path / "out")
        assert paths["json"].exists()
        assert paths["html"].exists()

    def test_the_json_round_trips(self, tmp_path):
        report = build_report(_result())
        paths = write_report(report, tmp_path, stem="run")
        assert json.loads(paths["json"].read_text())["run_id"] == report["run_id"]

    def test_the_written_html_matches_the_renderer(self, tmp_path):
        report = build_report(_result())
        paths = write_report(report, tmp_path, stem="run")
        assert paths["html"].read_text() == render_html(report)

    def test_a_stem_can_be_chosen(self, tmp_path):
        paths = write_report(build_report(_result()), tmp_path, stem="aapl_2026")
        assert paths["json"].name == "aapl_2026.json"


@pytest.mark.unit
class TestDeterminism:
    def test_two_reports_of_the_same_result_agree(self):
        """Apart from the generation timestamp, which is the only clock read."""
        first = build_report(_result())
        second = build_report(_result())
        first.pop("generated_at")
        second.pop("generated_at")
        assert first == second


@pytest.mark.unit
class TestExposureAndTurnoverAreMeasured:
    """The report's exposure and turnover must reflect the run, not read zero.

    Regression test for a real defect: ``build_report`` computed metrics with a
    direct ``compute_metrics`` call that passed no ``exposure`` series and no
    ``traded_notional``, so the report always showed a mean exposure of 0 and a
    turnover of 0 even for a fully invested run. It now uses
    ``metrics_from_result``, which supplies both from the equity curve and the
    fill events.
    """

    def test_a_fully_invested_run_reports_nonzero_exposure(self):
        report = build_report(_result())
        assert report["metrics"]["exposure"] > 0.0

    def test_the_report_exposure_matches_the_equity_curve(self):
        result = _result()
        report = build_report(result)
        mean_exposure = sum(p.exposure for p in result.equity_curve) / len(result.equity_curve)
        assert report["metrics"]["exposure"] == pytest.approx(mean_exposure)

    def test_a_run_that_traded_reports_nonzero_turnover(self):
        report = build_report(_result())
        assert report["metrics"]["turnover"] > 0.0
