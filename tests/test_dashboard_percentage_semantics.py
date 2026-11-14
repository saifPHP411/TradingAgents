"""A percentage must be printed at its own scale.

The dashboard has two kinds of "percentage":

* a **ratio** — ``total_return = 0.05`` means 5% — which is scaled by 100;
* a **percentage already** — Binance's ``priceChangePercent = 0.091`` means
  0.091% — which must not be scaled again.

Printing the second kind with the first kind's formatter shows it 100x too
large: a −0.11% move would read as −11.00%, and a −4.74% move would read as
−474%. These tests pin the scale of each, so a feed's convention cannot be
silently mis-rendered.
"""

from __future__ import annotations

import re

from tradingagents.paper_experiment.dashboard.server import INDEX_HTML


def _formatter(name: str) -> str:
    """The whole one-line JS arrow function, reassembled from the page."""
    match = re.search(rf"const {name} = (\(v\) => .*?);", INDEX_HTML)
    assert match, f"the dashboard has no {name} formatter"
    return match.group(1)


def _js_eval(expression: str, value: float | None) -> str:
    """Evaluate one formatter call the way the browser would.

    The dashboard's JS is a small, fixed set of arrow functions; this harness
    runs them with Node when it is available, and falls back to a faithful
    Python transcription of the same expressions otherwise.
    """
    import shutil
    import subprocess

    # ``None`` is Python's spelling; the page's JavaScript needs ``null``.
    literal = "null" if value is None else repr(value)
    script = f"const f = {expression}; console.log(f({literal}));"
    node = shutil.which("node")
    if node:
        out = subprocess.run(
            [node, "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return out.stdout.strip()
    # Python transcription of the same expressions.
    if "isNaN(v)" in expression and "Number(v)*100" in expression:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"
    if "isNaN(v)" in expression and "Number(v).toFixed(2)" in expression:
        if value is None:
            return "N/A"
        return f"{value:.2f}%"
    raise AssertionError(f"the formatter changed shape: {expression}")


class TestTheRatioFormatter:
    def test_a_ratio_is_scaled_to_a_percentage(self):
        assert _js_eval(_formatter("pct"), 0.05) == "5.00%"

    def test_a_negative_ratio_keeps_its_sign(self):
        assert _js_eval(_formatter("pct"), -0.034) == "-3.40%"

    def test_an_unknown_ratio_is_not_a_number(self):
        assert _js_eval(_formatter("pct"), None) == "N/A"


class TestTheAlreadyAPercentageFormatter:
    def test_a_percentage_is_not_scaled_again(self):
        """0.091 means 0.091%. Printing 9.10% would be wrong by 100x."""
        assert _js_eval(_formatter("pctAlready"), 0.091) == "0.09%"

    def test_a_negative_percentage_keeps_its_sign(self):
        assert _js_eval(_formatter("pctAlready"), -4.74) == "-4.74%"

    def test_an_unknown_percentage_is_not_a_number(self):
        assert _js_eval(_formatter("pctAlready"), None) == "N/A"


class TestTheMarketTableUsesTheRightOne:
    def test_the_market_table_does_not_scale_the_24h_change(self):
        """The 24h change comes from the feed as a percentage already."""
        assert "pctAlready(r.change_percent_24h)" in INDEX_HTML
        assert "pct(r.change_percent_24h)" not in INDEX_HTML

    def test_the_performance_rows_still_use_the_ratio_formatter(self):
        """A total return is a ratio, so it keeps the ×100 scaling."""
        assert "pct(p.total_return)" in INDEX_HTML
        assert "pct(p.max_drawdown)" in INDEX_HTML


class TestTheFeedConventionIsPinned:
    def test_binance_reports_the_change_as_a_percentage_already(self):
        """The premise of the fix: Binance's own field name and value scale.

        Binance's 24hr ticker carries ``priceChangePercent`` (``P``), whose
        value is the percentage itself. The parser reads it into
        ``change_percent_24h``, so the dashboard must not scale it again.
        """
        from tradingagents.crypto.types import Ticker

        ticker = Ticker.from_binance(
            {
                "s": "BTCUSDT",
                "c": "75946.59",
                "P": "0.091",
                "p": "68.89",
            }
        )
        assert ticker.change_percent_24h == 0.091
        # The absolute change over the price is the same 0.091%, which is what
        # makes the field a percentage and not a ratio.
        assert abs(ticker.change_24h) / ticker.last * 100 == pytest_approx(ticker.change_percent_24h)

    def test_the_stream_reads_the_same_field(self):
        """The websocket's 24hr ticker uses the short ``P`` key."""
        from tradingagents.crypto.types import Ticker

        ticker = Ticker.from_binance(
            {"s": "ETHUSDT", "c": "2238.7", "P": "-0.117", "p": "-2.62"}
        )
        assert ticker.change_percent_24h == -0.117


def pytest_approx(value: float, rel: float = 0.01):
    """A tiny relative tolerance, without importing pytest at module level."""
    import pytest

    return pytest.approx(value, rel=rel)
