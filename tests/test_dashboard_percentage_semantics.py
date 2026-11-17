"""The perpetual dashboard renders research percentages without rescaling."""

from __future__ import annotations

from tradingagents.paper_experiment.dashboard.server import INDEX_HTML


def test_perpetual_win_rate_and_return_are_rendered_as_already_percent_values():
    # The strategy-lab report contains percentage values, not decimal ratios.
    # Rendering them through a ratio formatter would multiply them by 100 twice.
    assert 'fmt(v.win_rate_pct,1)+"%"' in INDEX_HTML
    assert 'fmt(v.return_pct,2)+"%"' in INDEX_HTML
    assert "Number(v).toLocaleString" in INDEX_HTML


def test_unknown_dashboard_metrics_render_as_a_dash_not_nan():
    assert 'v===null||v===undefined||!Number.isFinite(Number(v))' in INDEX_HTML
    assert 'return "—"' in INDEX_HTML


def test_dashboard_no_longer_renders_the_legacy_spot_market_table():
    assert "r.change_percent_24h" not in INDEX_HTML
    assert "pctAlready(" not in INDEX_HTML
    assert "pct(p.total_return)" not in INDEX_HTML


def test_dashboard_uses_the_perpetual_chart_and_position_levels():
    assert 'id="chart"' in INDEX_HTML
    assert 'function drawChart(payload)' in INDEX_HTML
    for level in ("ENTRY", "STOP", "TARGET", "LIQUIDATION"):
        assert f'name:"{level}"' in INDEX_HTML
    assert "function renderPosition(d)" in INDEX_HTML
    assert "@media(max-width:620px)" in INDEX_HTML
    assert "@media(max-width:900px)" in INDEX_HTML


def test_binance_ticker_percentage_parser_keeps_its_feed_convention():
    # This parser remains a library compatibility surface, but it is no longer
    # wired into the Perpetual-only research dashboard.
    from tradingagents.crypto.types import Ticker

    ticker = Ticker.from_binance(
        {"s": "BTCUSDT", "c": "75946.59", "P": "0.091", "p": "68.89"}
    )
    assert ticker.change_percent_24h == 0.091


def test_binance_stream_percentage_parser_keeps_its_feed_convention():
    from tradingagents.crypto.types import Ticker

    ticker = Ticker.from_binance(
        {"s": "ETHUSDT", "c": "2238.7", "P": "-0.117", "p": "-2.62"}
    )
    assert ticker.change_percent_24h == -0.117
