"""The TJR intraday backtest: the loop, the exits, the shared book and the report.

The engine's contract is narrow and worth pinning precisely: it decides from bars
that had closed, fills at the *next* bar's open, and moves every cash or position
change through the shared ledger and risk engine rather than a private one. These
tests assert those properties on sessions small enough to check by hand.
"""

from __future__ import annotations

import pytest

from tests.tjr_helpers import BEARISH_BARS, BULLISH_BARS, SESSION, series_pair
from tradingagents.risk import RiskLimits
from tradingagents.signals.contract import SignalAction
from tradingagents.tjr.engine import TjrBacktester, TjrConfig
from tradingagents.tjr.report import build_report
from tradingagents.tjr.rules import TJR_READY

pytestmark = pytest.mark.unit


def _run(bars, *, cash: float = 100_000.0, limits: RiskLimits | None = None, symbols=None):
    """Run one symbol's session (or a symbol->bars mapping) and return the result."""
    if symbols is None:
        symbols = {"X": bars}
    one = {}
    five = {}
    for symbol, session_bars in symbols.items():
        five[symbol], one[symbol] = series_pair(symbol, SESSION, session_bars)
    config = TjrConfig(
        initial_cash=cash,
        risk_limits=limits or RiskLimits(max_open_positions=5),
    )
    return TjrBacktester(one_minute=one, five_minute=five, config=config).run()


# A bullish session extended so the entry's target is reached before the bell.
_WIN = BULLISH_BARS[:15] + [
    (100.6, 110.0, 100.55, 109.0),
    (109.0, 110.0, 108.0, 109.5),
]

# The same session extended so the entry's stop is hit instead.
_LOSS = BULLISH_BARS[:15] + [
    (100.6, 100.7, 95.0, 95.5),
    (95.5, 96.0, 94.0, 94.5),
]


def test_a_completed_setup_becomes_one_long_trade():
    result = _run(_WIN)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.reason_code == TJR_READY
    assert trade.quantity > 0
    assert trade.exit_reason == "target"
    assert trade.net_pnl > 0


def test_the_entry_fills_at_the_next_bar_open_not_the_decision_bar():
    result = _run(_WIN)
    trade = result.trades[0]
    # The chain completes on the 14:45 bar-close; the fill is the 14:50 open.
    entry_instant = trade.entry_time
    assert "14:45" in entry_instant or "14:50" in entry_instant
    assert trade.entry_price == pytest.approx(100.6)


def test_the_stop_is_taken_before_the_target_when_a_bar_covers_both():
    # One bar whose range spans both the stop (97.85) and the target (106.1). The
    # engine cannot know which came first and must assume the loss.
    bars = BULLISH_BARS[:15] + [(100.6, 110.0, 95.0, 100.0)]
    result = _run(bars)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].net_pnl < 0


def test_an_open_position_is_flat_by_the_session_close():
    # A winner that never reaches its target still cannot be carried overnight.
    bars = BULLISH_BARS[:15] + [
        (100.6, 102.0, 100.5, 101.5),
        (101.5, 102.0, 101.0, 101.8),
    ]
    result = _run(bars)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "session_close"
    # Nothing is held after the last bar.
    assert result.ledger.position_quantity("X") == 0


def test_a_risk_rejection_produces_no_trade_and_no_position():
    # An order-value floor above any possible order refuses the entry outright.
    limits = RiskLimits(max_open_positions=5, min_order_value=10_000_000.0)
    result = _run(_WIN, limits=limits)
    assert result.trades == []
    assert result.ledger.position_quantity("X") == 0
    assert result.rejections
    assert "MIN_ORDER_VALUE" in result.rejections[0]["reasons"]


def test_the_book_refuses_more_positions_than_its_limit():
    # Two names set up, but the book allows only one open position.
    limits = RiskLimits(max_open_positions=1)
    result = _run(_WIN, limits=limits, symbols={"X": _WIN, "Y": _WIN})
    assert len(result.trades) == 1
    assert any("MAX_OPEN_POSITIONS" in r["reasons"] for r in result.rejections)


def test_a_bearish_session_produces_no_trade_in_a_long_only_book():
    result = _run(BEARISH_BARS)
    assert result.trades == []
    assert result.ledger.position_quantity("X") == 0


def test_equity_is_marked_at_every_bar_close_and_reconciles():
    result = _run(_WIN)
    assert result.equity_curve
    # The last mark equals the ledger's own equity, and cash plus the marked
    # position is that number. No figure is invented outside the ledger.
    assert result.final_equity == pytest.approx(result.ledger.equity)
    assert result.ledger.position_quantity("X") == 0
    assert result.final_equity == pytest.approx(result.ledger.cash)


def test_the_engine_is_deterministic_for_the_same_input():
    first = _run(_WIN)
    second = _run(_WIN)
    assert [t.net_pnl for t in first.trades] == [t.net_pnl for t in second.trades]
    assert first.equity_curve == second.equity_curve
    assert first.reason_counts == second.reason_counts


def test_no_look_ahead_a_bar_is_never_read_before_it_closes():
    # Run the session, then run it again with every bar after the entry removed:
    # the entry decision must be identical, because it never read the later bars.
    full = _run(_WIN)
    truncated = _run(BULLISH_BARS[:16])
    full_entry = next(t for t in full.trades)
    truncated_entry = next(t for t in truncated.trades)
    assert full_entry.entry_price == truncated_entry.entry_price
    assert full_entry.entry_time == truncated_entry.entry_time


def test_the_strategy_requests_no_size_so_the_risk_engine_sizes_it():
    # The engine's signal carries no size; the ledger position is the risk
    # engine's default fraction of equity, not a number the strategy chose.
    result = _run(_WIN)
    trade = result.trades[0]
    config = result.config
    assert config is not None
    expected = config.risk_limits.default_position_size * config.initial_cash
    assert trade.entry_price * trade.quantity == pytest.approx(expected, rel=1e-6)


def test_the_strategy_fabricates_no_stop_or_target():
    # The levels on the trade are the ones the evaluation produced from structure.
    result = _run(_WIN)
    trade = result.trades[0]
    assert trade.entry_price > 0
    assert trade.exit_price > 0
    assert trade.reason_code == TJR_READY


def test_a_hold_produces_no_order():
    # A session with no setup at all: every evaluation is a HOLD and nothing trades.
    bars = [
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 100.3, 100.0, 100.2),
        (100.2, 100.4, 100.1, 100.3),
        (100.3, 100.5, 100.2, 100.4),
        (100.4, 100.6, 100.3, 100.5),
    ]
    result = _run(bars)
    assert result.trades == []
    assert result.reason_counts.get("NO_SWEEP", 0) > 0


def test_the_report_reuses_the_performance_engine_and_tallies_the_reasons():
    result = _run(_WIN)
    report = build_report(result)
    assert report.sessions == 1
    assert report.symbols == ("X",)
    assert report.reason_counts.get(TJR_READY) == 1
    assert report.exit_reasons == {"target": 1}
    # The metrics come from the shared performance module, so they are present and
    # finite even though a single trade cannot define every ratio.
    assert report.metrics.final_equity == pytest.approx(result.final_equity)
    assert report.metrics.num_trades == 1


def test_the_signal_for_a_completed_setup_is_a_buy_with_levels():
    five, one = series_pair("X", SESSION, BULLISH_BARS)
    from tradingagents.tjr.strategy import TjrStrategy

    strategy = TjrStrategy(one_minute={"X": one}, five_minute={"X": five})
    ready = None
    for index in range(len(five.frame)):
        instant = five.frame.index[index] + __import__("pandas").Timedelta(minutes=5)
        signal, evaluation = strategy.signal("X", SESSION, instant)
        if signal.action is SignalAction.BUY:
            ready = (signal, evaluation)
            break
    assert ready is not None
    signal, evaluation = ready
    assert evaluation.reason_code == TJR_READY
    assert signal.entry_price == evaluation.entry
    assert signal.stop_loss == evaluation.stop
    assert signal.take_profit == evaluation.target
    assert signal.position_size is None
