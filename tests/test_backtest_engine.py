"""The portfolio backtester: the event loop, its scheduling, and its accounting.

The scheduling tests are the important ones. A backtest's most expensive bug is
silent: if a decision fills at the price it already saw, every result is too good
and nothing looks wrong. So the tests pin the exact sessions and prices:

    decide on T  ->  order at T's close  ->  fill at T+1's open

and they pin the exit schedule the same way. The accounting tests then check that
the book the loop produces satisfies the same conservation identity the ledger
does on its own, so the engine cannot introduce money the ledger would not.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.backtesting import (
    BacktestConfig,
    MarketDataFeed,
    PortfolioBacktester,
)
from tradingagents.execution import CostModel
from tradingagents.risk import RiskLimits
from tradingagents.signals import SignalAction, TradingSignal

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]


def _feed(symbol="AAPL", *, opens=None, closes=None, highs=None, lows=None, dates=DATES):
    """A coherent bar frame: high/low default to the extremes of open and close.

    Defaults are derived rather than guessed so the bars are always internally
    consistent (``low <= open, close <= high``), which the ``Bar`` model requires.
    An explicit ``highs``/``lows`` still has to be coherent with them.
    """
    closes = closes if closes is not None else [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    opens = opens if opens is not None else closes
    if highs is None:
        highs = [max(o, c) * 1.005 for o, c in zip(opens, closes, strict=True)]
    if lows is None:
        lows = [min(o, c) * 0.995 for o, c in zip(opens, closes, strict=True)]
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * len(dates)},
        index=pd.to_datetime(dates),
    )
    return MarketDataFeed({symbol: frame})


def _signal(session, symbol="AAPL", action=SignalAction.BUY, **kwargs) -> TradingSignal:
    return TradingSignal(
        symbol=symbol, action=action, confidence=0.8,
        timestamp=datetime.strptime(session, "%Y-%m-%d").replace(hour=21, tzinfo=UTC),
        as_of_date=session, **kwargs,
    )


def _buy_once(session_to_buy="2026-01-05", **signal_kwargs):
    """A strategy that buys on one session and does nothing else."""
    def strategy(context):
        if context.session == session_to_buy:
            return _signal(context.session, **signal_kwargs)
        return None
    return strategy


def _run(strategy, feed=None, config=None, start=None, end=None):
    feed = feed or _feed()
    engine = PortfolioBacktester(feed, strategy, config=config)
    return engine.run(start, end), engine


# ---------------------------------------------------------------------------
# Scheduling: the look-ahead guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScheduling:
    def test_a_decision_fills_at_the_next_session_open_not_the_decision_close(self):
        """The decision saw T's close; filling there would trade on a known price."""
        feed = _feed(opens=[100.0, 111.0, 112.0, 113.0, 114.0, 115.0],
                     closes=[100.0, 110.0, 112.0, 113.0, 114.0, 115.0])
        config = BacktestConfig(holding_period_days=5, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, engine = _run(_buy_once("2026-01-05"), feed, config)

        # Entry is on 2026-01-06 (the session after the decision) at its open (111).
        trade_positions = engine.open_positions
        assert "AAPL" in trade_positions
        assert trade_positions["AAPL"].entry_session == "2026-01-06"
        assert trade_positions["AAPL"].entry_price == pytest.approx(111.0)
        # The decision close of 100 was never used as a fill price.
        assert trade_positions["AAPL"].entry_price != 100.0

    def test_the_order_is_created_on_the_decision_session(self):
        result, engine = _run(_buy_once("2026-01-05"), config=BacktestConfig(holding_period_days=5))
        created = [e for e in engine.ledger.events if e.type.value == "ORDER_CREATED"]
        assert created
        assert created[0].timestamp.date().isoformat() == "2026-01-05"

    def test_an_order_does_not_fill_when_the_symbol_has_no_bar_that_session(self):
        """A holiday for the symbol must expire the order, not price it."""
        feed = MarketDataFeed({
            # The calendar has a session on 01-06 (MSFT traded), but AAPL did not.
            "AAPL": pd.DataFrame(
                {"open": [100.0, 102.0], "high": [101.0, 103.0], "low": [99.0, 101.0],
                 "close": [100.0, 102.0], "volume": [1_000, 1_000]},
                index=pd.to_datetime(["2026-01-05", "2026-01-07"]),
            ),
            "MSFT": pd.DataFrame(
                {"open": [50.0], "high": [51.0], "low": [49.0], "close": [50.0],
                 "volume": [1_000]},
                index=pd.to_datetime(["2026-01-06"]),
            ),
        })
        config = BacktestConfig(holding_period_days=5)
        result, engine = _run(_buy_once("2026-01-05"), feed, config)
        # The order was scheduled for 01-06, when AAPL did not trade, so it expired
        # rather than filling at a neighbouring price.
        assert engine.open_positions == {}
        assert any("no bar" in reason for _s, _sym, reason in result.unfilled)

    def test_executing_on_the_decision_close_is_available_but_flagged_unsafe(self):
        """The unsafe mode exists to demonstrate the difference, not to be used."""
        config = BacktestConfig(holding_period_days=2, execute_on_next_open=False,
                                risk_limits=RiskLimits(max_position_size=0.1,
                                                       max_portfolio_exposure=None))
        result, _engine = _run(_buy_once("2026-01-05"), config=config)
        # The entry happened on the decision session itself, at its close.
        assert result.trades
        assert result.trades[0].entry_session == "2026-01-05"
        assert result.trades[0].entry_price == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExits:
    def test_a_position_exits_after_the_holding_period(self):
        config = BacktestConfig(holding_period_days=2, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05"), config=config)
        assert len(result.trades) == 1
        trade = result.trades[0]
        # entered 01-06, two sessions later is 01-08
        assert trade.entry_session == "2026-01-06"
        assert trade.exit_session == "2026-01-08"
        assert trade.reason == "holding_period"

    def test_a_stop_loss_exits_early_at_the_level(self):
        # Entry on 01-06 at open 100; a stop at 99 is touched on 01-07, whose open
        # is above the stop, so the stop is honoured at its own price.
        feed = _feed(opens=[100.0, 100.0, 100.0, 95.0, 95.0, 95.0],
                     closes=[100.0, 100.0, 95.0, 95.0, 95.0, 95.0],
                     highs=[101.0, 101.0, 101.0, 96.0, 96.0, 96.0],
                     lows=[99.0, 99.0, 90.0, 90.0, 90.0, 90.0])
        config = BacktestConfig(holding_period_days=5, use_stops=True, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05", stop_loss=99.0, entry_price=100.0), feed, config)
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.reason == "stop_loss"
        assert trade.exit_price == pytest.approx(99.0)
        assert trade.net_pnl < 0

    def test_a_stop_that_gaps_through_fills_at_the_open(self):
        """A stop cannot be honoured at its price if the session opened below it."""
        feed = _feed(opens=[100.0, 100.0, 80.0, 80.0, 80.0, 80.0],
                     closes=[100.0, 100.0, 80.0, 80.0, 80.0, 80.0],
                     highs=[101.0, 101.0, 81.0, 81.0, 81.0, 81.0],
                     lows=[99.0, 99.0, 79.0, 79.0, 79.0, 79.0])
        config = BacktestConfig(holding_period_days=5, use_stops=True, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05", stop_loss=95.0, entry_price=100.0), feed, config)
        assert result.trades[0].exit_price == pytest.approx(80.0)
        assert result.trades[0].reason == "stop_loss"

    def test_a_take_profit_exits_early_at_the_level(self):
        feed = _feed(opens=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                     closes=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                     highs=[101.0, 101.0, 120.0, 120.0, 120.0, 120.0],
                     lows=[99.0, 99.0, 99.0, 99.0, 99.0, 99.0])
        config = BacktestConfig(holding_period_days=5, use_stops=True, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05", take_profit=115.0, entry_price=100.0), feed, config)
        assert result.trades[0].reason == "take_profit"
        assert result.trades[0].exit_price == pytest.approx(115.0)
        assert result.trades[0].net_pnl > 0

    def test_stops_are_ignored_when_disabled(self):
        feed = _feed(opens=[100.0, 100.0, 100.0, 95.0, 95.0, 95.0],
                     closes=[100.0, 100.0, 95.0, 95.0, 95.0, 95.0],
                     highs=[101.0, 101.0, 101.0, 96.0, 96.0, 96.0],
                     lows=[99.0, 99.0, 90.0, 90.0, 90.0, 90.0])
        config = BacktestConfig(holding_period_days=4, use_stops=False, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05", stop_loss=99.0, entry_price=100.0), feed, config)
        assert result.trades[0].reason == "holding_period"

    def test_a_position_open_at_the_end_is_reported_not_invented(self):
        config = BacktestConfig(holding_period_days=10, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, engine = _run(_buy_once("2026-01-05"), config=config)
        assert result.trades == []
        assert "AAPL" in engine.open_positions

    def test_force_close_realises_the_remaining_positions(self):
        config = BacktestConfig(holding_period_days=10, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        result, engine = _run(_buy_once("2026-01-05"), config=config)
        engine.force_close_all("2026-01-12")
        assert len(result.trades) == 1
        assert result.trades[0].reason == "forced_close"


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOverlap:
    def test_two_decisions_on_consecutive_sessions_both_execute(self):
        """Overlap is modelled: the second order adds to the position."""
        def strategy(context):
            if context.session in ("2026-01-05", "2026-01-06"):
                return _signal(context.session, position_size=0.02)
            return None

        config = BacktestConfig(holding_period_days=5, risk_limits=RiskLimits(
            max_position_size=0.5, max_portfolio_exposure=None))
        _result, engine = _run(strategy, config=config)
        position = engine.open_positions["AAPL"]
        # 2% of ~100k = ~2000 -> 20 units each time, blended entry.
        assert position.quantity == pytest.approx(40.0, rel=0.05)

    def test_the_exposure_cap_stops_the_book_growing_without_bound(self):
        def always_buy(context):
            return _signal(context.session, position_size=0.9)

        config = BacktestConfig(holding_period_days=5, risk_limits=RiskLimits(
            max_position_size=0.2, max_portfolio_exposure=0.4))
        _result, engine = _run(always_buy, config=config)
        assert engine.ledger.total_exposure <= 0.4 + 1e-6

    def test_the_position_cap_bounds_a_single_name(self):
        def always_buy(context):
            return _signal(context.session, position_size=0.9)

        config = BacktestConfig(holding_period_days=5, risk_limits=RiskLimits(
            max_position_size=0.1, max_portfolio_exposure=None))
        _result, engine = _run(always_buy, config=config)
        position_value = engine.open_positions["AAPL"].quantity * engine.ledger.holdings[
            "AAPL"].average_entry_price
        equity = engine.ledger.equity
        assert position_value / equity <= 0.1 + 1e-3


# ---------------------------------------------------------------------------
# Risk enforcement inside the loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRiskInTheLoop:
    def test_a_sell_with_nothing_held_is_rejected_and_recorded(self):
        config = BacktestConfig(risk_limits=RiskLimits(max_portfolio_exposure=None))
        result, engine = _run(
            lambda c: _signal(c.session, action=SignalAction.SELL) if c.session == "2026-01-05" else None,
            config=config,
        )
        assert result.rejections
        assert "INSUFFICIENT_POSITION" in result.rejections[0][2]
        assert engine.ledger.position_quantity("AAPL") == 0.0

    def test_a_hold_produces_no_order_and_no_rejection_entry(self):
        result, engine = _run(
            lambda c: _signal(c.session, action=SignalAction.HOLD) if c.session == "2026-01-05" else None,
        )
        assert result.decisions == 1
        assert result.rejections == []
        assert engine.ledger.events == [] or all(
            e.type.value != "ORDER_FILLED" for e in engine.ledger.events
        )

    def test_the_book_is_never_short(self):
        """A strategy that tries to sell what it does not hold cannot create a short."""
        def always_sell(context):
            return _signal(context.session, action=SignalAction.SELL)

        _result, engine = _run(always_sell, config=BacktestConfig(risk_limits=RiskLimits(
            max_portfolio_exposure=None)))
        for symbol in engine.ledger.holdings:
            assert engine.ledger.position_quantity(symbol) >= 0.0


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAccounting:
    def test_the_equity_curve_has_one_point_per_session(self):
        result, _ = _run(lambda c: None)
        assert len(result.equity_curve) == len(DATES)
        assert [p.session for p in result.equity_curve] == DATES

    def test_a_flat_book_keeps_its_cash(self):
        result, _ = _run(lambda c: None)
        assert result.final_equity == pytest.approx(result.initial_cash)

    def test_costs_reduce_the_final_equity_on_a_round_trip(self):
        """Buy and sell at the same prices; only the costs should move the equity."""
        config = BacktestConfig(
            holding_period_days=1,
            costs=CostModel(commission_rate=0.001, half_spread_bps=5.0),
            risk_limits=RiskLimits(max_position_size=0.1, max_portfolio_exposure=None),
        )
        feed = _feed(opens=[100.0] * 6, closes=[100.0] * 6,
                     highs=[100.0] * 6, lows=[100.0] * 6)
        result, _ = _run(_buy_once("2026-01-05"), feed, config)
        assert result.final_equity < result.initial_cash

    def test_the_conservation_identity_holds_at_the_end_of_a_run(self):
        config = BacktestConfig(holding_period_days=2, costs=CostModel(commission_flat=1.0),
                                risk_limits=RiskLimits(max_position_size=0.1,
                                                       max_portfolio_exposure=None))
        _result, engine = _run(_buy_once("2026-01-05"), config=config)
        ledger = engine.ledger
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert (
            ledger.cash + basis + ledger.total_fees - ledger.realized_pnl
            == pytest.approx(ledger.initial_cash)
        )

    def test_equity_is_cash_plus_marked_positions_at_every_point(self):
        config = BacktestConfig(holding_period_days=3, risk_limits=RiskLimits(
            max_position_size=0.2, max_portfolio_exposure=None))
        result, _ = _run(_buy_once("2026-01-05"), config=config)
        for point in result.equity_curve:
            assert point.equity == pytest.approx(point.cash + point.market_value)

    def test_the_equity_series_is_indexed_by_session(self):
        result, _ = _run(lambda c: None)
        series = result.equity_series()
        assert list(series.index) == DATES


# ---------------------------------------------------------------------------
# The strategy contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyContract:
    def test_the_context_history_is_cut_at_the_session(self):
        seen = {}

        def strategy(context):
            seen["max_date"] = context.history("AAPL").index.max().strftime("%Y-%m-%d")
            seen["session"] = context.session
            return None

        _run(strategy, start="2026-01-05", end="2026-01-07")
        assert seen["max_date"] <= seen["session"]

    def test_the_context_exposes_the_session_close(self):
        seen = {}

        def strategy(context):
            seen["close"] = context.close("AAPL")
            return None

        _run(strategy, start="2026-01-05", end="2026-01-05")
        assert seen["close"] == 100.0

    def test_a_strategy_returning_none_is_not_a_decision(self):
        result, _ = _run(lambda c: None)
        assert result.decisions == 0

    def test_the_result_reports_its_own_configuration(self):
        config = BacktestConfig(initial_cash=50_000.0, holding_period_days=3)
        result, _ = _run(lambda c: None, config=config)
        assert result.initial_cash == 50_000.0
        assert result.config is config

    def test_an_empty_date_range_is_refused(self):
        engine = PortfolioBacktester(_feed(), lambda c: None)
        with pytest.raises(ValueError, match="no sessions"):
            engine.run("2027-01-01", "2027-02-01")


@pytest.mark.unit
class TestConfigValidation:
    def test_a_non_positive_initial_cash_is_refused(self):
        with pytest.raises(ValueError, match="initial_cash"):
            BacktestConfig(initial_cash=0.0)

    def test_a_non_positive_holding_period_is_refused(self):
        with pytest.raises(ValueError, match="holding_period_days"):
            BacktestConfig(holding_period_days=0)
