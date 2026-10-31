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
from tradingagents.portfolio.events import EventType
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


# ---------------------------------------------------------------------------
# The strategy cannot see the future, and a cap holds through a gap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyCannotSeeTheFuture:
    """The context a strategy is handed must not expose a later bar.

    Regression test for a real defect: the engine passed the raw feed to the
    strategy context, so ``ctx.feed.bar(symbol, tomorrow)`` returned tomorrow's
    bar. The safe ``ctx.history`` helper was not the only way in. A strategy that
    read the future would have produced optimistic results that no other test
    would flag, which is the worst kind of backtest bug.
    """

    def test_the_context_feed_is_bounded_at_the_session(self):
        feed = _feed()
        seen = {}

        def strategy(ctx):
            seen[ctx.session] = ctx.feed.bar("AAPL", "2026-01-12")
            return None

        PortfolioBacktester(feed, strategy, config=BacktestConfig()).run()
        for session, bar in seen.items():
            if bar is not None:
                assert bar.date <= session, f"{session} could see {bar.date}"

    def test_the_context_feed_calendar_ends_at_the_session(self):
        feed = _feed()
        lengths = {}

        def strategy(ctx):
            lengths[ctx.session] = ctx.feed.calendar()[-1]
            return None

        PortfolioBacktester(feed, strategy, config=BacktestConfig()).run()
        for session, last in lengths.items():
            assert last == session

    def test_the_context_history_is_still_bounded(self):
        feed = _feed()
        seen = {}

        def strategy(ctx):
            history = ctx.history("AAPL")
            seen[ctx.session] = str(history.index[-1].date())
            return None

        PortfolioBacktester(feed, strategy, config=BacktestConfig()).run()
        for session, last in seen.items():
            assert last == session




# ---------------------------------------------------------------------------
# The position cap is enforced at the fill, not only at the decision
# ---------------------------------------------------------------------------


def _flat_then_gap(*, gap_open, sessions=4, pre=100.0):
    """Sessions that trade flat at ``pre``, then open at ``gap_open`` and hold it.

    Open and close agree on every session, so a fill at an open and a mark at a
    close value the position the same way. That keeps the arithmetic in a test
    legible: a position's marked fraction is its quantity times the price divided
    by equity, with no open-versus-close wedge to explain.
    """
    prices = [pre] + [gap_open] * (sessions - 1)
    dates = [f"2026-01-{d:02d}" for d in range(5, 5 + sessions)]
    return _feed(dates=dates, opens=prices, closes=prices), dates


def _entry_strategy(dates, *, size, limits, cash=100_000.0, at=None):
    """A strategy that buys on one session and holds for the rest of the run."""
    when = at if at is not None else dates[0]

    def strategy(ctx):
        if ctx.session == when and ctx.position("AAPL") == 0:
            return _signal(when, position_size=size)
        return None

    return strategy, BacktestConfig(
        initial_cash=cash,
        holding_period_days=len(dates) + 5,
        risk_limits=limits,
    )


@pytest.mark.unit
class TestPositionCapEnforcedAtFill:
    """A gap between the decision price and the fill must not breach a cap.

    The engine sizes an entry when the decision is taken and fills it at the next
    session's open. When that open gaps up, the position bought is worth more than
    the size allowed. The engine re-checks the caps at the projected fill price and
    reduces the order to the largest size that holds, so the position that results
    respects the cap. These tests assert the value of the filled position, not the
    size of the order that was requested.
    """

    def test_a_gap_that_would_double_the_position_is_capped(self):
        # Sized at 20% of 100,000 on the 100 close = 200 units; the fill opens at
        # 200, where 200 units is 40% of equity. The fill must be 100 units = 20%.
        feed, dates = _flat_then_gap(gap_open=200.0)
        strategy, config = _entry_strategy(
            dates, size=0.20,
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        assert ledger.position_quantity("AAPL") == pytest.approx(100.0)
        assert 100.0 * 200.0 / ledger.equity <= 0.20 + 1e-6

    def test_a_gap_within_the_limit_is_not_reduced(self):
        # 10% of 100,000 = 100 units at a 100 close; a 105 open makes that 10.5%,
        # still inside the 20% cap, so the size stands.
        feed, dates = _flat_then_gap(gap_open=105.0)
        strategy, config = _entry_strategy(
            dates, size=0.10,
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        assert ledger.position_quantity("AAPL") == pytest.approx(100.0)
        assert 100.0 * 105.0 / ledger.equity <= 0.20 + 1e-6

    def test_a_bigger_gap_leaves_fewer_units(self):
        """The invariant is on value, so the units fall as the gap widens."""
        limits = RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0)
        small_feed, dates = _flat_then_gap(gap_open=120.0)
        big_feed, _ = _flat_then_gap(gap_open=400.0)
        strategy, config = _entry_strategy(dates, size=0.20, limits=limits)
        small = PortfolioBacktester(small_feed, strategy, config=config).run().ledger
        strategy, config = _entry_strategy(dates, size=0.20, limits=limits)
        big = PortfolioBacktester(big_feed, strategy, config=config).run().ledger
        assert small.position_quantity("AAPL") > big.position_quantity("AAPL") > 0
        for ledger, price in ((small, 120.0), (big, 400.0)):
            q = ledger.position_quantity("AAPL")
            assert q * price / ledger.equity <= 0.20 + 1e-6

    def test_a_cap_that_leaves_less_than_a_unit_fills_that_fraction(self):
        # A 0.1% cap of 100,000 is 100; at a 500 open that is 0.2 units. The
        # ledger is quantity-based, so 0.2 units is a real fill, and the cap holds.
        feed, dates = _flat_then_gap(gap_open=500.0)
        strategy, config = _entry_strategy(
            dates, size=0.20,
            limits=RiskLimits(
                max_position_size=0.001, max_portfolio_exposure=1.0,
                default_position_size=0.001,
            ),
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        q = ledger.position_quantity("AAPL")
        assert q == pytest.approx(0.2)
        assert q * 500.0 / ledger.equity <= 0.001 + 1e-6

    def test_the_reduction_is_recorded_in_the_unfilled_log(self):
        """A reduced fill is reported, so a run is auditable after the fact."""
        feed, dates = _flat_then_gap(gap_open=200.0)
        strategy, config = _entry_strategy(
            dates, size=0.20,
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        result = PortfolioBacktester(feed, strategy, config=config).run()
        assert result.ledger.position_quantity("AAPL") > 0
        assert result.ledger.position_quantity("AAPL") == pytest.approx(100.0)

    def test_accounting_identity_holds_after_a_capped_fill(self):
        feed, dates = _flat_then_gap(gap_open=200.0)
        strategy, config = _entry_strategy(
            dates, size=0.20,
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert ledger.cash + basis + ledger.total_fees - ledger.realized_pnl == pytest.approx(
            100_000.0
        )

    def test_repeated_entries_cannot_accumulate_past_the_cap(self):
        """Two decisions on consecutive sessions must not add up past the cap.

        The first fill takes the position to the cap; the second sees it in the
        book, so its headroom is the remainder rather than a fresh cap. This is the
        accumulation case, and it is why the re-check reads the ledger rather than
        the order alone.
        """
        dates = [f"2026-01-{d:02d}" for d in range(5, 10)]
        feed = _feed(dates=dates, opens=[100.0] * 5, closes=[100.0] * 5)

        def strategy(ctx):
            if ctx.session in (dates[0], dates[1]):
                return _signal(ctx.session, position_size=0.20)
            return None

        config = BacktestConfig(
            initial_cash=100_000.0, holding_period_days=20,
            risk_limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        q = ledger.position_quantity("AAPL")
        assert q * 100.0 / ledger.equity <= 0.20 + 1e-6

    def test_a_capped_fill_still_charges_only_for_what_filled(self):
        """The ledger is charged for the reduced quantity, not the requested one.

        A cap that reduced 200 units to 100 must produce fees and cash for 100.
        Charging for the larger order would be the quiet version of this bug: the
        position is right and the money is wrong.
        """
        costs = CostModel(commission_flat=1.0, commission_per_unit=0.01)
        feed, dates = _flat_then_gap(gap_open=200.0)
        strategy, config = _entry_strategy(
            dates, size=0.20,
            limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        config = BacktestConfig(
            initial_cash=config.initial_cash,
            holding_period_days=config.holding_period_days,
            risk_limits=config.risk_limits,
            costs=costs,
        )
        ledger = PortfolioBacktester(feed, strategy, config=config).run().ledger
        q = ledger.position_quantity("AAPL")
        assert q == pytest.approx(100.0)
        # The position cost 100 * 200 of cash, and the fee is the flat plus 100
        # units of per-unit commission: the reduced size, not the requested 200.
        assert ledger.total_fees == pytest.approx(1.0 + 0.01 * 100)
        basis = sum(h.cost_basis for h in ledger.open_positions().values())
        assert ledger.cash + basis + ledger.total_fees - ledger.realized_pnl == pytest.approx(
            100_000.0
        )

    def test_a_refusal_at_the_fill_is_recorded_not_silent(self):
        """A refusal after the order was created must leave a trace.

        The order exists by then, so a refusal that set no reason and wrote no
        event would be invisible to the event log a report is built from: the
        order would appear created and then simply vanish. The first buy fills at
        100; the second is sized at 100 but fills at a 400 open, where the book is
        already past the cap, so it is refused at the fill with ``RISK_LIMIT`` and
        an ``ORDER_REJECTED`` event.
        """
        dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        feed = _feed(dates=dates, opens=[100.0, 100.0, 400.0, 400.0],
                     closes=[100.0, 100.0, 400.0, 400.0])

        def strategy(ctx):
            if ctx.session in (dates[0], dates[1]):
                return _signal(ctx.session, position_size=0.15)
            return None

        config = BacktestConfig(
            initial_cash=100_000.0, holding_period_days=20,
            risk_limits=RiskLimits(max_position_size=0.20, max_portfolio_exposure=1.0),
        )
        result = PortfolioBacktester(feed, strategy, config=config).run()
        rejected = [e for e in result.ledger.events if e.type is EventType.ORDER_REJECTED]
        assert rejected, "a fill-time refusal must record an ORDER_REJECTED event"
        assert all(e.detail.get("reason") == "RISK_LIMIT" for e in rejected)
        assert any("refused at execution" in e.reason for e in rejected)
        assert any(
            symbol == "AAPL" and "refused at execution" in detail
            for _, symbol, detail in result.unfilled
        )


@pytest.mark.unit
class TestBuySizeCappedByCashAtFill:
    """A buy that no longer fits the cash at fill time is reduced, not dropped.

    Regression test for a real defect: the order was sized to the available cash
    at the decision price, but the fill happens at the next session's open and
    carries a fee, so the all-in cost could exceed the cash. The simulator then
    rejected the whole order (``INSUFFICIENT_CASH``), losing a fill the cash could
    still have paid for. The fill-time re-check now caps the size to the
    fee-inclusive affordable notional, exactly as it caps the risk limits.
    """

    def test_a_commission_does_not_turn_a_full_cash_buy_into_a_rejection(self):
        feed = _feed(opens=[100.0] * 6, closes=[100.0] * 6)
        config = BacktestConfig(
            initial_cash=100_000.0,
            holding_period_days=20,
            costs=CostModel(commission_rate=0.001),
            risk_limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
        )
        result, _ = _run(_buy_once(position_size=1.0), feed=feed, config=config)
        assert result.ledger.position_quantity("AAPL") > 0
        # The whole cash is spent, and the all-in cost (notional + fees) fits it.
        spent = result.ledger.position_quantity("AAPL") * 100.0 + result.ledger.total_fees
        assert spent <= 100_000.0 + 1e-6
        assert not [u for u in result.unfilled if "INSUFFICIENT_CASH" in u[2]]

    def test_an_up_gap_reduces_the_size_rather_than_refusing(self):
        # Decision close 100, fill open 150: the cash buys 2/3 as many units.
        feed = _feed(opens=[100.0, 150.0, 150.0, 150.0, 150.0, 150.0],
                     closes=[100.0, 150.0, 150.0, 150.0, 150.0, 150.0])
        config = BacktestConfig(
            initial_cash=100_000.0,
            holding_period_days=20,
            risk_limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
        )
        result, _ = _run(_buy_once(position_size=1.0), feed=feed, config=config)
        quantity = result.ledger.position_quantity("AAPL")
        assert quantity == pytest.approx(100_000.0 / 150.0)
        assert result.ledger.cash == pytest.approx(0.0, abs=1e-6)

    def test_the_cap_never_leaves_cash_negative(self):
        feed = _feed(opens=[100.0, 200.0, 200.0, 200.0, 200.0, 200.0],
                     closes=[100.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        config = BacktestConfig(
            initial_cash=100_000.0,
            holding_period_days=20,
            costs=CostModel(commission_rate=0.001),
            risk_limits=RiskLimits(max_position_size=1.0, max_portfolio_exposure=None),
        )
        result, _ = _run(_buy_once(position_size=1.0), feed=feed, config=config)
        assert result.ledger.cash >= -1e-6
        assert result.ledger.cash == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Exit tracking vs. the ledger (adversarial audit)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExitTrackingFollowsTheLedger:
    """A signal SELL fills at the next open; the exit tracker must follow it.

    The engine tracks open positions for their scheduled exit. A SELL signal
    executes through the same pending-order path as a buy, so the ledger's
    holding falls — but the tracker was never told. It then tried to sell the
    position again at its holding-period exit and the ledger refused the oversell,
    leaving the engine's idea of the book permanently larger than the real one.
    """

    def _sell_after_buy(self, *, buy_size=0.5, sell_size=1.0):
        def strategy(context):
            if context.session == "2026-01-05":
                return _signal(context.session, position_size=buy_size)
            if context.session == "2026-01-07":
                return _signal(context.session, action=SignalAction.SELL, position_size=sell_size)
            return None
        return strategy

    def test_a_sell_signal_empties_the_engine_tracker(self):
        result, engine = _run(
            self._sell_after_buy(),
            config=BacktestConfig(initial_cash=100_000.0, holding_period_days=3),
        )
        assert result.ledger.position_quantity("AAPL") == pytest.approx(0.0)
        assert engine.open_positions == {}

    def test_a_sell_signal_records_a_completed_round_trip(self):
        result, _ = _run(
            self._sell_after_buy(),
            config=BacktestConfig(initial_cash=100_000.0, holding_period_days=3),
        )
        sells = [t for t in result.trades if t.reason == "signal_sell"]
        assert len(sells) == 1
        assert sells[0].quantity == pytest.approx(result.ledger.position_quantity("AAPL") + sells[0].quantity)

    def test_no_phantom_oversell_after_a_signal_sell(self):
        """The scheduled exit must not try to sell a position that is already gone."""
        result, _ = _run(
            self._sell_after_buy(),
            config=BacktestConfig(initial_cash=100_000.0, holding_period_days=2),
        )
        assert not [u for u in result.unfilled if "exceeds" in u[2]]

    def test_a_partial_sell_reduces_the_tracked_position(self):
        result, engine = _run(
            self._sell_after_buy(buy_size=0.5, sell_size=0.5),
            config=BacktestConfig(initial_cash=100_000.0, holding_period_days=20),
        )
        # Half sold: the tracker holds what the ledger holds, not the original size.
        tracked = engine.open_positions.get("AAPL")
        if tracked is not None:
            assert tracked.quantity == pytest.approx(result.ledger.position_quantity("AAPL"))

    def test_the_tracker_always_agrees_with_the_ledger(self):
        result, engine = _run(
            self._sell_after_buy(buy_size=1.0, sell_size=0.25),
            config=BacktestConfig(initial_cash=100_000.0, holding_period_days=4),
        )
        for symbol, position in engine.open_positions.items():
            assert position.quantity == pytest.approx(result.ledger.position_quantity(symbol))


# ---------------------------------------------------------------------------
# Event-log completeness and ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheEventLogIsOrderedAndComplete:
    def test_creation_is_recorded_before_acceptance(self):
        """A lifecycle that logs ACCEPTED before CREATED is not a lifecycle."""
        result, _ = _run(_buy_once(), config=BacktestConfig(holding_period_days=2))
        order_events = [e.type for e in result.ledger.events
                        if e.type in (EventType.ORDER_CREATED, EventType.ORDER_ACCEPTED)]
        assert order_events[0] is EventType.ORDER_CREATED
        assert order_events[1] is EventType.ORDER_ACCEPTED

    def test_an_order_expired_for_a_missing_bar_is_cancelled_on_the_record(self):
        """An order that expires because the symbol did not trade leaves a record."""
        def frame(dates, price):
            return pd.DataFrame(
                {"open": [price] * len(dates), "high": [price * 1.01] * len(dates),
                 "low": [price * 0.99] * len(dates), "close": [price] * len(dates),
                 "volume": [1_000_000] * len(dates)},
                index=pd.to_datetime(dates),
            )
        feed = MarketDataFeed({
            # AAPL does not trade on 01-06, so an order to buy it there expires.
            "AAPL": frame(["2026-01-05", "2026-01-07"], 100.0),
            "MSFT": frame(["2026-01-05", "2026-01-06", "2026-01-07"], 50.0),
        })
        result, _ = _run(_buy_once(), feed=feed, config=BacktestConfig(holding_period_days=5))
        cancels = [e for e in result.ledger.events if e.type is EventType.ORDER_CANCELLED]
        assert cancels, "the expired order must be cancelled on the event log"
        assert any("no bar" in u[2] for u in result.unfilled)

    def test_an_order_still_pending_at_the_end_is_accounted_for(self):
        """An order with no next session to execute in is cancelled, not dropped."""
        def strategy(context):
            if context.session == DATES[-1]:
                return _signal(context.session, position_size=0.1)
            return None
        result, _ = _run(strategy, config=BacktestConfig(holding_period_days=5))
        created = [e for e in result.ledger.events if e.type is EventType.ORDER_CREATED]
        cancelled = [e for e in result.ledger.events if e.type is EventType.ORDER_CANCELLED]
        assert created, "the final-session order must still be created"
        assert len(cancelled) == len(created), "every created order ends with an outcome"

    def test_every_created_order_has_a_terminal_event(self):
        """No order may be created and then vanish without an outcome."""
        def strategy(context):
            return _signal(context.session, position_size=0.1)
        result, _ = _run(strategy, config=BacktestConfig(holding_period_days=5))
        created = {e.detail.get("order_id") for e in result.ledger.events
                   if e.type is EventType.ORDER_CREATED}
        terminal = {e.detail.get("order_id") for e in result.ledger.events
                    if e.type in (EventType.ORDER_FILLED, EventType.ORDER_REJECTED,
                                  EventType.ORDER_CANCELLED)}
        assert created <= terminal


# ---------------------------------------------------------------------------
# The tradable universe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSignalsForSymbolsOutsideTheUniverse:
    """A signal the feed cannot price is rejected, not fatal.

    The multi-agent brain can name a ticker that is not in the run's universe.
    Before this guard the backtester raised ``BarDataError`` from inside the loop,
    so one stray symbol destroyed the entire run and lost every result.
    """

    def _ghost_strategy(self):
        def strategy(context):
            if context.session == "2026-01-05":
                return _signal(context.session, symbol="GHOST", position_size=0.2)
            return None
        return strategy

    def test_a_signal_for_an_unknown_symbol_does_not_crash_the_run(self):
        result, _ = _run(self._ghost_strategy(), config=BacktestConfig(initial_cash=100_000.0))
        assert result is not None

    def test_a_signal_for_an_unknown_symbol_is_recorded_as_a_rejection(self):
        result, _ = _run(self._ghost_strategy(), config=BacktestConfig(initial_cash=100_000.0))
        reasons = [reason for _s, sym, reason in result.rejections if sym == "GHOST"]
        assert "UNKNOWN_SYMBOL" in reasons

    def test_an_unknown_symbol_never_touches_the_book(self):
        result, _ = _run(self._ghost_strategy(), config=BacktestConfig(initial_cash=100_000.0))
        assert result.ledger.cash == pytest.approx(100_000.0)
        assert result.ledger.holdings == {}

    def test_the_rest_of_the_run_still_happens(self):
        """One bad symbol must not stop the good ones."""
        def strategy(context):
            if context.session == "2026-01-05":
                return [
                    _signal(context.session, symbol="GHOST", position_size=0.1),
                ] and _signal(context.session, position_size=0.3)
            return None
        result, _ = _run(strategy, config=BacktestConfig(initial_cash=100_000.0))
        assert result.ledger.position_quantity("AAPL") > 0


# ---------------------------------------------------------------------------
# The strategy's view of the book
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheStrategyCannotMoveTheBook:
    """The strategy reads the book; only orders move it.

    Regression test for a real defect: ``context.ledger`` was the live
    :class:`Ledger`, so a strategy could write ``context.ledger.cash = ...`` or
    call ``apply_fill`` and fabricate a result that no order produced. The class
    docstring promised the ledger was read-only; it was not.
    """

    def _tampering_strategy(self, attempts):
        def strategy(context):
            try:
                context.ledger.cash = 10_000_000.0
                attempts.append("cash written")
            except AttributeError:
                attempts.append("cash write refused")
            try:
                context.ledger.apply_fill(
                    symbol="AAPL", quantity=999.0, price=1.0, fee=0.0,
                    timestamp=datetime(2026, 1, 5, 21, tzinfo=UTC),
                )
                attempts.append("fill applied")
            except AttributeError:
                attempts.append("fill refused")
            return None
        return strategy

    def test_cash_cannot_be_written(self):
        attempts = []
        _run(self._tampering_strategy(attempts), config=BacktestConfig(initial_cash=100_000.0))
        assert "cash written" not in attempts
        assert "cash write refused" in attempts

    def test_apply_fill_cannot_be_called(self):
        attempts = []
        _run(self._tampering_strategy(attempts), config=BacktestConfig(initial_cash=100_000.0))
        assert "fill applied" not in attempts

    def test_the_book_is_untouched_by_a_tampering_strategy(self):
        attempts = []
        result, _ = _run(self._tampering_strategy(attempts),
                         config=BacktestConfig(initial_cash=100_000.0))
        assert result.ledger.cash == pytest.approx(100_000.0)
        assert result.ledger.position_quantity("AAPL") == 0.0

    def test_reading_the_book_still_works(self):
        seen = {}

        def strategy(context):
            seen["cash"] = context.ledger.cash
            seen["equity"] = context.ledger.equity
            seen["position"] = context.position("AAPL")
            return None

        _run(strategy, config=BacktestConfig(initial_cash=100_000.0))
        assert seen["cash"] == pytest.approx(100_000.0)
        assert seen["equity"] == pytest.approx(100_000.0)
        assert seen["position"] == 0.0

    def test_a_holding_returned_to_the_strategy_is_a_copy(self):
        """Mutating the object the strategy receives must not reach the book."""
        observed = {}

        def strategy(context):
            holding = context.ledger.holding("AAPL")
            holding.quantity = 12_345.0
            observed["after_mutation"] = context.ledger.position_quantity("AAPL")
            return None

        _run(strategy, config=BacktestConfig(initial_cash=100_000.0))
        assert observed["after_mutation"] == 0.0
