from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.backtesting import BacktestConfig, MarketDataFeed, PortfolioBacktester
from tradingagents.execution import CostModel
from tradingagents.risk import RiskLimits
from tradingagents.signals.contract import OrderIntent, SignalAction, TradingSignal


@pytest.mark.unit
def test_limit_entry_and_exit_trade_fees_match_maker_fees_charged_to_ledger():
    dates = pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1_000_000] * 4,
        },
        index=dates,
    )
    feed = MarketDataFeed({"AAPL": frame})

    def strategy(context):
        if context.session == "2026-01-05":
            return TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                confidence=0.8,
                timestamp=datetime(2026, 1, 5, 21, tzinfo=UTC),
                as_of_date="2026-01-05",
                entry_price=100.0,
                position_size=0.10,
                order_intent=OrderIntent.LIMIT,
                limit_price=100.0,
            )
        if context.session == "2026-01-06" and context.position("AAPL") > 0:
            return TradingSignal(
                symbol="AAPL",
                action=SignalAction.SELL,
                confidence=0.8,
                timestamp=datetime(2026, 1, 6, 21, tzinfo=UTC),
                as_of_date="2026-01-06",
                entry_price=100.0,
                position_size=1.0,
                order_intent=OrderIntent.LIMIT,
                limit_price=100.0,
            )
        return None

    config = BacktestConfig(
        initial_cash=100_000.0,
        holding_period_days=5,
        costs=CostModel(maker_fee_bps=1.0, taker_fee_bps=10.0),
        risk_limits=RiskLimits(
            max_position_size=0.20,
            max_portfolio_exposure=None,
        ),
    )
    engine = PortfolioBacktester(feed, strategy, config=config)
    result = engine.run()

    assert len(result.trades) == 1
    trade = result.trades[0]

    # The position is $10,000 on each side. Two maker fills at 1 bp cost $2.
    assert engine.ledger.total_fees == pytest.approx(2.0)
    assert trade.fees == pytest.approx(engine.ledger.total_fees)
    assert trade.net_pnl == pytest.approx(engine.ledger.net_realized_pnl)
    assert result.final_equity == pytest.approx(99_998.0)
