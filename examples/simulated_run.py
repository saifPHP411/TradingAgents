"""A complete simulated run, from synthetic bars to a report. No network, no keys.

Run it directly:

    python examples/simulated_run.py

It builds a deterministic price series, runs a trivial strategy through the full
event loop (signal, risk verdict, order, fill, ledger), and writes an HTML and a
JSON report next to the script. Everything is offline and reproducible, so it
works from an Android shell or a Colab cell with no setup beyond the package.

The point is to show the whole chain in one file, and to give a template: replace
the synthetic frame with real bars and the trivial strategy with a TradingSignal
produced by the agent graph, and nothing else changes.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingagents.backtesting import (
    BacktestConfig,
    MarketDataFeed,
    PortfolioBacktester,
)
from tradingagents.execution import CostModel
from tradingagents.risk import RiskLimits
from tradingagents.signals import SignalAction, TradingSignal
from tradingagents.trading_report import build_report, write_report

SYMBOL = "DEMO"
SESSIONS = 60


def synthetic_bars(symbol: str = SYMBOL, sessions: int = SESSIONS) -> pd.DataFrame:
    """A deterministic price series: a slow rise with a sine wobble.

    Synthetic so the example needs no data vendor and gives the same answer every
    time. A real run would read bars from the feed's vendor instead.
    """
    dates = pd.bdate_range("2026-01-05", periods=sessions)
    closes = [100.0 + i * 0.35 + 3.0 * math.sin(i / 6.0) for i in range(sessions)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.008 for c in closes],
            "low": [c * 0.992 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * sessions,
        },
        index=dates,
    )


def make_strategy(first_session: str, symbol: str = SYMBOL):
    """Buy once, hold to the end. Deliberately trivial: the pipeline is the subject."""

    def strategy(context):
        if context.session != first_session or context.position(symbol) > 0:
            return None
        return TradingSignal(
            symbol=symbol,
            action=SignalAction.BUY,
            confidence=0.7,
            timestamp=datetime.now(UTC),
            as_of_date=context.session,
            position_size=0.5,
            stop_loss=90.0,
            take_profit=150.0,
            reasoning="Example signal: a single entry, held for the run.",
        )

    return strategy


def main() -> None:
    frame = synthetic_bars()
    feed = MarketDataFeed({SYMBOL: frame})
    first_session = frame.index[0].strftime("%Y-%m-%d")

    config = BacktestConfig(
        initial_cash=100_000.0,
        holding_period_days=SESSIONS / 2,
        costs=CostModel(commission_rate=0.0005, half_spread_bps=1.0, slippage_bps=2.0),
        risk_limits=RiskLimits(
            max_position_size=0.5,
            max_portfolio_exposure=0.8,
            max_open_positions=3,
        ),
    )

    engine = PortfolioBacktester(feed, make_strategy(first_session), config=config)
    result = engine.run()

    out = Path(__file__).resolve().parent.parent / "reports" / "example"
    paths = write_report(
        build_report(result, title=f"{SYMBOL} simulated run"),
        out,
        stem="simulated_run",
    )

    print(f"sessions:     {len(result.sessions)}")
    print(f"decisions:    {result.decisions}")
    print(f"trades:       {len(result.trades)}")
    print(f"final equity: {result.final_equity:,.2f}")
    if result.ledger is not None:
        print(f"fees paid:    {result.ledger.total_fees:,.2f}")
    print(f"report:       {paths['html']}")
    print(f"data:         {paths['json']}")


if __name__ == "__main__":
    main()
