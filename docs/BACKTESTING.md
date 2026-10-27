# Backtesting

The backtester is a portfolio and event loop, not a signal-to-return shortcut.
For each session, in order, it does the same thing a real system would:

```
market event  →  AI decision  →  risk validation  →  order
      →  execution  →  portfolio update  →  P&L  →  next event
```

`backtesting/engine.py` holds the loop. `backtesting/data.py` holds the feed.

## The loop, step by step

For each session `S`:

1. **Advance the calendar.** `MarketDataFeed.sessions()` yields the trading days
   present in the data, in order. Weekends and holidays are absent because they
   have no bars, so no session is invented.
2. **Fill yesterday's orders at today's open.** An order scheduled at session
   `S-1` fills against `S`'s open. This is the look-ahead guard.
3. **Enforce stops and targets.** For each open position, if `S`'s range crosses
   the stop or the target, the exit is taken at that level. A gap through the
   level fills at the open instead, which is the honest price, not the level.
4. **Expire positions past their holding period.** A position held for
   `holding_period_days` sessions exits at the next open.
5. **Ask the strategy for a decision.** The strategy is handed a
   `StrategyContext` whose history frames are already cut at `S`. It returns a
   `TradingSignal` or `None`.
6. **Run the signal through the risk engine.** The verdict is recorded whether
   it is an approval or a refusal, with the reason.
7. **Schedule the approved order** for session `S+1`'s open.
8. **Mark the book.** Every holding is marked at `S`'s close and an equity point
   is recorded.

## Timing, and why it is what it is

| Moment | What happens |
|---|---|
| Decision timestamp | The signal's own timestamp, at session `S`'s close. |
| Order timestamp | When the order is created: session `S`. |
| Execution timestamp | Session `S+1`'s open. |
| Price source | The open of `S+1`, from the feed. |
| Candle availability | Only bars up to and including `S` are visible to the decision. |

`execute_on_next_open` defaults to `True`. Setting it to `False` fills at the
decision bar's close — the price the decision already saw — and exists only so a
test can demonstrate the difference. It is documented as unsafe.

A strategy cannot see the future even if it tries. `StrategyContext.history()`
returns frames sliced at `S`, and the feed's `history()` refuses to return a bar
after the requested date. There is no accessor that returns the full frame.

## Missing and partial data

- A symbol with no bar on session `S` keeps its last known close when marked.
  Marking it to zero would invent a loss; marking it to a later price would be
  look-ahead.
- A pending order whose symbol has no bar at `S+1` is carried forward, not
  filled at a price nobody saw.
- The feed's `history()` raises for an unreadable date rather than returning
  everything, so a caller cannot accidentally widen the window.
- A signal for a symbol with no price at all is refused by the risk engine with
  a named reason, and the refusal is recorded.

## Corporate actions

Prices from the vendor are split-adjusted. That means a historical price is the
split-adjusted value rather than the price on the tape that day. A dividend is
therefore not credited as cash, and the equity curve reflects price return only.
This is a real limitation and is stated in `docs/TRADING_SYSTEM_AUDIT.md` rather
than hidden.

## Benchmarks

A benchmark is compared on the same sessions as the strategy, so a run and its
benchmark share a start and end. It is supplied to the metrics, not fetched.

## Walk-forward evaluation

`backtesting/walkforward.py` builds folds of train, validation and test windows.
A `Fold` validates its ordering at construction, so an out-of-sample window that
overlaps its training window cannot be built. The intent is that parameters are
chosen on train and validation, and the test window is read once.

## Reproducing a run

`BacktestConfig` holds everything that shapes a run: initial cash, holding
period, risk limits, costs, volume participation, stop enforcement and the fill
timing. Two runs with the same config and the same feed produce the same result.
The LLM calls upstream of the signal are not deterministic, which is why the
signal carries its own reasoning and id.

## Running a backtest

```python
import pandas as pd
from datetime import UTC, datetime

from tradingagents.backtesting import BacktestConfig, MarketDataFeed, PortfolioBacktester
from tradingagents.execution import CostModel
from tradingagents.risk import RiskLimits
from tradingagents.signals import SignalAction, TradingSignal

frame = pd.DataFrame(
    {"open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]},
    index=pd.to_datetime([...]),
)
feed = MarketDataFeed({"AAPL": frame})


def strategy(context):
    if context.position("AAPL") > 0:
        return None
    return TradingSignal(
        symbol="AAPL", action=SignalAction.BUY, confidence=0.7,
        timestamp=datetime.now(UTC), as_of_date=context.session,
        position_size=0.5, stop_loss=90.0, take_profit=120.0,
        reasoning="example",
    )


config = BacktestConfig(
    initial_cash=100_000.0,
    holding_period_days=5,
    costs=CostModel(commission_rate=0.0005, slippage_bps=2.0),
    risk_limits=RiskLimits(max_position_size=0.25, max_portfolio_exposure=0.9),
)
result = PortfolioBacktester(feed, strategy, config=config).run()
print(result.final_equity)
```

Then report it:

```python
from tradingagents.trading_report import build_report, write_report

paths = write_report(build_report(result), "reports", stem="aapl_run")
print(paths["html"], paths["json"])
```
