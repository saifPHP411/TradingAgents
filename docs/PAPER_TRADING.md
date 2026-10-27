# Paper trading

Paper trading runs a strategy against live quotes, using the same ledger, risk
engine and execution simulator a backtest uses. It is the step between "this
backtests well" and "I would trust this with money", and it is where the
accounting has to be the same in both modes or the comparison means nothing.

`execution/paper.py` holds it.

## What is different from a backtest

Two things, and both are in the caller:

| | Backtest | Paper |
|---|---|---|
| Clock | The historical session calendar | The wall clock |
| Quotes | A `MarketDataFeed` of bars | A caller-supplied quote source |

Everything else — the risk verdict, the order, the fill, the ledger — is the
same code. That is deliberate: if paper results drifted from backtest results,
the paper run would be telling a different story about the same strategy.

## What is not different, and cannot be

There is no live path. `PaperSession` resolves its mode through the same
`resolve_mode` guard a backtest uses, the simulator's constructor calls
`assert_simulation_only`, and a quote source is a plain callable that returns a
price. A live implementation would have to be a new class written deliberately,
not a setting. The tests assert that `live`, `real`, `production`, `broker` and
`exchange` all raise, and that `ExecutionMode` has no live member to select.

## Running a session

A quote source is any callable from symbol to `Quote`. The simplest is a dict:

```python
from datetime import UTC, datetime

from tradingagents.execution import Quote, dict_quote_source, new_session
from tradingagents.risk import RiskLimits
from tradingagents.signals import SignalAction, TradingSignal

quotes = dict_quote_source({"AAPL": 195.40}, timestamp=datetime.now(UTC))

session = new_session(
    initial_cash=100_000.0,
    quotes=quotes,
    limits=RiskLimits(max_position_size=0.10, max_portfolio_exposure=0.50),
)

signal = TradingSignal(
    symbol="AAPL", action=SignalAction.BUY, confidence=0.7,
    timestamp=datetime.now(UTC), as_of_date=datetime.now(UTC).date().isoformat(),
    position_size=0.05, stop_loss=180.0, take_profit=220.0,
    reasoning="Example only.",
)

step = session.on_signal(signal)
print(step.order_status, step.quantity, step.fill_price, step.fees)
print(session.ledger.equity, session.ledger.cash)
```

For live prices, supply your own callable. It must not reach a venue:

```python
def my_quotes(symbol: str) -> Quote:
    price = fetch_a_price_from_a_read_only_source(symbol)
    return Quote(symbol=symbol, price=price, timestamp=datetime.now(UTC))
```

## Marking the book

A session only knows the prices it is given, so mark before deciding. Risk is
sized against a current book, not the last trade's prices:

```python
session.mark([Quote(symbol="AAPL", price=196.10, timestamp=datetime.now(UTC))])
```

## Resting LIMIT orders

A signal can ask for a `LIMIT` order that rests until it fills. Build it with
`order_intent=OrderIntent.LIMIT`, an `entry_price` on the right side, and
`time_in_force=SignalTimeInForce.GTC`:

```python
from tradingagents.signals import OrderIntent, SignalTimeInForce

signal = TradingSignal(
    symbol="AAPL", action=SignalAction.BUY, confidence=0.7,
    timestamp=datetime.now(UTC), as_of_date=datetime.now(UTC).date().isoformat(),
    position_size=0.05, entry_price=196.10, limit_price=190.00,
    order_intent=OrderIntent.LIMIT, time_in_force=SignalTimeInForce.GTC,
)

session.on_signal(signal)
```

If the current quote has not reached the limit, the order is not cancelled: it is
held in `session.resting`. Advance it when fresh quotes arrive:

```python
session.mark([Quote(symbol="AAPL", price=189.50, timestamp=datetime.now(UTC))])
steps = session.work_resting()
print(steps[0].order_status, session.ledger.position_quantity("AAPL"))
```

An order whose symbol has no quote is left resting rather than cancelled — a
missing quote is not a reason to kill an order. `session.resting` and the JSON
snapshot's `resting_orders` list show what is still live. A caller that never
calls `work_resting` leaves the order resting, which is honest: an order nobody
advanced is an order that did not fill.

## Recording

Every decision is recorded as a `SessionStep`, whether it produced an order or
not. A HOLD and a refusal both appear, with their reason. The session writes a
snapshot as JSON:

```python
paths = session.write_json("runs/paper_2026-01-05.json")
```

The snapshot holds the mode, equity, cash, peak, realised and unrealised P&L,
total fees, every resting order, every open position and every step.

## The daily baseline

The session tracks the day's opening equity and resets it when the date changes,
so the daily-loss limit has a baseline to measure against. A signal on the same
day does not reset it.

## What a paper session is not

It is not a broker connection, and it is not a substitute for one. It fills
against the price you hand it, so it cannot tell you about slippage on a thin
book, a rejected order at a venue, or a partial fill you did not model. Those are
the things paper trading is meant to reveal, and this simulation can only reveal
the ones you put into the quote.
