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
missing quote is not a reason to kill an order. A price source that *raises* is
handled the same way for that order: it is kept live, the book is not touched,
and the other resting orders are still worked, so one broken symbol cannot stall
the queue. The exception is re-raised after the loop rather than swallowed, so a
caller can retry; nothing is filled against a price that was never read.
`session.resting` and the JSON snapshot's `resting_orders` list show what is
still live. A caller that never calls `work_resting` leaves the order resting,
which is honest: an order nobody advanced is an order that did not fill.

A GTC limit can also fill only *part* of its size when participation is
configured and the quote carries a volume; its remainder stays in
`session.resting` so the next `work_resting` continues it. When participation is
configured, the session's volume allowance for a symbol is **shared** by every
order worked against that symbol in one `work_resting` call and drawn down in
queue order, exactly as the backtester shares one session's liquidity — so two
orders cannot each fill the whole bar. A resting buy is re-checked against the
risk caps at the price it will fill at, exactly as a fresh entry is, so a gap to
the limit cannot leave a position past its cap. An order that reaches its
`expires_session` is cancelled with `EXPIRED` by `work_resting`, and a caller can
cancel a live order explicitly with `session.cancel_resting(order)`.

The session clock only moves forward: a quote stamped on a date earlier than the
session has already seen is treated as stale for the day roll, so it cannot move
the daily-loss baseline backwards.

## Persistence, restart and recovery

There are two distinct JSON files, and they answer different questions:

| | `write_json` / `snapshot()` | `save_state` / `state()` |
|---|---|---|
| Purpose | Reporting | Recovery |
| Holds | Headline figures, positions, steps, live orders | The whole ledger **including its event log and sequence counter**, risk limits, cost schedule, simulator settings, live orders, processed-signal set, day baseline |
| Versioned | Yes (`schema_version`) | Yes (`schema_version`) |
| Can it resume a run? | **No** | Yes |

A snapshot is a *view*: it deliberately drops the event log, the per-holding
realised P&L and the fee bookkeeping, so it cannot rebuild the book by replay.
Restoring from it would silently change the accounting. The state file is the
recovery format, and it is versioned so a reader refuses a file it does not
understand rather than misreading it.

To survive a restart, save the state and restore it into a fresh session:

```python
session.save_state("runs/paper_2026-01-05.state.json")

# ... the process restarts ...

from tradingagents.execution import PaperSession, dict_quote_source

restored = PaperSession.restore(
    {},                                          # or state=... directly
    quotes=dict_quote_source({"AAPL": 196.10}),  # a callable cannot be serialised
    path="runs/paper_2026-01-05.state.json",
)
restored.work_resting()   # the resting orders are still live
```

The price source is supplied again by the caller because a callable is not data.
Everything else is restored exactly: cash, positions, average entries, realised
P&L, fees, the event log with its sequence counter (so a resumed run's events
continue rather than collide), the live resting orders with their remaining size
and weighted average fill price, the risk limits, the cost schedule, the
simulator's participation setting, and the processed-signal set.

`PaperSession.restore` raises `PaperStateError` for a malformed file, an unknown
`schema_version`, a state claiming a live mode, **or a state that is internally
inconsistent** — a negative cash balance or holding, a non-finite figure, an
event log that is not dense and 1-based, a `seq` counter that disagrees with it,
or a resting order in a lifecycle state it could not have reached (a `FILLED`
order with a remainder, an `ACCEPTED` order with a fill). It never restores a
book that would be silently wrong. A restored session is still `PAPER` and still
cannot reach a venue.

A caller can cancel a restored order with `session.cancel_resting(order)`, and
that cancellation is durable: it removes the order from the live queue, records
exactly one terminal `ORDER_CANCELLED` event, survives a further save/restore,
and is never undone — a cancelled order is not filled when its price is reached
on a later session. Cancelling an already-terminal order is a no-op, so a
duplicate cancel does not log a second event.
(`tests/test_paper_recovery.py::TestCancellationAfterRestart`.)

### What recovery does and does not guarantee

The design is explicit save, restore from the last saved state — no write-ahead
log, no automatic checkpoint, no supervisor. That supports one claim:

- **Safe:** the system deterministically recovers from a *valid saved state*.
- **Not claimed:** the system is *crash-safe*. State written after the last
  `save_state` is lost on a hard crash. The caller owns the checkpoint cadence; a
  lost tail is lost data, not a corrupted book.
- **Not claimed:** a *fill* is guarded against a stale price. The engine fills
  against the quote its caller supplies; the caller owns quote freshness. The
  session enforces one freshness rule of its own — the day clock only moves
  forward, so a quote dated before the session's last date cannot roll the day or
  reset the daily-loss baseline — but a quote that is merely old is still a valid
  price to fill at, and a vendor source is expected to refuse a bar outside its
  staleness window before it ever reaches the session.

## Duplicate processing

A signal's `signal_id` is a deterministic digest of its decision, so replaying
the same decision — a crash-recovery re-run, a duplicated message — produces the
same id. The session records the ids it has processed and refuses a second one
with a `DUPLICATE_SIGNAL` step and no order, so a re-driven decision cannot
produce a second position or a second fee. The refusal is explicit, not silent.

A price-source *failure* is different: the signal is not marked processed, so a
data outage does not consume a decision that was never acted on. The caller can
retry the same signal once the source recovers.

## Determinism

Given the same quotes in the same order, two runs produce the same cash, the same
positions, the same fees and the same event semantics — whether or not one of
them was interrupted by a save and a restore. The recovery tests assert exactly
that: Scenario A (no restart), Scenario B (restart at the same point) and
Scenario C (restart mid-sequence) all agree. Event `seq` numbers are part of the
restored state, so even they match; only wall-clock timestamps, which come from
the caller, are compared by their semantic fields rather than by value.

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

It is also not a crash-proof service. The state file is a durable, versioned
snapshot written when the caller asks for it; there is no write-ahead log, no
automatic checkpoint on every step, and no supervisor. A restart recovers exactly
the state that was last saved, so any step after the last `save_state` is lost —
the caller decides how often to save. What the recovery mechanism *does* prove is
that a restored session continues with identical accounting and no duplicated
fills; what it does not provide is the durability policy, which belongs to the
caller.

A restored run continues on the bars it saved, and it does not re-fetch them. That
is deliberate on a quota-limited vendor, but it also means the run's data does not
grow on its own. The prospective paper experiment
([PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md#refreshing-vendor-data)) adds an
explicit, append-only *refresh* on top of recovery: before advancing, a vendor
daily source's newer sessions are appended to the same experiment. That is a
separate concern from recovery — a resume still never re-fetches — and the two
should not be described as the same thing.

## A broker connection, when you actually want one

The simulated session above fills against the price you hand it. When you want to
drive a real *paper* account — a broker that accepts orders and fills them against
a live feed, with its own simulated money — use
`tradingagents/paper_experiment/broker/` (Alpaca Paper). It is deliberately
separate: it keeps its own account state, never writes this session's ledger, and
cannot be pointed at a live venue. See
[PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md#alpaca-paper-broker-separate-opt-in-paper-only).
This session remains the offline simulator; the broker is the online paper venue.

The broker has a deliberate start/stop loop (`broker/trading.py`), with its own
durable, versioned state file (`worker_state.json`) and a dashboard **Paper
trading** tab. The dashboard **START** runs the worker **continuously**: one
background loop that waits `poll_interval_seconds` and then calls `run_once()`
once per interval until **STOP**, which signals the loop and waits for the
in-flight cycle to finish. It is opt-in (`--allow-paper-trading`) and paper-only; see
[PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md#the-paper-trading-worker-explicit-opt-in-paper-only).

## Running a session forward over time

This module is one decision at a time. To record an LLM strategy's decisions
forward in time, execute them at the next session's open and evaluate them as
later data arrives, use the prospective layer built on top of it:
[PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md). It drives this same session, adds a
durable hash-chained journal and a report, and keeps decision, execution and
evaluation time apart. It adds no live path.
