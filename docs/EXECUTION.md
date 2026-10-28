# Execution

Execution turns an approved order into a fill or a rejection, and charges what
the fill costs. It never reaches a venue.

`execution/orders.py` defines the order and its lifecycle.
`execution/costs.py` holds the cost schedule.
`execution/simulator.py` does the filling.
`execution/interface.py` holds the guard that makes a live path unreachable.

## The order

An `Order` carries the symbol, the side, a positive quantity, the type, the
creation time and the analysis date it came from. Direction lives in `side`, so
there is no negative-quantity convention for a sign error to exploit.

Lifecycle states, in `OrderStatus`:

```
CREATED → ACCEPTED → FILLED
                   → PARTIALLY_FILLED
                   → REJECTED
                   → CANCELLED
```

An order is never sent anywhere. The simulator fills it against a
`MarketContext` handed in by the caller, so there is nothing to send it to.

## Order types

Two, and only two:

- `MARKET` — fills at the market price plus costs, subject to available cash or
  position. Always `DAY`: it either fills on its bar or expires.
- `LIMIT` — fills only if the market reached the limit. With `DAY` it is worked
  once; with `GTC` it rests across sessions until it fills or expires.

A LIMIT order's reachability uses the bar's low/high when the caller supplies
them: a buy limit fills if the session traded down to it (the low proves it), and
a sell limit if it traded up to it (the high). Without a range the reference price
alone decides. The limit caps the price, it does not set it — a bar that traded
through the limit fills at the better price.

`TimeInForce` has two members:

| Member | Meaning |
|---|---|
| `DAY` | Worked once, on the first eligible bar; cancelled if that bar misses it |
| `GTC` | Rests across eligible bars until it fills or reaches its expiry |

`GTC` here means "rests until filled or expired", not the venue-level guarantee
of queue priority. A daily OHLC bar cannot prove the order sat at the front of a
queue, so a fill on a bar whose range covers the limit is a modelled fill, not a
claim about the real book.

There is no stop, stop-limit or trailing variant. Those need a price feed between
bars to trigger honestly, which a daily-bar simulation cannot provide. A stop
that silently triggered on a daily close would be a lie about what a real stop
does, so the backtester enforces stops at the bar level and says so.

## Rejections

Every rejection carries a named `RejectReason` from a closed vocabulary, so
refusals are countable rather than prose:

| Reason | When |
|---|---|
| `INVALID_QUANTITY` | Zero or negative quantity |
| `INVALID_PRICE` | A non-positive price |
| `INSUFFICIENT_CASH` | A buy the cash cannot cover |
| `INSUFFICIENT_POSITION` | A sell larger than the holding |
| `MARKET_CLOSED` | The market context says the session is closed |
| `NO_PRICE` | No price is available for the symbol |
| `RISK_LIMIT` | The risk engine refused it |
| `LIMIT_NOT_REACHED` | A limit order the bar did not reach was cancelled |
| `EXPIRED` | A resting order reached its expiry unfilled |
| `BUDGET_EXHAUSTED` | The run's budget ran out |

A rejection never touches the book. Every check happens before any state change.

## Costs

`CostModel` is the schedule. Every rate defaults to zero, so a run that does not
opt into costs is not silently charged:

| Field | Meaning |
|---|---|
| `commission_flat` | Flat fee per fill |
| `commission_per_unit` | Fee per unit traded |
| `commission_min` | Minimum commission per fill |
| `commission_rate` | Commission as a fraction of notional |
| `half_spread_bps` | Half the bid/ask spread, against the trade |
| `slippage_bps` | Base adverse price move |
| `slippage_volume_factor` | Extra slippage scaled by size against volume |

Costs are applied in the simulator and land in the ledger, not just in a report.
A run with costs has a lower equity than the same run without, and the difference
is the fees plus the price impact. `CostBreakdown` records the split, so a report
can show where the money went.

The spread is applied against the trade: a buy pays the offer, a sell receives
the bid. Slippage is always adverse, in the direction that costs the trader.

## Partial fills

`max_volume_participation` caps how much of a bar's volume an order may take. It
defaults to 0, which disables partial fills. When set, an order larger than its
share fills partially and the remainder is reported, rather than being filled
whole against a bar that could not have absorbed it.

A partial fill applies only the filled quantity to the book: cash, position, fees
and turnover all move by the piece that actually filled, never by the requested
size. `Order.filled_quantity` is the running total and `Order.average_fill_price`
is the quantity-weighted average of every piece, so the order's record of its own
fills agrees with the ledger's VWAP for the position.

## Persistent (resting) orders

A caller that works an order across several bars passes `rest_if_unfilled=True`
to `submit`. A LIMIT order the bar did not reach is then left `ACCEPTED` instead
of being cancelled, so the same order object can be worked again on a later bar.
The order is cancelled only when the caller asks, when it reaches its expiry, or
when the caller stops working it.

- Acceptance is recorded once. `Order.accepted_recorded` guards the
  `ORDER_ACCEPTED` event, so a resting order worked on ten bars still logs one
  acceptance.
- A resting order is never cancelled just because a bar missed it, and no event
  is recorded while it rests — a `CANCELLED` event would say the order is finished
  when it is not.
- Expiry is explicit. `Order.expires_session` is the last session the order may be
  worked on; `BacktestConfig.limit_order_ttl_sessions` bounds the sessions it may
  be worked for. Either bound is checked before the order is worked, so an order
  never fills on the session after it should have expired. An expired order is
  cancelled with `EXPIRED` and stamped with the expiry session's time.
- The remainder of a partially filled resting order stays live. Only the
  increment each session applies is registered with the exit tracker, so a
  three-session fill is one position, not three.

`BacktestConfig.resting_limit_orders` (default `True`) turns the whole path on or
off; off, every order is single-shot, exactly as before.

## Session liquidity

When `max_volume_participation` is set, every order worked against one session
draws from a shared per-symbol allowance: the configured share of that symbol's
own traded volume. Two orders for the same symbol therefore compete for the same
liquidity rather than each assuming the whole bar. The allowance is consumed in
queue order, so the outcome is deterministic. Orders for different symbols have
independent liquidity. Both engines share this rule: the backtester computes the
allowance in `_session_liquidity_budget` and the paper session in
`PaperSession._session_budget`, so a backtest and a paper run over the same thin
session fill the same size.

Two limitations are real and are stated rather than hidden:

- **A partial fill of a MARKET order is not re-queued by the backtester.** A
  MARKET order that fills part of its size ends `PARTIALLY_FILLED` and the
  remainder is reported in `BacktestResult.unfilled`; the backtester does not
  carry it to the next session. A resting LIMIT order's remainder *is* carried
  forward. A caller that wants a MARKET remainder filled must re-submit it (the
  simulator will fill the remaining size on a second `submit`).
- **A resting fill is a modelled fill.** A daily bar can prove the price traded,
  not that a queue would have filled the order there. `GTC` is therefore
  documented as "rests until filled or expired", not as a queue guarantee.

## The safety guard

`ExecutionMode` has exactly two members, `BACKTEST` and `PAPER`. There is no
`LIVE` to select.

- `resolve_mode` refuses any string naming a live mode — `live`, `real`,
  `production`, `broker`, `exchange` — by name, with a hard error rather than a
  warning.
- `ExecutionSimulator.__init__` calls `assert_simulation_only`, so even a
  subclass cannot be constructed in a mode that is not simulation.
- The simulator takes prices as arguments rather than fetching them, so there is
  nothing for a live implementation to hook into without a new class being
  written deliberately.

This is not a policy that could be configured away. It is the absence of a code
path.

## Persistence and restart recovery

Reporting and recovery are separate concerns with separate files:

- `Ledger.snapshot()` and `PaperSession.snapshot()` / `write_json()` are
  **reporting** views. A snapshot carries the headline figures and a summary of
  the live orders, but not the ledger's event log, per-holding realised P&L or
  fee bookkeeping, so it cannot rebuild a book. It is not a recovery mechanism.
- `Ledger.to_state()` / `Ledger.from_state()` and `PaperSession.state()` /
  `save_state()` / `restore()` are the **recovery** format. They capture the
  whole book — including the event log and the dense `seq` counter — plus the
  risk limits, the cost schedule, the simulator settings, the live resting orders
  with their remaining size and weighted average fill price, the processed-signal
  set and the day baseline.

Both files carry a `schema_version` (`PAPER_STATE_SCHEMA_VERSION`, currently `1`).
`PaperSession.restore` refuses a file whose version it does not understand, and
raises `PaperStateError` for a malformed file or a state claiming a live mode,
rather than restoring a book that would be silently wrong. A restored session is
still `PAPER`: recovery adds no path to a venue.

The accounting is restored exactly, not recomputed: a resumed run's cash,
positions, average entries, realised P&L and fees equal the original's, and its
event sequence continues rather than restarting. `tests/test_paper_recovery.py`
proves this by running the same sequence twice — once straight through, once with
a save/restore in the middle — and asserting the two agree on every figure and on
the event semantics.

Recovery is not a durability policy. There is no write-ahead log and no automatic
checkpoint; the caller decides when to call `save_state`, and only the last saved
state is recovered.

The prospective paper experiment ([PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md))
supplies that durability policy for its own use: it saves state after every
session, keeps an append-only hash-chained journal, and records the execution
session of every decision so a resumed run fills at the same open a straight run
would. It adds no order-routing path.
