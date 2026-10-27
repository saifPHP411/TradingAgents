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
  position.
- `LIMIT` — fills only if the market price is at or better than the limit.

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

Three limitations are real and are stated rather than hidden:

- **A partial fill is not re-queued.** A MARKET order that fills part of its size
  ends `PARTIALLY_FILLED` and the remainder is reported in `BacktestResult.unfilled`
  — it is not carried to the next session. A caller that wants the rest must
  re-submit (the simulator will fill the remaining size on a second `submit`).
- **A LIMIT order is single-shot.** The simulator is given one bar per `submit`;
  if the limit is not reached on that bar the order is cancelled with
  `LIMIT_NOT_REACHED` and is terminal, so it cannot fill on a later bar through
  the same object. There is no good-till-cancelled rest and no expiry date. This
  is consistent with the simulator's one-bar-per-submit contract, but it means a
  strategy cannot express "rest this limit for the next five sessions" without
  re-creating the order each session.
- **The backtester and the paper session only ever create MARKET orders.** Neither
  the engine nor the paper session emits a LIMIT order, so `LIMIT` is exercised
  through the simulator's own API and tests, not through a run.

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
