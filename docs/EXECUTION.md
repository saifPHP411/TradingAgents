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
