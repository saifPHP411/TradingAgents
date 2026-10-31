# Risk

The risk engine is where a model's opinion stops being an opinion. A signal is a
suggestion; a `RiskDecision` is a verdict. Nothing reaches an order without one.

`risk/engine.py` holds the engine and the limits.

## The limits

All of these live in `RiskLimits` and are configuration, not code:

| Limit | Meaning | Default |
|---|---|---|
| `max_position_size` | Most equity one position may represent, as a fraction | 0.20 |
| `max_portfolio_exposure` | Most equity all positions together may represent | 0.95 |
| `max_open_positions` | Most distinct open positions at once | 10 |
| `max_daily_loss` | Most of the day's opening equity lost in one day | disabled |
| `max_drawdown` | Most peak-to-trough decline tolerated | disabled |
| `max_order_value` | Largest single order notional | disabled |
| `min_order_value` | Smallest single order notional | disabled |
| `default_position_size` | Fraction used when a signal requests no size | 0.05 |
| `long_only` | Whether shorts are forbidden | True |

A limit of `None` disables that check. That is deliberate: a research simulation
without exposure caps should be able to say so explicitly rather than set a limit
to a huge number and hope. The disabled state is named in the decision's notes,
so a report can show which checks could not fire.

`RiskLimits` validates its own coherence at construction: `default_position_size`
must not exceed `max_position_size`, and `long_only=False` is refused because the
engine only implements long-only. A configuration error fails at construction
rather than changing behaviour quietly.

## Long-only, deterministically

There is no prompt involved in enforcing long-only. Three rules do it:

1. A sell larger than the holding is refused with `SHORT_NOT_ALLOWED`.
2. An order's quantity is always positive; direction lives in the side. There is
   no "negative quantity means sell" convention for a sign error to exploit.
3. A fill that would take a position negative raises in the ledger, before any
   state changes.

## Position sizing

Sizing happens in this order, and each step can only reduce the size:

1. Start from the signal's `position_size`, or `default_position_size` if absent.
2. Cap at `max_position_size` of current equity.
3. Cap at the remaining exposure headroom under `max_portfolio_exposure`.
4. Cap at available cash.
5. Cap at `max_order_value`, if set.
6. Refuse if the result is below `min_order_value`, if set.

The cap is applied to the *resulting position*, not to each order. That
distinction was a real bug: bounding each order let repeated buys accumulate past
the cap, because two orders of half the cap summed to the whole cap and the check
passed both times. The regression test is
`tests/test_risk_engine.py::TestPositionCap::test_repeated_buys_cannot_exceed_the_cap`.

## The cap is re-checked at the fill price

Sizing happens when the decision is taken, against the price the decision saw. The
fill happens later, at a price that can have moved: in a backtest it is the next
session's open, which can gap, and in paper mode the spread and slippage move the
fill off the quote. A cap satisfied at the decision price can therefore be breached
by the fill. A position sized at 20% of equity on a 100 close is worth 40% when the
fill opens at 200.

The engine is event-driven, so the check and the fill are necessarily one session
apart. `RiskEngine.review_fill` closes the gap: it re-runs the same cap arithmetic
against the price the fill will actually get, and both the backtester and the paper
session call it immediately before submitting a buy, when the fill price is known
and the ledger has not yet moved. A SELL is not re-checked: it reduces the position
and the exposure, so a gap can only make it smaller.

When a cap is exceeded the fill is *reduced* to the largest size that keeps every
cap, rather than refused. A smaller fill is still a fill, and reducing is what the
sizing path already does when a request is too large. The reduction is recorded as
a `RiskReason` and a note, so a run is auditable after the fact, and only a fill
that leaves no positive size is refused outright.

Two properties make the reduction safe to apply:

- `review_fill` never touches the ledger. It computes and returns a decision, so a
  size that is not executed is charged nothing, and no event is created for it.
- The size is projected twice. Slippage depends on size, so after a reduction the
  fill price is re-projected at the smaller quantity — which is the conservative
  one — and the caps are confirmed against that. The order that is finally
  submitted is the one whose value was checked.

The ledger's own constraints still hold through a gap: cash cannot go negative and
a sell cannot take a position short. It is only the equity-relative sizing caps
that a gap could breach, and this is the check that stops them.

## Book-level checks come first

A book that is already over a limit must not accept a new position, whatever the
individual order looks like. So the drawdown, daily-loss and open-position-count
checks run before sizing, and a failure there refuses the signal outright.

`equity_peak` and `day_start_equity` are supplied by the caller, because the
engine is stateless and reads only the ledger it is handed. When they are absent,
the drawdown and daily-loss checks cannot fire, and the decision says so in its
notes rather than passing silently.

## The verdict

A `RiskDecision` carries the verdict, the approved size, the reference price, the
notional, and a list of named `RiskReason` values with human-readable notes. The
reasons are a closed vocabulary, so a refusal is analysable: a report can count
how often each limit bound, rather than reading prose.

The engine is stateless apart from its limits. Two engines with the same limits
give the same verdict for the same book, which is what makes enforcement
testable.
