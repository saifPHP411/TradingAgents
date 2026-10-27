# Adversarial audit of the simulated trading system

A second, adversarial pass over the simulated trading layer: not "does the code
run", but "where does it quietly do the wrong thing". Every finding below was
reproduced against the working tree at `683895d`, and every fix carries a
regression test that fails on the old behaviour.

Method. The audit drove the real code paths — the ledger, the execution
simulator, the risk engine, the portfolio backtester and the paper session — with
inputs chosen to break an invariant, and checked the invariant directly rather
than trusting a passing suite. The four rules applied throughout: an invariant
must hold for *every* input, not the happy one; a number the system reports must
be measured, not defaulted; a constraint the system claims to enforce must be
enforced in code that a caller cannot skip; and a rejection must not be a silent
loss of a fill the book could still have made.

## Findings

### 1. A buy sized to available cash was refused, not reduced, at the fill

Severity: high. It silently drops trades and it interacts with every fee.

The risk engine sizes an order against the cash available at the decision price.
The fill happens at the next session's open, and it carries a commission. A buy
sized to the whole book therefore had an all-in cost above the cash the moment
either a commission or an up-gap applied, and the simulator rejected the entire
order with `INSUFFICIENT_CASH`.

Reproduced: a 100,000 book, a signal requesting `position_size=1.0`, a price of
100 flat, and `commission_rate=0.001`. The order for 1,000 units needs 100,100
with the fee; the 100,000 of cash is 100 short; the run ends flat with 100,000 in
cash and no trade. With no commission the same order fills. A 5% up-gap produces
the same total loss of the fill.

This is not a risk-limit rejection — the position would have been well inside
every cap — it is a fill the cash could still have paid for, discarded.

Fix: the fill-time re-check now caps a buy's size to the fee-inclusive affordable
notional, `CostModel.affordable_notional(cash, price, is_maker)`, computed at the
price the fill will actually get. The backtester (`backtesting/engine.py`) and
the paper session (`execution/paper.py`) both pass it to
`RiskEngine.review_fill(..., max_notional=...)`, so the reduction happens where
the fill price is known and the ledger has not moved. A size that affords nothing
is still refused. The simulator's own `INSUFFICIENT_CASH` guard is unchanged, so
a hand-built order that bypasses the re-check is still caught.

Tests: `tests/test_backtest_engine.py::TestBuySizeCappedByCashAtFill`,
`tests/test_paper_trading.py::TestPaperBuyCappedByCashAtFill`,
`tests/test_execution_costs.py::TestAffordableNotional`.

### 2. An OHLC-inconsistent bar aborted a run mid-flight

Severity: medium. It turns one bad vendor row into a lost run.

`MarketDataFeed._normalize` dropped rows that were non-finite or non-positive,
but not rows whose own prices disagreed with each other. A row with `high < low`,
or an open/close outside `[low, high]`, survived construction; `Bar`'s validator
rejected it only when `bar()` was called, which happens mid-backtest on the
session that row belongs to. A run holding a position died on that date.

Reproduced: five sessions where the third has `high=99, low=100`. A flat run
survived (nothing forced the bar to be read); a run holding a position through
that session raised `ValidationError` and aborted.

Fix: `_normalize` now applies `Bar`'s own consistency rule (same 1e-9 tolerance)
at construction, so a row that survives construction can never raise when served.
The date simply has no bar, which every caller already handles.

Tests: `tests/test_backtest_data.py::TestInconsistentOhlcBarsAreDroppedAtConstruction`.

### 3. The report's exposure and turnover were always zero

Severity: medium. A reported metric that is structurally zero is worse than a
missing one, because it reads as a measurement.

`build_report` computed metrics with a direct `compute_metrics` call that passed
no `exposure` series and no `traded_notional`. Both therefore defaulted, so a
fully invested run reported `exposure = 0.0` and `turnover = 0.0` in its JSON and
its HTML while the ledger's own `total_exposure` was 0.81.

Reproduced: an 80%-invested run reported `exposure = 0.0`, `turnover = 0.0`.

Fix: `build_report` now uses `performance.report.metrics_from_result`, which
supplies the exposure series from the equity curve and the traded notional from
the `ORDER_FILLED` events. The report and the standalone metrics path now agree by
construction rather than by coincidence.

Tests: `tests/test_backtest_report.py::TestExposureAndTurnoverAreMeasured`.

### 4. `maker_fee_bps` and `taker_fee_bps` were declared but never charged

Severity: low, but it is a silent no-op on a documented feature.

Both fields existed on `CostModel`, were validated, and were described in
`docs/ARCHITECTURE.md` as part of the cost schedule. No code path read them, so a
schedule that set them collected nothing, and a backtest understated its costs.

Fix: `CostModel.fee_for_fill(quantity, price, is_maker)` adds the maker rate for a
resting LIMIT fill and the taker rate for a MARKET fill on top of the commission,
and the execution simulator charges it. Both rates default to zero, so an existing
schedule is unchanged.

Tests: `tests/test_execution_costs.py::TestMakerTakerFees`.

### 5. The strategy could move the book it was given to read

Severity: high. It lets a run report a result no order produced.

`StrategyContext`'s docstring promised the ledger was "read-only for context ...
but no way to move it", and the field held the live `Ledger`. A strategy could
therefore write `context.ledger.cash = 10_000_000`, or call
`context.ledger.apply_fill(...)` and conjure a position at a price of its
choosing. Any such strategy invalidated the backtest it was supposedly testing,
and the corruption is invisible in the output: the numbers still add up.

Reproduced: a strategy that sets `cash` and calls `apply_fill` on the first
session ends the run with 10,000,001 in cash and 4,995 units of `X` at an entry
of 1.0 — a book no order could have produced. The probe ran on the working tree
at `dee70f1~1`.

Fix: `StrategyContext.ledger` is a `LedgerView`, a read-only window. Assignment
and deletion raise `AttributeError`, the mutating methods (`apply_fill`, `mark`,
`charge_fee`, `record_event`) are simply not on the view, and `holding`,
`open_positions` and `holdings` return copies, so mutating what the strategy
receives cannot reach the book. Reads — `cash`, `equity`, `position_quantity`,
`total_exposure`, `snapshot` — are unchanged, so a strategy that only inspects
the book behaves exactly as before. The view wraps the live ledger rather than
snapshotting it, so the figures stay current within a session.

It is not a security boundary: Python has no private attributes, and a caller
that goes looking for the wrapped object can find it. It closes the accidental
and the obvious path, which is the guarantee the docstring makes.

Tests: `tests/test_backtest_engine.py::TestTheStrategyCannotMoveTheBook` (five
tests, four of which fail on the old behaviour).

## What the audit confirmed holds

Each of these was attacked and did not break:

- **Accounting.** The conservation identity
  `cash + cost_basis + total_fees - realized_pnl == initial_cash` holds across a
  round trip and across a full backtest, with and without fees.
- **Long-only.** A negative explicit quantity is refused
  (`NON_POSITIVE_QUANTITY`), an oversell is refused (`INSUFFICIENT_POSITION`), and
  no path produced a negative position. `Order.quantity` cannot be constructed
  non-positive, so the sign-error route is closed at the model.
- **Non-finite inputs.** NaN and ±inf are rejected at the market price, the fill
  price, the fill quantity, the mark price and `max_notional`. NaN is the case
  that matters, because `x <= 0` is false for it.
- **Look-ahead.** The strategy's feed is bounded per session on every accessor:
  `history`, `bar`, `latest_bar_at_or_before` and `calendar` all refuse to answer
  past the current session, and `next_session`-then-`bar` returns `None`. No
  accessor leaked a future bar in the probe.
- **Collective caps.** Two buys of 60% each cannot take exposure past a 95% cap,
  `max_open_positions` holds across successive orders, and cash never goes
  negative across competing orders.
- **Fill-time caps.** A buy sized at 20% on a 100 close is reduced to 20% at every
  fill price from 100 to 1,000, and a down-gap (decision 200, fill 100) does not
  over-reject.
- **Order lifecycle.** A terminal order is not re-filled; a rejection leaves cash,
  fees and events unchanged; every fill is followed by a position event and the
  sequence numbers are dense.
- **Drawdown and daily loss.** Both fire at their configured thresholds, and both
  record a note rather than passing silently when the history they need is absent.
- **Determinism.** Two runs of the same feed and config produce identical final
  equity and byte-identical event logs.
- **Report safety.** Model reasoning is HTML-escaped, so a `<script>` in the
  reasoning is rendered as text; the report is JSON-serialisable.
- **Telemetry.** The cost ceiling and the call ceiling both raise before the call;
  an unpriced model records `None` rather than a fabricated cost.

## Residual limitations (unchanged)

- The fill-time re-check uses the *projected* fill price. For a MARKET order that
  is the price the simulator uses, so the two agree; an order type that filled at
  a price not derivable from the bar would need a post-fill reconciliation.
- Prices are split-adjusted, so a backtest reflects price return only; dividends
  are not credited as cash.
- LLM calls are not deterministic. The deterministic layers around them are
  reproducible from the run configuration; the calls are not.
