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

### 6. A partially filled order reported its last fill price as its average

Severity: high. It understates or overstates what an order actually paid, and it
is silent — the cash the ledger spent is right, so nothing in the output
contradicts the wrong figure.

`ExecutionSimulator._apply` set `order.average_fill_price = price` on every fill.
An order that fills in pieces at different prices — a MARKET order capped by
`max_volume_participation` over two sessions, or a limit worked over two bars —
therefore ended with the *last* piece's price in a field named
`average_fill_price`, and `Order.notional` (defined as `filled_quantity *
average_fill_price`) disagreed with the cash the ledger actually spent.

Reproduced on the working tree at `e7853e4`: a 1,000-unit order that filled 400
@ 10 and 600 @ 20 reported `average_fill_price = 20.0` and `notional = 20,000`,
against a true average of 16.0 and a true notional of 16,000. The ledger's own
`average_entry_price` was 16.0, so the two records of the same fills disagreed.

Fix: `_apply` accumulates a quantity-weighted average across every piece:
`average = (prior_notional + applied * price) / filled_quantity`. A single fill
is unchanged (`applied == quantity`, so the formula reduces to `price`), and the
order's notional now equals what the ledger spent. The ledger's VWAP was already
correct and is untouched.

Tests: `tests/test_execution_simulator.py::TestPartialFills` — the average across
two pieces, the 1000/400/600 example reconciled against the ledger, and fees
charged per piece. The average test fails on the old behaviour (obtained 20.0,
expected 16.0).

### 7. A LIMIT order was single-shot, and its remainder could not rest

Severity: medium. It made a whole class of strategy inexpressible, and the way it
was handled hid the fact: an unfilled limit was *cancelled*, so a run reported a
terminal outcome for an order that a real venue would have left working.

The simulator is given one bar per `submit`. A LIMIT order the bar did not reach
was cancelled with `LIMIT_NOT_REACHED` and became terminal, so it could never fill
on a later bar through the same object. There was no good-till-cancelled rest and
no expiry. Neither the backtester nor the paper session ever created a LIMIT
order, so the type was exercised only through the simulator's own API.

Two further defects sat behind that one:

- **Acceptance would be re-recorded.** The simulator appended an `ORDER_ACCEPTED`
  event on every `submit`. Working one order across several bars would have logged
  one acceptance per bar, so the event log would show an order accepted five times.
- **A resting remainder would double-count.** `_register_entry` used the order's
  cumulative `filled_quantity`, so an order that filled 40 then 60 units across
  two sessions would have registered 40 + 100 = 140 units of exit tracking against
  a 100-unit position.

Fix: an order can rest. `submit(..., rest_if_unfilled=True)` leaves a GTC LIMIT
the bar did not reach `ACCEPTED` instead of cancelling it, and records no event
while it rests. `Order.accepted_recorded` guards the acceptance so it is logged
once. `BacktestConfig.resting_limit_orders` and `limit_order_ttl_sessions`, plus
`Order.expires_session`, bound how long an order may rest; an expired order is
cancelled with the new `EXPIRED` reason and stamped with the expiry session's
time. LIMIT reachability now uses the bar's low/high when supplied, so a limit the
session traded down to (but did not close at) fills. The engine registers only the
*increment* each session via `_apply_fill_to_tracking`, so a multi-session fill is
one position. The backtester and the paper session now both emit LIMIT orders from
a signal's `order_intent`, and the paper session holds resting orders in
`session.resting`, advanced by `work_resting()`. Competing orders for one symbol
against one session share that symbol's liquidity allowance, consumed in queue
order.

Reproduced: a GTC buy limit at 95 with a session that opens at 100 and closes at
100 was cancelled on its first session, and a later session opening at 94 never
saw the order. A two-session partial fill registered 140 units of exit tracking
against a 100-unit position.

Tests: `tests/test_persistent_limit_orders.py` (resting, expiry, TTL, partial
remainder, low/high reachability, the single-acceptance guard, and the
single-position accounting through the backtester) and
`tests/test_paper_trading.py::TestPaperRestingLimits`.

## Phase 3 — paper-trading reliability, persistence and recovery

A second pass over the *paper session* specifically, asking not "does a backtest
survive this" but "does a long-running paper process survive a restart, a data
outage and a repeated step without changing the accounting". The reported Phase 2
state was taken as a claim to verify, not a fact; the findings below were each
reproduced against the working tree before any fix.

### 8. There was no recovery mechanism, only a reporting snapshot

Severity: high. The whole point of a paper run is that it can be stopped and
resumed, and it could not be.

`PaperSession.write_json` wrote a snapshot that looked like state but was a
report: it omitted the ledger's event log, the per-holding realised P&L, the
per-holding fee bookkeeping and the `seq` counter. Restoring from it would have
silently changed the accounting, and no restore path existed in any case.

Fix: a distinct, versioned recovery format. `Ledger.to_state` / `from_state` and
`PaperSession.state` / `save_state` / `restore` capture the whole book (event log
and sequence counter included), the risk limits, the cost schedule, the simulator
settings, the live resting orders with their remaining size and weighted average
fill price, the processed-signal set and the day baseline. A snapshot and a state
file are now documented as different things. `PaperSession.restore` raises
`PaperStateError` for a malformed file, an unknown `schema_version`, or a state
claiming a live mode.

Tests: `tests/test_paper_recovery.py::TestRestartEqualsNoRestart`,
`TestSnapshotIsNotRecovery`, `TestFilePersistence`.

### 9. A price-source exception mid-`work_resting` left the queue half-advanced

Severity: high. One broken symbol could corrupt the resting queue.

`work_resting` looped over the resting orders and rebuilt `self.resting` only at
the end. A quote source that raised for one symbol propagated out of the loop
with `self.resting` still holding the pre-loop list, so an order that had already
filled on an earlier iteration stayed in the queue marked `FILLED` — and the next
`work_resting` would skip it, silently. The book had moved for the fill, but the
queue and the step list disagreed with it.

Fix: the loop is exception-safe. A quote that cannot be read leaves that order
resting, the remaining orders are still worked, and the first failure is
re-raised *after* the loop so it is never swallowed. The book is never touched by
a failed read, and a filled order is never re-queued.

Tests: `tests/test_paper_recovery.py::TestMarketDataFailures`.

### 10. A closed market rejected a resting GTC order instead of skipping it

Severity: medium. It cancelled a live order for a reason that is not about the
order.

`work_resting` passed the quote's `is_open` straight into the simulator, so a
quote that arrived with `is_open=False` made the simulator reject the resting
order with `MARKET_CLOSED` — a terminal state for an order whose limit simply had
not been reached. A market that is closed is a reason to wait, not to kill the
order.

Fix: a closed quote now leaves the order resting, consistent with a missing
quote. The order is only ever cancelled when it expires or the caller asks.

Tests: `tests/test_paper_recovery.py::TestMarketDataFailures`
(`test_a_closed_market_leaves_a_resting_order_live`).

### 11. A partially filled resting order's remainder was dropped on the first work

Severity: medium. The remainder of a persistent limit could vanish.

`on_signal` only added an order to `self.resting` when its status was exactly
`ACCEPTED`, so a GTC limit that filled *part* of its size on the first attempt
(participation configured, a thin quote) ended `PARTIALLY_FILLED` and was never
queued — its remainder was silently lost, and a later `work_resting` could not
continue it.

Fix: an `ACCEPTED` *or* `PARTIALLY_FILLED` GTC order is queued, so the remainder
rests. A partial fill's increment is what reaches the ledger, and the order's
weighted average fill price blends the pieces.

Tests: `tests/test_paper_recovery.py::TestPartialFillRecovery`.

### 12. A resting order was not re-checked against the risk caps at its fill price

Severity: medium. A gap to the limit could breach a position cap.

The paper path re-checked a *fresh* buy at its projected fill price, but a
resting order's later fill went straight to the simulator with no re-check. A
resting limit sized at the cap could therefore fill at a price that left the
position past it.

Fix: `work_resting` re-checks a resting BUY's *remaining* size against the caps at
the projected fill price, exactly as a fresh entry is; the size is reduced to the
largest that holds, and the remainder is rejected with `RISK_LIMIT` (recorded)
only when no positive size remains.

Tests: `tests/test_paper_recovery.py::TestPartialFillRecovery::test_a_partial_fill_then_risk_rejection_of_the_remainder`.

### 13. A repeated signal filled a second position

Severity: high. A crash-recovery re-run or a duplicated message doubled the
position.

`on_signal` had no memory of what it had already processed. The same signal — the
same `signal_id`, a deterministic digest of the decision — driven twice filled
twice, charged two fees, and appended two `ORDER_FILLED` events.

Fix: the session records the `signal_id`s it has processed and refuses a second
one with an explicit `DUPLICATE_SIGNAL` step and no order. A price-source
*failure* is not marked processed, so a data outage does not consume a decision
that was never acted on.

Tests: `tests/test_paper_recovery.py::TestDuplicateProcessing`.

### 14. A resting fill reported zero fees, and the snapshot understated the order

Severity: low, but it is a reporting defect on the exact figures a report shows.

`work_resting` recorded `fees=0.0` on its steps even when the fill charged a fee,
and the snapshot's `resting_orders` listed only five fields, so a reader could not
tell `ACCEPTED` from `PARTIALLY_FILLED`, nor see the remaining quantity, the
average fill price, the time in force or the expiry. The snapshot also carried no
schema version and no current-time field.

Fix: `work_resting` records the fee the fill actually charged (from the ledger
delta), and the snapshot reports each order's real status, filled and remaining
quantities, weighted average fill price, time in force and expiry, plus a
`schema_version`, `current_time`, `day` and `day_start_equity`.

Tests: `tests/test_paper_recovery.py::TestSnapshotCorrectness`.

### 15. `work_resting` did not roll the day, so the daily-loss baseline went stale

Severity: low. It is a real defect for a session that only ever rests orders.

`on_signal` rolled the day's opening equity, but `work_resting` did not, so a
session whose activity was entirely resting-order fills kept a stale daily
baseline and a stale `day`.

Fix: `work_resting` rolls the day from the quote's timestamp before working the
queue.

Tests: `tests/test_paper_recovery.py::TestRestartEqualsNoRestart` (day baseline
survives the restart) and the day-roll assertions in `tests/test_paper_trading.py`.

## Phase 4 — final release-candidate adversarial audit

An independent pass over the whole simulated system, attacking accounting,
execution, order lifecycle, partial fills, persistent LIMIT orders, risk
controls, portfolio state, recovery, determinism, backtest/paper consistency,
data handling, reporting, walk-forward evaluation, event integrity and the
simulation boundary. It found three further defects, all fixed with a regression
test each.

### 16. `restore` accepted state a session could never have produced

Severity: high (for recovery). The recovery format's stated promise is that a
corrupt file is refused rather than turned into a different portfolio. It was
not: `Ledger.from_state` trusted the file, and `PaperSession.restore` accepted a
state with a **negative cash balance**, a **negative holding**, **negative
fees**, a **NaN cash figure**, a **rolled-back `seq` counter** (which then minted
**duplicate event sequence numbers** on the resumed run), and an **impossible
order lifecycle** — a `FILLED` order with a remaining quantity, or an `ACCEPTED`
order carrying fill data. Each of those resumed with silently wrong accounting,
which is the one outcome recovery must never have.

Fix: `Ledger.from_state` now re-checks every invariant the live ledger maintains
— cash and holdings non-negative, all figures finite, a held quantity has a
positive average entry, and the event log is dense and 1-based with a `seq`
counter that agrees with it — and raises `LedgerError` on a violation.
`Order.model_validate` gained a lifecycle validator that refuses a status/fill
combination no order could reach, so a restored resting order cannot be
`FILLED` with a remainder or `PARTIALLY_FILLED` with nothing filled.
`PaperSession.restore` wraps both, so a bad file raises `PaperStateError`.

Tests: `tests/test_paper_recovery.py::TestInconsistentStateIsRefused`.

### 17. Paper did not share a per-session liquidity budget across orders

Severity: medium (backtest/paper divergence). The backtester computes one
units-of-volume allowance per symbol per session and draws it down in queue
order, so two orders cannot each fill the whole bar. The paper session computed
the allowance *per order*, so two resting orders for one symbol filled twice the
session's volume — 800 units where a backtest fills 400. The two engines are
supposed to agree by construction; this was the one caller-side place they did
not.

Fix: `work_resting` now builds one per-symbol budget for the session and passes
the running remainder to each order, mirroring `_session_liquidity_budget` in the
backtester. The allocation is consumed in queue order, so it is deterministic.

Tests: `tests/test_paper_recovery.py::TestSessionLiquidityBudgetIsShared` and
`::TestBacktestPaperParity`.

### 18. The session clock could run backwards on a stale quote

Severity: low. `_roll_day` set the day from whatever quote it was handed, so an
out-of-order (earlier-dated) quote moved the day and the daily-loss baseline
*backwards* — resetting the risk history a late-arriving old quote should not be
able to touch, and stamping a fill in the past.

Fix: `_roll_day` only moves forward. A quote on an earlier date than the session
has already seen is ignored for the clock, so `day` and `day_start_equity` never
go backwards.

Tests: `tests/test_paper_recovery.py::TestSessionClockMonotonicity`.

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

Phase 4 added these to the "confirmed holds" list after attacking them directly:

- **Event integrity.** No path produced a duplicate lifecycle event, a `FILLED`
  after a `CANCELLED` or `EXPIRED`, or a non-monotonic / non-dense `seq`; a
  terminal order is never filled again, and acceptance is recorded once per order
  however many bars it rests on.
- **LIMIT boundaries.** A BUY limit fills at the exact boundary and not a tick
  above; a SELL limit fills at the exact boundary and not a tick below; a
  gap-through fills at the limit or better; a no-touch cancels.
- **Reporting.** `total_return`, `max_drawdown`, `win_rate` and the P&L figures
  recomputed by hand from the equity curve and trade list matched the
  implementation; an unfilled or rejected order contributes no trade, no notional
  and no fee.
- **Walk-forward.** A fold whose validation window reaches into its training
  window, or whose test window overlaps its validation window, is refused at
  construction; selection reads validation scores only, never test.
- **Safety boundary.** `ExecutionMode` has no `LIVE`; `resolve_mode` refuses every
  live-sounding string (`live`, `real`, `production`, `broker`, `exchange`,
  `mainnet`); no network, broker or credential import exists anywhere in
  `execution/`, `portfolio/`, `risk/` or `backtesting/`.
- **Recovery validation.** A state file is refused, not resumed, when it is
  malformed, carries an unknown `schema_version`, claims a live mode, or is
  internally inconsistent (see finding 16).

## Residual limitations (unchanged)

- The fill-time re-check uses the *projected* fill price. For a MARKET order that
  is the price the simulator uses, so the two agree; an order type that filled at
  a price not derivable from the bar would need a post-fill reconciliation. This
  is *not* an accounting or risk hole under the current architecture: the
  simulator fills a MARKET order at exactly the price `projected_fill_price`
  returns, and a LIMIT order fills at its limit or better, which the re-check also
  uses. The projection and the fill cannot disagree for either supported order
  type, so no cap can be breached through this path.
- Prices are split-adjusted, so a backtest reflects price return only; dividends
  are not credited as cash.
- LLM calls are not deterministic. The deterministic layers around them are
  reproducible from the run configuration; the calls are not.
- Point-in-time safety is per-source, not global. The market analyst's inputs are
  request-bounded and safe for a historical run; the news, sentiment and
  fundamentals sources are window-filtered or current-only and are withheld from
  one. A full-graph historical run is therefore refused, not faked; see
  `docs/LLM_POINT_IN_TIME_AUDIT.md`.

## Recovery: what may and may not be claimed

The persistence design is explicit save, restore from the last saved state, no
write-ahead log, no automatic checkpoint and no supervisor. That supports exactly
one claim:

- **Safe:** "The system can deterministically recover from a valid saved state."
  A restart from a `save_state` file reproduces cash, positions, fees, P&L, the
  order lifecycle and the event log exactly, and refuses a corrupt file rather
  than resuming with different accounting.
- **Unsafe, and not claimed:** "The system is crash-safe." There is no durable
  checkpoint guarantee, so state written after the last `save_state` is lost on a
  hard crash. A crash-safe claim would need a write-ahead log or a supervisor,
  which this release does not have and does not claim.

This limitation is acceptable for a simulation / paper release candidate: the
caller owns the checkpoint cadence, and a lost tail is a lost tail, not a
corrupted book.

## Phase 10: the prospective paper experiment

The Phase 10 layer ([PAPER_EXPERIMENT.md](PAPER_EXPERIMENT.md)) closes the
checkpoint gap for its own use, and adds three claims of its own, each with a
test that would fail if it were false:

- **Safe:** "A restart does not duplicate a fill." The experiment saves state
  after every session and keeps the execution session of every decision, so a
  resumed run fills at the same open a straight run would; the session's processed
  signal set and the journal's idempotent `event_id` stop a fill being applied
  twice. `tests/test_phase10_paper_experiment.py::TestResumeEqualsNoRestart`
  proves a straight run and a restarted run agree on every accounting figure and
  on the event semantics.
- **Safe:** "A decision is never executed at the price it was made from." A
  decision from session T's close is re-dated to T+1 before it reaches the paper
  session, and the config refuses `execute_on_next_open=False`, so the timing
  cannot be relaxed. `TestLookAheadAndTiming` pins it.
- **Safe:** "The history is tamper-evident." The journal is hash-chained; editing,
  removing or reordering an entry makes `ExperimentJournal.load` raise. The
  `TestJournal` group proves it.

- **Unsafe, and not claimed:** "The experiment is a live paper run." By default the
  price source is the frozen historical dataset, which ends in 2024. Real vendor
  sources now exist (`paper_experiment/providers/`) and are opt-in, so
  `current_market_source_available()` is `True`; but `READY FOR LIVE PAPER` also
  requires a source that can serve a session window, and a quote-only source
  (Finnhub) cannot, so it is not reachable on a quote alone. A vendor source that
  fails stops the run rather than silently substituting the frozen dataset, and the
  CLI refuses to resume a vendor run on frozen data. A missing credential yields a
  *blocked* decision, never a fabricated one, and a test stub is labelled
  `synthetic_fixture` and is never counted as genuine.
- **Safe:** "A vendor's free plan is not assumed to include the endpoint used."
  Alpha Vantage's free function is `TIME_SERIES_DAILY` (the adjusted variant is
  premium and unused, and `outputsize=full` is refused without an explicit
  opt-in); Finnhub's historical candle endpoint is premium, so its provider serves
  a current-day quote only and refuses every historical request. `TestAlphaVantage`
  and `TestFinnhub` pin both, and `docs/MARKET_DATA_PROVIDERS.md` cites the vendor
  pages.
- **Safe:** "A vendor payload cannot poison the book." Every provider's bars pass
  through one validator that refuses NaN, infinity, a non-positive price, a high
  below low, an open/close outside the range, a duplicate or out-of-order session,
  and a non-numeric column. A quota or bad-key notice inside an HTTP 200 is
  classified and raised, never parsed as data.
- **Unsafe, and not claimed:** "The paper result is evidence of profitability." A
  paper mark-to-market is not a realised trade, the costs are the declared model
  and not observed fills, and a sample below `min_trades_for_metrics` is reported
  as `N/A`, not as a number.
