# Independent review of PR #1

This is a second-pass review of the simulated trading system delivered on the
`trading-system-development` branch. The first pass (the audit and the initial
implementation) is in `docs/TRADING_SYSTEM_AUDIT.md` and
`docs/FINAL_DEVELOPMENT_REPORT.md`. This file records the review that followed,
the defects it found, and what was done about them.

The method was to read the code and try to break it, not to re-run its tests and
treat a green suite as a verdict. Every defect below was reproduced with a script
before it was touched, and every fix carries a regression test.

---

## Round 1 — input hardening

Seven defects were found that the 1,899-test suite did not catch. Six were fixed;
the seventh was documented and pinned.

| # | Defect | Severity | Outcome |
|---|---|---|---|
| 1 | NaN/inf price, quantity or fee silently corrupted the ledger | High | Fixed |
| 2 | NaN/inf market price produced an order marked `FILLED` | High | Fixed |
| 3 | Re-submitting an already-filled order filled it twice | High | Fixed |
| 4 | `StrategyContext.feed` exposed future bars | High | Fixed |
| 5 | A non-finite bar was accepted at construction, raised mid-backtest | Medium | Fixed |
| 6 | `Quote` accepted NaN/inf prices in paper mode | Medium | Fixed |
| 7 | Position cap could be exceeded when the fill gapped up | Medium | **Fixed in round 2** |

The NaN class was the most instructive. `x <= 0` and `x < 0` are both *false* for
NaN, so a NaN price passed every positivity guard, was applied, and turned `cash`
and `equity` into NaN for the rest of the run — and NaN then compares false against
every later bound, so the exposure check silently stopped firing too. The fix is to
test `math.isfinite` explicitly and never rely on a comparison.

Defect 7 was left open in round 1 because closing it properly is a design change,
not a patch. Round 2 is that change.

---

## Round 2 — the position cap at the fill price

### The defect

The risk engine sizes an order against the price the *decision* saw. The fill
happens at the next session's open, which can gap. So a cap satisfied at the
decision price can be breached by the fill.

Reproduced before the fix, with a 20% cap on a 100,000 book:

```
decision session close: 100.00   ->  20% of equity = 20,000 -> 200 units requested
fill session open:      200.00   ->  200 units = 40,000 = 40% of equity
```

The position that was actually bought was twice the size the limit allowed. The
ledger's own constraints still held — cash stayed positive, no short appeared — so
nothing else flagged it.

### The fix

`RiskEngine.review_fill` re-runs the same cap arithmetic at the price the fill will
actually get. Both execution paths call it immediately before submitting a buy,
which is the only point where the fill price is known and the ledger has not moved:

- `PortfolioBacktester._recheck_entry_at_fill` in the backtest loop, and
- `PaperSession.on_signal` in paper mode.

The fill is **reduced** to the largest size that keeps every cap, rather than
refused, because reducing is what the sizing path already does when a request is
too large and a smaller fill is still a fill. Only a fill that leaves no positive
size is refused. A SELL is not re-checked: it lowers the position and the exposure,
so a gap can only make it smaller.

Three details make the reduction safe:

1. **It never touches the ledger.** `review_fill` computes and returns a decision.
   A size that is not executed is charged nothing and creates no event, so there is
   no fee on an unfilled quantity and no duplicate fill.
2. **The size is projected twice.** Slippage depends on size, so after a reduction
   the fill price is re-projected at the *smaller* quantity — the conservative
   direction — and the caps are confirmed against that. The order finally submitted
   is the one whose value was checked, not an intermediate estimate.
3. **The reduction is recorded.** The binding `RiskReason` and a note go on the
   decision, so a report can say why the filled size differs from the requested one.

`ExecutionSimulator.projected_fill_price` was added for this: it resolves the fill
exactly as `submit` will, but applies nothing, so asking the price costs nothing.

A related reporting defect was found and fixed at the same time. `PaperSession`
recorded the *requested* quantity on a `SessionStep` even when the fill was
reduced, so a report could claim a larger position than the ledger held. It now
records the filled quantity.

### Result

```
before:  200 units x 200 = 40,000 = 40.00% of equity   (cap 20%)
after:   100 units x 200 = 20,000 = 20.00% of equity
```

Paper mode, with 100 bps of half-spread and 100 bps of slippage moving a 100 quote
to a 102 fill:

```
before:  200.0 units x 102 = 20,400 = 20.40% of equity  (cap 20%)
after:   196.08 units x 102 = 20,000 = 20.00% of equity
```

---

## What was tested

Every case the review raised, as a test rather than an assertion in prose.

**Risk engine** (`tests/test_risk_engine.py::TestFillTimeRecheck`) — the cap
arithmetic at the fill price, driven directly:

- a gap that doubles the position is capped to the limit;
- a fill inside the limit is not reduced;
- a gap too small to breach leaves the size alone;
- the reduction accounts for an existing position;
- the exposure cap binds across positions;
- `max_order_value` still applies at the fill price;
- a reduction below `min_order_value` is refused;
- no positive size remains when the book is at the cap;
- a sell is not capped, and is not rejected for exceeding the holding (that check
  is the ledger's, and duplicating it here would let the two disagree);
- NaN/inf fill price and NaN/inf/zero/negative quantity are refused;
- a reduction to a fractional size is still approved, because the ledger is
  quantity-based;
- the re-check does not move the ledger: no cash, equity, fee or event change.

**Backtester** (`tests/test_backtest_engine.py::TestPositionCapEnforcedAtFill`) —
the same guarantee through the real event loop:

- the 20%-to-40% gap is capped to 20% of the marked position;
- a gap within the limit is not reduced;
- a bigger gap leaves fewer units, and every case respects the cap;
- a cap that leaves less than one unit fills that fraction;
- the reduction is visible after the run;
- the accounting identity holds after a capped fill;
- two decisions on consecutive sessions cannot accumulate past the cap;
- a capped fill is charged fees for the reduced quantity only, not the requested one;
- a refusal at the fill is recorded, not silent: the order carries `RISK_LIMIT` and
  an `ORDER_REJECTED` event reaches the log a report is built from.

**Paper mode** (`tests/test_paper_trading.py::TestPaperCapEnforcedAtFill`) — the
same pipeline, so the same guarantee:

- adverse costs do not breach the cap;
- a fill inside the cap is not reduced;
- the step reports the filled quantity, not the requested one;
- a sell is not reduced, and a partial sell sells the requested fraction;
- the accounting identity holds after a capped fill;
- repeated entries cannot accumulate past the cap;
- a refusal at the fill is recorded on the step and in the event log.

### On the old test

`TestPositionCapThroughAGap::test_a_gap_up_fill_can_exceed_the_position_cap`
asserted that the breach *happened*, and documented it as a limitation. That test
was correct while the defect was open and is wrong now that it is closed, so it was
replaced by `TestPositionCapEnforcedAtFill`, which asserts the cap holds. Removing
it is not weakening the suite: it is the same coverage pointed at the fixed
behaviour, plus the ten new cases around it. No other test was deleted or relaxed.

---

## Verification

| Check | Result |
|---|---|
| Tests before this round | 1931 passed, 1 skipped, 1 deselected |
| Tests after this round | 1966 passed, 1 skipped, 1 deselected, 100 subtests passed |
| Lint (`ruff check .`) | Clean |
| Example run (`examples/simulated_run.py`) | Unchanged output: 103,569.69 equity, 52.25 fees |
| Multi-agent graph (`test_graph_end_to_end.py`) | 15 passed |
| Layering and prompt integrity | 14 passed |
| Long-only guard | A SELL with no position is rejected (`INSUFFICIENT_POSITION`) |
| Live trading path | None: `ExecutionMode` has only `BACKTEST` and `PAPER` |
| Secrets | None committed; no key in any new file |

The example run produces byte-identical numbers to before, which is the point: the
re-check changes nothing when no cap is breached, and only bites when one would be.

---

## Remaining limitations

These are real and are stated rather than hidden.

1. **The re-check uses the projected fill price, not the settled one.** For a MARKET
   order that is the same price `submit` uses, so the two agree. A future order type
   that fills at a price not derivable from the bar — a partial fill at an unknown
   depth, say — would need the re-check to move after the fill, which the ledger
   would then have to reconcile.
2. **The re-check runs per order, in arrival order.** Two orders for the same symbol
   that arrive in the same session are capped in sequence, so the second sees the
   first. That is correct, but it means the outcome depends on arrival order, which
   is deterministic here and would need a rule if order arrival ever became
   concurrent.
3. **The exposure cap is checked at the mark, and the mark moves.** The re-check
   values the rest of the book at its last mark. Between sessions the marks are
   stale by construction, which is honest but means the cap is enforced against the
   book as last valued, not as it would be valued at the fill instant.
4. **Sells are not re-checked at all.** Correct for a long-only book, since a sell
   only reduces exposure. A future short-capable mode would need its own re-check.
5. **Round 1's unfixed items stand** as recorded in the audit: no intra-bar stop, no
   per-series FRED release calendar, no snapshot-capable news/social provider.

---

## Verdict

**PASS WITH FINDINGS.**

The defect this review was opened to fix is fixed, in both execution paths, with the
cap enforced against the value of the position the fill actually creates. It is
covered by 36 new tests that assert the resulting position value, not the requested
order size, and the accounting identity holds after every capped fill.

The verdict is "with findings" rather than a clean pass because the limitations
above are real. None of them is a cap that can be breached today — each is a
boundary the current design does not cross, and a place a future change would have
to think. They are recorded here and in the code so that a later change meets them
deliberately.
