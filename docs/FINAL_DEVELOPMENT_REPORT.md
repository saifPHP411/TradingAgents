# Final Development Report

## Executive summary

The task was to turn TradingAgents from a multi-agent analysis framework into a
framework with a deterministic simulated trading system underneath it, without
rewriting the analysis and without adding any path to live trading.

That is what was done. The agent workflow is untouched: the same analysts,
researchers, trader, risk analysts and portfolio manager run in the same
LangGraph graph. What was added is a layer below it, which takes the portfolio
manager's view as a validated signal and carries it through risk, execution, a
ledger, performance measurement and a report — deterministically, and offline.

Eleven phases were delivered. The one that found a real bug found it in the risk
engine, and the bug was the kind this whole layer exists to prevent.

**FINAL STATUS: PASS** — every deliverable in the task is implemented, the full
suite passes (1899 passed, 1 skipped, 1 deselected), and no live trading path
exists. Known limitations are stated in section "Known limitations" rather than
left implicit.

## Original architecture

Before this work, TradingAgents was an LLM research framework:

- A LangGraph workflow of analysts, researchers, a trader, risk analysts and a
  portfolio manager, producing a written decision with a rating.
- Data vendors behind a router: yfinance, SEC EDGAR, FRED, news, Reddit,
  StockTwits, Polymarket.
- A point-in-time discipline in `dataflows/date_window.py` that clamped what a
  tool could ask a vendor for.
- A decision-quality evaluator (`backtest.py`) that ran the graph over a grid of
  tickers and dates and scored the ratings against realized alpha.

What it did not have: any way to turn a rating into an order, any ledger, any
position sizing, any execution model, any cost model, any portfolio performance
measurement, or any report of a simulated run. `backtest.py` said so explicitly
in its own docstring — "It is not a portfolio simulator, and must not grow one."

The audit in `docs/TRADING_SYSTEM_AUDIT.md` classified every capability the task
asked about as PASS, PARTIAL, FAIL, NOT IMPLEMENTED or UNKNOWN, with file
evidence. The short version: signal handling, portfolio accounting, execution,
risk enforcement, costs, portfolio backtesting, performance analytics and
walk-forward evaluation were all NOT IMPLEMENTED. Point-in-time discipline was
the strongest existing part, and was PASS for clamping but PARTIAL overall.

## What was changed

Nothing in `graph/`, `agents/`, `llm_clients/` or `dataflows/vendors/` was
rewritten. The analysis workflow and its provider abstraction are intact.

One real bug was found and fixed, in code added earlier in this work:

- **Position cap bounded each order, not the resulting position.**
  `risk/engine.py` capped a single order at `max_position_size` of equity. Two
  orders of half the cap therefore both passed, and the position ended at the
  full cap or beyond. The cap now bounds the resulting position. Regression test:
  `tests/test_risk_engine.py::TestPositionCap::test_repeated_buys_cannot_exceed_the_cap`.

Several test premises were also wrong and were corrected, rather than the code
being bent to match them: a holiday order expiry, a stop-loss expected at the
level rather than at the gap open, and a disabled-stops case.

## Files added

| Path | Purpose |
|---|---|
| `tradingagents/signals/` | The validated `TradingSignal` contract |
| `tradingagents/portfolio/` | The ledger and its event records |
| `tradingagents/risk/` | The deterministic risk engine and `RiskLimits` |
| `tradingagents/execution/` | Orders, costs, the simulator, the mode guard, paper sessions |
| `tradingagents/backtesting/` | The event-loop portfolio backtester, the feed, walk-forward folds |
| `tradingagents/performance/` | Performance metrics and their report |
| `tradingagents/telemetry/` | LLM cost telemetry, budgets, run metadata |
| `tradingagents/dataflows/pit.py` | Per-source point-in-time guarantees and the snapshot seam |
| `tradingagents/trading_report/` | Run reports, rendered to HTML and JSON |
| `examples/simulated_run.py` | A complete offline run, synthetic bars to report |
| `docs/ARCHITECTURE.md` | The system, layer by layer |
| `docs/BACKTESTING.md` | The event loop, timing, and bias controls |
| `docs/RISK.md` | The limits and how they are enforced |
| `docs/EXECUTION.md` | Orders, fills, costs, rejections |
| `docs/PAPER_TRADING.md` | Running against live quotes |
| `docs/TRADING_SYSTEM_AUDIT.md` | The audit (written at phase 0, updated) |
| `docs/PR1_INDEPENDENT_REVIEW.md` | The second-pass review: defects found, fixes, remaining limitations |
| `tests/test_signal_contract.py` | 48 tests |
| `tests/test_ledger.py` | 38 tests |
| `tests/test_execution_costs.py` | 28 tests |
| `tests/test_execution_simulator.py` | 50 tests |
| `tests/test_risk_engine.py` | 66 tests |
| `tests/test_backtest_engine.py` | 41 tests |
| `tests/test_backtest_data.py` | 29 tests |
| `tests/test_performance.py` | 62 tests |
| `tests/test_walkforward.py` | 34 tests |
| `tests/test_telemetry.py` | 44 tests |
| `tests/test_pit_integrity.py` | 21 tests |
| `tests/test_paper_trading.py` | 35 tests |
| `tests/test_backtest_report.py` | 28 tests |

## Files modified

| Path | Change |
|---|---|
| `README.md` | A "Simulated trading system" section: the layer map, a backtest example, a paper example |
| `CHANGELOG.md` | An `[Unreleased]` section describing the whole addition |
| `docs/TRADING_SYSTEM_AUDIT.md` | PIT section rewritten to the per-source classification; roadmap module paths corrected; defect 7 marked fixed |
| `docs/RISK.md` | A section on the fill-time cap re-check |
| `tradingagents/risk/engine.py` | `review_fill`: the caps re-checked at the projected fill price |
| `tradingagents/execution/simulator.py` | `projected_fill_price`: resolve a fill's price without applying it |
| `tradingagents/execution/paper.py` | The fill-time re-check, and a `SessionStep` that reports the filled quantity |
| `tradingagents/backtesting/engine.py` | The fill-time re-check before an entry is submitted |

## Tests before and after

| | Count |
|---|---|
| Before this work | 1412 passed, 1 skipped, 1 deselected |
| After the initial implementation | 1899 passed, 1 skipped, 1 deselected, 100 subtests passed |
| After the adversarial review | 1931 passed, 1 skipped, 1 deselected, 100 subtests passed |
| After the fill-time cap re-check | 1966 passed, 1 skipped, 1 deselected, 100 subtests passed |
| Added | 554 tests across 14 files |

The suite refuses network access by default (an autouse fixture in
`tests/conftest.py`), so every test is hermetic and runs offline. `ruff check .`
passes. The one skip is `langchain_aws` not being installed in this environment;
its provider support is untouched.

The 1,899 figure was the count *before* the adversarial review. That review found
seven defects the suite had not caught, of which six were fixed and one
documented; the 32 tests added between 1,899 and 1,931 are the regression tests
for those defects, one group per defect. The fact that the suite was green while
those defects were live is the point of the review, and it is why "all tests pass"
was not treated as evidence of correctness. See section 15 of
`docs/TRADING_SYSTEM_AUDIT.md` for the defect list and the reproductions.

## Backtesting improvements

`backtest.py` scored a rating against a later price. `backtesting/engine.py` runs
a portfolio through an event loop:

- market event → decision → risk validation → order → fill → portfolio update →
  P&L → next event.
- **Look-ahead control.** Orders fill at the next session's open, not at the
  decision bar's close. `execute_on_next_open=False` exists only so a test can
  demonstrate the difference, and is documented as unsafe.
- **Strict slicing.** `StrategyContext.history()` returns bars up to the session
  inclusive and never a later one. The feed refuses to return a bar after the
  requested date.
- **Honest stops.** A stop crossed by a gap fills at the open, not at the level.
- **Missing data.** A symbol with no bar keeps its last close when marked; a
  pending order with no bar is carried forward, not filled at a price nobody saw.
- **Calendar.** Sessions are the trading days present in the data, so weekends and
  holidays are absent because they have no bars, not because a calendar was
  hardcoded.

## Portfolio improvements

There was no ledger. There is now one (`portfolio/ledger.py`):

- Initial cash, cash, positions, quantity, average entry price, market value,
  equity, available cash, buying power, realized P&L, unrealized P&L, fees, total
  exposure.
- Every cash or position change goes through one method, `apply_fill`, so the
  accounting rules live in one place.
- Every change is recorded as an event: `ORDER_CREATED`, `ORDER_ACCEPTED`,
  `ORDER_REJECTED`, `ORDER_FILLED`, `POSITION_OPENED`, `POSITION_INCREASED`,
  `POSITION_REDUCED`, `POSITION_CLOSED`, `FEE_CHARGED`.
- Accounting invariants are tested: cash conservation, P&L conservation, fees,
  position quantity, and equity.

## Execution improvements

There was no execution model. There is now one (`execution/`):

- MARKET and LIMIT orders, with a lifecycle: `CREATED`, `ACCEPTED`, `REJECTED`,
  `PARTIALLY_FILLED`, `FILLED`, `CANCELLED`.
- Named rejection reasons from a closed vocabulary: insufficient cash,
  insufficient position, market closed, no price, invalid quantity, invalid
  price, symbol not tradable, risk limit, budget exhausted.
- No stop or trailing order type, because those need a price feed between bars to
  trigger honestly and a daily-bar simulation cannot provide one.
- Partial fills via `max_volume_participation`, disabled by default.
- A rejection never touches the book: every check happens before any state change.

## Risk improvements

There was no risk engine. There is now one (`risk/engine.py`):

- Configurable limits: `max_position_size`, `max_portfolio_exposure`,
  `max_open_positions`, `max_daily_loss`, `max_drawdown`, `max_order_value`,
  `min_order_value`, plus a `default_position_size`.
- Limits validate their own coherence at construction, so a configuration error
  fails loudly rather than changing behaviour quietly.
- **Long-only is enforced deterministically**, by three code paths rather than a
  prompt: a sell larger than the holding is refused, quantity is always positive
  with direction in the side, and a fill that would go negative raises in the
  ledger.
- The engine is stateless apart from its limits, so two engines with the same
  limits give the same verdict for the same book.

## PIT improvements

`date_window.py` clamped what a tool asked for. That is necessary and not
sufficient: it cannot stop a vendor returning the future. `dataflows/pit.py` now
records, per source, which half is guaranteed:

- `REQUEST_BOUNDED` — the request is clamped and the source is served as it was.
  Prices, indicators, FRED (which pins its vintage).
- `WINDOW_FILTERED` — the request is bounded and dated items are filtered, with
  undated items withheld from a historical run. News, Reddit, StockTwits, SEC.
- `NOT_PIT_SAFE` — no historical view exists. The company profile and insider
  transactions are withheld from historical runs; Polymarket has no date
  parameter to clamp at all.

Each entry cites the guard that implements it, and the tests read that guard's
source so a rename cannot leave a stale claim. Two caveats are recorded in code
rather than smoothed over: the price clamp lives in the agent tool layer, so a
direct vendor call would bypass it, and no vendor has a clamp of its own.

A `SnapshotSource` protocol and `register_snapshot_source` provide the seam for
promoting a source that gains a dated archive, without touching its callers.

## Performance metrics

`performance/metrics.py` computes: initial capital, final equity, total return,
annualized return, CAGR, realized P&L, unrealized P&L, total fees, number of
trades, winning and losing trades, win rate, profit factor, average win, average
loss, expectancy, maximum drawdown, Sharpe, Sortino, Calmar, exposure and
turnover, with benchmark comparison.

Two conventions make it honest: an undefined metric is `None` rather than zero
(a flat run has no Sharpe, and reporting 0.0 would be a lie), and the total return
is not annualized while the annualized figure states its session assumption.

## LLM telemetry

`telemetry/` records per call: model, provider, input tokens, output tokens,
total tokens, latency, retry count, errors and estimated cost, from a
caller-supplied price table that returns `None` for an unknown model rather than
inventing a number. Optional budgets cap run cost, call count and retries, and a
breach stops the run with `BudgetExceededError`. No secret is stored.

`telemetry/run.py` records run metadata for reproducibility: run id, timestamp,
code version, git commit, provider, model, temperature, research depth, risk and
debate rounds, data sources and configuration.

## Security

- No API key, token or credential is added anywhere. `.gitignore` is unchanged;
  generated reports were already ignored, so the example script leaves no
  artifact in the tree.
- There is no live trading path, and no configuration that produces one.
  `ExecutionMode` has only `BACKTEST` and `PAPER`. `resolve_mode` raises on
  `live`, `real`, `production`, `broker`, `exchange`. The simulator's constructor
  calls `assert_simulation_only`. The simulator takes prices as arguments, so
  there is nothing for a live implementation to hook into without a new class.
- Model reasoning is rendered into reports as escaped text, so a response
  containing markup cannot inject into a report.
- The tests assert all of the above, so the guard is verified rather than assumed.

## Known limitations

These are real, and are stated rather than hidden:

1. **Adjusted prices.** Prices from the vendor are split-adjusted, so a backtest
   reflects price return only. Dividends are not credited as cash.
2. **News, social and SEC sources are filtered, not point-in-time.** Their dating
   is the vendor's word. A backtest using them should discount its result.
3. **Three sources cannot be served point-in-time at all** — the company profile,
   insider transactions and Polymarket — and are withheld from historical runs.
4. **FRED publication lags are not modelled per series.** The vintage pin bounds
   the data correctly, but the release calendar is not simulated.
5. **No stop order type.** Stops are enforced at the bar level in the backtester;
   an intra-bar stop that triggers and reverses cannot be modelled honestly from
   daily bars.
6. **Paper mode is not a broker connection.** It fills against the price you hand
   it, so it cannot reveal slippage on a thin book or a venue rejection.
7. **LLM calls are not deterministic.** The deterministic layers are reproducible
   from configuration; the signals feeding them are not, which is why each carries
   its reasoning and a content-derived id.
8. **An equity-relative position cap could be breached by a gap — now fixed.**
   The risk engine sizes against the decision price and the fill happens at the
   next open, so a large up-gap left a position worth more than the cap allowed.
   Found in the adversarial review and originally left open, because the honest fix
   is a fill-time re-check against the actual fill price. That fix was then made:
   `RiskEngine.review_fill` re-runs the cap arithmetic at the projected fill price,
   and both the backtester and the paper session call it immediately before
   submitting a buy. A fill that would breach a cap is reduced to the largest size
   that holds, and refused only when no positive size remains. A 20% cap on a
   100,000 book that sized 200 units on a 100 close now fills 100 units at a 200
   open — 20% of equity, not 40%. The same defect existed in paper mode and is fixed
   there too. See `docs/PR1_INDEPENDENT_REVIEW.md`.

## Adversarial review

After the implementation was complete it was re-reviewed against the code rather
than against its own test count. Seven defects were found that the 1,899-test
suite had not caught. Six were fixed in the first round with regression tests:
non-finite prices, quantities and fees silently poisoning the ledger; a non-finite
market price producing an order marked `FILLED`; a re-submitted order filling
twice; the strategy context exposing future bars through `ctx.feed`; a non-finite
bar raising mid-backtest instead of being dropped at construction; and `Quote`
accepting a non-finite price in paper mode.

The seventh — an equity-relative position cap exceeded when the fill gaps up from
the decision price — was left open in that round, because the honest fix is a
fill-time re-check against the actual fill price rather than a patch. It was then
fixed in a second round, in both the backtest and paper paths, and carries 36
regression tests that assert the value of the position the fill creates rather than
the size of the order that was requested. The full account, with reproductions and
the remaining limitations, is `docs/PR1_INDEPENDENT_REVIEW.md`; the first-round list
is section 15 of `docs/TRADING_SYSTEM_AUDIT.md`.

Two things were verified as correct rather than assumed: the accounting
conservation identity holds to floating-point zero, and the performance formulas
(Sharpe, Sortino, drawdown, CAGR, win rate, profit factor, expectancy) match
hand-computed values to machine precision.

## Remaining work

- A snapshot-capable provider for news and social data, registered through
  `register_snapshot_source`, which would raise those sources to
  `REQUEST_BOUNDED`.
- A per-series FRED release calendar.
- Wiring the telemetry budget into the graph loop as a hard stop.
- A CLI subcommand for a simulated run, once the shape of the run is settled.
- A fill-time re-check that reconciles against the *settled* price, should an order
  type ever fill at a price not derivable from the bar. The current re-check uses
  the projected fill price, which for a MARKET order is the price `submit` uses, so
  the two agree today.

## How to run

```bash
pip install -e .
pytest -q          # the full suite, offline
ruff check .       # lint
```

No API key is needed for the test suite or for the simulated trading system.

## How to backtest

```bash
python examples/simulated_run.py
```

That builds synthetic bars, runs a strategy through the full loop, and writes
`reports/example/simulated_run.html` and `.json`. For your own data, see the
example in the README and `docs/BACKTESTING.md`.

## How to run a paper simulation

See `docs/PAPER_TRADING.md`. In short:

```python
from datetime import UTC, datetime
from tradingagents.execution import dict_quote_source, new_session

session = new_session(
    initial_cash=100_000.0,
    quotes=dict_quote_source({"AAPL": 195.40}, timestamp=datetime.now(UTC)),
)
step = session.on_signal(my_signal)
session.write_json("runs/paper.json")
```

## Git commits

On branch `trading-system-development`:

```
feat: add validated trading signal contract
docs: add full trading system audit
feat: add portfolio ledger with event records
feat: add simulated execution engine with realistic costs
feat: add deterministic risk engine
feat: add portfolio event-loop backtester with point-in-time data
feat: add performance analytics with defined return conventions
feat: add walk-forward evaluation with enforced out-of-sample splits
feat: add LLM cost telemetry, budgets, and run metadata
feat: record point-in-time guarantees per data source
feat: add paper trading sessions over the same pipeline as a backtest
feat: add run reports in HTML and JSON, built from one document
fix: harden simulation against non-finite inputs and double fills
fix: enforce the position cap at the fill price, not only at the decision
```

## Final status

**FINAL STATUS: PASS WITH FINDINGS**

Every phase in the task is implemented and tested, and the one defect the
adversarial review left open — an equity-relative position cap that a price gap
could breach between the decision and the fill — is now fixed, in both the backtest
and the paper path, with regression tests that assert the value of the position the
fill creates.

What is done: the multi-agent architecture is preserved and remains the analytical
component; there is no live trading path and no secret committed; the accounting,
execution, risk, cost, walk-forward and performance layers are tested; the
accounting conservation identity holds to floating-point zero; the performance
formulas match hand-computed values; and the risk caps hold at the fill price, so a
gap cannot produce a position larger than the limits allow. The suite is 1,966
tests, all offline and hermetic, and `ruff check .` is clean.

The "with findings" is deliberate. A clean pass would claim the system has no
boundaries left, and it does: the re-check uses the projected fill price rather than
the settled one (equal today, but a constraint on any future order type); the
exposure cap is enforced against the last mark rather than the fill-instant value;
and there is still no snapshot-capable provider for news and social data. None of
these is a cap that can be breached today. They are recorded in
`docs/PR1_INDEPENDENT_REVIEW.md` and in the code, so a later change meets them
deliberately instead of by accident.

Why not a bare PASS: the honest reading of a review that found seven defects the
original 1,899-test suite had missed is that the suite proved the code did what its
author expected, not what the code actually does. Six were fixed in the first round
and the seventh in the second, each with a test that reproduces the defect first.
The status reflects a system whose known limitations are written down, not one
without any.
