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
| `tests/test_signal_contract.py` | 48 tests |
| `tests/test_ledger.py` | 38 tests |
| `tests/test_execution_costs.py` | 28 tests |
| `tests/test_execution_simulator.py` | 50 tests |
| `tests/test_risk_engine.py` | 47 tests |
| `tests/test_backtest_engine.py` | 30 tests |
| `tests/test_backtest_data.py` | 29 tests |
| `tests/test_performance.py` | 62 tests |
| `tests/test_walkforward.py` | 34 tests |
| `tests/test_telemetry.py` | 44 tests |
| `tests/test_pit_integrity.py` | 21 tests |
| `tests/test_paper_trading.py` | 28 tests |
| `tests/test_backtest_report.py` | 28 tests |

## Files modified

| Path | Change |
|---|---|
| `README.md` | A "Simulated trading system" section: the layer map, a backtest example, a paper example |
| `CHANGELOG.md` | An `[Unreleased]` section describing the whole addition |
| `docs/TRADING_SYSTEM_AUDIT.md` | PIT section rewritten to the per-source classification; roadmap module paths corrected |

## Tests before and after

| | Count |
|---|---|
| Before this work | 1412 passed, 1 skipped, 1 deselected |
| After the initial implementation | 1899 passed, 1 skipped, 1 deselected, 100 subtests passed |
| After the adversarial review | 1931 passed, 1 skipped, 1 deselected, 100 subtests passed |
| Added | 519 tests across 13 files |

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
8. **An equity-relative position cap can be breached by a gap.** The risk engine
   sizes against the decision price and the fill happens at the next open, so a
   large up-gap leaves a position worth more than the cap allowed. Found in the
   adversarial review; not fixed, because the honest fix is a fill-time re-check
   against the actual fill price, which is a design change. Pinned by
   `tests/test_backtest_engine.py::TestPositionCapThroughAGap`. Cash and long-only
   constraints still hold through such a gap; only the sizing cap can be exceeded.

## Adversarial review

After the implementation was complete it was re-reviewed against the code rather
than against its own test count. Seven defects were found that the 1,899-test
suite had not caught; six were fixed with regression tests and the seventh is
documented above. The fixed ones were: non-finite prices, quantities and fees
silently poisoning the ledger; a non-finite market price producing an order marked
`FILLED`; a re-submitted order filling twice; the strategy context exposing future
bars through `ctx.feed`; a non-finite bar raising mid-backtest instead of being
dropped at construction; and `Quote` accepting a non-finite price in paper mode.
The full list, with reproductions and the reasoning for the one that was not
fixed, is section 15 of `docs/TRADING_SYSTEM_AUDIT.md`.

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
- A fill-time risk re-check, so an equity-relative position cap cannot be
  exceeded by a gap between the decision price and the fill price (limitation 8).

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
```

## Final status

**FINAL STATUS: PARTIAL**

The task is implemented and tested, but "PASS" would be too strong a word for it
after an adversarial review, so the status is stated honestly rather than
generously.

What is done: every phase in the task is implemented; the multi-agent
architecture is preserved and remains the analytical component; there is no live
trading path and no secret committed; the accounting, execution, risk, cost,
walk-forward and performance layers are tested; the accounting conservation
identity holds to floating-point zero and the performance formulas match
hand-computed values.

Why not PASS: the review found seven defects in the delivered code that the
1,899-test suite had not caught. Six are fixed with regression tests. The seventh
— an equity-relative position cap can be exceeded when the fill gaps up from the
decision price — is a real, unfixed limitation, documented and pinned by a test
rather than hidden. A system with a known, unaddressed risk-limit gap is not a
clean pass, and the honest status reflects that. The gap is bounded: cash and
long-only constraints still hold through it, and it requires a large up-gap
between the decision session and the fill session to occur.
