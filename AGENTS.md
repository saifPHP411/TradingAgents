# AGENTS.md

Repository-specific notes for future sessions.

## Commands

```bash
pip install -e .          # install
pytest -q                 # full suite; offline, no API key needed
python -m ruff check .    # lint
python -m pytest tests/test_backtest_engine.py -q   # one module
python examples/simulated_run.py                    # end-to-end simulated run
python -m tradingagents.baseline.run --out docs/baseline/official_baseline.json  # official baseline
```

`python -m tradingagents.baseline.fetch_data` is the only network entry point in
the baseline package, and it is run once by a human to freeze
`tradingagents/baseline/data/`. Never call it from a test or from a run. The
download goes through `dataflows/vendors/yahoo/ohlcv.load_ohlcv_range`, because
`tests/test_layering.py` allows only the data layer to import `yfinance`.

`pytest` refuses network access via an autouse fixture in `tests/conftest.py`, so
tests are hermetic. Do not add a test that needs a live vendor or an API key
without marking it to skip when unavailable.

`langchain_aws` is not installed in this environment, so
`tests/test_bedrock_provider.py` skips. That is expected, not a failure.

## Layout

The analysis workflow (`graph/`, `agents/`, `llm_clients/`, `dataflows/`) is one
system. The simulated trading layer (`signals/`, `portfolio/`, `risk/`,
`execution/`, `backtesting/`, `performance/`, `telemetry/`, `trading_report/`) is
another. They meet at `signals/contract.py:TradingSignal`. Do not move execution
or accounting logic into the agent layer, and do not let a prompt decide a
constraint that code can enforce.

Two similar-sounding modules are deliberately distinct:

- `tradingagents/backtest.py` — the decision-quality grid evaluator over the
  agent graph. Its docstring says it must not grow a portfolio.
- `tradingagents/backtesting/` — the portfolio, event-loop simulator.

- `tradingagents/reporting.py` — the agent-written report tree.
- `tradingagents/trading_report/` — reports for a simulated run (HTML and JSON).

Do not merge either pair; they answer different questions. A package shadowing a
module of the same name has already broken an import once here.

`tradingagents/baseline/` is the reproducible *reference* baseline: one
deterministic strategy, a frozen committed dataset, a versioned protocol, and an
independent cross-check. It is not a baseline of the LLM strategy, which is
non-deterministic and needs credentials; do not describe it as one. The frozen
data under `tradingagents/baseline/data/` is the source of truth — a re-fetch is
a new baseline version, not an update.

`tradingagents/llm_evaluation/` records and replays LLM decisions offline.
`point_in_time.py` in that package is the readiness audit: its dependency matrix
is derived from the graph spec, the router's vendor map and the PIT registry, so
do not hand-edit the matrix — change `dataflows/pit.py` and the tests will follow.
A source's classification is one-way from its registry level; do not claim a
`WINDOW_FILTERED` source is `PIT_VERIFIED` because it returned a historical date.

## Conventions

- Never add a live trading path. `ExecutionMode` has only `BACKTEST` and `PAPER`,
  and `resolve_mode` raises on live-sounding names. Tests assert this.
- Undefined metrics are `None`, not `0.0`. A flat run has no Sharpe ratio.
- Every cash or position change goes through `Ledger.apply_fill`, so the
  accounting rules stay in one place and the event log stays complete.
- Rejections carry a named reason from a closed vocabulary, never prose only.
- A point-in-time claim must cite the guard that implements it; see
  `dataflows/pit.py`. Do not call a source PIT-safe without one.
- Tests are written to catch a premise error, not to confirm the code. When a
  test fails, first check whether the test's premise is right.
- Every number that enters the ledger must be finite. NaN is the dangerous case,
  because `x <= 0` and `x < 0` are both false for it, so a NaN passes the usual
  positivity guards and then poisons cash and equity. Check `math.isfinite`
  explicitly, never rely on a comparison.
- An order may fill at most once. `ExecutionSimulator.submit` refuses an order
  that is already terminal, because the ledger cannot tell a re-submission from a
  second genuine order.
- A strategy's `context.feed` is a view bounded at the session, not the whole
  feed. Do not hand a strategy the raw feed; that re-opens look-ahead through
  `feed.bar(symbol, future_date)`.

## Known limitation

An equity-relative position or exposure cap is re-checked at the fill price
(`RiskEngine.review_fill`), so an up-gap cannot leave a position past its cap: the
fill is reduced to the largest size that holds, and refused only when no positive
size remains. Cash and long-only constraints hold through a gap, and a buy whose
all-in cost (notional plus fee) exceeds the cash is reduced to the affordable
notional rather than refused. The residual limitation is narrower: the re-check
uses the *projected* fill price, which for a MARKET order is the price the
simulator will use, so the two agree; an order type that filled at a price not
derivable from the bar would need the re-check to reconcile after the fill. See
`docs/PR1_INDEPENDENT_REVIEW.md`.
