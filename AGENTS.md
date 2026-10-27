# AGENTS.md

Repository-specific notes for future sessions.

## Commands

```bash
pip install -e .          # install
pytest -q                 # full suite; offline, no API key needed
python -m ruff check .    # lint
python -m pytest tests/test_backtest_engine.py -q   # one module
python examples/simulated_run.py                    # end-to-end simulated run
```

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
