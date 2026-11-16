# Bybit Demo paper-trade feedback loop

## Purpose

The paper-learning endpoint provides a read-only feedback loop over Bybit Demo Spot
executions. It is deliberately separate from the decision source and order path.

- Worker decisions are attributed to their submitted entry order in
  `execution_state/bybit_demo/worker_state.json`.
- The dashboard reads Bybit V5 execution rows with
  `GET /v5/execution/list` only.
- Executions are paired FIFO by symbol, appropriate to this long-only Spot worker.
- Actual `execFee` values are deducted only when both sides report fees in the
  symbol's quote currency.
- A truncated history, missing fee data, or a sample below 20 completed trades
  prevents a net-performance verdict.
- The report includes per-symbol outcomes and evidence-based observations.

## Endpoint

Authenticated dashboard GET route:

```text
GET /api/paper/learning
```

The route creates a stateless Bybit Demo reader, confirms the account, and reads
execution history. It does not construct or start the worker, submit/amend/cancel
orders, or change strategy/risk settings.

## What "learning" means here

This first iteration is an observational learning loop: collect decisions and
venue fills, calculate realized outcomes after verified fees, and surface review
suggestions. It is **not** an online model trainer and does not change the TJR
strategy. The active crypto decision source remains deterministic TJR.

Automatic parameter changes are intentionally disabled. Any future candidate
must be tested offline on chronological walk-forward data with realistic costs
and an untouched final holdout, then explicitly reviewed before promotion.

## Known limits

- FIFO matching assumes long-only Spot inventory and pairs buys with later sells.
- Bybit's execution-history API is paginated; if the configured page limit is
  reached, the report marks history incomplete and withholds performance metrics.
- The worker's existing state only contains orders it knows about. Manual orders
  or fills not attributed to worker order IDs may appear as unmatched sells or
  remain outside this report.
- Fees charged in a non-quote asset are not converted using a guessed FX rate;
  net PnL is withheld instead.
- This is an execution/outcome diagnostic, not proof of a profitable strategy.
