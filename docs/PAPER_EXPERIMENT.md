# Prospective paper experiment

A prospective paper experiment records an LLM strategy's decisions *forward in
time* and evaluates them as later data arrives — without real money and without a
live trading path. It is the Phase 10 layer, built from pieces that already exist
and are already tested.

This document explains how to run it, what it can and cannot prove, and where its
files live. Read [PAPER_TRADING.md](PAPER_TRADING.md) first for the paper session
itself, and [EXECUTION.md](EXECUTION.md) for the fill model.

## What it is, and what it is not

It **is**:

- a durable, append-only, hash-chained journal of an experiment's history;
- a cycle that separates decision time (T), execution time (T+1's open) and
  evaluation time (T+1+lag), so no result uses a price that did not exist when
  the decision was made;
- a report that states its own sample size and every reason it is blocked.

It is **not**:

- a live trading system. There is no order routing, no broker, no credential in
  any artifact, and no `ExecutionMode.LIVE`;
- a current-market run by default. The default price source reads the committed
  historical dataset, which ends in 2024. A real vendor source (Alpha Vantage
  daily bars, Finnhub quotes) can be selected, but it is opt-in and declares its
  own provenance, and a vendor that fails stops the run rather than silently
  falling back to 2024 data;
- a profitability claim. A paper mark-to-market is not a realised trade.

## Three storage concepts, kept apart

Conflating these is how a report gets mistaken for a recovery file:

| Concept | File | Purpose |
|---|---|---|
| Journal | `<root>/journal.json` | Append-only, hash-chained history. Removing, reordering or editing an entry breaks the chain. |
| Session state | `<root>/session_state.json` | The recovery format: the paper book (ledger, resting orders, processed signals), every decision's lifecycle, and the processed-session set. |
| Report | anywhere you ask | A read-only view. Regenerating it changes nothing. |
| Snapshots | `<root>/snapshots/*.json` | The reproducible price window each decision is anchored to. |

The **journal** is not the recovery file, and the **report** is not the journal. A
reporting snapshot alone cannot restore a book; the session state can.

## Commands

All commands are paper-only. The default price source is the frozen dataset.

```bash
# Start an experiment directory (writes the journal's creation entry).
python -m tradingagents.paper_experiment start --root runs/exp1

# Process the next N sessions, resuming if the directory exists.
python -m tradingagents.paper_experiment run --root runs/exp1 --sessions 20

# Write the report.
python -m tradingagents.paper_experiment report --root runs/exp1 --out runs/exp1/report.json

# Check the journal's hash chain.
python -m tradingagents.paper_experiment verify --root runs/exp1

# List the selectable price sources.
python -m tradingagents.paper_experiment sources --root runs/exp1
```

### Choosing a price source

`--source` picks the source for a **new** run; `--fallback` names a source to use
if the primary cannot be built. The fallback is explicit and logged; without it, a
source failure stops the run rather than silently changing the data.

```bash
# Real daily bars from Alpha Vantage (free plan; set ALPHA_VANTAGE_API_KEY).
python -m tradingagents.paper_experiment start --root runs/av \
    --source alpha_vantage --symbol IBM

# The same, with an explicit fallback to the local dataset if the vendor fails.
python -m tradingagents.paper_experiment start --root runs/av \
    --source alpha_vantage --fallback frozen --symbol IBM
```

A vendor source is fetched **once**, when the experiment is created. `report`,
`verify` and a resumed `run` read the saved state and never refetch; the CLI
refuses to resume a vendor run on the frozen dataset, so a vendor failure cannot
be hidden behind 2024 data. See [MARKET_DATA_PROVIDERS.md](MARKET_DATA_PROVIDERS.md)
for what each provider can and cannot provide on its free plan.

`--stub` uses a non-genuine stub decision source, for an offline demonstration.
Its decisions are labelled `synthetic_fixture` and are **never** counted as
genuine inferences. Without `--stub`, the experiment uses the real provider seam:
one bounded request per decision session, with `max_retries=0`.

## Verifying price data

A decision is only recorded if its price window is reproducible and anchored to a
named snapshot whose hash matches the record's. Verify the committed SPY snapshot
directly, offline:

```bash
python -m tradingagents.llm_evaluation.snapshot --verify docs/baseline/SPY_price_snapshot.json
```

The experiment also refuses a decision whose window does not end at the decision
date, so a record can never be built from bars the model could not have seen.

## The cycle, stated once

1. **Decide at T.** The window is every bar up to and including T. The model (or
   a blocked marker) produces a decision dated T. A decision is only taken on a
   cadence session (every `decision_frequency_sessions`), but settlement runs on
   every session.
2. **Execute at T+1's open.** A decision from T's close cannot fill at T's close.
   The signal is re-dated to the execution session before it reaches the paper
   session, so nothing downstream can read the fill as happening at T.
3. **Evaluate at T+1+lag.** The outcome is measured once enough later bars exist.

Three timestamps are recorded on every decision — `decision_session`,
`execution_session`, `evaluation_session` — so the timing is auditable rather than
implied.

A resting GTC limit left by a decision is worked against each later session's bar
before the next decision is taken, so a limit the earlier session did not reach can
fill here. Setting `max_volume_participation` (in the config, or via
`from_protocol`) models a thin book: an order larger than the bar can absorb fills
partially across sessions, and the remaining size rests until it is worked again.

## When the key is missing, or the provider fails

If no credential is configured, or the provider call fails, or the response has
no readable rating, the session records a **blocked** decision with the reason.
It never substitutes a fabricated decision, and it never counts a blocked
decision as a result. The report lists blocked decisions separately and names the
missing prerequisite.

The experiment does not prompt for a key and does not read one from anywhere but
the environment. A blocked run exits cleanly.

## Stopping without losing state

State is written atomically after every session. Stop the process at any point;
`run` resumes from the last processed session and does not reprocess one already
in `processed_sessions`. The journal is idempotent on `event_id`, so a re-run
cannot duplicate an event.

## Distinguishing a fixture from a real inference

- A record's `source` is `llm_inference` only when a real provider call produced
  the decision. The label is derived from what the client actually did, not from
  the caller's word.
- A stub or test client yields `synthetic_fixture`, and the report's
  `genuine_inferences` count excludes it.
- The report's readiness status is `BLOCKED` until at least one genuine inference
  exists. `READY FOR LIVE PAPER` needs a genuine inference *and* a current-market
  source that can serve a session window; a historical source never qualifies, and
  a quote-only source is current but cannot drive a multi-session run.

## Where files live and how they are protected

```
<root>/
  journal.json          append-only, hash-chained, atomic writes
  session_state.json    versioned recovery state, atomic writes
  snapshots/            one verifiable snapshot per decision
```

Every payload is scanned for credential-shaped values before it is written; a hit
raises rather than writing the file. Writes go to a temporary file and are
renamed, so a crash leaves either the old file or the new one, never a truncated
one. `verify` recomputes the chain and reports `JOURNAL INVALID` on tampering.

## Readiness, honestly

Run the report and read `readiness`:

- `BLOCKED` — no genuine inference yet, or the source is not the current market.
- `OFFLINE PAPER (reproducible, not live)` — genuine inferences exist, but the
  source is historical. This is the expected state for the frozen dataset and for
  a vendor's daily bars, which are historical.
- `READY FOR LIVE PAPER` — genuine inferences on a current-market source that can
  serve a session window. A quote-only source (Finnhub) is current but cannot, so
  it does not reach this state on its own.

The report's `limitations` list and `readiness.blockers` are the authoritative
statement of what the experiment does not prove. A paper result is not evidence
of profitability, and a small sample is reported as `N/A`, not as a number.
