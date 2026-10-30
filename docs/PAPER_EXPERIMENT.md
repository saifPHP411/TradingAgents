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
`verify` and a resumed `run` read the saved state and never refetch; a
frame-backed vendor source stores its bars in `session_state.json`, so a resume
rebuilds the same source locally. The CLI refuses to resume a source that carries
no persistable bars (a quote-only source) rather than substituting the frozen
dataset, so a vendor failure cannot be hidden behind 2024 data. See
[MARKET_DATA_PROVIDERS.md](MARKET_DATA_PROVIDERS.md)
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

1. **Decide at T.** The window is every bar up to and including T. The model is
   shown that window — the trailing `WINDOW_BARS` (30) rows, oldest first, as
   `date,open,high,low,close,volume` — and produces a decision dated T. A session
   with no reproducible window is blocked *before* any request: the model is
   never asked to decide with no data. A decision is only taken on a cadence
   session (every `decision_frequency_sessions`), but settlement runs on every
   session.
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

## Reading the counters: settled is not filled

The report's `counts` block and its `orders` funnel answer two different
questions, and conflating them is how "`executed: 1`, `filled orders: 0`" reads
as a lost order when nothing is lost.

- `counts.executed` counts **decisions carried to their execution session**. A
  HOLD is one: the decision was taken, the session settled it, and no order was
  ever going to exist. So `executed` is not a count of orders and must not be
  read as `filled`.
- The `orders` funnel counts what the orders themselves did, once each:
  `settled`, `submitted`, `filled`, `partially_filled`, `rejected`, `cancelled`,
  `resting`, and `without_order` (with `without_order_reasons` tallying why).

The identity that always holds:

```
settled == submitted + without_order
submitted == filled + partially_filled + rejected + cancelled + resting
```

A run whose every decision is a HOLD reports `submitted: 0`, `filled: 0` and
`without_order: 20` with `without_order_reasons: {"no order: HOLD": 20}` — the
decisions were all settled, and none was ever an order. A tradeable decision
does produce a simulated fill; the funnel shows `submitted` and `filled` rise
together. To get fills, the model must return a tradeable rating (Buy,
Overweight, Underweight, Sell) rather than Hold, and risk must approve the size.

### Why a run returned Hold on every decision

A run that returns Hold on every decision is only informative if the model was
given something to decide *from*. The first real-inference prompt was not: it
asked for a rating while telling the model "given no further data", so `Hold` was
the only safe answer and 20 HOLD out of 20 decisions followed by construction.
The seam's signature was `(symbol, session)`, so the window the experiment
already built could not reach the prompt; it was used only afterwards, to hash
and anchor the record. The record therefore attested to an input the model never
saw.

The fix passes the session's point-in-time window into the seam and renders it
into the request (`PAPER_INFERENCE_PROMPT_VERSION = "paper-inference-2"`). Two
invariants now hold and are tested in `tests/test_paper_inference_prompt.py`:

- the request contains only bars up to and including the decision session, capped
  at the last `WINDOW_BARS` (30) bars, so it can never carry a future bar;
- a window with a non-finite value, or no window at all, blocks the decision
  *before* any request is made, so no paid call is spent and no decision is taken
  blind.

A windowless session is blocked with `no reproducible price window …`, not
decided. If a future run still returns Hold on every decision, the model was
shown real data and genuinely chose Hold — the counters can then be read as a
result.

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

A frame-backed source (the frozen dataset, a static fixture, or a vendor daily
source) persists its validated bars and provenance in `session_state.json`, so a
resume rebuilds the **same** source without a second fetch. On a quota-limited
vendor (Alpha Vantage's free tier allows 25 calls a day) that matters: a restart
must not spend a call, and it must not silently continue on different data. A
source that cannot be persisted — Finnhub's quote source, which carries no bar
history — is refused on resume rather than swapped for the frozen dataset.

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

## Currency is re-checked on restore, not trusted from the file

`is_current_market` is a *claim about the clock*, so it is re-evaluated when a
state file is read, not carried over as a fact. A vendor source records the
freshness rule it applied (`max_stale_days`), and on restore the flag is recomputed
against the current date from the newest bar. A snapshot persisted while its bar
was fresh therefore stops claiming the current market once the bar is stale, so a
report cannot describe months-old data as current. A source with no freshness rule
(the frozen dataset, a test fixture) keeps its recorded flag: its currency is a
property of the source, not of the day it is read.

## Alpaca paper broker (separate, opt-in, paper-only)

`tradingagents/paper_experiment/broker/` is a **separate** path for Alpaca Paper
Trading. It is not the internal simulator and it is not a live path.

- It has its **own** account state and its own recovery file; it never imports or
  writes the internal ledger or the simulator, so a broker fill cannot be
  double-counted as an internal fill.
- The endpoint is hard-validated to Alpaca's paper host. Every live host, and any
  unapproved host, is refused. Credentials come from
  `ALPACA_PAPER_API_KEY`/`ALPACA_PAPER_API_SECRET` only; live variable names are
  refused, not used as a fallback.
- Constructing the broker performs **no** request. `connect()` is a read-only
  account check. An order is placed only with an explicit `allow_orders=True`,
  after a confirmed account read.
- Each order carries a stable `client_order_id`. Intent is persisted before the
  request, a lost response is reconciled by that id rather than re-submitted, and
  a fill is applied at most once (a persisted `applied_fills` set).

```bash
# Report whether the paper broker is configured, without printing a secret.
python -m tradingagents.paper_experiment broker-check
# The same command under an explicit Alpaca name.
python -m tradingagents.paper_experiment alpaca-diagnose
# Also read the account (read-only; still places no order).
python -m tradingagents.paper_experiment alpaca-diagnose --connect
```

`--connect` performs one read-only `GET /v2/account` and reports the account id,
status, currency, cash, equity, buying power and any active restrictions
(`TRADING_BLOCKED`, `ACCOUNT_BLOCKED`, `TRANSFERS_BLOCKED`,
`TRADE_SUSPENDED_BY_USER`, `SHORTING_DISABLED`). A successful read proves the
paper environment, because the paper host is the only URL the adapter can
request. It fails safe — non-zero exit, clear message, no live fallback — if a
credential is missing, authentication fails, the request times out, or the
response is not the expected object.

There is no CLI flag that submits an order. `ExecutionMode` is unchanged
(`BACKTEST` and `PAPER` only); this adapter does not add a live path. A gated
integration test (`tests/test_live_providers_integration.py::TestAlpacaPaperLive`)
runs the real read-only check when both paper credentials are present, and skips
otherwise, so the default suite needs no secret.

## The paper trading worker (explicit opt-in, paper-only)

`tradingagents/paper_experiment/broker/trading.py` turns the read-only adapter
above into a *deliberate* paper trading loop. It is still not a live path, and it
is off unless an operator opts in.

- **One worker per state file.** `start()` runs preflight — broker connection,
  account status, positive buying power, an available market-data source, and a
  valid risk configuration — and places **no** order. A failed preflight leaves
  the worker `ERROR`, never `RUNNING`. On success `start()` launches one
  background loop; a second `start()` while it is live raises rather than starting
  a second loop.
- **START means "run continuously".** The loop waits `poll_interval_seconds`
  *first*, then calls `run_once()` once per interval until `stop()`. Waiting before
  the first cycle is deliberate: it gives an operator a window to press STOP
  before any order, and it makes "start places no order" true for a whole interval.
- **A decision is one order, at most.** `run_once()` processes at most one
  strategy decision and places at most one order. It refuses unless `RUNNING`.
- **Local risk checks run before the broker.** Buying power, a per-order notional
  cap, a daily order count, an open-position count, an optional symbol allow-list,
  and long-only (no short) are enforced by the worker and the broker's own
  `_check_risk` before any request leaves the process. The per-order cap defaults
  to `25.0`, sized to a ~50 USD paper account, so it is a real second layer rather
  than a nominal one; the target `order_notional` must not exceed it.
- **The daily order count is durable.** It is persisted in the broker's state
  file with the UTC day it belongs to, so a restart on the same day keeps the
  count and a new day starts fresh without an explicit reset. Only an order the
  broker accepted is counted: a local refusal or a broker rejection does not
  count, and a partial fill of one order is still one order.
- **Idempotent by construction.** Each decision maps to a deterministic
  `client_order_id`. A decision already processed is skipped; a decision whose
  order the worker already knows is *reconciled* through the broker by that id
  rather than re-submitted. A lost response is reconciled, not guessed.
- **A restart comes back `STOPPED`.** The worker's state (processed decision,
  orders, positions, account, timestamps) is written to
  `<root>/worker_state.json`, versioned with `schema_version`. It is read back as
  `STOPPED`; nothing resumes `RUNNING` on its own.
- **Stop is graceful.** `stop()` signals the loop, waits for the in-flight cycle
  to finish rather than killing the thread, and closes no position and cancels no
  order. `poll_interval_seconds` is the loop's interval; the default is 60 s.
- **A cycle failure is contained.** An unexpected exception inside a cycle is
  recorded, the state goes `ERROR`, the loop stops, and the process keeps running.
  A later manual `start()` reconciles rather than re-submitting.
- **A bad price refuses the order.** A missing, `NaN`, `inf`, zero or negative
  price, or a price-source exception, is surfaced as `MarketDataError` and the
  cycle does nothing — no fabricated fill, no state corruption.

### Start and stop from the CLI

```bash
# One explicit cycle: preflight, then at most one decision. Places an order only
# if a decision is tradeable and every risk check passes.
python -m tradingagents.paper_experiment paper-trade --root runs/exp1

# Preflight and report, without processing a decision or placing an order.
python -m tradingagents.paper_experiment paper-trade --root runs/exp1 --dry-run
```

### Start and stop from the dashboard

The dashboard's **Paper trading** tab has START and STOP buttons, and shows the
account, positions, orders, last cycle and risk limits. **START now runs the
worker continuously** — it launches the loop described above; **STOP** stops it.
The control is **off by default**: a dashboard started without
`--allow-paper-trading` reports the feature as disabled and refuses every
start/stop. Both endpoints require the bearer token *and* an `X-Paper-Confirm: yes`
header, and a live endpoint cannot be configured in — the worker builds its own
paper-only broker. `GET /api/paper/status` reports the live state (`STOPPED`,
`STARTING`, `RUNNING`, `ERROR`), the loop flag, the next-cycle time and the cycles
completed. On process shutdown the dashboard stops the loop gracefully.

```bash
python -m tradingagents.paper_experiment serve --root runs/exp1 \
  --allow-paper-trading --token "$TRADINGAGENTS_DASHBOARD_TOKEN"
```

`ExecutionMode` remains `BACKTEST` and `PAPER` only. There is no live broker, no
order routing, no credential in any artifact, and no flag that reaches a live
endpoint.
