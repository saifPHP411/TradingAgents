# LLM strategy evaluation: record, replay, audit

The reference baseline (`docs/STRATEGY_BASELINE.md`) measures a deterministic SMA
rule. It does not measure the TradingAgents LLM strategy, for two reasons that are
properties of the design: the pipeline needs model credentials, and it is
non-deterministic, so a run cannot be replayed to a single number.

This document describes the layer that closes as much of that gap as can be closed
honestly: capture each decision with the data window it saw, replay the decisions
through the frozen engine, and report a result — or report exactly what is
blocking one.

Read the first section before quoting any number.

## What this is, and what it is not

It is a **decision-replay evaluation**. A `DecisionRecord` carries everything the
signal path needs to rebuild a decision without calling a model, plus the
provenance to audit it. `ReplayStrategy` feeds those records into the existing
`PortfolioBacktester`, through the same `TradingSignal` contract a live decision
uses, so a replayed decision travels the same
decision → risk → order → fill → ledger path as any other strategy.

It is **not** a live LLM run, and it does **not** measure the news, sentiment or
fundamentals agents. The frozen dataset carries daily OHLCV only. The graph is
built to read news, sentiment and fundamentals as well as prices, and those
sources are not in the dataset and are not point-in-time safe for a historical run
(`dataflows/pit.py`). A price-only run is therefore a *restricted* evaluation: it
measures the pipeline on the one input the baseline and the LLM share. That gap is
recorded in the artifact, not hidden, and it is why a price-only result is a lower
bound on what the full pipeline could do.

## Classification

**Replay path implemented and tested. Genuine LLM inference path implemented but
blocked: no model credential is available in this environment.**

No number in this document is LLM-strategy performance. This is a research
scaffold, not a production result, and not live-trading readiness.

## The blocker, stated plainly

The graph hard-wires the news, fundamentals and sentiment analysts
(`graph/setup.py`, `graph/analyst_execution.py`). Those inputs are not in the
frozen dataset and are not point-in-time safe, so a historical LLM run cannot be
reproduced from the frozen data. Two consequences:

1. A genuine evaluation needs a credential and live (or recently captured) inputs,
   not the frozen dataset.
2. A price-only historical run is possible only by replaying *recorded* decisions,
   which is what this layer does.

`build_evaluation()` reports `LLM BASELINE BLOCKED` with the exact missing
prerequisite when neither a credential nor a record set is present:

```
$ python -m tradingagents.llm_evaluation.run
LLM BASELINE BLOCKED (mode: blocked)
  - no LLM provider credential is available in this environment; set one of
    OPENAI_API_KEY, ... (or configure a local OpenAI-compatible endpoint) to run
    genuine inference
  - no captured decision record set was supplied; pass one to build_evaluation()
    to evaluate real model output via replay
```

It never invents a number to fill the gap.

## Point-in-time readiness, and why a full-graph historical run is refused

`input_scope="full"` is refused because the news, sentiment and fundamentals
analysts read sources that are not point-in-time safe for a historical run. That
is not an assumption; it is derived from the code and reported by
`graph_readiness(scope)` in `tradingagents/llm_evaluation/point_in_time.py`, which
builds the agent-by-agent dependency matrix from the graph spec, the router's
vendor map and the PIT registry in `dataflows/pit.py`.

| Scope | Ready for a historical run? | Why |
|---|---|---|
| `price_only` (market analyst) | **Yes** | Reads only `yahoo_prices` and `yahoo_indicators`, both `PIT_VERIFIED` — served as they stood at the run's date. |
| `full` (all analysts) | **No** | News/sentiment sources are window-filtered (dating is the vendor's word) and fundamentals/social sources are current-only (must be withheld). |

A `price_only` run is a **restricted** evaluation: it measures the market analyst
and the decision-to-execution pipeline on the one input the frozen dataset carries.
It does **not** measure the news, sentiment or fundamentals pipeline. The artifact
records this in `evaluation_scope` and `limitations` so a restricted result cannot
be read as a full-graph one.

The full audit — every source, its classification, the guard that implements it,
the residual leak channels, and what would unblock a full-graph pilot — is in
[`docs/LLM_POINT_IN_TIME_AUDIT.md`](LLM_POINT_IN_TIME_AUDIT.md). The
machine-readable form is the `point_in_time` block of the evaluation artifact.

## How to run a genuine pilot (when unblocked)

The scaffold is built so that a genuine pilot is a data-and-credential step, not a
code change. It is currently blocked on both.

**Step 1 — credentials.** Set one provider key, or point a local
OpenAI-compatible endpoint at the graph. No key is present in this environment.
For a documented free tier (Groq, Google Gemini, OpenRouter) or a local endpoint,
see [`LLM_PROVIDER_AUDIT.md`](LLM_PROVIDER_AUDIT.md), which records each claim
with its official source. The recognised variables are listed by
`credential_status()` (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY`, `GOOGLE_API_KEY`, …). The provider audit's
`provider_readiness` block reports which providers are usable *now*, by env-var
name and boolean only — never a value — and contacts nothing.

**Step 2 — data.** For a `price_only` pilot, the frozen dataset already suffices
and no network call is needed. For a `full` pilot, first register a snapshot source
(see the audit's "What would unblock a full-graph historical pilot"), so every
source in the matrix is `PIT_VERIFIED`. `graph_readiness("full")` must report ready
before the pilot runs.

**Step 3 — capture decisions, then replay.** Run the graph on the protocol's
cadence, record each decision as a `DecisionRecord` (the record carries the window
hash, the config hash and the source), then evaluate it offline. Note that **no
capture path exists in this release**: `DecisionRecord` is built only in tests, so
a capture step must be added before a genuine pilot can produce a record set.

```bash
python -m tradingagents.llm_evaluation.run --records records.json --out eval.json
```

The replay runs the recorded decisions through the frozen engine with no model
call (enforced by `no_llm_calls`), so the evaluation is reproducible and free. A
result is marked `genuine_inference: true` only when the record set's source is
`llm_inference`; a fixture is never presented as genuine model output.

**Step 4 — report honestly.** Compare against the frozen reference baseline
(`docs/STRATEGY_BASELINE.md`) over the same window, costs and timing. Report the
scope: a `price_only` result is a lower bound, not the full pipeline's result.

## The protocol

`tradingagents/llm_evaluation/protocol.py` fixes the evaluation *before* any result
is seen, so it cannot be tuned after the test window. It inherits the baseline's
universe, period, costs, risk limits and execution timing verbatim — by reading the
baseline's own cost and limit *objects*, not by copying their values, so the two
runs cannot drift apart on costs or limits.

| Item | Value |
|---|---|
| Inherited from baseline | universe, benchmark, period, costs, risk limits, execution timing |
| Decision cadence | one decision every 5 sessions |
| Max decisions | 120 (a hard ceiling on model calls) |
| Input scope | `price_only` (`full` is refused, not faked) |
| Prompt version | `tradingagents-default` (shipped prompts, unmodified) |
| Default position size | 20% of equity (the baseline's) |
| Protective stop | 20% (the baseline's) |
| Temperature | provider default (0.0 does not guarantee identical output) |

`config_hash()` is a stable hash of everything that shapes a decision's inputs and
costs. Every record carries it, so a replay can prove it ran under the same
configuration the model did. `input_scope="full"` raises, naming
`dataflows/pit.py`, rather than producing a fake full-pipeline result.

## Decision records

`tradingagents/llm_evaluation/records.py`. Three sources are kept strictly apart,
because conflating them is how a fixture gets reported as performance:

- `llm_inference` — a real provider/model call produced this decision.
- `operator_supplied` — a human captured decisions from a real run elsewhere.
- `synthetic_fixture` — a test fixture. **Never** strategy performance.

Each record is validated at construction:

- The action is `BUY`/`HOLD`/`SELL`, the confidence is in `[0, 1]`, every number is
  finite.
- `window_end` must equal `as_of_date`: the model may not have seen a later bar.
- The action must match the rating it came from (`action_for_rating`). A rating
  with no tradeable action (`REVIEW`) must carry `HOLD` and no size.
- A `HOLD` carries no size, stop or limit. A `MARKET` intent carries no limit.
  Only a `LIMIT` intent may be `GTC`. These mirror the production signal contract.
- `data_window_hash` and `config_hash` are 64-character SHA-256 digests, or the
  record is refused.

The rationale is kept as a bounded excerpt (`≤2000` characters), never
chain-of-thought, and it is scanned for credentials at construction time — so a
credential in the reasoning cannot reach a serialiser even if `to_dict()` is never
called.

### Credential safety

Every record, record set and artifact is scanned before it can be written. Two
checks: a credential-shaped literal (OpenAI, AWS, GitHub, Google, Slack patterns)
and the exact value of any environment variable whose *name* marks it as a
credential. A hit raises `CredentialLeakError`, naming the field and never the
secret. `build_evaluation` calls `assert_secret_free` on the whole artifact before
returning it.

## Replay

`tradingagents/llm_evaluation/replay.py`. The adapter sits where a strategy would,
so nothing in the engine changes. Three guarantees are enforced rather than
assumed:

1. **No model call.** `no_llm_calls()` replaces the LLM client factory with one
   that raises, so a replay that reached for a model fails loudly instead of making
   a paid call.
2. **No fabricated signal.** A session with no record yields `None`. A record whose
   ticker is outside the feed is still emitted, so the engine records it as an
   `UNKNOWN_SYMBOL` rejection; the adapter does not silently drop it.
3. **The window is the same window.** `window_hash` recomputes the record's
   `data_window_hash` from the session-bounded feed and refuses a mismatch.

`signal_from_record` rebuilds the signal through `signal_from_decision`, the same
function the live path uses, so a replayed signal is subject to every validation a
live one is.

## The artifact

`build_evaluation()` returns an `EvaluationOutcome` whose `artifact` is
JSON-serialisable and versioned (`artifact_schema_version`). It contains:

- `protocol` — every value that shapes the run, plus the dataset checksums.
- `point_in_time` — the agent dependency matrix, each source's classification and
  guard, and the readiness verdict for the run's scope.
- `evaluation_scope` — which analysts were evaluated and which were omitted, and
  whether the run is a restricted (price-only) one.
- `credentials` — which providers have a credential, **by variable name and
  boolean only, never a value**.
- `provider_readiness` — per-provider transport, credential requirement, env-var
  name, whether a usable key is present, whether structured output is needed, and
  any documented free tier with its official source. Derived from the client
  registries; contacts nothing.
- `reference_context` — the SMA reference and buy-and-hold SPY over the same
  windows, labelled as context, **not** LLM results.
- `llm_results` — the per-window replay result, or `null` when no record set was
  supplied.
- `provenance` — strategy id, graph version, decision count, sessions.
- `blockers` — the exact reasons a genuine result is not established.
- `genuine_inference` — `true` only when the record set's mode is
  `llm_inference`.
- `reproducibility` — which determinism claims are verified by tests and which
  require a working credential.
- `limitations` — the scope the run actually covered and what it did not measure.

Two builds of the same records are byte-identical except for `generated_at`.

## Determinism

The replay is deterministic. Tests assert that two replays of the same records
produce identical accounting (equity, fees, position, cash, PnL) and identical
semantic event outcomes, and that two builds of the same records produce identical
artifacts apart from the wall-clock stamp.

What is **not** claimed: that a genuine LLM run is reproducible. Setting
temperature to 0.0 does not guarantee identical output across calls, and the
protocol leaves temperature at the provider default rather than implying a
guarantee it cannot make. `inference_repeatability` is reported as "not tested:
requires a working credential".

## How to use it

Report the blocker (offline, no key):

```bash
python -m tradingagents.llm_evaluation.run
```

Replay captured decisions through the frozen engine:

```bash
python -m tradingagents.llm_evaluation.run --records path/to/records.json \
    --out docs/baseline/llm_evaluation.json
```

Skip the (slow) reference-context windows while iterating:

```bash
python -m tradingagents.llm_evaluation.run --no-reference-context --records records.json
```

From Python:

```python
from tradingagents.llm_evaluation import build_evaluation, DecisionRecordSet

record_set = DecisionRecordSet.load("records.json")
outcome = build_evaluation(record_set=record_set)
print(outcome.status)          # established_with_findings, or blocked
print(outcome.artifact["llm_results"])
```

## What the numbers may and may not be used for

May be used for:

- Auditing that a captured decision replays to the same result every time.
- Confirming the decision → risk → execution → ledger pipeline behaves correctly
  on real model output.
- Comparing a future captured run against the reference baseline on the same
  protocol and data.

May **not** be used for:

- Any claim about the LLM strategy's real performance from a fixture or
  price-only replay.
- Any claim that the news, sentiment or fundamentals agents were evaluated. They
  were not.
- Any claim of profitability, production readiness, or live-trading readiness.

## Known limitations

- **Genuine inference is blocked.** No credential in this environment, and the
  full pipeline needs inputs the frozen dataset does not carry. See
  [`docs/LLM_POINT_IN_TIME_AUDIT.md`](LLM_POINT_IN_TIME_AUDIT.md) for the
  source-by-source audit and the named blockers, and
  [`docs/LLM_PROVIDER_AUDIT.md`](LLM_PROVIDER_AUDIT.md) for the provider side
  (documented free tiers, exact env vars, and the missing capture path).
- **No capture path.** Nothing constructs a `DecisionRecord` from a real graph
  run, so a credential alone cannot produce a record set to evaluate.
- **Price-only by construction.** The restricted evaluation measures one input.
  `graph_readiness("full")` reports the graph not ready for a historical run, with
  named blockers, until a snapshot source is registered.
- **One decision per session.** The production graph makes one decision per ticker
  per date, so the honest mapping is one decision per session; more than one in a
  session raises rather than inventing a priority.
- **Reference context is slow.** The SMA and SPY windows run the full engine over
  the frozen dataset and take about a minute; `--no-reference-context` skips them.
- **Non-determinism of the model itself is not addressed.** Only the replay is
  deterministic.

## No live trading

This layer adds no live trading path. Replay travels the same `BACKTEST`/`PAPER`
engine the audited stack uses. `ExecutionMode` still has only `BACKTEST` and
`PAPER`, and `resolve_mode` still raises on live-sounding names.
