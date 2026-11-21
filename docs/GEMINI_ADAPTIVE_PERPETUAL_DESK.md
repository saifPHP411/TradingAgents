# Gemini adaptive desk for Bybit Demo

The desk starts in shadow mode. It evaluates the existing EMA multi-timeframe
scalper for USDT linear perpetuals, tracks hypothetical outcomes and sends
qualified candidate snapshots to Gemini when the API key is configured. The
same signal implementation is used in shadow and when the Demo worker is armed.

## Decision and learning

| Role | Responsibility |
| --- | --- |
| Sniper | The deterministic EMA-reclaim rule uses a forming 1-minute Mark Price and completed 5- and 15-minute trend filters. |
| Analyst | Gemini receives the candidate plus a summary of matured outcomes. Its structured response can approve or abstain for this symbol, direction and candle. |
| Professor | Code checks the bracket, stress costs, the learned performance gate, market freshness and reference-price drift. The existing Demo engine enforces venue constraints, exposure, protective orders and risk limits. |

Gemini runs asynchronously with an 8-second HTTP timeout. The ledger budgets at
most 24 requests per UTC day by default and records the exact features and
response per request. Each entry requires a later fresh quote with the same
signal and direction before the minute expires. Missing keys, errors, invalid
responses and vetoes prevent new automated entries. Failed inference leaves
existing position management independent of Gemini.

Shadow observations use later sampled Mark Prices. A candidate that cannot be
filled at a subsequent eligible quote, or an observation with a gap after
entry, is excluded from learning and counted separately. The simulation models
one opportunity per symbol at a time, deducts 15 bps per side as a stress
assumption for fees and impact, and uses observed prices at adverse exits.
These observations do not measure actual spread, funding, order fills, margin
availability, portfolio equity or strategy performance on Bybit Demo. A
75% observed win rate on this sample provides a gate for a Demo trial and
cannot establish an expected live win rate.

SQLite retains the observation history and updates the learning context from
the latest 100 settled shadows that were available before each decision.
Each eligible opportunity updates the baseline regardless of the Gemini vote.
The Demo gate requires at least 30 eligible closes, at least 75% net winners
and a positive lower mean-return estimate. A direction with sufficiently
negative historical evidence may still be vetoed. The estimate is a screening
heuristic and can overstate certainty when trades are correlated. This is
online learning from observations. The Gemini model itself is used for
inference through Google's API.

## Railway setup

Use an isolated service pointing at the feature branch, with a single replica
and a persistent volume at `/data`. Use the repository Dockerfile, configure
`/healthz` as the health check, and set this start command on the service:

```sh
/bin/sh -c "exec python -m tradingagents.paper_experiment.perpetual_demo.server --root /data --host 0.0.0.0 --port $PORT --gemini-adaptive"
```

Run the service with `RAILWAY_RUN_UID=0` when the mounted volume requires it.
From a Railway service's Variables tab:

| Variable | Purpose |
| --- | --- |
| `TRADINGAGENTS_DASHBOARD_TOKEN` | A strong password-equivalent token for the mobile dashboard. Required for the public bind. |
| `GEMINI_API_KEY` | Your Gemini API key, added directly to the service variables. |
| `TRADINGAGENTS_GEMINI_MODEL` | Optional. Defaults to `gemini-3.8-flash`. |
| `TRADINGAGENTS_GEMINI_DAILY_LIMIT` | Optional call budget, 24 by default. |
| `TRADINGAGENTS_PERPETUAL_DEMO_SYMBOLS` | Optional comma- or space-separated contract list. |
| `BYBIT_LINEAR_DEMO_API_KEY`, `BYBIT_LINEAR_DEMO_API_SECRET` | Dedicated Demo account credentials. Shadow collection uses public market data without these. |
| `TRADINGAGENTS_GEMINI_DEMO_ENABLED` | Set to `true` to allow authenticated Demo activation after the learning gate qualifies. Default is off. |

The desk polls public market data with a 15-second pause between scans.
The repository's separate Donchian WebSocket research tool has its own
one-second observation loop and evidence. This Gemini desk uses the EMA worker.

Use **Start Scalping Auto** on the authenticated dashboard to arm Demo execution
once the gate has passed. **Stop strategy** pauses Demo entries and returns to
shadow collection. Existing orders retain exchange-side protection. After a
process restart the desk returns to shadow and requires a fresh operator
activation. Use an account dedicated to this desk.

`/healthz` checks that the service is listening. The authenticated
`/api/learning` shows the latest scan, sample counts, gate, request budget
and recent Gemini votes. A key configuration indicator alone does not verify
Google access. A recorded Gemini verdict verifies a completed call.

References: [Gemini content generation](https://ai.google.dev/api/generate-content)
and [Bybit Demo API](https://bybit-exchange.github.io/docs/v5/demo).
