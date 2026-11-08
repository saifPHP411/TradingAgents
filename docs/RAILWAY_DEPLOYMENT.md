# Railway deployment readiness

This document states what deploying the **prospective paper experiment** to
Railway requires, and what it does not. It describes a simulation-only workload:
no order routing, no credential beyond a market-data and an LLM provider key plus
the optional read-only Alpaca paper check, and no `ExecutionMode.LIVE` (which does
not exist). The one broker-facing command is a **read-only** Alpaca paper account
check that places no order.

## What kind of service this is

A **worker**, not a web service. The experiment has no HTTP surface, no port to
listen on, and no request to serve. It runs a bounded batch of sessions and exits:

```
start  → create an experiment directory (journal creation entry + state)
run    → process sessions, appending to the journal and state after each
report → write a report JSON
verify → recompute the journal hash chain
```

Because it exits, the right Railway shape is one of:

- a **cron job** — a schedule runs `run` and the process exits, or
- a **one-off / manual** service started on demand.

A permanently-running web service is the wrong shape: nothing needs to stay
resident between sessions, and a service that never exits would be restarted by
Railway's restart policy for no benefit.

> Railway cron services "are expected to execute a task, and terminate as soon as
> that task is finished" and a previous execution still running causes the next
> scheduled run to be **skipped**
> ([Cron Jobs](https://docs.railway.com/reference/cron-jobs)). The runner exits on
> its own, so it satisfies that requirement.

An **optional dashboard service** is the one web-shaped service in this repo: a
small standard-library HTTP server (`serve`) that reads the experiment directory
and can advance the simulation. It is *not* required to run the experiment, and
it is not a trading surface. See "Monitoring dashboard" below.

## Start command

Railway's start command defaults to the Dockerfile's `ENTRYPOINT`/`CMD`
([Build and Start Commands](https://docs.railway.com/builds/build-and-start-commands)).
The image's entrypoint is the `tradingagents` CLI, which is the *analysis* CLI,
not the paper runner. Override the start command for the paper service, e.g.:

```bash
python -m tradingagents.paper_experiment run --root /data/paper-experiment --sessions 20
```

Note the `--root`: the paper runner has no default directory. State must point at
the mounted volume (below), never at the container's ephemeral filesystem.

Railway states that for a Dockerfile/image service "the start command **overrides
the image's ENTRYPOINT in exec form**"
([Start command](https://docs.railway.com/deployments/start-command)), so the
override above replaces `tradingagents` rather than being appended to it. If you
need variable expansion in the command, wrap it in a shell —
`/bin/sh -c "exec python -m tradingagents.paper_experiment run ..."` — because
"commands ran in exec form do not support variable expansion".

Both forms were exercised against the built image with the volume mounted at
`/data`, running as root (the effect of `RAILWAY_RUN_UID=0`): the run processed
sessions, wrote `journal.json`, `session_state.json` and `snapshots/` to the
mount, and a second invocation resumed. The same run on the image's default
non-root user failed to write the mount with `PermissionError`, which is why
`RAILWAY_RUN_UID=0` below is not optional.

## Python version and dependencies

- `requires-python = ">=3.11"`; the `Dockerfile` builds on `python:3.13-slim`.
- No extra dependency is needed for the vendor providers: they use the standard
  library (`urllib`, `json`) plus `pandas`, which is already a core dependency.
  The Groq LLM path uses the existing OpenAI-compatible client (`requests`).
- `.dockerignore` already excludes `.git`, `**/.env*`, `results`, `build`, `dist`.

### One interpreter for the app and for a shell

The image installs the project into a virtualenv at `/opt/venv` and prepends it
to `PATH` with `ENV PATH="/opt/venv/bin:$PATH"`. That is the interpreter the
`ENTRYPOINT` and any `sh -c`/exec-form command use, so the running service and
the dashboard always see the project's dependencies.

A **login shell is different**. Debian's `/etc/profile` *overwrites* `PATH`
outright, which discards the `ENV` prepend; `python` then resolves to the base
image's interpreter, which has none of the project's dependencies. Running a
maintenance command that way fails with `ModuleNotFoundError`:

```
$ bash -lc 'python -m tradingagents.paper_experiment report --root /data/paper-experiment'
ModuleNotFoundError: No module named 'dotenv'
```

`dotenv` is *not* missing: `python-dotenv` is a declared dependency
(`pyproject.toml`) and is installed in `/opt/venv`. The fault is the shell
choosing the wrong interpreter. The `Dockerfile` closes this by writing
`/etc/profile.d/10-venv-path.sh`, which re-prepends `/opt/venv/bin`. Because
`/etc/profile` resets `PATH` *before* sourcing `/etc/profile.d/*`, the venv wins
again for login shells too, and the manual command above works unchanged.

If you prefer not to rely on that (or run an image built before the fix), call
the interpreter explicitly — this is always correct regardless of shell:

```
/opt/venv/bin/python -m tradingagents.paper_experiment report --root /data/paper-experiment
```

Both forms are covered by `tests/test_docker_interpreter.py`, which asserts the
dependency is declared, the package imports it, and the Dockerfile restores the
venv path for login shells.

## Environment variables

| Variable | Needed for | Notes |
|---|---|---|
| `GROQ_API_KEY` | genuine LLM inference | provider `groq`, model `openai/gpt-oss-20b` |
| `ALPHA_VANTAGE_API_KEY` | `--source alpha_vantage` | free daily bars; 25 calls/day |
| `FINNHUB_API_KEY` | `--source finnhub` | free current-day quote only |
| `ALPACA_PAPER_API_KEY` | the read-only Alpaca **paper** diagnostic | paper key id; **no order is placed** |
| `ALPACA_PAPER_API_SECRET` | the read-only Alpaca **paper** diagnostic | paper secret; **no order is placed** |
| `TRADINGAGENTS_DASHBOARD_TOKEN` | the optional dashboard | bearer token; **required** for any non-loopback bind |
| `TRADINGAGENTS_EXPERIMENT_ROOT` | the optional dashboard (module form) | experiment dir; lets `--root` be omitted |
| `TRADINGAGENTS_DASHBOARD_HOST` | the optional dashboard (module form) | bind interface; defaults to loopback |
| `TRADINGAGENTS_AUTO_ADVANCE_SESSIONS` | the optional dashboard | sessions per auto-advance tick; `0` (default) = monitor only |
| `TRADINGAGENTS_AUTO_ADVANCE_INTERVAL` | the optional dashboard | seconds between ticks; default `60` |

Set these as Railway **service variables** (never in the repository). The code
reads them from the environment by name only; it never prints, logs or stores a
value. If a variable is missing, a decision is recorded as **blocked** rather
than fabricated, and the run exits cleanly.

The two `ALPACA_PAPER_*` variables are used only by the read-only diagnostic
below. They are the **paper** credentials — the adapter refuses every live Alpaca
host and refuses the live variable names (`APCA_API_KEY_ID`, `ALPACA_API_KEY`, …)
rather than using them as a fallback. No value is ever placed in a URL or a log.

## Read-only Alpaca paper connection check

Once the two `ALPACA_PAPER_*` service variables are set, confirm the paper
connection from the Railway shell without placing any order:

```bash
# Presence-only check: no network call, no order.
python -m tradingagents.paper_experiment alpaca-diagnose

# Read-only account check: one GET /v2/account, no order, no state change.
python -m tradingagents.paper_experiment alpaca-diagnose --connect
```

`broker-check` is the same command under its original name. On success the
`--connect` form prints the account id, status, currency, cash, equity, buying
power and any active restrictions, and states that the paper environment is
confirmed and that no order was placed. It fails safe (non-zero exit, clear
message, no fallback) if a credential is missing, authentication fails, the
request times out, or the paper environment cannot be proven. It never submits,
modifies or cancels an order — there is no flag that does.

## Monitoring dashboard (optional)

A small dashboard can be run to monitor and — if explicitly enabled — advance the
simulation. It is built from the standard library only (no extra dependency), and
it is simulation-only: the one action it can take drives
`PaperExperiment.process_session`, which never reaches a venue. There is no route
that places, modifies or cancels a broker order, and no credential is ever
returned to the browser.

**The dashboard does not run the engine by itself.** `serve` binds an HTTP server
and waits: it advances nothing until either a caller presses Run (`POST
/api/run`, needs `--allow-control`) or `--auto-advance` is set. This is deliberate
— `serve` is a monitor first — but it means a service started with a bare
`serve` shows an idle experiment (cash at its opening balance, no sessions, no
orders). If you want one Railway service to both run and monitor, pass
`--auto-advance`.

```bash
# Monitor only (loopback): reads the experiment, advances nothing.
python -m tradingagents.paper_experiment serve --root /data/paper-experiment

# One service that runs AND monitors: advance 3 sessions every 60s.
# Reachable from Railway: bind the interface Railway provides and set a token.
# Railway injects $PORT; `serve` reads it when --port is omitted.
TRADINGAGENTS_DASHBOARD_TOKEN=<a-long-random-string> \
python -m tradingagents.paper_experiment serve \
  --root /data/paper-experiment --host 0.0.0.0 --port "$PORT" \
  --allow-control --auto-advance 3 --auto-interval 60

# The module form is equivalent and takes its root from the environment, so the
# Railway start command can be one line:
TRADINGAGENTS_EXPERIMENT_ROOT=/data/paper-experiment \
TRADINGAGENTS_DASHBOARD_HOST=0.0.0.0 \
TRADINGAGENTS_DASHBOARD_TOKEN=<a-long-random-string> \
TRADINGAGENTS_AUTO_ADVANCE_SESSIONS=3 \
python -m tradingagents.paper_experiment.dashboard.server --allow-control
```

> Use the module form only through its entry point, as above. Importing the
> module without running it (`python -c "import ...dashboard.server"`) does not
> start the server; `dashboard/server.py` binds and serves under
> `if __name__ == "__main__"`. Running it with no `--root` and no
> `TRADINGAGENTS_EXPERIMENT_ROOT` exits `2` with a clear message rather than
> listening on nothing.

### Which price the paper worker sizes from

The paper worker sizes each order from a *reference price*. That price comes from
one of two seams, chosen by `--paper-market-data` (or
`$TRADINGAGENTS_PAPER_MARKET_DATA`); the flag wins over the environment:

- `bar_open` (the default) — the **historical** daily-bar open of the experiment's
  latest session. This is a past-session price, not a live quote. It is fine for
  exercising the plumbing, but it is not a current price.
- `alpaca_snapshot` — a **genuinely current** price read from Alpaca market data
  (`GET /v2/stocks/snapshots`, `feed=iex`). It needs `ALPACA_PAPER_API_KEY` and
  `ALPACA_PAPER_API_SECRET`. The price is the **newest** of the latest trade, the
  latest quote midpoint and the minute-bar close, compared by their own timestamps,
  so a symbol that has stopped printing trades on the IEX feed is not mistaken for
  a stale market. A snapshot whose newest field is older than
  `--paper-snapshot-max-age-seconds` (default 300s, or
  `$TRADINGAGENTS_PAPER_SNAPSHOT_MAX_AGE_SECONDS`) is **refused**: the worker
  raises and sizes no order rather than trading a stale number. There is no silent
  fallback from `alpaca_snapshot` to `bar_open`.

Both the `serve` subcommand and the module form select the seam the same way, from
the same code, so the two entrypoints cannot diverge. The chosen provider is
printed to stderr at startup and shown on the dashboard's **Market data** card.

```bash
# A paper service that sizes from a current Alpaca price, SPY, fractional shares
# allowed (SPY ~580 makes whole shares floor to zero under a $25 cap):
ALPACA_PAPER_API_KEY=<key> ALPACA_PAPER_API_SECRET=<secret> \
TRADINGAGENTS_DASHBOARD_TOKEN=<a-long-random-string> \
python -m tradingagents.paper_experiment serve \
  --root /data/paper-experiment --host 0.0.0.0 --port "$PORT" \
  --allow-control --auto-advance 3 \
  --paper-market-data alpaca_snapshot --paper-fractional-shares
```

`--paper-fractional-shares` (or `$TRADINGAGENTS_PAPER_FRACTIONAL_SHARES`) lets the
worker size a fractional quantity. Without it, whole shares only, and a high-priced
symbol under a small notional floors to zero and the order is refused with a clear
message. The account must support fractional trading or the broker rejects the
order; the $25 notional cap and the long-only rule are unchanged either way.

### One writer per experiment directory

State is a read-modify-write of one JSON file (`session_state.json`) plus an
append to `journal.json`. Two processes writing the *same* directory lose data:
the second `os.replace` overwrites the first, so the first writer's session
disappears. The dashboard serialises its own writers behind a lock, and
`--auto-advance` uses that same lock, so one process is safe. Across processes
there is no lock, so run **one writer**. Either:

- one dashboard service with `--auto-advance` (runs and monitors), and no cron; or
- a cron service running `run` for the engine, and a dashboard with **no**
  `--auto-advance` and **no** `--allow-control` (monitor only) — but even then the
  two must not overlap: a cron tick during a manual Run can still race.

If you need both, prefer the single auto-advancing dashboard service.

### Decisions need a real seam

A session advanced from the dashboard decides through the *same* inference seam a
terminal `run` would: the provider named in the persisted config, or the labelled
stub when `--stub` is set. Without a credential the decision is recorded as
**blocked** — honestly, not fabricated — but the session is still marked
processed, so those sessions cannot be re-decided later. Start the experiment
with the provider credential present, or use `--stub` for an offline
demonstration.

Safety properties, all enforced in code and covered by tests:

- **Auth is mandatory off loopback.** Binding a non-loopback host with no token
  is *refused* at startup, so the surface is never served open on a public
  interface. Every `/api` route requires `Authorization: Bearer <token>`, compared
  in constant time. `/healthz` is the only unauthenticated route (a liveness
  probe) and returns no experiment data.
- **Read-only by default.** `--allow-control` is required before the dashboard can
  advance the simulation. Without it, `POST /api/run` returns 403 and
  `--auto-advance` is refused.
- **Control is deliberate.** `POST /api/run` needs the bearer token *and* an
  explicit `X-Control-Confirm: yes` header, is bounded per call, and is refused
  while another run is in progress (the state file is single-writer).
- **State is the same file the CLI uses.** A session advanced from the dashboard
  is processed and persisted by the same `process_session`, so the dashboard, the
  CLI and a cron run cannot diverge *in code*; they must still not run at once.

Railway settings for this service, when you use one: start command
`python -m tradingagents.paper_experiment serve --root /data/paper-experiment --host 0.0.0.0 --port $PORT --allow-control --auto-advance 3 --paper-market-data alpaca_snapshot --paper-fractional-shares`
(or the module form above, which reads the root and host from service
variables), the same `/data` volume mounted, `TRADINGAGENTS_DASHBOARD_TOKEN` and
the `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_API_SECRET` pair set as service
variables (needed for `alpaca_snapshot`). Give it a **public domain** in
Railway's networking settings so the phone can reach it; without the token the
server refuses to start on that bind.

## Persistent volume and where state lives

Railway volumes persist across deploys but are mounted at a path **you** choose
([Volumes](https://docs.railway.com/reference/volumes)). Mount one and point
`--root` at it. The deployment in this repository mounts the volume at
`/home/appuser/app/data` and points `--root` at the same path, so the experiment
writes:

```
/home/appuser/app/data/
  journal.json          append-only, hash-chained, atomic writes
  session_state.json    versioned recovery state (includes the source's bars)
  snapshots/            one verifiable snapshot per decision
  worker_state.json     the worker's processed/ordered decision sets
  crypto_source_state.json
```

The worker's `processed_decisions` / `ordered_decisions` sets are what stop a
restart from placing an order for a decision that already produced one. They are
files in this directory, so **the volume is what makes a redeploy idempotent**.

These three are the durable state. **They must be on the volume.** A `--root` on
the container filesystem is lost on redeploy, which would discard the journal and
the recovered positions.

> "Docker images that run as a non-root UID by default will have permissions
> issues when performing operations within an attached volume. If you are
> affected by this, you can set `RAILWAY_RUN_UID=0`"
> ([Volumes — Caveats](https://docs.railway.com/reference/volumes)). The
> `Dockerfile` runs as the non-root `appuser`, so **set `RAILWAY_RUN_UID=0`** on
> the service, or otherwise ensure `appuser` can write the mount path, or the
> first write to the volume will fail.

## Restart behaviour and recovery

- State is written **atomically** (temp file + `os.replace`) after every session,
  so a crash mid-write leaves the old file intact, never a truncated one.
- On restart, `run` resumes from `session_state.json`: it does not reprocess a
  session already in `processed_sessions` and cannot duplicate a fill or a fee.
- A frame-backed source persists its validated bars, so a **vendor** run resumes
  on the *same* data **without a second API call** — important on a free tier.
- Railway's default restart policy is `On Failure` (max 10 restarts)
  ([Restart Policy](https://docs.railway.com/deployments/restart-policy)). Because
  the runner exits `0` on success, a clean finish does **not** trigger a restart;
  a non-zero exit does. If you deploy it as a cron service, the schedule — not the
  restart policy — drives the next run.

## Resource usage and API quotas

The runner is a short batch that processes a bounded number of sessions and
exits, so its footprint is small and brief: a single Python process, no
resident worker pool, and state files measured in kilobytes (one journal entry
and one decision per session, plus one snapshot per decision). The optional
dashboard is the only long-lived process, and only when you choose to run it. A
small experiment fits comfortably inside the smallest Railway plan; this is a
*shape* statement, not a bill. Railway's own plan limits are:

> "Free For running small apps with $1 of free credit per month" and, per the
> plan table, Free allows "1" replica with "0.5 GB" RAM, "1 vCPU", "0.5 GB"
> volume storage; the Hobby plan "includes $5 of resource usage per month"
> ([Plans](https://docs.railway.com/reference/pricing/plans),
> [FAQs](https://docs.railway.com/reference/pricing/faqs)).

No specific monthly charge is asserted here: Railway bills measured resource
usage, and the runner exits rather than idling, so a stopped service accrues no
compute. Verify current pricing before budgeting.

The binding constraint is not Railway but the **free vendor quotas**, which the
runner must not exhaust:

| Provider | Free-tier limit | Why it matters here |
|---|---|---|
| Alpha Vantage | "25 API requests per day" ([Premium](https://www.alphavantage.co/premium/)) | one fetch per *new* run; a restart reuses persisted bars and makes **no** call |
| Finnhub | "60 API calls/minute" on Free; OHLC is blank on Free ([Pricing](https://finnhub.io/pricing)) | one quote fetch per run; `/quote` cannot drive a multi-session window |
| Groq | per-model RPM/RPD/TPM/TPD limits ([Rate limits](https://console.groq.com/docs/rate-limits)) | one inference per session, `max_retries=0`, so one session is one request |

The design keeps polling minimal by construction: a source fetches **once**, at
construction, and freezes the frame; `run --sessions N` processes a bounded
batch; and a resume reads persisted bars instead of re-fetching. There is no
timer, no polling loop, and no per-tick vendor call. A scheduled (cron) service
should be given a cadence well inside the daily quota — e.g. at most one new
Alpha Vantage fetch per day on the free plan.

## Config files: what is committed and why

- **`.railway/railway.ts`** is the deployment's Infrastructure as Code. It is
  the file Railway reads going forward, and it describes the crypto dashboard
  service: the start command, the `/healthz` health check and the non-secret
  service variables
  ([Infrastructure as Code](https://docs.railway.com/infrastructure-as-code/reference)).
- **`railway.json`** is the deprecated Config as Code. Railway still reads it
  for existing services but stops reading it on **2026-12-01**
  ([Config as Code](https://docs.railway.com/reference/config-as-code)), so it
  is kept as a short-lived bridge for a service that has not yet migrated. The
  two files describe the **same** service, and
  `tests/test_crypto_deployment_wiring.py` fails if they drift apart.
- **No `Procfile`.** Railpack reads one, but it is "not recommended" and is
  ignored entirely when a Dockerfile is present — and this repo has a Dockerfile
  ([Build Configuration](https://docs.railway.com/builds/build-configuration)).
- **No startup script.** The two things that must be configured — the start
  command and the volume mount path — are service settings, set in the dashboard
  or via IaC, not files in the repository.

### The start command must be shell-wrapped

Railway runs a Dockerfile service's start command in **exec form**, where
`$PORT` is not expanded, so the command is wrapped:

```
/bin/sh -c "exec python -m tradingagents.paper_experiment.dashboard.server --root /home/appuser/app/data --host 0.0.0.0 --port $PORT ..."
```

Without the wrapper the server would receive the literal string `$PORT` and
fail to bind ([Start command](https://docs.railway.com/deployments/start-command)).

### `--allow-paper-trading` permits, it does not start

The flag **permits** the dashboard's START button to build a paper worker. The
worker itself starts only on an authenticated `POST /api/paper/start`, and the
broker it may reach is Alpaca's **paper** endpoint — the live host is refused in
code. A deployment that is never pressed therefore places no order.

## The crypto dashboard service

The committed configuration serves the **crypto dashboard**: the live crypto
market section (Binance public market data, no API key), the TJR strategy view
over the same live feed, and the Alpaca **paper** worker controls.

| Setting | Value | Notes |
|---|---|---|
| Start command | the module form, shell-wrapped | see above |
| Health check | `/healthz` | unauthenticated liveness probe; returns no experiment data |
| Decision source | `crypto` | the unchanged TJR rule chain over Binance 1m/5m bars, no model |
| Market data | `binance_snapshot` | Binance public REST; no key, no registration |
| Order notional | `25` | the shared portfolio's hard cap |
| Price deviation | `0.05` | the largest tolerated gap between the decision and execution prices |
| Snapshot max age | `15` | a price older than this is refused, not used |
| Volume | `/home/appuser/app/data` | the durable state; without it a redeploy can repeat an order |
| `RAILWAY_RUN_UID` | `0` | the image's non-root user cannot write to the volume |

The Alpaca **paper** credentials, the dashboard token and any LLM key are **not**
in the repository. They are Railway service variables, set in the dashboard.

### The two symbol lists are not the same list

`TRADINGAGENTS_CRYPTO_SYMBOLS` is what the live section *displays*;
`TRADINGAGENTS_EXPERIMENT_SYMBOLS` is what the *experiment* is created with.
Both are requests, and both are validated against the venues' own registries
before use. The committed values are Binance's USDT pairs — the market-data
venue's own spelling — and the execution venue's USD pair is resolved from
Alpaca's own listing rather than assumed.

## What is not ready without live credentials

Deploying the image is possible today, but a run on a vendor source or a genuine
inference needs the keys above set in the Railway service. Without them, the
experiment runs on the frozen dataset and records blocked decisions — a valid,
offline, reproducible state, but not a live paper run. See
`docs/MARKET_DATA_PROVIDERS.md` for what each vendor can and cannot serve.

## Operator checklist

The steps below are the ones that need a Railway account and an Alpaca account.
They are written here because the agent that prepared this deployment had
neither account, so **nothing in this repository has been deployed and no
broker command has been executed**. Everything that *could* be verified offline
was verified: the image builds, the start command boots, `/healthz` answers, the
Binance WebSocket connects and streams, the TJR chain evaluates live bars, and
the paper-only guards refuse the live host.

### 1. Create the service

1. Push this branch and create a Railway service from it. Railway finds the
   repository `Dockerfile` automatically; the committed `railway.json` and
   `.railway/railway.ts` describe the same build.
2. Add a **volume** to the service and mount it at `/home/appuser/app/data`.
   This is the step that makes a redeploy safe: without it the worker's
   processed/ordered decision sets are lost and a decision that already produced
   an order could be placed again.
3. Set `RAILWAY_RUN_UID=0` on the service. The image's default user is the
   non-root `appuser`, which cannot write to a Railway volume mount; the first
   write fails with `PermissionError` and the state never lands.

### 2. Set the variables

Set these as Railway service variables. None of them is in the repository, and
no secret was committed.

| Variable | Required | Value |
|---|---|---|
| `TRADINGAGENTS_DASHBOARD_TOKEN` | yes | `openssl rand -hex 32` — the dashboard is unauthenticated without it |
| `ALPACA_PAPER_API_KEY` | for paper trading | from <https://app.alpaca.markets/signup>, the **paper** account |
| `ALPACA_PAPER_API_SECRET` | for paper trading | the paper secret, not the live one |
| `TRADINGAGENTS_CRYPTO_SYMBOLS` | no | the pairs the live section displays; the committed default is fine |
| `TRADINGAGENTS_EXPERIMENT_SYMBOLS` | no | the pairs the experiment is created with; the committed default is fine |

The Alpaca **paper** keys are not the live keys. The project refuses live hosts
and live-sounding credential names outright, so a live key set here does not
enable live trading — it makes the paper path fail loudly instead.

### 3. Open the dashboard and start paper trading

1. Open the Railway-generated domain. The dashboard shows an **Access token**
   field: paste `TRADINGAGENTS_DASHBOARD_TOKEN` there. The page keeps it in the
   browser's local storage and sends it as `Authorization: Bearer <token>` on
   every API call, so it is entered once per browser. There is no token in the
   URL — the dashboard serves the shell without a token and refuses every API
   call until one is entered.
2. The **Live Market** section fills from Binance's public WebSocket. It needs
   no key and no registration.
3. The **TJR Strategy** section shows the chain's own evaluation per pair. It
   reads the decision source's persisted records, so it says `NO SIGNAL` with
   the chain's real reason until a cycle has run.
4. Paper trading is **off** until someone presses START. The button requires
   an explicit confirmation (`X-Paper-Confirm: yes`), and the worker only
   starts after that authenticated POST.
5. Press **Stop New Entries** to halt new orders. Open positions and open
   orders are left alone: stopping does not close a position.

### 4. What to check after the first deploy

- `GET /healthz` returns `{"status": "ok"}`.
- The service logs show the Binance stream connected, with a rising message
  count and `duplicates: 0`.
- `journal.json`, `worker_state.json` and `crypto_source_state.json` appear
  under `/home/appuser/app/data` — their presence proves the volume is writable.
- A redeploy keeps those files, and the worker does not re-place an order for a
  decision that already produced one.

### 5. If crypto paper trading is refused

Alpaca's crypto support is not universal: the account must be enabled for
crypto, and the pair must be in Alpaca's own listing. The project reads
Alpaca's asset registry and refuses a pair Alpaca does not serve, so the
symptom is a clear refusal rather than a failed order. The dashboard reports
the reason under **System Health**.
