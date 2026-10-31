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
`python -m tradingagents.paper_experiment serve --root /data/paper-experiment --host 0.0.0.0 --port $PORT --allow-control --auto-advance 3`
(or the module form above, which reads the root and host from service
variables), the same `/data` volume mounted, and `TRADINGAGENTS_DASHBOARD_TOKEN`
set as a service variable. Give it a **public domain** in Railway's networking
settings so the phone can reach it; without the token the server refuses to start
on that bind.

## Persistent volume and where state lives

Railway volumes persist across deploys but are mounted at a path **you** choose
([Volumes](https://docs.railway.com/reference/volumes)). Mount one at `/data` and
point `--root` at a directory under it. The experiment writes:

```
/data/paper-experiment/
  journal.json          append-only, hash-chained, atomic writes
  session_state.json    versioned recovery state (includes the source's bars)
  snapshots/            one verifiable snapshot per decision
```

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

## Config files: why none are added

- **No `railway.toml` / `railway.json`.** Railway has deprecated Config as Code;
  it is no longer read for new services and stops being read on **2026-12-01**;
  the replacement is Infrastructure as Code (`.railway/railway.ts`)
  ([Infrastructure as Code](https://docs.railway.com/infrastructure-as-code)).
  Committing a soon-dead config file would be a trap, not a convenience.
- **No `Procfile`.** Railpack reads one, but it is "not recommended" and is
  ignored entirely when a Dockerfile is present — and this repo has a Dockerfile
  ([Build Configuration](https://docs.railway.com/builds/build-configuration)).
- **No startup script.** The two things that must be configured — the start
  command and the volume mount path — are service settings, set in the dashboard
  or via IaC, not files in the repository.

## What is not ready without live credentials

Deploying the image is possible today, but a run on a vendor source or a genuine
inference needs the keys above set in the Railway service. Without them, the
experiment runs on the frozen dataset and records blocked decisions — a valid,
offline, reproducible state, but not a live paper run. See
`docs/MARKET_DATA_PROVIDERS.md` for what each vendor can and cannot serve.
