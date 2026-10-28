# Railway deployment readiness

This document states what deploying the **prospective paper experiment** to
Railway requires, and what it does not. It describes a simulation-only workload:
no broker connection, no order routing, no credential beyond a market-data and an
LLM provider key, and no `ExecutionMode.LIVE` (which does not exist).

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

## Python version and dependencies

- `requires-python = ">=3.11"`; the `Dockerfile` builds on `python:3.13-slim`.
- No extra dependency is needed for the vendor providers: they use the standard
  library (`urllib`, `json`) plus `pandas`, which is already a core dependency.
  The Groq LLM path uses the existing OpenAI-compatible client (`requests`).
- `.dockerignore` already excludes `.git`, `**/.env*`, `results`, `build`, `dist`.

## Environment variables

| Variable | Needed for | Notes |
|---|---|---|
| `GROQ_API_KEY` | genuine LLM inference | provider `groq`, model `openai/gpt-oss-20b` |
| `ALPHA_VANTAGE_API_KEY` | `--source alpha_vantage` | free daily bars; 25 calls/day |
| `FINNHUB_API_KEY` | `--source finnhub` | free current-day quote only |

Set these as Railway **service variables** (never in the repository). The code
reads them from the environment by name only; it never prints, logs or stores a
value. If a variable is missing, a decision is recorded as **blocked** rather
than fabricated, and the run exits cleanly.

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
