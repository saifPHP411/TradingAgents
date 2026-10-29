# Changelog

All notable changes to TradingAgents are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changes that need action when upgrading are listed under "Upgrading from" in their release.

## [Unreleased] — simulated trading system

Adds a deterministic simulated trading system alongside the existing multi-agent
analysis, so a decision can become a position with honest books kept about it.
The agent workflow is unchanged and stays the analytical component; execution,
sizing, constraints and accounting are separate code. There is no live trading
path, and no configuration that produces one.

### Added

- Persistent LIMIT orders (`execution/`, `backtesting/`): an order can rest
  across sessions instead of being single-shot. `TimeInForce.DAY`/`GTC`, an
  `expires_session` and a `limit_order_ttl_sessions` bound how long it may rest;
  an expired order is cancelled with the new `EXPIRED` reason. A GTC limit the
  bar did not reach stays `ACCEPTED` and is worked again, recording its
  acceptance once (`Order.accepted_recorded`). Reachability uses the bar's
  low/high when supplied, so a limit the session traded to (but did not close at)
  fills. The backtester and the paper session both emit LIMIT orders from a
  signal's `order_intent`, and the paper session advances resting orders with
  `PaperSession.work_resting()`.
- Signal order intent (`signals/`): `OrderIntent` (`MARKET`/`LIMIT`) and
  `SignalTimeInForce` (`DAY`/`GTC`) on `TradingSignal`, with `limit_price` and
  `expires_session`. A LIMIT intent needs a limit on the correct side of the
  entry; a MARKET intent refuses one; only a LIMIT order may be GTC.
- A validated signal contract (`signals/`): a decision becomes a `TradingSignal`
  only if its action is BUY/HOLD/SELL, its confidence is in range, and its size
  and stop/target are coherent. A HOLD cannot carry a size.
- A portfolio ledger (`portfolio/`): cash, positions, average entry, market
  value, equity, realized and unrealized P&L, fees and exposure, with an event
  log (`ORDER_FILLED`, `POSITION_OPENED`, `FEE_CHARGED`, and the rest). Every
  cash or position change goes through one method.
- A deterministic risk engine (`risk/`): configurable limits on position size,
  portfolio exposure, open positions, daily loss, drawdown and order value.
  Long-only is enforced in code, not by prompt. A position cap bounds the
  resulting position, not each order, so repeated buys cannot accumulate past it.
- A simulated execution engine (`execution/`): MARKET and LIMIT orders with a
  full lifecycle, named rejection reasons, and configurable costs — commission,
  half-spread, slippage and volume-scaled impact. Costs land in the ledger.
- A portfolio, event-loop backtester (`backtesting/`): market event, decision,
  risk validation, order, fill, portfolio update, P&L. Orders fill at the next
  session's open, bars are sliced strictly at each session, and stops fill at the
  open on a gap rather than at the level.
- Performance analytics (`performance/`): total and annualized return, CAGR,
  realized and unrealized P&L, fees, trade counts, win rate, profit factor,
  average win/loss, expectancy, max drawdown, Sharpe, Sortino, Calmar, exposure,
  turnover, and benchmark comparison. An undefined metric is `None`, never zero.
- Walk-forward evaluation (`backtesting/walkforward.py`): train, validation and
  test windows, with fold ordering validated at construction.
- LLM cost telemetry and budgets (`telemetry/`): per-call tokens, latency, retries
  and estimated cost, a run cost ceiling, a call ceiling, and run metadata for
  reproducibility.
- Paper trading (`execution/paper.py`): the same pipeline driven by live quotes
  from a caller-supplied source instead of historical bars.
- Paper-session persistence and restart recovery (`execution/paper.py`,
  `portfolio/ledger.py`): `Ledger.to_state`/`from_state` and
  `PaperSession.state`/`save_state`/`restore` write and rebuild the *complete*
  session — the whole book including its event log and dense `seq` counter, the
  risk limits, the cost schedule, the simulator settings, the live resting orders
  with their remaining size and weighted average fill price, the processed-signal
  set and the day baseline. It is versioned (`PAPER_STATE_SCHEMA_VERSION`), and
  `PaperSession.restore` raises `PaperStateError` for a malformed file, an unknown
  version, or a state claiming a live mode. This is deliberately distinct from the
  reporting snapshot (`snapshot`/`write_json`), which cannot rebuild a book.
- Duplicate-signal refusal in a paper session: a `signal_id` already processed is
  refused with a `DUPLICATE_SIGNAL` step and no order, so a re-driven decision
  (crash recovery, a duplicated message) cannot create a second position or fee.
- `PaperSession.cancel_resting` cancels a live resting order, recording exactly
  one `ORDER_CANCELLED` event.
- `Quote` accepts an optional intraday `low`/`high` range and validates it, so a
  LIMIT order's reachability can use the session range in paper mode as it does
  against a bar, and a malformed range is refused.
- Reports (`trading_report/`): one document covering decisions, execution,
  portfolio and performance, rendered to self-contained HTML and JSON.
- Point-in-time guarantees per data source (`dataflows/pit.py`): each source is
  classified `REQUEST_BOUNDED`, `WINDOW_FILTERED` or `NOT_PIT_SAFE`, with the
  guard that implements it cited and checked by tests.
- Documentation: `docs/ARCHITECTURE.md`, `docs/BACKTESTING.md`, `docs/RISK.md`,
  `docs/EXECUTION.md`, `docs/PAPER_TRADING.md`, `docs/TRADING_SYSTEM_AUDIT.md`
  and `docs/FINAL_DEVELOPMENT_REPORT.md`, plus `examples/simulated_run.py`.
- An official strategy baseline (`tradingagents/baseline/`): a frozen, versioned
  protocol (`BaselineProtocol`, `PROTOCOL_VERSION`), a committed dataset under
  `tradingagents/baseline/data/` (fetched once by
  `python -m tradingagents.baseline.fetch_data`, the only entry point in the
  package that reaches the network, and it does so through the data layer), a
  deterministic pre-registered reference strategy
  (`SmaCrossover`, 20/100, fixed a priori), and a harness that runs it through the
  existing backtester over a development/validation/test split and a rolling
  walk-forward. It measures with the existing performance engine and recomputes
  the headline metrics independently (`crosscheck.py`) before publishing. The
  artifact (`docs/baseline/official_baseline.json`) records the protocol, the
  data checksums, the strategy id and the seed, so a result always names the spec
  that produced it. Documented in `docs/STRATEGY_BASELINE.md`. This is a
  *reference* baseline: it does not measure the non-deterministic LLM strategy,
  which needs credentials and cannot be replayed to one number.
- A decision-replay evaluation of the LLM strategy (`llm_evaluation/`): the layer
  the reference baseline above is missing. A `DecisionRecord` captures one agent
  decision with the hash of the data window the model saw and the config hash it
  ran under; `ReplayStrategy` feeds recorded decisions into the *existing*
  `PortfolioBacktester` through the production `TradingSignal` contract, so a
  replayed decision travels the same decision → risk → order → fill → ledger path
  as any other strategy. The replay calls no model (`no_llm_calls()` replaces the
  client factory with one that raises), refuses a window-hash mismatch, and
  refuses a rating/action mismatch. `build_evaluation()` reports
  `LLM BASELINE BLOCKED` with the exact missing prerequisite — a provider
  credential or a captured record set — and never invents a number. It records the
  news/sentiment/fundamentals gap explicitly: the frozen dataset is price-only, so
  a replay is a restricted, lower-bound evaluation, not the full pipeline's
  result. Three record sources (`llm_inference`, `operator_supplied`,
  `synthetic_fixture`) are kept strictly apart so a fixture cannot be reported as
  performance. Every record, record set and artifact is scanned for
  credential-shaped values before it can be written; a hit raises
  `CredentialLeakError` naming the field, never the secret. Artifacts are
  versioned (`artifact_schema_version`) and two builds of the same records are
  byte-identical except for the timestamp. Documented in
  `docs/LLM_EVALUATION.md`. No live trading path is added.
- A point-in-time readiness audit of the agent graph
  (`llm_evaluation/point_in_time.py`): the agent-by-agent dependency matrix is
  derived from the code — the graph spec (`graph/analyst_execution.py`), the
  router's tool→vendor map (`dataflows/router.py`) and the PIT registry
  (`dataflows/pit.py`) — so it cannot go stale without a test failing. Each source
  gets a classification (`PIT_VERIFIED`, `PIT_PARTIALLY_VERIFIED`,
  `PIT_UNVERIFIED`, `CURRENT_ONLY`, `UNAVAILABLE`) and the guard that implements
  it, and `graph_readiness(scope)` returns a readiness verdict with named
  blockers. `price_only` (the market analyst) is ready for a historical run — it
  reads only request-bounded sources; `full` is not, because the news/sentiment
  sources are window-filtered and the fundamentals/social sources are current-only.
  The verdict, matrix and classifications are embedded in the evaluation
  artifact's new `point_in_time`, `evaluation_scope` and `limitations` blocks, so
  a restricted result cannot be read as a full-graph one. Documented in
  `docs/LLM_POINT_IN_TIME_AUDIT.md`. No live trading path is added.
- A provider-readiness audit (`llm_evaluation/provider_audit.py`): readiness is
  derived from the client registries and the *names* of the env vars they declare,
  so it reports which providers this environment can reach without reading a
  credential value or contacting a provider. Each provider carries its transport,
  credential requirement, env var, whether structured output is needed, and any
  documented free API tier with the official URL it was checked against. The
  report is embedded in the evaluation artifact as `provider_readiness`, and a
  blocked evaluation now names the documented free-tier options. Documented in
  `docs/LLM_PROVIDER_AUDIT.md`. No live trading path is added.
- A real decision-capture path (`llm_evaluation/capture.py`): turns a completed
  graph run into a validated `DecisionRecord`. It reads the rating, the
  session-bounded price window (hashed, so a replay can prove it saw the same
  bars), the config hash and the model/provider metadata, and refuses a rating
  with no tradeable action rather than degrading it to Hold. It invents no size,
  stop or limit. The `llm_inference` label is derived from the **graph**, not the
  caller: the graph carries an `InferenceObserver`
  (`llm_clients/observed.py`) on its real provider clients, so a scripted or
  stubbed model can never be captured as genuine inference — a mock run yields a
  `synthetic_fixture` record with a note saying why. Replay gained an optional
  `expected_config_hash`, so a record captured under a different protocol is
  refused rather than replayed as if it were this one.
- A bounded, opt-in provider smoke test (`llm_evaluation/smoke.py`,
  `python -m tradingagents.llm_evaluation.smoke`): the only path that may make a
  real model call. It checks configuration before any request, names the exact
  env var to set and stops if absent, makes **at most one** small request with a
  timeout, validates the returned decision with the production rating parser, and
  writes a record only when the decision and metadata are valid. It never places
  an order and never touches execution or risk. Default provider is Groq (a
  documented free tier with an OpenAI-compatible API; `openai/gpt-oss-20b` is a
  documented production model). An injected client (the test seam) is reported as
  a fixture, never a genuine observation, so a mocked test can never be counted
  as real inference.
- An `evaluation_mode` and `capture_status` block in the evaluation artifact, so
  the report cannot conflate `REAL_INFERENCE_CAPTURE`, `REPLAY` and `MOCK_TEST`.
  `capture_status` states, field by field, whether a provider is configured,
  whether a genuine request succeeded, whether a record was captured, whether its
  inputs are reproducible, whether they are point-in-time safe, and whether
  historical evaluation is allowed — for the price-only scope only, since the
  full graph's inputs remain not PIT-safe. Input reproducibility is reported
  separately from point-in-time safety. No live trading path is added.
- A `failure_category` on the smoke report (`llm_evaluation/smoke.py`): a failed
  provider call is classified from a closed vocabulary (authentication failure,
  model unavailable, rate limit, timeout, malformed response, provider error,
  provider unreachable, configuration error, other) instead of only echoing an
  error string. The smoke path also sets `max_retries=0`, so the SDK's default
  three attempts on a 5xx cannot turn one opt-in request into three billable
  ones.

### Phase 9.5 — reproducible price input

The first genuine Groq run (Phase 9) produced a real decision but recorded
`bar_count: 0` and `data_snapshot_id: feed:unavailable`: the inference was real,
its input was not reproducible. Two causes, both fixed.

The smoke CLI never loaded a price feed, and its default session was *today*,
which is past the frozen dataset's last session (2024-12-30). A record requires
its window to end at the decision date, so a window ending 2024-12-30 for a
decision dated today was dropped. The smoke path now loads the frozen dataset
when no feed is injected, defaults the session to the dataset's last covered
session, and snapshots the window it captures.

Added:

- A versioned, self-verifying price snapshot (`llm_evaluation/snapshot.py`,
  `python -m tradingagents.llm_evaluation.snapshot`): `snapshot_id` is
  `<symbol>-<start>..<end>-<hash12>`; `content_hash` is the same SHA-256 the
  decision record carries (`replay.window_hash`), so
  `snapshot.matches_record(record)` is a checkable claim. `load` recomputes the
  hash and refuses a tampered or drifted file; an empty window is refused at
  construction rather than recorded as `unavailable`. A committed snapshot lives
  at `docs/baseline/SPY_price_snapshot.json`.
- `window_reproducible` and `experiment_ready` on the smoke report.
  `experiment_ready` requires a genuine decision **and** a reproducible window
  that matches its snapshot — a genuine inference alone is not enough.
- `snapshot_id` on `build_record`, so a record names a reloadable artifact
  rather than a bare date range.

Changed:

- `prospective_paper_experiment_possible` now requires a reproducible window, not
  only a captured record. The `feed:unavailable` record no longer satisfies it.
- The smoke report no longer claims "one genuine model call" when an injected
  (test) client was used.

### Phase 12 — restart recovery of a vendor-backed paper experiment

Phase 10 proved a restart is idempotent **when the caller hands the restored
session the same price source**. That hid a real defect: a vendor source fetches
once at construction, so a resume that rebuilt the source from scratch either
re-fetched (a second, quota-consuming call — and a failure on the free tier) or
silently resumed on different data. This phase persists the source so a restart
recovers the *same* run.

Added:

- **Persisted source bars** (`sources.FramePriceSource.to_dict`/`from_dict`,
  `SourceProvenance.from_dict`): a frame-backed source (the frozen dataset, a
  static fixture, or a vendor daily source) writes its validated bars and
  provenance into `session_state.json` under a `source.restore` block.
  `PaperExperiment.load(root)` now rebuilds the source from that block, so a
  **vendor** run resumes without a second API call and on the same bars. A stored
  frame is re-validated on restore, so a corrupted state cannot poison the book.
- **A versioned, explicit restore path**: `load(root, source=None)` reads the
  persisted bars; passing a source explicitly still works. A source that cannot
  be persisted (Finnhub's quote source, which carries no bar history) is refused
  with a clear error rather than silently swapped for the frozen dataset.
- **Phase 12 tests** (`tests/test_phase12_restart_source_persistence.py`): a
  restart *without* an explicit source matches a straight run on accounting and
  events; a vendor run restarts without refetching (asserted by counting the
  mocked transport calls); multiple symbols survive; a corrupted or absent
  restore block is refused; and a credential in the environment never reaches the
  state file.
- **Credential-gated live tests** (`tests/test_live_providers_integration.py`):
  marked `integration` and skipped unless the relevant key is present, so the
  default suite stays hermetic. They make the minimum real calls (one Alpha
  Vantage, one Finnhub, one LLM inference) and never print a key.
- **`docs/RAILWAY_DEPLOYMENT.md`**: what deploying the paper runner to Railway
  requires — a worker/cron shape, the start-command override, the volume and the
  `RAILWAY_RUN_UID=0` caveat, and why no Config-as-Code file is committed.

Changed:

- `docs/PAPER_EXPERIMENT.md`, `docs/MARKET_DATA_PROVIDERS.md`: state the resume
  mechanism precisely — a frame-backed vendor run is resumed from its own stored
  bars, and the refusal applies only to a source with no persistable bars.

### Phase 3 (reliability) — read-only Alpaca paper connection verification

Extends the Alpaca paper broker adapter from a mocked-only check to a **real,
read-only** connection check, without adding any order path.

**Added — `alpaca-diagnose` (alias of `broker-check`), a read-only paper
diagnostic.** It reports, without any secret value, whether the paper credential
env vars (`ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_API_SECRET`) are present and
whether the endpoint is the approved paper host. With `--connect` it performs one
read-only `GET /v2/account` and reports the account id, status, currency, cash,
equity, buying power and active restrictions. It places no order and changes
nothing; there is no flag that submits one. It fails safe — non-zero exit, clear
secret-free message, no live fallback — when a credential is missing,
authentication fails, the request times out, or the response is not the expected
object.

**Added — richer, documented account fields.** `BrokerAccount` now reads the
optional fields Alpaca documents on `GET /v2/account` (`account_number`,
`portfolio_value`, `last_equity`, `long_market_value`, `short_market_value`,
`multiplier`, `shorting_enabled`, `transfers_blocked`, `trade_suspended_by_user`,
`daytrade_count`) and exposes a `restrictions()` list from a closed vocabulary. A
missing optional field is reported as unknown (`None`), never a fabricated zero.

**Added — a credential-gated live check.**
`tests/test_live_providers_integration.py::TestAlpacaPaperLive` runs the real
read-only paper account check when both paper credentials are present and skips
otherwise, so the default suite needs no secret and CI stays hermetic.

The adapter remains paper-only and separate: the endpoint is hard-validated to
Alpaca's paper host (every live host is refused), live credential variable names
are refused rather than used as a fallback, and the module never imports or writes
the internal ledger or simulator. `ExecutionMode` still has only `BACKTEST` and
`PAPER`. Tests: `tests/test_phase3_broker.py`.

### Phase 3 (reliability) — restored currency, and an isolated Alpaca paper broker

Two changes, both aimed at making the paper path more honest and safer. Neither
adds a live path: `ExecutionMode` still has only `BACKTEST` and `PAPER`.

**Fixed — a restored provenance kept claiming the current market after its bar
went stale.** `SourceProvenance.is_current_market` was computed once at fetch time
and trusted verbatim on restore, so a vendor snapshot persisted while its newest
bar was fresh still reported `current market: True` weeks later — the stale date
that showed up in a report. The freshness *rule* is now persisted
(`SourceProvenance.max_stale_days`) and re-evaluated against the current date on
restore, through the one shared helper (`bar_is_fresh`) a fresh fetch also uses. A
source with no freshness rule (the frozen dataset, a fixture) keeps its recorded
flag. Tests:
`tests/test_phase12_restart_source_persistence.py::TestRestoredFreshnessIsReevaluated`.

**Added — `tradingagents/paper_experiment/broker/`, an Alpaca *paper* broker
adapter, isolated from the internal simulator.** Alpaca Paper has its own
simulated account, so the adapter keeps its **own** account state and recovery
file and never imports or writes the internal ledger or simulator — a broker fill
cannot be double-counted. Paper-only is enforced structurally:
`assert_paper_endpoint` refuses every live Alpaca host and any unapproved host;
credentials come from `ALPACA_PAPER_API_KEY`/`ALPACA_PAPER_API_SECRET` only (live
variable names are refused, not used as a fallback); construction performs no
request; an order needs an explicit `allow_orders=True` after a read-only
`connect()`. Each order carries a stable `client_order_id`; intent is persisted
before the request; a lost response is reconciled by that id rather than
re-submitted; and a fill is applied at most once across a restart (a persisted
`applied_fills` set). A read-only `broker-check` CLI command reports
configuration without printing a secret and places no order. Tests:
`tests/test_phase3_broker.py`.

### Phase 14 — paper monitoring dashboard (simulation-only)

A small, phone-friendly dashboard for a paper experiment, added because the
project had no web surface at all (the CLI was the only way to look at a run).
It is a **monitoring and control surface for the simulation**, not a trading
surface, and it keeps every existing simulation-only guard.

- `tradingagents/paper_experiment/dashboard/` — a standard-library HTTP server
  (no new dependency). `GET /` serves one self-contained HTML page (no external
  assets), `GET /healthz` is an unauthenticated liveness probe, and
  `GET /api/summary`, `/api/decisions`, `/api/journal`, `/api/control/status`
  are read-only JSON views of the persisted experiment. `POST /api/run` advances
  the simulation by N sessions.
- Auth: every `/api` route requires `Authorization: Bearer <token>`, compared in
  constant time. Binding a **non-loopback** host with no token is **refused at
  startup**, so the surface is never served unauthenticated on a public
  interface. The token comes from `TRADINGAGENTS_DASHBOARD_TOKEN` or `--token`,
  and is never logged.
- Control is **off by default** (`--allow-control` enables it), needs a bearer
  token *and* an explicit `X-Control-Confirm: yes` header, is bounded per call,
  and refuses a concurrent run so the state file stays single-writer.
- `tradingagents/paper_experiment/runner.py` — one shared
  `process_pending`/`pending_sessions` path, so the CLI's `run` and the
  dashboard's control action process and persist sessions through the *same*
  code. There is no second implementation to drift.
- CLI: `serve` (`--host`, `--port`, `--token`, `--allow-control`), reading
  `$PORT` when `--port` is omitted. `docker-compose.yml` gains an optional
  `dashboard` service under a `dashboard` profile.
- Safety boundary unchanged: no live broker execution, no order routing, no
  credential, no `ExecutionMode.LIVE`, no weakened guard. There is no route that
  places, modifies or cancels a broker order.

Documented in `docs/RAILWAY_DEPLOYMENT.md` (a "Monitoring dashboard (optional)"
section) and covered by `tests/test_phase13_dashboard.py`.

### Fixed — the dashboard module entry point did not start the server

`python -m tradingagents.paper_experiment.dashboard.server` — the start command
a platform (Railway) would reach for — emitted a runpy `RuntimeWarning`
("'...dashboard.server' found in sys.modules after import of package
'...dashboard', but prior to execution") and then **exited without listening**:
`dashboard/__init__.py` imported `.server` eagerly, so runpy found it already
loaded, and `server.py` had no `__main__` guard, so the module ran and stopped.

- `dashboard/__init__.py` now re-exports `Dashboard`/`serve` lazily (PEP 562), so
  importing the package no longer pulls `.server` into `sys.modules` and the
  runpy warning is gone. `from ...dashboard import Dashboard, serve` still works.
- `dashboard/server.py` gains a `main()` entry point and an
  `if __name__ == "__main__"` guard. `--root` falls back to
  `TRADINGAGENTS_EXPERIMENT_ROOT`, `--host` to `TRADINGAGENTS_DASHBOARD_HOST`
  (default loopback) and `--port` to `$PORT`, so the start command can be one
  line. A missing root exits `2`; a non-loopback bind with no token is still
  refused before the socket is opened.

Verified by running the real module as a subprocess: it serves `/healthz` on
`0.0.0.0:$PORT`, and its output carries no `RuntimeWarning`.

### Phase 13 — pre-Railway deployment verification

A verification pass, not a feature. Its job was to prove the deployable image
and the deployment story, not to add capability. Nothing in the trading system
changed: no live path, no new source, no accounting change.

Verified empirically (against the built image, not asserted from reading code):

- **The image builds and runs.** `docker build` succeeds on `python:3.13-slim`.
  The Railway start-command override — `python -m tradingagents.paper_experiment
  run --root /data/paper-experiment ...` — was exercised in the container with a
  volume at `/data`; it processed sessions, wrote `journal.json`,
  `session_state.json` and `snapshots/`, and a second invocation **resumed**.
- **The volume caveat is real, not theoretical.** On the image's default non-root
  user the first write to the mount failed with `PermissionError`; running with
  `RAILWAY_RUN_UID=0` (as `docs/RAILWAY_DEPLOYMENT.md` instructs) fixed it. The
  doc already required the variable; this confirms *why*.
- **The image carries no secrets.** A scan of the built image found no `.env` and
  no key literal; `.dockerignore` excludes `**/.env*`.
- **Error classification is intact offline.** With no credential,
  `--source alpha_vantage` / `--source finnhub` fail closed (exit `2`) with a
  named reason and no silent fallback to the frozen dataset.

Documented:

- `docs/RAILWAY_DEPLOYMENT.md` gains a **Resource usage and API quotas** section
  (Railway plan limits and the Alpha Vantage / Finnhub / Groq free-tier limits,
  each cited) and a note that the start command *overrides* the image ENTRYPOINT
  in exec form, with the shell-wrapped alternative for variable expansion.

Not done, and why: the real-API checks (one Alpha Vantage daily fetch, one
Finnhub quote, one genuine Groq inference) and the resulting end-to-end paper
run were **not executed**, because no provider credential was present in the
environment. The gated tests remain in `tests/test_live_providers_integration.py`
and skip cleanly when a key is absent, so CI needs no secret. See the Phase 13
report for the exact remaining blocker.

### Fixed — Alpha Vantage sent no `apikey`; then rejected its own session order

`AlphaVantageDailySource` read `ALPHA_VANTAGE_API_KEY`, refused a missing key,
and passed the key to `fetch_json` — but only as a *redaction secret*. The
outgoing query carried `function`, `symbol`, `outputsize` and `datatype` and
**no `apikey`**. Alpha Vantage answers such a request with

> "the parameter apikey is invalid or missing"

no matter how valid the key is, so every `--source alpha_vantage` run failed with
a message that looked like a bad key and was not. The credential is now sent as
the `apikey` query parameter, assembled in one `build_request_params` helper so
the parameter cannot be dropped from one call path while another keeps it.
Credentials are also trimmed of surrounding whitespace, an all-whitespace key is
refused like a missing one, and a vendor message that echoes the key back is
redacted before it is raised. A regression test asserts that the key in the
environment reaches the outgoing `apikey` parameter (with a fake key, never a
real one).

With the credential fixed, the next run failed with

> "vendor_daily: SPY sessions are not in ascending order"

`TIME_SERIES_DAILY` lists its bars **newest-first**, but the shared price-source
validator requires **oldest-first**, and the parser handed the vendor's raw order
straight to it. The parsed frame is now sorted by session label before
validation. The sort is stable, so a genuine duplicate still reaches the
validator's duplicate check rather than being merged away, and no OHLCV value is
changed — each bar stays attached to its own date. The reorder is a sort, not a
reversal. The old test fixture happened to build its body in ascending order, so
it never exercised the vendor's real ordering; the new tests build the body
newest-first and would fail without the fix.

A new diagnostic reports a deployment's credential state without revealing the
secret — `python -m tradingagents.paper_experiment diagnose --source
alpha_vantage` prints whether the env var is present and non-blank and whether
the request would carry `apikey`, with parameter *names* only and never the
value, any part of it, or its length. `--root` is now optional and required only
by the commands that act on an experiment directory.

### Phase 11 — real market-data providers

Adds real vendor price sources (`paper_experiment/providers/`) so a paper
experiment can run on live market data, still with no live trading path. A
provider fetches data, never an order; every provider declares its own
provenance; and a vendor failure stops the run rather than silently substituting
historical data.

Added:

- **Alpha Vantage daily OHLCV** (`providers/alpha_vantage.py`): the free
  `TIME_SERIES_DAILY` function (`outputsize=compact`), fetched once and validated.
  The premium `TIME_SERIES_DAILY_ADJUSTED` endpoint is deliberately not used, and
  `outputsize=full` is refused unless the caller explicitly accepts a paid call.
  A quota or bad-key notice that arrives inside an HTTP 200 body is classified
  (`ProviderRateLimitedError` / `ProviderAuthError`) and never parsed as data; a
  bar whose date is older than the staleness window raises
  `ProviderStaleDataError`.
- **Finnhub quotes** (`providers/finnhub.py`): the free `/quote` endpoint, which
  returns a current-day quote only. Finnhub's historical candle endpoint is
  premium, so this source is a `vendor_quote` that *refuses* every historical
  request (`bars`/`window`/`sessions`/`covers` raise) and can never stand in for a
  bar history.
- **A shared, bounded, redacting HTTP layer** (`providers/base.py`): one timeout,
  a credential scrubbed from any error, an injectable transport (so tests never
  touch a socket), and a bounded retry. It never returns a partially parsed body
  and never swallows a failure.
- **Explicit source selection with a logged fallback** (`providers/selection.py`):
  `select_source(requested, fallback=…)` switches only to a named fallback and
  logs it; without one, a source failure raises. An experiment runs on exactly one
  source's bars, so a snapshot never mixes two vendors, and the CLI refuses to
  resume a vendor run on the frozen dataset.
- **Source provenance** (`sources.SourceProvenance`): name, kind, current-market
  flag, symbol, fetch time, last bar date, timezone, adjusted flag, dataset
  window, snapshot id and content hash, recorded in the experiment state, the
  journal's creation entry and the report's `data` block.
- **A shared frame validator** (`sources._validate_frame`): every source's bars
  pass through one place that refuses NaN, infinity, a non-positive price, a high
  below low, an open/close outside the range, a duplicate or out-of-order session,
  and a non-numeric column. `FramePriceSource` factors out the validation, session
  set and point-in-time window shared by `StaticPriceSource` and the vendors.
- CLI `--source`/`--fallback` and a `sources` command; `docs/MARKET_DATA_PROVIDERS.md`
  stating, with citations, what each provider can and cannot provide on its free
  plan.

Changed:

- `current_market_source_available()` now returns `True` (a current-market source
  exists), but `READY FOR LIVE PAPER` additionally requires a source that can
  serve a session window, so a quote-only source still cannot reach it. The
  existing Phase 10 test was tightened rather than weakened to assert this.
- The experiment's session timestamp is now source-overridable (a vendor's own
  timezone), still a pure function of the session so signal ids stay stable.

### Phase 10 — prospective paper experiment

A resumable, prospective paper experiment (`paper_experiment/`) records an LLM
strategy's decisions forward in time and evaluates them as later data arrives.
It adds no live trading path and reuses the existing book, risk engine, simulator,
ledger, snapshot, record and metrics rather than reimplementing any of them.

Added:

- A durable, append-only, **hash-chained journal** (`paper_experiment/journal.py`).
  Every entry carries the previous entry's hash, so removing, reordering or
  editing one breaks the chain and `ExperimentJournal.load`/`verify` raise instead
  of trusting the file. Writes are atomic (temp file + `os.replace`), and an entry
  with an `event_id` already present is not appended twice, so a re-run after a
  restart cannot duplicate an event. Every payload is scanned for
  credential-shaped values before it is written.
- An **experiment cycle** (`paper_experiment/experiment.py`) that separates three
  times on every decision: `decision_session` (T), `execution_session` (T+1's
  open) and `evaluation_session` (T+1+lag). A decision made from T's close is
  re-dated to T+1 before it reaches the paper session, so no fill can be read as
  happening at the decision's own close. Settlement runs on every session while a
  decision is only *taken* on a cadence session, so a decision is still executed
  at its own T+1 open even when T+1 is not itself a decision session.
- A **versioned experiment state** (`session_state.json`) that restores the paper
  book, every decision's lifecycle and the processed-session set, so a restart
  resumes without reprocessing a session or duplicating a fill. A decision's full
  `DecisionRecord` is journalled, so execution rebuilds it exactly rather than
  reconstructing it from prose.
- **Price sources with explicit provenance** (`paper_experiment/sources.py`).
  `FrozenPriceSource` reads the committed historical dataset point-in-time and
  reports `is_current_market = False`; `StaticPriceSource` is a validated,
  in-memory series for tests. `current_market_source_available()` returns `False`,
  so the experiment cannot claim live readiness on frozen data. A malformed bar
  (NaN, infinity, non-positive, high below low) is refused at construction.
- A **config frozen before results** (`paper_experiment/config.py`), with a
  content hash recorded on the journal. `execute_on_next_open` is forced `True`:
  filling a decision at the close it was made from is look-ahead, so the
  configuration is refused rather than defaulted.
- An **honest report** (`paper_experiment/report.py`): sessions processed, decision
  counts by state, cash/equity/positions, return and drawdown, the cost model and
  its assumptions, the data source and its provenance, and every blocker. A metric
  below `min_trades_for_metrics` is `None` (rendered `N/A`), never `0.0`, with a
  `sample_limited` flag and the sample size shown.
- A **CLI** (`python -m tradingagents.paper_experiment start|run|report|verify`)
  and a **real-inference seam** (`paper_experiment/inference.py`) that makes one
  bounded request per session with `max_retries=0`, checks the credential before
  the call, and raises to a *blocked* decision when there is none.
- `tests/test_phase10_paper_experiment.py` (68 tests): journal chaining and
  tamper detection, restart-equals-no-restart, duplicate processing, blocked
  decisions, fixture-vs-genuine labelling, look-ahead timing, malformed data,
  multi-symbol recovery, and the CLI.

Changed:

- Nothing existing. The paper session, simulator, ledger, risk engine, snapshot,
  record and metrics are consumed as-is.

### Phase 9 — first genuine inference attempt (blocked)

The capture path added above is exercised by the offline suite, but no genuine
provider request had been made. Phase 9 attempted the first one and stopped at
the credential gate: this environment holds no provider key (`GROQ_API_KEY` or
any other) and no local OpenAI-compatible endpoint answers, so no request was
made and no record was fabricated. The result is recorded as **PROVIDER BLOCKED**,
not as a captured decision.

The offline proof that the genuine-inference discriminator is real — not merely
asserted — is `tests/test_phase9_genuine_inference.py`: it drives the **real**
provider client through an `httpx.MockTransport`, so the observer fires because
the client dispatched a LangChain `on_llm_end` callback, and it asserts a caller
cannot force the `llm_inference` label, a missing credential blocks before any
request, a provider failure yields no record, and replay makes zero provider
calls. No live trading path is added.

### Fixed

- A non-finite price, quantity or fee is refused by the ledger. A NaN passed the
  `price <= 0` and `fee < 0` guards (both are false for NaN) and was applied,
  turning cash, equity and the exposure check into NaN for the rest of the run;
  infinity passed the same guards. Both are now rejected before any state change.
- A non-finite market price is rejected by the execution engine instead of
  producing an order marked `FILLED` with a NaN fill price.
- An order that has already reached a terminal state is not re-evaluated.
  Submitting a filled order a second time filled it twice, because the ledger
  cannot distinguish a re-submission from a second genuine order.
- A strategy's context is bounded at its session, so `context.feed.bar(symbol,
  tomorrow)` no longer returns tomorrow's bar. The look-ahead guarantee was a
  property of `context.history` only, and one attribute deeper the feed answered
  for any date.
- A bar with a non-finite price is dropped when the feed is built rather than
  raising mid-backtest on an arbitrary later session.
- A paper-mode `Quote` refuses a non-finite price.
- An equity-relative position cap is re-checked at the fill price. The engine
  sized an order against the price the decision saw and filled it at the next
  session's open, so an up-gap left a position worth more than the cap allowed: a
  20% cap on a 100,000 book sized 200 units on a 100 close and filled them at a
  200 open, a position worth 40% of equity. `RiskEngine.review_fill` now re-runs
  the caps against the price the fill will actually get, and both the backtester
  and the paper session call it immediately before submitting a buy. A fill that
  would breach a cap is reduced to the largest size that holds, and refused only
  when no positive size remains. The same defect existed in paper mode, where
  spread and slippage move the fill off the quote, and is fixed there too.
- A paper session's `SessionStep` reports the quantity that was filled, not the
  quantity that was requested, so a reduced fill is not reported as a larger
  position than the ledger holds.
- An order sized to the available cash is reduced, not refused, when the fill
  price or the fill fee pushes its all-in cost past the cash. The simulator
  rejected the whole order with `INSUFFICIENT_CASH`, so a buy sized to the book
  was lost entirely once a commission or an up-gap applied; the fill-time
  re-check now caps the size to the fee-inclusive affordable notional
  (`CostModel.affordable_notional`), and both the backtester and the paper
  session pass it. A size that affords nothing is still refused.
- A bar whose OHLC values are internally inconsistent (`high < low`, or an
  open/close outside the range) is dropped when the feed is built rather than
  raising mid-backtest. `Bar` validated the row, but `_normalize` did not, so a
  vendor row with `high < low` survived construction and aborted a run on an
  arbitrary later session while it held a position. The construction filter now
  mirrors `Bar`'s own validator.
- The report's mean exposure and turnover are measured rather than always zero.
  `build_report` computed metrics with a direct `compute_metrics` call that
  passed no exposure series and no traded notional, so a fully invested run
  reported `exposure = 0` and `turnover = 0`; it now uses `metrics_from_result`.
- The `maker_fee_bps` and `taker_fee_bps` cost fields are charged. They were
  declared and documented but never read, so a schedule that set them collected
  nothing; the execution path now charges `CostModel.fee_for_fill`, which adds
  the maker rate for a resting LIMIT fill and the taker rate for a MARKET fill.
- A price-source exception during `PaperSession.work_resting` no longer leaves the
  resting queue half-advanced. The loop rebuilt `self.resting` only at the end, so
  an exception for one symbol propagated out with an already-filled order still
  queued (marked `FILLED`) and the book moved ahead of the queue. The loop is now
  exception-safe: a failed read leaves that order resting, the remaining orders
  are still worked, the book is never touched by a failed read, and the first
  failure is re-raised after the loop so it is never swallowed.
- A closed market no longer rejects a resting GTC order. `work_resting` passed the
  quote's `is_open` to the simulator, so `is_open=False` produced a terminal
  `MARKET_CLOSED` rejection for an order whose limit simply had not been reached;
  a closed market is now a reason to wait, and the order stays live.
- A partially filled resting GTC order keeps its remainder. `on_signal` queued an
  order only when its status was exactly `ACCEPTED`, so a GTC limit that filled
  part of its size was never queued and its remainder was silently lost; an
  `ACCEPTED` or `PARTIALLY_FILLED` GTC order is now queued.
- A resting buy is re-checked against the risk caps at its projected fill price
  before it fills, so a gap to the limit cannot leave a position past its cap. The
  size is reduced to the largest that holds, and the remainder is rejected with
  `RISK_LIMIT` (recorded) only when no positive size remains.
- `PaperSession.work_resting` records the fee each fill actually charged rather
  than `0.0`, and rolls the day's opening equity from the quote's timestamp so a
  session whose only activity is resting fills still has a current daily-loss
  baseline.
- A paper session's `resting_orders` snapshot reports each order's real status,
  filled and remaining quantities, weighted average fill price, time in force and
  expiry, so a snapshot cannot claim an order is `FILLED` when it is `ACCEPTED` or
  `PARTIALLY_FILLED`. The snapshot also carries a `schema_version`, `current_time`,
  `day` and `day_start_equity`.
- `ExecutionSimulator.mode` is resolved to the `ExecutionMode` enum at
  construction rather than stored as given, so a caller that passes the string
  form (as a restored session does) still gets a value the rest of the code can
  compare and serialise.
- A paper `Quote` whose `low` is above its `high`, or whose range is non-finite,
  is refused rather than used to decide a LIMIT order's reachability. A quote
  source that answers with the wrong symbol for the request is refused rather than
  filled against.
- A price-source failure in `on_signal` no longer consumes the signal: the
  `signal_id` is recorded as processed only once the quote is in hand, so a data
  outage does not mark a decision that was never acted on, and the caller can
  retry it.
- `Ledger.from_state` validates the persisted book rather than trusting it, and
  `Order` refuses an impossible lifecycle at construction. A hand-edited or
  corrupted state file carrying a negative cash balance or holding, a non-finite
  figure, a `seq` counter rolled back below the event log (which minted duplicate
  sequence numbers on the resumed run), or a resting order marked `FILLED` with a
  remainder — or `ACCEPTED` with a fill — used to resume with silently wrong
  accounting. Every invariant the live ledger maintains is now re-checked, and
  `PaperSession.restore` raises `PaperStateError` on a violation instead of
  resuming from it.
- A paper session shares one per-session liquidity budget across the orders it
  works against a symbol. `work_resting` computed the volume allowance per order,
  so two resting orders for one symbol each filled the whole bar — 800 units where
  the backtester fills 400. The allowance is now computed once per symbol per
  session and drawn down in queue order, so paper and backtest agree.
- The paper session's clock only moves forward. `_roll_day` set the day from any
  quote it was handed, so an out-of-order (earlier-dated) quote moved the day and
  the daily-loss baseline backwards; a quote on a date earlier than the session
  has already seen is now ignored for the day roll.

### Notes

- Prices are split-adjusted, so a backtest reflects price return only; dividends
  are not credited as cash. Documented rather than hidden.
- LLM calls are not deterministic, so a run's signals carry their own reasoning
  and a content-derived id; the deterministic layers around them are reproducible
  from the run configuration.
- The fill-time cap re-check uses the *projected* fill price, which for a MARKET
  order is the price the simulator will use, so the two agree. An order type that
  filled at a price not derivable from the bar would need the re-check to reconcile
  after the fill. Recorded in `docs/PR1_INDEPENDENT_REVIEW.md`.
- Paper-session recovery restores the state that was last saved, and there is no
  write-ahead log or automatic checkpoint: the caller decides when to call
  `save_state`, and any step after it is lost on a crash. The recovery mechanism
  proves a restored session continues with identical accounting and no duplicated
  fills; it does not provide the durability policy. Documented in
  `docs/PAPER_TRADING.md`.
- The claim recovery supports is "deterministically recovers from a valid saved
  state", not "crash-safe". A crash-safe claim would need a durable checkpoint,
  which this release does not have. Recorded in `docs/ADVERSARIAL_AUDIT.md`.


## [0.7.1] — 2026-10-27

Decisions are scored only on final closes and settle across long market holidays, a model's unpadded date no longer reads filings made after the run date, and fixes to portfolios, prices, SEC statements, reports, Azure and Claude settings.

### Upgrading from 0.7.0

- Docker images no longer include `.env.enterprise`. For Azure OpenAI in Docker, put its keys in `.env`.

### Fixed

- A date a model passes without zero padding, such as `2024-7-5`, is read as that date. It could compare as later than a filing made after the run date, and SEC EDGAR then served that filing. (#1488)
- A decision is no longer scored on the bar of a session still trading, and one made before a market closure longer than a week, such as a national holiday, settles once trading resumes instead of staying pending. (#1548, #1539)
- A backtest reports the failures of its last settlement pass, and two started in the same second no longer share one run. (#1557, #1549)
- A holding listed under a broker's spelling (`BTCUSDT`, `700.HK`) is matched to the instrument analyzed, and a portfolio number that is not finite is refused when the file loads. (#1556, #1509)
- Prices keep their significant digits: a sub-cent close no longer prints as 0.00, and a currency pair keeps its pips. (#1567)
- SEC EDGAR statements list one row per period, so a model no longer reads a figure against the wrong quarter across a long history, and a figure under one million no longer prints as 0. (#1517, #1589)
- Saving a report to a folder used before removes the earlier run's sections. (#1566)
- A date window that ends before it starts is sent back to the model instead of returning no data. (#1595)
- A statement frequency other than annual or quarterly is sent back to the model instead of being served annual figures under that word. (#1593)
- The memory's most recent lessons, and the entries rotation keeps, follow decision dates, so a backfill of older dates no longer displaces them. (#1603)
- Prediction-market odds near 0% or 100% show their distance from it to two significant digits. (#1598)
- Azure OpenAI receives `max_tokens` and a configured endpoint. (#1584, #1600)
- Claude's effort setting is decided correctly for dated model IDs such as `claude-sonnet-4-20250514`. (#1585)
- The CLI checks a ticker with the same rule as the run, so `XAUUSD+` is accepted and `..` is refused at the prompt. (#1604)
- Docker images no longer include `.env` files from any folder.
- Tests that failed on Windows and macOS for platform reasons are fixed. (#1506, #1485, #1541)

### Contributors

[@alanfate3](https://github.com/alanfate3), [@aranlucas](https://github.com/aranlucas), [@BichengWang](https://github.com/BichengWang), [@Bortlesboat](https://github.com/Bortlesboat), [@hoffman988](https://github.com/hoffman988), [@jhill-cmd](https://github.com/jhill-cmd), [@muhib-karim](https://github.com/muhib-karim), [@pei711](https://github.com/pei711), [@peiji1981](https://github.com/peiji1981), [@rudycelekli](https://github.com/rudycelekli), [@Shubham-Padkonde](https://github.com/Shubham-Padkonde), [@stevexyz2003](https://github.com/stevexyz2003), [@SuperMarioYL](https://github.com/SuperMarioYL), [@sytfly](https://github.com/sytfly), [@yuv009](https://github.com/yuv009).

## [0.7.0] — 2026-10-16

Runs on your ChatGPT plan, backtests scored from the first close after each decision, a note at start-up when a newer release is out, and a wheel attached to each GitHub release.

### Added

- **Your ChatGPT plan.** The `chatgpt` provider runs TradingAgents on a ChatGPT Plus or Pro plan instead of an API key, through OpenAI's sign-in for open-source apps. Choose "Subscription" in the CLI's provider menu; for unattended runs, Docker and Python (`llm_provider="chatgpt"`), sign in once with `tradingagents login chatgpt`. `tradingagents logout chatgpt` revokes the sign-in. The CLI lists the plan's models, and reasoning effort applies as it does for OpenAI. (#1194, #1475)
- **What's new at start-up.** The interactive CLI mentions a newer release, with the command that upgrades this install when it can tell how TradingAgents was installed, and notices from Tauric Research for the running version. It checks at most once a day in the background, so a run never waits on it; runs without a terminal never check, and `TRADINGAGENTS_NO_UPDATE_CHECK=1` turns it off.
- **Release downloads.** Each GitHub release carries the source archive and a wheel, so a version installs with `pip install` from its release page without cloning.

### Upgrading from 0.6.0

- `cryptography` 42 and `packaging` 24 or later are installed.
- Decisions are now scored from the first close after their date. Outcomes already settled in a memory log keep the earlier entry, the close of the decision date, so a backtest resumed with `run_id` mixes the two; start a new run to compare like with like.

### Fixed

- A decision is no longer scored against news it could not have traded on: a run dated *d* reads news published after the close on *d*, so the memory log and backtests enter it at the first close after *d*, not the close of *d*. (#1518)
- A run without a terminal, such as a scheduled job or CI, no longer contacts the announcements server before it starts.
- A model tier on another provider keeps its configured reasoning effort, thinking level or Claude effort when the main provider has none. (#1440)

### Contributors

[@LeonWTW](https://github.com/LeonWTW), [@OleksanderSS](https://github.com/OleksanderSS), [@sovorn-c](https://github.com/sovorn-c).

## [0.6.0] — 2026-10-03

HTML reports, a provider per model tier, past decisions settled for every ticker while the analysts work, and company news read while Yahoo's news feed is down.

### Added

- **HTML report.** A saved report also includes `complete_report.html`, the report as one page with its sections listed beside the text, that reads well in a browser, on a phone and in print. Answering the save question at the prompt also asks about the page and can open it in your browser; `--no-html` or `save_reports(..., html=False)` skips it. The page is self-contained: it loads nothing and runs nothing. (#1419)
- **A provider per model tier.** The quick and deep models can run on different providers, for example the managers on Claude while the analysts run on OpenAI: set `deep_think_provider` or `quick_think_provider` (`TRADINGAGENTS_DEEP_THINK_PROVIDER`, `TRADINGAGENTS_QUICK_THINK_PROVIDER`) with the tier's model. The report header names each tier's provider. (#1440)
- **Past decisions settle alongside the analysts.** While a run's analysts work, every logged decision whose holding period has passed is settled, for any ticker, not only the one being analyzed. If settling fails, the run goes on and the report header says so. (#1428, #1445)
- **Crypto on Reddit.** For a crypto ticker the Sentiment Analyst reads r/CryptoCurrency and r/CryptoMarkets, plus r/Bitcoin, r/ethereum or r/solana for those coins, instead of the stock subreddits. (#1461)
- `settle_all_pending()` settles every due decision without running an analysis, for a scheduled job; it and `settle_pending()` return the decisions settled and any that could not be. (#1445)
- The report header names the run's rating.
- Euro-area macro series by name: `ecb_deposit_rate`, `ecb_main_refi_rate`, `euro_hicp`, `euro_core_hicp`, `euro_real_gdp`, `germany_10y`, `france_10y` and `eur_usd`. (#1465)

### Models

- A Gemini call times out after 600 seconds, as OpenAI and Anthropic calls do, so a stalled call no longer hangs the run. (#1417)

### Upgrading from 0.5.2

- markdown-it-py 4.0 or later is installed.
- A saved report also writes `complete_report.html`; pass `--no-html`, or `html=False` to `save_reports()`, to keep only the Markdown.
- A run no longer settles its ticker's decisions before it starts; the Memory Log step settles every ticker's during the run.
- `settle_pending()` returns a `Settlement` with `settled` and `failed` (was `None`).
- Checkpoints from 0.5.2 are not resumed.

### Fixed

- Company news is read from Yahoo search while Yahoo's news feed returns nothing. When Yahoo has no article about the symbol, its news is reported unavailable, not absent, and a configured next news vendor is tried. (#1467, #1469)
- A rating quoted from someone else, such as "Consensus rating: Buy", no longer replaces the Portfolio Manager's own. (#1466)
- A Yahoo news article with a null field no longer loses the ticker's news. (#1458)
- A start date a model writes with stray characters is sent back to it to correct, instead of failing the news or price request. (#1476)
- A request Alpha Vantage rejects is reported as unavailable, not served as data; the API key is kept out of every Alpha Vantage error. (#1442)
- Yahoo fundamentals print each money figure in its own currency: London prices in GBp, market cap and per-share figures in GBP, and an ADR's income figures in its reporting currency, such as TWD for TSMC. (#1456)
- Jev screening uses the endpoint `TYPESAFE_BASE_URL` names, so it can run through OpenRouter. (#1416)
- Runs sharing a memory log settle each decision once, and an interrupted settlement keeps the outcomes it finished.
- The report header names the sentiment analyst `sentiment`, as the CLI does.
- `requirements.txt` says the dependencies are declared in `pyproject.toml`.

### Contributors

[@amitbhansali](https://github.com/amitbhansali), [@AyushJain-601](https://github.com/AyushJain-601), [@BichengWang](https://github.com/BichengWang), [@byte7m](https://github.com/byte7m), [@Dangerboi](https://github.com/Dangerboi), [@davidalmeida90](https://github.com/davidalmeida90), [@Jackzigen](https://github.com/Jackzigen), [@jhill-cmd](https://github.com/jhill-cmd), [@Recnelis0](https://github.com/Recnelis0), [@seaaxe](https://github.com/seaaxe), [@vaegev](https://github.com/vaegev).

## [0.5.2] — 2026-09-29

Parallel analysts, unattended CLI runs, reports that record what produced them,
and past runs served only what was known on their date.

### Added

- **Parallel analysts.** The selected analysts run at the same time, so the analyst phase takes about as long as the slowest one; the CLI shows each analyst's progress and time as it finishes, and the debate starts when all reports are in. (#1255, #433)
- **Unattended CLI runs.** Flags answer the per-run questions, and `TRADINGAGENTS_*` variables the rest, so a scheduled job or script asks nothing: `tradingagents --ticker NVDA --date 2026-09-26 --analysts market,news --save --no-show`. Without a terminal, a missing answer stops the run before it starts and names what to set. (#1127, #1133)
- **Reports record what produced them.** Each report opens with the analysis date, version, provider, models, analysts, rounds and data vendors; the state log keeps them as `run_settings`. (#752, #1197)
- **Docker data folder.** `TRADINGAGENTS_DATA_DIR` keeps results, reports and the memory log in a host folder. (#865)
- **A cap on an analyst's tool calls.** After `max_tool_rounds` rounds (default 20, `TRADINGAGENTS_MAX_TOOL_ROUNDS`) the analyst writes its report from what it has. (#1420)
- `tradingagents backtest` shows each cell as it starts and the `--run-id` to continue; `--analysts` accepts `sentiment`.
- Benchmarks for Taiwan, Korea, Singapore and the main European exchanges. (#1392)
- Python 3.14 support; the Docker image runs Python 3.13.

### Models

- `deepseek-flash` is treated as a thinking model, so structured output works. (#1388)

### Upgrading from 0.5.1

- Python 3.11 or later is required; pandas 3 is installed.
- Minimum versions: yfinance 1.7.0, langchain-core 1.6.5, langchain-openai 1.6.6, langchain-anthropic 1.7.4, langchain-google-genai 4.4.0, langgraph 1.2.12, langgraph-checkpoint-sqlite 3.1.1, langchain-aws 1.7.9 (`bedrock` extra).
- Statements default to SEC EDGAR, then Yahoo (`fundamental_data = "sec_edgar,yfinance"`).
- A past-dated run gets no Yahoo or Alpha Vantage statements or insider trades, which carry no filing date, and only the company's current name, marked as such.
- The memory log is the `tradingagents.memory` package (was `tradingagents.decision_log`); settlement and reflection moved there from `tradingagents.graph`, and `Reflector` is no longer exported from `tradingagents.graph`.
- `process_signal` is removed; the rating is `propagate()`'s second value or `final_state["final_rating"]`.
- A rating is read only from its `Rating:` label; a free-text decision without one is `REVIEW`.
- Data-layer functions take `as_of_date=` (was `curr_date=`); `build_instrument_context` and `resolve_instrument_context` take `trade_date=`.
- `VendorRateLimitError` is `VendorUnavailableError`.
- Data tools take the instrument from the run's state; tool calls no longer carry `symbol` or `ticker`.
- State log: `trader_investment_plan` (was `trader_investment_decision`), `judge_decision` removed, `final_rating` and `run_settings` added.
- Checkpoints from 0.5.1, or saved under different settings, are not resumed.

### Fixed

- Data tools serve the run's instrument, even when the model passes another symbol.
- A failed vendor request is reported as unavailable, not as a missing or delisted symbol; one vendor's "no data" no longer speaks for a chain in which another failed. (#1386)
- The rating is the Portfolio Manager's own, not one its thesis quotes (#1383); its line keeps an English label in other output languages (#1435).
- Backtest cells given twice run once (#1415), and settlement compares a decision with its benchmark over the same dates.
- A snapshot for a symbol with no prices no longer ends the run.
- Concurrent runs no longer drop memory log entries.
- SEC EDGAR quarterly figures filed only year to date are served, labelled with their span.
- Yahoo dividend yield and debt-to-equity print as percentages. (#1414)
- The FRED change line names the dates it spans. (#1397)
- `stream_run` serves the graph's own config to its tools.
- The test suite stays off the network and out of the user's files, and passes in any order; integration tests run with `-m integration`. (#1395)

### Contributors

[@1Wizzy](https://github.com/1Wizzy), [@BichengWang](https://github.com/BichengWang), [@Chaoqi31](https://github.com/Chaoqi31), [@davidalmeida90](https://github.com/davidalmeida90), [@djconnexion77](https://github.com/djconnexion77), [@fuzing](https://github.com/fuzing), [@hailampy123](https://github.com/hailampy123), [@hesam-shams](https://github.com/hesam-shams), [@HEYALT](https://github.com/HEYALT), [@Jackzigen](https://github.com/Jackzigen), [@kagura-agent](https://github.com/kagura-agent), [@loulanyue](https://github.com/loulanyue), [@mannubaveja007](https://github.com/mannubaveja007), [@olivergpt](https://github.com/olivergpt), [@rita112025-cpu](https://github.com/rita112025-cpu), [@shuyan-code](https://github.com/shuyan-code), [@SingTheCode](https://github.com/SingTheCode), [@sjq597](https://github.com/sjq597), [@xiaodu55](https://github.com/xiaodu55), [@Youholdme](https://github.com/Youholdme).

## [0.5.1] — 2026-09-24

A package layout organised by what each module holds, social posts screened by
TypeSafe's Jev when a key is set, GPT-6 Sol and Luna as the default models, and
fixes to run isolation, SEC EDGAR statements and historical runs.

### Upgrading from 0.5.0

Some modules moved, and the old import paths are gone. Update imports as follows:

- `tradingagents.dataflows.interface` is `tradingagents.dataflows.router`, and `dataflows.symbol_utils` is `dataflows.symbols`. `dataflows.utils` is gone: `get_current_date` is in `dataflows.date_window`, `safe_ticker_component` in `dataflows.symbols`, and `get_scrubbed` and `vendor_reachable` in `dataflows.net`.
- Vendor modules live under `tradingagents.dataflows.vendors`: `yahoo` (`ohlcv`, `market`, `fundamentals`, `news`, `snapshot`, from the former `stockstats_utils`, `y_finance`, `yfinance_news` and `market_data_validator`), `alpha_vantage` (a package, from the `alpha_vantage_*` modules), and `sec_edgar`, `fred`, `polymarket`, `reddit`, `stocktwits`.
- `tradingagents.agents.utils` is gone: the agent tools are in `agents.tools`, and `agent_utils`, `agent_states`, `rating` and `structured` are `agents.context`, `agents.state`, `agents.rating` and `agents.structured`.
- The decision log is `tradingagents.decision_log` (was `agents.utils.memory`), and `cli.utils` is `cli.prompts`.
- `backtest.summarize` takes a `run_backtest` result or the path of a decision log, in place of a `TradingMemoryLog`.
- Removed: `SignalProcessor` (the rating is parsed by `process_signal`), the `create_social_media_analyst` alias (use `create_sentiment_analyst`), `symbol_utils.is_yahoo_safe`, the unused `project_dir` config key, and the graph attributes `curr_state`, `ticker` and `log_states_dict`, which held the previous run's state.

### Added

- **Jev post screening.** With `TYPESAFE_API_KEY` set, TypeSafe's Jev reads each StockTwits and Reddit post the Sentiment Analyst fetches: posts that are not about the company are dropped, and each source opens with a count of the rest by stance. Without the key nothing changes. (#1376)
- B3 tickers (`.SA`) are benchmarked against the Ibovespa. (#1366)

### Models

- GPT-6 Sol and GPT-6 Luna are the default deep and quick models, and Claude Opus 5.5 replaces Opus 5 in the picker. Opus 5 and GPT-5.4 Mini remain valid model IDs.

### Fixed

- Several graphs in one process each read their own data vendors. (#1369)
- SEC EDGAR cash flow statements find capital expenditure for filers that moved it to purchases of productive assets (NVIDIA since fiscal 2022, Amazon since 2016), whose recent periods read as empty. (#1370)
- SEC EDGAR annual statements list fiscal years only.
- A historical run is no longer told today's date, or a date after the run, in coverage notices and the instrument context.
- The Fundamentals Analyst can call the insider transactions tool.
- An unreachable Yahoo on insider transactions is reported as unavailable, not as a symbol without data.
- A graph reused across runs, as in a backtest, no longer keeps every run's full state.
- A checkpointed CLI run says whether it resumed or started fresh.
- A blank path variable (`TRADINGAGENTS_RESULTS_DIR` and the like) keeps the default path.
- StockTwits messages reach the prompt as plain text rather than HTML-escaped.
- The test suite is independent of the developer's `.env`, time zone and network. (#1368, #1372)

### Contributors

Thanks to everyone who reported these or sent a fix:

[@codify88](https://github.com/codify88), [@davidalmeida90](https://github.com/davidalmeida90), [@duongylinh](https://github.com/duongylinh), [@jccl2](https://github.com/jccl2), [@yuina368](https://github.com/yuina368).

## [0.5.0] — 2026-09-18

Point-in-time integrity across every dated path, decisions that are recorded as
they were made, backtesting over a grid of tickers and dates, the caller's
portfolio as run input, and SEC EDGAR fundamentals served as filed.

### Highlights

- **Fundamentals as filed.** SEC EDGAR serves US company statements as they stood on the run's date: a period that has ended but has not been filed is not served, and a figure restated later still reads as first reported. Keyless, opt-in via the vendor chain.
- **Backtesting.** `run_backtest` runs the pipeline over a ticker and date grid into its own decision log, and `summarize` scores the settled cells; `tradingagents backtest` does the same from the CLI.
- **Portfolio context.** `propagate(..., portfolio=...)` and `--portfolio` let the trader, risk and portfolio agents size against real holdings. A run without one is never treated as a flat book.
- **Decisions are recorded as made.** An unreadable decision is flagged for review everywhere instead of becoming a tradeable Hold, and a rating argued against is no longer read as the call.

### Point-in-time and honest attribution

- Dated tools take the run's date from graph state, so an omitted or later date cannot reach a vendor. (#1331, #1319, #1118)
- Insider filings and prediction-market odds are bounded by the run date; insider rows state that a trade becomes public when its Form 4 is filed.
- A feed that never observed a window reports it as unavailable rather than as an absence, across news, Reddit and StockTwits.
- The resolved company identity says when it describes today rather than the run date.
- The verification snapshot quotes the prices the vendor reported, never a gap-filled value.
- A vendor failure is a vendor failure: yfinance raises instead of returning its errors as text, an outage is not reported as a company with no data, and a chain where every vendor is unavailable says so instead of ending the run.
- The macro vintage pin is clamped to the vendor's own clock, so a run dated today cannot ask for a vintage it does not have.
- A historical run is not served a present-day company profile by either fundamentals vendor. (#1300)

### Decisions and evaluation

- The labelled rating decides, whatever separates it, and prose naming several ratings is reviewed rather than guessed.
- Decision prompts state the shape of their answer, so a provider without structured output still returns a readable decision.
- A report that was not produced says so, instead of appearing as an empty section.
- Backtest scoring reads the direction each rating claimed: a Sell that fell is a hit, and Hold reports no hit rate.
- The outcome window is configurable (`holding_period_days`), and reflection states the window it judges.
- A settled decision is not logged twice, and a failed reflection no longer stops the next run. (#645)
- The trader states entry and stop levels as prices, so a percentage no longer fails the whole proposal. (#1288)

### CLI

- `tradingagents backtest`, with `--run-id` to continue an interrupted sweep. (#1234)
- The previous run's selections come back as prompt defaults. (#1236, #920)
- A run with no readable rating says so; the live view no longer scrolls; messages that read like Python values are shown. (#649, #784)
- The state log keeps non-ASCII readable. (#1081)

### Data sources

- SEC EDGAR fundamentals vendor (US filers, keyless).
- Hong Kong and Shanghai tickers resolve to the symbols Yahoo serves. (#1342, #957, #1260)
- Reddit is fetched as one combined request per run. (#1286)
- One OHLCV cache file per symbol. (#1330)

### Models

- Current lineups for every provider: GPT-6 Astra and the GPT-5.6 family, Gemini 3.8 Flash, Claude Opus 5 and Fable 5.1, Grok 4.6, DeepSeek Flash, Qwen 3.8, GLM-5.3, MiniMax M3, Kimi K3 and the current Mistral snapshots.
- Every provider accepts a model ID the picker does not list.
- GLM traffic goes to the platform its key belongs to, and Ollama structured output no longer sends a tool_choice it rejects. (#1062)

### Changed

- The memory log records `REVIEW` for a decision with no readable rating, where it previously recorded `Hold`.
- Optional fields the model did not provide are named as such rather than omitted.
- Removed dependencies nothing imports: backtrader, redis, setuptools, langchain-experimental, parsel, tqdm. (#1353, #1070)

### Contributors

Thanks to everyone who reported these or sent a fix:

[@akashkpfreelancer](https://github.com/akashkpfreelancer), [@angziii](https://github.com/angziii), [@anupamme](https://github.com/anupamme), [@AyushKar2005](https://github.com/AyushKar2005), [@bulkypanda](https://github.com/bulkypanda), [@CadeYu](https://github.com/CadeYu), [@chiang21fcb](https://github.com/chiang21fcb), [@dajiaohuang](https://github.com/dajiaohuang), [@dewrama](https://github.com/dewrama), [@DogInfantry](https://github.com/DogInfantry), [@emitov](https://github.com/emitov), [@farukerdem34](https://github.com/farukerdem34), [@flydragon2018](https://github.com/flydragon2018), [@fusshell](https://github.com/fusshell), [@Ganesh1729-ui](https://github.com/Ganesh1729-ui), [@gyx09212214-prog](https://github.com/gyx09212214-prog), [@hamzabudeir](https://github.com/hamzabudeir), [@ihsieh31](https://github.com/ihsieh31), [@jaylew20250206](https://github.com/jaylew20250206), [@kaushik-yadav](https://github.com/kaushik-yadav), [@kbnnf](https://github.com/kbnnf), [@kevinkda](https://github.com/kevinkda), [@LudwigJMarx](https://github.com/LudwigJMarx), [@lx7720](https://github.com/lx7720), [@malandrindev](https://github.com/malandrindev), [@mhd325ic-hash](https://github.com/mhd325ic-hash), [@minhdn90](https://github.com/minhdn90), [@miznan](https://github.com/miznan), [@mmssix](https://github.com/mmssix), [@mrbob-git](https://github.com/mrbob-git), [@newnewself](https://github.com/newnewself), [@prithvirajrh](https://github.com/prithvirajrh), [@PyriteResearch](https://github.com/PyriteResearch), [@Rajatendu1](https://github.com/Rajatendu1), [@Recnelis0](https://github.com/Recnelis0), [@Rodvask](https://github.com/Rodvask), [@samhoooo](https://github.com/samhoooo), [@sheiun-xu](https://github.com/sheiun-xu), [@shivsin25](https://github.com/shivsin25), [@SmileShaun](https://github.com/SmileShaun), [@SonnyRajagopalan](https://github.com/SonnyRajagopalan), [@taro0915](https://github.com/taro0915), [@wupengbo125](https://github.com/wupengbo125), [@wxggzz](https://github.com/wxggzz), [@Yixiang-Wu](https://github.com/Yixiang-Wu), [@ZahirBodrike](https://github.com/ZahirBodrike), [@ZHUYAWEI](https://github.com/ZHUYAWEI), [@zkwang616](https://github.com/zkwang616).

## [0.4.0] — 2026-08-31

Look-ahead and point-in-time fixes across the data and memory layers, clearer
decision signals, working CLI checkpoint resume, and the GPT-5.6 / GLM-5.3 models.

### Fixed

- **FRED macro look-ahead.** Historical macro requests were served from today's
  data vintage, leaking later revisions into a backtest; both the observations
  and metadata requests now pin the vintage to the as-of date. (#1275)
- **Social sentiment look-ahead.** StockTwits and Reddit were fetched with no
  date, so a historical run showed today's chatter as if it were from the as-of
  date; the social path is now trimmed to the analysis window, via one shared
  UTC half-open window rule (`dataflows/date_window`) used by news too. (#1220)
- **Memory point-in-time guard.** `get_past_context` returned every resolved
  lesson regardless of the run date; each resolved entry now records the date
  its outcome became known, and a historical run only sees lessons resolved by
  the trade date. (#1251)
- **Premature reflection.** A decision was settled on a partial return if a rerun
  happened before its holding window fully traded; resolution now waits for the
  full window. (#1169)
- **Latest OHLCV bar dropped.** The newest bar with a NaN close was silently
  dropped before the date cutoff, making the previous trading day look like the
  latest; dates are normalized per element (DST- and non-US-market safe) and a
  missing latest close raises rather than falling back. (#1201)
- **Debate opening fabrication.** The first speaker in each debate round rebutted
  an empty opponent response, fabricating the other side; all five debators now
  open with their own case when no opponent has spoken. (#1176)
- **Silent Hold.** An unparseable Portfolio Manager rating (including a fullwidth
  colon) was coerced to a tradeable Hold; it now surfaces a `REVIEW` sentinel,
  with `parse_rating` keeping its silent default for compatibility callers. (#1170)
- **`--checkpoint` was a no-op on the CLI.** Checkpoint setup lived only in
  `propagate()`; the CLI streamed the checkpointer-less graph. The lifecycle is
  now shared, and a resume feeds `None` so LangGraph continues the interrupted
  run instead of duplicating messages. (#1249)
- **DeepSeek via OpenRouter.** `deepseek/<id>` fell through to default
  capabilities and had object-form `tool_choice` forced on it; the official
  namespace is stripped so it reuses the native DeepSeek quirks. (#1199)
- **Trader price grounding.** The Trader saw only the digested plan; it now also
  receives the technical market report so entry/stop levels anchor to real price
  structure. (#1167)

### Added

- **Configurable output-token cap.** `max_tokens` / `TRADINGAGENTS_MAX_TOKENS`,
  forwarded to every provider (Gemini as `max_output_tokens`), so a model that
  emits unbounded reasoning can be bounded instead of hanging. (#1204)
- **Latest models.** Added the GPT-5.6 family (`gpt-5.6` / `gpt-5.6-terra` /
  `gpt-5.6-luna`) and GLM-5.3 (`glm-5.3`, `glm-5.3-flash`). The default models
  are now `gpt-5.6` (deep) and `gpt-5.6-luna` (quick).

### Contributors

Thanks to everyone who reported these or sent a fix:

[@PyriteResearch](https://github.com/PyriteResearch), [@yiran1268](https://github.com/yiran1268), [@fabiolenine](https://github.com/fabiolenine), [@lx7720](https://github.com/lx7720), [@taro0915](https://github.com/taro0915), [@Jaswanth-Sriram-Veturi](https://github.com/Jaswanth-Sriram-Veturi), [@ariesy](https://github.com/ariesy), [@liangzj1999](https://github.com/liangzj1999), [@zkwang616](https://github.com/zkwang616), [@aniketshukla1](https://github.com/aniketshukla1), [@loulanyue](https://github.com/loulanyue), [@hudsonwa](https://github.com/hudsonwa), [@daleselaji-dev](https://github.com/daleselaji-dev), [@wolfoswald777-crypto](https://github.com/wolfoswald777-crypto).

## [0.3.1] — 2026-07-05

Correctness and stability patch: data look-ahead, graph-router crash-safety,
checkpoint identity, crypto sentiment sources, and configurable resilience.

### Fixed

- **Alpha Vantage look-ahead filter now runs.** The fundamentals payload is a
  JSON string, so the dict-only guard skipped filtering and future-dated reports
  leaked into historical runs; parse before filtering. (#1115, @zachthebird)
- **News analyst prompt matches the tool.** The prompt advertised
  `get_news(query, ...)` but the tool takes a ticker; aligned to stop
  hallucinated free-text query calls. (#1116, @shcheuk)
- **Shared debate/risk routers can't crash mid-run.** Both routers return more
  targets than any one edge mapped; every edge now shares the complete path map,
  so a fall-through under prompt/i18n/refactor drift stays routable.
  (#1088, @Fr3ya, @sa7an7, @Sushanth012)
- **Checkpoint resume respects graph shape.** The thread id folds in selected
  analysts, debate/risk depth, and asset mode, so a resume under different
  choices no longer continues the wrong graph. (#1089, @bossjoker1, @Ghraven)
- **Crypto sentiment sources resolve.** StockTwits lists crypto as `<BASE>.X`
  (Yahoo's `BTC-USD` 404s) and Reddit needs the base symbol to match; the social
  path now maps crypto correctly for both. (#1113, @suremadoreai)

### Added

- **Configurable LLM retry budget.** `llm_max_retries` /
  `TRADINGAGENTS_LLM_MAX_RETRIES` is forwarded to every provider, so a transient
  429 burst no longer aborts a run. (#1091, @yanggaome)
- **Bedrock API-key auth.** `AWS_BEARER_TOKEN_BEDROCK` authenticates Amazon
  Bedrock without AWS access keys and takes precedence over an ambient
  `AWS_PROFILE`. (#1103, @praxstack)
- **Latest Claude models.** Added Claude Sonnet 5 (`claude-sonnet-5`) and
  Fable 5 (`claude-fable-5`); effort control now covers the Claude 5 line.

## [0.3.0] — 2026-06-22

Stabilization and extensibility release: a CI gate, a unified verified
data-access contract, a provider and data-vendor registry, and a maintenance
sweep that hardened config precedence, the model catalog, data resilience, and
structured output.

### Added

- **CI gate.** GitHub Actions runs the pytest suite across Python 3.10-3.13,
  strict `ruff`, and a clean-install smoke that imports the package and CLI to
  catch undeclared dependencies. (#994, #197)
- **Provider registry.** OpenAI-compatible providers register as a single spec,
  and a generic `openai_compatible` endpoint covers vLLM, LM Studio, and relays.
  Adds NVIDIA NIM, Kimi, Groq, Mistral, and a native Amazon Bedrock client.
- **Macro and prediction-market vendors.** FRED macro indicators and Polymarket
  event probabilities, surfaced to the news and macro analysts.
- **Programmatic report output.** `TradingAgentsGraph.save_reports()` writes the
  same report tree the CLI produces, for headless and API runs. (#1037)
- **Env-configurable reasoning depth** via `TRADINGAGENTS_OPENAI_REASONING_EFFORT`,
  `TRADINGAGENTS_GOOGLE_THINKING_LEVEL`, and `TRADINGAGENTS_ANTHROPIC_EFFORT`,
  each gated to the models that accept it.

### Changed

- **Verified data-access contract.** Symbol normalization on every vendor path
  (identity, returns, CLI, news); the configured vendor list is the exact
  resolution chain with no silent fallback to unselected vendors; a typed
  `VendorError` taxonomy; look-ahead-safe news windows; stale-OHLCV rejection;
  inclusive yfinance date ranges.
- **Config precedence.** An explicit `TRADINGAGENTS_*` value or CLI flag now wins
  over interactive defaults for debate and risk round counts,
  `--checkpoint / --no-checkpoint`, and the Docker provider profile; invalid
  boolean env values fail loudly. (#975, #976, #977)
- **Current-generation model catalog.** Refreshed provider lineups; retired
  `gpt-4.1`, Claude Sonnet 4.5, and the Gemini 2.5 line.
- **Optional vendors degrade** instead of aborting a run: a failed macro or
  prediction-market lookup returns a no-data sentinel.
- **Analyst prompts lead with the current date** so tool-call date ranges anchor
  to the run date rather than the model's training cutoff. (#836)

### Fixed

- **Instrument identity.** Deterministic ticker-to-company resolution prevents
  wrong-company hallucination, and a verified market-data snapshot grounds price
  and indicator claims. (#814, #830)
- **Social and market data sources.** Reddit RSS-first with 429 backoff,
  StockTwits transport hardening, and Alpha Vantage timeout plus
  key-versus-rate-limit handling.
- **Structured output.** Local OpenAI-compatible servers no longer reject
  object-form `tool_choice`; a thinking model that returns no parsed result falls
  back to free text; null-ish strings in optional price fields coerce to `None`.
  (#1038, #1051, #1057)

### Removed

- The no-op `analyst_concurrency_limit` config knob; parallel analyst execution
  is planned for a later release. (#979)
- The unused committed `uv.lock`. (#1030)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

[@CadeYu](https://github.com/CadeYu), [@Zavianx](https://github.com/Zavianx), [@weijianz-opc](https://github.com/weijianz-opc), [@naltun](https://github.com/naltun), [@brahmasky](https://github.com/brahmasky), [@nik2208](https://github.com/nik2208), [@thieucong98](https://github.com/thieucong98), [@Derekko-web](https://github.com/Derekko-web), [@LukiPrince](https://github.com/LukiPrince), [@Eddieargenal](https://github.com/Eddieargenal), [@Ghraven](https://github.com/Ghraven), [@ms32035](https://github.com/ms32035), [@yting27](https://github.com/yting27), [@nyxst4ck](https://github.com/nyxst4ck), [@KenCheung-AIxFinance](https://github.com/KenCheung-AIxFinance), [@yangyusheng2n](https://github.com/yangyusheng2n), [@fareloj](https://github.com/fareloj), [@haosenwang1018](https://github.com/haosenwang1018), [@octo-patch](https://github.com/octo-patch), [@seifenk](https://github.com/seifenk), [@CaoYuhaoCarl](https://github.com/CaoYuhaoCarl), [@mihailnica10](https://github.com/mihailnica10), [@Dado-hash](https://github.com/Dado-hash), [@Handsomemikezzz](https://github.com/Handsomemikezzz), [@ydhawesome](https://github.com/ydhawesome), [@macd2](https://github.com/macd2), [@AyushKar2005](https://github.com/AyushKar2005), [@wildhuman](https://github.com/wildhuman), [@robert23kim](https://github.com/robert23kim), [@bngness](https://github.com/bngness), [@tedix-rodrigo](https://github.com/tedix-rodrigo), [@malaccan](https://github.com/malaccan), [@rfalken78](https://github.com/rfalken78), [@dengli1971-droid](https://github.com/dengli1971-droid), [@proofconcept39](https://github.com/proofconcept39), [@prasta1](https://github.com/prasta1), [@liximin](https://github.com/liximin), [@jeffhuen](https://github.com/jeffhuen), [@mazar](https://github.com/mazar), [@soyangelromero](https://github.com/soyangelromero), [@CNQQC](https://github.com/CNQQC), [@dovetaill](https://github.com/dovetaill), [@fperdigon](https://github.com/fperdigon), [@gyx09212214-prog](https://github.com/gyx09212214-prog), [@RSXLX](https://github.com/RSXLX).

## [0.2.5] — 2026-05-11

### Added

- **Grounded Sentiment Analyst.** The renamed `sentiment_analyst` now reads
  real Yahoo News, StockTwits, and Reddit data before generating its report,
  replacing the prior flow that could fabricate social posts under prompt
  pressure. (#557, #607)
- **MiniMax provider** with the full M2.x catalog (M2.7 / M2.5 / M2.1 / M2
  plus highspeed variants, 204K context). Dual-region: Global
  (`MINIMAX_API_KEY`) and China (`MINIMAX_CN_API_KEY`).
- **Dual-region Qwen and GLM** with separate keys per region — international
  (`DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`) and China (`DASHSCOPE_CN_API_KEY`,
  `ZHIPU_CN_API_KEY`), selectable via a secondary region prompt. (#758)
- **`TRADINGAGENTS_*` env-var configurability for `DEFAULT_CONFIG`.** Override
  `llm_provider`, deep/quick model IDs, `backend_url`, `output_language`,
  debate-round counts, checkpoint flag, and benchmark ticker via `.env` with
  type-aware coercion (string / int / bool). (#602)
- **Interactive API-key detection in the CLI.** When the selected provider's
  key is missing, the CLI prompts for it and persists the value to `.env`
  so the analysis run continues without restart.
- **Remote Ollama support.** `OLLAMA_BASE_URL` points the CLI and the
  programmatic client at a remote `ollama-serve`. The CLI surfaces the
  resolved endpoint and warns on common malformed inputs. Adds a
  `"Custom model ID"` option for models pulled via `ollama pull`. (#648, #768)
- **Configurable news-fetch parameters** in `DEFAULT_CONFIG` — per-ticker
  article limit, macro headline limit, lookback window, and macro search
  queries. (#606, #683)
- **Configurable alpha benchmark** for non-US tickers. Replaces hardcoded
  SPY with regional indices for `.NS` (^NSEI), `.T` (^N225), `.HK` (^HSI),
  `.L` (^FTSE), `.TO` (^GSPTSE), `.AX` (^AXJO), `.BO` (^BSESN); explicit
  `benchmark_ticker` override available. Eliminates FX drift dominating
  alpha for non-USD listings. (#628, #684)
- **Multi-language output covers every user-facing agent** — researchers,
  risk debators, research manager, and trader, ending the previous
  partial-localization reports. (#575)
- **Model catalog refresh.** OpenAI GPT-5.5 frontier, Anthropic Claude Opus
  4.7, Gemini 3.1 Flash-Lite GA, xAI Grok 4.20, Qwen 3.6 line. Versioned IDs
  only; auto-shifting aliases moved to the `"Custom model ID"` option.

### Changed

- **Sentiment Analyst** is now consistently named across the CLI dropdown,
  status panel, and final reports (previously the backend was renamed but
  the CLI still said "Social Analyst"). The `AnalystType.SOCIAL = "social"`
  wire value is kept for saved-config back-compat.

### Fixed

- **Structured output works on DeepSeek V4 / reasoner and MiniMax M2.x.**
  Those providers reject `tool_choice` per their tool-calling docs; the
  binding flow now skips it automatically via a capability table.
- **`pip install .` installations pick up the project `.env`** when running
  the CLI as a console script. (#747)
- **Reports save end-to-end** — streamed chunks were previously dropped from
  `complete_report.md`. (#719, #736)
- **Ticker prompt preserves exchange suffixes** (`.SH`, `.SZ`, `.SS`, `.HK`,
  `.T`, etc.) for A-share, HK, Tokyo, and other non-US flows. (#770)
- **Docker permission errors** no longer block first-run write to
  `~/.tradingagents/`. (#519, #627, #672, #771)
- **Config state no longer leaks between runs** when sub-dicts are mutated;
  `set_config` partial updates preserve sibling defaults. (#788)
- **`max_recur_limit` config actually applies** — previously read but not
  forwarded to the propagator. (#764)
- **Missing-API-key error** names the exact env var to set. (#680)
- **Quieter startup** — suppressed the noisy upstream
  `LangChainPendingDeprecationWarning` from langgraph-checkpoint; will be
  removed once that package ships its fix.

### Security

- **Ticker path-traversal validation** at every filesystem-path site (cache,
  checkpoint database, results) so a malicious ticker cannot escape its
  intended directory. (#618)

## [0.2.4] — 2026-04-25

### Added

- **Structured-output decision agents.** Research Manager, Trader, and Portfolio
  Manager now use `llm.with_structured_output(Schema)` on their primary call
  and return typed Pydantic instances. Each provider's native structured-output
  mode is used (`json_schema` for OpenAI / xAI, `response_schema` for Gemini,
  tool-use for Anthropic, function-calling for OpenAI-compatible providers).
  Render helpers preserve the existing markdown shape so memory log, CLI
  display, and saved reports keep working unchanged. (#434)
- **LangGraph checkpoint resume** — opt-in via `--checkpoint`. State is saved
  after each node so crashed or interrupted runs resume from the last
  successful step. Per-ticker SQLite databases under
  `~/.tradingagents/cache/checkpoints/`. `--clear-checkpoints` resets them. (#594)
- **Persistent decision log** replacing the per-agent BM25 memory. Decisions
  are stored automatically at the end of `propagate()`; the next same-ticker
  run resolves prior pending entries with realised return, alpha vs SPY, and
  a one-paragraph reflection. Override path with `TRADINGAGENTS_MEMORY_LOG_PATH`.
  Optional `memory_log_max_entries` config caps resolved entries; pending
  entries are never pruned. (#578, #563, #564, #579)
- **DeepSeek, Qwen (Alibaba DashScope), GLM (Zhipu), and Azure OpenAI**
  providers, plus dynamic OpenRouter model selection.
- **Docker support** — multi-stage build with separate dev and runtime images.
- **`scripts/smoke_structured_output.py`** — diagnostic that exercises the
  three structured-output agents against any provider so contributors can
  verify their setup with one command.
- **5-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell) used
  consistently by Research Manager, Portfolio Manager, signal processor, and
  the memory log; Trader keeps 3-tier (Buy / Hold / Sell) since transaction
  direction is naturally ternary.
- **Pytest fixtures** — lazy LLM client imports plus placeholder API keys so
  the test suite runs cleanly without credentials. (#588)

### Changed

- **`backend_url` default is now `None`** rather than the OpenAI URL. Each
  provider client falls back to its native default. The previous default
  leaked the OpenAI URL into non-OpenAI clients (e.g. Gemini), producing
  malformed request URLs for Python users who switched providers without
  overriding `backend_url`. The CLI flow is unaffected.
- All file I/O passes explicit `encoding="utf-8"` so Windows users no longer
  hit `UnicodeEncodeError` with the cp1252 default. (#543, #550, #576)
- Cache and log directories moved to `~/.tradingagents/` to resolve Docker
  permission issues. (#519)
- `SignalProcessor` reads the rating from the Portfolio Manager's rendered
  markdown via a deterministic heuristic — no extra LLM call.
- OpenAI structured-output calls default to `method="function_calling"` to
  avoid noisy `PydanticSerializationUnexpectedValue` warnings emitted by
  langchain-openai's Responses-API parse path. Same typed result, no warnings.

### Fixed

- Empty memory no longer triggers fabricated past-lessons in agent prompts;
  the memory-log redesign makes this structurally impossible since only the
  Portfolio Manager consults memory and only when entries exist. (#572)
- Tool-call logging processes every chunk message, not just the last one, and
  memory score normalization handles empty score arrays. (#534, #531)

### Removed

- `FinancialSituationMemory` (the per-agent BM25 system) and the dead
  `reflect_and_remember()` plumbing; subsumed by the persistent decision log.
- Hardcoded Google endpoint that caused 404 when `langchain-google-genai`
  changed its API path. (#493, #496)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

- [@claytonbrown](https://github.com/claytonbrown) — checkpoint resume (#594), test fixtures (#588), design feedback on cost tracking (#582) and structured validation (#583)
- [@Bcardo](https://github.com/Bcardo) — memory-log redesign (#579), empty-memory hallucination report (#572), encoding fix proposal (#570)
- [@voidborne-d](https://github.com/voidborne-d) — memory persistence design (#564), portfolio manager state fix (#503)
- [@mannubaveja007](https://github.com/mannubaveja007) — structured-output feature request (#434)
- [@kelder66](https://github.com/kelder66) — RAM-only memory issue (#563)
- [@Gujiassh](https://github.com/Gujiassh) — tool-call logging fix (#534), test stub PR (#533)
- [@iuyup](https://github.com/iuyup) — memory score normalization fix (#531)
- [@kaihg](https://github.com/kaihg) — Google base_url fix (#496)
- [@32ryh98yfe](https://github.com/32ryh98yfe) — Gemini 404 report (#493)
- [@uppb](https://github.com/uppb) — OpenRouter dynamic model selection (#482)
- [@guoz14](https://github.com/guoz14) — OpenRouter limited-model report (#337)
- [@samchenku](https://github.com/samchenku) — indicator name normalization (#490)
- [@JasonOA888](https://github.com/JasonOA888) — y_finance pandas import fix (#488)
- [@tiffanychum](https://github.com/tiffanychum) — stale import cleanup (#499)
- [@zaizou](https://github.com/zaizou) — Docker permission issue (#519)
- [@Stosman123](https://github.com/Stosman123), [@mauropuga](https://github.com/mauropuga), [@hotwind2015](https://github.com/hotwind2015) — Windows encoding bug reports (#543, #550, #576)
- [@nnishad](https://github.com/nnishad), [@atharvajoshi01](https://github.com/atharvajoshi01) — encoding fix proposals (#568, #549)

## [0.2.3] — 2026-03-29

### Added

- **Multi-language output** for analyst reports and final decisions, with a
  CLI selector. Internal agent debate stays in English for reasoning quality. (#472)
- **GPT-5.4 family models** in the default catalog, with deep/quick model split.
- **Unified model catalog** as a single source of truth for CLI options and
  provider validation.

### Changed

- `base_url` is forwarded to Google and Anthropic clients so corporate proxies
  work consistently across providers. (#427)
- Standardised the Google `api_key` parameter to the unified `api_key` form.

### Fixed

- Backtesting fetchers no longer leak look-ahead data when `curr_date` is in
  the middle of a fetched window. (#475)
- Invalid indicator names from the LLM are caught at the tool boundary instead
  of crashing the run. (#429)
- yfinance news fetchers respect the same exponential-backoff retry as price
  fetchers. (#445)

### Contributors

- [@ahmedk20](https://github.com/ahmedk20) — multi-language output (#472)
- [@CadeYu](https://github.com/CadeYu) — model catalog typing (#464)
- [@javierdejesusda](https://github.com/javierdejesusda) — unified Google API key parameter (#453)
- [@voidborne-d](https://github.com/voidborne-d) — yfinance news retry (#445)
- [@kostakost2](https://github.com/kostakost2) — look-ahead bias report (#475)
- [@lu-zhengda](https://github.com/lu-zhengda) — proxy/base_url support request (#427)
- [@VamsiKrishna2021](https://github.com/VamsiKrishna2021) — invalid indicator crash report (#429)

## [0.2.2] — 2026-03-22

### Added

- **Five-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell)
  introduced for the Portfolio Manager.
- **Anthropic effort level** support for Claude models.
- **OpenAI Responses API** path for native OpenAI models.

### Changed

- `risk_manager` renamed to `portfolio_manager` to match the role description
  shown in the CLI display.
- Exchange-qualified tickers (e.g. `7203.T`, `BRK.B`) preserved across all
  agent prompts and tool calls.
- Process-level UTF-8 default attempted for cross-platform consistency
  (note: this approach did not actually take effect; replaced in v0.2.4 with
  explicit per-call `encoding="utf-8"` arguments).

### Fixed

- yfinance rate-limit errors are retried with exponential backoff. (#426)
- HTTP client SSL customisation is supported for environments that need
  custom certificate bundles. (#379)
- Report-section writes handle list-of-string content gracefully.

### Contributors

- [@CadeYu](https://github.com/CadeYu) — exchange-qualified ticker preservation (#413)
- [@yang1002378395-cmyk](https://github.com/yang1002378395-cmyk) — HTTP client SSL customisation (#379)

## [0.2.1] — 2026-03-15

### Security

- Patched `langchain-core` vulnerability (LangGrinch). (#335)
- Removed `chainlit` dependency affected by CVE-2026-22218.

### Added

- `pyproject.toml` build-system configuration; the project now installs via
  modern packaging tooling.

### Removed

- `setup.py` — dependencies consolidated to `pyproject.toml`.

### Fixed

- Risk manager reads the correct fundamental report source. (#341)
- All `open()` calls receive an explicit UTF-8 encoding (initial pass).
- `get_indicators` tool handles comma-separated indicator names from the LLM. (#368)
- `Propagation` initialises every debate-state field so risk debaters never
  see missing keys.
- Stock data parsing tolerates malformed CSVs and NaN values.
- Conditional debate logic respects the configured round count. (#361)

### Contributors

- [@RinZ27](https://github.com/RinZ27) — `langchain-core` security patch (#335)
- [@Ljx-007](https://github.com/Ljx-007) — risk manager fundamental-report fix (#341)
- [@makk9](https://github.com/makk9) — debate-rounds config issue (#361)

## [0.2.0] — 2026-02-04

This is the largest release since the initial public version. The framework
moved from single-provider to a multi-provider architecture and grew several
production-ready surfaces.

### Added

- **Multi-provider LLM support** (OpenAI, Google, Anthropic, xAI, OpenRouter,
  Ollama) via a factory pattern, with provider-specific thinking configurations.
- **Alpha Vantage** integration as a configurable primary data provider, with
  yfinance as a community-stability fallback.
- **Footer statistics** in the CLI: real-time tracking of LLM calls, tool
  calls, and token usage via LangChain callbacks.
- **Post-analysis report saving** — the framework writes per-section markdown
  files (analyst reports, debate transcripts, final decision) when a run
  completes.
- **Announcements panel** — fetches updates from `api.tauric.ai/v1/announcements`
  for the CLI welcome screen.
- **Tool fallbacks** so a single vendor outage does not stop the pipeline.

### Changed

- Risky / Safe risk debaters renamed to **Aggressive / Conservative** for
  consistency with the displayed agent labels.
- Default data vendor switched to balance reliability and quota across
  community deployments.
- Ollama and OpenRouter model lists updated; default endpoints clarified.

### Fixed

- Analyst status tracking and message deduplication in the live display.
- Infinite-loop guard in the agent loop; reflection and logging hardened.
- Various data-vendor implementation bugs and tool-signature mismatches.

### Contributors

This release is the first with substantial outside contributions; many community
PRs from late 2025 also landed here.

- [@luohy15](https://github.com/luohy15) — Alpha Vantage data-vendor integration (#235)
- [@EdwardoSunny](https://github.com/EdwardoSunny) — yfinance fetching optimisations (#245)
- [@Mirza-Samad-Ahmed-Baig](https://github.com/Mirza-Samad-Ahmed-Baig) — infinite-loop guard, reflection, and logging fixes (#89)
- [@ZeroAct](https://github.com/ZeroAct) — saved results path support (#29)
- [@Zhongyi-Lu](https://github.com/Zhongyi-Lu) — `.env` gitignore (#49)
- [@csoboy](https://github.com/csoboy) — local Ollama setup (#53)
- [@chauhang](https://github.com/chauhang) — initial Docker support attempt (#47, later reverted; the merged Docker support shipped in v0.2.4)

## [0.1.1] — 2025-06-07

### Removed

- Static site assets that had been bundled with v0.1.0; the public site now
  lives separately.

## [0.1.0] — 2025-06-05

### Added

- **Initial public release** of the TradingAgents multi-agent trading
  framework: market / sentiment / news / fundamentals analysts; bull and bear
  researchers; trader; aggressive, conservative, and neutral risk debaters;
  portfolio manager. LangGraph orchestration, yfinance data, per-agent
  BM25 memory, single-provider OpenAI integration, interactive CLI.

[0.2.4]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/TauricResearch/TradingAgents/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/TauricResearch/TradingAgents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TauricResearch/TradingAgents/releases/tag/v0.1.0
