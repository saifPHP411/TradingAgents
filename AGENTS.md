# AGENTS.md

Repository-specific notes for future sessions.

## Commands

```bash
pip install -e .          # install
pytest -q                 # full suite; offline, no API key needed
python -m ruff check .    # lint
python -m pytest tests/test_backtest_engine.py -q   # one module
python examples/simulated_run.py                    # end-to-end simulated run
python -m tradingagents.baseline.run --out docs/baseline/official_baseline.json  # official baseline
```

`python -m tradingagents.baseline.fetch_data` is the only network entry point in
the baseline package, and it is run once by a human to freeze
`tradingagents/baseline/data/`. Never call it from a test or from a run. The
download goes through `dataflows/vendors/yahoo/ohlcv.load_ohlcv_range`, because
`tests/test_layering.py` allows only the data layer to import `yfinance`.

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

`tradingagents/paper_experiment/` is the Phase 10 prospective layer. It owns no
accounting of its own: it drives the existing `PaperSession` and records the
history. Three of its artifacts are deliberately distinct — the hash-chained
**journal** (`journal.json`) is the experiment's history, the **session state**
(`session_state.json`) is the recovery format for the book, and the **report** is
a read-only view. Do not treat a report as recovery. Its one timing rule is fixed:
a decision is made from bars through session T, executed at T+1's open, and
evaluated at T+1+lag; the config refuses `execute_on_next_open=False`, so a
decision can never fill at the close it was made from. The default price source
is still the frozen historical dataset, so by default the experiment reports
itself as offline paper. Real vendor sources live in `paper_experiment/providers/`
(Alpha Vantage daily bars, Finnhub quotes); they are opt-in, fetched once, and
never the default. A vendor failure must stop the run rather than silently
substituting the frozen dataset — the CLI enforces this. A current-market source
now exists, so `current_market_source_available()` is `True`, but `READY FOR LIVE
PAPER` additionally needs a source that can serve a session window; a quote-only
source (Finnhub) is current but cannot, so it does not qualify. Do not call any of
this a live paper run, and do not describe a report snapshot as durable recovery.

`tradingagents/baseline/` is the reproducible *reference* baseline: one
deterministic strategy, a frozen committed dataset, a versioned protocol, and an
independent cross-check. It is not a baseline of the LLM strategy, which is
non-deterministic and needs credentials; do not describe it as one. The frozen
data under `tradingagents/baseline/data/` is the source of truth — a re-fetch is
a new baseline version, not an update.

`tradingagents/llm_evaluation/` records and replays LLM decisions offline.
`point_in_time.py` in that package is the readiness audit: its dependency matrix
is derived from the graph spec, the router's vendor map and the PIT registry, so
do not hand-edit the matrix — change `dataflows/pit.py` and the tests will follow.
A source's classification is one-way from its registry level; do not claim a
`WINDOW_FILTERED` source is `PIT_VERIFIED` because it returned a historical date.

`tradingagents/llm_evaluation/provider_audit.py` reports which LLM providers this
environment can reach. It reads env-var *names* only and contacts nothing; do not
make it read a key value or probe a provider. A free-tier claim in it must carry
the official URL it was checked against — no URL, no claim. Readiness and
point-in-time data are separate blockers; neither implies the other.

`tradingagents/llm_evaluation/capture.py` is the only producer of a
`DecisionRecord` from a real run. The `llm_inference` label is derived from
`graph.inference_observed`, which an `InferenceObserver` on the *real* provider
clients sets — never from a caller's argument. Do not add a way to set that label
from a parameter, and do not let a scripted/stubbed model set the observer: a
mock must stay `synthetic_fixture`, and `tests/test_llm_capture.py` asserts it.
The capture invents no size or levels; sizing is the risk engine's.

`tradingagents/llm_evaluation/smoke.py` is the only path that may make a real
model call. It is opt-in, bounded to one request, gated on a credential before
any request, and never runs in the test suite. Do not call it from a test, a
run, or any automatic path. An injected `client_factory` is the test seam and
must never be reported as a genuine observation. It sets `max_retries=0` on the
client so a provider 5xx cannot multiply one request into three, and it reports a
`failure_category` from a closed vocabulary rather than an error string alone.

`tests/test_phase9_genuine_inference.py` is how the genuine-inference
discriminator is proven without a credential: it drives the *real* provider
client through an `httpx.MockTransport`, so the observer fires because the client
dispatched a LangChain `on_llm_end` callback, not because a test said so. Keep
that shape — an offline test that swaps in a stub proves nothing about the
discriminator, because a stub is exactly what the discriminator must reject. A
genuine `llm_inference` record still requires an actual provider response; no
test may assert otherwise.

`tradingagents/llm_evaluation/snapshot.py` is the reproducible price input a
record anchors to. Its `content_hash` must be the *same* function the record
carries (`replay.window_hash`), or `snapshot.matches_record(record)` proves
nothing — do not give it a second hashing scheme. A snapshot refuses an empty
window at construction; do not let a missing window become a snapshot. The smoke
default session is the frozen dataset's last covered session, not today: today is
past the data, and a window that does not end at the decision date is dropped.
`experiment_ready` requires a genuine decision *and* a reproducible window, so a
genuine inference with no window must not claim it.

The offline suite must never make an external provider call. If you add a test
that needs a credential, skip it when the credential is absent rather than
faking a success.

## Conventions

- Never add a live trading path. `ExecutionMode` has only `BACKTEST` and `PAPER`,
  and `resolve_mode` raises on live-sounding names. Tests assert this.
- Undefined metrics are `None`, not `0.0`. A flat run has no Sharpe ratio.
- `counts.executed` in the paper report counts *decisions settled at their
  execution session*, and a HOLD is one. It is not a fill count; the `orders`
  funnel is. Never read `executed` as `filled` — a run can legitimately report
  `executed: 1, filled_orders: 0` when the tail decision is a HOLD. See
  `tests/test_paper_experiment_counters.py`.
- The paper experiment's inference seam takes `(symbol, session, window)`, and the
  window is the session's point-in-time slice. A model asked for a directional
  call with no data can only answer Hold, so never strip the window from the
  request, and never spend a request on a session whose window cannot be anchored.
  A non-finite window value blocks before any call. See
  `tests/test_paper_inference_prompt.py`.
- Every cash or position change goes through `Ledger.apply_fill`, so the
  accounting rules stay in one place and the event log stays complete.
- Rejections carry a named reason from a closed vocabulary, never prose only.
- A point-in-time claim must cite the guard that implements it; see
  `dataflows/pit.py`. Do not call a source PIT-safe without one.
- Tests are written to catch a premise error, not to confirm the code. When a
  test fails, first check whether the test's premise is right.
- Every number that enters the ledger must be finite. NaN is the dangerous case,
  because `x <= 0` and `x < 0` are both false for it, so a NaN passes the usual
  positivity guards and then poisons cash and equity. Check `math.isfinite`
  explicitly, never rely on a comparison.
- An order may fill at most once. `ExecutionSimulator.submit` refuses an order
  that is already terminal, because the ledger cannot tell a re-submission from a
  second genuine order.
- A strategy's `context.feed` is a view bounded at the session, not the whole
  feed. Do not hand a strategy the raw feed; that re-opens look-ahead through
  `feed.bar(symbol, future_date)`.
- A paper experiment must be restart-safe *without* the caller re-supplying the
  price source. A vendor source fetches once and persists its validated bars into
  `session_state.json`, so `PaperExperiment.load(root)` rebuilds the same source
  with no second fetch. The frozen dataset is not stored (2,515 sessions); it
  reloads from disk and the recorded window is checked instead. A source that can
  persist neither is refused, never silently swapped for the frozen dataset.
- The broker decision bridge (`paper_experiment/paper_trading.py`) must re-read the
  experiment on every decision; it must never cache a decision list at build time.
  A worker is often built before any decision exists (the dashboard builds it on
  START), and `auto-advance` records decisions from another writer, so a cached
  list freezes the worker at "no decision is due" forever. Two guards keep it from
  going wrong the other way: a build-time baseline suppresses decisions already
  *settled* when the source was built (START must not replay history), and the
  worker's durable `processed_decisions` set suppresses any decision it has already
  ordered. See `tests/test_phase17_decision_delivery.py`.
- `PaperTradingController.status()` must not build the worker. A status poll that
  constructs the worker (and so the decision source) is what froze a live view; the
  status path reads the persisted `worker_state.json` as `STOPPED` instead.

## Current-market decision source (Phase 27)

The worker draws its decisions from one of two **separate** sources, selected by
`--paper-decision-source` / `$TRADINGAGENTS_PAPER_DECISION_SOURCE`, default
`history`:

- `history` — the existing `ExperimentDecisionSource` over the experiment's
  recorded decisions. Bounded by the experiment's sessions, so it reports
  `HISTORY_EXHAUSTED` once they are consumed. Unchanged.
- `current_market` — the independent `CurrentMarketDecisionSource`
  (`paper_experiment/current_source.py`). It is **not** an extension of the
  historical source and is never blocked by a consumed history. Do not make
  `ExperimentDecisionSource` the current-market source, and do not merge the two:
  one replays a recorded experiment, the other decides live.

The two paths differ in exactly one place: where the decision's input window comes
from. `render_strategy_prompt` (`inference.py`) is the shared builder both paths
send, so the current path reuses the *same* strategy — never grow a second one.
The current path's market-data and market-hours seams are their own modules
(`current_market_data.py`, `market_hours.py`), and its decision data
(`data.alpaca.markets`, feed `iex`) is kept apart from the worker's execution price
(the existing `--paper-market-data` seam). Do not price an order from the decision
window, and do not fetch decision data from the paper/order host.

Its cadence is one decision **per symbol per trading session**, keyed on the
market clock's session date; `decision_id` is `current-<symbol>-<session>`, so it
is deterministic and a restart within a session replays the same id without a
second model call. The source persists its own state in
`current_source_state.json` — never the experiment's `session_state.json` /
`journal.json` — so it is not a second writer to the experiment directory. It
fabricates no size, stop or target: the notional is the worker's, and a rating with
no tradeable action is an honest `NO_SIGNAL`, never a fake signal. Market-hours
failures are `MARKET_CLOSED`; a stale, future, malformed or unreachable window is
`MARKET_DATA_UNAVAILABLE`. Do not treat a stale bar as current, and do not let an
unknown clock be read as open. The path is opt-in and stays paper-only; no live
endpoint, and risk limits are unchanged.

## Paper-trading observability

- The paper worker records *why* it placed no order as a stable code from a closed
  vocabulary (`NO_TRADE_REASONS` in `paper_experiment/broker/trading.py`), carried
  on `CycleResult.reason_code` and persisted as `WorkerState.last_no_trade_code`.
  The experiment decision source names its own code (`ExperimentDecisionSource.skip_code`),
  so a fully-consumed history reports `HISTORY_EXHAUSTED` (new data required) rather
  than a generic "no decision". Add a new code to the vocabulary, never a prose-only
  reason.
- `engine_view()` derives a single `blocker` (`{code, detail, operator_action_required}`)
  from persisted state only. It is `None` for a stopped worker (off, not blocked)
  and `AWAITING_FIRST_EVALUATION` before the first cycle. It must not touch the
  broker or the network.
- A price seam must describe itself: the historical/static seams derive
  `market_data_status` from `available()` so the view never reports the market data
  down while the worker is pricing an order from it. An absent status is *unknown*,
  resolved from worker evidence, never read as "broken".
- The engine view is read at RUNNING time in tests: a stopped worker has no blocker,
  so assert the blocker before `stop()`.
- A current-market decision is consumed only once it *produced an order*
  (`WorkerState.ordered_decisions`, written by `_mark_ordered` at the two sites
  that reach the broker: a submit and a reconcile). `processed_decisions` still
  means "do not retry" and still records a refusal, because the historical path
  relies on that bound; the current-market guard (`_is_processed`) and the current
  source's `is_consumed` filter read the order-only set instead. Reading the
  refusal-inclusive set there is what reported `CURRENT_MARKET_NO_NEW_DATA` for a
  session whose only decision was a no-op. Both sets are additive to the state
  file; an older file reads back with no ordered decisions.
- Two decision tallies answer "why are there so few trades", and they are kept
  apart because they answer different questions. The **strategy tally**
  (`CurrentMarketDecisionSource.statistics()`, persisted in the source's own
  state) counts genuine model evaluations by rating (BUY/SELL/HOLD) and keeps the
  model's own last HOLD reason; only a real call bumps it, so a replay, a
  market-closed poll or a data failure counts nothing. The **worker tally**
  (`WorkerState.decision_outcomes` → `_decision_statistics()`) counts settled
  outcomes: `DECISION_OUTCOME_ORDERED` for a decision that produced an order,
  otherwise the `NO_TRADE_REASONS` code that says why (including `RISK_REJECTED`).
  A source that does not describe itself (the historical one) contributes no
  strategy tally, reported as `None`, never a fabricated zero. Both are
  observational: they change no cadence, no sizing, and no risk limit.
- `DECISION_OUTCOME_ORDERED` is deliberately *not* in `NO_TRADE_REASONS`: it is the
  absence of a no-trade reason, and a consumer iterating the vocabulary must not
  read it as one.

## Deployment (Railway) and the crypto paper path

- A Railway deployment installs the **wheel**, so anything the dashboard reads
  at startup must be declared in `[tool.setuptools.package-data]`. The frozen
  baseline dataset (`tradingagents/baseline/data/*.csv`) is committed but is
  not Python source; without that declaration the installed package has no data
  and the dashboard refuses to start with "the frozen dataset is unavailable".
  This failure appears only after the image is built — verify a change by
  installing the wheel into a clean venv, not just by running the checkout.
- A Railway volume mount is writable only by root, and the image's default user
  is the non-root `appuser`. `RAILWAY_RUN_UID=0` is what makes the volume
  writable; without it the first state write fails with `PermissionError` and
  the processed/ordered decision sets never land.
- **The image's `ENTRYPOINT` must be the program the platform should run.**
  A Dockerfile `ENTRYPOINT` is *prepended* to whatever the platform runs, so an
  entrypoint of `tradingagents` (the analysis CLI, a Typer app) turns a
  Railway start command into `tradingagents /bin/sh -c ...` and the service
  dies with "No such command '/bin/sh'". The image's `ENTRYPOINT` is the
  paper dashboard, which is also the only program that serves the `/healthz`
  the image's own `HEALTHCHECK` probes — an image whose default process is the
  analysis CLI carries a health check that can never pass. The analysis CLI
  moved to an explicit `entrypoint:` in `docker-compose.yml`.
- The deployment's start command is checked against the **real** argument
  parser in `tests/test_crypto_deployment_wiring.py`, not against a copy of it:
  the test extracts the flags from `railway.json`, feeds them to `build_parser()`
  and asserts the effective settings. A renamed flag breaks the test rather
  than the deployment. `railway.json` and `.railway/railway.ts` must carry the
  identical command string.
- Starting the dashboard must create the experiment's state and **nothing
  else**: no worker, no broker request. The paper worker starts only when an
  operator presses START (bearer token **and** `X-Paper-Confirm: yes`), and an
  order is placed only inside a worker cycle. Keep those four layers
  distinct — a test guards both HTTP libraries (`requests` and `urllib`), and
  the guard is itself tested so it cannot become a no-op.
- The worker's `processed_decisions` / `ordered_decisions` sets are what stop a
  restart from placing the same order twice, so the experiment root must be on
  the persistent volume, not in the container filesystem.
- The crypto path reads two different venues: the **decision** price is Binance's
  `BTC/USDT` and the **execution** price is Alpaca's `BTC/USD`. They are not the
  same instrument, so an order must be sized from an execution-venue price; when
  it cannot be read, or it differs from the decision price by more than
  `max_price_deviation`, the worker places **no** order
  (`EXECUTION_PRICE_UNAVAILABLE` / `PRICE_DEVIATION`).
- Every Alpaca host is checked at the point of use: the market-data client
  refuses a live or unapproved host **at construction, before it reads any
  credential**, and the crypto asset registry runs `assert_paper_endpoint`
  itself rather than trusting the caller to have done it. A new provider that
  takes a `base_url` must do the same — the equity adapter's single gate is not
  automatically inherited by a new client.
- Alpaca's **time-in-force vocabulary is per asset class**. Equity orders take
  `day`/`gtc`/`opg`/`cls`/`ioc`/`fok`; **crypto orders take only `gtc` and `ioc`**,
  and anything else is refused with HTTP 422 `invalid crypto time_in_force`. The
  adapter therefore resolves the time-in-force from an explicit `crypto` flag
  (`_resolve_time_in_force`), and refuses an unsupported value **before** the
  request. The flag travels from the controller's crypto branch through
  `PaperTradingConfig.crypto` to both `submit_order` call sites (the entry and
  the exit). Do not infer the asset class from a symbol string — `BTC/USD` and
  `AAPL` are both strings, so that would be a guess. The controller's
  `_worker_config()` and `_build_worker()` must agree on the flag, or the status
  view reports settings the worker does not enforce.
- A crypto order's **quantity is small**, and `str()` renders a small float in
  scientific notation (`1e-06`), which Alpaca does not accept as a `qty`. The
  adapter formats it with `Decimal(str(value))` (`_format_quantity`): the
  notation changes, the value does not. Do not reintroduce `str(quantity)` here.

## Known limitation

An equity-relative position or exposure cap is re-checked at the fill price
(`RiskEngine.review_fill`), so an up-gap cannot leave a position past its cap: the
fill is reduced to the largest size that holds, and refused only when no positive
size remains. Cash and long-only constraints hold through a gap, and a buy whose
all-in cost (notional plus fee) exceeds the cash is reduced to the affordable
notional rather than refused. The residual limitation is narrower: the re-check
uses the *projected* fill price, which for a MARKET order is the price the
simulator will use, so the two agree; an order type that filled at a price not
derivable from the bar would need the re-check to reconcile after the fill. See
`docs/PR1_INDEPENDENT_REVIEW.md`.

## TJR intraday strategy

`tradingagents/tjr/` is a self-contained intraday strategy: the TJR rule chain
(liquidity sweep → 5m reversal confirmation → 5m continuation confluence → 1m
entry), its own backtest loop, and a report. It does not run inside the daily
engine, because that engine marks every bar to a midnight date and cannot express
a 5-minute bar-close. It reuses everything *around* the loop: the same `Ledger`,
`RiskEngine`, `CostModel` and `TradingSignal`. Do not give it a private ledger or
risk path.

- The closed-bar rule lives in `frames.IntradaySeries`. `available(as_of)` is the
  only reader and never returns a bar whose close is after `as_of`. Do not add a
  method that returns a later bar; that re-opens look-ahead.
- `rules.evaluate` is a pure function of the bars visible at an instant. A session
  can sweep several levels, so it tries every recent sweep and keeps the most
  *progressed* setup (a completed entry beats a bare sweep). Do not go back to
  picking only the most recent sweep: a small sweep made during a retrace would
  then replace the sweep that is actually setting up.
- The confirmation's break-of-structure reference is the most recent swing **as it
  stood at the sweep bar**, fixed there. Recomputing it per bar lets a later swing
  move the reference and un-confirm a setup that had already confirmed.
- The strategy is long-only. A complete bearish setup returns
  `SHORT_SETUP_LONG_ONLY`, never a fabricated long. Every "no trade" carries a
  reason code from a closed vocabulary; do not replace a reason code with prose.
- The strategy never requests a size, a stop or a target it did not derive from
  structure. Sizing is the risk engine's; the levels come from the sweep, the leg
  and the next draw on liquidity.
- The engine fills at the **next** 5m bar's open, and a bar that covers both the
  stop and the target is read as the stop first (the conservative choice). It is
  flat by the session close, or by the last bar the data carries when the feed
  stops short of 16:00.
- Real data comes only from `dataflows/vendors/yahoo/intraday.py` (the data layer,
  which `tests/test_layering.py` enforces). `tjr/fetch_data.py` is the package's
  only network entry point, run once by a human; `tjr/run.py` is offline. Yahoo
  serves 1m bars only for the last 30 days and 5m bars for the last 60, so an
  honest study is a few recent weeks, not years. A window the vendor will not serve
  raises; it is never shortened into a frame that looks complete.

`tjr/validation.py` is the ten-symbol historical study harness. It runs the
*unchanged* rule chain over the fetched frames under `paper_parity_limits()` and
reports the metrics, the per-symbol split, the cost scenarios and a chronological
development/test split. It changes no rule and no limit; if a change there would
alter a decision, it belongs in `rules.py` and its tests instead.

- The harness memoizes the pure evaluation (`_MemoStrategy`) keyed on
  `(symbol, session, as_of)`. The rule chain is a pure function of the bars closed
  by an instant, so a cached answer is the same answer; the cache is exact and
  `tests/test_tjr_validation.py` pins a cached run equal to a cold one. Do not
  memoize anything that depends on the book — the cache key carries no book state,
  because none of the rule chain reads it.
- A sub-period (development/test) is a *view* of the one shared-book run
  (`period_view`), rebased to the starting cash, not a second run. The book is flat
  at every session close, so a session set is self-contained. Label it a view; do
  not present it as an independent run.
- The look-ahead probe truncates the feed and re-runs: any entry at or before the
  cut must be identical. It is the executable form of the closed-bar rule.
- The real window is ~1 month (the vendor cap), not the 6–12 months a study would
  want. Never describe this as a long-horizon test, and never fill the gap.

## Crypto market data on the paper worker

- A crypto pair has **three spellings** and they name the same instrument: the
  canonical `BTC/USDT`, Binance's `BTCUSDT` and Alpaca's `BTC/USD`. The paper
  worker's symbol mapping holds the **venue's** spelling (`market_data_symbol`),
  while the rest of the system holds the canonical one. A price map keyed by only
  one spelling therefore hides a perfectly good price — the production DOGE/USDT
  failure was exactly this, and it affected *every* symbol, not just DOGE.
  `ticker_24h()` keys its result by both spellings, and `_find_ticker()` in
  `paper_experiment/broker/trading.py` compares on the canonical pair so a symbol
  can never match a different instrument by accident.
- When checking a live price, read the **canonical pair** and compare the two
  spellings through *one* response map. Two separate `price()` calls are two
  separate HTTP requests, so a small difference is the market moving, not a
  lookup bug.
- Binance's batch ticker endpoint **works**; a previous fix that "corrected" its
  JSON separators was fixing a premise that was not broken. Binance rejects the
  *whole* batch with HTTP 400 when it holds a symbol it does not list, so the
  fallback to one request per symbol is for 400/404 only — a throttle or a
  server fault would be met again, and retrying per symbol only multiplies the
  load.
- A provider method that a real run calls with **no** injected seam must have a
  default that works. `AlpacaCryptoClient.assets()` chose its HTTP seam with
  `transport or self._transport`, and in production both were `None`, so it
  called `None` (`'NoneType' object is not callable`). The fallback is
  `providers/base._default_transport`, the same one every other provider read
  uses, so the timeout, headers and params stay the ones the project already
  relies on. The provider imports that symbol *by name*, so a test must patch
  it in the provider's namespace, not in `base`.
- A diagnostic record that reaches the dashboard must be scrubbed. The venue's
  own `code` and `msg` are kept — they are what explains the failure — but a
  credential the venue echoed back is redacted from the message first, and the
  record carries no header, no query string and no raw body.
- A failed read is **not** an empty registry. `EXECUTION_REGISTRY_UNAVAILABLE`
  means the venue's list could not be read; it must never be reported as "this
  venue trades nothing", and a coin is never accepted without the venue's own
  registry.

## TJR on the paper worker

`paper_experiment/tjr_source.py` is the third `DecisionSource`: it runs the
deterministic TJR rule chain (no model) over current *intraday* Alpaca bars and
hands the worker a decision the existing risk layer and broker already understand.
It is a *separate* source from `ExperimentDecisionSource` (historical replay) and
`CurrentMarketDecisionSource` (live, but through the LLM inference seam). Do not
make one source do another's job, and do not route TJR through the LLM.

- Its decision data is current 1m/5m bars from `tjr_intraday_data.py`, on Alpaca's
  **market-data** host and the `iex` feed only — never the paper/order host. This
  is the *decision* window; the *execution* price stays the worker's existing
  `--paper-market-data` seam. Do not price an order from the decision window.
- The seam drops every bar that had not *closed* at the decision instant, refuses
  a stale or future-dated newest bar, and validates every bar. The closed-bar rule
  lives in `tjr/frames.py`; the seam only ever hands back bars the caller could
  already have seen. A missing window raises; it is never shortened into a frame
  that looks complete.
- One decision per symbol per session, id `tjr-<symbol>-<session>`, mirroring the
  TJR backtest's one-attempt-per-session rule. The id is deterministic, so a
  restart within a session replays the same id without re-evaluating the chain.
  Like `current_market`, consumption is **order-only** (`ordered_decisions` via
  `has_order_for`), not the refusal-inclusive `processed_decisions`: a session
  decision that produced no order must be re-offered, not mistaken for consumed.
- A completed setup is put through the project's own `RiskEngine` with
  `paper_parity_limits()` — the same caps the backtest and the paper account use.
  Only an *approved* signal becomes a decision, carrying the RiskEngine's sized
  quantity and the chain's stop/target. The book the engine sizes against is a
  read-only mirror of the worker's broker account/positions (`_book_snapshot_from_worker`),
  never a fabricated book; with no book the engine is handed an empty book and
  refuses to size. The source invents no size, no stop and no target.
- Every no-trade carries a code from the worker's closed vocabulary: a closed
  market is `MARKET_CLOSED`, an unreadable clock or an unreachable/stale/malformed
  window is `MARKET_DATA_UNAVAILABLE`, a RiskEngine refusal is `RISK_REJECTED`, an
  incomplete chain is `NO_SIGNAL`, and a session whose symbols are all decided is
  `TJR_NO_NEW_DATA` (the worker's own code, not the current-market one). The rule
  chain's own reason code (`NO_1M_ENTRY` and friends) rides in the record detail
  for the audit trail, never as the worker's code.
- The source persists its own state in `tjr_source_state.json` under the
  experiment root — never the experiment's `session_state.json`/`journal.json` —
  so it is not a second writer to the experiment directory.
- It is opt-in via `--paper-decision-source tjr` / `$TRADINGAGENTS_PAPER_DECISION_SOURCE`,
  paper-only, and changes no risk limit. Best paired with
  `--paper-market-data alpaca_snapshot` so an order sizes at a current price.
  `tests/test_phase30_tjr_paper_source.py` drives the whole chain offline.
- **A no-signal does not close the session.** The backtest's engine re-evaluates
  the chain at *every* 5m bar close and marks `(symbol, session)` attempted only
  when a setup *completed*; the live source must mirror that. A record whose chain
  did not complete (a bare sweep, a short setup on a long-only book) leaves the
  session open, so the bars are re-read on the next cycle and a later, complete
  setup is still seen. Closing the session on the first no-signal makes the live
  path strictly narrower than the backtest it claims to mirror — and it silently
  stops the fetch, so the data seam's `tjr_data_last_fetch_at` goes stale while
  `last_strategy_evaluation_at` keeps advancing. A *completed* setup — sized, or
  refused by the RiskEngine — does close the session: re-reading the same bars
  would only re-run the same judgement.
- **A no-decision cycle must refresh the no-trade fields.** The worker's
  `last_no_trade_decision_id` / `last_no_trade_decision_code` name the *current*
  cycle's outcome. A cycle that evaluates and finds nothing has to clear them (or
  record its own decision id), or an earlier `BROKER_REJECTED` survives beside a
  fresh `NO_SIGNAL` and the dashboard reads as if the current cycle had failed.
  The same applies to `last_error`: a clean cycle clears an error left over from
  an earlier one, and `_sync_observability` re-records a read failure if one just
  happened. The statistics stay a cumulative tally and are never overwritten — the
  two answer different questions. See `tests/test_phase31_production_runtime_audit.py`.

## Deployment and venue endpoints

- `assert_paper_endpoint()` accepts **one** URL: `https://paper-api.alpaca.markets`.
  It checks the whole URL — scheme, exact hostname, port, userinfo, path, query,
  fragment — not just the hostname. A hostname-only check accepted an HTTP
  downgrade, a credential-bearing URL and a different port, each of which is a
  different destination from the approved one. `assert_paper_request_url()` is
  the request-layer counterpart: it allows a path (a request URL needs one) but
  re-checks scheme, host, port and userinfo, so the request layer is guarded
  too. See `tests/test_endpoint_security.py`.
- A missing venue credential is a **configuration** problem, so the dashboard's
  START path answers `503`, not `500` (a server fault) or `502` (a broker
  fault). `BrokerNotConfiguredError` and `ProviderNotConfiguredError` both map
  to `503` with a secret-free message; every other `BrokerError` stays `502`.
- A venue that does not *state* whether an instrument is fractionable leaves the
  answer **unknown** (`TradingRules.fractionable is None`), which is not a
  refusal. Only an explicit `False` rejects a symbol, and only when a
  fractional quantity is actually required.
- Symbol rejection carries a stable `code` from a closed vocabulary
  (`crypto_symbol_config.py`), so the dashboard can tell a venue that does not
  list a pair from one whose registry could not be read at all. The execution
  registry being unavailable is its own state
  (`EXECUTION_REGISTRY_UNAVAILABLE`) — it must not be reported as "no symbols
  rejected".
- A dashboard percentage has **two scales**. A ratio (`total_return = 0.05`
  means 5%) is scaled by 100; a feed's own percentage (Binance's
  `priceChangePercent = 0.091` means 0.091%) is not. Printing the second kind
  with the first kind's formatter shows it 100x too large. `pctAlready()` is
  the formatter for an already-a-percentage value; see
  `tests/test_dashboard_percentage_semantics.py`, which evaluates the page's
  real JavaScript.
- The dashboard refuses to bind `0.0.0.0` without an access token, and every
  mutating paper route needs both the token and a confirmation header. The
  token is stored only in the browser, never echoed into a log or a response.
