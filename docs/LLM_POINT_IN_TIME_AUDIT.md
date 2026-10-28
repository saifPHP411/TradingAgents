# Point-in-time data integrity audit

This document answers one question: **for a run dated in the past, is every input
the agents would read actually available as of that date?**

It is an audit, not a result. It does not measure the strategy. Its purpose is to
decide, before any genuine LLM pilot, whether a historical decision can be made
from information that existed at the decision time — and to say plainly which
agents cannot, so a restricted experiment is labelled restricted and a full-graph
historical run is refused rather than faked.

Everything below is derived from the code, not asserted. The dependency matrix is
built from `tradingagents/graph/analyst_execution.py` (which agents exist and what
tools they hold), `tradingagents/dataflows/router.py` (which vendor each tool can
reach) and `tradingagents/dataflows/pit.py` (each source's guarantee and the guard
that enforces it). A test (`tests/test_llm_point_in_time.py`) fails if any of the
three moves out from under this matrix.

The machine-readable form is the `point_in_time` block of
`docs/baseline/llm_evaluation.json`.

## Two distinctions this audit keeps

**Historical data availability is not point-in-time validity.** A vendor that
returns a bar dated 2024-01-05 has proven nothing about whether that value was
known on 2024-01-05. Adjusted prices, restated fundamentals and re-dated news all
carry a past date and can all contain information from later. So a source is not
called safe because it returns a historical date; it is called safe only when a
guard serves it *as it stood at the run's date*.

**Publication time is not the time information became public.** A filing is dated
by the period it covers and published weeks later; an insider trade is dated by
when it happened and disclosed up to two business days later; a macro series is
dated by its reference period and released on a schedule, then revised. Each of
these is a way a run can read something that was not yet known. The guards below
address them individually.

## Classification vocabulary

| Classification | Meaning |
|---|---|
| `PIT_VERIFIED` | Evidence establishes the data was available at the decision time. Only a source served as it stood then qualifies. |
| `PIT_PARTIALLY_VERIFIED` | Only some fields or periods are proven safe; the dating depends on the vendor being correct. |
| `PIT_UNVERIFIED` | Historical data exists but availability at decision time cannot be established. (No current source falls here; the bucket is kept so the vocabulary is closed.) |
| `CURRENT_ONLY` | The source returns present-day information with no trustworthy historical snapshot. |
| `UNAVAILABLE` | Required data cannot be obtained. (No current source falls here.) |

The mapping from `pit.py`'s three registry levels is conservative and one-way:
`REQUEST_BOUNDED → PIT_VERIFIED`, `WINDOW_FILTERED → PIT_PARTIALLY_VERIFIED`,
`NOT_PIT_SAFE → CURRENT_ONLY`.

## Agent dependency matrix

Derived from the graph spec. The sentiment analyst fetches its sources *before*
calling the model, so its tool tuple is empty; its feeds are recorded explicitly
from `agents/analysts/sentiment_analyst.py`.

| Agent | Tools | Sources | Classifications |
|---|---|---|---|
| Market Analyst | `get_stock_data`, `get_indicators`, `get_verified_market_snapshot` | `yahoo_prices`, `yahoo_indicators` | both `PIT_VERIFIED` |
| News Analyst | `get_news`, `get_global_news`, `get_macro_indicators`, `get_prediction_markets` | `yahoo_news`, `fred`, `polymarket` | `PIT_PARTIALLY_VERIFIED`, `PIT_VERIFIED`, `CURRENT_ONLY` |
| Sentiment Analyst | *(pre-fetched)* | `yahoo_news`, `stocktwits`, `reddit` | all `PIT_PARTIALLY_VERIFIED` |
| Fundamentals Analyst | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_insider_transactions` | `yahoo_profile`, `sec_edgar`, `insider_transactions` | `CURRENT_ONLY`, `PIT_PARTIALLY_VERIFIED`, `CURRENT_ONLY` |

## Source classifications and evidence

Each row cites the guard that implements it. The full note for each source is in
`pit.py` and in the artifact; the summary here is the audit's read of it.

| Source | Class | Guard | Evidence / limitation |
|---|---|---|---|
| `yahoo_prices` (OHLCV) | `PIT_VERIFIED` | `agents.tools.get_stock_data → date_window.as_of_window` | The tool clamps the window's end to the run date before the vendor is called, so a later bar is never requested. Prices are split/dividend-adjusted, so a historical price is the adjusted value, not the tape value that day. |
| `yahoo_indicators` | `PIT_VERIFIED` | `agents.tools.get_indicators → date_window.as_of_window` | Computed from the clamped price window; no future bar enters the calculation. |
| `fred` (macro) | `PIT_VERIFIED` | `fred.get_macro_data` (observation_end + vintage pinning) | Observations are requested up to the run date and the series is pinned to a vintage, so a later revision does not appear in an earlier run. The clamp bounds the window but does **not** model each series' release calendar: a run may see a release before its figures were ordinarily public. |
| `sec_edgar` (filings) | `PIT_PARTIALLY_VERIFIED` | `date_window.withhold_undated_statements` | EDGAR serves US filings as filed, which is the right shape. A statement whose filing date the vendor does not expose is *withheld* from a historical run rather than assumed to predate it. A filing dated by acceptance time can be a day late. |
| `yahoo_news` | `PIT_PARTIALLY_VERIFIED` | `agents.tools.get_news → as_of_window + in_window + coverage_gap` | Yahoo news search returns its latest items whatever window is asked. Dated items are filtered to the window; an undated item is withheld from a historical run; a window the feed never observed is reported unavailable, not empty. Depends on the vendor dating items correctly. |
| `stocktwits` | `PIT_PARTIALLY_VERIFIED` | `date_window.in_window + stocktwits._within_window + coverage_gap` | The public feed serves the latest messages; only recent windows are observed and an older one is reported unavailable. Filtered on `created_at`. |
| `reddit` | `PIT_PARTIALLY_VERIFIED` | `date_window.in_window + reddit._within_window + coverage_gap` | Same shape as StockTwits: recent-only feed, filtered on `created_utc`, older windows reported unobserved. |
| `yahoo_profile` (sector/industry/name) | `CURRENT_ONLY` | `date_window.withhold_live_profile` | Present-day snapshot with no historical vintage. Withheld entirely from a historical run: companies rename and are reclassified, so serving it would inject post-decision information. |
| `insider_transactions` | `CURRENT_ONLY` | `date_window.withhold_undisclosed_trades` | The vendor dates a trade by when it happened, not when its Form 4 became public. Withheld from a historical run. |
| `polymarket` | `CURRENT_ONLY` | none | Prediction-market prices are served as of now with no historical endpoint and no date parameter to clamp. Must not be used in a backtest. |

Verified: 3 sources (`yahoo_prices`, `yahoo_indicators`, `fred`). Weak: 4
(`yahoo_news`, `stocktwits`, `reddit`, `sec_edgar`). Current-only: 3
(`polymarket`, `yahoo_profile`, `insider_transactions`).

## Readiness verdict

`graph_readiness(scope)` returns whether a declared scope can run on trustworthy
inputs.

**`price_only` (market analyst alone): READY.** It reads only `yahoo_prices` and
`yahoo_indicators`, both `PIT_VERIFIED`, both served as they stood at the run's
date. This is the scope the frozen dataset supports.

**`full` (all four analysts): NOT READY.** The news, sentiment and fundamentals
analysts read sources that are window-filtered (dating is the vendor's word) or
current-only (must be withheld). A full-graph historical run therefore cannot be
point-in-time safe today. This is a property of the sources, not a bug to work
around, and the readiness verdict says so with named blockers.

## The historical information boundary

The boundary is implemented in `tradingagents/dataflows/date_window.py` and
consumed by every dated tool. It is not new; this audit confirms it exists and
covers the cases the pilot depends on:

- **Future data rejected.** `as_of` / `as_of_window` clamp any requested date or
  window to the run's `trade_date`; a window wholly after it is moved back to end
  there. A model that passes today's date instead of the analysis date cannot walk
  past the guards.
- **Timezones normalized.** `to_utc` normalizes every timestamp to UTC-aware; a
  naive value is assumed UTC. `in_window` compares in UTC with an upper bound
  exclusive at midnight after `end`.
- **Missing publication timestamps handled.** `in_window(pub_dt=None, …)` keeps an
  undated item only when the window reaches the present, so a backtest never sees
  an item it cannot prove is not from the future.
- **Latest-value APIs cannot silently replace a snapshot.** `withhold_live_profile`
  and `withhold_undisclosed_trades` replace present-day data with a notice in a
  historical run rather than serving it.
- **Coverage gaps are explicit.** `coverage_gap` reports a window a recent-only
  feed never observed as unavailable, not as an empty (absence) result.
- **Derived features are bounded.** Indicators are computed from the clamped price
  window, so no future bar enters them.
- **Memory cannot cross the boundary.** Resolved memory-log lessons carry the date
  their outcome became known and are filtered by `get_past_context(as_of=…)`
  (`tests/test_memory_pointintime.py`).

### Residual leak channels (recorded, not hidden)

- **Split/dividend adjustment.** A "historical" price is the adjusted value, so a
  backtest on adjusted prices reflects a later corporate action. The baseline
  documents this (price return only, no dividend cash).
- **Vendor dating.** The window-filtered sources trust the vendor's timestamps. If
  a vendor mis-dates an item, the filter cannot catch it.
- **FRED release calendar.** The vintage pin prevents later revisions but does not
  model when each series is ordinarily released.
- **Current news during historical simulation.** `get_news` / `get_global_news`
  route to a live vendor; the guard is the window clamp and the coverage gap, not a
  frozen archive. There is no point-in-time news snapshot in this repository.

## Provider alternatives and verified constraints

This section records what was checked, and what could not be verified from this
environment. **No provider was purchased, subscribed to, or contacted for paid
data.**

| Option | Coverage | PIT guarantee | Credential | Usable here? |
|---|---|---|---|---|
| Frozen dataset (in repo) | Daily OHLCV for the baseline universe, 2022–2024 | Committed bytes; byte-identical each run | None | Yes — the only source the `price_only` scope uses |
| Yahoo (prices/indicators/news) | Broad | prices request-bounded; news window-filtered | None | Network reachable, but live-only for news and rate-limited (HTTP 429 observed) |
| SEC EDGAR | US filings as filed | window-filtered (needs filing dates) | `SEC_EDGAR_USER_AGENT` contact string, no key | Reachable (verified 2026-10-28); `filed`-date filtering is in `vendors/sec_edgar.py`, but no frozen capture in repo |
| FRED | US + mirrored euro macro | request-bounded with vintage pin | `FRED_API_KEY` (free) | Reachable; key not present here |
| StockTwits / Reddit | Recent social posts | window-filtered, recent-only | None / OAuth | Not usable for a historical window — feeds are recent-only |
| Polymarket | Live prediction markets | none | None | Not usable in a backtest |

Network is reachable from this environment (verified: `api.openai.com` returned
HTTP 401, i.e. reachable but unauthenticated; `fred.stlouisfed.org` returned HTTP
200; `data.sec.gov` returned HTTP 200 with a contact User-Agent; Yahoo returned
HTTP 429). Reachability is not authorization and not a point-in-time guarantee.

### SEC EDGAR is the strongest PIT fundamentals path already in the repo

`vendors/sec_edgar.py` serves US filings as they were filed: it reads the
`filed` date on every XBRL fact and serves only facts filed on or before
`as_of_date`, taking a restated period at its latest filing *before* that date
(verified 2026-10-28 against `data.sec.gov`, which exposes `filed`, `form`,
`start`, `end`, `val`, `accn` per fact). The registry still classifies it
`WINDOW_FILTERED` / `PIT_PARTIALLY_VERIFIED` because a statement with no exposed
filing date is *withheld* rather than assumed old, and a filing dated by
acceptance time can be a day late. It is the closest existing source to a
point-in-time fundamentals feed and needs no credential — only a contact
User-Agent.

**A snapshot seam exists but is unused.** `dataflows/pit.py` defines a
`SnapshotSource` protocol and a `SNAPSHOT_SOURCES` registry (currently empty).
Registering a source that can serve data as it stood at a past date raises its
`effective_level` to `REQUEST_BOUNDED` everywhere at once, without touching
callers. This is the smallest path to promoting a weak source: capture daily
snapshots of the news/social/fundamentals responses into a local store and register
an adapter. No such store exists in this repository.

## Explicit blockers

A **full-graph historical LLM pilot** is blocked by:

1. **No point-in-time news/sentiment/fundamentals.** The three sources the
   non-market analysts depend on are window-filtered or current-only. There is no
   frozen historical capture in the repository and no snapshot source registered.
   (SEC EDGAR is the strongest existing PIT path for fundamentals; see above.)
2. **No model credential.** No provider key is present in this environment.
   Confirmed by inspecting environment variable names only; no value was read or
   printed. Documented free-tier options and the exact env var to set are in
   [`LLM_PROVIDER_AUDIT.md`](LLM_PROVIDER_AUDIT.md).
3. **No capture path.** Nothing constructs a `DecisionRecord` from a real graph
   run; records are built only in tests. So even with a credential, there is
   nothing for `build_evaluation` to evaluate. This is a code gap, not a data gap,
   and it is deliberately left unfilled until (1) is resolved — a capture over
   inputs the graph cannot honestly obtain for a historical date would record
   decisions made from non-point-in-time data.

A **price-only restricted pilot** is not blocked by data — it reads only
`PIT_VERIFIED` sources — but it does not measure the news, sentiment or
fundamentals agents, so it is not a full-graph evaluation. It remains blocked by
the missing credential (2) and the missing capture path (3).

## What would unblock a full-graph historical pilot

1. **Register a snapshot source.** Capture the news/social/fundamentals responses
   once per session into a dated local store, implement `SnapshotSource` over it,
   and `register_snapshot_source(...)`. The classification rises to
   `PIT_VERIFIED` for that source automatically.
2. **Provide a credential** for one supported provider (a documented free tier or
   a local endpoint is enough) and a request budget.
3. **Add a capture path** that runs the graph over the frozen dataset and writes
   its decisions to a `DecisionRecordSet` with the window hash and config hash.
4. **Re-run** `graph_readiness("full")` — it should report ready once every source
   in the matrix is `PIT_VERIFIED` — then run the bounded pilot declared in
   `docs/LLM_EVALUATION.md`.

Until (1), (2) and (3) hold, the honest status is `LLM BASELINE BLOCKED`.
