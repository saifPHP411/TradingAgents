# TradingAgents — Trading System Audit

Audit of the repository at commit `3e44faa` (`release: 0.7.1`), branch `v0.7.1`.

This document records what the system does today, with evidence from the code,
and what it does not do. Nothing is marked `PASS` without a file and line
reference or a named test. Every claim of "no implementation" is backed by a
grep count over `tradingagents/` and `cli/`.

Legend:

- **PASS** — implemented and covered by tests.
- **PARTIAL** — implemented in some form, with a material gap.
- **FAIL** — present but incorrect or unsafe.
- **NOT IMPLEMENTED** — no code path exists.
- **UNKNOWN** — cannot be established from the repository alone.

---

## 1. Architecture

TradingAgents is a **multi-agent LLM research framework**. It produces one
qualitative decision per (ticker, analysis date) and records it in a memory log
for later reflection. It is **not** a trading system: it has no portfolio, no
orders, no execution, no cash ledger and no risk enforcement.

### 1.1 What actually runs

```
START ─┬─ Market Analyst      ─┐
       ├─ Sentiment Analyst    │
       ├─ News Analyst         ├─ (parallel, each a sub-graph with its own tools)
       ├─ Fundamentals Analyst ┘
       └─ Memory Log  (settles past decisions, loads lessons)
              │
              ▼
        Bull ⇄ Bear Researcher debate
              │
              ▼
        Research Manager  ──► Investment plan (5-tier rating)
              │
              ▼
        Trader            ──► TraderProposal (Buy/Hold/Sell + levels)
              │
              ▼
        Aggressive ⇄ Conservative ⇄ Neutral risk debate
              │
              ▼
        Portfolio Manager ──► PortfolioDecision (5-tier rating)  ──► END
```

Evidence: `tradingagents/graph/setup.py` (node and edge wiring),
`tradingagents/graph/trading_graph.py` (graph construction),
`tradingagents/agents/state.py` (`AgentState`).

### 1.2 Layers present

| Layer | Location | Notes |
|---|---|---|
| Agent roles | `tradingagents/agents/` | 4 analysts, 2 researchers, 3 risk debators, RM, trader, PM |
| Graph orchestration | `tradingagents/graph/` | LangGraph `StateGraph`, conditional debate routing, SQLite checkpointing |
| Structured output | `tradingagents/agents/schemas.py` | Pydantic schemas for RM, Trader, PM, Sentiment |
| Data vendors | `tradingagents/dataflows/` | yfinance, SEC EDGAR, FRED, Alpha Vantage, Reddit, StockTwits, Polymarket |
| Point-in-time rules | `tradingagents/dataflows/date_window.py` | `as_of`, `in_window`, withholding rules |
| Memory / reflection | `tradingagents/memory/` | Append-only markdown log, settlement, reflection |
| Evaluation | `tradingagents/backtest.py` | Grid of single-shot decisions, scored by alpha |
| Reporting | `tradingagents/reporting.py`, `report_html.py` | Markdown tree + one HTML page |
| CLI | `cli/` | Typer app, interactive + headless |

### 1.3 LLM provider abstraction

`tradingagents/llm_clients/` implements a provider factory
(`factory.py`, `base_client.py`) with concrete clients for OpenAI-compatible
endpoints, Anthropic, Google, Azure, Bedrock, plus ChatGPT-plan auth. Tier
separation (quick/deep) is supported per provider (`default_config.py`).

**Assessment: PASS** — provider-neutral, no single-vendor coupling found.
Tests: `tests/test_tier_providers.py`, `tests/test_provider_registry.py`,
`tests/test_openai_compatible_provider.py`, `tests/test_bedrock_provider.py`.

---

## 2. Current capabilities

| Capability | Status | Evidence |
|---|---|---|
| Multi-agent analysis pipeline | PASS | `graph/setup.py`, `tests/test_graph_end_to_end.py` |
| Structured agent output with free-text fallback | PASS | `agents/structured.py`, `tests/test_structured_agents.py` |
| Deterministic rating extraction | PASS | `agents/rating.py`, `tests/test_signal_processing.py` |
| Point-in-time date clamping for tools | PASS | `dataflows/date_window.py`, `tests/test_date_boundaries.py` |
| Historical-run data withholding (profile, insider, statements) | PASS | `date_window.py`, `tests/test_fundamentals_lookahead.py` |
| Social/news look-ahead trimming | PASS | `date_window.py`, `tests/test_social_lookahead.py`, `tests/test_news_lookahead.py` |
| Memory log with settled outcomes and PIT-filtered lessons | PASS | `memory/log.py`, `memory/settlement.py`, `tests/test_memory_pointintime.py` |
| Checkpoint / resume | PASS | `graph/checkpointer.py`, `tests/test_checkpoint_resume.py` |
| Caller portfolio *context* (read-only prompt block) | PASS | `portfolio.py`, `tests/test_portfolio_context.py` |
| Report tree (markdown + HTML) | PASS | `reporting.py`, `report_html.py`, `tests/test_reporting.py` |
| Decision-quality evaluation over a date grid | PARTIAL | `backtest.py` — see §4 |
| LLM call/token counters for the live CLI view | PARTIAL | `cli/stats_handler.py` — in-memory display only, not persisted |

---

## 3. Current limitations

The framework stops at the decision. Everything between "the model said Buy"
and "the book changed" is absent:

- No order object, no order lifecycle, no fills.
- No cash ledger, no positions, no P&L.
- No fees, slippage or spread anywhere.
- No deterministic risk enforcement.
- No portfolio-level backtest; only independent decision scoring.
- No performance metrics beyond per-rating mean alpha.
- No walk-forward split.
- No LLM cost accounting or budget limits.
- No execution abstraction, so no path to paper trading.

Grep evidence over `tradingagents/` and `cli/` (excluding tests):

```
realized_pnl      0     slippage        0     sharpe          0
unrealized_pnl    0     commission      0     sortino         0
take_profit       0     position_size   0     max_drawdown    0
telemetry         0     walk_forward    0     win_rate        0
                                               profit_factor   0
```

---

## 4. Backtesting assessment

`tradingagents/backtest.py` is an **evaluation harness**, and its own module
docstring is explicit that it is not a portfolio simulator:

> "Scope: this evaluates decision quality. It is not a portfolio simulator, and
> must not grow one." — `backtest.py`

How it works:

1. `iter_grid(start, end, every_n_days)` builds analysis dates, clamped to today.
2. For each `(ticker, date)` cell, `TradingAgentsGraph.propagate` runs the full
   agent graph.
3. The decision is written to a **run-local** memory log.
4. `settle_pending` measures the outcome; `summarize` groups settled cells by
   rating and reports hit rate and mean alpha.

| Requirement | Status | Evidence |
|---|---|---|
| Entry timing | PARTIAL | Entry is the first close after the analysis date (`memory/settlement.py`). Documented, but there is no order, so no decision→order→fill separation. |
| Exit timing | PARTIAL | Exit is the close `holding_period_days` sessions later (`settlement.py`). |
| Decision timestamps | PASS | The analysis date is the grid date; validated `<= today` (`trading_graph.py`). |
| Price selection | PARTIAL | Daily closes only (`vendors/yahoo/market.py:get_closes`). No open/high/low, no intraday. |
| Holding periods | PASS | Configurable `holding_period_days` (default 5). |
| Overlapping positions | NOT IMPLEMENTED | Cells are independent; two dates can "hold" the same ticker with no interaction. |
| Overlapping decisions | NOT IMPLEMENTED | No position state, so overlap is not modelled at all. |
| Market calendars | PARTIAL | The trading sessions come from the vendor's own bars, not from an explicit calendar. No holiday table. |
| Weekends / holidays | PARTIAL | Handled implicitly by missing bars (`settlement.py`); no explicit calendar assertion. |
| Incomplete candles | PARTIAL | Today's bar is excluded from settlement (`settlement.py`). |
| Look-ahead bias | PASS (decision path) | `as_of` clamps every dated tool to the trade date; social/news are window-trimmed. Tests: `test_fundamentals_lookahead.py`, `test_news_lookahead.py`, `test_social_lookahead.py`. |
| Survivorship bias | NOT IMPLEMENTED | No universe construction; the caller supplies the tickers. |
| Adjusted prices | UNKNOWN | yfinance `history()` returns adjusted prices, but nothing asserts or records which series was used. |
| Corporate actions | NOT IMPLEMENTED | No split/dividend handling in the ledger (there is no ledger). |
| Benchmark calculation | PASS | Regional benchmark by suffix, alpha = raw − benchmark (`memory/settlement.py`). |
| Portfolio equity curve | NOT IMPLEMENTED | No cash, no positions, no equity series. |

**Assessment: FAIL as a portfolio backtester; PASS as a decision-quality
evaluator.** The module is honest about this in its docstring, but the task
requires a portfolio/event-based engine, which does not exist.

---

## 5. Portfolio assessment

There are two different things called "portfolio" in this repo, and neither is
a book:

1. `tradingagents/portfolio.py` — `PortfolioContext`, the **caller's** holdings
   and cash, rendered into a prompt block so the agents can size against a real
   position. It is read-only input; nothing updates it. It has no P&L, no fees,
   no ledger, no equity.

| Field | Present? | Where |
|---|---|---|
| `initial_cash` | NOT IMPLEMENTED | — |
| `cash` | PARTIAL | `PortfolioContext.cash` — an input field, never mutated |
| `positions` | PARTIAL | `PortfolioContext.positions` — an input list, never mutated |
| `quantity` | PARTIAL | `Position.quantity` — input only |
| `average_entry_price` | PARTIAL | `Position.average_price` — input only |
| `market_value` | NOT IMPLEMENTED | — |
| `equity` | NOT IMPLEMENTED | — |
| `available_cash` | NOT IMPLEMENTED | — |
| `realized_pnl` | NOT IMPLEMENTED | — |
| `unrealized_pnl` | NOT IMPLEMENTED | — |
| `fees` | NOT IMPLEMENTED | — |
| `total_exposure` | NOT IMPLEMENTED | — |
| `buying_power` | NOT IMPLEMENTED | — |
| Portfolio constraints | NOT IMPLEMENTED | — |
| Position limits | NOT IMPLEMENTED | — |

2. `memory/settlement.py` — measures a *return* for one decision against a
   benchmark. This is an outcome measurement, not portfolio accounting.

**Assessment: NOT IMPLEMENTED.** No ledger, no P&L, no equity. The word
"portfolio" appears in `PortfolioContext` and `PortfolioManager`, which are
respectively an input DTO and an LLM agent — not accounting.

---

## 6. Execution assessment

| Requirement | Status |
|---|---|
| Orders | NOT IMPLEMENTED |
| Order lifecycle | NOT IMPLEMENTED |
| Fills | NOT IMPLEMENTED |
| Partial fills | NOT IMPLEMENTED |
| Market orders | NOT IMPLEMENTED |
| Limit orders | NOT IMPLEMENTED |
| Rejected orders | NOT IMPLEMENTED |
| Slippage | NOT IMPLEMENTED |
| Bid/ask spread | NOT IMPLEMENTED |
| Fees | NOT IMPLEMENTED |
| Execution timestamps | NOT IMPLEMENTED |

The only "execution" in the repo is LangGraph graph execution. `TraderProposal`
carries `entry_price`, `stop_loss` and a free-text `position_sizing`
(`agents/schemas.py`), but nothing consumes them: they are rendered into
markdown for a human to read.

**Assessment: NOT IMPLEMENTED.**

---

## 7. Risk assessment

| Requirement | Status | Evidence |
|---|---|---|
| Position sizing | NOT IMPLEMENTED | `position_sizing` is free text ("5% of portfolio"), never parsed or enforced |
| Max position | NOT IMPLEMENTED | — |
| Max exposure | NOT IMPLEMENTED | — |
| Stop loss | NOT IMPLEMENTED | `stop_loss` is recorded, never acted on |
| Take profit | NOT IMPLEMENTED | No field, no logic |
| Daily loss limit | NOT IMPLEMENTED | — |
| Drawdown limit | NOT IMPLEMENTED | — |
| Volatility constraints | NOT IMPLEMENTED | ATR is available as data, unused as a constraint |
| Liquidity constraints | NOT IMPLEMENTED | — |
| Long-only enforcement | NOT IMPLEMENTED | `Position.quantity` is documented as "negative is short" (`portfolio.py`) with no guard anywhere |

The "risk" in this repo is a **debate**, not an engine: three LLM personas
(`agents/risk_mgmt/`) argue, and the Portfolio Manager decides. That is a
reasoning device. It cannot be relied on to enforce a limit, because a prompt is
not a constraint.

**Assessment: NOT IMPLEMENTED.** This is the largest safety gap: a suggestion
from an LLM is currently indistinguishable from a permitted trade.

---

## 8. Data / point-in-time assessment

The PIT work here is the strongest part of the repository, and it is
concentrated in `dataflows/date_window.py`:

- `as_of(requested, trade_date)` clamps any model-supplied date to the run date
  and canonicalises it.
- `as_of_window` clamps a window's end and refuses an unreadable start.
- `in_window` trims dated items to the window and drops undated items for
  historical runs.
- `coverage_gap` reports an unreachable window as *unavailable*, not *empty* —
  avoiding a fabricated absence.
- `withhold_live_profile`, `withhold_undisclosed_trades`,
  `withhold_undated_statements` refuse to serve data with no historical vintage.

Each source is now classified in `dataflows/pit.py`, at three levels, and the
tests read the cited guard so a classification cannot go stale:

| Source | Level | Evidence |
|---|---|---|
| yfinance OHLCV | REQUEST_BOUNDED | `agents/tools.py:get_stock_data` clamps the window end via `as_of_window` before the vendor is called. Adjusted prices mean a historical price is the split-adjusted value, not the price on the tape. |
| yfinance indicators | REQUEST_BOUNDED | Computed from the same clamped window, so no future bar enters. |
| FRED | REQUEST_BOUNDED | `vendors/fred.py:get_macro_data` bounds `observation_end` and pins the vintage with `realtime_start`/`realtime_end` (`fred.py:204`), so a later revision does not appear in an earlier run. |
| SEC EDGAR | WINDOW_FILTERED | Served as filed, which is the right shape; a statement with no filing date is withheld (`withhold_undated_statements`) rather than assumed to predate the run. |
| yfinance news | WINDOW_FILTERED | Dated items filtered to the window; undated items withheld from a historical run; `coverage_gap` reports an unreachable window as unavailable. |
| Reddit / StockTwits | WINDOW_FILTERED | Same feed limitation; filtered on the item's own timestamp, unreachable windows reported. |
| company profile | NOT_PIT_SAFE | Present-day snapshot with no historical view; withheld entirely from a historical run. The withholding is the mitigation, not a guarantee. |
| insider transactions | NOT_PIT_SAFE | Not dated by public disclosure, so withheld from a historical run. |
| Polymarket | NOT_PIT_SAFE | No date parameter to clamp at all; a backtest must not use it. |

Two caveats are recorded in code rather than smoothed over: the price clamp lives
in the agent tool layer, so a direct vendor call would bypass it, and no vendor
has a clamp of its own. That is why the backtester reads bars from
`MarketDataFeed` rather than calling a vendor.

`dataflows/pit.py` also provides a `SnapshotSource` protocol and
`register_snapshot_source`, so a source that gains a dated archive is promoted to
`REQUEST_BOUNDED` in one place without touching its callers. None is registered,
because none of the current vendors has one, and the tests assert that rather
than leaving it implied.

**Assessment: PASS for the clamping/withholding discipline and for stating each
source's limit in code; PARTIAL overall** because news, social and SEC sources
are filtered rather than served as they stood, and three sources cannot be served
point-in-time at all. The difference from the baseline audit is that the limit is
now recorded per source and checkable, instead of being a general caveat.

---

## 9. Testing assessment

Baseline at `3e44faa`: **1412 passed, 1 skipped, 1 deselected** in ~15s
(`pytest -q`, network disabled by an autouse fixture in `tests/conftest.py`).

Strengths:

- The suite refuses network access by default (`conftest.py`), so tests
  are hermetic.
- Config and file locations are isolated per test (`conftest.py`).
- There is an end-to-end graph test with scripted models
  (`tests/test_graph_end_to_end.py`).
- Prompt-integrity and layering tests assert architectural invariants
  (`tests/test_prompt_integrity.py`, `tests/test_layering.py`).

Gaps, which this work must close:

- No accounting tests (no accounting to test).
- No execution tests.
- No risk-enforcement tests.
- No fee / slippage / spread tests.
- No walk-forward or date-leakage tests.
- No performance-metric tests.

---

## 10. Security assessment

| Check | Status | Evidence |
|---|---|---|
| No secrets in the repo | PASS | `.gitignore` ignores `.env`, `.env.enterprise`; only `.env.example` is tracked |
| Ticker used in a path is validated | PASS | `safe_ticker_component` (`dataflows/symbols.py`), used in `trading_graph.py`, `cli/run.py`, `backtest.py` |
| No live credentials or broker keys | PASS | None present |
| HTML report escapes model output | PASS | CommonMark with raw HTML off + CSP (`report_html.py`) |
| No network from the execution path | PASS (vacuously) | No execution path exists yet |
| Log/telemetry must not persist secrets | NOT IMPLEMENTED | No telemetry exists; this must be a requirement for Phase 11 |

**Assessment: PASS today.** The risk is prospective: adding an execution engine
must not introduce a path to a live venue or leak credentials into telemetry.

---

## 11. Performance assessment

`iter_grid` × `propagate` runs the **full agent graph per cell**. At 4 analysts,
2 debate rounds and 3 risk rounds, one cell is tens of LLM calls. A 2-ticker,
9-date grid is ~18 full graph runs. There is no batching, no cache reuse across
cells, and no cost ceiling.

| Aspect | Status |
|---|---|
| Per-cell cost bound | NOT IMPLEMENTED |
| Cross-cell cache reuse | PARTIAL (vendor cache only, `dataflows/vendors/yahoo/ohlcv.py`) |
| Parallel cells | NOT IMPLEMENTED |
| Token accounting persisted | NOT IMPLEMENTED |

---

## 12. Cost assessment

LLM cost is invisible. `cli/stats_handler.py` counts calls and tokens for the
live display, but nothing persists them, nothing estimates cost, and nothing
stops a run that is burning budget.

| Requirement | Status |
|---|---|
| Model/provider recorded per call | NOT IMPLEMENTED |
| Input/output/total tokens persisted | NOT IMPLEMENTED |
| Latency, retries, errors recorded | NOT IMPLEMENTED |
| Estimated cost | NOT IMPLEMENTED |
| `max_run_cost` / `max_llm_calls` / `max_retries` budgets | NOT IMPLEMENTED |

**Assessment: NOT IMPLEMENTED.**

---

## 13. Recommended architecture

Keep the multi-agent brain exactly as it is. Add a deterministic spine
downstream of it, so the LLM proposes and code decides:

```
                 ┌──────────────────────┐
                 │     Market Data      │   dataflows/ (unchanged)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   TradingAgents AI   │   agents/ + graph/ (unchanged)
                 │       Brain          │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Signal Normalization │   signals/          (new)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │    Risk Engine       │   risk/             (new)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   Order Engine       │   execution/        (new)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Execution Simulator  │   execution/        (new)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Portfolio / Ledger   │   portfolio/        (new)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Performance Engine   │   performance/      (new)
                 └──────────────────────┘
```

Supporting modules: `backtesting/` (portfolio engine + walk-forward),
`telemetry/` (LLM cost and budgets), `reproducibility.py` (run metadata).

Design rules for the new layers:

1. **Long-only, spot/equity semantics.** Shorts are rejected deterministically.
2. **No network in the execution path.** Prices arrive as data, never fetched
   from the engine.
3. **No live trading.** The execution interface has `BACKTEST` and `PAPER`
   modes only; any other mode is a construction error.
4. **Every state change is an event.** The ledger is auditable by replay.
5. **The LLM never sizes or prices.** It proposes a direction; the risk engine
   sizes, the execution engine prices.
6. **Determinism where it is achievable.** Given the same prices, signals and
   config, the ledger, execution and metrics are bit-reproducible. LLM sampling
   is documented as non-deterministic.

---

## 14. Development roadmap

| Phase | Deliverable | Module | Status after this work |
|---|---|---|---|
| 1 | Validated signal contract | `signals/` | implemented |
| 2 | Portfolio ledger + events | `portfolio/` | implemented |
| 3 | Order & execution simulator | `execution/` | implemented |
| 4 | Deterministic risk engine | `risk/` | implemented |
| 5 | Fees, slippage, spread in the ledger | `execution/costs.py` | implemented |
| 6 | Portfolio/event-based backtester | `backtesting/` | implemented |
| 7 | Performance analytics + benchmark | `performance/` | implemented |
| 8 | Walk-forward / out-of-sample | `backtesting/walkforward.py` | implemented |
| 9 | PIT integrity review + provider abstraction | `dataflows/pit.py` + docs | implemented, with per-source caveats |
| 10 | Reproducibility run metadata | `telemetry/run.py` | implemented |
| 11 | LLM cost telemetry + budgets | `telemetry/` | implemented |
| 12 | Paper-trading abstraction | `execution/paper.py` | implemented |
| 13 | Reports with decision/execution/portfolio/performance | `trading_report/` | implemented |

Two roadmap entries named a module that does not exist in the delivered tree
(`reproducibility.py`, `walk_forward.py`); the corrected paths are above. The
reports module is `trading_report/` rather than an extension of `reporting.py`,
because `reporting.py` is the agent-written report tree and the two answer
different questions.

See `docs/FINAL_DEVELOPMENT_REPORT.md` for what was actually delivered.

---

## 15. Adversarial review of the delivered system

The implementation above was re-reviewed adversarially against the actual code
rather than against its own passing suite. The suite passing 1,899 tests was
treated as a claim to check, not as evidence of correctness. Each item below was
reproduced with a script before it was fixed, and each fix carries a regression
test. These are the defects the suite did not catch.

| # | Defect | Severity | Status |
|---|---|---|---|
| 1 | NaN/inf price, quantity or fee silently corrupted the ledger (`cash`, `equity` became NaN; `<= 0` guards are false for NaN) | High | Fixed |
| 2 | NaN/inf market price produced an order marked `FILLED` with a NaN fill | High | Fixed |
| 3 | Re-submitting an already-filled order filled it a second time | High | Fixed |
| 4 | `StrategyContext.feed` exposed future bars (`ctx.feed.bar(sym, tomorrow)` returned tomorrow) | High | Fixed |
| 5 | A non-finite bar was accepted at feed construction and raised mid-backtest | Medium | Fixed |
| 6 | `Quote` accepted NaN/inf prices in paper mode | Medium | Fixed |
| 7 | Position/exposure cap can be exceeded when the fill gaps up from the decision price | Medium | Fixed (round 2) |

Defect 7 was left open in the first review round, on the reasoning that the honest
fix is a fill-time re-check against the actual fill price — a deliberate design
change rather than a patch — so the behaviour was pinned by a test and recorded as
a known limitation. That change was then made. `RiskEngine.review_fill` re-runs the
cap arithmetic at the projected fill price, and both the backtester and the paper
session call it immediately before submitting a buy. A fill that would breach a cap
is reduced to the largest size that holds, and refused only when no positive size
remains. The ledger is never touched by the re-check, so no fee is charged on an
unfilled quantity and no event is created for it.

Before the fix, a 20% cap on a 100,000 book sized 200 units on a 100 close and
filled them at a 200 open: a position worth 40% of equity. After the fix the same
case fills 100 units, worth 20%. The same defect existed in paper mode, where
spread and slippage move the fill off the quote, and is fixed there too. The
full account is in `docs/PR1_INDEPENDENT_REVIEW.md`; the tests are
`TestFillTimeRecheck`, `TestPositionCapEnforcedAtFill` and
`TestPaperCapEnforcedAtFill`.

Two things were checked and found sound, and are worth recording because they are
easy to get wrong:

- **Accounting conservation holds exactly.** `cash + cost_basis + total_fees -
  realized_pnl == initial_cash` was verified to floating-point zero across mixed
  buys, a partial exit, and a full round trip, and fees are counted once
  (`total_fees` = 20 for a two-leg trade), not double counted.
- **The performance formulas are correct.** Sharpe, Sortino, max drawdown, CAGR,
  annualised return, win rate, profit factor, average win/loss and expectancy were
  each recomputed by hand and matched to machine precision.

One earlier concern was withdrawn. The branch is *graph-wise* behind `main` by one
commit, but `git diff main HEAD` on the shared paths is empty: the PR's tree is a
strict superset of `main`'s, so merging it cannot revert anything. A content diff,
not the commit graph, is what decides that, and the content diff is clean.
