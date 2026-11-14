# Strategy baseline

This document defines the official, reproducible performance baseline for the
TradingAgents simulated trading system, states exactly what it measures, and
records the result. Its purpose is to give every future strategy change a fixed
number to be compared against, produced the same way every time.

Read the first section before quoting any number below.

## What this baseline is, and what it is not

It is a **reference strategy baseline**: one deterministic, pre-registered
rule, run through the existing backtester over a frozen dataset under a frozen
protocol, measured by the existing performance engine, and cross-checked
independently.

It is **not** a baseline of the TradingAgents LLM strategy. The LLM strategy is
the multi-agent workflow in `tradingagents/graph/`. It cannot be reduced to a
single reproducible number, for two reasons that are properties of the design and
not bugs:

- It needs model credentials. No provider key is available in this environment,
  and a run cannot be replayed without one.
- It is non-deterministic. The same inputs can produce different ratings across
  runs, so a "baseline number" for it would be one sample, not a value. The
  repository says so itself in the README: "Treat the framework as a research
  scaffold for studying multi-agent analysis, not as a strategy with a fixed,
  replicable return."

So this baseline does **not** measure the LLM strategy, and no number here should
be attributed to it. What it does is establish the measurement apparatus: a
protocol, a frozen dataset, a deterministic reference point, and an independent
check on the metrics. When the LLM strategy is eventually evaluated, it is
evaluated against this file, through the same protocol.

## Classification

**Reference baseline established, out-of-sample validated. LLM-strategy baseline
not established (blocked: model credentials and non-determinism).**

This is a research scaffold, not a production result, and not live-trading
readiness.

The blocker on the LLM strategy is now partly addressed by a separate layer,
`tradingagents/llm_evaluation/`, which captures each decision with the data window
it saw and replays it through this same engine under the same protocol. That layer
reports `LLM BASELINE BLOCKED` with the exact missing prerequisite when no genuine
run is possible, and never invents a number. It does not change anything here: the
baseline still does not measure the LLM strategy. See
[`docs/LLM_EVALUATION.md`](LLM_EVALUATION.md).

## The protocol

Every value that shapes a run lives in `tradingagents/baseline/protocol.py` as a
frozen `BaselineProtocol`, versioned by `PROTOCOL_VERSION`. The artifact records
the whole protocol, so a result always names the spec that produced it.

| Item | Value |
|---|---|
| Universe | SPY, QQQ, IWM, TLT, GLD |
| Benchmark | SPY, buy-and-hold |
| Period | 2015-01-02 to 2024-12-30 (daily bars) |
| Initial capital | 100,000 USD |
| Sizing | 20% of equity per entry, one entry per symbol at a time |
| Leverage | 1.0 (none) |
| Holding period | 40 sessions |
| Costs | 5 bps commission, 1 bp half-spread, 2 bps slippage, 5% max volume participation |
| Risk | 25% max position, 95% max exposure, 5 max open positions |
| Execution | MARKET, decided at session T's close, filled at T+1's open |
| Splits | dev 2015–2018, validation 2019–2021, test 2022–2024 (exact dates below) |
| Walk-forward | 504 train / 252 validate / 252 test sessions, step 252 |
| Seed | 20260101 (reserved; nothing draws a random number) |

The dev/validation/test boundaries are `dev_end=2018-12-31`,
`validation_start=2019-01-01`, `validation_end=2021-12-31`,
`test_start=2022-01-01`, `test_end=2024-12-30`.

## The reference strategy

`tradingagents/baseline/reference.py` fixes the *form* of the rule before any
result is seen. It is deliberately simple:

- **Entry.** On the first session a symbol is flat and its 20-session simple
  moving average is above its 100-session simple moving average, buy 20% of
  equity in it, at the next session's open.
- **Exit.** The engine's fixed 40-session holding-period exit, plus a 20%
  protective stop.
- **Sizing.** 20% of equity per position, up to five positions, no leverage.

The moving-average lengths (20/100) are conventional and were chosen a priori,
not tuned on this data. They are shorter than the textbook 50/200 because a
walk-forward test window is 252 sessions and a 200-session warmup would leave a
fold with nothing to trade; 20/100 warms up inside a fold. The exact same
strategy object is used, unchanged, in development, validation, test and every
walk-forward fold. There is exactly one candidate, so there is nothing to
overfit.

## The dataset

The dataset is **frozen and committed** under
`tradingagents/baseline/data/`, one CSV per symbol, produced once by
`python -m tradingagents.baseline.fetch_data` (the only code in this package
that touches the network). Every run reads the committed files, so the baseline
is offline and byte-identical from one machine to the next. The artifact records
the SHA-256 of each file.

Re-fetching restates split and dividend adjustments, so a re-fetch is a *new*
baseline version, not an update of this one. The checksums are what make that
explicit.

## How to reproduce

```bash
python -m tradingagents.baseline.run --out docs/baseline/official_baseline.json
```

No network, no API key. The run is deterministic: two runs differ only in the
`generated_at` wall-clock stamp. The test
`tests/test_baseline.py::TestDeterminism::test_two_full_baselines_are_identical`
asserts exactly that.

## Result

See `docs/baseline/official_baseline.json` for the machine-readable artifact.
The headline table below is copied from it. The reference strategy is a trend
filter, so it is long during uptrends and flat otherwise; that is why its return
is well below buy-and-hold in a decade that was mostly up, and why its exposure
averages below 80%.

| Metric | Development | Validation | Test |
|---|---|---|---|
| Sessions | 1,006 | 757 | 752 |
| Total return | +9.32% | +46.99% | +12.98% |
| Annualised return | +2.26% | +13.68% | +4.17% |
| CAGR | +2.26% | +13.72% | +4.17% |
| Max drawdown | 15.29% | 16.03% | 17.35% |
| Volatility (ann.) | +7.50% | +10.09% | +9.18% |
| Sharpe | 0.34 | 1.32 | 0.49 |
| Sortino | 0.29 | 1.16 | 0.46 |
| Calmar | 0.15 | 0.85 | 0.24 |
| Trades | 88 | 73 | 62 |
| Win rate | +59.09% | +67.12% | +66.13% |
| Profit factor | 1.27 | 2.97 | 1.61 |
| Exposure (mean) | 68.21% | 77.14% | 65.85% |
| Turnover | 34.55x | 29.18x | 25.14x |
| Fees | 1,733.98 | 1,752.65 | 1,205.39 |
| Benchmark return (SPY B&H) | +31.76% | +99.68% | +28.66% |
| Excess vs benchmark | -22.44% | -52.69% | -15.69% |

The independent cross-check passes for every window (`cross_check_passes: true`).

### Walk-forward

Six folds roll a 504-session training window, a 252-session validation window
and a 252-session test window across the calendar. The mean out-of-sample test
return is **+4.77%**, against a mean best-in-sample return of **+15.79%** — an
overfitting gap of **11.02 percentage points**. The selected test scores are
`-1.90%, +13.21%, +16.74%, +5.60%, -10.77%, +5.73%`. Two of the six test windows
are negative. This is the honest read: the rule generalises weakly, and the
in-sample number is not the number to trust.

### Cost sensitivity (test window)

| Scenario | Total return | Net P&L | Fees |
|---|---|---|---|
| Zero cost | +15.23% | 15,232.86 | 0.00 |
| Baseline | +12.98% | 12,977.12 | 1,205.39 |
| Double cost | +10.76% | 10,764.22 | 2,386.74 |
| Quadruple cost | +6.46% | 6,463.97 | 4,679.11 |

Costs are material but not fatal: quadrupling them costs about half the net P&L.

### Concentration and regime (test window)

- Net P&L 12,977.12, gross profit 38,225.66. The single best trade is 5.97% of
  gross profit, the best three 15.71%, the best five 24.81%. The result is not
  one lucky trade.
- GLD contributes the most (7,182.59); the largest single trade is 2,280.19.
- By the benchmark's trailing trend: 508 sessions bull (+33.74% cumulative),
  244 bear (-15.53% cumulative). The rule does what a trend filter should — it
  earns in uptrends and gives some back in downtrends — and is not a hedge.

### How to read this

This is a **reference point**, not a recommendation. It underperformed
buy-and-hold on every window, which is the expected shape of a simple long-only
trend filter in a mostly-rising decade. Its value is that the number is fixed,
reproducible, independently cross-checked, and out-of-sample. A future strategy
must beat *this file*, on the same protocol and the same data, to have earned a
claim of improvement.


## What the artifact contains

- `protocol` — every value that shapes the run, plus the dataset checksums.
- `strategy` — id, version, kind, and the fixed parameters.
- `results` — per window (development, validation, test): return, annualised
  return, CAGR, max drawdown and its dates, volatility, Sharpe, Sortino, Calmar,
  trade counts, win rate, profit factor, expectancy, exposure, turnover, fees,
  benchmark return, excess return, and the rejection/unfilled counts.
- `independent_cross_check` — the headline metrics recomputed by
  `tradingagents/baseline/crosscheck.py`, which shares no code with the
  performance engine, and whether the two agree.
- `regime_analysis` — the strategy's return bucketed by the benchmark's
  trailing trend and volatility, computed from trailing data only.
- `concentration` — how much of the gross profit comes from the best trades and
  from one symbol.
- `cost_sensitivity` — the test window rerun at zero, baseline, double and
  quadruple costs.
- `walk_forward` — the rolling folds, their selected test scores, and the
  in-sample-versus-out-of-sample gap.

## Independent cross-check

`tradingagents/baseline/crosscheck.py` recomputes total return, net P&L, max
drawdown, trade counts, win rate, profit factor and total fees from the equity
curve, the trade list and the ledger event log, with plain arithmetic that
shares no code with `performance/metrics.py`. The two must agree, to a relative
tolerance of 1e-6, or `cross_check_passes` is `False` and the baseline is not
publishable. This catches an off-by-one, a double annualisation, or a drawdown
measured from the wrong peak.

## What the numbers may and may not be used for

May be used for:

- Comparing a future strategy change against a fixed, reproducible reference.
- Confirming the accounting and metric pipeline produces a stable, sane result
  on real data.
- A worked example of the full loop: decision, risk, execution, ledger, P&L.

May **not** be used for:

- Any claim about the TradingAgents LLM strategy. This baseline does not measure
  it.
- Any claim of profitability, production readiness, or live-trading readiness.
- Any inference beyond the universe, period and assumptions above. Five ETFs
  over ten years, at one cost assumption, is a small sample.

## Known limitations

- **The LLM strategy is not baselined.** Blocked on credentials and
  non-determinism, as above. This is the single largest gap. A decision-replay
  layer (`docs/LLM_EVALUATION.md`) captures and replays recorded decisions through
  this same engine, but it is price-only and does not measure the news, sentiment
  or fundamentals agents.
- **No slippage model beyond a fixed bps and a volume cap.** Real fills depend
  on the book, which the simulator does not model.
- **Daily bars only.** No intraday path exists to test.
- **No borrow, short or leverage.** The reference is long-only and unlevered, so
  the baseline says nothing about a strategy that uses them.
- **One candidate.** The walk-forward selection is trivial with a single
  candidate; the value is in the disjoint test windows, not in the selection.
- **Regime and concentration analyses are descriptive.** They describe this run,
  not a forecast.
- **Cost sensitivity is a sensitivity test.** It shows dependence on the cost
  assumption; it is not a tuning step and nothing feeds back into the strategy.

## How to compare a future strategy

1. Load the same protocol and the same frozen dataset.
2. Run the candidate through `PortfolioBacktester` over the same windows.
3. Produce an artifact with the same schema (`build_baseline`).
4. Compare window by window: return, max drawdown, Sharpe, fees, turnover, and
   the benchmark-relative excess. A candidate is not "better" on return alone;
   the drawdown and the cost sensitivity matter as much.
5. Compare the walk-forward `selected_test_scores`, not the in-sample scores.
   The `overfitting_gap` is the number to watch.
