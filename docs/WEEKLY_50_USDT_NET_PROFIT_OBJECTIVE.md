# Current objective — at least 50 USDT net profit per week

**Effective decision: 20 November 2026.** The owner withdrew the former
50 → 1,000 USDT in seven days research objective. That is **not** the current
target and should not be used for any new candidate selection, readiness
decision, progress display, or reporting.

## Exact account goal

- Starting simulated account equity: **50 USDT**, no deposits or external cash.
- Required net account profit: **at least +50 USDT every complete seven-day
  interval**, after trading fees, spread, adverse fills, funding, and losses.
- First-week simulated ending equity must therefore be **at least 100 USDT**.
- For a continuously growing account, week 2 starts from the actual week-1
  closing equity, and must earn **another +50 USDT**, not another 100%.
  E.g. 50 → 100 → 150 → 200 → 250 means +50 per week, not an
  unrealistic requirement to double again every seven days.
- Separately restarting a **50 USDT research account** every week is also
  supported. Every week then needs to finish at **at least 100 USDT**,
  because its initial capital is reset only in the simulator.
- The previously requested **70% net winning-trade rate** remains a
  *secondary research screening preference*, explicitly reported apart
  from the absolute-dollar target. A profitable portfolio can have a lower
  win rate; do not suppress the net-dollar outcome when that happens.
- The existing strict combined evidence gate additionally requests at
  least 30 completed net-of-cost trades in each of four consecutive
  independent weeks, no liquidation, flat week boundaries, and verified
  ledger/equity reconciliation. The strict combined gate is *not* a claim
  that 70% wins or a fixed number of trades guarantees +50 USDT.

A 50-USDT initial account earning 50 USDT net in week one is **+100% in
seven days**, still extremely aggressive and vulnerable to total-loss
outcomes. It is a research target, not a guarantee, forecast, or
authorization to trade with leverage.

## How to audit results

Use `tradingagents/crypto/backtest/weekly_goal.py`. The
`weekly_goal_bridge.py` translates existing read-only historical scan
artifacts for **every** tested strategy, week and cost assumption into this
account audit. It separately emits:

- `weekly_net_profit_usdt` on each complete seven-day interval,
- `weekly_profit_goal_met` — >=50 USDT net account growth after
  reconciliation and without liquidation,
- `weekly_win_rate_70pct_met` — net closed-trade win-rate screen,
- `weekly_target_met` — strict joint evidence screen including minimum
  sample, win rate and required profit,
- `weeks_meeting_50usdt_profit` and
  `weeks_meeting_50usdt_profit_and_70pct` for every strategy/scenario.

The audit rejects unsupported capital flows, incomplete weeks, cherry-picked
date gaps, fake account snapshots, and trade P&L that does not reconcile to
equity. A self-reported historical quality flag is not independent proof of
exchange fills. The output never allows automatic deployment.

## Actual evidence and constraints

Earlier October–November exploratory scans found **no proven** 50-USDT
weekly net profit strategy. Six single-direction strategies and two
two-leg pair-relative-value strategies were explored using official
Binance USD-M minute *candles* and modeled trading costs. These were not
Bybit order fills. In the fixed samples, most configurations lost money;
the earlier Donchian 15m study's *best* October week earned only about
+1.75%, far below +100% on 50 USDT.

Execution constraints remain open: the public Bybit V5 instrument/ticker
endpoints tested from GitHub Actions returned HTTP 403 from both official
hosts. $10 total notional under the current 20% cap (or roughly $5 per
paired leg) may not pass minimum-order rules, but the answer is **UNKNOWN**
until reliable venue/account validation is available. Funding, liquidation
tier and genuine IOC fills were not independently validated.

Keep the existing Bybit Demo leverage ceiling and maximum notional fraction
unchanged. Do not bypass venue filters or stage/activate Demo/live trading
just to reach a numeric target. No credentials, orders, or deposits are
needed to run the research.

The old `docs/WEEKLY_20X_70PCT_OBJECTIVE.md` is preserved only as a
historical record of the **superseded** research requirement.
