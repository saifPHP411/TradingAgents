# Frozen 24h/4h cross-sectional trial — 15 consecutive weeks (research only)

**Date:** 20 November 2026. **Draft PR:** #22 on
`research/independent-weekly-demo-audit-20261120`.
**Current user objective:** >=50 USDT net profit per complete seven-day week,
starting from 50 USDT, with no deposits. An earlier 20× weekly target was
superseded and is no longer evaluated.

## Predeclared hypothesis

Distinct from the nine intraday single-symbol candidates and the two
two-leg relative-value studies, this study ranks BTCUSDT/ETHUSDT/XRPUSDT/
LTCUSDT by their prior **completed 24h** return at UTC 00:00 and applies a
completed **4h** direction check. Next-minute-open entry, at most one
position per UTC day, single shared 50-USDT account, independent weekly
capital resets. No leverage or risk increases.

- `daily_relative_winner_long`: buy the strongest 24h return if >=1.5%
  and its preceding 4h return is positive.
- `daily_relative_loser_short`: short the weakest 24h return if <=-1.5%
  and its preceding 4h return is negative.
- `daily_relative_loser_rebound_long`: buy the weakest if <=-3% on the
  prior 24h but its completed 4h return is at least +0.2%.

**Frozen exits:** first stop at 4% adverse price movement, take profit
at 8% favorable price movement, mandatory exit by end of the same UTC
day. Stop wins any same-minute OHLC target/stop collision, including
the entry minute; adverse gaps through stop use the worse opening price.
At most one 20%-of-equity notional entry per day, one position at a time.
Final and daily balance checkpoints are flat. Entry fee is taken out
of account equity at the open; the full net P&L of the closed trade
reconciles to account equity.

Three frozen scenarios: zero-cost optimistic upper bound,
realistic modeled 5.5 bps taker commission + 3 bps adverse impact **per side**,
stress at 8 bps + 7 bps. Historical candles are **official public
Binance USD-M 1-minute archived OHLCV**, not actual Bybit book fills.

## Evidence

[GitHub Actions — 15-week cross-sectional replay and independent-account
audit #44777119809](https://github.com/saifPHP411/TradingAgents/actions/runs/44777119809).

**All 15 consecutive, non-overlapping 7-day periods** were included:
2 August through 14 November 2026 UTC, inclusive, spanning 5 August-into-
September weeks, 4 subsequent September weeks, 4 October weeks and 2
November weeks. August was a previously unevaluated historical month
for these rules, **not a prospective forward holdout**.

Results for realistic estimated costs only; every weekly account is reset
to **50 USDT**, so these are independent experiment P&L values rather than
one continuously funded account:

| Frozen hypothesis | Positive weeks | Median weekly net USDT | Mean weekly net USDT | Best / worst weekly net | Modeled trades | Pooled net win rate |
|---|---:|---:|---:|---:|---:|---:|
| Best 24h winner LONG + 4h confirm | 5/15 | -0.02024 | +0.07179 | +1.91696 / -0.48738 | 21 | 47.62% |
| Worst 24h loser SHORT + 4h confirm | 3/15 | -0.12358 | -0.08260 | +1.18557 / -0.72773 | 27 | 29.63% |
| Worst 24h loser LONG after 4h rebound | 3/15 | 0.00000 | +0.02097 | +0.20941 / 0.00000 | 3 | 100.00% (only 3 trades) |

The strict account audit examined **3 hypotheses × 3 cost scenarios × 15
weeks = 135 week-model evaluations**. **ZERO** generated +50 USDT of net
profit in any week, including the unrealistic zero-cost upper bound.
The rebound candidate's 100% sample accuracy comes from just **three
small winning trades across fifteen weeks**; 12 weeks produced no trades.
It is *not* robust or deployable.

## Analytic feasibility check — reject before expensive data scans

These **specific** daily candidates cannot possibly reach the user's
+$50/week objective **under their frozen simulator rules**, regardless
of the historical data. Even if all seven trades in a week hit an
8% take-profit, with maximum position notional of only 20% of the
current account, and ignoring **all** commissions, adverse impact,
funding and losing trades:

```text
Max weekly profit (unrealistic perfect-win bound):
$50 × [(1 + 0.20 × 0.08)^7 − 1] = $5.87608379
```

That is less than $6, far below the desired $50 weekly profit. This
arithmetic bound applies **only to this model** (one entry per day,
20%-of-equity notional, 8% take-profit exit), not to every possible
crypto trading strategy. It is now exported by
`weekly_relative_strength.theoretical_weekly_net_profit_ceiling_before_costs()`
and asserted in an automated regression test, so no one can claim the
candidate qualifies by cherry-picking candles. Do **not** raise the
leverage or order-size caps automatically to force a result.

## Limitations and responsible decision

- **No proven +$50 weekly net strategy.** None of the fifteen weeks
  reached the objective; median returns were nonpositive.
- Historical 1m trades assume next-bar execution. They do not verify
  Bybit Demo order minima, quantity steps, real stop triggering, actual
  IOC fills, funding, liquidation tiers, mark-price conditions,
  instantaneous margin drawdown or price impact.
- The earlier Bybit public-instruments proof failed with HTTP 403 in
  the tested GitHub runner. This is a missing-evidence blocker,
  *not* evidence of an executable 50-USDT account.
- This is research with frozen hypotheses and archived historical
  periods, **not independently validated forward performance**.
  Comparisons to the pre-existing 11 models are descriptive and
  cannot establish persistent superiority.
- No Bybit Demo or real order was submitted. No deposit, withdrawal,
  PR merge, Railway deployment or policy/leveraged-risk change.

### Follow-up research policy

Before adding any candidate, compute whether its maximum gross
trade opportunity under the current capital, max notional and take-profit
rules can even theoretically approach **+$50 net/week**. Reject
mechanically impossible configurations without running expensive
backtests. For feasible candidate designs, require independent
chronological holdouts with realistic costs and verified minimum
order size, position/stop behavior, funding and margin requirements.
Allow **no trade** when evidence is inadequate.

Academic sources motivating *the hypothesis*, not endorsing this setup:

- [Momentum in the Cryptocurrency Market: A Comprehensive Analysis under
  Realistic Assumptions](https://academic.oup.com/raps/advance-article/doi/10.1093/rapstu/raag011/8914819)
- [Cross-cryptocurrency return predictability](https://www.sciencedirect.com/science/article/pii/S0165188924000551)
- [Cryptocurrency anomalies and economic constraints](https://ideas.repec.org/a/eee/finana/v94y2024ics1057521924001509.html)
