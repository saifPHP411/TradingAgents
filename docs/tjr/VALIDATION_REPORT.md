# TJR strategy validation — historical study

Generated (UTC): 2026-11-05T13:39:14.796626+00:00

This study runs the **unchanged** TJR rule chain over real Yahoo intraday bars under the paper account's own risk limits. No rule, stop, target or limit was changed to improve the result.

## 1. Data window (what the vendor actually served)

- Universe: AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AMD, AVGO, JPM
- Sessions: **21** (2026-10-07 .. 2026-11-04, ET)
- Yahoo serves 1m bars for only the last ~30 days and 5m bars for the last ~60; the real window is this short, not the 6-12 months asked for
- **This is ~1 month, not the 6–12 months requested.** The limit is the vendor's; no bar was invented to cover the gap, and the study is not claimed to be long-horizon.

### Data-quality audit

- Sessions aligned across all symbols: **True** (union 21 sessions)
- 1m vs 5m agreement (each 5m bar against the five 1m bars that build it):
  - AAPL: 1637 bars, max close rel-diff 0.00e+00
  - AMD: 1637 bars, max close rel-diff 0.00e+00
  - AMZN: 1637 bars, max close rel-diff 0.00e+00
  - AVGO: 1637 bars, max close rel-diff 0.00e+00
  - GOOGL: 1637 bars, max close rel-diff 0.00e+00
  - JPM: 1638 bars, max close rel-diff 0.00e+00
  - META: 1637 bars, max close rel-diff 0.00e+00
  - MSFT: 1637 bars, max close rel-diff 0.00e+00
  - NVDA: 1637 bars, max close rel-diff 0.00e+00
  - TSLA: 1637 bars, max close rel-diff 0.00e+00
- Sessions with fewer than 390 one-minute bars: {'AAPL': ['2026-10-07'], 'AMD': ['2026-10-07'], 'AMZN': ['2026-10-07'], 'AVGO': ['2026-10-07'], 'GOOGL': ['2026-10-07'], 'META': ['2026-10-07'], 'MSFT': ['2026-10-07'], 'NVDA': ['2026-10-07'], 'TSLA': ['2026-10-07']}

## 2. Risk limits (unchanged, applied to the shared book)

- Source: `tradingagents.risk.paper_parity.paper_parity_limits()`
- max_order_notional = 25.0 USD, max_open_positions = 5, max_daily_orders = 10, allow_short = False, fractional_shares = True
- Applied to: the shared portfolio book, not each symbol separately

## 3. Cost scenarios (shared portfolio)

| scenario | trades | net P/L (USD) | net return | B&H return | win rate | profit factor | max DD | Sharpe | Sortino | exposure | turnover | rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero | 197 | 0.02 | +0.000% | -5.871% | +37.563% | 1.003 | +0.002% | 0.005 | 0.004 | +0.028% | 0.099 | 3 |
| project | 197 | -7.86 | -0.008% | -5.871% | +32.487% | 0.375 | +0.009% | -1.818 | -1.325 | +0.028% | 0.099 | 3 |
| high | 197 | -11.80 | -0.012% | -5.871% | +23.350% | 0.235 | +0.013% | -2.607 | -1.831 | +0.028% | 0.099 | 3 |

The Buy & Hold return is an equal-weight basket of the traded symbols at 1m closes, over the same index as the equity curve (see `equal_weight_benchmark`).

Rejection reasons (closed vocabulary):
- **zero**: {'MAX_OPEN_POSITIONS': 3}
- **project**: {'MAX_OPEN_POSITIONS': 3}
- **high**: {'MAX_OPEN_POSITIONS': 3}

Winning / losing / flat sessions (by each session's closing equity change):
- **zero**: 10 / 10 / 0
- **project**: 2 / 18 / 0
- **high**: 0 / 20 / 0

- **zero**: frictionless (no costs), for comparison only
- **project**: the project's frozen schedule: 5 bps commission, 1 bp half-spread, 2 bps slippage (the baseline/paper protocol's own costs)
- **high**: a harsher assumption: 5 bps commission, 2 bps half-spread, 5 bps slippage, to show the result's sensitivity to friction

Costs are not hidden: the frictionless figure is the highest and is a comparison only; the project schedule and the harsher schedule both reduce the result, which is the expected direction.

## 4. Per-symbol results (each symbol alone, project costs)

| symbol | trades | net P/L | net return | B&H return | win rate | avg planned R:R | avg realized R |
|---|---|---|---|---|---|---|---|
| AAPL | 21 | -0.69 | -0.001% | +4.979% | +28.571% | 1.597 | -1.186 |
| MSFT | 21 | -0.61 | -0.001% | +5.176% | +33.333% | 1.725 | -0.896 |
| NVDA | 19 | -1.16 | -0.001% | -16.012% | +21.053% | 1.661 | -1.152 |
| AMZN | 19 | -0.65 | -0.001% | +4.784% | +36.842% | 1.760 | -0.895 |
| META | 21 | -1.60 | -0.002% | -13.818% | +28.571% | 1.723 | -0.955 |
| GOOGL | 20 | -0.93 | -0.001% | -16.397% | +25.000% | 1.607 | -1.050 |
| TSLA | 20 | -0.12 | -0.000% | +1.693% | +45.000% | 1.512 | -0.515 |
| AMD | 20 | -1.05 | -0.001% | -24.353% | +30.000% | 1.430 | -0.694 |
| AVGO | 20 | -0.57 | -0.001% | -12.875% | +40.000% | 1.617 | -0.630 |
| JPM | 19 | -0.54 | -0.001% | +8.117% | +36.842% | 1.584 | -0.543 |

## 5. Out-of-sample read (chronological development / test split)

Method: one contiguous chronological cut at 50% of the 21 sessions; the rules were fixed before the run, so the test period is not used to choose anything

| period | sessions | net P/L | net return | trades | win rate | profit factor | max DD | win/loss days |
|---|---|---|---|---|---|---|---|---|
| development | 10 | -3.75 | -0.004% | 93 | +32.258% | 0.377 | +0.005% | 1/8 |
| test | 11 | -4.11 | -0.004% | 104 | +32.692% | 0.373 | +0.004% | 1/9 |

## 6. Robustness

- Look-ahead probe (AAPL, cut 2026-11-04T17:30:00+00:00): the full feed has 21 entries; 20 of them are entered at or before the cut, and the feed truncated at the cut enters the same 20 at the same time, size and price. Identical: **True**. (1 trade was open at the cut and force-closed by the truncation, so only the entry is compared.)
- Look-ahead probe (MSFT, cut 2026-11-04T17:30:00+00:00): the full feed has 21 entries; 21 of them are entered at or before the cut, and the feed truncated at the cut enters the same 21 at the same time, size and price. Identical: **True**. (1 trade was open at the cut and force-closed by the truncation, so only the entry is compared.)
- Survivorship bias: **not measured.** The ten symbols exist today; names that delisted are not in the universe, and no data source here can add them.
- Cost sensitivity: the three scenarios above span frictionless to a harsher-than-project schedule, so the result's dependence on friction is visible.
- The rules were fixed before this run and no parameter was tuned on the test period, so the test read is not selected. With this many sessions it is still descriptive, not statistically significant.

## Prior reports (kept for comparison)

These are the earlier committed TJR runs, over a shorter window. They are shown beside this study, not merged with it.

| report | window | trades | net return | win rate | profit factor | Sharpe | fees |
|---|---|---|---|---|---|---|---|
| real_data_backtest.json | 2026-10-15..2026-11-04 | 137 | +0.001% | +36.496% | 1.190 | 0.300 | zero |
| real_data_backtest_costs.json | 2026-10-15..2026-11-04 | 137 | -0.000% | +36.496% | 0.921 | -0.146 | charged |

- real_data_backtest.json: costs are zero in this run; the figures are frictionless and are not a claim about net-of-fee performance

## 7. Limitations (why this is not proof of future profitability)

1. **Window.** ~1 month of intraday data (vendor cap), not 6–12 months. Too short for a stable estimate of edge.
2. **Regime.** One market regime, one season. A different regime may behave differently.
3. **Costs.** The vendor's bars carry no quoted spread; the spread/slippage here are assumptions, not measurements. A wider real spread worsens the result.
4. **Survivorship.** The universe is ten current large caps; delisted names are absent.
5. **Position cap.** A 25 USD order ceiling on a 100,000 USD book makes the strategy's footprint tiny, so absolute P/L is small by construction; the ratios, not the dollars, are what carry information.
6. **Selection.** No parameter was tuned here, but the rules themselves originate elsewhere; this study validates an implementation, it does not prove a market edge.

