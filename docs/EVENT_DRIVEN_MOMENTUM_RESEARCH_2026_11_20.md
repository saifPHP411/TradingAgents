# Frozen event-driven research: volatility shock versus hourly momentum and reversal

**Study date:** 20 November 2026. **Branch:** `research/independent-weekly-demo-audit-20261120`.
**Research only:** no orders, deposits, Demo enablement, changes to leverage, or Railway deployment.

## Why these three distinct hypotheses?

Academic research identifies potentially regime-dependent intraday time-series momentum
and reversal; it also finds that the apparent value of technical signals can disappear
under trading costs. These papers are *hypothesis generators*, not evidence that the
specific thresholds below work:

- [Dynamic time series momentum of cryptocurrencies](https://www.sciencedirect.com/science/article/pii/S1062940821000590)
- [Bitcoin intraday time-series momentum](https://doi.org/10.1111/fire.12290)
- [On the Performance of Lagged Momentum and Reversal Strategies Across Daytime and Overnight Sessions in Bitcoin and Ethereum Cryptocurrencies](https://www.mdpi.com/1911-8074/19/9/692)
- [Technical analysis in cryptocurrency markets: Do transaction costs and bubbles matter?](https://www.sciencedirect.com/science/article/pii/S1042443122000816)
- [Momentum in the Cryptocurrency Market: A Comprehensive Analysis under Realistic Assumptions](https://academic.oup.com/raps/advance-article/doi/10.1093/rapstu/raag011/8914819)

The three configurations were specified *before* reviewing their October–November
results, and the additional September weeks were requested without adjusting
their thresholds to fit those initial results.

## Frozen candidates (no parameter sweep)

All triggers are evaluated on **completed** UTC minute candles and enter at
the **next minute's open**, with estimated adverse fill impact.

| Candidate name | Trigger on completed bar | Position | Stop / target | Max hold |
|---|---|---|---|---|
| `volatility_spike_5m_long_12h` | 5m absolute return >=0.8%, >2× mean abs 5m return of previous twelve 5m blocks, 5m volume >=1.2× prior-hour typical 5m volume | LONG only | 2% / 3.5% | 720 minutes |
| `hourly_impulse_momentum_6h` | Absolute 60m return in [1.2%, 8%], hour volume >=1.2× average of four preceding hourly volumes | Direction of impulse (long/short) | 1.5% / 3% | 360 minutes |
| `hourly_impulse_reversal_6h` | Exactly same trigger as momentum | Opposite impulse (long/short) | 1.5% / 3% | 360 minutes |

Repeated signals cannot create overlapping open positions. All four
BTCUSDT/XRPUSDT/ETHUSDT/LTCUSDT histories compete for **one shared** simulated
50-USDT account. Notional per entry is limited to 20% of account equity
(initially about 10 USDT), with a 2× leverage policy ceiling carried over
from the existing research simulator. A position must close at the weekly
boundary and every independent seven-day experiment restarts at 50 USDT.

Modeled stop and take-profit collisions are resolved **stop-first**, including
the entry minute. Gaps through a stop receive the worse observed open, plus
modeled fill impact. Ledger fees and realized P&L are reconciled to account
equity. Every week and every losing trade is preserved.

Cost scenarios use the existing fixed standard:
zero-cost diagnostic upper bound (not executable), realistic estimate at
5.5 bps taker commission + 3 bps adverse impact *per side*, and stress at
8 bps commission + 7 bps impact per side. **No fee-free win is interpreted
as executable.**

## First frozen six-week comparison (4 October + 2 November)

Realistic modeled costs, 50 USDT reset per week:

| Study | Total modeled trades | Pooled net win rate | Positive weeks | Best weekly net profit |
|---|---:|---:|---:|---:|
| 5m volatility-shock long | 45 | 35.56% | 1/6 | +0.373 USDT |
| 1h impulse continuation | 46 | 50.00% | 3/6 | +0.378 USDT |
| 1h impulse reversal | 48 | 41.67% | 2/6 | +0.146 USDT |

Verified GitHub Actions evidence:
[Exploratory weekly candidate replay #44763660142](https://github.com/saifPHP411/TradingAgents/actions/runs/44763660142).

**None achieved the minimum +50-USDT weekly net-profit objective.**
The comparatively highest positive-week count among these three was the
hourly-impulse continuation, but it lost money in three other weeks. Its
best week gained only about 0.38 USDT, not 50 USDT.

## Additional frozen-period check

The updated `.github/workflows/weekly-candidates.yml` adds a four-week
study from **6 September through 3 October 2026 UTC**, before the existing
October and November samples, giving 10 complete nonoverlapping weekly
experiments. The historical relative-value benchmark
`.github/workflows/weekly-pairs.yml` is extended to the same period.

The new crosscheck downloads the three chronological scan artifacts into
`weekly_goal_bridge.py`, which records the `weekly_net_profit_usdt` for
every week, the count meeting **>=50 USDT net** regardless of win rate,
and a separate count additionally meeting the retained **>=70%**
winning-trade preference. It also produces an exploratory ranking by
**median weekly net USD after estimated realistic costs**, with
worst-week results and positive-week counts. This is an **in-sample ranking,
not untouched holdout or independent proof**.

## Critical restrictions

- No proof of real Bybit IOC fills, order-book adverse selection or
  instrument minQty/minNotional/qtyStep. Public Bybit instruments-info from
  previous GitHub runners returned HTTP 403. A nominal 10-USDT entry is
  not independently confirmed to be executable.
- Funding, tiered liquidation and mark-price-triggered stops are not
  modeled/verified. The historical source is official Binance USD-M klines,
  not Bybit executed trades. Daily snapshots understate intraday drawdown.
- 50 USDT of **net weekly profit from a 50-USDT account is +100% per week**.
  Neither a strong historical win rate nor 2× metadata makes this
  target reliably achievable, and it is not a basis for escalating risk.
- No candidate becomes Demo eligible without verifying quantity filters,
  funding, liquidation, prospective evidence, correct trade ledger, and
  meaningful net profitability **after** transaction costs.

The full existing objective is in
[`WEEKLY_50_USDT_NET_PROFIT_OBJECTIVE.md`](WEEKLY_50_USDT_NET_PROFIT_OBJECTIVE.md).
