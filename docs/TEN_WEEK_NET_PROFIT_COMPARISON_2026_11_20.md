# Ten-week frozen strategy comparison — $50 net/week research objective

**Study completed:** 20 November 2026 (research only). **Current objective:**
at least **50 USDT of NET account profit per full seven-day period** from
a simulated starting balance of 50 USDT (the superseded 20x weekly objective
is not used here).

## Auditable original evidence

- [Nine directional families, ten complete independent weeks — GitHub Actions #44764449180](https://github.com/saifPHP411/TradingAgents/actions/runs/44764449180)
- [Two paired long/short families, ten weeks — GitHub Actions #44764449117](https://github.com/saifPHP411/TradingAgents/actions/runs/44764449117)
- Exact source: `tradingagents/crypto/backtest/weekly_candidates.py`
  and `weekly_pairs.py`; independent-account reconciliation and
  fees/venue-evidence audit: `weekly_goal_bridge.py` and `weekly_goal.py`.
- The dated input windows are 6 Sep–3 Oct 2026 (4 weeks),
  4–31 Oct 2026 (4 weeks) and 1–14 Nov 2026 (2 weeks), all UTC.
- Source: historical public **Binance USD-M perpetual** 1m OHLCV
  ZIP archives, not actual Bybit prices, fills, order-book executions
  or confirmed Bybit Demo account balances.

Every strategy was run on a **single simulated 50-USDT portfolio** across
four symbols (or a fixed two-leg pair), with each seven-day week independently
reset to 50 USDT. Directional order notional was capped at 20% of simulated
equity; paired strategies split that combined notional between legs. The
research policy used a 2x leverage ceiling without increasing exposure
to force a backtest result.

Three cost scenarios were compared for every strategy and period:
zero-cost optimistic upper bound (**not executable**), realistic estimated
taker commission 5.5 bps + adverse price impact 3 bps **each side**, and
stress 8 bps + 7 bps each side. The table below is **REALISTIC ESTIMATE ONLY**.
All entries use the full ten-week record, including every loss.

## Exploratory ranking — median USDT net profit per week

**Ranking is retrospective descriptive research**, not untouched
out-of-sample validation. Independent weeks each start at 50 USDT.
Each row is one strategy applied to all ten weekly periods.

| Rank | Strategy | Positive weeks | Median weekly net USDT | Average weekly net USDT | Worst weekly net USDT | Pooled net win rate | Modeled closed trades |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Donchian 15m Trend | 5/10 | **+0.0385** | +0.0170 | -0.9171 | 44.12% | 170 |
| 2 | Hourly impulse momentum 6h | 4/10 | -0.1616 | -0.2367 | -1.1220 | 40.82% | 98 |
| 3 | Hourly impulse reversal 6h | 4/10 | -0.2150 | -0.0794 | -1.2770 | 42.27% | 97 |
| 4 | VWAP 15m reclaim | 1/10 | -0.2450 | -0.2032 | -0.3679 | 31.93% | 119 |
| 5 | 5m volatility spike long, 12h hold | 2/10 | -0.3529 | -0.3601 | -1.1769 | 35.05% | 97 |
| 6 | EMA60 RSI pullback | 0/10 | -0.8141 | -0.7765 | -1.3422 | 33.91% | 516 |
| 7 | XRP/LTC dollar-neutral pair | 0/10 | -1.2883 | -1.2707 | -1.6521 | 24.84% | 781 |
| 8 | BTC/ETH dollar-neutral pair | 0/10 | -1.5519 | -1.5566 | -1.8361 | 4.42% | 905 |
| 9 | EMA9/21 RSI momentum | 0/10 | -4.6434 | -4.5684 | -5.8660 | 23.54% | 2783 |
| 10 | Donchian20 1m breakout | 0/10 | -5.2963 | -5.4110 | -7.2733 | 22.21% | 3093 |
| 11 | Bollinger RSI 1m reversion | 0/10 | -5.5920 | -5.5361 | -5.9934 | 25.67% | 3397 |

**0/110** realistic strategy-week trials earned >=50 USDT of net profit.
**0/330** total strategy × cost scenario × week trials met that goal,
including unrealistic zero-cost upper-bound studies. The paired and
directional strict account-evidence audits succeeded technically, and
none of the modeled strategies met the required weekly objective.

The highest-ranked **Donchian 15m** was profitable in only 5/10
weeks. Its median weekly net gain was **less than four cents**, and the
worst week lost about 92 cents. A +$0.04 historical median on a $50
account is **not** a meaningful edge nor a way to obtain $50/week.
The three additional event studies and extra September window did not
justify moving any configuration to trading.

## Anti-overfitting and limitations

Signals were frozen before evaluating their first October–November
six-week comparison and remained unchanged when the four September
weeks were added. However, strategies had already been explored on
related data, and the ranking is now sample-selected. Do **not** claim
independent holdout validity, forward profitability, or statistical
significance without genuinely unseen future periods and a realistic
multiple-testing protocol.

The expected source/venue gap remains unresolved:

- Official Bybit public instruments-info/ticker endpoints returned
  HTTP 403 from previously tested GitHub Actions runners. Current
  minimum notional, lot step, quantity step, price filters and actual
  available collateral are **unknown**. A modeled ~$10 notional entry
  and particularly a ~$5 paired leg may not be executable.
- Actual two-leg atomic execution is not supported by this research
  replay; pair trades could fill only one leg. Book depth, partial
  IOC fills, mark-price stops, funding, and venue liquidation tiers
  are not independently validated.
- Historical Binance OHLCV is *not* a Bybit Demo fill transcript.
  Fills here rely on modeled next-minute open, stop-first candle tie
  handling, estimated adverse price impact and commission assumptions.
- Daily equity snapshots can miss intraday drawdown and margin events.
  No strategy changes to real accounts, Demo settings, leverage,
  broker credentials, Railway deployments, or order placement were made.

## Research decision

No configuration among the eleven examined earned the requested
**+50 USDT net/week** from 50 USDT, and none passed an exchange-execution
readiness gate. Donchian 15m ranks highest *only within the losing/near-flat
experimental set*; it is **not approved** for deployment or a profit
forecast.

Next independent studies should test a fundamentally new return
mechanism (e.g. longer-horizon relative-strength allocation or a
fully priced funding/basis trade), verify executable minimum order
size before using tiny notional positions, and avoid retuning rules
on these now-seen weeks. Only previously unseen periods or prospective
Demo observations can test the final strategy's generalization.

Related frozen hypothesis details:
[`EVENT_DRIVEN_MOMENTUM_RESEARCH_2026_11_20.md`](EVENT_DRIVEN_MOMENTUM_RESEARCH_2026_11_20.md).
