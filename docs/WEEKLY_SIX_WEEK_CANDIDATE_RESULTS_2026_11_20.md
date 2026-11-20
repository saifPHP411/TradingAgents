# Six-week read-only candidate evaluation — 2026-11-20

## Decision

**None of six tested long/short strategy families satisfies either the
repeatable 70% net win-rate requirement or the 50→1,000 USDT seven-day
return target. No strategy is approved for Demo execution.**

This is an *exploratory* historical-candle study, not a prospective
out-of-sample validation or actual Bybit execution test. It is not
evidence that the tested strategies must fail in every market.

## Recorded source runs

- Four nonoverlapping weeks, October 4–31, 2026:
  [GitHub Actions 44732366754](https://github.com/saifPHP411/TradingAgents/actions/runs/44732366754),
  job `october-4weeks`.
- Two nonoverlapping weeks, November 1–14, 2026:
  [same run](https://github.com/saifPHP411/TradingAgents/actions/runs/44732366754),
  job `november-2weeks`.
- Full run JSON artifacts `research-weekly-candidates-october-4weeks`
  and `research-weekly-candidates-november-2weeks` preserve individual
  closed-trade simulated P&L and UTC daily equity.
- Synthetic tests: `tests/test_weekly_candidates.py`; the CI success
  status must be checked against the latest PR head SHA independently.

## Predeclared simulation assumptions

| Setting | Value |
|---|---|
| Source | Binance USD-M USDT perpetual daily 1m historical ZIPs, not Bybit |
| Symbols | BTCUSDT, XRPUSDT, ETHUSDT, LTCUSDT |
| Evaluation | Six disjoint weeks; one shared account reset to 50 USDT each week |
| Action | Completed minute or completed 5m/15m confirmation; enter next 1m open |
| Equity exposure | Maximum notional per position = 20% of account equity |
| Concurrent positions | 1 |
| Leverage ceiling reported | 2x; no extra notional leverage beyond 20% account exposure |
| Execution model | 5.5 bps commission plus 3.0 bps adverse price impact *per side* |
| Alternative assumptions | Zero cost (non-executable upper bound), and stress (8+7 bps per side) |
| Exits | Fixed stop/TP, conservative stop-first for same-bar collisions; time exit |
| Trading orders submitted | 0 |

Crucially, Binance's public candle ZIPs do **not** verify Bybit order
fills, tick and lot-step constraints, minimum notional, bid–ask liquidity,
funding, order rejection, or liquidation risk tiers. Funding and quantity
filters in this standalone scan are not applied, so realized Demo trading
can differ considerably. Trading performance below is purely simulated.

## Net-of-modeled-cost results

Pooled net win rate is trade-weighted across six independent 50-USDT
weeks. A "profitable week" means final equity greater than that week's
50-USDT start, not a claim of statistical significance.

| Candidate | Closed trades (6 weeks) | Pooled net win rate | Profitable weeks / 6 | Weekly net return percentages, chronological |
|---|---:|---:|---:|---|
| EMA9/EMA21 RSI momentum | 1,633 | 21.56% | 0/6 | -9.38, -8.46, -8.56, -9.84, -9.19, -11.73 |
| Bollinger RSI reversion | 1,992 | 25.20% | 0/6 | -9.81, -11.74, -10.35, -11.70, -10.34, -10.95 |
| Donchian20 breakout | 1,820 | 21.65% | 0/6 | -11.46, -9.26, -11.89, -9.06, -10.73, -11.28 |
| EMA60 RSI pullback (5m) | 300 | 33.67% | 0/6 | -2.27, -2.68, -0.45, -1.43, -1.68, -1.58 |
| Donchian 15m trend | 94 | 45.74% | 3/6 | +0.79, +1.75, +0.84, -1.07, -0.21, -0.31 |
| Anchored VWAP 15m reclaim | 66 | 31.82% | 1/6 | -0.09, -0.70, -0.48, +0.19, -0.58, -0.68 |

### Important detail about apparent 70% winners

Donchian 15m achieved **72.73% (8 wins/11 trades)** in the second
October week and **72.22% (13 wins/18 trades)** in the third October
week, with only +1.75% and +0.84% weekly account growth, respectively.
Both **fail** the 30-closed-trade sample threshold and **fail** the
+1,900% net growth requirement. In the next three chronological weeks
its net win rate was materially lower, including October's -1.07%
fourth-week return. The best-week result is not a viable 70% strategy.

### How much did fees matter?

The same recorded candles and frozen candidates were also replayed
with *zero* trading costs. The zero-cost case is not executable and
must never be presented as net return.

- EMA9/21 was positive in only 1/6 zero-cost weeks.
- Bollinger RSI was positive in only 3/6 zero-cost weeks.
- Donchian20 was positive in only 2/6 zero-cost weeks.
- EMA60 pullback was positive in only 3/6 zero-cost weeks.
- Donchian 15m was positive in 5/6 zero-cost weeks, **but only 3/6
  after modeled costs** and far below 20x returns.
- VWAP 15m was positive in 2/6 zero-cost weeks.

This indicates trading friction is substantial, but no tested family
has an independently demonstrated advantage large enough to achieve
the user's simultaneous goals.

## Research interpretation and next test

The results **reject these six specific fixed configurations under
the stated model**; they do not prove impossibility for all strategies.
The clear next technical improvement is *execution realism*, not merely
changing the stop/TP parameters to fit the inspected weeks:

1. Validate Bybit quantity/step/minimum notional and genuine fill
   feasibility for the 50-USDT / 20%-notional sizing.
2. Use the L2 replay in PR #20 for realistic fill caps and slip
   assumptions; record actual 1m-mark signal timestamps where possible.
3. Freeze any **new, structurally distinct** strategy specification
   before a genuinely untouched prospective evaluation.
4. Compare on shared-account equity with no deposits and all costs,
   full weekly sample counts, losses and trade-level audit evidence.
5. Keep all trading execution disabled until account/venue validation
   and an independent Demo sign-off.

**No profitable 20×/70% strategy has been found, implemented or
validated by this research. No PR merge, live-trading action or Railway
deployment was performed.**
