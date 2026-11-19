# Perpetual Strategy Research Report — 2026-11-17

## Executive result

**No strategy is currently eligible for Bybit Demo execution.**

A five-family development-only tournament completed over 16 disjoint chronological folds, four Bybit USDT linear perpetual symbols, 1x/2x simulated leverage, and three modeled transaction-cost scenarios. The development-only top-ranked candidate was EMA14/EMA50 + EMA200 + ADX18 at 2x. However, this is the same fixed strategy specification that had already failed the previously inspected independent final holdout at both 1x and 2x. It is therefore disqualified from reselection, despite its positive development-fold results.

After applying that prior-holdout exclusion, no remaining candidate passes all realistic, conservative, and stress gates. No candidate is selected and the leveraged execution path remains disabled.

## Scope and methodology

- Venue/data: Bybit V5 public market data; linear USDT perpetuals only.
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT.
- Timeframe: completed 4-hour candles.
- History request: 730 days, aligned to common timestamps across all symbols.
- Development-only boundary: strictly before the \`final_holdout_start\` timestamp recorded in the earlier EMA walk-forward report. That previous final holdout and every later candle were excluded from this tournament.
- Walk-forward: 16 disjoint chronological folds; 260-bar warmup; 200 test bars per fold.
- Leverage tested: 1x and 2x. The research engine still caps leverage at 2x and defaults to 1x.
- Candidates: EMA14/50 ADX18; EMA9/30 ADX22; EMA20/60 ADX16; Donchian20 ADX18; Donchian55 ADX18. All use an EMA200 regime filter and ATR trailing exit.
- Donchian channels are shifted one candle, so a candle cannot contribute its own high/low to the breakout threshold used to evaluate that same candle.

### Modeled costs

| Scenario | Commission | Half-spread | Slippage |
|---|---:|---:|---:|
| Realistic estimate | 5.5 bps | 1.0 bps | 2.0 bps |
| Conservative | 8.0 bps | 2.0 bps | 5.0 bps |
| Stress | 12.0 bps | 4.0 bps | 10.0 bps |

These are modeled assumptions, not measured account-specific Bybit fills. Funding costs are incorporated from the public funding-rate history. Liquidation accounting is approximate and is not a replacement for Bybit risk-tier calculations.

### Development acceptance gates

A candidate must pass every gate in every cost scenario:

- At least 30 trades.
- Profit Factor at least 1.20 (realistic), 1.05 (conservative), and 1.00 (stress).
- Positive equal-weight compounded return.
- At least 60% of development folds positive.
- At least 75% of symbols positive.
- Worst fold drawdown no greater than 10%.
- Zero simulated liquidations.

Passing a development gate is not independent validation.

## Prior independent final-holdout result: EMA14/EMA50

The following previously inspected final-holdout report applies to the same fixed EMA14/EMA50 + EMA200 + ADX18 strategy specification. Every row failed its original acceptance gate:

| Leverage | Scenario | Win rate | Return | Profit Factor | Trades | Gate |
|---|---|---:|---:|---:|---:|---|
| 1x | Realistic estimate | 23.5% | -0.02% | 0.968 | 17 | Fail |
| 1x | Conservative | 23.5% | -0.07% | 0.910 | 17 | Fail |
| 1x | Stress | 23.5% | -0.14% | 0.823 | 17 | Fail |
| 2x | Realistic estimate | 23.5% | -0.16% | 0.855 | 17 | Fail |
| 2x | Conservative | 23.5% | -0.23% | 0.801 | 17 | Fail |
| 2x | Stress | 23.5% | -0.27% | 0.768 | 17 | Fail |

This old holdout was not re-run or used to score alternative candidates. It is used only to prevent the already-rejected EMA14/EMA50 specification from being promoted again.

## Development tournament results

All values below are from the 16 development folds only. Returns are equal-weight compounded across chronological fold returns; Win Rate and Profit Factor are pooled trade metrics. Values are rounded to three decimals.

| Candidate | Lev. | Cost model | Win rate | Return | PF | Trades | Positive folds | Positive symbols | Liquidations | Scenario gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| EMA14/50 ADX18 | 2x | Realistic | 44.578% | +2.088% | 1.720 | 83 | 10/16 | 4/4 | 0 | Pass development only; prior holdout rejects |
| EMA14/50 ADX18 | 2x | Conservative | 44.578% | +1.921% | 1.660 | 83 | 10/16 | 4/4 | 0 | Pass development only; prior holdout rejects |
| EMA14/50 ADX18 | 2x | Stress | 41.975% | +1.086% | 1.363 | 81 | 10/16 | 3/4 | 0 | Pass development only; prior holdout rejects |
| EMA14/50 ADX18 | 1x | Realistic | 40.541% | +1.082% | 1.458 | 74 | 10/16 | 3/4 | 0 | Pass development only; prior holdout rejects |
| EMA14/50 ADX18 | 1x | Conservative | 40.541% | +0.952% | 1.397 | 74 | 10/16 | 3/4 | 0 | Pass development only; prior holdout rejects |
| EMA14/50 ADX18 | 1x | Stress | 38.356% | +0.378% | 1.150 | 73 | 10/16 | 3/4 | 0 | Pass development only; prior holdout rejects |
| EMA20/60 ADX16 | 1x | Realistic | 41.250% | +1.116% | 1.511 | 80 | 8/16 | 4/4 | 0 | Fail |
| EMA20/60 ADX16 | 1x | Conservative | 41.250% | +0.887% | 1.391 | 80 | 8/16 | 4/4 | 0 | Fail |
| EMA20/60 ADX16 | 1x | Stress | 40.000% | +0.518% | 1.210 | 80 | 7/16 | 4/4 | 0 | Fail |
| EMA20/60 ADX16 | 2x | Realistic | 41.111% | +1.740% | 1.547 | 90 | 8/16 | 4/4 | 0 | Fail |
| EMA20/60 ADX16 | 2x | Conservative | 41.573% | +1.176% | 1.369 | 89 | 8/16 | 4/4 | 0 | Fail |
| EMA20/60 ADX16 | 2x | Stress | 40.449% | +0.417% | 1.123 | 89 | 7/16 | 2/4 | 0 | Fail |
| Donchian55 ADX18 | 1x | Realistic | 39.490% | +1.035% | 1.180 | 157 | 8/16 | 2/4 | 0 | Fail |
| Donchian55 ADX18 | 1x | Conservative | 38.065% | +0.799% | 1.140 | 155 | 8/16 | 2/4 | 0 | Fail |
| Donchian55 ADX18 | 1x | Stress | 36.774% | +0.378% | 1.067 | 155 | 8/16 | 1/4 | 0 | Fail |
| Donchian55 ADX18 | 2x | Realistic | 39.655% | +0.747% | 1.111 | 174 | 8/16 | 2/4 | 0 | Fail |
| Donchian55 ADX18 | 2x | Conservative | 37.791% | +0.416% | 1.064 | 172 | 8/16 | 2/4 | 0 | Fail |
| Donchian55 ADX18 | 2x | Stress | 35.088% | -0.046% | 0.997 | 171 | 8/16 | 1/4 | 0 | Fail |
| Donchian20 ADX18 | 1x | Realistic | 41.827% | +1.620% | 1.213 | 208 | 10/16 | 3/4 | 0 | Pass this scenario only |
| Donchian20 ADX18 | 1x | Conservative | 40.097% | +1.058% | 1.137 | 207 | 9/16 | 3/4 | 0 | Fail |
| Donchian20 ADX18 | 1x | Stress | 37.255% | -0.157% | 0.984 | 204 | 9/16 | 2/4 | 0 | Fail |
| Donchian20 ADX18 | 2x | Realistic | 41.558% | +2.670% | 1.281 | 231 | 10/16 | 3/4 | 0 | Pass this scenario only |
| Donchian20 ADX18 | 2x | Conservative | 40.000% | +1.690% | 1.176 | 230 | 9/16 | 3/4 | 0 | Fail |
| Donchian20 ADX18 | 2x | Stress | 37.168% | -0.326% | 0.971 | 226 | 9/16 | 2/4 | 0 | Fail |
| EMA9/30 ADX22 | 1x | Realistic | 27.451% | -0.549% | 0.717 | 51 | 5/16 | 1/4 | 0 | Fail |
| EMA9/30 ADX22 | 1x | Conservative | 27.451% | -0.643% | 0.679 | 51 | 5/16 | 1/4 | 0 | Fail |
| EMA9/30 ADX22 | 1x | Stress | 24.000% | -0.796% | 0.615 | 50 | 5/16 | 1/4 | 0 | Fail |
| EMA9/30 ADX22 | 2x | Realistic | 28.070% | -0.907% | 0.642 | 57 | 6/16 | 0/4 | 0 | Fail |
| EMA9/30 ADX22 | 2x | Conservative | 28.070% | -0.999% | 0.614 | 57 | 6/16 | 0/4 | 0 | Fail |
| EMA9/30 ADX22 | 2x | Stress | 25.000% | -1.188% | 0.552 | 56 | 6/16 | 0/4 | 0 | Fail |

## Interpretation

1. EMA14/50 is the apparent development winner, but it is the same strategy family and fixed specification that failed the earlier independent final holdout at both leverage levels. It is disqualified from selection.
2. Donchian20 at 2x produced the highest realistic development return (+2.670%), but its stress result was negative (-0.326%) with PF 0.971. It is not robust to the modeled stress costs.
3. EMA20/60 at 1x remained positive across all three scenarios and had a minimum scenario PF of about 1.210, but only 7–8 of 16 folds were positive. It fails the 60% fold-consistency requirement.
4. EMA9/30 lost money in all scenarios. Donchian55 did not clear the required PF/breadth gates.
5. No un-rejected candidate passes all the gates across every cost model. The maximum development Win Rate belongs to the already-rejected EMA14/50 candidate (44.578%); no tested alternative approaches a 79% Win Rate target.

## Deployment and execution status

- Isolated research service is configured as perpetual-only; the legacy Spot/equity and paper-worker stack is not initialized in that mode.
- The dashboard's public Bybit market chart is read-only. The optional Demo position reader is GET-only and does not submit or amend orders.
- No order has been submitted. No account leverage has been changed. No strategy is Demo-ready.
- Original production service was left unchanged.
- GitHub CI run associated with this report was still running when the report was prepared; do not treat the full CI matrix as passing until it reaches a terminal success state.

## Next valid research step

Do not tune the same 16 development folds again and claim independent validation. Freeze a non-rejected candidate, wait for a new chronological period of data after the current sample, then evaluate that frozen candidate on a fresh untouched holdout with realistic, conservative and stress costs. Until that new validation passes, keep execution disabled.
