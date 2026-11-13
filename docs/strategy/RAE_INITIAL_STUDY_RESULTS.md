# RAE Initial Historical Study — Findings and Decision

**Status: research only. Not approved for Bybit Demo or live execution.**

## Dataset and execution assumptions

- Binance public spot OHLCV, completed 1-hour candles only.
- Coverage per symbol: 17,519 bars, from 2024-11-13 08:00 UTC through 2026-11-13 06:00 UTC.
- Symbols: BTC/USDT, ETH/USDT, BNB/USDT.
- Chronological partitions, independently capitalized per symbol:
  - Development: 60% (10,511 bars), 2024-11-13 to 2026-01-25.
  - Validation: 20% (3,504 bars), 2026-01-25 to 2026-06-20.
  - Holdout: 20% (3,504 bars), 2026-06-20 to 2026-11-13.
- Signals are evaluated on completed bars and executed at the next bar open. Stops are gap-aware and checked against intrabar lows.
- Starting capital is 10,000 USDT per symbol and partition. Base costs are 10 bps fee and 5 bps slippage per side. Stress costs multiply both by 1.5.
- Each symbol is a separate single-symbol account. Results are not added into a synthetic portfolio.

## Ensemble results

Net return and maximum drawdown are percentages. The table shows the ensemble variant, not the separate trend/range/breakout variants.

| Partition | Symbol | Trades | Win rate | Profit factor | Net return | Max DD | Net return at 1.5x costs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | BNB | 2 | 0.0% | 0.00 | -0.164% | 0.208% | -0.194% |
| Development | BTC | 10 | 10.0% | 0.006 | -0.873% | 0.917% | -1.021% |
| Development | ETH | 3 | 66.7% | 2.04 | +0.253% | 0.381% | +0.208% |
| Validation | BNB | 1 | 0.0% | 0.00 | -0.130% | 0.130% | -0.145% |
| Validation | BTC | 2 | 50.0% | 10.68 | +0.280% | 0.214% | +0.249% |
| Validation | ETH | 1 | 0.0% | 0.00 | -0.093% | 0.093% | -0.108% |
| Holdout | BNB | 1 | 0.0% | 0.00 | -0.111% | 0.114% | -0.126% |
| Holdout | BTC | 2 | 50.0% | 304.86 | +0.104% | 0.137% | +0.074% |
| Holdout | ETH | 0 | — | — | 0.000% | 0.000% | 0.000% |

The holdout BTC profit factor is mathematically high but comes from only two trades; it is not meaningful evidence of a repeatable edge. Across all three holdout symbols, the ensemble produced only three trades. The separate recent-30-day run produced zero ensemble trades on all three symbols.

## Unchanged TJR 30-day reference

The TJR rule chain was run separately on 30 days of 1-minute market data. It is not an apples-to-apples comparison with the RAE 1-hour study; it is retained only as a same-project reference.

- Frictionless: 93 trades, PF 1.22, net +70.42 USDT on 100,000 USDT.
- Realistic costs: 93 trades, PF 0.061, net -207.49 USDT; fees 924.83 USDT and slippage 277.45 USDT.
- Conservative costs: 93 trades, PF 0.024, net -575.07 USDT; fees 1,381.54 USDT and slippage 644.72 USDT.

The frictionless result is not actionable. Under the modeled realistic and conservative costs, this TJR reference loses money.

## Decision

**RAE v1 fails the research acceptance gate.** Its trade sample is far too small to support conclusions, development results are weak/negative on BTC and BNB, and the recent 30-day sample has no activity. The attractive holdout BTC PF is a two-trade statistic and must not be treated as proof.

1. Do not connect RAE to the paper worker, do not deploy it to Railway, and do not route orders through it.
2. Keep the existing TJR implementation unchanged as a reference baseline.
3. Do not tune against the already inspected validation/holdout partitions. The next candidate version must be developed using development data only and evaluated on a newly reserved future holdout.
4. Improve signal coverage and regime logic without simply loosening every filter. Add trade-count and activity diagnostics, then test causal trend-pullback/time-series-momentum and range modules separately.
5. Require a materially larger out-of-sample sample, positive net expectancy after realistic costs, robustness across symbols and rolling folds, and 1.5x cost stress before any activation discussion.

This report records the first study outcome; it is a rejection of the current candidate for deployment, not a claim that every regime-adaptive approach is ineffective.
