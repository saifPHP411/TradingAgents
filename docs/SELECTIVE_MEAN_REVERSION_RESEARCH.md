# Selective Bollinger / RSI Mean-Reversion Candidate

**Status: research-only proposal. It is not validated, Demo-ready, or connected to an order path.**

## Objective

Investigate whether a deliberately selective range mean-reversion candidate can achieve a **79% net-of-cost win rate** without sacrificing positive expectancy, profit factor, or drawdown control. The 79% figure is an acceptance target, not a forecast or claim.

A high win rate is not sufficient by itself. A system that wins many small trades but loses more on a few stops is rejected. The research screen therefore requires all of the following in realistic, conservative, and stress cost cases:

- At least 79% pooled win rate after modeled fees, spread, slippage, and funding.
- At least 30 closed trades.
- Positive average net PnL per trade and positive compounded portfolio return.
- Profit factor of at least 1.20 (realistic), 1.05 (conservative), and 1.00 (stress).
- At least 60% positive chronological folds and at least 75% positive symbols.
- Worst fold drawdown no greater than 10%, with zero simulated liquidations.
- No promotion based on development data alone; a frozen specification must subsequently pass an untouched chronological holdout.

## Frozen entry specification

The initial candidate is intentionally narrow:

- Instrument scope: Bybit USDT-settled linear perpetuals, simulated isolated margin, one-way position mode.
- Timeframe: completed 4-hour candles only.
- Volatility envelope: 20-bar simple moving average with bands at ±2.0 population standard deviations.
- Long setup: the previous close was at/below the lower band; the current completed close re-enters above the lower band; RSI(14) was at/below 32 and turns upward; ADX(14) is at/below 18.
- Short setup: symmetric upper-band re-entry; RSI(14) was at/above 68 and turns downward; ADX(14) is at/below 18.
- Execution: signal is known only after the candle closes; entry is simulated at the next 4-hour open with modeled adverse costs.
- Target: the signal candle's middle band, frozen at signal time. Reject the entry if the next open has already crossed that target after costs.
- Stop: initial 2.0 ATR(14) adverse stop. The existing liquidation-distance guard and position sizing/risk caps remain in force.
- Exit: stop before target when both are touched in the same OHLC candle (conservative ambiguity handling); otherwise use the frozen mean target, ADX regime-break exit, or 72-bar maximum holding limit.
- No adding to a losing position, martingale, averaging down, or adaptive leverage.

This is a hypothesis designed to avoid fading strong trends and to make the expected target explicit before entry. It may still fail during volatility regime changes, news shocks, persistent trends, or when costs exceed the modeled assumptions.

## Evaluation design

The candidate is added as a fixed member of the existing multi-symbol Bybit perpetual research tournament. It inherits the tournament's chronological folds, four-symbol scope, 1x/2x research comparison, historical funding, and realistic/conservative/stress execution-cost scenarios. Its parameters are not optimized against a final holdout.

Compare it to the existing fixed candidates and a no-trade/cash baseline. Preserve every failed scenario. Report pooled and per-symbol trade count, win rate, average win/loss, expectancy, profit factor, net compounded return, drawdown, funding, fees, and fold consistency. If the 79% win-rate target is reached but expectancy or return is non-positive, the candidate still fails.

A passing development screen is only permission to freeze and evaluate a genuinely untouched future holdout. It does not authorize paper or live execution.

## Safety and promotion

- The feature branch is isolated from production.
- No API credentials, authenticated account endpoints, leverage-setting endpoints, or order endpoints are used by this strategy.
- The research leverage ceiling remains 2x, with the existing sizing and liquidation checks.
- Do not connect this candidate to the paper worker or Demo execution unless it passes independent after-cost validation and a separate execution review.
- If the sample is too small to assess 79% meaningfully, report the uncertainty rather than widening the sample, lowering costs, or tuning the rules on the holdout.
