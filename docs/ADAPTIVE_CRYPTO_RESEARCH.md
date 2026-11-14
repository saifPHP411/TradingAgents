# Adaptive crypto research engine

## Scope and safety
Research-only. Not imported by the paper worker, dashboard order path, or exchange
order endpoints. It emits an analysis candidate, never an order. It is long-only
Spot; bearish, high-volatility and uncertain regimes abstain.

## Process
1. Validate chronological finite OHLCV and require fresh closed bars and a known spread.
2. Calculate causal EMA, ATR, ADX, path-efficiency, RSI, prior-bar range and
   relative volume. Range and volume thresholds are shifted to exclude the signal bar.
3. Classify bullish trend, bearish trend, range, high volatility, or uncertain.
4. Activate compatible experts: bullish trend pullback; lower-range mean reversion;
   prior-range breakout with relative-volume confirmation; otherwise no-trade.
5. Reject candidates when spread/data quality or estimated round-trip cost fails.
   Stops and targets are reference levels, not exchange instructions.

## Current limitations
This first implementation does not yet consume order-book depth, funding, open
interest, liquidation flow, macro/news, or exchange-specific historical fills.
Spread and cost assumptions must be supplied/measured; defaults are not verified
Bybit fees. It does not size executable quantities. Do not connect it to orders
without a separate risk/execution integration and validation.

## Research foundations (hypotheses, not profit guarantees)
- Moskowitz, Ooi & Pedersen (2012), Time Series Momentum.
- AQR, A Century of Evidence on Trend-Following Investing.
- What drives cryptocurrency returns? A sparse statistical jump model approach.
- Bailey et al., The Probability of Backtest Overfitting.
- Optimal trend-following with transaction costs.

These works motivate candidate families and validation controls; none proves this
implementation profitable in crypto or on Bybit.

## Required gates before paper/demo integration
Causal/no-lookahead and stale-data tests; multi-symbol chronological walk-forward
across market cycles; measured Bybit fees/spread/slippage; untouched final holdout;
selection-bias analysis; buy-and-hold and cash baselines; per-symbol and per-regime
expectancy, profit factor, compounded return, drawdown, turnover and uncertainty.
