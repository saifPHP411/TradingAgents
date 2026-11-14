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

## Backtest and execution assumptions
The research backtester uses closed hourly bars, next-hour-open entries, stop-first
intrabar handling, gap-aware stop exits, time exits, and explicit per-fill commission,
half-spread and slippage scenarios. Simulated quantity is bounded by risk-per-trade
and notional caps. The backtest is single-position per symbol; it is not a shared
multi-asset execution portfolio and it does not create executable order quantities.
The CI study prefers Binance public Spot 1h candles, with completed 1m aggregation
as a fallback. The research workflow fetches a 2-year hourly history for ten liquid
USDT pairs, while the legacy benchmark cache retains its 120-day 1m sample.
Binance history is only a cross-venue proxy for Bybit Demo Spot.

## Current limitations
The engine does not yet consume order-book depth, funding, open interest, liquidation
flow, macro/news, or exchange-specific historical fills. Spread and cost assumptions
are explicit estimates, not verified Bybit fees or realized market impact. Do not
connect it to orders without separate risk/execution integration and validation.

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
across multiple market cycles; measured Bybit fees/spread/slippage; untouched final holdout;
selection-bias analysis; buy-and-hold and cash baselines; per-symbol and per-regime
expectancy, profit factor, compounded return, drawdown, turnover and uncertainty.
