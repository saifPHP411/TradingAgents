# Crypto strategy research catalog

This is a hypothesis library for research, not a list of guaranteed profitable setups.
Every method must be defined mechanically, tested with causal data, and evaluated
after realistic fees, spread, slippage, funding (when applicable), and market impact.
A strategy is eligible only in the market state where its economic premise is plausible.

## 1. Trend and momentum families

| Family | Core hypothesis | Useful evidence/features | Failure modes to test |
|---|---|---|---|
| Time-series momentum | An asset's own medium-horizon return can persist | Multi-horizon returns, moving-average slope, volatility scaling | Sideways chop, crash/reversal, crowded exits |
| Cross-sectional momentum | Relative winners can continue outperforming losers | Rank assets by volatility-adjusted returns; rebalance on fixed schedule | Universe survivorship, turnover, correlation spikes |
| Trend following / channel systems | Persistent directional moves can be captured with delayed confirmation | Donchian channels, moving-average structure, ADX, path efficiency | Whipsaw, late entry, gap risk |
| Pullback continuation | Retracements within an established trend may offer better entry geometry | Higher-timeframe trend, swing structure, EMA/VWAP reclaim, volume normalization | Pullback becomes reversal; subjective swing labeling |
| Breakout / volatility expansion | A close outside a prior balance area can precede price discovery | Prior-bar range, ATR compression/expansion, relative volume, close location | False breakouts, news gaps, thin books |

## 2. Mean-reversion and relative-value families

| Family | Core hypothesis | Useful evidence/features | Failure modes to test |
|---|---|---|---|
| Range mean reversion | Price excursions inside a stable range can revert toward fair value | ADX/efficiency, rolling range, RSI, Bollinger z-score | Trend transition; repeated averaging down |
| VWAP reversion | Price displaced from volume-weighted average may normalize | Session VWAP, anchored VWAP, deviation bands, volume | 24/7 session definition; no stable anchor |
| Statistical pairs / spread trading | Related assets may temporarily diverge from a stable relationship | Cointegration tests, rolling hedge ratio, spread z-score | Relationship breaks, borrow/short constraints, fees |
| Cross-asset relative strength | Capital rotates among assets; leaders may persist | BTC-relative returns, sector/universe ranks, correlation regime | Benchmark regime changes, thin altcoin liquidity |
| Market-neutral baskets | Long/short exposure can isolate relative returns | Factor exposure, beta neutrality, covariance estimates | Short availability, funding, borrow and liquidation risk |

## 3. Volatility and market-structure families

| Family | Core hypothesis | Useful evidence/features | Failure modes to test |
|---|---|---|---|
| Volatility targeting | Position risk can be normalized across volatility states | ATR/realized volatility, volatility forecasts | Volatility jumps after sizing; unstable estimates |
| Volatility breakout | Expansion after compression can produce directional follow-through | ATR percentile, Bollinger bandwidth, realized-volatility ratios | Compression can persist; direction is unknown |
| Range/market profile | Repeated acceptance/rejection zones may reveal balance areas | Volume-at-price, point of control, value area, session profile | Exchange-specific volume, profile parameter sensitivity |
| Order-flow imbalance | Aggressive buying/selling and queue changes may precede short-term moves | Trades, bid/ask depth, order-book imbalance, cancellations | Spoofing, feed latency, hidden liquidity, queue position |
| Liquidity-aware execution | Participation and order size relative to depth determine realized cost | Spread, depth, expected impact, fill ratio, adverse selection | Snapshot depth is not executable certainty |
| Funding/open-interest context | Perpetual positioning and leverage can signal crowded exposure | Funding, open interest changes, liquidations, basis | Perpetual-only data; exchange differences; reflexive events |

## 4. Risk and portfolio methods

- Fixed fractional risk: size from equity and stop distance; never increase risk because
  recent trades won.
- Volatility scaling: reduce notional when realized volatility rises; cap leverage at
  zero for this Spot-only research path.
- Correlation-aware allocation: estimate rolling correlations and prevent a portfolio
  from treating correlated coins as independent bets.
- Drawdown controls: define daily/weekly loss stops, peak-to-trough limits, and
  cool-downs before any paper execution.
- Kelly sizing: use only as a theoretical upper bound; estimation error makes full
  Kelly dangerous. Any fractional-Kelly candidate must be capped and stress-tested.
- Tail-risk checks: gap stress, clustered losses, expected shortfall, and adverse
  liquidity scenarios; stop orders do not guarantee stop-price fills.
- Portfolio constraints: max simultaneous positions, total exposure, per-symbol
  concentration, minimum order size, precision, fees, and residual dust.

## 5. Regime diagnosis before strategy selection

No single indicator reliably labels market state. Combine independent evidence:
- Direction: multi-timeframe EMA slope, swing highs/lows, return persistence.
- Trend quality: ADX plus path-efficiency / directional movement.
- Balance: low trend strength, repeated range acceptance, mean-reverting residuals.
- Volatility: ATR percentile, realized-volatility ratio, compression/expansion.
- Participation: relative volume, trade imbalance and volume-at-price when available.
- Liquidity: spread, depth, expected impact and feed freshness.
- Cross-asset context: BTC direction, broad market breadth and correlation regime.

The classifier must be causal, calibrated on training windows only, and allowed to
return uncertain/no-trade. A transition state should not be forced into a trend or range.

## 6. Validation methods that prevent self-deception

1. Chronological walk-forward: fit/choose parameters only on the past, evaluate on the
   next untouched block, then roll forward.
2. Purging and embargo: remove overlapping label/trade horizons around validation
   boundaries when samples share future information.
3. Combinatorially symmetric cross-validation / probability of backtest overfitting:
   quantify how often the selected in-sample winner ranks poorly out of sample.
4. Multiple-testing correction: report every attempted variant; do not hide failed
   configurations or repeatedly tune against the final holdout.
5. Reality-check / SPA-style comparisons: account for data snooping across a family of
   candidate rules.
6. Bootstrap uncertainty: resample trade sequences or blocks, preserving dependence
   as far as practical; report confidence intervals, not only point estimates.
7. Benchmarks: cash, buy-and-hold, simple trend, and simple range baselines on the
   same timestamps and with the same cost assumptions.
8. Execution realism: next-event/next-bar fills, stop-first intrabar ambiguity,
   gap-through stops, spread, commissions, slippage, partial fills, exchange filters,
   and measured venue-specific paper fills.
9. Robustness: symbols, market cycles, timeframes, parameter neighborhoods, regime
   buckets, and high-cost scenarios.
10. Sample adequacy: publish trade count and effective independent sample size;
    a small number of wins is not evidence of a durable edge.

## 7. Evidence references

- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*,
  https://doi.org/10.1016/j.jfineco.2011.11.003
- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*,
  https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
- Lo, Mamaysky & Wang (2000), *Foundations of Technical Analysis*,
  https://doi.org/10.1111/0022-1082.00265
- AQR, *A Century of Evidence on Trend-Following Investing*,
  https://www.aqr.com/Insights/Research/White-Papers/A-Century-of-Evidence-on-Trend-Following-Investing
- Bailey et al., *The Probability of Backtest Overfitting*,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Almgren & Chriss (2001), *Optimal Execution of Portfolio Transactions*,
  https://doi.org/10.21314/JOR.2001.041

These papers support research hypotheses and validation discipline. They do not
validate this repository's strategy, the crypto asset class, or Bybit execution.
