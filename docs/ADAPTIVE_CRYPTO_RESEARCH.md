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
as a fallback. The 1m fallback emits only hourly buckets containing all 60 minute
candles; partial hours are discarded. The walk-forward runner validates every common
timestamp against an explicit expected interval (1h by default), so uniformly skipped
bars cannot silently masquerade as complete hourly history. The research workflow fetches a 2-year hourly history for ten liquid
USDT pairs, while the legacy benchmark cache retains its 120-day 1m sample.
Binance history is only a cross-venue proxy for Bybit Demo Spot.

## Fee-aware and passive-execution research additions

- The entry gate now requires the modeled gross reward distance to exceed the
  estimated round-trip cost by a configurable multiple (default 3x), in addition
  to the minimum net reward/risk ratio. The same gate is rechecked at the next
  simulated open so a gap cannot silently invalidate the setup.
- Position sizing includes the estimated round-trip cost in risk per unit.
- Range mean reversion now has rolling Bollinger lower-band and rolling
  volume-weighted-average features, while retaining the ADX/path-efficiency
  range regime filter. The VWAP here is a rolling OHLCV proxy, not exchange
  session VWAP or order-book microprice.
- `tradingagents/crypto/backtest/bybit_execution.py` builds a Bybit V5
  `Limit`/`PostOnly` payload and rounds quotes to tick size. It is a pure
  helper: it does not authenticate, submit, amend, cancel, or poll orders.
  Buy quotes rest at/below best bid; sell quotes rest at/above best ask.
- A PostOnly instruction prevents taking liquidity at submission, but it does
  not guarantee a fill, a maker fee tier, or a profitable exit. OHLCV alone
  cannot model queue position, partial fills, cancellations, adverse selection,
  or maker-to-taker conversion at exit. Do not count a hypothetical passive
  quote as filled without order-book/trade replay and conservative queue rules.
- Commission inputs remain explicit scenario assumptions. They must be replaced
  with the account's actual Bybit fee schedule and instrument category before
  any venue-specific conclusion; VIP tier rates are not hardcoded.
- Funding-rate cash-and-carry is deferred. It requires synchronized Spot and
  Perpetual order books, historical funding timestamps/rates, basis, borrow or
  inventory costs, two-leg partial-fill handling, and liquidation/margin modeling.
  A positive funding rate alone is not a risk-free arbitrage signal.

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

## Leakage-safe online learning overlay

The walk-forward study also runs a research-only online expert-learning overlay.
Each symbol has an independent learner. It receives only closed trades from
completed earlier OOS folds, measured after the selected cost scenario. It
never sees outcomes from the fold currently being evaluated. During the first
20 matured outcomes per expert it remains in shadow/exploration mode. Thereafter
it blocks an expert only when the lower 95% confidence bound of its mean net
trade return is not positive. The learner has bounded observation memory and
requires timezone-aware chronological timestamps.

The study reports two independent learning overlays: one trained and evaluated
using realistic-cost outcomes, and one trained and evaluated using conservative-
cost outcomes only. Their learners are separate per symbol and per cost scenario;
conservative labels are never mixed with realistic labels. Both are updated only
after each fold closes. This tests whether the gating effect survives a harsher
cost assumption, but it is not nested hyperparameter selection and does not create
an untouched holdout.

These overlays may abstain from an expert; they cannot change strategy parameters,
position sizing, risk fractions, notional caps, stops, targets, or cost
assumptions. They do not train on open or hypothetical trades, and have no
exchange, worker, or order integration. Learning-overlay reports are evaluated
separately from the fixed baseline and are not evidence of a live learning
advantage. Any future model/parameter adaptation must use nested chronological
selection and a genuinely unseen future holdout; the already inspected historical
sample is not a locked final holdout.


## Verified current-head evidence (2026-11-16)

The following values were read from the JSON artifacts uploaded by the completed
GitHub Actions run for commit
`5017336dd403d7a3561b6a84170383cd30d5e64a` (run `43879827452`). They are
simulated research results on Binance public Spot candles, not Bybit fills.

### Adaptive study: 10 symbols, 82 chronological folds

| Scenario | Compounded OOS return | Aggregate PF | Trades | Positive folds | Max fold-return DD |
|---|---:|---:|---:|---:|---:|
| Fixed baseline, realistic costs | -0.1130% | 0.9899 | 730 | 25/82 | 1.9385% |
| Fixed baseline, conservative costs | -0.5153% | 0.9403 | 501 | 21/82 | 1.8491% |
| Online learning, realistic outcomes | +0.5528% | 1.0972 | 435 | 24/82 | 0.8749% |
| Online learning, conservative outcomes | +0.1140% | 1.0240 | 305 | 16/82 | 0.9778% |
| Frictionless baseline (diagnostic only) | +1.3998% | 1.1293 | 972 | 33/82 | 1.3969% |

The learning overlays use only prior completed-fold outcomes and do not change
execution costs or risk limits. Their aggregate figures are positive, but only
24/82 realistic-cost folds and 16/82 conservative-cost folds were positive.
The baseline remains negative after costs, results vary substantially by symbol,
and the learning overlay is not yet robust enough to justify Demo selection.
The historical sample has already been inspected and is not an untouched final
holdout. These figures do not establish a profitable edge.

### TJR quick screen: BTC/USDT, 14-day 1m history

The exact current-head artifact contains 15 trades. With the configured realistic
cost assumptions, final equity was 99,811.59 from 100,000 (approximately -0.1884%),
profit factor 0.1834, with 1 winning and 14 losing trades. Under conservative
costs, final equity was 99,676.91 (approximately -0.3231%), profit factor 0.0933,
with 1 winning and 14 losing trades. The frictionless result was 100,006.44
(+0.0064%) with profit factor 1.0981, showing that the tiny apparent gross edge
does not survive modeled costs. This is a short screen, not a validated strategy.

### Decision

**NO-GO for paper/Demo strategy activation.** CI passing proves the code and
research workflow completed; it does not prove profitability or venue readiness.
Before reconsideration: collect a genuinely future untouched holdout, validate
the strategy on Bybit Demo market data and instrument filters, measure actual
account fees/spread/slippage, model partial fills and order lifecycle, and pass
predefined after-cost robustness gates across symbols and market regimes.
The research branch remains isolated; none of these studies is connected to the
paper worker or dashboard order path.
