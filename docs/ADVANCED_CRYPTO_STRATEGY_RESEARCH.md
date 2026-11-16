# Advanced crypto strategy research: evidence and implementation plan

**Status: research-only. No strategy in this document is approved for the paper worker or Demo execution.**

## Research conclusion

There is no defensible universal "most profitable" crypto strategy. Results depend on universe construction, venue, execution costs, market regime, turnover and validation design. Complexity is not a proxy for edge. The strongest evidence reviewed supports investigating cross-sectional return prediction conditioned on liquidity, risk and momentum, while treating high-frequency order-flow as a separate data/execution problem.

### Evidence reviewed

1. **Cross-sectional crypto machine learning (2024).** Cakici, Shahzad, Będowska-Sójka and Zaremba, *Machine learning and the cross-section of cryptocurrency returns*, International Review of Financial Analysis (2024). The study reports that simple ML models can predict cross-sectional crypto returns; price, past alpha, illiquidity and momentum are important predictors. The reported return premia are concentrated in smaller, less liquid and more volatile assets, which creates a major capacity and transaction-cost caveat. DOI: https://doi.org/10.1016/j.irfa.2024.103244
2. **Interactions among crypto predictors (2025).** Mercik, Będowska-Sójka, Karim and Zaremba, *Cross-sectional interactions in cryptocurrency returns*, International Review of Financial Analysis (January 2025). The study evaluates 40 characteristics across more than 500 crypto assets (2017–2023) and reports an out-of-sample long/short strategy Sharpe above 1 for selected interactions. Liquidity constraints and trading costs can prevent implementation. DOI: https://doi.org/10.1016/j.irfa.2024.103809
3. **Order flow and crypto returns (2026).** Anastasopoulos et al., *Order flow and cryptocurrency returns*, Journal of Financial Markets (2026). The paper reports economically useful out-of-sample prediction from world order flow and nonlinear ML, including robustness to high transaction costs. Its international order-flow inputs are not equivalent to hourly OHLCV or a single exchange's basic trade-imbalance feed. DOI: https://doi.org/10.1016/j.finmar.2026.101047
4. **Cointegration basket example (GitHub).** https://github.com/sujith-kamme/statistical-arbitrage-crypto reports a Johansen/Ornstein-Uhlenbeck basket strategy with walk-forward selection and a claimed net OOS Sharpe of 1.76 after 20 bps. Its own README says the result is based on only nine round trips in roughly 220 days; that sample is too small to establish a durable edge. The method also requires reliable short/hedge execution and stable cointegration, neither of which is supplied by the current Spot-only paper path.
5. **Order-flow replication example (GitHub).** https://github.com/Rubin-S/order-flow-imbalance reports that its event-level BTC perpetual order-flow signal predicted only a small fraction of future 1-second price movement; its estimated gross edge was below 4–10 bps round-trip fees, and it did not break even even under its maker-fee assumptions. This is a useful warning that predictive R-squared is not tradable profit.
6. **Open-source bot infrastructure.** https://github.com/freqtrade/freqtrade provides a mature crypto backtesting, dry-run and FreqAI platform. It is useful as an implementation reference, not as evidence that any community strategy is profitable.

## Candidate implemented in this branch

The module tradingagents/crypto/backtest/factor_interaction.py implements a deliberately constrained version of the cross-sectional interaction hypothesis:

- Ten liquid USDT Spot pairs from the existing two-year Binance public OHLCV cache.
- Causal 1h features: 24h/72h/7d/14d momentum, 6h reversal, 24h/7d realized volatility, EMA distance, dollar-volume liquidity proxy and relative-volume proxy.
- Cross-sectional ranks and five predeclared interactions between momentum, liquidity, volatility, reversal, volume and BTC market regime.
- Weekly-refit ridge regression. The training window is the trailing 180 days; the target is next-open to next-open 24h return.
- Purged labels: every training outcome must be fully realized before the OOS fold begins.
- Seven-day non-overlapping OOS folds, six daily decision points per fold, long-only Spot, maximum two 10% sleeves, no leverage, and cash held when the BTC regime or expected-return gate is not met.
- Realistic and conservative cost scenarios; the predicted gross return must exceed 1.25 times modeled round-trip costs.
- No parameter sweep, hyperparameter optimization, exchange integration, or order submission.

This is not a replication of the 500-asset academic study. The current ten-asset universe is much smaller, uses Binance Spot as a cross-venue proxy for a future Bybit Spot target, and does not contain order-book, funding, open-interest, liquidation or global order-flow data.

## Validation and acceptance

- CI tests and Ruff validate code correctness only.
- A successful research workflow means the model ran, not that it is profitable.
- A candidate must show positive compounded OOS returns after both realistic and conservative costs, positive net expectancy and PF above 1, adequate independent trade count, tolerable drawdown, and consistency across symbols, folds and distinct market regimes.
- Compare against cash, buy-and-hold and simple baselines on identical timestamps and costs.
- Account for model/family selection with nested purged validation, multiple-testing disclosure and a deflated Sharpe / probability-of-backtest-overfitting analysis where sample size permits.
- Preserve a genuinely future untouched holdout. The existing two-year cache has already been examined by earlier studies and is not a final locked holdout.
- Before any paper integration, measure Bybit account fees, spread, slippage, order filters and actual paper fill behavior. Binance history is not a Bybit execution record.

## Rejected or deferred families

- EMA9/VWAP 3m crossover: current 14-day BTC/ETH/BNB test lost before costs and deteriorated substantially after costs.
- TJR: the current 14-day BTC screen remains negative after modeled costs.
- Regime breakout and current adaptive expert mixture: current-head studies have not demonstrated robust positive after-cost results.
- Perpetual funding carry: deferred because the hosted runner could not obtain valid Binance perpetual klines; five focused offline unit tests passed, but no market result was produced.
- High-frequency order-flow: deferred until synchronized trade/depth data, reliable venue access, event-time replay and queue/latency-aware fills are available.
- Cointegration/OU baskets: deferred until the project can support reliable hedge/short execution and a broader, point-in-time universe.

**Safety:** Keep PR #6 open and draft. Do not merge, deploy, start/restart the Railway worker, submit orders, expose/change secrets, enable leverage/shorting, or increase risk limits as part of this research.
