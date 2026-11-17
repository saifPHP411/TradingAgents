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

## 4h Donchian trend candidate and preservation record (2026-11-17)

- The best candidate found in the current research sweep is the existing fixed-rule 4h Donchian trend strategy with an optional **individual-asset ADX >= 20** entry confirmation. This is a research variant, not a default or execution approval.
- The original baseline was preserved before changing it in branch [backup/turtle-trend-positive-20261117](https://github.com/saifPHP411/TradingAgents/tree/backup/turtle-trend-positive-20261117), pointing to commit `042965ea009329ee6b068b8930d87f3552be1a92`. A direct rerun of the baseline after adding disabled-by-default filters produced identical summary and fold records.
- The ADX >=20 candidate has its own recovery branch [backup/turtle-trend-adx20-candidate](https://github.com/saifPHP411/TradingAgents/tree/backup/turtle-trend-adx20-candidate), pinned to commit `43bf543ebad686017b7366d0842df89dfe3a4508`. Both recovery refs are separate from this working research branch; preserve them unchanged.
- On 10 Binance Spot USDT pairs, 4h candles resampled from complete 1h bars, 22 non-overlapping chronological folds and 248 baseline trades, the baseline returned **+0.4043% realistic** (PF 1.103) but **-0.0134% conservative** (PF 1.000). These are estimated costs, not measured Bybit fills.
- A predeclared filter comparison ran four variants on the same already-inspected two-year development history. The ADX >=20 variant returned **+0.5197% realistic** (PF 1.152; 220 trades; +$2.40 net expectancy/trade) and **+0.1526% conservative** (PF 1.044; 220 trades; +$0.73 expectancy/trade), with reported max drawdown across fold returns 0.683% realistic / 0.782% conservative. This was better than baseline on those aggregate metrics.
- Important weakness: only **4 of 22 folds were positive** for ADX >=20; 5 folds had no trades. Fold-return t-statistics were very small (about 0.38 realistic and 0.12 conservative), and a naïve IID bootstrap interval crossed zero in both scenarios. Profits were concentrated in a few folds. Six of ten symbols were positive, so symbol-level breadth was not universal.
- The relative-volume >=1.2 filter alone and combined ADX+relative-volume variant were weaker than ADX alone. Therefore relative volume was not selected as a default filter.
- The latest rolling-cache refinement rerun (10 Binance Spot pairs, 4h bars, 22 chronological folds) compared seven predeclared variants on the same already-inspected development history. The best aggregate candidate was `ADX >= 20` + rising ADX + `+DI > -DI`: **+0.9055% realistic** (PF 1.288; 210 trades) and **+0.5488% conservative** (PF 1.167; 210 trades); reported max drawdown across fold returns was 0.661% realistic / 0.760% conservative. The ADX-only result was +0.8520% realistic / +0.4896% conservative (214 trades); adding `+DI > -DI` without rising ADX did not change the result in this sample.
- This is an improvement in the historical comparison, not proof of an edge. Only **5 of 22 folds were positive** in both cost cases. The simple fold-level t-statistics were approximately 0.62 realistic and 0.39 conservative (two-sided p-values about 0.54 and 0.70); neither is statistically significant. Seven of ten symbols were positive in both scenarios, but losses remained concentrated in BTC and LINK. The result is exploratory because all seven variants reuse the already-inspected history; do not select/promote from this sample alone.
- The earlier DMI artifact reported +0.5730% realistic / +0.2115% conservative on its older rolling cache. The newer figures above supersede that snapshot for current research reporting; preserve the older artifact for reproducibility rather than overwriting it.
- Recovery branches: [baseline](https://github.com/saifPHP411/TradingAgents/tree/backup/turtle-trend-positive-20261117) at `042965ea009329ee6b068b8930d87f3552be1a92`; [ADX-only](https://github.com/saifPHP411/TradingAgents/tree/backup/turtle-trend-adx20-candidate) at `43bf543ebad686017b7366d0842df89dfe3a4508`; [rising-ADX combination](https://github.com/saifPHP411/TradingAgents/tree/backup/turtle-trend-rising-adx-20261117) at `c7020b17df2d140ea1ed4c0a2d5cc7b7b4f8aa56`.
- Latest DMI artifact: [research workflow run](https://github.com/saifPHP411/TradingAgents/actions/runs/44077380150), artifact `14325947451`; CI run `44077380193` succeeded on Python 3.11–3.14, Ruff, and clean-install. Earlier artifacts: [filter comparison run](https://github.com/saifPHP411/TradingAgents/actions/runs/44071967660), artifact `14323586288`; [baseline study in the same run](https://github.com/saifPHP411/TradingAgents/actions/runs/44071967660), artifact `14322723886`.

## Rejected or deferred families

- EMA9/VWAP 3m crossover: current 14-day BTC/ETH/BNB test lost before costs and deteriorated substantially after costs.
- TJR: the current 14-day BTC screen remains negative after modeled costs.
- Regime breakout and current adaptive expert mixture: current-head studies have not demonstrated robust positive after-cost results.
- Perpetual funding carry: the first market run failed before strategy evaluation because the runner received an empty HTML response from Binance USD-M Futures REST klines; five focused offline unit tests passed, but no market result was produced. The research branch now includes an official Binance USD-M Futures public-archive fallback for hourly klines, per-host error diagnostics, and a machine-readable failure artifact. Funding-rate history still requires the public Futures REST endpoint; an archive fallback for funding observations has not been verified.
- High-frequency order-flow: deferred until synchronized trade/depth data, reliable venue access, event-time replay and queue/latency-aware fills are available.
- Cointegration/OU baskets: deferred until the project can support reliable hedge/short execution and a broader, point-in-time universe.

**Safety:** Keep PR #6 open and draft. Do not merge, deploy, start/restart the Railway worker, submit orders, expose/change secrets, enable leverage/shorting, or increase risk limits as part of this research.
