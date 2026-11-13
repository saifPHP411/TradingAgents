# Regime-Adaptive Crypto Strategy Ensemble — Research Specification

Status: research-only proposal. This document does not change the active Bybit Demo strategy or enable orders.

## Research conclusion

Do not replace one named retail setup with a pile of indicators or a simple majority vote. Build a regime-aware portfolio of independent, testable signal families, and let each family trade only in the market state where its premise is valid.

The initial candidate is a **Regime-Adaptive Ensemble (RAE)** for liquid Binance spot pairs:

1. **Trend following / time-series momentum (primary):** multi-horizon return and moving-average trend alignment, with Donchian breakout confirmation. This is the default module in persistent directional markets.
2. **Range mean reversion (conditional):** deviation from a rolling mean / volatility band with RSI confirmation, enabled only when trend strength is low and the market is demonstrably range-bound. It is disabled during strong trends to avoid repeatedly buying falling markets or shorting rising ones.
3. **Volatility breakout (secondary confirmation):** ATR-normalized range expansion and volume confirmation. It confirms a new move; it is not a standalone reason to chase an already extended candle.
4. **Cross-asset risk filter:** BTC trend/volatility state can reduce or block altcoin exposure when broad market conditions are hostile. This is a risk overlay, not a predictive guarantee.
5. **Execution and cost filter:** reject stale prices, excessive spread, insufficient liquidity, and signals whose conservative expected move does not clear estimated round-trip fees plus slippage.

The modules are combined by **regime gating and calibrated scores**, not by unconditional indicator stacking. In a trend regime, trend and breakout modules may contribute; in a range regime, mean reversion may contribute; in an ambiguous/high-volatility regime, the system stays flat.

## Initial rule families (hypotheses, not optimized settings)

### A. Trend regime and entries
- Determine trend on completed higher-timeframe candles only (initial candidate: 4h; execution/entry on 1h).
- Use EMA 50/200 direction and slope plus ADX as regime descriptors.
- Require multi-horizon time-series momentum agreement (for example 20/60-bar returns) and a Donchian breakout or controlled pullback.
- Avoid entries when price is too extended relative to ATR.
- Exit with ATR-based initial protection and a Chandelier/ATR trailing stop; evaluate optional time stop.

### B. Range regime and entries
- Require low trend strength, flat long EMA slope, and bounded rolling range.
- Enter only at statistically stretched deviations from a rolling mean / volatility band with RSI confirmation and a reversal close.
- Exit near the mean or at a predeclared volatility-based stop. Do not average down.
- Disable this module when the trend filter flips or volatility expands sharply.

### C. Position sizing and portfolio risk
- Size from stop distance and a fixed fraction of current equity at risk; never size from a fixed notional alone.
- Apply hard per-trade, aggregate exposure, correlation/concentration, daily-loss, drawdown, and maximum-position caps.
- Keep the current spot-only / long-only execution constraint until a separate, explicitly tested derivatives implementation exists.
- Do not loosen existing broker-side limits. The broker's verified protective orders remain a separate mandatory gate.

## Research and acceptance protocol

1. Preserve the TJR implementation as a reproducible baseline. Do not delete it or silently change the deployed source.
2. Implement each candidate module independently and unit-test it on synthetic OHLCV fixtures, including look-ahead and candle-completion boundaries.
3. Compare baseline TJR, trend-only, range-only, breakout-only, and the regime-adaptive ensemble on identical symbols, bars, sizing, and costs.
4. Use at least 2 years of hourly history where available, plus a separate higher-frequency execution-cost sample. The current 30-day 1-minute study is not sufficient to establish robustness.
5. Use chronological development / validation / untouched holdout splits and rolling walk-forward folds. Freeze parameters before the final holdout.
6. Include exchange maker/taker fees appropriate to the actual account tier, spread, conservative slippage, rejected/partial fills, and turnover. Report gross and net results separately.
7. Report net expectancy, profit factor, Sharpe/Sortino, max drawdown, exposure, turnover, trade count, fees, slippage, and per-symbol/per-regime results. Include confidence intervals or bootstrap uncertainty where practical.
8. Reject any model whose profitability depends on one symbol, one short date window, one parameter point, or frictionless fills. Prefer a stable parameter plateau to the single best optimization result.
9. Only after independent out-of-sample acceptance, wire the selected strategy into the paper decision source behind an explicit feature flag. Keep live trading disabled; first run shadow decisions, then demo-only paper validation.

## Initial fail-closed acceptance gates

These are conservative engineering gates to be reviewed against the account's actual constraints, not promises of profit:
- Positive net expectancy after conservative costs on untouched holdout.
- Profit factor > 1.15 on holdout, with enough trades to make the estimate meaningful.
- No single symbol or fold accounts for most of the total profit.
- Maximum drawdown stays within the configured portfolio risk budget.
- Results remain directionally positive under a 1.5x cost stress.
- No look-ahead, survivorship, data-gap, or fill-timing defects.
- All open positions have broker-verified protective stop orders; recorded dashboard levels alone do not count.

If the gates fail, do not deploy the ensemble. Preserve the negative result and investigate rather than retuning against the holdout.

## Research references

- Chen et al. (2026), *Technical analysis, fundamental analysis, and momentum strategies: Review and integration*: https://doi.org/10.1007/s11156-026-01540-7
- *Cryptocurrency trading: a comprehensive survey* (2021), systematic crypto trading, technical rules, pairs trading, and econometric methods: https://doi.org/10.1186/s40854-021-00321-6
- Zarattini, Pagani & Barbon (2025), *Catching Crypto Trends*: multi-Donchian trend ensemble, volatility sizing, and explicit transaction-cost treatment: https://concretumgroup.com/wp-content/uploads/2026/02/Catching-Crypto-Trends.pdf
- Monash (2023), *Trend-following Strategies for Crypto Investors*: trend following evidence and material transaction-cost sensitivity: https://www.monash.edu/__data/assets/pdf_file/0011/3744821/Trend-following-Strategies-for-Crypto-Investors.pdf
- Grobys (2025), *Cryptocurrency Momentum: Is It an Illusion?*: risk and distributional caveats for crypto momentum: https://doi.org/10.1002/ijfe.70036
- Karassavidis (2025/2026), *Quantitative Evaluation of Volatility-Adaptive Trend-Following Models in Cryptocurrency Markets*: multi-timeframe trend, RSI, ATR risk and trailing exits; working paper, not proof of deployable edge: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5821842

## Important limitations

Published backtests are hypotheses, not a menu of guaranteed-profitable strategies. Results depend on sample, horizon, universe, fees, spread, execution, and market regime. The implementation must be tested against the project's own Bybit Demo execution path before it can replace TJR.


## Implementation status (research branch)

The current branch contains a deliberately isolated prototype:

- `tradingagents/crypto/ensemble.py`: completed-bar regime classification, EMA/DI/ADX trend filter, Donchian breakout, momentum/RSI/volume confirmation, range z-score + RSI reversal, ATR initial stop, explicit Donchian/mean exits, max-hold/time stop and post-stop cooldown, fail-closed BTC trend overlay for altcoins, and distinct `should_exit` semantics so "no fresh entry" does not force an open trend trade to close.
- `tradingagents/crypto/ensemble_backtest.py`: one-symbol long-only simulation, signal at completed-bar close, next-bar-open execution, gap-aware intrabar ATR stop, stop-distance sizing, notional cap, per-side fee and slippage accounting, and trade/equity logs.
- `tradingagents/crypto/fetch_ensemble_history.py`: explicit fetch of completed Binance Spot 1h candles into the offline research cache.
- `tradingagents/crypto/run_ensemble_study.py`: chronological 60/20/20 development/validation/holdout comparison for trend-only, range-only, breakout-only, and ensemble variants, with a 1.5x fee/slippage stress. Each symbol and partition is an independent account; its returns must not be summed as if they were one portfolio.
- `.github/workflows/crypto-ensemble-study.yml`: manually dispatched two-year (17,520 hourly bars) public-data study; the JSON report is uploaded as an artifact.

This is **not yet the complete proposed system**. The following remain before any strategy selection: 4h trend context; spread/depth/liquidity and expected-move cost filter; trailing/Chandelier stop; multi-symbol shared-equity portfolio with correlation/exposure/daily-loss caps; comparison against the unchanged TJR baseline on exactly matched data and costs; and a higher-frequency execution-cost sample. The current single-symbol backtest is a controlled prototype, not a production portfolio engine.

The Binance fetch workflow uses public market data and has no API credentials. The 10 bps per-side fee and 5 bps per-side slippage defaults are research assumptions, **not a claim about the user's Bybit fee tier or actual fills**. The 1.5x run stress-tests those assumptions. Replace them with verified account-tier fees and measured execution costs before interpreting results.

Commands for a local research environment:

```bash
python -m tradingagents.crypto.fetch_ensemble_history --bars 17520 --symbols BTC/USDT ETH/USDT BNB/USDT --cache .research-cache
python -m tradingagents.crypto.run_ensemble_study --symbols BTC/USDT ETH/USDT BNB/USDT --cache .research-cache --out rae_study.json
```

The workflow can also be started from GitHub Actions → **Crypto RAE historical study** → Run workflow. It is manual by design; no recurring job, paper worker, Railway deployment, or broker order is started by it.
