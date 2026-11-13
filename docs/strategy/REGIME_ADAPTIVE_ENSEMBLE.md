# Regime-Adaptive Crypto Strategy Ensemble — Research Specification

Status: research-only proposal. This document does not change the active Bybit Demo strategy or enable orders.

## Research conclusion

Do not replace one named retail setup with a pile of indicators or a simple majority vote. Build a regime-aware portfolio of independent, testable signal families, and let each family trade only in the market state where its premise is valid.

The initial candidate is a **Regime-Adaptive Ensemble (RAE)** for liquid Binance spot pairs:

1. **Trend following / time-series momentum (primary):** multi-horizon return and moving-average trend alignment, with Donchian breakout confirmation. This is the default module in persistent directional markets.
2. **Range mean reversion (conditional):** deviation from a rolling mean / volatility band with RSI confirmation, enabled only when trend strength is low and the market is demonstrably range-bound. It is disabled during strong trends to avoid repeatedly buying falling markets or shorting rising ones.
3. **Trend pullback continuation (additional candidate):** EMA pullback reclaim while the higher-level trend, directional movement and time-series momentum remain aligned. It is evaluated separately from Donchian breakouts to avoid hiding which entry family is responsible for results.
4. **Volatility breakout (secondary confirmation):** ATR-normalized range expansion and volume confirmation. It confirms a new move; it is not a standalone reason to chase an already extended candle.
5. **Cross-asset risk filter:** BTC trend/volatility state can reduce or block altcoin exposure when broad market conditions are hostile. This is a risk overlay, not a predictive guarantee.
6. **Execution and cost filter:** reject stale prices, excessive spread, insufficient liquidity, and signals whose conservative expected move does not clear estimated round-trip fees plus slippage.

The modules are combined by **rule-based regime gating and explicit module priority**, not by unconditional indicator stacking. The numeric signal score is diagnostic only; it is not a calibrated probability. In a trend regime, Donchian breakout and pullback continuation may contribute; in a range regime, mean reversion may contribute; in an ambiguous/high-volatility regime, the system stays flat.

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
3. Compare baseline TJR, trend-only, trend-pullback-only, range-only, breakout-only, and the regime-adaptive ensemble on identical symbols, bars, sizing, and costs.
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

- `tradingagents/crypto/ensemble.py`: completed-bar regime classification, EMA/DI/ADX trend filter, Donchian breakout, distinct EMA pullback-reclaim entry, momentum/RSI/volume confirmation, range z-score + RSI reversal, ATR initial stop, explicit Donchian/mean exits, max-hold/time stop and post-stop cooldown, fail-closed BTC trend overlay for altcoins, and distinct `should_exit` semantics so "no fresh entry" does not force an open trend trade to close.
- `tradingagents/crypto/ensemble_backtest.py`: one-symbol long-only simulation, signal at completed-bar close, next-bar-open execution, gap-aware intrabar ATR stop, stop-distance sizing, notional cap, exact inclusive max-hold exit (the entry candle is holding bar one), post-stop cooldown, per-side fee and slippage accounting, and trade/equity logs.
- `tradingagents/crypto/ensemble_portfolio.py`: shared-equity BTC/ETH/BNB spot portfolio simulation with aligned-hourly-data checks, deterministic BTC-first simultaneous-entry priority, exact inclusive max-hold exit, max positions, altcoin concentration, gross-exposure, daily-loss and peak-drawdown entry gates, BTC risk overlay, correlation and aggregate stop-risk caps, and shared cash/fees/slippage accounting.
- `tradingagents/crypto/fetch_ensemble_history.py`: explicit fetch of completed Binance Spot 1h candles into the offline research cache.
- `tradingagents/crypto/run_ensemble_study.py`: chronological 60/20/20 development/validation/holdout comparison for trend-only, trend-pullback-only, range-only, breakout-only, and ensemble variants, with a 1.5x fee/slippage stress, plus a separate shared-equity portfolio result for each partition and the recent 30-day sample.
- `.github/workflows/crypto-ensemble-study.yml`: manually dispatched two-year (17,520 hourly bars) public-data study; the JSON report is uploaded as an artifact.

This is **not yet the complete proposed system**. The research branch now contains both a single-symbol simulator and a shared-equity multi-symbol portfolio simulator. The portfolio applies max-open-position, altcoin-concentration, gross-exposure, daily-loss and peak-drawdown entry gates, plus a fail-closed BTC trend overlay for altcoins. The current RAE v1.1 portfolio study is recorded in `docs/strategy/RAE_V1_1_PORTFOLIO_STUDY_RESULTS.md`; it failed the acceptance gate: development and holdout were negative, the validation gain came from only nine trades, and the recent 30-day period produced no trades. It must not be activated.

Still outstanding before any strategy selection are: completed 4h trend context; stronger multi-horizon momentum validation; spread/depth/liquidity and expected-move cost filters; exchange quantity-step/minimum-notional handling; verified Bybit account-tier fees and measured execution costs; and a genuinely matched TJR-vs-RAE comparison on identical data and execution assumptions. The trailing stop, correlation cap, aggregate stop-risk cap, inclusive max-hold timing, and deterministic BTC-first entry order are implemented and covered by regression tests, but their presence does not establish a profitable edge. The current simulators remain research prototypes, not production execution engines.

The Binance fetch workflow uses public market data and has no API credentials. The 10 bps per-side fee and 5 bps per-side slippage defaults are research assumptions, **not a claim about the user's Bybit fee tier or actual fills**. The 1.5x run stress-tests those assumptions. Replace them with verified account-tier fees and measured execution costs before interpreting results.

Commands for a local research environment:

```bash
python -m tradingagents.crypto.fetch_ensemble_history --bars 17520 --symbols BTC/USDT ETH/USDT BNB/USDT --cache .research-cache
python -m tradingagents.crypto.run_ensemble_study --symbols BTC/USDT ETH/USDT BNB/USDT --cache .research-cache --out rae_study.json
```

The workflow can also be started from GitHub Actions → **Crypto RAE historical study** → Run workflow. It is manual by design; no recurring job, paper worker, Railway deployment, or broker order is started by it.


## RAE v1.2 engineering changes (research-only)

The v1.1 shared-portfolio result is rejected and must not be tuned against its already inspected validation or holdout windows. A new research iteration adds two risk controls without changing the entry thresholds:

- **Chandelier-style ATR trailing stop:** the stop can only tighten using the completed bar's high and ATR after that bar's intrabar stop check. The updated stop becomes active on the following bar. This avoids same-bar look-ahead. The initial ATR stop remains in force until the trailing level is higher.
- **Aggregate open-risk cap:** total modeled stop risk across all open positions cannot exceed 1% of current equity by default. A new position's risk budget is reduced by the stop risk already reserved by open positions; when no capacity remains, the entry is blocked.
- **Altcoin correlation cap:** before a new altcoin order is filled at a bar open, its rolling return correlation with existing altcoin positions is measured using completed returns only through the prior bar. Correlation at or above the configured cap blocks the entry. Insufficient history or undefined correlation fails closed. BTC is excluded from this pairwise altcoin cap because the separate BTC regime overlay already governs broad-market risk.

Defaults are a 3.0 ATR trailing distance, 1.0% maximum aggregate open stop risk, 0.85 maximum altcoin correlation, and 168 completed hourly returns. These are initial risk-control hypotheses, not optimized parameters. Unit tests cover next-bar trailing-stop activation, correlation blocking, and invalid settings.

The old v1.1 validation/holdout have already been inspected and are no longer untouched data. Any performance report produced after these changes is exploratory only; do not use it as a fresh out-of-sample claim. Freeze the v1.2 implementation and collect a new future holdout before evaluating activation. Do not connect it to Bybit Demo or Live.

A later correctness audit fixed a one-bar-late `max_hold_bars` exit in both simulators and made BTC the first deterministic symbol considered when several entry signals arrive together. The corrected rerun is documented in `docs/strategy/RAE_V1_2_PORTFOLIO_STUDY_RESULTS.md` and remains rejected: development and the previously inspected holdout are negative, the validation result is based on only nine trades, and the most recent 30 days produced zero trades. No trade in the corrected study exited due to `max_hold_bars`, so this timing correction did not change those aggregate portfolio metrics.
