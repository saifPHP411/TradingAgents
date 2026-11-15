# RAE Walk-Forward Validation Protocol

**Status: research-only. No broker, Bybit Demo, live execution, worker, or Railway integration.**

## Why this protocol exists

The published RAE v1.2 study is not sufficient for activation: development and the previously inspected holdout were negative, validation contained only nine trades, the prior holdout contains only six trades, and the recent 30-day window had no trades. Those validation/holdout dates have already been inspected and must not be reused as independent evidence.

This protocol freezes the current RAE rules and evaluates them on chronological, disjoint out-of-sample windows over refreshed market data. However, the default two-year history overlaps the dates already inspected in the RAE v1.1/v1.2 studies. These results are therefore exploratory diagnostics, **not an independent future holdout**. A genuinely untouched holdout can only be collected after the code and settings are frozen and new future candles arrive. This runner does not tune parameters, select the best fold, or claim that bootstrap simulations predict future returns.

## Frozen evaluation defaults

| Setting | Default |
|---|---:|
| Symbols | BTC/USDT, ETH/USDT, BNB/USDT |
| Data | Binance Spot completed 1-hour OHLCV |
| Requested history | 17,520 bars per symbol (about two years) |
| Training context | 4,000 bars |
| Purge gap | 24 bars |
| Embargo gap | 24 bars |
| OOS test window | 720 bars |
| OOS step | 720 bars (non-overlapping tests) |
| Monte Carlo | 2,000 circular moving-block bootstrap paths |
| Bootstrap block | 24 hourly returns |
| Seed | 42 |
| Costs | 10 bps fee + 5 bps slippage per side, as research assumptions |
| Stress costs | Must also be evaluated at 1.5x before acceptance |

The training context is used only for causal indicator warm-up. There is no parameter fitting in this runner. Purge and embargo bars are no-trade gaps. Every fold begins with fresh capital; the aggregate fold return compounds the fold-level returns in chronological order. These are fixed-rule forward tests, not purged cross-validation for a fitted machine-learning model.

## Execution and accounting requirements

- Signals are evaluated only on completed hourly candles.
- Entries and scheduled exits execute at the following candle's open.
- Stop orders are checked intrabar; a gap through a stop fills at the worse opening price.
- Fees are charged on both sides and adverse slippage is accounted for on entry and exit.
- Position sizing includes both-side fee estimates and adverse stop-exit slippage.
- Quantity is capped by risk budget, notional, gross exposure, and available cash including entry fees.
- The shared portfolio enforces BTC-first entry priority, maximum positions, altcoin concentration, correlation, aggregate stop risk, daily-loss and drawdown entry gates.
- A missing, misaligned, non-finite, duplicated, or non-hourly data series fails closed.

The modeled stop loss cannot guarantee a maximum realized loss: gaps, liquidity, market impact, exchange outages, and rejected/partial fills can produce worse execution.

## Reported metrics

Each fold reports its exact train-context and OOS date ranges, bar count, trades, net return, max drawdown, profit factor, and blocked-entry reasons. The runner evaluates all five RAE modes (trend, trend-pullback, range, breakout, and ensemble) under both base costs and a 1.5x fee/slippage stress. Each mode uses the same symbols, folds, portfolio limits, and starting capital. The combined report includes:

- Chronologically compounded OOS return (not a sum of fold percentages).
- Trade count, win rate, net expectancy in USDT and R multiples, and profit factor.
- Hourly Sharpe and Sortino, annualized with 24 × 365 periods.
- Worst fold drawdown.
- Gross-winning-PnL concentration by symbol, fold, and individual trade; the report flags whether one symbol or fold contributes more than 50% of gross winning PnL, and marks concentration unassessable when there are no gross winners.
- Circular moving-block bootstrap distribution of terminal equity and maximum drawdown, including probability of loss and 5th/50th/95th percentiles.

Bootstrap paths preserve short contiguous clusters of hourly returns. They still assume that the observed return blocks are representative; they do not capture new regimes, exchange failure, market impact, or parameter uncertainty. A small trade count remains a small trade count regardless of the number of bootstrap simulations.

## TJR comparison: current limitation

The existing crypto TJR study uses a shared-book intraday engine with 1-minute entry triggers and 5-minute/hourly/four-hour context. RAE currently uses completed 1-hour bars and next-hour-open fills. Running those two existing engines on their native timeframes is not an apples-to-apples comparison, even if the calendar dates overlap.

Do not label the existing TJR 1-minute 30-day study as a matched benchmark for RAE. The runner also reports an equal-weight buy-and-hold benchmark on the same hourly bars, with the same modeled entry/exit fees and slippage. This is a useful market baseline, not a TJR comparison. A fair TJR comparison requires either:
1. a separately reviewed hourly TJR implementation with identical hourly signals/fill timing and the same portfolio, fees, slippage, and universe; or
2. an RAE execution adapter using the same 1-minute data, fill engine, and cost model as TJR.

Until that work is completed, report RAE against same-data buy-and-hold and risk-free/cash baselines only, and keep TJR results side-by-side with an explicit timeframe/methodology caveat.

## Acceptance gates (must all pass before any activation discussion)

1. New, untouched future holdout after the code and settings are frozen.
2. At least 100 completed OOS trades overall, with meaningful representation across symbols and folds; more data may be required.
3. Positive net expectancy and profit factor above 1.15 after conservative costs.
4. Results remain directionally positive at 1.5x costs.
5. No single symbol or fold explains most of the total profit.
6. Drawdown remains within the predeclared portfolio risk budget.
7. Monte Carlo lower-tail outcomes are acceptable under a predeclared loss tolerance.
8. No look-ahead, timing, accounting, data-gap, or fill-model defects.
9. A matched TJR comparison is completed before claiming superiority to TJR.
10. A separate shadow-only integration and Bybit Demo verification is completed after independent offline acceptance.

Passing unit tests or CI does not satisfy these performance gates. A favorable backtest does not establish future profitability.

## Reproduction

Run the GitHub Actions workflow **Crypto RAE walk-forward validation** manually. It fetches current public Binance Spot data, writes `artifacts/rae_walkforward.json`, and uploads the report as an artifact. It does not use credentials or any trading endpoint.

The workflow is manual by design. Its historical walk-forward output is exploratory because it overlaps previously inspected dates; it does not satisfy the future-holdout acceptance gate. It does not change the active TJR strategy, the Bybit Demo branch, Railway, broker settings, or API credentials.
