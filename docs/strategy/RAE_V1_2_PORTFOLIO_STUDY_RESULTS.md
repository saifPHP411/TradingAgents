# RAE v1.2 — Shared-Portfolio Historical Study

**Decision: reject for activation. Research-only. Do not connect to Bybit Demo or Live.**

## Run and data provenance

- Corrected GitHub Actions run: [43399778930](https://github.com/saifPHP411/TradingAgents/actions/runs/43399778930) (completed successfully).
- Code revision studied: `714eef2f0c0c67c7897209002b114f71e07e0f96`. This revision includes the corrected inclusive `max_hold_bars` semantics and deterministic BTC-first simultaneous-entry priority. Later commits only update tests and documentation.
- Binance Spot completed 1-hour OHLCV; 17,519 bars per symbol.
- Symbols: BTC/USDT, ETH/USDT, BNB/USDT.
- Coverage: 2024-11-13 10:00 UTC through 2026-11-13 08:00 UTC.
- Chronological partitions, each with fresh 10,000 USDT capital:
  - Development: 10,511 bars, 2024-11-13 10:00 to 2026-01-25 08:00 UTC.
  - Validation: 3,504 bars, 2026-01-25 09:00 to 2026-06-20 08:00 UTC.
  - Previously inspected holdout: 3,504 bars, 2026-06-20 09:00 to 2026-11-13 08:00 UTC.
  - Recent activity check: 720 bars, 2026-10-14 09:00 to 2026-11-13 08:00 UTC.
- Signals use completed candles and execute at the next bar open. Stops use gap-aware fills.
- Base costs: 10 bps fee and 5 bps slippage per side. Stress: 1.5x both. These are research assumptions, not verified Bybit account-tier fees or measured execution costs.

## v1.2 risk controls

- Initial stop: 2.5 ATR.
- Chandelier-style trailing stop: 3.0 ATR from the highest high since entry. It is calculated only after the current candle's intrabar stop check and becomes active on the next candle.
- Maximum aggregate open stop risk: 1.0% of current equity.
- Maximum open positions: 2; maximum altcoin positions: 1.
- Maximum gross exposure: 25%; per-trade risk budget: 0.5%; per-position notional cap: 10%.
- BTC trend risk filter; daily-loss entry gate 2%; peak-drawdown entry gate 10%.
- Altcoin correlation cap: 0.85 over 168 completed hourly returns. Missing or undefined correlation fails closed.

## Correctness audit after v1.2

A post-study code audit found and fixed two backtest-engine issues before this corrected rerun:

- **Maximum holding period was one completed candle too long.** Both the single-symbol and shared-portfolio simulators now count the entry candle as holding bar one and schedule the exit at the next bar open once exactly `max_hold_bars` completed candles have elapsed.
- **Simultaneous entries did not actually prioritize BTC.** Portfolio processing now gives the configured BTC risk-filter symbol the first deterministic entry slot, then processes remaining symbols alphabetically.

Regression tests cover exact max-hold timing in both simulators and BTC-first entry priority. The corrected study's trade logs contain no `max_hold_bars` exits in these partitions, so the holding-period correction did not change the primary portfolio outcome figures below. The BTC-first ordering is now explicit and reproducible; it is not a performance optimization.

## Primary shared-portfolio results

| Partition | Trades | Win rate | Profit factor | Net return | Max drawdown | Net return at 1.5x costs |
|---|---:|---:|---:|---:|---:|---:|
| Development | 21 | 23.81% | 0.460 | -0.790% | 1.073% | -1.102% |
| Validation | 9 | 33.33% | 4.378 | +0.966% | 0.395% | +0.829% |
| Previously inspected holdout | 6 | 16.67% | 0.405 | -0.242% | 0.417% | -0.331% |
| Recent 30 days | 0 | — | — | 0.000% | 0.000% | 0.000% |

### Cost and activity details

| Partition | Fees paid | Slippage paid | BTC-filter blocks | Correlation blocks | Aggregate-risk blocks |
|---|---:|---:|---:|---:|---:|
| Development | 41.71 USDT | 20.86 USDT | 27 | 0 | 0 |
| Validation | 18.29 USDT | 9.14 USDT | 8 | 0 | 0 |
| Previously inspected holdout | 11.99 USDT | 5.99 USDT | 10 | 0 | 0 |
| Recent 30 days | 0.00 USDT | 0.00 USDT | 0 | 0 | 0 |

The trailing stop tightened above its initial level in 12/21 development trades, 7/9 validation trades, and 6/6 previously inspected holdout trades. This confirms the trailing-stop mechanism was exercised. The correlation and aggregate-risk gates were not triggered in this sample; the default maximum-altcoin-position limit of one also means the correlation cap had little opportunity to engage.

## Comparison with v1.1

The shared-portfolio v1.2 result is only marginally less negative than v1.1:
- Development improved from -0.846% to -0.790%.
- Validation moved from +0.970% to +0.966%.
- Previously inspected holdout improved from -0.259% to -0.242%.
- The recent 30-day window still generated zero trades.

This is not evidence of a durable edge. The sample is tiny (only 6 holdout trades), development and holdout remain negative, and the strategy is inactive in the most recent month. The new risk controls improved stop management but did not solve the entry system's weak or sparse signal quality.

## Decision

1. **Reject RAE v1.2 for activation.** Do not merge into the active trading branch, deploy to Railway, connect it to the paper worker, or send orders.
2. Keep the active TJR source unchanged as a reproducible baseline.
3. The validation and holdout windows were already inspected during v1.1 and v1.2 development. They are not untouched data and must not be described as independent proof.
4. Do not retune thresholds against these dates. Any next candidate must be developed using development data only, frozen, and evaluated on a newly collected future holdout with a materially larger trade sample.
5. Before activation is reconsidered, collect measured spread/depth and account-tier fees, validate exchange quantity/minimum-notional rules, and test shadow-only broker integration after independent offline acceptance.

The artifact for the corrected rerun is the `crypto-rae-and-tjr-study` artifact on GitHub Actions run 43399778930. The optional TJR rerun was not enabled, so this artifact is an RAE study, not a matched TJR-vs-RAE comparison. It supersedes the earlier v1.2 report run (43397216756).