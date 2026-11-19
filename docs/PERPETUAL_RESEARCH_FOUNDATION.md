# Bybit USDT Perpetual Research Foundation

## Scope

This is an isolated research/backtesting subsystem on `strategy-research-cost-aware`. It does not modify the existing Spot broker, paper worker, production service, Railway settings, or PR base branch. It does not authenticate, read account balances, change leverage, submit orders, or start a worker.

The first implementation is deliberately limited to **linear USDT-settled perpetual contracts**, **isolated margin**, and **one-way positions**. Inverse contracts, cross/portfolio margin, hedge mode, multi-symbol netting, partial live fills, and order management are not implemented.

## Candidate strategy

The disabled-by-default research candidate uses completed 4-hour candles:

- EMA 14/50 crossover for direction.
- EMA 200 as the long/short regime filter.
- ADX filter (default 18) to avoid low-directional-strength conditions.
- ATR(14) stop at 2.5 ATR, trailed only after a completed candle.
- Entry at the next candle open after a completed signal; opposite crossover and max holding period exit.
- One isolated position per symbol in the single-symbol simulator.

## Leverage and risk

- Default simulated leverage is 1x; hard research ceiling is 2x.
- Position size is the minimum of stop-risk, initial-margin, notional, instrument quantity and instrument max-market-quantity limits.
- Default risk budget is 0.5% of equity per trade (hard configuration ceiling 1%).
- Default margin cap is 10% of equity and notional cap is 20%.
- Daily loss, maximum drawdown and consecutive-loss halts are enforced in the simulator.
- Venue qty-step rounding floors quantity; it never rounds up through a risk limit.

Leverage changes margin requirements; it does not create strategy edge. It may amplify losses and accelerate liquidation.

## Data and execution assumptions

The read-only Bybit V5 adapter downloads current instrument filters, 4-hour traded-price candles, aligned mark-price candles, and historical funding events. It discards still-forming bars. Missing/irregular funding data, candle gaps, timestamp mismatches, malformed rows and unsupported instruments fail closed.

The default per-fill cost assumptions are 5.5 bps commission, 1 bp half-spread and 2 bps slippage. They are adjustable estimates, not a verified account-specific fee schedule. Funding cash flow is applied at the exact returned settlement event and uses mark price.

The isolated liquidation price is an explicit approximation. It does not reproduce Bybit's risk-tier maintenance-margin deduction, closing-fee and account calculation. Any future Demo connector must use Bybit's own position/liquidation fields and instrument risk tiers instead of trusting this estimate.

## Running a read-only research screen

Install the project's normal dependencies, then run:

```bash
python -m tradingagents.crypto.backtest.perpetual_ema_study --symbol BTCUSDT --days 365 --leverage 1.0
```

This fetches public market data and writes a JSON report under `results/`. It does not place orders. The script is an exploratory screen, not evidence that the candidate is profitable or Demo-ready. Evaluate multiple symbols, chronological out-of-sample folds, realistic/conservative/high-cost scenarios, funding and liquidation stress before any Demo consideration.

## Go / No-Go

A passing unit test or completed data download proves only that code paths ran. It is not proof of profitability. The existing research history contains weak or cost-sensitive candidates; this EMA perpetual strategy has no validated return claim yet. The report marks `demo_ready=false`. Do not integrate it into the paper worker or production execution path until independent review and robust after-cost out-of-sample evidence exist.

## Safety corrections and implementation status (2026-11-17)

- Added a configurable 2% default separation buffer between the initial ATR stop and the simplified liquidation estimate. Entries are rejected when the estimated liquidation level is too close to, or beyond, the stop. This is an additional research guard, not a replacement for the exchange's live liquidation calculation.
- Fixed pending daily/drawdown exits so they are cleared when their owning position closes; a queued exit cannot leak into and close a later position. Signals are not queued while the daily-loss halt is active.
- Added regression tests for both conditions and for liquidation-buffer validation.
- The simulator remains single-symbol and research-only. The read-only market adapter still has no authenticated account, set-leverage, place-order, cancel-order, or position-management methods. It does not connect to the dashboard or paper worker.
- A real Bybit linear-perpetual execution adapter would need to use `POST /v5/position/set-leverage` for leverage and the `linear` category for USDT perpetual orders. The `isLeverage` flag belongs to Spot margin borrowing, not the perpetual leverage setting.
- Before any Demo consideration, fetch the applicable `/v5/market/risk-limit` tier, use the exchange's current position/liquidation fields for reconciliation, and validate the actual UTA account mode, isolated/cross behavior, tick/quantity/minimum-notional rules, reduce-only protection, partial fills, clock synchronization, and order idempotency. The simple liquidation estimate in this module is not a substitute for Bybit's tier-aware calculation.
- No production branch, Railway service, worker, credentials, or orders were touched. `demo_ready` remains false.
