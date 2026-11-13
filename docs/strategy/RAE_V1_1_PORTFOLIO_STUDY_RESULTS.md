# RAE v1.1 — Shared-Portfolio Historical Study

**Decision: reject for activation. Research-only. Do not connect to Bybit Demo or Live.**

## Dataset and assumptions

- Binance Spot completed 1-hour OHLCV candles; 17,519 bars per symbol.
- Symbols: BTC/USDT, ETH/USDT, BNB/USDT.
- Coverage: 2024-11-13 09:00 UTC through 2026-11-13 07:00 UTC.
- Chronological partitions, with each partition run from fresh capital:
  - Development: 10,511 bars, 2024-11-13 to 2026-01-25.
  - Validation: 3,504 bars, 2026-01-25 to 2026-06-20.
  - Holdout: 3,504 bars, 2026-06-20 to 2026-11-13.
  - Recent activity check: last 720 bars (30 days).
- Signals use completed bars and are filled at the next bar open. Stops use gap-aware fills. Spot long-only.
- Single-symbol diagnostic runs start with 10,000 USDT per symbol. The shared portfolio starts with 10,000 USDT total across all three symbols.
- Base costs: 10 bps fee and 5 bps slippage per side. Stress: 1.5x both. These are research assumptions, not verified Bybit account fees or measured fills.
- Shared portfolio controls: max 2 open positions, max 1 altcoin, 25% gross exposure, 2% daily-loss entry gate, and 10% peak-drawdown entry gate. BTC trend state blocks new altcoin risk.

## Shared-equity portfolio — primary result

| Partition | Trades | Win rate | Profit factor | Net return | Max drawdown | Net return at 1.5x costs |
|---|---:|---:|---:|---:|---:|---:|
| Development | 21 | 23.81% | 0.461 | -0.846% | 1.179% | -1.157% |
| Validation | 9 | 33.33% | 4.338 | +0.970% | 0.462% | +0.833% |
| Holdout | 6 | 16.67% | 0.388 | -0.259% | 0.435% | -0.349% |
| Recent 30 days | 0 | — | — | 0.000% | 0.000% | 0.000% |

The validation profit factor is based on only 9 trades and is not reliable evidence of an edge. The untouched holdout lost money at both base and stressed costs. The latest 30-day period generated no trades. Development was also negative. The portfolio did not meet the positive-net-expectancy, trade-sample, or robustness requirements.

## Separate entry-family diagnostics

The following are independent single-symbol accounts, not one combined portfolio. Returns are averaged over three separate 10,000-USDT accounts only to make the aggregate scale explicit.

### Development partition

| Variant | Trades | Aggregate net return across 30,000 USDT | Profit factor |
|---|---:|---:|---:|
| Trend / Donchian | 13 | -0.238% | 0.410 |
| Trend pullback | 11 | -0.135% | 0.440 |
| Range mean reversion | 2 | -0.023% | 0.071 |
| Volatility breakout | 8 | -0.224% | 0.047 |
| Regime-adaptive ensemble | 21 | -0.265% | 0.493 |

### Validation and holdout caveat

- Trend pullback was positive in validation (7 trades; aggregate +0.376%) but negative in holdout (5 trades; aggregate -0.071%).
- The ensemble portfolio was positive in validation but negative in development and holdout.
- No individual mode has enough trades to support a robust performance claim.
- Do not tune thresholds against these already inspected validation or holdout partitions.

## Risk-gate activity

In development the shared portfolio recorded 27 altcoin entry blocks from the BTC risk filter, one max-alt-position block and one max-open-position block. No daily-loss, gross-exposure, or max-drawdown gate was reached. In holdout it recorded 10 BTC risk-filter blocks. This is descriptive, not proof that the limits will control every future market path.

## Decision and next work

1. Keep the active TJR decision source unchanged. RAE v1.1 is not approved for Bybit Demo or Live.
2. Do not deploy this branch to Railway, do not enable the paper worker for this strategy, and do not send orders.
3. Preserve these results as a rejection, not a reason to retune the already inspected validation/holdout dates.
4. Any new candidate must be developed only on development data and tested on a newly reserved future holdout. It must produce a materially larger sample before any edge claim.
5. Before any future activation discussion, add and validate 4h trend context, trailing/Chandelier exits, a correlation cap, spread/depth/liquidity and expected-move cost filters, Bybit instrument min-quantity/step/notional rules, and measured account-tier fees/slippage. Then run a shadow-only integration and demo validation only after independent offline gates pass.

The previous TJR 30-day reference is documented separately in `RAE_INITIAL_STUDY_RESULTS.md`. It used 1-minute data and is not an apples-to-apples comparison with this 1-hour RAE study; its realistic-cost report showed a negative net equity change. The RAE v1.1 study artifact is attached to GitHub Actions run 43395328090.


## Subsequent engineering iteration (RAE v1.2)

After rejecting v1.1, a separate risk-control iteration added a 3.0 ATR Chandelier-style trailing stop and a fail-closed altcoin correlation cap (0.85 over 168 completed hourly returns). The trailing level is calculated after each candle's intrabar stop check and is active only on the next candle. Correlation is evaluated at the next-bar entry open using returns through the preceding completed bar; missing/undefined history blocks the entry. These controls were added without retuning the entry thresholds.

The v1.1 results above do not measure the performance of these v1.2 controls. Because the historical validation and holdout periods have already been inspected, rerunning them is exploratory and cannot restore holdout independence. A new future holdout is required after the v1.2 code is frozen. The strategy remains rejected for activation and research-only.
