# Frozen Bollinger + RSI validation on Bybit

The source is `perpetual_strategy_lab_latest.json`, SHA-256
`86425cae59139a08bd8d49af3a0432a3430b7170c207b7cc1ce738a49496aaf5`.
The accompanying ZIP contains the identical JSON. Its
`selective_bollinger_rsi_range` report records 8/9 wins for realistic
costs at 2x and 7/9 for conservative costs on Binance Vision. It does
not reproduce the earlier user-reported 9/12 result. All 14 parameters
in the new replay match the archived candidate exactly. The archive's
development period started 2024-11-01 and ended before 2026-11-01 UTC.

The new runs use Bybit USDT perpetual trade and mark candles, historical
funding observations, current Bybit quantity and minimum notional rules,
and current first-tier maintenance margin. The shared project simulator
runs the rule at 2x with a 50 USDT starting balance per symbol, a 20%
position notional cap, and the existing modeled transaction costs.
Every symbol is a
separate replay and the summed P&L is a diagnostic across those runs. Historical order
book liquidity and contract-rule changes are unobserved.

| Window (UTC, end exclusive) | Universe | Signals | Trades, conservative | Win rate | Profit factor | Sum of net P&L, USDT |
|---|---:|---:|---:|---:|---:|---:|
| 2024-05-01 to 2024-11-01 | 4 | 0 | 0 | — | — | 0 |
| 2024-05-01 to 2024-11-01 | 24 | 24 | 11 | 27.27% | 0.4275 | -0.7263 |
| 2026-11-01 to 2026-11-20 | 4 or 24 | 0 | 0 | — | — | 0 |
| 2026-11-13 to 2026-11-20 | 4 or 24 | 0 | 0 | — | — | 0 |

The seven-day test starts flat at 50 USDT and is a subset of the November
window. All 24 symbols passed data checks. Seven symbols produced modeled
trades in the historical comparison. Thirteen entry attempts in the
conservative case failed quantity or notional limits. The worst
conservative symbol-level drawdown was 0.84% and there were no modeled
liquidations. The original four symbols were BTCUSDT, ETHUSDT, SOLUSDT,
and BNBUSDT.

Decision: the rule fails the return and win-rate criteria. This research
did not progress to a shared-capital execution trial on Railway or an
exchange order. The compressed JSON includes input filters, all cost
cases, and per-symbol closed trade ledgers. Run
`python bollinger_bybit_validation.py` to regenerate it. All balances
were reconciled, and 13 focused project tests passed.
