# Stop risk and weekly profit concentration

This diagnostic adapts the trade-level RR_gain and time-distribution checks in Ondřej Buček, *Analyzing trading strategies for cryptocurrencies charts* (2022), sections 5.1, 5.4.2 and 5.5. The study calculates net realized P&L divided by position quantity times initial stop distance. Fees, funding, slippage and modeled liquidation from the existing replay are included in the numerator. The denominator is the planned price risk at entry. It can differ from the realized loss. R values are summed as normalized trade observations, not as a portfolio return.

The audited configuration is the previously selected Binance relative momentum strategy with the strong-signal gate on 24 contracts at 50x. The test combines its original 12 weeks with the 20 additional confirmation weeks, using the same 15-minute execution model and contract assumptions. Every weekly balance starts at 50 USDT. The highest-R week is removed once per cost case to check sensitivity to that period.

| Cost case | Weeks | Trades | Total net R | Best week net R | Net R excluding best week |
|---|---:|---:|---:|---:|---:|
| Base | 32 | 535 | -110.08 | +9.70 | -119.78 |
| Stress | 32 | 463 | -208.33 | +9.56 | -217.89 |

The week starting 2023-09-11 was the only profitable week and the highest-R week in both cost cases. All 64 ending balances matched the previously recorded replays. Run `python risk_units_audit.py` to regenerate the weekly and trade-level diagnostics.
