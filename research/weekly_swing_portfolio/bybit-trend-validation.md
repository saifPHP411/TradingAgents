# Bybit 4-hour entry-rule comparison

Decision: keep both rules in research. Across independent periods neither sustained positive net P&L and a profit factor above 1 under conservative costs. Neither reached the 75% win-rate target on the expanded holdout sample.

## Method

- Markets: Bybit USDT perpetuals. The core list contains BTC, ETH, SOL and BNB. The expanded list includes those four and 20 more contracts from `wide-bybit-filters.json`.
- Volume breakout: close outside the previous 24-bar price channel, aligned with EMA200, with USDT turnover at least 1.5 times the preceding 24-bar median.
- Trend pullback: close crosses back over EMA20 in the direction of the EMA50/EMA200 trend.
- The rules were fixed for this comparison. Entries use the open after the signal closes. Both rules use a two-ATR stop and a maximum hold of 42 bars. A stop or liquidation can occur in the entry bar.
- Each symbol starts with an independent 50 USDT at 2x. The simulation applies historical funding, mark-price liquidation, current exchange order limits and the repository's three fee and slippage cases. Aggregated P&L sums those standalone runs. A single 50 USDT account sharing capital across 24 contracts would require a portfolio simulation. Candle-level fills and today's order limits are approximations.

## 2025 holdout with conservative costs

From 1 May through 31 July, UTC. All 24 markets had complete input data for every window.

| Rule | Markets | Wins / trades | Win rate | Net P&L sum (USDT) | Profit factor | Worst symbol drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Volume breakout | 4 | 1 / 4 | 25.0% | +0.042 | 1.16 | 1.02% |
| Volume breakout | 24 | 32 / 77 | 41.6% | +6.485 | 1.89 | 2.21% |
| Trend pullback | 4 | 0 / 3 | 0.0% | −0.313 | 0.00 | 0.63% |
| Trend pullback | 24 | 16 / 78 | 20.5% | −3.497 | 0.66 | 2.37% |

Expanded volume breakout was also positive under stress costs on the holdout (PF 2.28), but fell to PF 0.40 in May–October 2024 and PF 0.35 on 1–19 November 2026 under conservative costs. Expanded trend pullback had PF 1.10 in 2024 and PF 1.99 in the recent check under conservative costs, with PF 0.66 on the holdout. There were no simulated liquidations. The drawdowns are per-symbol figures.

## Reproduce

From the repository root with the project dependencies installed:

```sh
PYTHONPATH=. python research/weekly_swing_portfolio/bybit_trend_validation.py
```

The script writes `bybit-trend-validation.json` with the full trade ledger. [The committed summary](bybit-trend-summary.json) records the three windows, cost cases, tested markets, P&L, drawdown, and order rejections.
