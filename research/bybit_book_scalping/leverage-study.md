# Higher leverage on one shared 50 USDT account

Five policies replay the same fixed cross-exchange basis-deviation signal at isolated leverage 2x, 10x, 25x, 50x, or a volatility-selected multiple. No daily quota, loss-streak lock or fixed cooldown is imposed. There can be one net position per symbol, including simultaneous LTC and XRP positions. Each seven-day trial resets to 50 USDT. The combined trials cover September 3 through October 21, 2026 UTC in seven disjoint weeks, including three October weeks outside the prior study. Separate single-contract controls cover those dates and four later LTC weeks. October is a held-aside historical comparison, not prospective evidence.

At signal time the sizing rule measures realized volatility from the last 60 five-second midpoint changes. Initial margin ranges from 10% to 60% of portfolio equity, scaled inversely with volatility. Adaptive leverage is 50x below or at 0.10% realized volatility, 25x through 0.20%, 10x through 0.40%, and 5x above. Actual size is reduced by uncommitted collateral, displayed top depth, and quantity rounding. Fixed-leverage controls use the same variable margin. For overlapping positions, mark open P&L as of the new entry; reserve isolated collateral and fees and disallow financing new positions with unrealized gains. Every closed P&L reconciles to the end of the sampled portfolio equity curve.

Costs include the observed spread, market fees of 5.5 bp per side and extra slippage of 1.5 bp per side (4.5 bp in stress). The exit target is 0.60%, the stop 0.30%, and the planned maximum duration 15 minutes. Exit fills may be delayed for available size. Hour-crossing trades are avoided for funding. An isolated liquidation model uses a sampled midpoint proxy with a 0.50% assumed maintenance rate, or a 1% rate plus a 0.50% adverse mark divergence in stress. A modeled liquidation consumes the isolated margin and entry fee. These parameters are sensitivity assumptions. Historical Bybit mark prices, account-specific tier rates, quantity limits and private fills were not recovered. Actual liquidations may differ.

For a combined replay run the README download/decoding commands in one directory per contract for the same dates; three-week tests use 21 days. From the LTC directory run:

```bash
python "$STUDY/leverage_replay.py" LTCUSDT 2026-10-01 3 --other-dir ../xrp-oct
```

The CSV records all 250 weekly policy/cost cases (180 individual, 70 shared-portfolio). Base and stress are separate simulations. On the seven shared weeks, 50x ranged from 33.13 to 52.17 USDT in ending balance with one profitable week, while adaptive leverage ranged from 47.49 to 50.55 USDT. In the three new weeks every policy finished below 50. No production settings or worker changes are included.
