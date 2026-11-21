# Regime-aware multi-strategy portfolio — shadow research only

This version adds six deterministic **candidate families**, selected separately
for each USDT perpetual by a reproducible regime classifier on complete 5m
and 15m traded candles: EMA pullback, Donchian breakout, MACD momentum,
Bollinger+RSI reversion, VWAP+RSI reversion, and ATR-filtered volatility breakout.

Regimes are `trend_up`, `trend_down`, `range`, `volatile`,
`transition`, and `extreme_risk`. Predeclared priorities:
- Trend: Donchian → EMA pullback → MACD.
- Range: Bollinger+RSI → VWAP.
- Volatile: ATR-confirmed breakout.
- Extreme risk or transition: **no trade**.
Any disagreeing candidate directions are an unconditional abstention.

### Data provenance and chronology

Selection consumes only the public API's **completed** 5m/15m candles.
Monotonic UTC timestamps, uninterrupted 5m/15m chronology, valid OHLCV,
and most-recent-candle freshness are required. Forming 1m prices are not
inputs to the strategy signals; live Mark Price is used only to project
model stop/target distances. Missing data fails closed.

Research stop/target estimates require a maximum 1.5% stop fraction and
a two-to-one target greater than the assumed 30 bps round-trip stress
cost plus 10 bps surplus. They are **not** executable Bybit protections.

### Research journal and access

The adaptive desk writes one report per (symbol, finished 5m candle)
to its SQLite volume (with a 14-day bounded retention window) and exposes the most recent per-symbol observations
via the existing *authenticated* `/api/learning` endpoint. The burgundy
mobile dashboard shows the current regime, selected candidate, direction,
reason and stale state. Reports explicitly say `shadow_only=true` and
`execution_authorized=false`. The legacy Gemini sniper + Professor path
remains unchanged, including 2x leverage and all venue risk controls.

### Exclusions and honest limits

This is a real, automated **regime selector for research**, not a
validated profitable multi-strategy execution engine. No candidate
can be auto-promoted based on a favorable in-sample regime score.
Every family still needs a comparable, fee/funding/lot/slippage-aware
multi-week walk-forward study, then independent holdout and prospective
Bybit Demo verification **before** consideration for execution.

Multi-leg funding-basis, pairs/statistical arbitrage, orderbook market
making, and cross-sectional momentum are labeled *not implemented* because
candles and the current single-position broker cannot model those honestly.
The EMA scalper's losing historical Binance USD-M proxy is **not**
evidence of a profitable edge. No PR is merged, no Railway service changed,
and no order is submitted by this feature.
