# Adaptive crypto research engine

## Scope and safety
Research-only. Not imported by the paper worker, dashboard order path, or exchange
order endpoints. It emits an analysis candidate, never an order. It is long-only
Spot; bearish, high-volatility and uncertain regimes abstain.

## Process
1. Validate chronological finite OHLCV and require fresh closed bars and a known spread.
2. Calculate causal EMA, ATR, ADX, path-efficiency, RSI, prior-bar range and
   relative volume. Range and volume thresholds are shifted to exclude the signal bar.
3. Classify bullish trend, bearish trend, range, high volatility, or uncertain.
4. Activate compatible experts: bullish trend pullback; lower-range mean reversion;
   prior-range breakout with relative-volume confirmation; otherwise no-trade.
5. Reject candidates when spread/data quality or estimated round-trip cost fails.
   Stops and targets are reference levels, not exchange instructions.

## Backtest and execution assumptions
The research backtester uses closed hourly bars, next-hour-open entries, stop-first
intrabar handling, gap-aware stop exits, time exits, and explicit per-fill commission,
half-spread and slippage scenarios. Simulated quantity is bounded by risk-per-trade
and notional caps. The backtest is single-position per symbol; it is not a shared
multi-asset execution portfolio and it does not create executable order quantities.
The CI study prefers Binance public Spot 1h candles, with completed 1m aggregation
as a fallback. The research workflow fetches a 2-year hourly history for ten liquid
USDT pairs, while the legacy benchmark cache retains its 120-day 1m sample.
Binance history is only a cross-venue proxy for Bybit Demo Spot.

## Fee-aware and passive-execution research additions

- The entry gate now requires the modeled gross reward distance to exceed the
  estimated round-trip cost by a configurable multiple (default 3x), in addition
  to the minimum net reward/risk ratio. The same gate is rechecked at the next
  simulated open so a gap cannot silently invalidate the setup.
- Position sizing includes the estimated round-trip cost in risk per unit.
- Range mean reversion now has rolling Bollinger lower-band and rolling
  volume-weighted-average features, while retaining the ADX/path-efficiency
  range regime filter. The VWAP here is a rolling OHLCV proxy, not exchange
  session VWAP or order-book microprice.
- `tradingagents/crypto/backtest/bybit_execution.py` builds a Bybit V5
  `Limit`/`PostOnly` payload and rounds quotes to tick size. It is a pure
  helper: it does not authenticate, submit, amend, cancel, or poll orders.
  Buy quotes rest at/below best bid; sell quotes rest at/above best ask.
- A PostOnly instruction prevents taking liquidity at submission, but it does
  not guarantee a fill, a maker fee tier, or a profitable exit. OHLCV alone
  cannot model queue position, partial fills, cancellations, adverse selection,
  or maker-to-taker conversion at exit. Do not count a hypothetical passive
  quote as filled without order-book/trade replay and conservative queue rules.
- Commission inputs remain explicit scenario assumptions. They must be replaced
  with the account's actual Bybit fee schedule and instrument category before
  any venue-specific conclusion; VIP tier rates are not hardcoded.
- Funding-rate cash-and-carry is deferred. It requires synchronized Spot and
  Perpetual order books, historical funding timestamps/rates, basis, borrow or
  inventory costs, two-leg partial-fill handling, and liquidation/margin modeling.
  A positive funding rate alone is not a risk-free arbitrage signal.

## Current limitations
The engine does not yet consume order-book depth, funding, open interest, liquidation
flow, macro/news, or exchange-specific historical fills. Spread and cost assumptions
are explicit estimates, not verified Bybit fees or realized market impact. Do not
connect it to orders without separate risk/execution integration and validation.

## Research foundations (hypotheses, not profit guarantees)
- Moskowitz, Ooi & Pedersen (2012), Time Series Momentum.
- AQR, A Century of Evidence on Trend-Following Investing.
- What drives cryptocurrency returns? A sparse statistical jump model approach.
- Bailey et al., The Probability of Backtest Overfitting.
- Optimal trend-following with transaction costs.

These works motivate candidate families and validation controls; none proves this
implementation profitable in crypto or on Bybit.

## Required gates before paper/demo integration
Causal/no-lookahead and stale-data tests; multi-symbol chronological walk-forward
across multiple market cycles; measured Bybit fees/spread/slippage; untouched final holdout;
selection-bias analysis; buy-and-hold and cash baselines; per-symbol and per-regime
expectancy, profit factor, compounded return, drawdown, turnover and uncertainty.


## Bybit public-data execution replay (separate research branch)

The `execution_replay` module records public Spot or Linear `orderbook.{depth}.{symbol}`
and `publicTrade.{symbol}` WebSocket messages to JSONL. Install the optional
collector dependency with `pip install ".[market-data]"`. Collection is read-only
and does not use API credentials. Preserve raw receive timestamps alongside Bybit
exchange timestamps; discard intervals with disconnects, stale books, or sequence
gaps rather than interpolating missing market state.

The offline replay:
- Applies L2 snapshots/deltas and uses Bybit public trade taker side to identify
  eligible passive fills. Queue ahead is initialized from displayed size times a
  configurable queue factor (default 1.5).
- Credits fills only after eligible trade volume consumes modeled queue ahead.
  Book-size decreases are not treated as fills because public L2 cannot identify
  whether size disappeared due to cancellation or execution.
- Models submission latency, PostOnly crossing rejection at activation, TTL expiry,
  and partial fills. Queue cancellation credit is zero by default.
- Reports any-fill/full-fill rates, partial/unfilled/rejected orders, time-to-first-fill,
  and signed 1s/5s/30s mid-price markouts net of the supplied maker-fee assumption.

Important limitations:
- Public L2 does not reveal individual order identities or exact FIFO queue priority;
  queue factor is a conservative sensitivity assumption, not observed position.
- The markout is a post-fill mid-price diagnostic, not realized round-trip PnL.
  A complete strategy evaluation must model exit orders, exit taker/maker fees,
  inventory exposure, and realized cash/equity paths.
- The collector records a bounded session to JSONL and reconnects with backoff;
  disconnect/error markers invalidate the replay book until a fresh snapshot. Replay
  also invalidates on detected orderbook update-ID gaps. Do not use incomplete sessions
  for conclusions. No order placement, cancel endpoint, authentication, or private
  account access exists in this module. Linear market-data capture is supported, but
  funding cashflows, settlement, margin, and liquidation are not modeled; do not use
  this prototype to claim net perpetual returns.
- Instrument validation is a pure function over V5 instrument metadata. Spot uses
  `minOrderAmt`; linear/inverse use `minNotionalValue`, alongside tickSize,
  qtyStep, minOrderQty, and maxOrderQty where supplied. Fetch fresh metadata and
  book state before any future order adapter; this branch does not send orders.
