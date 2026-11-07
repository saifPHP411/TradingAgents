# Crypto TJR

This project runs the **unchanged** TJR intraday rule chain on a 24/7 crypto
market. It does not re-implement the strategy for crypto, and it does not let an
LLM invent a signal: the same rules, the same risk engine, the same ledger, the
same broker.

```
Binance public REST + WebSocket
  -> tradingagents/crypto/          normalization, symbols, instruments, freshness
    -> tradingagents/tjr/           the unchanged rule chain over a UTC-day session
      -> tradingagents/risk/        RiskEngine (the only place a limit is enforced)
        -> tradingagents/paper_experiment/   the paper broker, orders, fills, ledger
          -> tradingagents/paper_experiment/dashboard/   the dashboard API and view
```

## The pieces

| Module | What it owns |
| --- | --- |
| `crypto/symbols.py` | `BASE/QUOTE` parsing and canonicalization (`BTC/USD` -> `BTCUSDT` on Binance) |
| `crypto/binance.py` | The public REST client: ticker, book, candles, exchange info, server time |
| `crypto/stream.py` | The public WebSocket: ticker stream, reconnection, heartbeat, stale detection |
| `crypto/engine.py` | The live engine: per-symbol state, freshness, `stream_status` |
| `crypto/instruments.py` | Trading rules per pair: minimum notional, quantity/price precision, order types |
| `crypto/universe.py` | Resolving a requested pair list against the venue's tradable pairs |
| `crypto/backtest/` | History fetch, cache, the multi-period study, and the report |

## Market data

The provider is **Binance's public market data**. It needs **no API key** and no
account: every endpoint used is public and documented at
<https://developers.binance.com/docs/binance-spot-api-docs>.

* **REST** — `ticker/24hr`, `ticker/bookTicker`, `ticker/price`, `klines`,
  `exchangeInfo`, `time`.
* **WebSocket** — the combined stream (`/stream?streams=...`) carrying the
  `ticker` and `bookTicker` streams.

What the live view shows, and where each field comes from:

| Field | Source |
| --- | --- |
| Last price | `ticker/24hr` last price, refreshed by the `ticker` stream |
| 24h change / high / low / volume | `ticker/24hr` |
| Bid / ask | `bookTicker` (stream and REST) |
| Spread | computed from the bid and ask, in the quote currency |
| 1m / 5m candle state | `klines`, closed candles only for decisions |

**Freshness is explicit.** Every snapshot carries the instant it was received and
the instant the venue stamped it. The view reports `fresh`, `stale` or `no_data`;
a stale price is never shown as a live price. A disconnected stream leaves the
last known values in place *labelled* as stale, with the disconnect visible.

**Reconnection** is built into the stream: a dropped connection retries with
backoff, and duplicate events (a repeated ticker for the same symbol and event
time) are ignored.

### Trading rules

`crypto/instruments.py` reads the venue's `exchangeInfo` and records, per pair:
the tradable status, the minimum notional, the quantity precision, the price
precision and the supported order types. A pair the venue does not list, or lists
as not tradable, is **refused** — never sent.

## The TJR chain on a 24/7 market

The rule chain is the equity one, unchanged:

1. draws on liquidity (swing highs/lows, equal highs/lows, session levels),
2. a liquidity sweep,
3. a 5m reversal confirmation (BOS or IFVG),
4. a 5m continuation confluence (FVG or equilibrium),
5. a 1m entry confirmation (BOS or IFVG),
6. entry, stop and target taken from the structure and the liquidity.

What had to be decided for a market that never closes is **what a session is**.
The equity chain labels bars against an exchange session (09:30-16:00 ET). For
crypto the default is `utc_day_session`: the session is the **UTC calendar day**,
00:00-24:00 UTC. This is an explicit choice, not an accident:

* midnight UTC is **not** treated as a market open — there is no opening gap, no
  auction and no "first bar of the day" special case;
* the session high and low are the high and low of the UTC day, which is what the
  liquidity levels are measured against;
* the last entry window is the end of the UTC day, not 16:00 ET.

`sessions.py` also offers named policies (London, New York, Asia) so the
sensitivity of the strategy to the session definition can be measured rather
than assumed. The study's sensitivity section runs the named policies and reports
the difference.

The strategy is long-only. A complete bearish setup returns
`SHORT_SETUP_LONG_ONLY`, never a fabricated long.

## Backtesting

The study runs the unchanged chain over **real** Binance history. It is separate
from the live path: the history is fetched once by a human-run command, cached,
and then read offline.

```bash
# Fetch the history (the only network entry point in the backtest package).
python -m tradingagents.crypto.backtest.fetch --symbols BTC/USD ETH/USD --days 30

# Run the study.
python -m tradingagents.crypto.backtest.run --periods 7 30 --out study.json --markdown study.md
```

The study reports, per period and per symbol: final equity, net return, win rate,
profit factor, maximum drawdown, Sharpe, Sortino, average R, the planned R:R, the
trade counts, the fees, the slippage, the exposure, the turnover, the rejections
and an equal-weight buy-and-hold benchmark. It also runs a walk-forward (as
disjoint out-of-sample folds), a regime split (bull / bear / sideways), a cost
scenario set and a session-policy sensitivity set.

**Sizing.** A study sizes a position as a fraction of the book's own equity
(`study_risk_limits`). It does **not** use the paper account's 25-USD order cap:
that cap is the operator's live limit, and against a 100,000 USD research book it
would make every position worth 0.025% of it, so the reported P/L would measure
rounding rather than the strategy. The paper worker still uses
`paper_parity_limits()` unchanged.

**Costs.** A run with no cost model charges nothing, and says so. The study's
default figures come from the explicit scenarios:

| Scenario | Commission | Half-spread | Slippage |
| --- | --- | --- | --- |
| `frictionless` | 0 | 0 | 0 |
| `realistic` | 10 bps (0.10%) | 1 bp | 2 bps |
| `conservative` | 15 bps (0.15%) | 2 bps | 5 bps |

The realistic scenario uses Binance's published **spot taker rate of 10 bps** as
its commission, with a stated 1 bp half-spread and 2 bps slippage. These are
stated assumptions about round-trip friction, not a quote from a specific
account's fee schedule. Change them in `cost_scenarios()` to match an account's
real fees.

**Honesty about coverage.** A requested 90-day window that the venue only has 30
days of is labelled `90d_requested_30d` and reports 30 sessions. It is never
padded with synthetic bars.

## Paper trading

The paper worker runs on the **Alpaca paper endpoint only**
(`https://paper-api.alpaca.markets`). The live endpoint is never read and there
is no fallback path to it: a request to a live host is refused, not redirected.

The risk limits are the paper account's, unchanged and shared across the whole
portfolio:

| Limit | Value |
| --- | --- |
| Max order notional | 25 USD |
| Max open positions | 5 |
| Max daily orders | 10 |
| Shorts | not allowed |
| Fractional quantity | allowed (crypto quantity, not whole shares) |

No decision source and no dashboard control can bypass the `RiskEngine`: the
worker sizes and approves every order through it, and an order it refuses is
recorded with a named reason.

**Decision sources** (three, deliberately separate):

* `history` — replays a recorded experiment's decisions.
* `current_market` — asks the LLM over a live, point-in-time market window.
* `tjr` — runs the deterministic TJR chain over live Binance 1m/5m bars. No
  model, no invented signal.

**Market data for the order's price** is a separate seam from the decision's
data: the decision reads Binance's market-data feed, and the order is priced by
the worker's own `--paper-market-data` seam. An order is never priced from the
decision window.

### Running it

```bash
# Paper trading, continuous, TJR decisions, live Binance pricing.
TRADINGAGENTS_PAPER_DECISION_SOURCE=tjr \
TRADINGAGENTS_PAPER_MARKET_DATA=binance_snapshot \
python -m tradingagents.paper_experiment --paper-continuous

# One cycle only.
python -m tradingagents.paper_experiment --paper-once

# The dashboard, with the live crypto section.
python -m tradingagents.paper_experiment.dashboard.server --crypto-symbols BTC/USD ETH/USD --crypto-live
```

Paper trading is **off by default** and never starts on its own. The dashboard's
start/stop buttons control it, and stopping blocks new entries while leaving the
open positions and orders visible.

## The dashboard

The crypto section reads the real backend: the live market rows come from the
crypto engine, the strategy rows come from the decision source's own evaluation,
and the health row reports the actual connection and freshness state. Nothing is
a placeholder; where there is no data, the view says `no_data` rather than
showing a zero.

The dashboard polls the crypto endpoints on their own timer (faster than the
base dashboard poll) and shows a stale or disconnected state clearly.

## What is *not* here

* **No live trading path.** `ExecutionMode` has only `BACKTEST` and `PAPER`.
* **No crypto derivatives.** Spot only. Funding fees are not modelled because
  futures are not supported.
* **No automated exchange execution.** Alpaca is the broker; Binance is the
  market-data source. They are deliberately not mixed.
