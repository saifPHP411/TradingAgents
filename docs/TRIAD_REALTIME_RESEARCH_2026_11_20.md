# Triad research: Sniper / Analyst / Professor with one-second market watch

**Status:** Draft PR #23, 20 Nov 2026. **Research-only:** no live/demo
execution, no order entry, no new API credentials, no Railway deployment.

## Architecture: three roles, one precommitted signal, three vetoes

This is a deterministic, testable *three-agent voting architecture*, **not**
three independent LLM API calls every second. At one-second frequency,
three model calls would add latency, expense and inconsistent approval state;
the current agents are explicit policies that can be audited.

1. **Sniper (القناص):** the fixed `donchian_15m_trend` entry rule already
   selected as the least weak historical median among eleven frozen
   strategies. Must see a completed 15m breakout against the previous
   3h high/low, aligned EMA trend, adequate recent volume and RSI.
   A forming candle or an expired entry candidate cannot qualify.
2. **Analyst (المحلل):** independently checks RSI, directional 60/240
   minute EMA separation and the direction of the 15m close-to-close
   impulse. Opposite momentum/vague trend => veto.
3. **Professor (البروفيسور):** independently vetoes stale Mark or
   orderbook quotes older than **5s**, wide bid/ask spread above
   **10 bps**, excessive mark/mid deviation, notional above **20% of
   current account equity**, leverage above **2x**, and insufficient
   conservative stop/target payoff after **stress** cost assumptions
   (8 bps taker fee + 7 bps adverse impact per side).

All roles receive the **same symbol, direction, completed candle timestamp
and event ID**; an opportunity requires three approvals. A repeated check
of the same event does not create another opportunity. Missing inputs,
disconnection, stale book, incomplete candles, out-of-order timestamps,
gapped history or model errors -> **no approval**.

**Critical safety:** even unanimous approval yields only a
`new_unanimous_shadow_opportunities` JSON record and
`execution_authorized=false`. There is NO trade-submission client,
stop-loss placement, or path to credentials in the read-only modules.
Broker lot/tick/notional/step rules, actual available margin, funding,
liquidation risk and protective stop placement have **not** been verified
for an account with 50 USDT.

## Real-time data flow — not periodic REST scans

`tradingagents/crypto/triad_live_shadow.py` optionally establishes one
**public linear perpetual WebSocket** at
`wss://stream.bybit.com/v5/public/linear`, subscribing by symbol to:

- `tickers.SYMBOL` for the Mark Price;
- `orderbook.1.SYMBOL` for live best bid/ask;
- `kline.1.SYMBOL` for explicitly **confirmed/closed** one-minute bars.

The market feed is venue-push-driven, not one HTTP request per second.
A local cached-state check runs **each wall-clock second**, including on
a quiet stream, and reports stale quotes as a veto. EMA/Donchian
indicators use closed 1-minute bars; a new 15m strategy signal is
eligible only after that 15m candle closes. A one-second market refresh
is **not** one new trading signal per second.

On startup, the watcher attempts to bootstrap up to 540 completed
1-minute bars from the read-only Bybit public API. If the upstream API
is unavailable (e.g. HTTP 403 on the GitHub runner), it reports
`not_ready` and **does not invent historical candles**. It can warm up
from valid incoming confirmed candles, but cannot approve entries
without enough chronological history. On disconnect it clears live
quotes; on WebSocket re-connect it subscribes again, with bounded
exponential backoff. One-second checks do not silently fall back to
stale data. There is a bound on historical cache size (540 bars/symbol)
and reported duplicate opportunities (256 IDs).

**Manual read-only local research run** after checkout and installing:
```bash
pip install -e ".[realtime]"
python -m tradingagents.crypto.triad_live_shadow \
  --symbols BTCUSDT ETHUSDT XRPUSDT LTCUSDT
```

**Bounded read-only stream check**, no orders:
```bash
python -m tradingagents.crypto.triad_live_shadow \
  --symbols BTCUSDT --max-seconds 25 \
  --out artifacts/triad-public-feed-status.json
```

The optional GitHub Actions workflow `triad-public-feed-probe.yml` is
an *upstream access smoke test only*: it can report unavailable/blocked
public market data. Successful public subscription alone does **not**
verify account readiness, correctness of 15m bootstrapping,
strategy profitability, or live/real order execution. Consult
[Bybit public linear WS connect](https://bybit-exchange.github.io/docs/v5/ws/connect)
and [Bybit L1 orderbook snapshot docs](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook).

## Controlled historical comparison of the best existing candidate

The original `donchian_15m_trend` rule remains **unchanged**.
`donchian_15m_triad_shadow` keeps the exact same:

- entry signal source, 0.8% stop, 1.8% target, 240-minute maximum hold,
- 50 USDT capital reset every seven days and one shared portfolio across
  BTCUSDT/ETHUSDT/XRPUSDT/LTCUSDT,
- 20% of current account equity **maximum position notional**,
  2x policy ceiling (not a multiplier applied twice),
- next-minute modeled open, stop-first same-minute OHLC conservative
  treatment, actual simulated fill fees/impact, full closed-trade ledger,
- identical **zero-cost / realistic / stress** cost scenarios.

Only the additional Analyst and Professor vetoes are different; the
Professor's historical market spread/Mark freshness inputs are
**synthetic fixed assumptions** because the Binance archived 1-minute
candles do not contain real Bybit snapshots/marks. They therefore
cannot certify that historical entry quotes were fresh or executable.
This historical three-agent test is specifically a model of the
*additional filters*, not a true one-second L2 replay.

### Full ten-week historical sample (realistic estimated costs)

From fixed 6 Sep–14 Nov 2026 UTC experimental windows:

| Metric | Original Donchian15m | Original + unanimous triad |
|---|---:|---:|
| Total 7-day independent windows | 10 | 10 |
| Closed trades | 170 | 158 |
| Pooled net win rate | 44.12% | 44.94% |
| Positive weeks | 5/10 | 6/10 |
| Median weekly net profit, USDT | +0.03850 | **+0.12479** |
| Mean weekly net profit, USDT | +0.01701 | +0.05698 |
| Worst weekly net loss, USDT | -0.91708 | **-0.96099** |
| Best weekly net profit, USDT | +0.87452 | +0.84881 |
| Weeks earning >=50 USDT net | **0/10** | **0/10** |

Evidence: [GitHub Actions triad-vs-original run #44814850247](https://github.com/saifPHP411/TradingAgents/actions/runs/44814850247).
The historical-account gate covers both strategies, every one of the
ten weekly windows, three cost cases, no unexamined gaps, account
reconciliation and explicit missing execution evidence.

**Result: marginal retrospective improvement in number of positive weeks
and median P&L, but slightly worse worst-week drawdown and absolutely no
support for +50 USDT/week.** Gains of pennies must not be presented as
"50 USD/week achieved." Do not treat the filtered result as a new
out-of-sample success: these historical windows were already examined
while choosing Donchian as the starting hypothesis.

## Readiness gates (must remain BLOCKED)

The experiment is intentionally on a *stacked Draft PR*, based on
the previous research Draft #22; earlier Bybit Demo entry protection
PR #19 and IOC execution-replay PR #20 remain separate/unmerged.

Before integrating any unanimous vote into authenticated Bybit Demo
worker: prove actual exchange minQty/qtyStep/minNotional from the
correct account/region, venue snapshot/fill reconciliation, all
stop/target protections for partial IOC fills, real Mark/quote timing,
funding and liquidation path, precise risk/position lock, and
prospective forward evidence of **net** profits. Otherwise remain
read-only. **Never assume three agreeing signals create an economic
trading edge.** No change to leverage, capital, Railway deployment,
or order submission is authorized by this research.

