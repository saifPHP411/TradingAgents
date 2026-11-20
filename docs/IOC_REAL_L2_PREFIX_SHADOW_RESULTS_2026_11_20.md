# Real-L2 execution check — Market IOC vs 10 bps Limit IOC

**20 November 2026. Research-only diagnostic; not a scalper strategy backtest.**

## What was actually tested

The previous PR #20 Market-versus-Limit simulator already existed and
passed synthetic tests, but it had **no genuine saved October 2026
intraminute scalper signal trace**, so it could not perform the desired
strategy-level historical experiment. This new stage produces a narrower,
honestly labeled **quote-only execution comparison** on public Bybit linear
order-book messages with identical *predeclared pseudo-signals* fed to
both modeled execution policies.

**Source/replication:** [GitHub Actions run #44812451750](https://github.com/saifPHP411/TradingAgents/actions/runs/44812451750) —
all eight archive-quote jobs succeeded. Code:
`tradingagents/crypto/backtest/ioc_quote_prefix_benchmark.py`;
tests `tests/test_ioc_quote_prefix_benchmark.py`;
workflow `.github/workflows/ioc-quote-prefix-benchmark.yml`.

- Symbols: **XRPUSDT, LTCUSDT** (Bybit linear futures/perpetual books).
- Precommitted dates: **Oct 1, 8, 15, and 21, 2026** for *each* symbol.
- For each archive, fetch only the **first 8 MiB** of the compressed
  ZIP via HTTP Range 206, decode only complete L2 JSONL events and
  reconstruct snapshots + deltas. **Not the complete trading day**.
  These observations overweight the earlier hours of each date.
- At first valid quote per **15-second UTC bucket**, assign LONG for
  even buckets and SHORT for odd buckets; alternate deterministically
  without using a future outcome. Use the current best-bid/best-ask
  **midpoint as a proxy for the true Mark price**.
- Hypothetical notional **10 USDT** per independent shadow request.
  Assumed research tick increments XRP 0.0001 and LTC 0.01 have **not**
  been independently verified against historic Bybit instrument filters.
  No shared account was actually operated.
- Market IOC benchmark: walk available depth without modeled cap. **Actual
  Bybit Market IOC is exchange-protected internally**, so this is an
  optimistic execution benchmark, not its true venue-specific behavior.
- Limit IOC: directional cap **±10 bps relative to the quote midpoint
  proxy**, not the original live signal Mark.
- Only consume book updates up to order arrival with three delays:
  **0, 250 and 1,000 ms**; disqualify stale books.
- Estimated taker entry fee: 5.5 bps; no measured exits, stop orders,
  funding or realized account P&L.

## Results: 8 bounded historical source excerpts

At **250 ms** latency:

| Symbol | Date (UTC) | Fixed quote samples | Market full modeled fills | 10bp Limit full modeled fills | Market vs Limit mean adverse entry slippage (bps) |
|---|---|---:|---:|---:|---:|
| XRPUSDT | Oct 1 | 714 | 100% | 100% | 0.32885 / 0.32885 |
| XRPUSDT | Oct 8 | 579 | 100% | 100% | 0.35142 / 0.35142 |
| XRPUSDT | Oct 15 | 528 | 100% | 100% | 0.38516 / 0.38516 |
| XRPUSDT | Oct 21 | 795 | 100% | 100% | 0.37470 / 0.37470 |
| LTCUSDT | Oct 1 | 1557 | 100% | 100% | 0.75329 / 0.75329 |
| LTCUSDT | Oct 8 | 1689 | 100% | 100% | 0.76120 / 0.76120 |
| LTCUSDT | Oct 15 | 1266 | 100% | 100% | 0.84115 / 0.84115 |
| LTCUSDT | Oct 21 | 1970 | 100% | 100% | 0.80422 / 0.80422 |
| **ALL** | four dates | **9098** | **100%** | **100%** | **0.66468 / 0.66468** |

These are **9,098 independently sampled hypothetical entry opportunities**,
**not 9,098 real scalper trades or venue-confirmed fills**, and cannot
be added into a 50-USDT account's return. Because both policies found
the same displayed liquidity at tiny 10-USDT notional size and the
recorded quote samples were mostly within 10 bps, the 10-bp limit
did not improve entry price or reduce full fill rates at 250 ms.

At **0 ms** latency the results were also all 9,098 full modeled fills
under both policies; pooled adverse entry slippage was ~0.66992 bps
from the midpoint. At **1,000 ms**, 9,097 of 9,098 pseudo-signal
observations had a fresh enough book; Market filled all 9,097 in
the visible-depth model, while the Limit reached full size on
approximately 99.956% (only **four** of the 9,097 lost full-size
eligibility: three in LTC Oct 15, one in LTC Oct 8).
When both types filled, their modeled prices were the same.
A smaller fill quantity is **not** evidence of a more profitable
strategy; it is the mechanism by which Limit price protection rejects
unfavorable liquidity. No exit profitability was computed.

## Interpretation and hard blockers

1. **Execution benchmark, not trading profitability.** We have no
   original, timestamped live scalper entry signals; a pseudo-signal
   schedule is not equivalent to recovering those signals from archived
   candles. No authentic Bybit historical Mark series is matched here.
2. **Only archive beginnings examined.** Full-day/archive CRC, overnight
   news/move spikes, and illiquid events were not sampled. Strong claims
   about market impact, average daily fill rate, or complete October
   performance would be invalid.
3. **Venue size/price filters unverified.** User's 50-USDT starting
   capital and the fixed 20% entry-notional policy imply ~10-USDT
   orders, but there is no verified historical minNotional, qtyStep,
   active account margin or venue order acceptance. Some requests
   might be rejected entirely on exchange.
4. **No independent exits or account P&L.** Entry-price/fee estimates
   alone cannot establish Win Rate, Profit Factor, final capital or
   a claimed >=50-USDT weekly net return. All output fields for net
   strategy profit are **null**.
5. **Market IOC model differs from Bybit.** As stated in
   [the official Bybit V5 Create Order documentation](https://bybit-exchange.github.io/docs/v5/order/create-order),
   Market orders convert into protected IOC limit orders; the real
   exchange's bounds can reject liquidity too. The unlimited Market
   case here is only a sensitivity baseline.

**Recommendation for next valid strategy comparison:** collect
actual prospective paper/Demo scalper timestamps, signal Mark,
instrument filters, estimated quantity, response/fill records and
independent exit fills **before viewing results**. Then run the
existing `ioc_compare.py` against matching full Bybit L2 archives,
keeping development and untouched holdout periods separate.
Do **not** enable real/Demo order submission based solely on this
diagnostic. No PR was merged, no Railway environment was changed.
