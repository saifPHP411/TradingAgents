# Bybit Demo IOC execution comparison — research only

This comparator is intentionally separate from the live/Demo trading worker. It
does **not** generate signals, submit orders, or enable scalping.

## Purpose

Compare *the same recorded, timestamped 1m/5m/15m scalper signals* under:

- **Market IOC (modeled):** walks the visible Bybit linear perpetual order
  book at the modeled order send time. By default, there is no market-price cap
  in this model. Actual Bybit Market IOC uses its own protective price limit, so
  this case is an approximation, **not** a replay of real exchange fills.
- **Limit IOC (10 bps):** buy cap = floor_to_tick(signal Mark × 1.001);
  sell floor = ceil_to_tick(signal Mark × 0.999), mirroring PR #19.

Both models take only the liquidity available at the first orderbook event at
or after \`signal_ts_ms + latency_ms\`; unfilled quantity is canceled. No
maker-price assumption, no queue priority, no invented fills on a later candle.

Historical Bybit L2 daily dumps for both LTCUSDT and XRPUSDT exist at:

\`https://quote-saver.bycsi.com/orderbook/linear/{SYMBOL}/{YYYY-MM-DD}_{SYMBOL}_ob200.data.zip\`

Use the **linear** archive, not the spot archive. Review archive integrity and
coverage before drawing conclusions.

## Required signal trace

Provide a CSV containing real strategy signal timestamps, not timestamps
chosen retrospectively from profitable candles:

\`\`\`csv
signal_id,symbol,signal_ts_ms,side,signal_mark,quantity,tick_size,signal_expires_ms,exit_price
\`\`\`

- \`signal_ts_ms\`: actual UTC millisecond timestamp when the signal was
  generated and eligible for order submission, **not** the candle's timestamp.
- \`signal_mark\`: live Mark Price at that instant.
- \`side\`: \`long\` or \`short\`.
- \`quantity\`: sizing approved by the broker with the **50 USDT starting
  equity and all applicable risk limits**. The comparator does not size trades.
- \`tick_size\`: historical instrument price tick.
- \`signal_expires_ms\`: end of the current 1m candle (optional: derived from
  \`signal_ts_ms\`).
- \`exit_price\` (optional): independently observed, valid exit for a
  *conditional* P&L sensitivity calculation. This is **not** an accurate
  strategy-level TP/SL replay because the two order types can enter at
  different times, quantities, and prices.

Signals must be recorded *before* outcome analysis. A signal from a November
Demo session is not a legitimate October signal. If no October signal trace
exists, reconstructing one from 1m candle **closes** instead of the intraminute
live Mark trigger would create a different strategy. This tool deliberately
refuses to invent those signals.

## Run

Download only the daily archives needed by the real signal trace; then run:

\`\`\`bash
python -m tradingagents.crypto.backtest.ioc_compare \
  --signals research_signals.csv \
  --archive 2026-10-01_XRPUSDT_ob200.data.zip \
  --archive 2026-10-01_LTCUSDT_ob200.data.zip \
  --output results/ioc_compare_2026-10-01.json \
  --latency-ms 250 --max-book-delay-ms 1000 \
  --fee-per-fill 0.00055
\`\`\`

Compare alternative latency assumptions (0, 250, 1000 ms) and, separately,
use \`--market-cap-bps\` to test hypothetical Market IOC matching bounds.
The fee flag is **a research assumption** and must be replaced by the actual
Demo account's taker fee tier. Exits need separately measured slippage,
stop/target timing, funding where relevant, and position tracking before a
true net strategy return or 50 USDT equity curve can be published.

The output reports sample count, excluded/no-book signals, any/full/quantity
fill rates, signed average adverse entry slippage, and entry taker fees.
Conditional P&L is explicitly null without exits. If all fills fail or data are
missing, the comparator reports no result instead of a fabricated zero-return
study. Sample must be split into development and independent holdout periods.

## Limitations / reproducibility

Historical archived books do not establish that exchange matching would have
given exactly the displayed quantity. The model excludes changes during
request latency after the sampled book, hidden liquidity, dynamic venue price
limits, queueing, order rejections, margin/funding, and dynamic trade exit
logic. Timestamp and fee assumptions must accompany every result. Do not
turn on the Demo worker solely because modeled entry slippage improves.

## Bounded real-archive access check

The optional `IOC archive prefix validation (research)` GitHub workflow probes
the first 512 KiB of the actual 2026-10-01 XRPUSDT and LTCUSDT linear ZIPs.
It uploads individual JSON reports, including *unavailable_or_invalid* evidence
when network access or schema validation fails. A passing workflow job does
**not** imply the remote archive passed validation: inspect the individual
report's `status`, which must be `prefix_verified`. This check establishes
only file prefix schema/reachability; it does not verify a day's CRC, full
coverage, an intraminute historical mark feed, or profitability.

To check a single symbol locally without downloading the full day:

```bash
python tradingagents/crypto/backtest/archive_probe.py \
  --symbol XRPUSDT --date 2026-10-01 \
  --out results/l2-prefix-XRPUSDT.json
```

The research PR never deploys or enables the Demo worker.


## Bounded real-L2 quote-only shadow comparison (new)

**This does not bypass the missing historical strategy signal trace.**
The true Market IOC vs signal-anchored Limit IOC backtest still requires
the original pre-existing intraminute scalper signal timestamps, actual
**Bybit Mark prices**, trade sizes, and independent exit fills. None of
those fields are constructed from profitable-looking future candles.

To obtain a **real order-book execution *diagnostic*** in the meantime,
`ioc_quote_prefix_benchmark.py` examines the first **8 MiB compressed**
of public Bybit **linear** L2 ZIPs from each of four *predeclared* days
(2026-10-01, 08, 15 and 21), for XRPUSDT and LTCUSDT. It does not
download or assume a complete historical day. The workflow is
`.github/workflows/ioc-quote-prefix-benchmark.yml`.

A deterministic **quote-proxy sampling schedule** takes the first valid
L2 quote in each completed 15-second UTC bucket and alternates long and
short sides by the fixed bucket number. The quote midpoint is a **proxy**
for Mark price, not the actual archived Bybit mark or a strategy trigger.
It sizes each hypothetical request to about **10 USDT notional**, using
UNVERIFIED research tick assumptions (XRP: 0.0001, LTC: 0.01). Book
snapshots and subsequent deltas are replayed to the exact arrival
timestamp without reading future deltas. Three bounded latency cases
are checked: 0, 250 and 1000 ms, with a 1000 ms maximum quote age.

Both entry policies consume **identical** sampled quote-proxy events:
- Market IOC: unlimited modeled visible book walk, a deliberately
  optimistic fill benchmark. Real Bybit Market is **not unlimited**:
  according to its V5 documentation, it is internally converted to
  an IOC limit order with venue-managed price protection, and can
  partially fill/cancel.
- Limit IOC: a directionally rounded price cap of 10 bps against
  the **sampled midpoint proxy**, not against the live signal mark.
  Partial fills and immediate cancel of the unfilled remainder are
  included. Taker fees are *estimated* at 5.5 bps per side.

Outputs are deliberately named `sampled_quote_only_not_strategy_backtest`.
Report only measurable simulated **entry** properties (any/full/quantity
fill rates, average adverse entry slippage, paid estimated entry fees,
extra Market-filled quantity). Never present a fixed quote-sample rate as
a forward or full-day strategy fill rate. **Strategy P&L and net weekly
profit are NULL** because authentic signals, independent exits, Bybit
funding, mark-price triggers, and venue order constraints are unverified.

Run manually with:

```bash
python tradingagents/crypto/backtest/ioc_quote_prefix_benchmark.py \
  --symbol XRPUSDT --date 2026-10-01 \
  --output results/ioc-shadow-XRPUSDT-2026-10-01.json
```

For a **genuine scalper backtest** use the original `ioc_compare.py`
with recorded historical signal CSV and full archive, and reconcile the
exit fills and real 50-USDT account trade ledger. Do not substitute
these quote-proxy observations for original signal records.

Official source for Market IOC's actual hidden venue protection:
[Bybit V5 Place Order](https://bybit-exchange.github.io/docs/v5/order/create-order).

