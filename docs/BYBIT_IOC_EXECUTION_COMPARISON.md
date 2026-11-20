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
