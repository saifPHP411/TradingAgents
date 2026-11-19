# Seven-day perpetual futures portfolios

Research benchmark for a 50 USDT account with a 1,000 USDT realized target. The data universe is BTC, ETH, SOL, XRP and DOGE linear perpetuals. Each weekly portfolio starts flat with 50 USDT; no external deposits. Only one position at a time, long or short. Weekly trials cannot be added as a continuous account.

## Candidate comparison

Three rules use completed four-hour candles. Trend rotation ranks ATR-normalized 24-hour movement aligned with the fast/slow trend. Channel breakout ranks the distance of a closing breakout above or below the preceding 12 bars' extreme, scaled by ATR. Range reversion fades two-standard-deviation deviations during weak trends. Each strategy selects the highest absolute qualifying score across the five markets with feasible order size, at the next 15-minute open, 15 minutes after signal confirmation. The 14-bar ATR sets the stop and target. The time limit is 24 hours for range reversion or 48 hours for other strategies.

Half the settled balance funds isolated margin at leverage 2, 10, 25 or 50 (maximum initial exposure 1, 5, 12.5 or 25 times balance). Quantities round down. Stop, target, funding, forced liquidation, and last-bar exits affect a shared balance. The target price may be reduced to close when the account reaches 1,000. The assumed taker fee is 0.055% each fill plus 0.03% slippage (0.10% in stress). Funding uses historical signed settlements. A long pays positive funding. No funding is charged for entry after a settlement at the same opening timestamp. The liquidation screen uses historical mark highs/lows, assumed first-tier maintenance (0.5% Binance baseline, 1% stress) and a fee allowance. Liquidation takes precedence over a stop and the stop over a target if both are possible inside a bar. Liquidation consumes isolated margin, entry fee and any modeled funding debit. It is an approximation; candle extrema do not supply path, queue or fill evidence.

The 51 Binance USD-M weeks start 2025-11-03. First 35 weeks are development, final 16 are evaluation. Choosing by highest development median would have selected range reversion at 2x. The 50x channel breakout was examined more closely after observing the outstanding evaluation week; that observation is subject to selection bias. Daily mark data for 2026-06-29 replaces a missing day in the June monthly archive. Leverage and instrument filter assumptions for Binance are encoded in weekly.py. All closed balances, sampled peaks, drawdowns, modeled liquidations and funding totals appear in compressed CSV format.

## Replication

Minute trade and mark candles replay all 16 evaluation weeks of 50x breakout, retaining the same signals, entry clock and duration. Eight earlier weeks beginning 2025-05-05 serve as a separate disjoint block without changing the rules. The Bybit public V5 replay reconstructs the August week and two subsequent November weeks with that venue's own trade candles, mark candles, funding, four-hour signals, and current first-tier risk specifications. Historical tier/filter changes, executable depth, partial fills, tick rounding, and latency beyond modeled slippage remain unverified. August was selected on Binance; the same week on another venue is a cross-venue reproduction of one market event. November is a distinct period. Conditional on the price crossing the modeled goal-exit level, the fill can produce exactly 1,000. Prices are simulated and funds never leave the paper account.

Run with numpy and pandas in an empty data directory with these scripts available on the Python path:

```sh
python data.py
python weekly.py
python minute_check.py
python minute_check.py --replication
python bybit_check.py
```

Scripts access public endpoints only and cache data in the working directory. All outputs describe simulations, including the August trade ledgers. Download swing-weekly.csv.gz and decompress it to examine all 1,224 coarse scenarios. minute-check.csv, replication.csv and bybit-check.csv list the granular results. The bybit-instrument-assumptions file records quantity increment, minimum notional, maintenance margin rate, maximum leverage and first-tier risk limit in that order.

The target was hit in one of 51 base-case breakout weeks on the coarse Binance replay. On the one-minute replay it was hit in one of 16 evaluation weeks (median 10.12 USDT) and zero of eight additional early weeks (median 15.54 USDT). For the selected week, public Bybit candles and funding reproduce 1,000 USDT base or 415.46 stressed. Two subsequent Bybit weeks finish at 1.00 and 244.51 base, or 0.75 and 0.13 stressed. This is evidence for a retrospective occurrence, not reliable weekly returns. Nothing is deployed and no exchange orders were submitted.
