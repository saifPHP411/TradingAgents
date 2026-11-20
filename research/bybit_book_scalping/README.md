# Bybit scalping with a 50 USDT account

Historical research with read-only public data. Five fixed entry rules use the same exits and sizing. Each seven-day episode and each strategy starts with 50 USDT. Contract episodes are evaluated separately. All times are UTC.

## Replay

Compile the order book decoder with a C++17 compiler. Install pandas and numpy for Python. Set STUDY to the absolute path of this directory. In an empty working directory for one symbol, run:

```bash
g++ -O3 -std=c++17 "$STUDY/book.cpp" -o book
python "$STUDY/download.py" LTCUSDT 2026-10-22 28
for archive in 20??-??-??.zip; do
  (set -o pipefail; unzip -p "$archive" | ./book > "${archive%.zip}.csv") || exit 1
done
python "$STUDY/verify.py" 2026-10-22
python "$STUDY/study.py" LTCUSDT 2026-10-22
```

Repeat in separate directories for LTCUSDT and XRPUSDT starting 2026-09-03. The cross-exchange variant downloads Binance USD-M minute closes and uses each close only after the completed minute plus five seconds. A missing reference observation cannot generate an entry.

## Design

The decoder applies daily Bybit linear snapshot and delta events, drops unsynchronized books until the next snapshot, and records the first quote in each five-second bucket. Signals see current and past quotes. Simulated market entries arrive at the next observation, using the opposite best quote and visible size. Market exits use executable-side quotes and sampled stop/target observations. A reactive exit waits an additional observation, capped by the time exit. Risk limits approximate the demo worker: 20% maximum entry notional per position (2x leverage, 10% margin), one position, 0.3% stop, 0.6% target, up to 15 minutes, three minutes cooldown, daily lock after three consecutive losses or 3% daily loss, and lock at 10% drawdown until the weekly reset. Order quantity is rounded down to a research increment of 0.1 LTC or 1 XRP, with a 5 USDT notional floor. Historical symbol filters were unavailable; these are model assumptions. The time-to-exit may exceed the deadline by a sample or more if top size is insufficient.

Market fees are 5.5 bp per side and additional modeled slippage is 1.5 bp (base) or 4.5 bp (stress). The passive strategy pays 2 bp for its simulated passive entry. It posts at the observed same-side quote with a 30-second lifetime. The next observation must permit a post-only entry; a subsequent opposite quote must move strictly through its price. Queue priority, private fills, partial fills, and cancel latency are unobserved. All exits are market orders. New trades avoid the last 15m45s of each UTC hour to avoid crossing a funding timestamp.

Rules include a five-minute breakout aligned with book pressure, a three-minute price shock followed by reverse pressure, a compressed-range breakout, a passive retest following momentum, and convergence to a Binance reference after removing the trailing median basis. Pressure uses displayed depth at the top five levels; changes from cancellations and executed trades are not individually identified. Market rules are checked on five-second samples. Stop triggers use executable quotes, not Bybit Mark Price stops. No leverage multiple is applied again to P&L once position size is set.

## Reading results

The JSON contains weekly ending equity, peak sampled liquidation equity including exit costs, maximum drawdown, wins, trade count and profit factor. Results are historical and sensitive to fill assumptions. Existing November candle studies overlap some dates. September provides a different historical regime. Simulations for multiple strategies or contracts must not be summed into a single account.

The perfect-timing benchmark is deliberately non-tradable. For each exit sample, dynamic programming considers all earlier entries in the last 15 minutes, both sides, and staying flat. It applies actual spread and 5.5 bp fees on both legs, allows fractional quantities with 20% exposure, ignores slippage, funding and risk locks, and permits immediate re-entry. Its value is the optimum **inside that one-market, sampled, taker-only model**. It does not bound passive fills, other symbols, other sampling grids or future weeks. The verification script compares it with exhaustive enumeration on a small price path and separately checks one hour of the decoded order book against a Python implementation.

## Observed weekly outcomes

All base-cost account episodes reset to 50 USDT. The 12 contract-weeks cover eight distinct calendar weeks.

| Contract | Week starting | Best ending balance across five rules | Rule | Perfect-timing market-order benchmark |
|---|---|---:|---|---:|
| LTCUSDT | 2026-09-03 | 49.8988 | reference_gap | 111.60 |
| LTCUSDT | 2026-09-10 | 50.0283 | reference_gap | 91.35 |
| LTCUSDT | 2026-09-17 | 49.8661 | reference_gap | 176.73 |
| LTCUSDT | 2026-09-24 | 49.9077 | reference_gap | 282.96 |
| XRPUSDT | 2026-09-03 | 49.8390 | exhaustion | 110.44 |
| XRPUSDT | 2026-09-10 | 50.1909 | reference_gap | 154.85 |
| XRPUSDT | 2026-09-17 | 49.6200 | reference_gap | 224.70 |
| XRPUSDT | 2026-09-24 | 49.9517 | reference_gap | 178.65 |
| LTCUSDT | 2026-10-22 | 50.0151 | exhaustion | 96.14 |
| LTCUSDT | 2026-10-29 | 49.9116 | exhaustion | 110.01 |
| LTCUSDT | 2026-11-05 | 50.0635 | reference_gap | 102.53 |
| LTCUSDT | 2026-11-12 | 50.0594 | reference_gap | 163.10 |

See weekly-results.json for every variant and stress case.
