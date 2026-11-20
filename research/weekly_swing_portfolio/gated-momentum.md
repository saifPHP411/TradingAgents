# Relative momentum gates on 24 perpetual contracts

This study compares the existing Bitcoin-relative momentum score with three predetermined entry screens. One requires an absolute score of at least 2.5; one requires alignment with the hourly Bitcoin EMA trend and a net breadth of at least three signals in the same direction; the last applies both. The simulator ranks the available signals, trades at most one position at a time, and uses completed hourly data with a 15-minute entry delay. Model costs include fees, slippage, historical funding, and mark-price liquidation. The 50x research configuration uses half the current account balance for isolated margin. The simulation caps realized equity at the target via a price-triggered exit.

The 24 instruments are the previously verified Bybit list. Binance trade, mark price, and funding archives provide the historical test inputs. Current Bybit quantity limits approximate a subset of the historical filters, and the Binance replay assumes a common base maintenance margin of 0.5%. A missing November 10 mark candle in both monthly and daily archives led to using October for development and December for holdout. A separate August–September block reruns the same four screens. There are 384 scenarios across 12 distinct weeks, including four leverage levels and two execution assumptions.

## Results

Only one distinct week met both targets, beginning September 11, 2023. The strong-signal screen at 50x finished at 1,000 USDT from 50. The minute replay confirms seven net winning trades out of seven at base costs and eight out of eight in stress, with intraperiod drawdowns of 42.0% and 42.5%. These are two execution assumptions applied to the same market week. The minute replay identifies the trades in gated-september-base.json and gated-september-stress.json.

The other eleven weeks lost money in that configuration. Across all twelve coarse weeks its base win rate was 68 of 189 trades, and the median ending balance was 1.73 USDT. The stress results were 52 of 171 and a median ending balance of 0.15 USDT. The strategy qualified as a historical occurrence in a parameter screen. The surrounding results provide no evidence of repeatability.

Run `python gated_momentum.py`, then `python gated_minute.py` to reproduce the Binance replays. Run `python gated_bybit.py` to compare the same historical week on Bybit public market data and current first-tier contract constraints.

## Bybit cross-check

For September 11–17, the same entry rule was recomputed on Bybit data with hours aligned to UTC and a 13-day warm-up. The comparison uses Bybit historical funding and mark candles plus current first-tier leverage and maintenance limits. The 15-minute replay ended at 34.88 USDT with 9 of 18 trades winning. The 1-minute replay, run with --minute, ended at 43.84 USDT with 9 of 17 winning and three modeled liquidations. Under stress the 1-minute replay ended at 1.15 USDT with 10 of 23 trades winning. Historical execution liquidity, tier changes, and tick rounding are not observed. No exchange orders were submitted.
