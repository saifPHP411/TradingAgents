# Funding-rate carry research screen

This is an isolated, research-only study of long Spot + short USD-M perpetual carry.
It does not connect to the paper worker, dashboard order path, Railway, or exchange
order endpoints.

## Data and scope

The current CI runner can reach Binance public Spot and USD-M perpetual history,
but the Bybit public REST hosts returned HTTP 403 from the hosted runner. Therefore
the first historical study uses synchronized Binance Spot + USD-M perpetual data
as a **cross-venue proxy**, not as Bybit evidence. It must not be used to claim
Bybit execution readiness. A direct Bybit run requires a reachable public market
data source and separately measured account fee tier.

The bounded default is 180 days for BTCUSDT, ETHUSDT and SOLUSDT. The collector
fetches hourly Spot and perpetual candles plus realized funding events, preserving
actual timestamps and failing closed when a symbol has insufficient history.

## Causal simulation

At funding timestamp t, the already-settled funding rate at t is observable. That
lagged rate alone decides whether to enter for the *next* interval. The next
funding event's realized rate is credited only after the simulated position has
remained open through that settlement. Entry is approximated at the first hourly
open at or after t+1h; exit at the first hourly open at or after the next funding
timestamp +1h. No future funding observation is used to select the position.

The candidate is long Spot + short Perpetual with equal base quantity. PnL includes
the actual equal-quantity price-difference/basis movement, the next realized
funding payment, and four costed executions (Spot buy/sell and Perp sell/buy).
Realistic and conservative cost schedules are reported separately. Spot
buy-and-hold is shown as a context benchmark, not a risk-matched comparator.

## Known limitations / acceptance gates

- Binance is a proxy for Bybit; no Bybit fills, fees, spread, queue position or
  market impact are measured.
- Hourly opens approximate execution; the model does not simulate margin,
  liquidation, collateral haircuts, partial fills, borrow, or exchange failure.
- Only positive-funding long-Spot/short-Perp carry is tested. The reverse
  short-Spot/long-Perp trade is excluded because borrow availability and cost
  are not modeled.
- Funding rates can change; the prior settled rate is only a lagged predictor.
- No leverage or orders are used. This report is not a recommendation to trade.
- Do not tune against a window after inspecting its results. Use nested
  chronological validation and reserve genuinely future data for a locked holdout.

A positive result is only a candidate for further validation. It is not Demo
approval. The study must show positive net results under conservative costs,
sufficient event count, acceptable drawdown, stability by symbol and month,
and then be repeated with measured Bybit data/fees and a margin/liquidation model.
