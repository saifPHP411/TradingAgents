# Perpetual-only research dashboard

## Research scope

The isolated research deployment can be run in perpetual-only mode with:

    TRADINGAGENTS_PERPETUAL_ONLY=1

In that mode the server does not create or refresh the legacy Spot/equity experiment, does not start the Spot market-data engine, disables the legacy paper-worker and simulation controls, and rejects the legacy Spot API routes. This setting is intended only for the isolated research service; it must not be applied to the original production service.

## Chart and actual Demo position

The dashboard fetches completed 4-hour candles from the public Bybit V5 linear USDT-perpetual market-data endpoints for BTCUSDT, ETHUSDT, SOLUSDT and BNBUSDT.

When both BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET are configured, an isolated read-only client signs only GET /v5/position/list against the fixed api-demo.bybit.com host. It does not expose methods to place, amend, cancel orders or change leverage. It rejects multiple open legs and hedge-mode positions. Missing credentials produce an unavailable state, not a false flat position. A failed signed request produces an error state and does not fall back to a potentially stale local position.

The chart overlays entry, stop loss, take profit and liquidation price only when a genuine Demo position or an explicitly validated local futures snapshot is available. The liquidation figure is supplied by the exchange position response; it is not the simulator's simplified liquidation estimate. A missing stop, target or liquidation price is omitted instead of being fabricated.

## Strategy tournament

The development-only tournament compares five fixed configurations:

- EMA 14/50 with ADX 18
- EMA 9/30 with ADX 22
- EMA 20/60 with ADX 16
- Donchian 20 breakout with ADX 18
- Donchian 55 breakout with ADX 18

Each candidate is evaluated at 1x and 2x under realistic, conservative and stress cost assumptions. It uses disjoint chronological development folds and excludes all candles on or after the final_holdout_start timestamp from the previously inspected EMA report. It does not evaluate or overwrite that consumed holdout. Repeated tournament execution on the same development sample is blocked to reduce selection bias.

A development candidate is shortlisted only if every cost scenario passes its minimum trade count, Profit Factor, compounded return, positive-fold ratio, positive-symbol breadth, drawdown and zero-liquidation checks.

The original EMA14/EMA50 + EMA200 + ADX18 candidate is explicitly disqualified from reselection: it is the same fixed specification evaluated in the earlier independent final holdout, where both 1x and 2x failed the realistic, conservative and stress acceptance gates. Its favourable result in the later development-only tournament is reported for comparison, but cannot override that prior out-of-sample failure.

After applying that exclusion, none of the remaining tested candidates passed all three cost scenarios and consistency gates. Therefore no strategy is selected or approved for Demo.

## Limits and safety

This is research, not proof of a future trading edge. Cost assumptions are estimates, not measured account-specific fills. The strategy simulator's research leverage cap remains 2x and defaults to 1x. The Demo reader is GET-only; the research dashboard does not submit orders, change account leverage, start a worker or approve a strategy for trading.
