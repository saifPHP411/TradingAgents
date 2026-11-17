# Perpetual-only dashboard

The isolated research deployment supports a perpetual-only mode through:

\`\`\`text
TRADINGAGENTS_PERPETUAL_ONLY=1
\`\`\`

In this mode the dashboard does not create the legacy Spot/equity experiment, does not start the Binance Spot market-data engine, disables paper-worker controls, rejects simulation-advance requests, and returns no Spot positions. The existing production service is not changed by this setting.

## Market chart

The chart endpoint is authenticated and reads only Bybit V5 public linear USDT perpetual data:

\`\`\`text
GET /api/perpetual/chart?symbol=BTCUSDT
\`\`\`

Supported symbols are BTCUSDT, ETHUSDT, SOLUSDT and BNBUSDT. The chart uses completed 4-hour candles. Forming bars are not used. It renders candles with price and volume context, and overlays entry, stop, target and estimated liquidation levels only when a validated futures-position snapshot is available.

## Futures position snapshot contract

No authenticated Bybit Futures position worker is implemented by this research-only change. Consequently the dashboard reports \`position_status=unavailable\` unless a separate, trusted worker writes the snapshot below. It never infers a futures position from the Spot paper account.

File location (relative to the dashboard data root): \`perpetual_open_position.json\`.

Flat state:

\`\`\`json
{"state":"FLAT","symbol":"BTCUSDT"}
\`\`\`

Open state:

\`\`\`json
{
  "state": "OPEN",
  "symbol": "BTCUSDT",
  "side": "long",
  "quantity": 0.01,
  "leverage": 2.0,
  "entry_price": 100.0,
  "mark_price": 103.0,
  "stop_loss": 97.0,
  "take_profit": 109.0,
  "liquidation_price": 51.0,
  "unrealized_pnl": 0.03,
  "updated_at": "2026-01-01T12:00:00+00:00"
}
\`\`\`

The position data is validated for symbol, side, finite positive size, leverage and price fields before rendering. Missing or invalid snapshots remain unavailable; no placeholder position is created. The liquidation price is labelled as estimated and must come from authoritative Bybit position data before any live use.

## Safety boundary

This dashboard mode is research-only. Public chart retrieval does not authenticate to a Bybit account. It does not set leverage, read private balances, submit, amend or cancel orders, or start a worker. The existing perpetual simulator still has a research leverage cap of 2x and defaults to 1x. Passing tests does not establish an after-cost trading edge.
