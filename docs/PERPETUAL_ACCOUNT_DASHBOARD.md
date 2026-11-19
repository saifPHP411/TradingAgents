# Perpetual Demo Account Dashboard

## Scope and safety

The isolated research service runs in linear USDT perpetual-only mode. The account panel is a monitoring surface, not an execution surface. Its exchange client is fixed to \`https://api-demo.bybit.com\` and implements signed GET reads only:

- \`GET /v5/account/wallet-balance\` for Unified account equity, wallet balance, margin balance, available balance, and perpetual unrealized P&L.
- \`GET /v5/position/closed-pnl\` for recent closed linear-perpetual P&L and the venue-provided open/close fee fields.
- \`GET /v5/position/list\` for the selected symbol's open-position chart overlay.

There are no order-create, amend, cancel, set-leverage, margin-mode, or transfer calls in this account reader. A successful account read does not authorize or start trading.

## Capital baseline and return calculation

Set \`TRADINGAGENTS_INITIAL_CAPITAL_USDT\` on the isolated Railway research service to the actual starting capital if it is known. The dashboard then calculates:

- Net P&L since baseline = current Bybit \`totalEquity\` − configured initial capital.
- Return since baseline = net P&L ÷ configured initial capital × 100.
- Margin utilization indicator = Bybit \`totalInitialMargin\` ÷ \`totalEquity\` × 100, when both values are available.

If the variable is not configured, the service records the first successfully verified account equity as a provisional baseline in its persistent data volume. This is explicitly labelled as a first-observed baseline, not a claim about the account's original deposit.

Deposits and withdrawals are not currently reconciled against the baseline. If external capital flows occur, the displayed return is not a flow-adjusted investment return. Configure the actual original capital and reconcile those flows before relying on the percentage as a performance statistic. A zero first-observed equity cannot become a denominator; the dashboard asks for an explicit positive baseline instead.

## Charts and history

- Account equity snapshots are persisted at most once per minute, capped at 1,000 points. History begins when the dashboard first reads the account successfully; it is not historical exchange equity data.
- Closed-PnL records are requested from Bybit's default recent seven-day window, with a maximum of 100 records per refresh. The cumulative chart is cumulative over the returned sample, not all-time realized P&L.
- Closed-trade rows show symbol, side, quantity, entry/exit average prices, closed P&L, and open/close fees where returned.
- If the wallet endpoint succeeds but closed-PnL history is restricted or unavailable, the wallet metrics remain visible and the realized-PnL panel is marked unavailable.
- If wallet authentication or validation fails, balances are hidden; the dashboard never substitutes zero or a cached balance.

## Required Bybit API permissions

Use a dedicated Bybit Demo API key with read-only permissions sufficient for Account/Wallet and Contract/Position reads. Do not enable Trade, Order, Transfer, Withdraw, or leverage/margin mutation permissions for this dashboard reader. The key and secret must be configured only as Railway service variables named \`BYBIT_DEMO_API_KEY\` and \`BYBIT_DEMO_API_SECRET\`; never put them in source code, issue comments, logs, or chat messages.

## UI

The isolated Perpetual Lab uses a responsive burgundy palette. It includes the account/equity summary, available margin, perpetual unrealized P&L, baseline-relative return, tracked equity history, recent realized P&L curve, closed-trade ledger, and the existing candlestick chart with entry, mark, stop, target, and liquidation overlays. Profit/loss colors remain semantically distinct for readability.
