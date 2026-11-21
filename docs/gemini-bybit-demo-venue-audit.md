# Actual Bybit Demo closed P&L is separate from modeled shadow

The Gemini desk can record a **matched** Bybit Demo closed-PnL response as
an independent, read-only, append-only audit in the same SQLite database.
The venue close order ID is unique, so restarts or repeat reconciliation
do not count the same close twice.

- Source: authenticated Bybit **Demo**, not real-money execution.
- The venue field `closed_pnl` is already net of fees and funding as described
  in the Bybit closed-PnL calculation. We do **not** deduct fee fields again.
- The source order is exchange-reported, but linking it to one of our strategy
  entries uses the existing matching heuristic (side, entry price/time).
  It is therefore labeled `matched_exchange_record_heuristic_strategy_link`.
- Missing, unmatched, malformed, or non-finite records are excluded.
- The reported net USDT and profit factor are *not* a fabricated equity curve.
  Account ROI, drawdown, and risk limits must use the separately verified
  account balance/equity, not this close-only series.
- This data is NEVER silently mixed with Mark-model shadow observations or
  used to relax entry constraints, self-promote strategies, or activate Demo.
- No backfill is invented. Previous venue closes are not added automatically
  unless their close is reconciled by the active strategy after this code deploys.
- No live-money trading, Bybit orders, API keys, Railway settings, or worker
  start is performed by this research change.

Independent reference: https://bybit-exchange.github.io/docs/v5/position/close-pnl
