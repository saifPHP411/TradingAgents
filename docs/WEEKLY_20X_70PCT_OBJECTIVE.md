# 7-day 20× / 70% objective — independent evidence gate

**User's target:** grow one 50 USDT account to at least 1,000 USDT in
seven days (+1,900%, 20×), with at least 70% net-profitable closed trades.

This is a **research objective, not a forecast or guaranteed trading
outcome**. The required geometric average is about +53.4% per day for seven
days. A high historical win rate alone does not imply a positive expectancy.

## Implementation

The standalone research-only tool:

- tradingagents/crypto/backtest/weekly_goal.py
- tests/test_weekly_goal.py

Example:

```bash
python -m tradingagents.crypto.backtest.weekly_goal \
  --input results/single_account_replay.json \
  --output results/weekly_goal_audit.json
```

Input format (illustrative structure, **not** observed market data):

```json
{
  "initial_equity_usdt": 50,
  "source": "historical_replay",
  "data_quality": {
    "independent_holdout": false,
    "fees_included": true,
    "slippage_included": true,
    "funding_included": true,
    "liquidations_included": true,
    "unrealized_pnl_included": true,
    "no_external_cash_flows": true,
    "venue_filters_enforced": true,
    "position_risk_limits_enforced": true,
    "no_lookahead": true,
    "order_fills_verified_or_conservatively_modeled": true
  },
  "account_equity": [
    {"timestamp": "2026-10-01T00:00:00Z", "equity_usdt": 50,
     "open_positions_count": 0}
  ],
  "closed_trades": [
    {"trade_id": "example-001", "closed_at": "2026-10-01T09:00:00Z",
     "net_pnl_usdt": 0.04, "liquidated": false}
  ]
}
```

This abbreviated JSON intentionally fails validation. A real report needs
**eight consecutive UTC-midnight account-equity checkpoints per week** and
the complete exchange/replay trading ledger. A qualifying week must start and
end flat (zero open positions) and its trade-level net P&L must reconcile
to the change in account equity. Account equity includes unrealized P&L.
Never manufacture zeros or fill missing days by interpolation.

The tool does not multiply independent symbol backtests together: all capital
must be tracked in a **single chronological account**, respecting open
positions, margin and competing trades. It reports **every nonoverlapping
7-day week** rather than selecting the best-looking one. Its evidence gate
requires:

- Weekly ending/start equity multiple >=20.
- Closed net-of-cost trade win rate >=70%.
- At least 30 closed trades per week.
- No liquidations in any qualifying week.
- Reconciliation of closed P&L against 7-day account equity delta and no open
  positions at the beginning/end of the week.
- At least four complete independent seven-day weeks, **all** satisfying
  those criteria. This is an extra robustness gate, not a guarantee of
  future performance.
- Explicit provenance flags covering untouched holdout, fees, slippage,
  funding, liquidations, unrealized account equity, no deposits, venue filters,
  risk limits, no lookahead and order-fill modeling.

Even a passing historical score is explicitly
`reported_historical_goal_passed_not_forward_verified`, not a live-trading
authorization. Flags are self-reported by the producer, so the report uses
reported_evidence_gate_passed and keeps independently_verified_by_this_tool
false: an independent review of raw fills and equity is still required. Daily drawdown samples
are not evidence of maximum *intraday* drawdown.

## Research track — strategy independent

Market type is **not fixed** by this objective. Spot and USDT perpetual
long/short variants can be investigated separately, provided each strategy
is implemented with its real venue fees, lot-size minimums, order rejections,
mark/last prices, funding and liquidation logic. Bybit Demo's existing
2× ceiling and 20% account-notional cap stay unchanged unless separately
reviewed with explicit risk permission; these limits make a 20× weekly
target especially unlikely, but they must not be bypassed simply to force
a historical win.

Experiments to compare without hindsight leakage:

1. Current 1m/5m/15m scalper with Market IOC and 10 bps Limit IOC.
2. Trend/momentum breakout with explicit regime and liquidity conditions.
3. Range/mean-reversion with strict independent OOS validation.
4. Spot or hedged funding variants as benchmarks; note that low-risk
   arbitrage cannot reasonably be expected to earn 1,900% weekly.

Always log *every trial*, including losses and skipped signals. Use
development, validation and untouched holdout intervals; avoid tuning
against the holdout. Compare complete portfolio equity curves over
consecutive weeks, not pooled trade win rate or the best historical week.

**Current status:** no end-to-end trade/equity dataset in this branch has
demonstrated the 20×/70% goal. The tool is a rigorous screen only.
Trading worker remains disabled; no PR merge or Railway deployment.
