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

## New two-week historical candidate probe (exploratory)

The accompanying read-only module is
`tradingagents/crypto/backtest/weekly_candidates.py`. Its GitHub Actions
workflow is `.github/workflows/weekly-candidates.yml`.

It requests official Binance USD-M perpetual traded 1m OHLCV daily archives
for **BTCUSDT and XRPUSDT, November 1–14, 2026**, plus one prior warmup day.
This is **not** the Bybit data feed and is **not** an untouched holdout.
It replays three fixed signal families: EMA9/21+RSI momentum, Bollinger+RSI
reversion, and Donchian20 volume-confirmed breakout. All decisions use
completed bars; entries use the **next** 1m open. Stop/target collisions are
resolved stop-first. Every individual seven-day trial resets one shared
simulated account to 50 USDT, with a single position at a time and maximum
notional of 20% of equity (2x). Modeled taker commission is 5.5 bps per
side, with 3 bps additional adverse fill impact per side.

```bash
python tradingagents/crypto/backtest/weekly_candidates.py \
  --start 2026-11-01 --days 14 \
  --symbols BTCUSDT XRPUSDT \
  --output artifacts/weekly-candidate-probe.json
```

**Caveats:** tick size/lot step, minimum executable notional, genuine Bybit
slippage, partial IOC fills, funding and Bybit liquidation tiers are not
verified by public Binance candle ZIPs. Any reported trades are modeled,
not exchange-confirmed. The results must be labeled exploratory and
must **never** be promoted to Demo execution based on this study alone.
The strict 20×/70% portfolio gate is a separate evaluator; research
candidate snapshots here are available for subsequent auditing only after
its missing venue/funding provenance is resolved.

## Fee ablation for the fixed candle candidates

The research scan also reports all fixed candidates under three transparent
cost assumptions: zero fee/zero impact **upper bound (not executable)**,
realistic-estimate 5.5 bps commission plus 3 bps adverse price impact per
fill, and stress-estimate 8 bps plus 7 bps. All scenarios use the same
initial capital, source bars and predeclared signals. The lower-cost bound
is diagnostic: if a candidate cannot make money before costs, reducing
commission does not cure its directional signal.

## Independent 50-USDT trial input (preferred)

For the literal question "Can 50 USDT become 1,000 USDT in a week?",
the audit supports a top-level `weekly_runs` list. Each element contains
its own eight daily `account_equity` checkpoints and complete
`closed_trades` array. All four or more historical trials are *chronologically
consecutive*, **each independently resets to 50 USDT**, and all account
positions must be flat at each week's start and end. Trade IDs must be globally
unique across the trials.

This option prevents the mistaken requirement to compound one 50-USDT account
into 8 million USDT over four weeks: the intended claim is reproducibility of
a **50 -> 1,000 USDT weekly experiment**, not infinite capital scaling.
The separate original account-equity mode can evaluate a real continuously
compounding account, but requires appropriate actual portfolio evidence.
Neither mode asserts that a one-week historical success guarantees repetition.
