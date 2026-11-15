# Bybit Demo Spot — Railway deployment and one-week paper run

This is the operator runbook for the current Railway configuration. The worker sends
orders only to Bybit's **Demo Spot** API. It does not use Bybit Live, Testnet,
Futures, Margin or leverage.

## What the deployment runs

- Dashboard: long-running phone-friendly web service.
- Decision source: `crypto`, the deterministic TJR rule chain over current Binance
  1m/5m crypto bars.
- Market-data snapshot: Binance public market data.
- Execution broker: `bybit_demo`, explicitly selected in both the Railway start
  command and environment.
- Execution endpoint: `https://api-demo.bybit.com`.
- Instruments: validated against Binance's data registry and Bybit Demo Spot's
  instrument registry before a worker can start.
- Position direction: long-only Spot.

The command explicitly passes `--paper-execution-provider bybit_demo`; the
environment also sets `TRADINGAGENTS_PAPER_EXECUTION_PROVIDER=bybit_demo`.
This is intentional defence in depth. A missing or invalid Bybit Demo credential
must stop preflight; it must never fall back to Alpaca.

## Required Railway service variables

Set these in Railway's service Variables panel. Do not commit values to GitHub.

| Variable | Required | Purpose |
|---|---:|---|
| `TRADINGAGENTS_DASHBOARD_TOKEN` | Yes | Strong bearer token protecting the dashboard and START/STOP actions |
| `BYBIT_DEMO_API_KEY` | Yes to trade | API key created inside Bybit Demo Trading |
| `BYBIT_DEMO_API_SECRET` | Yes to trade | Matching Demo API secret |

The repository config supplies the non-secret settings:

- `TRADINGAGENTS_PAPER_EXECUTION_PROVIDER=bybit_demo`
- `TRADINGAGENTS_PAPER_DECISION_SOURCE=crypto`
- `TRADINGAGENTS_PAPER_MARKET_DATA=binance_snapshot`
- `TRADINGAGENTS_PAPER_ORDER_NOTIONAL=25`
- `TRADINGAGENTS_PAPER_MAX_PRICE_DEVIATION=0.05`
- `TRADINGAGENTS_PAPER_SNAPSHOT_MAX_AGE_SECONDS=15`

Use **Demo Trading credentials only**. Do not paste credentials into chat, GitHub
issues, commits, logs or dashboard screenshots. The dashboard's read-only
`/api/paper/account-check` endpoint checks the selected Bybit Demo account without
starting the worker or submitting an order.

## Start the one-week run

1. Confirm the Bybit account is in **Demo Trading**, has demo USDT available and
   the API key is a Demo key with Spot trading permission.
2. Add the three required variables above in Railway. Redeploy after changing
   variables if Railway does not restart the service automatically.
3. Open the dashboard and confirm its banner says **BYBIT DEMO — NOT LIVE**,
   its execution provider is `bybit_demo`, and its endpoint is
   `https://api-demo.bybit.com`.
4. Check account connectivity and symbol validation. Do not press START while
   the account, endpoint, symbol registry, market data or protection status is
   unhealthy.
5. Press **START** once. The API requires the dashboard bearer token and the
   explicit paper confirmation header. START performs read-only preflight and
   starts the worker loop; it does not place an order immediately. The first
   cycle waits for the configured poll interval (60 seconds by default).
6. Leave the dashboard service running for at least seven days. Monitor open
   positions, verified stop/target protection, fills, fees, rejected orders,
   stale-data refusals and daily order counts.
7. At the end of the observation period, press **STOP** and export/review the
   trade journal and worker state before changing any strategy rule or risk cap.

The current risk policy remains the existing conservative paper policy:
$25 maximum target notional per order, up to five open positions, up to ten
orders per day, long-only, with mandatory stop-loss/take-profit protection.
Venue minimums and precision can cause an order to be refused or rounded down;
the worker must not force a trade to meet a target count.

## Important: STOP does not flatten the account

STOP prevents new worker cycles. It does **not** close open positions or cancel
orders already accepted by Bybit Demo. Existing venue-side protective orders
remain a broker-side responsibility. After STOP, inspect the Demo account and
make sure every remaining position is still protected; close it manually in the
Bybit Demo interface if you want the trial to end flat. Do not assume STOP means
"close everything".

A process restart returns the worker to STOPPED. The Railway volume preserves
its worker/broker state and idempotency records, but the worker must be started
again deliberately from the dashboard.

## What "learning" means in this first run

The configured `crypto` source is the TJR rules engine, not an online-learning
LLM. During the week the system records decisions, no-trade reasons, submitted
orders, venue fills, position protection and realised outcomes. Those records
make a forward paper sample that can be analysed after the run.

It does **not** rewrite its rules or increase risk automatically while trading.
That would let a noisy handful of trades change the strategy mid-test and make
the results impossible to audit. After at least a week, evaluate the logged
trades net of fees and slippage, group them by symbol and market regime, identify
which setups helped or hurt, then test any proposed rule changes on a separate
holdout before adopting them in a new version. A week with few trades is still
useful evidence; the bot must not manufacture trades just to create a larger
sample.

## Go/no-go checks

- [ ] Dashboard banner is BYBIT DEMO — NOT LIVE.
- [ ] Provider is `bybit_demo`; endpoint is `https://api-demo.bybit.com`.
- [ ] Demo account check succeeds and shows the expected account type.
- [ ] Every selected symbol is accepted by both venue registries.
- [ ] Market data is fresh; no stale or future-dated snapshot is being used.
- [ ] Open positions show verified protection.
- [ ] No live Bybit credentials or live endpoint are configured.
- [ ] START is pressed manually; no worker auto-starts on deployment/restart.

No profitability is guaranteed. This is a forward **Demo** observation run, not
evidence that the strategy will make money in a live account.
