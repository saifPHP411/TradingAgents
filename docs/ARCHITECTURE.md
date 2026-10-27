# Architecture

TradingAgents is two systems that meet at one interface. One is the analytical
brain: a multi-agent LLM workflow that reads data and argues its way to a view.
The other is a deterministic simulated trading system that turns a view into a
position and keeps honest books about it. The interface between them is a
validated trading signal.

The split exists because the two have different failure modes. A language model
is good at reading a filing and bad at arithmetic under constraints. A ledger is
exact and has no opinion. Keeping them apart means a bad model response produces
a rejected signal, not a corrupt book.

```
                 ┌──────────────────────┐
                 │     Market Data      │
                 │  dataflows/ vendors/ │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   TradingAgents AI   │
                 │  graph/ agents/      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Signal Normalization │
                 │       signals/       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │    Risk Engine       │
                 │        risk/         │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   Order Engine       │
                 │     execution/       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Execution Simulator  │
                 │  execution/simulator │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Portfolio / Ledger   │
                 │      portfolio/      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Performance Engine   │
                 │     performance/     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │       Reports        │
                 │    trading_report/   │
                 └──────────────────────┘
```

## The analytical brain

The existing multi-agent workflow is unchanged. It lives in `graph/` and
`agents/`, and its roles are:

| Role | Module |
|---|---|
| Fundamental analyst | `agents/analysts/fundamentals_analyst.py` |
| Sentiment analyst | `agents/analysts/sentiment_analyst.py` |
| News analyst | `agents/analysts/news_analyst.py` |
| Technical analyst | `agents/analysts/market_analyst.py` |
| Bull researcher | `agents/researchers/bull_researcher.py` |
| Bear researcher | `agents/researchers/bear_researcher.py` |
| Trader | `agents/trader/trader.py` |
| Risk analysts | `agents/risk_mgmt/` |
| Portfolio manager | `agents/managers/portfolio_manager.py` |

They debate, and the portfolio manager writes a final view. That view is the
model's opinion, and it is where the deterministic system starts.

Providers are abstracted in `llm_clients/`, so the workflow runs against any
supported provider and is not tied to one model. See the provider section of the
README.

## The signal contract

`signals/contract.py` defines `TradingSignal`. A model's view becomes a signal
only if it validates:

- `action` is one of `BUY`, `HOLD`, `SELL`. An arbitrary action is refused.
- `confidence` is in `[0, 1]`.
- `as_of_date` is a real date, and the timestamp is timezone-aware.
- A `HOLD` must not carry a `position_size`.
- A `BUY` must carry a size; a `SELL` must not.

The contract also fixes the signal's identity: `signal_id` is derived from the
symbol, action, date and a content hash, so two runs of the same decision produce
the same id. That is what makes a backtest reproducible.

The reasoning field is carried through to the report verbatim. It is never
parsed for instructions and never reaches the execution path.

## Risk, execution, portfolio

`risk/engine.py` takes a signal and a ledger and returns a `RiskDecision`: the
verdict, the size, and a named reason. It is the only thing that decides whether a
trade happens. The limits are configuration (`RiskLimits`): position size,
portfolio exposure, open-position count, daily loss, drawdown, and the order-value
band. Long-only is enforced deterministically — a sell larger than the holding is
refused as a short.

`execution/orders.py` defines the order and its lifecycle. `execution/costs.py`
holds the cost schedule: commission, maker and taker fees, slippage and spread.
`execution/simulator.py` turns an order into a fill or a rejection against a
supplied `MarketContext`, and applies the fill to the ledger. `execution/paper.py`
drives the same chain from live quotes instead of historical bars.

`portfolio/ledger.py` is the book. Every cash or position change goes through
`apply_fill`, so the accounting rules live in one place, and every change is
recorded as an event. Cash, positions, average entry, market value, equity,
realised and unrealised P&L, fees and exposure are all derived from the same
records, which is why the report cannot disagree with the book.

## The two drivers

The same chain is driven two ways:

- **Backtest** — `backtesting/engine.py`. A `MarketDataFeed` supplies bars,
  sliced strictly at each session, so a strategy cannot see the future. Orders
  are scheduled for the next session's open by default.
- **Paper** — `execution/paper.py`. A caller-supplied quote source supplies
  prices and the clock is the wall clock.

Both use the same ledger, risk engine and simulator. A paper result and a
backtest result are therefore comparable, because the accounting is identical by
construction.

## Safety

There is no live trading path, and there is no configuration that produces one.

- `ExecutionMode` has exactly two members, `BACKTEST` and `PAPER`. There is no
  `LIVE` to select.
- `resolve_mode` refuses any string naming a live mode, by name.
- `ExecutionSimulator.__init__` calls `assert_simulation_only`.
- The simulator takes prices as arguments rather than fetching them, so there is
  nothing for a live implementation to hook into without a new class being
  written deliberately.

This is covered by tests in `tests/test_paper_trading.py` and
`tests/test_execution_simulator.py`.

## Where to look next

- `docs/BACKTESTING.md` — the event loop, timing and bias controls.
- `docs/RISK.md` — the limits and how they are enforced.
- `docs/EXECUTION.md` — orders, fills and costs.
- `docs/PAPER_TRADING.md` — running against live quotes.
- `docs/TRADING_SYSTEM_AUDIT.md` — what was there before, and what was missing.
