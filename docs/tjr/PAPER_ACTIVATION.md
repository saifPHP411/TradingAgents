# TJR on Alpaca paper trading — activation

This note records how the TJR rule chain is wired into the Alpaca **paper** worker,
what it does and does not do, and how it is exercised offline. It is the Phase 2
counterpart to `VALIDATION_REPORT.md`, which measures the *strategy*; this document
describes the *execution path* that carries it.

Nothing here changes a TJR rule, a stop, a target or a risk limit. The strategy
validated in Phase 1 is the strategy this path runs.

## What was added

- `paper_experiment/tjr_source.py` — `TjrDecisionSource`, the third
  `DecisionSource`. It runs the deterministic TJR rule chain (no model) over
  current intraday bars and yields a `TradingDecision`.
- `paper_experiment/tjr_intraday_data.py` — the current 1m/5m decision-data seam
  (`AlpacaIntradayBars`, `StaticIntradayBars`), separate from the daily
  current-market seam and from the execution-price seam.
- `DECISION_SOURCE_TJR = "tjr"` and `NO_TRADE_TJR_NO_NEW_DATA` in
  `broker/trading.py`, added to the closed vocabularies.
- Controller dispatch (`dashboard/paper_control.py`), CLI choice
  (`--paper-decision-source tjr`) and the operator-facing description in
  `market_data_wiring.py`.

## How it differs from the other two sources

| source | strategy | data | cadence | consumption |
|---|---|---|---|---|
| `history` | recorded LLM ratings | the experiment | the experiment's sessions | processed set (refusals included) |
| `current_market` | the LLM inference seam | current **daily** Alpaca bars | one decision per symbol per session | order-only |
| `tjr` | the **TJR rule chain** | current **1m/5m** Alpaca bars | one decision per symbol per session | order-only |

The three are deliberately separate modules. TJR is never routed through the LLM,
and the current-market source is never used as the TJR source.

## Safety properties, and where each is enforced

1. **Closed bars only.** `tjr_intraday_data` drops every bar that had not closed
   at the decision instant; the closed-bar rule itself lives in `tjr/frames.py`.
   A bar that is still forming cannot enter a decision.
2. **No stale or future data.** A newest bar older than the freshness limit, or
   dated after the decision session, is refused. The window is never shortened
   into a frame that looks complete.
3. **Market-data host only.** The seam talks to `data.alpaca.markets` on the `iex`
   feed; the paper/order host and every live host are refused before any request.
4. **RiskEngine gate.** A completed setup becomes a `TradingSignal` and is put
   through the project's own `RiskEngine` under `paper_parity_limits()` — the same
   caps the backtest and the paper account use (25 USD order notional, 5 open
   positions, 10 daily orders, long-only, fractional shares). Only an approved
   signal becomes a decision, carrying the engine's sized quantity.
5. **Real book, no fabrication.** The engine sizes against a read-only mirror of
   the worker's broker account and positions. With no book it is handed an empty
   book and refuses to size. The source invents no size, stop or target.
6. **One decision per symbol per session.** Id `tjr-<symbol>-<session>`,
   deterministic, so a restart within a session replays the same id without a
   second evaluation. Consumption is order-only: a session decision that produced
   no order is re-offered, not mistaken for consumed.
7. **Own state file.** The source persists to `tjr_source_state.json` under the
   experiment root, never the experiment's `session_state.json`/`journal.json`.
8. **Paper-only.** The path is opt-in, uses only the paper host and the market-data
   host, and no code here reaches a live endpoint or places an order itself.

## Honest no-trades

Every no-trade carries a code from the worker's closed vocabulary:

| situation | code |
|---|---|
| market closed / unknown clock | `MARKET_CLOSED` / `MARKET_DATA_UNAVAILABLE` |
| unreachable, stale, future or malformed window | `MARKET_DATA_UNAVAILABLE` |
| chain did not complete | `NO_SIGNAL` |
| RiskEngine refused the entry | `RISK_REJECTED` |
| every symbol already decided this session | `TJR_NO_NEW_DATA` |

The rule chain's own reason code (`NO_1M_ENTRY` and friends) is kept in the record
detail for the audit trail; it is never what the worker records as its code.

## How it is exercised

`tests/test_phase30_tjr_paper_source.py` drives the whole chain offline — the real
`TjrDecisionSource`, the real `PaperTradingWorker`, the real risk checks — with the
market clock, the intraday fetch and the broker transport injected. No test reaches
the network and none needs an Alpaca key. The suite pins, among others: the chain is
the strategy (no model), closed bars only, deterministic ids, restart replay, the
RiskEngine gate under the parity caps, honest no-trades, state isolation, and the
full source → worker → risk → broker path.

## How to run it (paper only)

```bash
export ALPACA_PAPER_API_KEY=...      # paper credentials, not live
export ALPACA_PAPER_API_SECRET=...
python -m tradingagents.paper_experiment paper-trade \
  --paper-decision-source tjr \
  --paper-market-data alpaca_snapshot \
  --dry-run
```

`--dry-run` performs preflight and reports what would happen without placing an
order. Dropping it runs one cycle and stops. This environment had no Alpaca
credentials, so no live paper cycle was run here; the path is proven offline by the
test suite, and the first real run is a deliberate operator action.

## Limitations

- **No live run here.** The activation is validated offline. A first real paper
  cycle is an operator action; this note is not a claim that a real cycle ran.
- **Data window.** Alpaca's Basic plan serves a limited intraday history; the
  decision window is a few recent sessions, and the vendor's cap — not the study —
  sets it.
- **Paper only.** The paper account is a simulation. A paper result is not a
  live-trading result, and neither is evidence of future profitability.
- **Costs.** The worker's fills carry the project's frozen cost model; a real
  broker's spread and slippage may differ, and the Phase 1 study shows the result
  is sensitive to friction.
- **No edge claim.** This path carries the unchanged TJR rule chain. Phase 1's
  historical read was negative net of the project's costs; this activation does not
  change that and makes no claim that the strategy is profitable.
