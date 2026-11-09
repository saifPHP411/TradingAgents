"""The paper execution chain, end to end, with a mocked broker transport.

The dashboard complaint this suite answers: a worker can be ``RUNNING`` and an
Alpaca account ``ACTIVE`` while nothing is analysed, decided, ordered or filled,
and a single "RUNNING" badge hides that. So these tests walk the *whole* chain
against real worker code and a mocked HTTP transport —

    market data -> strategy evaluation -> decision -> risk -> order ->
    broker acknowledgement -> fill -> position -> exit decision -> closed trade

— and assert on the observable record each stage leaves.

No test reaches the network (the suite's autouse guard refuses it) and no test
needs an Alpaca key: the transport is injected, as the phase-15 loop tests do.

The claims pinned here:

- a decision the experiment recorded reaches the worker and becomes an order;
- the price the order was sized from is recorded, with the seam that produced it;
- a decision already processed is not ordered twice;
- a local risk refusal and a broker rejection are named apart;
- a fill is recorded and a position is read back;
- a recorded stop/target produces exactly one exit order, and the exit is not
  repeated;
- the engine view reports *why* nothing traded as a stable code, never a bare
  "no decision";
- the engine view never reports the market data as unavailable while the worker
  is pricing orders from it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    NO_TRADE_ALREADY_PROCESSED,
    NO_TRADE_HISTORY_EXHAUSTED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_ORDER_EXISTS,
    NO_TRADE_REASONS,
    NO_TRADE_SIZE_TOO_SMALL,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    SourceMarketData,
    StaticMarketData,
)
from tradingagents.paper_experiment.dashboard.engine_view import (
    engine_view,
    positions_view,
    trades_view,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.paper_trading import ExperimentDecisionSource
from tradingagents.paper_experiment.sources import StaticPriceSource

SYMBOL = "SPY"

# An account rich enough that a 25 USD order is never refused for buying power.
_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "5000.00",
    "equity": "5000.00",
    "buying_power": "5000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


def _frame(periods: int = 12, *, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-10-01", periods=periods, freq="B")
    opens = [start + i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1e6] * periods,
        },
        index=index,
    )


def _order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": SYMBOL,
        "side": "buy",
        "qty": "2",
        "filled_qty": "0",
        "filled_avg_price": None,
        "status": "accepted",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _position(**over):
    payload = {
        "symbol": SYMBOL,
        "qty": "2",
        "side": "long",
        "avg_entry_price": "100.0",
        "market_value": "220.0",
        "cost_basis": "200.0",
        "unrealized_pl": "20.0",
    }
    payload.update(over)
    return payload


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _transport(routes, *, calls=None):
    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _routes(*, orders=None, positions=None, by_client_order_id=None):
    return {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, positions if positions is not None else []),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id or _Resp(200, _order()),
    }


def _broker_factory(routes, *, calls=None, limits=None):
    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes, calls=calls),
            limits=limits or PaperRiskLimits(),
        )

    return build


def _experiment(root, *, periods=12, processed=1, rating="Buy", lag=2):
    """An experiment advanced ``processed`` sessions, with a StubInference decision.

    The stub proposes the levels the mandatory-protection gate requires: a stop
    below the entry and a target above it. The last session's bar opens around
    111.0, so the levels are chosen around that.
    """
    source = StaticPriceSource({SYMBOL: _frame(periods)}, name="static_test")
    config = ExperimentConfig.from_protocol(
        symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=lag
    )
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=root,
        inference=StubInference(
            rating=rating,
            stop_loss=100.0,
            take_profit=130.0,
        ),
    )
    experiment.ensure_created()
    for session in list(source.sessions())[:processed]:
        experiment.process_session(session)
    return experiment, source


def _worker(
    root,
    experiment,
    source,
    *,
    routes=None,
    whole_shares=False,
    market_data=None,
    limits=None,
    broker_limits=None,
    order_notional=25.0,
):
    """A worker wired exactly as the dashboard controller wires one."""
    config = PaperTradingConfig(
        strategy_id=experiment.config.strategy_id,
        order_notional=order_notional,
        whole_shares=whole_shares,
        state_path=root / "worker_state.json",
        market_data_provider="bar_open",
        limits=limits or PaperRiskLimits(),
    )
    market = market_data if market_data is not None else SourceMarketData(source, source.sessions()[-1])
    source_obj = ExperimentDecisionSource(
        experiment,
        order_notional=order_notional,
        reload=lambda: PaperExperiment.load(root, inference=None),
    )
    worker = PaperTradingWorker(
        config=config,
        decision_source=source_obj,
        market_data=market,
        broker_factory=_broker_factory(routes or _routes(), limits=broker_limits),
    )
    return worker, source_obj


@pytest.mark.unit
class TestTheChainReachesAnOrder:
    """market data -> strategy -> decision -> risk -> order -> fill -> position."""

    def test_a_recorded_decision_becomes_an_order_with_a_fill(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(
            root,
            experiment,
            source,
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(qty="0.25", filled_qty="0.25", filled_avg_price="111.0", status="filled"),
                )
            ),
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        # The chain reached a real order, sized from the seam, and recorded it.
        assert result.ran is True
        assert result.reason_code is None
        assert result.reference_price == pytest.approx(111.0)
        assert result.market_data_provider == "bar_open"
        assert result.market_data_kind == "historical"
        assert result.order is not None
        assert result.order.filled_quantity == pytest.approx(0.25)
        assert result.order.status == "filled"

    def test_the_order_is_persisted_and_survives_a_restart(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(
            root,
            experiment,
            source,
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(qty="0.25", filled_qty="0.25", filled_avg_price="111.0", status="filled"),
                )
            ),
        )
        worker.start()
        worker.run_once()
        client_order_id = next(iter(worker.state.orders))
        worker.stop()

        reloaded = PaperTradingWorker(
            config=worker.config,
            decision_source=worker.decision_source,
            market_data=worker.market_data,
            broker_factory=_broker_factory(_routes()),
        )
        assert client_order_id in reloaded.state.orders
        # A restart never resumes RUNNING on its own.
        assert reloaded.state.status == "STOPPED"

    def test_the_same_decision_is_not_ordered_twice(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=2, lag=1)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        first = worker.run_once()
        # Re-arm the source so the *same* decision is offered again: the worker's
        # processed-decision memory, not the source's cursor, must stop the re-order.
        worker.decision_source.reset()
        second = worker.run_once()
        worker.stop()

        assert first.ran is True
        assert second.ran is False
        # The second cycle saw the same decision and refused to re-order it.
        assert second.reason_code in (NO_TRADE_ALREADY_PROCESSED, NO_TRADE_ORDER_EXISTS)
        assert len(worker.state.orders) == 1


@pytest.mark.unit
class TestNoTradeReasonsAreNamed:
    """Every no-order cycle carries a code from the closed vocabulary."""

    def test_an_exhausted_history_is_named_not_generic(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=12)
        worker, decision_source = _worker(root, experiment, source)
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_HISTORY_EXHAUSTED
        assert "every available session" in result.skipped_reason
        # The source exposed the same code, so the view can match on it.
        assert decision_source.skip_code == NO_TRADE_HISTORY_EXHAUSTED

    def test_an_unpriceable_symbol_is_named_market_data(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        # A price map with no entry for the decision's symbol.
        worker, _ = _worker(
            root, experiment, source, market_data=StaticMarketData({"OTHER": 10.0})
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_a_too_small_whole_share_order_is_named_size(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        # 25 USD at ~111 USD/share floors to zero whole shares.
        worker, _ = _worker(root, experiment, source, whole_shares=True)
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_SIZE_TOO_SMALL
        assert "fractional" in result.error

    def test_every_code_is_in_the_closed_vocabulary(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        codes = set()
        for _ in range(2):
            result = worker.run_once()
            if result.reason_code is not None:
                codes.add(result.reason_code)
        worker.stop()
        assert codes <= set(NO_TRADE_REASONS)


@pytest.mark.unit
class TestRiskAndRejectionAreDistinct:
    def test_a_local_risk_refusal_is_recorded(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        # A daily-order cap of 1, pre-consumed for *today*, so the local risk check
        # (not the broker) refuses the order. The config-time notional check is
        # satisfied, so the refusal genuinely comes from the risk engine.
        from datetime import UTC, datetime

        broker_limits = PaperRiskLimits(max_daily_orders=1)
        worker, _ = _worker(root, experiment, source, broker_limits=broker_limits)
        worker.start()
        worker._broker._state.daily_orders_date = datetime.now(UTC).date().isoformat()
        worker._broker._state.daily_orders_count = 1
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        # A local refusal is named apart from a broker rejection.
        assert result.reason_code == "RISK_REJECTED"
        assert isinstance(result.error, str) and result.error

    def test_a_broker_rejection_is_named_broker(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        routes = _routes(orders=_Resp(422, {"message": "insufficient qty available"}))
        worker, _ = _worker(root, experiment, source, routes=routes)
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == "BROKER_REJECTED"


@pytest.mark.unit
class TestExitPolicyProducesAClosedTrade:
    def test_a_recorded_stop_produces_exactly_one_exit(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=12)
        # The stop sits above the price the seam returns, so it has triggered.
        stop = 200.0
        worker, _ = _worker(
            root,
            experiment,
            source,
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(
                        client_order_id="ptx-exit",
                        side="sell",
                        qty="1",
                        filled_qty="1",
                        filled_avg_price="111.0",
                        status="filled",
                    ),
                ),
                positions=[_position(qty="1", avg_entry_price="100.0", market_value="220.0")],
            ),
            # A notional cap wide enough to close the seeded 1-share position.
            broker_limits=PaperRiskLimits(max_order_notional=1000.0),
            limits=PaperRiskLimits(max_order_notional=1000.0),
            order_notional=100.0,
        )
        worker.start()
        # Seed the worker as if it had already opened a long with a recorded stop.
        worker.state.stops[SYMBOL] = {"symbol": SYMBOL, "stop_loss": stop, "take_profit": None}
        worker.state.positions = {SYMBOL: {"symbol": SYMBOL, "quantity": 1.0}}
        worker._save()

        first = worker.run_once()
        worker.stop()

        # The stop triggered once, an exit order was placed, and the position is
        # marked exited so it is never exited again.
        assert first.ran is True
        assert first.order is not None
        assert first.order.side == "sell"
        assert first.reference_price == pytest.approx(111.0)
        assert worker.state.exits[SYMBOL]["reason"] == "stop_loss"

    def test_an_exit_is_not_repeated(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=12)
        worker, _ = _worker(
            root,
            experiment,
            source,
            # The broker must report the held position: the worker's position view is
            # the broker's own, refreshed before the exit check, not a local guess.
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(
                        client_order_id="ptx-exit",
                        side="sell",
                        qty="1",
                        filled_qty="1",
                        filled_avg_price="111.0",
                        status="filled",
                    ),
                ),
                positions=[_position(qty="1", avg_entry_price="100.0", market_value="220.0")],
            ),
            broker_limits=PaperRiskLimits(max_order_notional=1000.0),
            limits=PaperRiskLimits(max_order_notional=1000.0),
            order_notional=100.0,
        )
        worker.start()
        worker.state.stops[SYMBOL] = {"symbol": SYMBOL, "stop_loss": 200.0, "take_profit": None}
        first = worker.run_once()
        # A second cycle must not place a second exit for the same symbol.
        second = worker.run_once()
        worker.stop()

        assert first.ran is True
        assert SYMBOL in worker.state.exits
        exit_orders = [oid for oid in worker.state.orders if oid.startswith("ptx-")]
        assert len(exit_orders) == 1
        assert second.client_order_id is None

    def test_no_recorded_level_fabricates_no_exit(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=12)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        # A held position with no recorded stop/target: nothing to exit, so no
        # order is fabricated for it.
        worker.state.positions = {SYMBOL: {"symbol": SYMBOL, "quantity": 1.0}}
        result = worker.run_once()
        worker.stop()
        assert result.ran is False
        assert worker.state.exits == {}


@pytest.mark.unit
class TestEngineViewIsHonest:
    def test_market_data_is_not_reported_down_while_an_order_prices(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(
            root,
            experiment,
            source,
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(qty="0.25", filled_qty="0.25", filled_avg_price="111.0", status="filled"),
                )
            ),
        )
        worker.start()
        worker.run_once()
        worker.stop()

        view = engine_view(worker.status())
        # The historical seam reports its own status, so availability is true...
        assert view["states"]["market_data_available"] is True
        assert view["market_data"]["status"] == "ok"
        assert view["states"]["order_filled"] is True

    def test_the_blocker_names_the_history_gate(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=12)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        result = worker.run_once()
        # Read the view while the worker is still RUNNING: a stopped worker is off,
        # not blocked, so its blocker is intentionally None.
        view = engine_view(worker.status())
        worker.stop()

        assert result.reason_code == NO_TRADE_HISTORY_EXHAUSTED
        assert view["blocker"]["code"] == NO_TRADE_HISTORY_EXHAUSTED
        assert view["blocker"]["operator_action_required"] is True

    def test_a_stopped_worker_has_no_blocker(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(root, experiment, source)
        # Never started: the worker is off, which is not "blocked".
        view = engine_view(worker.status())
        assert view["blocker"] is None
        assert view["states"]["worker_running"] is False

    def test_a_filled_order_is_not_a_terminal_without_fill(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(
            root,
            experiment,
            source,
            routes=_routes(
                orders=_Resp(
                    200,
                    _order(qty="0.25", filled_qty="0.25", filled_avg_price="111.0", status="filled"),
                )
            ),
        )
        worker.start()
        worker.run_once()
        worker.stop()

        view = engine_view(worker.status())
        assert view["terminal_without_fill"] == []
        assert view["counts"]["filled"] == 1

    def test_positions_view_reconciles_broker_against_internal(self, tmp_path):
        status = {
            "positions": {
                SYMBOL: {
                    "symbol": SYMBOL,
                    "quantity": 2.0,
                    "average_entry_price": 100.0,
                    "market_value": 220.0,
                    "opened_at": "2026-10-29T13:30:00Z",
                }
            },
            "stops": {SYMBOL: {"stop_loss": 90.0, "take_profit": 130.0}},
            "exits": {},
        }
        # The internal ledger agrees, so the row is reconciled.
        rows = positions_view(status, {SYMBOL: {"quantity": 2.0}})
        assert len(rows) == 1
        assert rows[0]["reconciled"] is True
        assert rows[0]["stop_loss"] == 90.0

        # The internal ledger disagrees, and the mismatch is shown, not hidden.
        mismatched = positions_view(status, {SYMBOL: {"quantity": 3.0}})
        assert mismatched[0]["reconciled"] is False
        assert mismatched[0]["internal_quantity"] == 3.0
        assert mismatched[0]["broker_quantity"] == 2.0

    def test_trades_view_does_not_call_an_unfilled_order_a_trade(self):
        status = {
            "orders": {
                "c1": {
                    "client_order_id": "c1",
                    "broker_order_id": "bo-1",
                    "symbol": SYMBOL,
                    "side": "buy",
                    "quantity": 2.0,
                    "filled_quantity": 0.0,
                    "status": "accepted",
                    "submitted_at": "2026-10-29T13:30:00Z",
                }
            }
        }
        view = trades_view([], status)
        assert view["submitted_orders"] == 1
        assert view["filled_orders"] == 0
        assert view["orders"][0]["is_filled"] is False

    def test_the_view_is_json_safe(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        worker.run_once()
        worker.stop()

        # Round-tripping proves no internal object leaked into the payload.
        engine_payload = json.loads(json.dumps(engine_view(worker.status())))
        positions_payload = json.loads(json.dumps(positions_view(worker.status(), {})))
        assert isinstance(engine_payload, dict)
        assert isinstance(positions_payload, list)


@pytest.mark.unit
class TestDuplicateAndIdempotency:
    def test_a_repeated_cycle_does_not_double_charge_or_double_fill(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        worker, _ = _worker(root, experiment, source)
        worker.start()
        worker.run_once()
        worker.run_once()
        worker.run_once()
        worker.stop()

        # One decision, one order, no matter how many cycles run.
        assert len(worker.state.orders) == 1

    def test_an_order_the_worker_already_knows_is_reconciled_not_resubmitted(self, tmp_path):
        root = tmp_path / "exp"
        experiment, source = _experiment(root, processed=1)
        calls: list[dict] = []
        worker, _ = _worker(root, experiment, source, routes=_routes())
        worker._broker_factory = _broker_factory(_routes(), calls=calls)
        worker.start()
        worker.run_once()
        posts_after_first = [c for c in calls if c["method"] == "POST"]
        worker.run_once()
        posts_after_second = [c for c in calls if c["method"] == "POST"]
        worker.stop()

        # The second cycle placed no second order.
        assert len(posts_after_first) == 1
        assert len(posts_after_second) == 1
