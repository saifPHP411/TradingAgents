"""Phase 28: a current-market decision that placed no order is not "consumed".

The defect these tests answer, reproduced against the pre-fix build: on the
current-market path a decision that reached the worker but produced no order was
still recorded as *processed*, and the source's consumed-filter then treated the
whole session as done. The worker reported the generic
``CURRENT_MARKET_NO_NEW_DATA`` — "every symbol already has a decision for this
session" — with ``orders == {}`` and no honest reason. That is exactly the
reported Railway state: a SPY SELL in ``current_source_state.json``, the same id
in ``worker_state.processed_decisions``, no order, and
``last_no_trade_code == CURRENT_MARKET_NO_NEW_DATA``.

The fix keeps two memories apart. ``processed_decisions`` means "do not retry" and
still records a refusal (the historical path relies on that bound). The new
``ordered_decisions`` means "an order exists", and the current-market path filters
on it, so a decision that produced no order is re-offered and its honest reason
(``NO_POSITION``, ``RISK_REJECTED``, …) survives instead of being masked.

Everything here is offline: the broker transport, the market clock and the
decision data are all injected, so no test reaches the network (the suite's
autouse guard refuses it) and none needs an Alpaca key. The tests drive the real
``CurrentMarketDecisionSource``, the real ``PaperTradingWorker`` and the real risk
checks, not stubs of them.
"""

from __future__ import annotations

import json

import pytest

from tests.test_phase27_current_market_source import (
    SESSION,
    SYMBOL,
    _broker_factory,
    _current_source,
    _routes,
)
from tradingagents.paper_experiment.broker import (
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_HISTORY,
    NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_NO_POSITION,
    NO_TRADE_REASONS,
    NO_TRADE_RISK_REJECTED,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
)

DECISION_ID = f"current-{SYMBOL}-{SESSION}"


def _wired_worker(root, *, rating="Sell", routes=None, market_price=101.0, broker_limits=None):
    """A worker whose source's consumed-filter consults the worker's order memory.

    This is the real controller wiring: ``is_consumed`` is late-bound to the
    worker, so a decision is filtered as consumed only when it produced an order.

    ``broker_limits`` tightens the *broker's* limits only, so a refusal comes from
    the risk layer rather than from configuration validation (which checks
    ``order_notional`` against the config's own cap at construction).
    """
    holder: dict[str, PaperTradingWorker] = {}
    source = _current_source(
        rating=rating,
        state_path=root / "current_source_state.json",
        is_consumed=lambda decision_id: holder["worker"].has_order_for(decision_id),
    )
    worker = PaperTradingWorker(
        config=PaperTradingConfig(
            strategy_id="current-market",
            order_notional=25.0,
            whole_shares=False,
            state_path=root / "worker_state.json",
            decision_source=DECISION_SOURCE_CURRENT_MARKET,
        ),
        decision_source=source,
        market_data=StaticMarketData(prices={SYMBOL: market_price}),
        broker_factory=_broker_factory(routes or _routes(), limits=broker_limits),
    )
    holder["worker"] = worker
    return worker


def _seed_stale_processed(root) -> None:
    """The exact reported shape: a decision consumed with no order behind it."""
    (root / "worker_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "processed_decisions": [DECISION_ID],
                "last_processed_decision_id": DECISION_ID,
                "orders": {},
                "status": "STOPPED",
            }
        )
    )


@pytest.mark.unit
class TestANoOrderDecisionIsNotConsumed:
    def test_a_sell_with_no_position_is_re_offered_not_masked(self, tmp_path):
        """The reported failure: SELL, flat book, no order, and it must stay named.

        Before the fix the first cycle risk-rejected the SELL and consumed the
        decision, and the second cycle reported ``CURRENT_MARKET_NO_NEW_DATA``.
        Now the honest ``NO_POSITION`` is reported on every cycle.
        """
        worker = _wired_worker(tmp_path, rating="Sell")
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        worker.stop()

        assert first.reason_code == NO_TRADE_NO_POSITION
        assert second.reason_code == NO_TRADE_NO_POSITION
        assert second.reason_code != NO_TRADE_CURRENT_MARKET_NO_NEW_DATA
        assert worker.state.orders == {}
        assert worker.has_order_for(DECISION_ID) is False
        # The no-trade is not a risk breach and never a short.
        assert first.reason_code != NO_TRADE_RISK_REJECTED

    def test_the_reported_stale_state_self_heals(self, tmp_path):
        """A pre-fix state file (consumed, no order) must not freeze the session.

        This is the Railway state verbatim: the decision id is in
        ``processed_decisions`` but no order was ever placed. The current-market
        guard reads the order-only set, so the decision is re-evaluated and named.
        """
        _seed_stale_processed(tmp_path)
        worker = _wired_worker(tmp_path, rating="Sell")
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.reason_code == NO_TRADE_NO_POSITION
        assert worker.state.orders == {}

    def test_the_generic_code_is_still_the_closed_vocabulary_default(self):
        # The new path must not have removed the code other readers match on.
        assert NO_TRADE_CURRENT_MARKET_NO_NEW_DATA in NO_TRADE_REASONS

    def test_a_risk_refused_current_decision_stays_named(self, tmp_path):
        """A local refusal is recorded, not swallowed into "no new data"."""
        worker = _wired_worker(
            tmp_path,
            rating="Buy",
            broker_limits=PaperRiskLimits(max_order_notional=0.5),
        )
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        worker.stop()

        assert first.reason_code == NO_TRADE_RISK_REJECTED
        assert second.reason_code == NO_TRADE_RISK_REJECTED
        assert worker.state.orders == {}


@pytest.mark.unit
class TestAnOrderedDecisionIsConsumed:
    def test_a_buy_that_ordered_is_not_re_offered(self, tmp_path):
        worker = _wired_worker(tmp_path, rating="Buy")
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        worker.stop()

        assert first.ran is True
        assert first.order is not None and first.order.side == "buy"
        assert worker.has_order_for(DECISION_ID) is True
        assert second.reason_code == NO_TRADE_CURRENT_MARKET_NO_NEW_DATA
        assert len(worker.state.orders) == 1

    def test_a_restart_does_not_duplicate_an_ordered_current_decision(self, tmp_path):
        worker = _wired_worker(tmp_path, rating="Buy")
        worker.start()
        first = worker.run_once()
        worker.stop()
        assert first.order is not None
        client_order_id = first.order.client_order_id

        restarted = _wired_worker(tmp_path, rating="Buy")
        restarted.start()
        second = restarted.run_once()
        restarted.stop()

        assert second.reason_code in (
            NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
            "ALREADY_PROCESSED",
            "ORDER_EXISTS",
        )
        assert set(restarted.state.orders) == {client_order_id}
        assert restarted.has_order_for(DECISION_ID) is True


@pytest.mark.unit
class TestHonestReasonIsNotMasked:
    def test_a_hold_is_an_honest_no_trade_with_no_order(self, tmp_path):
        worker = _wired_worker(tmp_path, rating="Hold")
        worker.start()
        first = worker.run_once()
        worker.stop()
        assert first.ran is False
        assert first.order is None
        assert worker.state.orders == {}

    def test_market_closed_is_named_and_places_no_order(self, tmp_path):
        holder: dict[str, PaperTradingWorker] = {}
        source = _current_source(
            is_open=False,
            is_consumed=lambda decision_id: holder["worker"].has_order_for(decision_id),
        )
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="current-market",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source=DECISION_SOURCE_CURRENT_MARKET,
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=_broker_factory(_routes()),
        )
        holder["worker"] = worker
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.reason_code == NO_TRADE_MARKET_CLOSED
        assert worker.state.orders == {}

    def test_stale_decision_data_is_unavailable_and_places_no_order(self, tmp_path):
        from tests.test_phase27_current_market_source import _frame

        holder: dict[str, PaperTradingWorker] = {}
        source = _current_source(
            frame=_frame(end="2026-09-29"),  # a month before the session
            is_consumed=lambda decision_id: holder["worker"].has_order_for(decision_id),
        )
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="current-market",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source=DECISION_SOURCE_CURRENT_MARKET,
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=_broker_factory(_routes()),
        )
        holder["worker"] = worker
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.reason_code == NO_TRADE_MARKET_DATA_UNAVAILABLE
        assert worker.state.orders == {}


@pytest.mark.unit
class TestPersistenceAndSchema:
    def test_ordered_decisions_round_trips(self, tmp_path):
        worker = _wired_worker(tmp_path, rating="Buy")
        worker.start()
        worker.run_once()
        worker.stop()

        blob = json.loads((tmp_path / "worker_state.json").read_text())
        assert DECISION_ID in blob["ordered_decisions"]
        assert DECISION_ID in blob["processed_decisions"]

        reloaded = PaperTradingWorker.load(tmp_path / "worker_state.json")
        assert reloaded.ordered_decisions == {DECISION_ID}

    def test_an_older_state_file_reads_back_with_no_ordered_decisions(self, tmp_path):
        (tmp_path / "worker_state.json").write_text(
            json.dumps({"schema_version": 1, "processed_decisions": [DECISION_ID]})
        )
        state = PaperTradingWorker.load(tmp_path / "worker_state.json")
        assert state.ordered_decisions == set()
        assert state.processed_decisions == {DECISION_ID}


@pytest.mark.unit
class TestTheHistoryPathIsUnchanged:
    def test_history_refusal_is_still_bounded_not_re_offered(self):
        # The processed set is still what the historical path filters on, so a
        # refused historical decision is not retried every cycle. Pinned here so
        # the current-market change cannot silently re-open that bound.
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="history",
                order_notional=25.0,
                state_path=None,
                decision_source=DECISION_SOURCE_HISTORY,
            ),
            decision_source=_OneShot(),
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=_broker_factory(_routes()),
        )
        worker.start()
        worker.state.processed_decisions.add("hist-1")
        assert worker._is_processed("hist-1") is True  # noqa: SLF001
        assert worker._is_processed("hist-2") is False  # noqa: SLF001
        worker.stop()

    def test_the_default_decision_source_remains_history(self):
        assert PaperTradingConfig().decision_source == DECISION_SOURCE_HISTORY


class _OneShot:
    """A decision source that never yields — the history-path shape under test."""

    def next_decision(self):
        return None
