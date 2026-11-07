"""Phase 29: decision statistics for the current-market paper path.

The question these tests answer: the worker runs, the current source works and
Alpaca's snapshot is OK, yet no BUY or SELL appears. Is the scarcity a strategy
problem, a data problem, or a cadence problem? The dashboard could not tell,
because nothing counted how often the strategy chose HOLD or *why* the worker
placed no order.

This suite pins the observability that answers it, and nothing else:

- the source keeps a *lifetime* tally of strategy ratings (BUY/SELL/HOLD) and the
  model's own last HOLD reason, persisted so a restart does not reset it;
- the worker keeps a *settled-outcome* tally (orders produced vs no-trade, split
  by the ``NO_TRADE_REASONS`` code, including ``RISK_REJECTED``), persisted too;
- the two are reported separately, because only the source sees a rating the
  worker then refused to trade;
- the tallies are observations: no order, no strategy change, no cadence change,
  and the risk limits are untouched.

Everything is offline and deterministic — the same injected broker transport,
market clock and bar fetch the phase-27 suite uses — so no test reaches the
network and none needs an Alpaca key.
"""

from __future__ import annotations

import json

import pytest

from tests.test_phase27_current_market_source import (
    SYMBOL,
    _current_source,
    _routes,
    _worker,
)
from tradingagents.paper_experiment.broker import (
    DECISION_OUTCOME_ORDERED,
    NO_TRADE_NO_POSITION,
    NO_TRADE_REASONS,
    NO_TRADE_RISK_REJECTED,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
    WorkerState,
    worker_status_view,
)
from tradingagents.paper_experiment.dashboard.engine_view import engine_view

# ---------------------------------------------------------------------------
# 1. The source's strategy tally
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyRatingTally:
    def test_a_hold_is_counted_and_named(self, tmp_path):
        source = _current_source(rating="Hold", state_path=tmp_path / "current_source_state.json")
        source.next_decision()
        stats = source.statistics()
        assert stats["evaluations"] == 1
        assert stats["side_counts"]["hold"] == 1
        assert stats["side_counts"]["buy"] == 0
        assert stats["rating_counts"]["Hold"] == 1
        assert stats["last_rating"] == "Hold"
        assert stats["last_side"] is None
        # The model's own words are kept, so the dashboard can say *why* HOLD.
        assert stats["last_hold_reason"]

    def test_a_buy_is_counted_as_a_buy_not_a_hold(self, tmp_path):
        source = _current_source(rating="Buy", state_path=tmp_path / "s.json")
        source.next_decision()
        stats = source.statistics()
        assert stats["side_counts"]["buy"] == 1
        assert stats["side_counts"]["hold"] == 0
        assert stats["last_side"] == "buy"
        # A tradeable rating clears the HOLD reason — it is no longer a HOLD.
        assert stats["last_hold_reason"] is None

    def test_a_sell_is_counted_as_a_sell(self, tmp_path):
        source = _current_source(rating="Sell", state_path=tmp_path / "s.json")
        source.next_decision()
        assert source.statistics()["side_counts"]["sell"] == 1

    def test_conviction_tiers_map_to_their_side(self, tmp_path):
        # Overweight is a BUY tier and Underweight a SELL tier, so the tally must
        # agree with the action mapping rather than counting them as HOLD.
        overweight = _current_source(rating="Overweight", state_path=tmp_path / "a.json")
        overweight.next_decision()
        assert overweight.statistics()["side_counts"]["buy"] == 1
        underweight = _current_source(rating="Underweight", state_path=tmp_path / "b.json")
        underweight.next_decision()
        assert underweight.statistics()["side_counts"]["sell"] == 1

    def test_the_tally_survives_a_restart(self, tmp_path):
        state = tmp_path / "current_source_state.json"
        first = _current_source(rating="Hold", state_path=state)
        first.next_decision()
        # A fresh source over the same state file must not reset the count.
        restarted = _current_source(rating="Hold", state_path=state)
        assert restarted.statistics()["evaluations"] == 1
        # A new session evaluates again and increments.
        restarted.next_decision()
        assert restarted.statistics()["evaluations"] == 1  # same session: replay, no new call

    def test_a_market_closed_poll_does_not_count_as_an_evaluation(self, tmp_path):
        source = _current_source(is_open=False, state_path=tmp_path / "s.json")
        source.next_decision()
        stats = source.statistics()
        assert stats["evaluations"] == 0
        assert stats["last_rating"] is None

    def test_a_market_data_failure_does_not_count_as_an_evaluation(self, tmp_path):
        source = _current_source(
            bars_error=RuntimeError("vendor down"), state_path=tmp_path / "s.json"
        )
        source.next_decision()
        assert source.statistics()["evaluations"] == 0

    def test_the_tally_is_persisted_in_the_source_state(self, tmp_path):
        state = tmp_path / "current_source_state.json"
        source = _current_source(rating="Buy", state_path=state)
        source.next_decision()
        payload = json.loads(state.read_text())
        assert payload["statistics"]["side_counts"]["buy"] == 1

    def test_the_tally_does_not_write_the_experiment_state(self, tmp_path):
        root = tmp_path / "exp"
        root.mkdir()
        source = _current_source(state_path=root / "current_source_state.json")
        source.next_decision()
        names = sorted(p.name for p in root.iterdir())
        assert "current_source_state.json" in names
        assert "session_state.json" not in names
        assert "journal.json" not in names

    def test_a_describe_view_carries_the_tally(self, tmp_path):
        source = _current_source(rating="Hold", state_path=tmp_path / "s.json")
        source.next_decision()
        described = source.describe()
        assert described["decision_source_statistics"]["side_counts"]["hold"] == 1

    def test_a_replayed_decision_is_not_counted_twice(self, tmp_path):
        # A HOLD yields no decision object, so the second call in the same session
        # returns None; the model was asked exactly once, so the tally is 1.
        source = _current_source(rating="Hold", state_path=tmp_path / "s.json")
        assert source.next_decision() is None
        assert source.next_decision() is None
        assert source.statistics()["evaluations"] == 1


# ---------------------------------------------------------------------------
# 2. The worker's settled-outcome tally
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkerOutcomeTally:
    def test_an_order_is_recorded_as_ordered(self, tmp_path):
        source = _current_source()
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.order is not None
        stats = worker_status_view(state=worker.state, config=worker.config)["decision_statistics"]
        assert stats["orders_produced"] == 1
        assert stats["no_trade_total"] == 0

    def test_a_risk_refusal_is_counted_and_named(self, tmp_path):
        from tests.test_phase27_current_market_source import _transport
        from tradingagents.paper_experiment.broker import AlpacaPaperBroker

        source = _current_source()
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="current-market",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source="current_market",
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=lambda: AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport(_routes()),
                limits=PaperRiskLimits(max_order_notional=5.0),
            ),
        )
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.reason_code == NO_TRADE_RISK_REJECTED
        stats = worker_status_view(state=worker.state, config=worker.config)["decision_statistics"]
        assert stats["risk_rejections"] == 1
        assert stats["no_trade_by_code"][NO_TRADE_RISK_REJECTED] == 1
        assert stats["orders_produced"] == 0

    def test_a_sell_with_nothing_held_is_named_no_position(self, tmp_path):
        source = _current_source(rating="Sell")
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.reason_code == NO_TRADE_NO_POSITION
        stats = worker_status_view(state=worker.state, config=worker.config)["decision_statistics"]
        assert stats["no_trade_by_code"][NO_TRADE_NO_POSITION] == 1

    def test_every_no_trade_code_is_from_the_closed_vocabulary(self, tmp_path):
        source = _current_source(rating="Sell")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        worker.stop()
        stats = worker_status_view(state=worker.state, config=worker.config)["decision_statistics"]
        for code in stats["no_trade_by_code"]:
            assert code in NO_TRADE_REASONS

    def test_the_tally_survives_a_restart(self, tmp_path):
        source = _current_source()
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        worker.stop()
        # A fresh worker over the same state file reads the tally back.
        restarted = PaperTradingWorker(
            config=worker.config,
            decision_source=source,
            market_data=worker.market_data,
            broker_factory=lambda: None,
        )
        stats = worker_status_view(state=restarted.state, config=restarted.config)[
            "decision_statistics"
        ]
        assert stats["orders_produced"] == 1

    def test_the_outcome_map_round_trips_through_the_state_file(self, tmp_path):
        state = WorkerState(
            decision_outcomes={"d1": DECISION_OUTCOME_ORDERED, "d2": NO_TRADE_NO_POSITION}
        )
        again = WorkerState.from_dict(state.to_dict())
        assert again.decision_outcomes == {"d1": DECISION_OUTCOME_ORDERED, "d2": NO_TRADE_NO_POSITION}

    def test_an_older_state_file_reads_back_with_no_outcomes(self, tmp_path):
        payload = WorkerState().to_dict()
        payload.pop("decision_outcomes", None)
        assert WorkerState.from_dict(payload).decision_outcomes == {}


# ---------------------------------------------------------------------------
# 3. The engine view reports both tallies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineViewStatistics:
    def test_the_view_separates_strategy_and_worker_tallies(self, tmp_path):
        source = _current_source(rating="Hold", state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()  # a HOLD -> no order
        worker.stop()
        view = engine_view(worker.status())
        stats = view["statistics"]
        assert stats["strategy"]["side_counts"]["hold"] == 1
        assert stats["worker"]["orders_produced"] == 0
        assert stats["last_evaluated_at"] is not None

    def test_a_historical_source_reports_no_strategy_tally(self):
        # The historical source does not describe itself, so the strategy tally is
        # None — never a fabricated zero.
        view = engine_view({"status": "RUNNING", "loop_running": True})
        assert view["statistics"]["strategy"] is None
        assert view["statistics"]["worker"] is None

    def test_the_view_is_secret_free(self, tmp_path):
        source = _current_source(state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        worker.stop()
        blob = json.dumps(engine_view(worker.status()), default=str)
        assert "paper-secret" not in blob
        assert "paper-key" not in blob


# ---------------------------------------------------------------------------
# 4. The statistics change nothing about trading or the risk limits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatisticsAreObservational:
    def test_the_risk_limits_are_unchanged(self, tmp_path):
        source = _current_source()
        worker = _worker(tmp_path, source)
        view = worker_status_view(state=worker.state, config=worker.config)
        limits = view["risk_limits"]
        assert limits["max_order_notional"] == 25.0
        assert limits["max_open_positions"] == 5
        assert limits["max_daily_orders"] == 10
        assert limits["allow_short"] is False

    def test_collecting_statistics_does_not_submit_an_order(self, tmp_path):
        # A HOLD produces no order, and counting it must not change that.
        source = _current_source(rating="Hold", state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.order is None
        assert worker.state.orders == {}

    def test_the_dashboard_page_shows_the_statistics_card(self):
        from tradingagents.paper_experiment.dashboard.server import INDEX_HTML

        assert 'id="eng-stats"' in INDEX_HTML
        assert "Decision statistics" in INDEX_HTML
        assert "renderDecisionStats" in INDEX_HTML

    def test_a_second_run_on_the_same_decision_does_not_double_count(self, tmp_path):
        # The same session's decision is re-offered after it already produced an
        # order; the re-run places no second order and adds no second outcome.
        source = _current_source(rating="Buy", state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        worker.stop()
        assert first.order is not None
        assert second.order is None
        stats = worker_status_view(state=worker.state, config=worker.config)["decision_statistics"]
        assert stats["decisions_settled"] == 1
        assert stats["orders_produced"] == 1
