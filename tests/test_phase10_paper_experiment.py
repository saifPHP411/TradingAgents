"""Phase 10: the prospective paper experiment.

These tests are written to catch a premise error, not to confirm the code. The
headline claims — that a restart does not duplicate a fill, that a decision is
never executed at the price it was made from, that a missing credential records a
blocked decision rather than a fabricated one, that a small sample is not reported
as a metric — are each pinned by a test that would fail if the claim were false.

No test here calls a real provider. The inference seam is a deterministic stub,
and the price source is a fixed in-memory frame.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    ExperimentConfigError,
    ExperimentError,
    ExperimentJournal,
    JournalError,
    PaperExperiment,
    PriceSourceError,
    StaticPriceSource,
    build_report,
    current_market_source_available,
)
from tradingagents.paper_experiment.experiment import (
    STATE_BLOCKED,
    STATE_EVALUATED,
    STATE_EXECUTED,
    STATE_PENDING,
    STATE_UNEVALUABLE,
    InferenceUnavailable,
    SessionQuoteSource,
)
from tradingagents.paper_experiment.inference import StubInference

SESSIONS = 24


def _frame(periods: int = SESSIONS, *, start: float = 100.0, step: float = 1.0):
    """A deterministic ascending OHLC frame, one row per business day."""
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * periods,
        },
        index=index,
    )


def _source(frames=None):
    return StaticPriceSource(frames or {"SPY": _frame()})


def _config(**overrides):
    base = {
        "symbols": ("SPY",),
        "decision_frequency_sessions": 1,
        "evaluation_lag_sessions": 5,
    }
    base.update(overrides)
    return ExperimentConfig.from_protocol(**base)


def _experiment(tmp_path, *, inference=None, config=None, source=None, name="exp"):
    source = source or _source()
    config = config or _config()
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=tmp_path / name,
        inference=inference if inference is not None else StubInference(rating="Buy"),
    )
    experiment.ensure_created()
    return experiment


def _run_all(experiment, source):
    for session in source.sessions():
        experiment.process_session(session)
    return experiment


def _accounting(experiment):
    ledger = experiment.session.ledger
    return {
        "cash": round(ledger.cash, 6),
        "equity": round(ledger.equity, 6),
        "realized_pnl": round(ledger.realized_pnl, 6),
        "unrealized_pnl": round(ledger.unrealized_pnl, 6),
        "total_fees": round(ledger.total_fees, 6),
        "positions": {s: round(h.quantity, 6) for s, h in ledger.open_positions().items()},
    }


def _event_semantics(experiment):
    return [(e.type.value, e.symbol) for e in experiment.session.ledger.events]


# ---------------------------------------------------------------------------
# Phase B: the journal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJournal:
    def test_append_is_hash_chained_and_verifies(self, tmp_path):
        journal = ExperimentJournal(tmp_path / "journal.json")
        journal.append("experiment_created", {"a": 1}, event_id="e1", timestamp="2024-01-01T00:00:00+00:00")
        journal.append("session_processed", {"b": 2}, event_id="e2", timestamp="2024-01-02T00:00:00+00:00")
        journal.verify()
        assert len(journal) == 2
        assert journal.entries[0].prev_hash == "0" * 64
        assert journal.entries[1].prev_hash == journal.entries[0].entry_hash

    def test_reappending_the_same_event_id_does_not_duplicate(self, tmp_path):
        journal = ExperimentJournal(tmp_path / "journal.json")
        first = journal.append("session_processed", {"b": 2}, event_id="e2")
        second = journal.append("session_processed", {"b": 999}, event_id="e2")
        assert len(journal) == 1
        assert first.entry_hash == second.entry_hash
        assert journal.entries[0].payload == {"b": 2}

    def test_a_tampered_entry_is_detected(self, tmp_path):
        path = tmp_path / "journal.json"
        journal = ExperimentJournal(path)
        journal.append("session_processed", {"b": 2}, event_id="e2")
        raw = json.loads(path.read_text())
        raw["entries"][0]["payload"]["b"] = 999
        path.write_text(json.dumps(raw))
        with pytest.raises(JournalError):
            ExperimentJournal.load(path)

    def test_a_removed_entry_breaks_the_chain(self, tmp_path):
        path = tmp_path / "journal.json"
        journal = ExperimentJournal(path)
        journal.append("session_processed", {"b": 1}, event_id="e1")
        journal.append("session_processed", {"b": 2}, event_id="e2")
        raw = json.loads(path.read_text())
        del raw["entries"][0]
        path.write_text(json.dumps(raw))
        with pytest.raises(JournalError):
            ExperimentJournal.load(path)

    def test_an_unknown_schema_version_is_refused(self, tmp_path):
        path = tmp_path / "journal.json"
        ExperimentJournal(path).append("session_processed", {"b": 1}, event_id="e1")
        raw = json.loads(path.read_text())
        raw["schema_version"] = 999
        path.write_text(json.dumps(raw))
        with pytest.raises(JournalError):
            ExperimentJournal.load(path)

    def test_a_credential_shaped_value_is_refused(self, tmp_path):
        from tradingagents.llm_evaluation.records import CredentialLeakError

        journal = ExperimentJournal(tmp_path / "journal.json")
        with pytest.raises(CredentialLeakError):
            journal.append(
                "session_processed",
                {"key": "sk-abcdefghijklmnopqrstuvwxyz012345"},
                event_id="e1",
            )
        assert len(journal) == 0


# ---------------------------------------------------------------------------
# Phase C: the cycle and resume
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCycle:
    def test_a_decision_is_made_on_a_cadence_session(self, tmp_path):
        source = _source()
        config = _config(decision_frequency_sessions=5)
        experiment = _experiment(tmp_path, config=config, source=source)
        sessions = source.sessions()
        for session in sessions:
            experiment.process_session(session)
        # Sessions at index 0, 5, 10, 15, 20 are decision sessions.
        assert len(experiment.decisions) == 5
        for decision in experiment.decisions.values():
            assert sessions.index(decision.decision_session) % 5 == 0

    def test_decision_executes_at_the_next_session_open_not_the_decision_close(self, tmp_path):
        # A frame with a visible gap between T's close and T+1's open, so "filled
        # at the next open" and "filled at the decision close" are distinguishable.
        source = _source({"SPY": _frame(step=3.0)})
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[0])
        experiment.process_session(sessions[1])
        decision = experiment.decisions[f"SPY-{sessions[0]}"]
        assert decision.state == STATE_EXECUTED
        assert decision.execution_session == sessions[1]
        next_bar = source.bars("SPY", sessions[1])
        decision_bar = source.bars("SPY", sessions[0])
        # The fill is at T+1's open (adjusted only by the declared costs), never at
        # T's close.
        assert decision.average_fill_price == pytest.approx(next_bar.open, rel=0.01)
        assert decision.average_fill_price != pytest.approx(decision_bar.close, rel=0.001)

    def test_evaluation_happens_lag_sessions_after_execution(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        for session in sessions[:8]:
            experiment.process_session(session)
        decision = experiment.decisions[f"SPY-{sessions[0]}"]
        assert decision.state == STATE_EVALUATED
        assert decision.evaluation_session == sessions[1 + 5]

    def test_a_decision_stays_pending_until_its_evaluation_session(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[0])
        experiment.process_session(sessions[1])
        decision = experiment.decisions[f"SPY-{sessions[0]}"]
        assert decision.state == STATE_EXECUTED
        assert decision.evaluation_session == sessions[6]


@pytest.mark.unit
class TestResumeEqualsNoRestart:
    def _run(self, tmp_path, *, restart_at=None):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        for index, session in enumerate(sessions):
            if restart_at is not None and index == restart_at:
                experiment.save_state()
                experiment = PaperExperiment.load(
                    tmp_path / "exp", source=source, inference=StubInference(rating="Buy")
                )
            experiment.process_session(session)
        return experiment

    def test_scenario_a_and_b_agree_on_accounting(self, tmp_path):
        straight = self._run(tmp_path / "a")
        restarted = self._run(tmp_path / "b", restart_at=10)
        assert _accounting(straight) == _accounting(restarted)

    def test_scenario_a_and_b_agree_on_event_semantics(self, tmp_path):
        straight = self._run(tmp_path / "a")
        restarted = self._run(tmp_path / "b", restart_at=10)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_the_restored_event_sequence_continues_rather_than_restarting(self, tmp_path):
        restarted = self._run(tmp_path / "b", restart_at=10)
        seqs = [e.seq for e in restarted.session.ledger.events]
        assert seqs == list(range(1, len(seqs) + 1))

    def test_a_restart_does_not_duplicate_a_decision(self, tmp_path):
        straight = self._run(tmp_path / "a")
        restarted = self._run(tmp_path / "b", restart_at=10)
        assert len(straight.decisions) == len(restarted.decisions)
        assert set(straight.decisions) == set(restarted.decisions)

    def test_the_journal_is_not_duplicated_across_a_restart(self, tmp_path):
        restarted = self._run(tmp_path / "b", restart_at=10)
        ids = [e.event_id for e in restarted.journal.entries]
        assert len(ids) == len(set(ids))
        restarted.journal.verify()


@pytest.mark.unit
class TestDuplicateProcessing:
    def test_reprocessing_a_session_is_a_no_op(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        for session in sessions[:6]:
            experiment.process_session(session)
        before = _accounting(experiment)
        events_before = len(experiment.session.ledger.events)
        journal_before = len(experiment.journal)

        outcome = experiment.process_session(sessions[3])
        assert outcome.processed is False
        assert _accounting(experiment) == before
        assert len(experiment.session.ledger.events) == events_before
        assert len(experiment.journal) == journal_before

    def test_reprocessing_after_a_restart_is_a_no_op(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        for session in sessions[:6]:
            experiment.process_session(session)
        experiment.save_state()
        reloaded = PaperExperiment.load(
            tmp_path / "exp", source=source, inference=StubInference(rating="Buy")
        )
        before = _accounting(reloaded)
        reloaded.process_session(sessions[3])
        assert _accounting(reloaded) == before

    def test_a_session_earlier_than_one_already_processed_is_refused(self, tmp_path):
        """Forward-only: a backwards step would reorder the book silently."""
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[5])
        with pytest.raises(ExperimentError, match="out of order"):
            experiment.process_session(sessions[2])

    def test_the_refused_backwards_step_changes_nothing(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[5])
        before = _accounting(experiment)
        events_before = len(experiment.session.ledger.events)
        with pytest.raises(ExperimentError):
            experiment.process_session(sessions[2])
        assert _accounting(experiment) == before
        assert len(experiment.session.ledger.events) == events_before
        assert sessions[2] not in experiment.processed_sessions

    def test_a_backwards_step_after_a_restart_is_also_refused(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        # Leave the first two sessions unprocessed, then advance past them.
        for session in sessions[2:8]:
            experiment.process_session(session)
        experiment.save_state()
        reloaded = PaperExperiment.load(
            tmp_path / "exp", source=source, inference=StubInference(rating="Buy")
        )
        assert sessions[1] not in reloaded.processed_sessions
        with pytest.raises(ExperimentError, match="out of order"):
            reloaded.process_session(sessions[1])

    def test_forward_processing_still_works_after_a_gap(self, tmp_path):
        """A skipped session is not a backwards step, so it stays allowed."""
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[0])
        experiment.process_session(sessions[6])
        assert sessions[6] in experiment.processed_sessions
        assert sessions[6] > sessions[0]


# ---------------------------------------------------------------------------
# Phase D: honesty safeguards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHonestySafeguards:
    def test_no_credential_records_a_blocked_decision_not_a_fabricated_one(self, tmp_path):
        def failing(*, symbol, session, window=None):
            raise InferenceUnavailable("no credential for provider 'groq'")

        source = _source()
        experiment = _experiment(tmp_path, inference=failing, source=source)
        outcome = experiment.process_session(source.sessions()[0])
        assert outcome.decision_state == STATE_BLOCKED
        assert experiment.counts()[STATE_BLOCKED] == 1
        assert experiment.counts()[STATE_EVALUATED] == 0
        # No order, no fill, no position.
        assert experiment.session.ledger.cash == experiment.session.ledger.initial_cash
        assert experiment.session.ledger.open_positions() == {}

    def test_a_missing_inference_callable_blocks_rather_than_fabricates(self, tmp_path):
        source = _source()
        experiment = PaperExperiment(
            source=source, config=_config(), root=tmp_path / "exp", inference=None
        )
        experiment.ensure_created()
        outcome = experiment.process_session(source.sessions()[0])
        assert outcome.decision_state == STATE_BLOCKED

    def test_a_provider_failure_blocks_and_does_not_raise(self, tmp_path):
        def exploding(*, symbol, session, window=None):
            raise RuntimeError("upstream 500")

        source = _source()
        experiment = _experiment(tmp_path, inference=exploding, source=source)
        outcome = experiment.process_session(source.sessions()[0])
        assert outcome.decision_state == STATE_BLOCKED
        assert "provider failure" in outcome.blocked_reason

    def test_a_stub_decision_is_labelled_synthetic_fixture(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, inference=StubInference(rating="Buy"), source=source)
        experiment.process_session(source.sessions()[0])
        decision = experiment.decisions[f"SPY-{source.sessions()[0]}"]
        assert decision.source == "synthetic_fixture"

    def test_a_synthetic_fixture_is_never_counted_as_genuine(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        report = build_report(experiment).to_dict()
        assert report["counts"]["genuine_inferences"] == 0
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"

    def test_the_frozen_source_is_not_reported_as_the_current_market(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        report = build_report(experiment).to_dict()
        assert report["data"]["is_current_market"] is False
        assert "NOT the current market" in report["data"]["note"]

    def test_a_current_market_source_exists_but_is_quote_only(self):
        """Phase 11 added real providers; the honesty claim is now stronger.

        A current-market source now ships (Finnhub's quote), so the old premise
        "none exists" is obsolete. The claim that matters is kept and tightened:
        the current-market source is a *quote* that cannot drive a multi-session
        run, so it can never make an experiment a live paper run on its own.
        """
        from tradingagents.paper_experiment.providers import (
            FinnhubQuoteSource,
            current_market_sources,
        )

        assert current_market_source_available() is True
        assert FinnhubQuoteSource in current_market_sources()
        assert FinnhubQuoteSource.kind == "vendor_quote"

        # The quote source refuses every historical request, so it cannot stand in
        # for the bar history a multi-session experiment needs.
        from tradingagents.paper_experiment.providers import ProviderUnsupportedError

        source = FinnhubQuoteSource.__new__(FinnhubQuoteSource)
        source.symbol = "SPY"
        source.name = "finnhub_quote"
        with pytest.raises(ProviderUnsupportedError):
            source.sessions()
        with pytest.raises(ProviderUnsupportedError):
            source.bars("SPY", "2024-01-02")
        with pytest.raises(ProviderUnsupportedError):
            source.window("SPY", "2024-01-02")

    def test_a_snapshot_window_ends_at_the_decision_date(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        session = source.sessions()[0]
        experiment.process_session(session)
        entry = experiment.journal.get(f"{experiment.experiment_id}:decision:SPY-{session}")
        record = entry.payload["record"]
        assert record["window_end"] == session
        assert record["as_of_date"] == session

    def test_the_window_never_includes_a_future_bar(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        session = source.sessions()[5]
        experiment.process_session(session)
        snapshot = experiment._snapshot("SPY", session)
        assert snapshot.window_end == session
        assert all(bar["date"] <= session for bar in snapshot.bars)

    def test_no_secret_appears_in_the_state_or_journal(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        text = (tmp_path / "exp" / "session_state.json").read_text()
        text += (tmp_path / "exp" / "journal.json").read_text()
        for token in ("api_key", "API_KEY", "sk-", "secret", "token"):
            assert token.lower() not in text.lower() or token.lower() == "token"


# ---------------------------------------------------------------------------
# Phase D/E: evaluation honesty and the report
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluationAndReport:
    def test_an_evaluated_decision_records_the_outcome(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        evaluated = [d for d in experiment.decisions.values() if d.state == STATE_EVALUATED]
        assert evaluated
        for decision in evaluated:
            assert decision.evaluation is not None

    def test_a_decision_with_no_later_data_stays_unevaluable(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        # The last decision session's evaluation window does not exist.
        for session in sessions:
            experiment.process_session(session)
        last_decision = experiment.decisions[f"SPY-{sessions[-1]}"]
        assert last_decision.state in (STATE_PENDING, STATE_EXECUTED, STATE_UNEVALUABLE)

    def test_a_hold_decision_produces_no_fill(self, tmp_path):
        source = _source()
        experiment = _experiment(
            tmp_path, inference=StubInference(rating="Hold"), source=source
        )
        _run_all(experiment, source)
        assert experiment.session.ledger.open_positions() == {}
        assert experiment.session.ledger.cash == experiment.session.ledger.initial_cash

    def test_the_report_shows_counts_and_sample_size(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        report = build_report(experiment).to_dict()
        assert report["counts"]["sessions_processed"] == len(source.sessions())
        assert "sample" in report["performance"]
        assert report["performance"]["sample"]["closed_outcomes"] >= 0

    def test_a_small_sample_reports_metrics_as_none_not_zero(self, tmp_path):
        source = _source()
        config = _config(min_trades_for_metrics=100)
        experiment = _experiment(tmp_path, config=config, source=source)
        _run_all(experiment, source)
        report = build_report(experiment).to_dict()
        assert report["performance"]["sample_limited"] is True
        assert report["performance"]["max_drawdown"] is None
        assert report["performance"]["sharpe_ratio"] is None
        assert report["performance"]["win_rate"] is None

    def test_the_report_does_not_claim_live_readiness_on_frozen_data(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        report = build_report(experiment).to_dict()
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"
        assert report["readiness"]["blockers"]

    def test_the_report_is_json_serialisable_and_secret_free(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        report = build_report(experiment)
        text = json.dumps(report.to_dict())
        assert "sk-" not in text
        assert report.write(tmp_path / "report.json").exists()

    def test_the_report_names_the_costs_and_their_nature(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        report = build_report(experiment).to_dict()
        assert "model" in report["costs"]
        assert "modelled" in report["costs"]["note"]

    def test_the_report_includes_limitations(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        report = build_report(experiment).to_dict()
        assert report["limitations"]
        assert any("paper only" in item for item in report["limitations"])


# ---------------------------------------------------------------------------
# Phase D: malformed and missing data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMalformedData:
    def test_a_nan_price_is_refused_at_construction(self):
        with pytest.raises(PriceSourceError):
            StaticPriceSource(
                {"SPY": pd.DataFrame(
                    {"open": [float("nan")], "high": [1.0], "low": [1.0], "close": [1.0]},
                    index=pd.to_datetime(["2024-01-01"]),
                )}
            )

    def test_a_high_below_low_is_refused(self):
        # Validated eagerly at construction, so a malformed fixture fails before a
        # run starts rather than part-way through it.
        with pytest.raises(PriceSourceError):
            StaticPriceSource(
                {"SPY": pd.DataFrame(
                    {"open": [1.0], "high": [1.0], "low": [2.0], "close": [1.0]},
                    index=pd.to_datetime(["2024-01-01"]),
                )}
            )

    def test_an_unknown_symbol_is_refused(self):
        source = _source()
        with pytest.raises(PriceSourceError):
            source.bars("NOPE", source.sessions()[0])

    def test_an_uncovered_session_is_refused(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        with pytest.raises(ExperimentError):
            experiment.process_session("1999-01-01")

    def test_an_empty_window_is_not_a_valid_snapshot(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        # A window before the first session cannot exist.
        assert experiment._snapshot("SPY", "1999-01-01") is None


# ---------------------------------------------------------------------------
# Config and boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigAndBoundary:
    def test_next_open_execution_cannot_be_disabled(self):
        with pytest.raises(ExperimentConfigError):
            _config(execute_on_next_open=False)

    def test_config_hash_changes_with_the_configuration(self):
        a = _config()
        b = _config(decision_frequency_sessions=3)
        assert a.config_hash() != b.config_hash()

    def test_config_round_trips_and_detects_tampering(self):
        config = _config()
        payload = config.to_dict()
        assert ExperimentConfig.from_dict(payload).config_hash() == config.config_hash()
        payload["decision_frequency_sessions"] = 99
        with pytest.raises(ExperimentConfigError):
            ExperimentConfig.from_dict(payload)

    def test_the_experiment_is_paper_only(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        assert experiment.session.mode.value == "PAPER"
        assert experiment.mode.value == "PAPER"

    def test_the_session_quote_source_refuses_a_mismatched_symbol(self):
        from tradingagents.execution.paper import Quote

        source = SessionQuoteSource()
        source.set(
            Quote(symbol="SPY", price=100.0, timestamp=datetime(2024, 1, 1, tzinfo=UTC))
        )
        with pytest.raises(ExperimentError):
            source("QQQ")
        source.clear()
        with pytest.raises(ExperimentError):
            source("SPY")


# ---------------------------------------------------------------------------
# Partial-fill recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialFillRecovery:
    def _thin_source(self):
        # A thin book: each session trades 100 units, so a large order can only
        # fill a slice per session and must be worked across sessions.
        index = pd.date_range("2024-01-01", periods=24, freq="B")
        opens = [100.0 + i for i in range(24)]
        return StaticPriceSource(
            {"SPY": pd.DataFrame(
                {
                    "open": opens,
                    "high": [o + 2 for o in opens],
                    "low": [o - 1 for o in opens],
                    "close": [o + 1 for o in opens],
                    "volume": [100.0] * 24,
                },
                index=index,
            )}
        )

    def _config(self):
        return _config(max_volume_participation=0.1)

    def _run(self, tmp_path, *, restart_at=None):
        source = self._thin_source()
        experiment = _experiment(tmp_path, config=self._config(), source=source)
        sessions = source.sessions()
        for index, session in enumerate(sessions):
            if restart_at is not None and index == restart_at:
                experiment.save_state()
                experiment = PaperExperiment.load(
                    tmp_path / "exp", source=source, inference=StubInference(rating="Buy")
                )
            experiment.process_session(session)
        return experiment, source

    def test_the_thin_book_produces_partial_fills(self, tmp_path):
        experiment, _ = self._run(tmp_path / "a")
        statuses = [d.order_status for d in experiment.decisions.values() if d.order_status]
        assert "PARTIALLY_FILLED" in statuses

    def test_a_partial_fill_survives_a_restart_with_identical_accounting(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        assert _accounting(straight) == _accounting(restarted)

    def test_a_partial_fill_survives_a_restart_with_identical_events(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_no_fill_is_applied_twice_across_a_restart(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        straight_fills = [e for e in straight.session.ledger.events if e.type.value == "ORDER_FILLED"]
        restarted_fills = [
            e for e in restarted.session.ledger.events if e.type.value == "ORDER_FILLED"
        ]
        assert len(straight_fills) == len(restarted_fills)
        assert sum(e.detail["quantity"] for e in straight_fills) == pytest.approx(
            sum(e.detail["quantity"] for e in restarted_fills)
        )

    def test_fees_are_charged_once_per_fill_across_a_restart(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        assert straight.session.ledger.total_fees == pytest.approx(
            restarted.session.ledger.total_fees
        )
        fee_events_straight = [
            e for e in straight.session.ledger.events if e.type.value == "FEE_CHARGED"
        ]
        fee_events_restarted = [
            e for e in restarted.session.ledger.events if e.type.value == "FEE_CHARGED"
        ]
        assert len(fee_events_straight) == len(fee_events_restarted)

    def test_the_restarted_position_equals_the_straight_one(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        straight_pos = straight.session.ledger.open_positions().get("SPY")
        restarted_pos = restarted.session.ledger.open_positions().get("SPY")
        assert straight_pos is not None and restarted_pos is not None
        assert straight_pos.quantity == pytest.approx(restarted_pos.quantity)


# ---------------------------------------------------------------------------
# Resting GTC limit recovery
# ---------------------------------------------------------------------------


def _limit_signal(symbol: str, session: str, limit_price: float, *, expires=None):
    from tradingagents.signals import OrderIntent, SignalAction, SignalTimeInForce, TradingSignal

    return TradingSignal(
        symbol=symbol,
        action=SignalAction.BUY,
        confidence=0.8,
        timestamp=datetime.fromisoformat(f"{session}T21:00:00+00:00"),
        as_of_date=session,
        position_size=1.0,
        order_intent=OrderIntent.LIMIT,
        limit_price=limit_price,
        time_in_force=SignalTimeInForce.GTC,
        expires_session=expires,
        entry_price=limit_price,
    )


@pytest.mark.unit
class TestRestingLimitRecovery:
    def _seed_resting(self, experiment, source, session, limit_price, *, expires=None):
        """Put a GTC limit into the session's book, as a decision would."""
        from tradingagents.execution.paper import Quote

        bar = source.bars("SPY", session)
        experiment._quote_source.set(
            Quote(
                symbol="SPY",
                price=bar.open,
                timestamp=datetime.fromisoformat(f"{session}T21:00:00+00:00"),
                is_open=True,
                low=bar.low,
                high=bar.high,
                average_daily_volume=bar.volume,
            )
        )
        try:
            step = experiment.session.on_signal(
                _limit_signal("SPY", session, limit_price, expires=expires)
            )
        finally:
            experiment._quote_source.clear()
        return step

    def _flat_source(self, closes):
        """A source whose sessions have the given closes, flat OHLC."""
        index = pd.date_range("2024-01-01", periods=len(closes), freq="B")
        return StaticPriceSource(
            {"SPY": pd.DataFrame(
                {
                    "open": closes,
                    "high": [c + 0.5 for c in closes],
                    "low": [c - 0.5 for c in closes],
                    "close": closes,
                    "volume": [1_000_000.0] * len(closes),
                },
                index=index,
            )}
        )

    def test_a_resting_limit_survives_a_restart_and_fills_once(self, tmp_path):
        # A limit below the market that the first session does not reach, then a
        # later session that trades down through it.
        closes = [100.0, 101.0, 102.0, 90.0, 90.0, 90.0]
        source = self._flat_source(closes)
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()

        # Session 0: submit a limit at 95, not reached (market is 100).
        step = self._seed_resting(experiment, source, sessions[0], 95.0)
        assert step.order_status == "ACCEPTED"
        assert len(experiment.session.resting) == 1
        experiment.save_state()

        # Restart, and process sessions until the price trades down through 95.
        reloaded = PaperExperiment.load(
            tmp_path / "exp", source=source, inference=StubInference(rating="Hold")
        )
        assert len(reloaded.session.resting) == 1
        resting = reloaded.session.resting[0]
        assert resting.status.value == "ACCEPTED"
        assert resting.remaining_quantity == pytest.approx(resting.quantity)

        for session in sessions[1:]:
            reloaded.process_session(session)

        fills = [e for e in reloaded.session.ledger.events if e.type.value == "ORDER_FILLED"]
        assert len(fills) == 1
        assert fills[0].detail["quantity"] == pytest.approx(resting.quantity)
        assert len(reloaded.session.resting) == 0

    def test_a_limit_not_reached_stays_resting_across_a_restart(self, tmp_path):
        closes = [100.0, 100.0, 100.0, 100.0]
        source = self._flat_source(closes)
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        self._seed_resting(experiment, source, sessions[0], 50.0)
        experiment.save_state()
        reloaded = PaperExperiment.load(
            tmp_path / "exp", source=source, inference=StubInference(rating="Hold")
        )
        for session in sessions[1:]:
            reloaded.process_session(session)
        assert len(reloaded.session.resting) == 1
        assert reloaded.session.ledger.position_quantity("SPY") == 0.0
        assert not [e for e in reloaded.session.ledger.events if e.type.value == "ORDER_FILLED"]

    def test_an_expiring_limit_transitions_to_expired_exactly_once(self, tmp_path):
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]
        source = self._flat_source(closes)
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        # A limit far below the market, expiring after session 1.
        self._seed_resting(experiment, source, sessions[0], 50.0, expires=sessions[1])
        experiment.save_state()

        reloaded = PaperExperiment.load(
            tmp_path / "exp", source=source, inference=StubInference(rating="Hold")
        )
        for session in sessions[1:]:
            reloaded.process_session(session)
        expired = [e for e in reloaded.session.ledger.events if e.type.value == "ORDER_CANCELLED"]
        assert len(expired) == 1
        # The rejection carries the named reason, not prose only.
        assert expired[0].detail.get("reason") == "EXPIRED"
        assert len(reloaded.session.resting) == 0

    def test_the_resting_order_state_is_json_safe(self, tmp_path):
        closes = [100.0, 100.0, 100.0]
        source = self._flat_source(closes)
        experiment = _experiment(tmp_path, source=source)
        self._seed_resting(experiment, source, source.sessions()[0], 95.0)
        text = json.dumps(experiment.state())
        # The recovery state carries the live resting order, JSON-safe.
        assert '"resting"' in text
        assert "95" in text


# ---------------------------------------------------------------------------
# Multi-symbol and multi-order recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiSymbolRecovery:
    def _multi_source(self):
        return StaticPriceSource({"SPY": _frame(start=100.0), "QQQ": _frame(start=200.0)})

    def _run(self, tmp_path, *, restart_at=None):
        source = self._multi_source()
        config = _config(symbols=("SPY", "QQQ"), decision_frequency_sessions=2)
        experiment = _experiment(tmp_path, config=config, source=source)
        sessions = source.sessions()
        for index, session in enumerate(sessions):
            if restart_at is not None and index == restart_at:
                experiment.save_state()
                experiment = PaperExperiment.load(
                    tmp_path / "exp", source=source, inference=StubInference(rating="Buy")
                )
            experiment.process_session(session)
        return experiment, source

    def test_two_symbols_are_decided_independently(self, tmp_path):
        experiment, source = self._run(tmp_path / "a")
        symbols = {d.symbol for d in experiment.decisions.values()}
        assert symbols == {"SPY", "QQQ"}

    def test_multi_symbol_accounting_survives_a_restart(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=8)
        assert _accounting(straight) == _accounting(restarted)

    def test_multi_symbol_event_semantics_survive_a_restart(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=8)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_two_positions_are_held_after_the_run(self, tmp_path):
        experiment, _ = self._run(tmp_path / "a")
        positions = experiment.session.ledger.open_positions()
        assert set(positions) == {"SPY", "QQQ"}


@pytest.mark.unit
class TestLookAheadAndTiming:
    def test_the_decision_session_differs_from_the_execution_session(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        executed = [d for d in experiment.decisions.values() if d.state == STATE_EXECUTED]
        assert executed
        for decision in executed:
            assert decision.execution_session > decision.decision_session

    def test_the_evaluation_session_is_after_the_execution_session(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        evaluated = [d for d in experiment.decisions.values() if d.state == STATE_EVALUATED]
        assert evaluated
        for decision in evaluated:
            assert decision.evaluation_session is not None
            assert decision.evaluation_session > decision.execution_session

    def test_an_evaluation_price_is_never_a_decision_time_price(self, tmp_path):
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        _run_all(experiment, source)
        for decision in experiment.decisions.values():
            if decision.evaluation and decision.evaluation.get("exit_price"):
                assert decision.evaluation_session > decision.decision_session

    def test_the_execution_uses_the_recorded_execution_session(self, tmp_path):
        """A gap in processing must not move execution to a later bar."""
        source = _source()
        experiment = _experiment(tmp_path, source=source)
        sessions = source.sessions()
        experiment.process_session(sessions[0])
        # Skip several sessions, then process a later one. The pending decision
        # must settle at its own execution session (T+1), not the current session.
        experiment.process_session(sessions[6])
        decision = experiment.decisions[f"SPY-{sessions[0]}"]
        assert decision.execution_session == sessions[1]
        assert decision.state in (STATE_EXECUTED, STATE_EVALUATED)


@pytest.mark.unit
class TestFrozenSourceIntegration:
    def test_a_run_on_the_frozen_dataset_is_reproducible(self, tmp_path):
        from tradingagents.baseline.data import load_feed
        from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL
        from tradingagents.paper_experiment import FrozenPriceSource

        feed = load_feed(DEFAULT_PROTOCOL)
        source = FrozenPriceSource(feed)
        assert source.is_current_market is False
        assert source.dataset_start == "2015-01-02"
        assert source.dataset_end == "2024-12-30"
        assert len(source.sessions()) == 2515

        config = _config(symbols=("SPY",), decision_frequency_sessions=20)
        experiment = PaperExperiment(
            source=source,
            config=config,
            root=tmp_path / "frozen",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        sessions = source.sessions()
        for session in sessions[:40]:
            experiment.process_session(session)
        report = build_report(experiment).to_dict()
        assert report["data"]["sessions_available"] == 2515
        assert report["counts"]["sessions_processed"] == 40
        # The frozen window the decision saw ends at the decision date.
        first = experiment.decisions[f"SPY-{sessions[0]}"]
        assert first.data_window_hash is not None
        assert report["readiness"]["status"] != "READY FOR LIVE PAPER"

    def test_the_frozen_source_refuses_a_session_it_does_not_cover(self):
        from tradingagents.baseline.data import load_feed
        from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL
        from tradingagents.paper_experiment import FrozenPriceSource

        source = FrozenPriceSource(load_feed(DEFAULT_PROTOCOL))
        with pytest.raises(PriceSourceError):
            source.bars("SPY", "1999-01-01")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunnerCLI:
    def test_start_run_report_verify(self, tmp_path, capsys):
        from tradingagents.paper_experiment.__main__ import main

        root = str(tmp_path / "cli")
        assert main(["start", "--root", root, "--stub"]) == 0
        assert main(["run", "--root", root, "--stub", "--sessions", "4"]) == 0
        report_path = str(tmp_path / "report.json")
        assert main(["report", "--root", root, "--out", report_path]) == 0
        assert main(["verify", "--root", root]) == 0
        assert json.loads((tmp_path / "report.json").read_text())["counts"]["sessions_processed"] == 4
        captured = capsys.readouterr()
        assert "journal OK" in captured.out

    def test_verify_reports_a_tampered_journal(self, tmp_path):
        from tradingagents.paper_experiment.__main__ import main

        root = tmp_path / "cli"
        assert main(["start", "--root", str(root), "--stub"]) == 0
        journal_path = root / "journal.json"
        raw = json.loads(journal_path.read_text())
        raw["entries"][0]["payload"]["config_hash"] = "0" * 64
        journal_path.write_text(json.dumps(raw))
        assert main(["verify", "--root", str(root)]) == 1

