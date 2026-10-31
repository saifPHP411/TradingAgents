"""Tests for the LLM-strategy evaluation: records, replay, determinism, safety.

Two things these tests deliberately do NOT do: call a model, or report a number
as LLM performance. They exercise the replay path with clearly-labelled
*synthetic fixtures*, and they assert that a fixture result is never presented as
a genuine inference result.

The most important tests here are the ones that could catch a lie:

- replay never constructs an LLM client;
- a fixture record set is marked ``synthetic_fixture`` and never ``llm_inference``;
- a missing credential produces ``LLM BASELINE BLOCKED`` with no fabricated row;
- a credential-shaped value cannot be serialised into a record set;
- a replay on different data is refused by the window hash.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.backtesting import MarketDataFeed
from tradingagents.llm_evaluation import (
    DEFAULT_PROTOCOL as LLM_PROTOCOL,
    STATUS_BLOCKED,
    STATUS_ESTABLISHED_WITH_FINDINGS,
    CredentialLeakError,
    DecisionRecord,
    DecisionRecordError,
    DecisionRecordSet,
    ReplayError,
    ReplayStrategy,
    build_evaluation,
    credential_status,
    no_llm_calls,
    redact_reasoning,
    signal_from_record,
    window_hash,
)
from tradingagents.llm_evaluation.records import (
    SOURCE_LLM_INFERENCE,
    SOURCE_OPERATOR_SUPPLIED,
    SOURCE_SYNTHETIC_FIXTURE,
)

# ---------------------------------------------------------------------------
# A small, deterministic feed and record factory
# ---------------------------------------------------------------------------

# The replay tests run through the real engine over the protocol's *test*
# window, so the fixture's sessions sit inside 2022-2024 and the window is wide
# enough for a decision at T to fill at T+1.
SESSIONS = [f"2022-03-{d:02d}" for d in range(1, 15)]  # 14 sessions
CLOSES = [100.0 + i for i in range(14)]


def _frame(closes=None, dates=None) -> pd.DataFrame:
    closes = closes if closes is not None else CLOSES
    dates = dates if dates is not None else SESSIONS
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=pd.to_datetime(dates),
    )


@pytest.fixture
def feed() -> MarketDataFeed:
    # SPY is present because the harness measures a buy-and-hold benchmark on the
    # protocol's benchmark symbol; the replay tests run through that harness.
    return MarketDataFeed(
        {"AAPL": _frame(), "SPY": _frame([200.0 + i for i in range(len(CLOSES))])}
    )


def make_record(feed, *, session="2022-03-06", ticker="AAPL", action="BUY",
                rating="Buy", source=SOURCE_SYNTHETIC_FIXTURE, **overrides) -> DecisionRecord:
    frame = feed.history(ticker, session)
    fields = {
        "decision_id": f"{ticker}-{session}",
        "ticker": ticker,
        "as_of_date": session,
        "action": action,
        "rating": rating,
        "confidence": 0.6,
        "signal_timestamp": f"{session}T20:00:00+00:00",
        "window_start": SESSIONS[0],
        "window_end": session,
        "bar_count": len(frame),
        "data_snapshot_id": "fixture",
        "data_window_hash": window_hash(frame),
        "config_hash": LLM_PROTOCOL.config_hash(),
        "prompt_version": "tradingagents-default",
        "graph_version": "test-graph",
        "source": source,
        "reasoning_excerpt": "fixture decision",
        "position_size": 0.20,
        "stop_loss": None,
        "order_intent": "MARKET",
        "time_in_force": "DAY",
    }
    fields.update(overrides)
    return DecisionRecord(**fields)


def make_set(decisions, *, mode=SOURCE_SYNTHETIC_FIXTURE, strategy_id="fixture") -> DecisionRecordSet:
    return DecisionRecordSet(
        strategy_id=strategy_id,
        graph_version="test-graph",
        protocol_version=LLM_PROTOCOL.version,
        mode=mode,
        created_at="2026-01-01T00:00:00+00:00",
        decisions=tuple(decisions),
    )


# ---------------------------------------------------------------------------
# Records: validation and serialisation
# ---------------------------------------------------------------------------


class TestDecisionRecord:
    def test_a_valid_record_round_trips(self, feed):
        record = make_record(feed)
        again = DecisionRecord.from_dict(record.to_dict())
        assert again == record

    def test_the_record_is_json_safe(self, feed):
        text = json.dumps(make_record(feed).to_dict())
        assert "AAPL" in text

    def test_an_unknown_schema_version_is_refused(self, feed):
        payload = make_record(feed).to_dict()
        payload["schema_version"] = 999
        with pytest.raises(DecisionRecordError, match="schema_version"):
            DecisionRecord.from_dict(payload)

    def test_a_missing_field_is_refused(self, feed):
        payload = make_record(feed).to_dict()
        del payload["rating"]
        with pytest.raises(DecisionRecordError, match="missing"):
            DecisionRecord.from_dict(payload)

    def test_an_unknown_field_is_refused(self, feed):
        payload = make_record(feed).to_dict()
        payload["surprise"] = 1
        with pytest.raises(DecisionRecordError, match="unknown fields"):
            DecisionRecord.from_dict(payload)

    def test_a_future_window_is_refused(self, feed):
        # The window must end at the decision date; a later end means the model
        # saw a bar after the decision.
        with pytest.raises(DecisionRecordError, match="window_end"):
            make_record(feed, window_end="2022-03-05", session="2022-03-06")

    def test_an_unparseable_date_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="yyyy-mm-dd"):
            make_record(feed, session="06/01/2026", window_end="06/01/2026")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_confidence_is_refused(self, feed, bad):
        with pytest.raises(DecisionRecordError, match="finite"):
            make_record(feed, confidence=bad)

    def test_a_confidence_outside_zero_one_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match=r"\[0, 1\]"):
            make_record(feed, confidence=1.5)

    def test_an_unknown_action_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="action"):
            make_record(feed, action="YOLO", rating="Buy")

    def test_an_unknown_source_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="source"):
            make_record(feed, source="something_else")

    def test_a_market_record_may_not_carry_a_limit_price(self, feed):
        with pytest.raises(DecisionRecordError, match="limit_price"):
            make_record(feed, order_intent="MARKET", limit_price=99.0)

    def test_a_day_record_may_not_be_gtc(self, feed):
        with pytest.raises(DecisionRecordError, match="GTC"):
            make_record(feed, order_intent="MARKET", time_in_force="GTC")

    def test_a_non_positive_stop_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="positive"):
            make_record(feed, stop_loss=0.0)

    def test_a_short_hash_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="SHA-256"):
            make_record(feed, config_hash="abc")


class TestDecisionRecordSet:
    def test_a_round_trip_preserves_the_set(self, feed):
        record_set = make_set([make_record(feed)])
        again = DecisionRecordSet.from_dict(record_set.to_dict())
        assert again.decisions == record_set.decisions
        assert again.mode == record_set.mode

    def test_a_duplicate_decision_id_is_refused(self, feed):
        record = make_record(feed)
        with pytest.raises(DecisionRecordError, match="duplicate"):
            make_set([record, record])

    def test_a_source_not_matching_the_mode_is_refused(self, feed):
        record = make_record(feed, source=SOURCE_OPERATOR_SUPPLIED)
        with pytest.raises(DecisionRecordError, match="match the record set's mode"):
            make_set([record], mode=SOURCE_SYNTHETIC_FIXTURE)

    def test_only_an_llm_inference_set_is_genuine(self, feed):
        assert not make_set([make_record(feed)]).is_genuine_inference
        genuine = make_set(
            [make_record(feed, source=SOURCE_LLM_INFERENCE)], mode=SOURCE_LLM_INFERENCE
        )
        assert genuine.is_genuine_inference

    def test_save_and_load_round_trip(self, feed, tmp_path):
        record_set = make_set([make_record(feed)])
        path = record_set.save(tmp_path / "decisions.json")
        assert DecisionRecordSet.load(path).decisions == record_set.decisions

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(DecisionRecordError, match="no decision record file"):
            DecisionRecordSet.load(tmp_path / "nope.json")

    def test_a_malformed_file_is_refused(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(DecisionRecordError, match="not valid JSON"):
            DecisionRecordSet.load(path)


# ---------------------------------------------------------------------------
# Credential safety
# ---------------------------------------------------------------------------


class TestCredentialsNeverReachArtifacts:
    def test_a_credential_shaped_literal_is_refused(self, feed):
        with pytest.raises(CredentialLeakError, match="credential-shaped"):
            make_record(feed, reasoning_excerpt="the key is sk-" + "a" * 40)

    def test_redaction_scrubs_a_credential_shape(self):
        scrubbed = redact_reasoning("token sk-" + "a" * 40 + " end")
        assert "sk-" not in scrubbed
        assert "<redacted>" in scrubbed

    def test_redaction_bounds_the_excerpt(self):
        assert len(redact_reasoning("x" * 5000)) <= 2001

    def test_an_environment_secret_cannot_be_serialised(self, feed, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "supersecretvalue123")
        with pytest.raises(CredentialLeakError, match="OPENAI_API_KEY"):
            make_record(feed, reasoning_excerpt="leaking supersecretvalue123 here")

    def test_the_artifact_is_scanned_for_secrets(self, monkeypatch):
        # A credential in the environment must not end up in the artifact even
        # if it is not in a record field.
        from tradingagents.baseline.data import load_feed
        from tradingagents.baseline.protocol import DEFAULT_PROTOCOL as BASE

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value-1234567890")
        outcome = build_evaluation(
            record_set=None, feed=load_feed(BASE), include_reference_context=False
        )
        assert "sk-ant-secret-value" not in json.dumps(outcome.artifact)
        assert outcome.status == STATUS_BLOCKED


# ---------------------------------------------------------------------------
# Signal reconstruction
# ---------------------------------------------------------------------------


class TestSignalFromRecord:
    def test_a_buy_record_builds_a_buy_signal(self, feed):
        signal = signal_from_record(make_record(feed, action="BUY", rating="Buy"))
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.action.value == "BUY"
        assert signal.as_of_date == "2022-03-06"

    def test_a_hold_record_builds_a_hold_signal(self, feed):
        signal = signal_from_record(
            make_record(feed, action="HOLD", rating="Hold", position_size=None)
        )
        assert signal is not None
        assert signal.action.value == "HOLD"

    def test_a_review_record_builds_no_signal(self, feed):
        # A decision nobody can read is not a trade, exactly as in production.
        record = make_record(
            feed, action="HOLD", rating="REVIEW", position_size=None, stop_loss=None
        )
        assert signal_from_record(record) is None

    def test_a_hold_record_may_not_request_a_position_size(self, feed):
        # The signal contract refuses this, so the record must too.
        with pytest.raises(DecisionRecordError, match="HOLD record"):
            make_record(feed, action="HOLD", rating="Hold")

    def test_an_action_rating_mismatch_is_refused(self, feed):
        with pytest.raises(DecisionRecordError, match="does not match rating"):
            make_record(feed, action="BUY", rating="Sell")

    def test_the_signal_carries_the_record_timestamp(self, feed):
        signal = signal_from_record(make_record(feed, session="2022-03-07"))
        assert signal.timestamp.date().isoformat() == "2022-03-07"


# ---------------------------------------------------------------------------
# Replay: no model, deterministic, honest
# ---------------------------------------------------------------------------


class TestReplayNeverCallsAModel:
    def test_a_model_call_inside_a_replay_raises(self):
        with no_llm_calls(), pytest.raises(ReplayError, match="must not create an LLM client"):
            from tradingagents.llm_clients import factory

            factory.create_llm_client("openai", "gpt-4o-mini")

    def test_the_factory_is_restored_afterwards(self):
        from tradingagents.llm_clients import factory

        original = factory.create_llm_client
        with no_llm_calls():
            pass
        assert factory.create_llm_client is original


class TestReplayStrategy:
    def test_a_session_with_no_record_yields_no_signal(self, feed):
        record_set = make_set([make_record(feed, session="2022-03-06")])
        strategy = ReplayStrategy(record_set)
        from tradingagents.backtesting import SessionClock, StrategyContext
        from tradingagents.portfolio.ledger import Ledger

        context = StrategyContext(
            session="2022-03-07",
            feed=feed,
            ledger=Ledger(initial_cash=100_000.0),
            clock=SessionClock.for_session("2022-03-07"),
            symbols=feed.symbols,
        )
        assert strategy(context) is None

    def test_two_decisions_in_one_session_are_refused(self, feed):
        # The engine takes one signal per session; picking among several would
        # invent a priority the graph does not have.
        records = [
            make_record(feed, session="2022-03-06", ticker="AAPL"),
            make_record(feed, session="2022-03-06", ticker="AAPL", decision_id="other"),
        ]
        strategy = ReplayStrategy(make_set(records))
        from tradingagents.backtesting import SessionClock, StrategyContext
        from tradingagents.portfolio.ledger import Ledger

        context = StrategyContext(
            session="2022-03-06",
            feed=feed,
            ledger=Ledger(initial_cash=100_000.0),
            clock=SessionClock.for_session("2022-03-06"),
            symbols=feed.symbols,
        )
        with pytest.raises(ReplayError, match="one signal per session"):
            strategy(context)

    def test_a_window_hash_mismatch_is_refused(self, feed):
        # A record that claims a window the replay cannot reproduce is a fiction.
        record = make_record(feed, config_hash=LLM_PROTOCOL.config_hash())
        tampered = DecisionRecord(**{**record.__dict__, "data_window_hash": "0" * 64})
        strategy = ReplayStrategy(make_set([tampered]))
        from tradingagents.backtesting import SessionClock, StrategyContext
        from tradingagents.portfolio.ledger import Ledger

        context = StrategyContext(
            session="2022-03-06",
            feed=feed,
            ledger=Ledger(initial_cash=100_000.0),
            clock=SessionClock.for_session("2022-03-06"),
            symbols=feed.symbols,
        )
        with pytest.raises(ReplayError, match="window mismatch"):
            strategy(context)

    def test_the_window_hash_is_stable_for_the_same_data(self, feed):
        assert window_hash(feed.history("AAPL", "2022-03-06")) == window_hash(
            feed.history("AAPL", "2022-03-06")
        )

    def test_the_window_hash_differs_for_different_data(self, feed):
        other = MarketDataFeed({"AAPL": _frame([c * 2 for c in CLOSES])})
        assert window_hash(feed.history("AAPL", "2022-03-06")) != window_hash(
            other.history("AAPL", "2022-03-06")
        )


# ---------------------------------------------------------------------------
# Replay through the real engine: determinism, no duplicate fills
# ---------------------------------------------------------------------------


def _run_replay(feed, records, *, mode=SOURCE_SYNTHETIC_FIXTURE, strategy_id="fixture"):
    from tradingagents.baseline import harness

    record_set = make_set(records, mode=mode, strategy_id=strategy_id)
    strategy = ReplayStrategy(record_set)
    window = harness.evaluate_window(feed, LLM_PROTOCOL, strategy, "test", SESSIONS[0], SESSIONS[-1])
    return window, strategy


def _semantic_snapshot(window, strategy) -> dict:
    """The fields two runs must agree on. Timestamps and run ids excluded.

    ``result.trades`` holds *closed* round-trips, so a single entry that the
    protocol still holds open at the end of the window is not a trade yet. The
    ledger is the source of truth for what is actually held, so the snapshot
    reads position quantity from there.
    """
    m = window.metrics
    ledger = window.result.ledger
    return {
        "final_equity": round(m.final_equity, 6),
        "total_return": round(m.total_return, 8),
        "total_fees": round(m.total_fees, 8),
        "num_trades": m.num_trades,
        "realized_pnl": round(m.realized_pnl, 6),
        "unrealized_pnl": round(m.unrealized_pnl, 6),
        "sessions": m.sessions,
        "position_quantity": round(ledger.position_quantity("AAPL"), 6),
        "cash": round(ledger.cash, 6),
        "signals_emitted": strategy.stats.signals_emitted,
        "cross_check": window.cross_check_passes,
    }


class TestReplayThroughTheEngine:
    def test_a_replayed_buy_travels_the_risk_and_execution_path(self, feed):
        records = [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        window, strategy = _run_replay(feed, records)
        assert strategy.stats.signals_emitted == 1
        # The engine scheduled exactly one signal and the ledger holds the fill.
        assert len(window.result.signals) == 1
        assert window.result.ledger.position_quantity("AAPL") > 0
        assert window.cross_check_passes

    def test_a_replay_is_deterministic(self, feed):
        records = [
            make_record(feed, session="2022-03-06", action="BUY", rating="Buy"),
            make_record(feed, session="2022-03-08", action="BUY", rating="Overweight"),
        ]
        first, s1 = _run_replay(feed, records)
        second, s2 = _run_replay(feed, records)
        assert _semantic_snapshot(first, s1) == _semantic_snapshot(second, s2)

    def test_the_same_order_is_not_filled_twice(self, feed):
        # One BUY record produces one order and one fill, never two.
        records = [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        window, _ = _run_replay(feed, records)
        ledger = window.result.ledger
        filled = [e for e in ledger.events if e.type.value == "ORDER_FILLED"]
        fees = [e for e in ledger.events if e.type.value == "FEE_CHARGED"]
        assert len(filled) == 1
        # The fee is charged once, on the one fill.
        assert len(fees) == 1
        # The position equals the single fill quantity, not twice it.
        assert ledger.position_quantity("AAPL") == pytest.approx(
            filled[0].detail["quantity"]
        )

    def test_an_unknown_symbol_is_a_rejection_not_a_crash(self, feed):
        # A record naming a ticker outside the feed must be rejected by the
        # engine's own guard, not silently dropped. The record is built without
        # asking the feed for a ZZZZ window, since the feed has none.
        base = make_record(feed, session="2022-03-06")
        record = DecisionRecord(
            **{
                **base.__dict__,
                "decision_id": "ZZZZ-2022-03-06",
                "ticker": "ZZZZ",
                "data_window_hash": "0" * 64,
                "bar_count": 0,
            }
        )
        window, _ = _run_replay(feed, [record])
        assert any("UNKNOWN_SYMBOL" in reason for _, _, reason in window.result.rejections)

    def test_a_hold_record_produces_no_order(self, feed):
        records = [
            make_record(feed, session="2022-03-06", action="HOLD", rating="Hold",
                        position_size=None)
        ]
        window, strategy = _run_replay(feed, records)
        assert strategy.stats.signals_emitted == 1
        assert window.result.ledger.position_quantity("AAPL") == 0
        assert len(window.result.trades) == 0

    def test_the_result_does_not_claim_to_be_genuine_inference(self, feed):
        records = [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        record_set = make_set(records)
        assert not record_set.is_genuine_inference


# ---------------------------------------------------------------------------
# Evaluation status: never fabricate a result
# ---------------------------------------------------------------------------


class TestEvaluationStatus:
    def test_no_credentials_and_no_records_is_blocked(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        assert outcome.status == STATUS_BLOCKED
        assert outcome.is_blocked
        assert outcome.artifact["llm_results"] is None
        assert outcome.artifact["genuine_inference"] is False

    def test_the_blocker_names_the_missing_credential(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        import os
        print("DBGVAL", os.environ.get("OPENAI_API_KEY"))
        assert any("credential" in b for b in outcome.blockers)

    def test_a_fixture_record_set_is_never_genuine_inference(self, feed):
        records = [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        outcome = build_evaluation(
            record_set=make_set(records), feed=feed, include_reference_context=False
        )
        assert outcome.status == STATUS_ESTABLISHED_WITH_FINDINGS
        assert outcome.artifact["genuine_inference"] is False
        assert any("fixture" in b for b in outcome.blockers)

    def test_an_llm_inference_record_set_is_marked_genuine(self, feed):
        records = [
            make_record(feed, session="2022-03-06", action="BUY", rating="Buy",
                        source=SOURCE_LLM_INFERENCE)
        ]
        outcome = build_evaluation(
            record_set=make_set(records, mode=SOURCE_LLM_INFERENCE),
            feed=feed,
            include_reference_context=False,
        )
        assert outcome.artifact["genuine_inference"] is True

    def test_credential_status_never_leaks_a_value(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-abcdefghijklmnopqrstuvwx")
        status = credential_status()
        assert "openai" in status["providers_with_credentials"]
        assert "sk-live" not in json.dumps(status)

    def test_a_placeholder_key_does_not_count_as_a_credential(self, monkeypatch):
        # The test suite's autouse fixture sets every provider key to the literal
        # "placeholder". If that counted, the evaluation would claim a runnable
        # LLM strategy in an environment with no usable key.
        monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        status = credential_status()
        assert status["any_credential_available"] is False
        assert status["providers_with_credentials"] == []

    def test_a_plausible_key_is_accepted(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdefghijklmnopqrstuvwx")
        status = credential_status()
        assert "anthropic" in status["providers_with_credentials"]
        assert status["any_credential_available"] is True

    def test_write_evaluation_produces_valid_json(self, feed, tmp_path):
        from tradingagents.llm_evaluation import write_evaluation

        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        path = write_evaluation(outcome, tmp_path / "eval.json")
        payload = json.loads(path.read_text())
        assert payload["status"] == STATUS_BLOCKED
        assert payload["artifact_schema_version"]


# ---------------------------------------------------------------------------
# Point-in-time: the replay cannot see a future bar
# ---------------------------------------------------------------------------


class TestPointInTime:
    def test_the_replay_window_never_exceeds_the_session(self, feed):
        # A record's window is bounded at its own as_of_date; the replay
        # recomputes the same bound from the session-bounded feed.
        record = make_record(feed, session="2022-03-06")
        frame = feed.history("AAPL", "2022-03-06")
        assert len(frame) == record.bar_count
        assert frame.index.max() <= pd.Timestamp("2022-03-06")

    def test_a_record_whose_window_runs_past_its_date_is_refused(self, feed):
        # A window ending after the decision date means the model saw a later
        # bar. That is the look-ahead this layer exists to prevent, so refuse it.
        with pytest.raises(DecisionRecordError, match="window_end must equal as_of_date"):
            make_record(feed, session="2022-03-06", window_end="2022-03-10")

    def test_a_future_dated_window_is_rejected_by_the_feed(self, feed):
        # The feed itself cannot return a bar after the session, so a replay
        # asking for one gets nothing rather than tomorrow's price.
        assert feed.bar("AAPL", "2027-01-01") is None

    def test_the_protocol_refuses_a_full_scope_without_pit_data(self):
        from tradingagents.llm_evaluation import LLMEvaluationProtocol

        with pytest.raises(ValueError, match="point-in-time"):
            LLMEvaluationProtocol(input_scope="full")


# ---------------------------------------------------------------------------
# Artifact determinism and the command-line entry point
# ---------------------------------------------------------------------------


class TestArtifactDeterminism:
    def test_two_replays_of_the_same_records_agree(self, feed):
        # Everything except the wall-clock stamp must be identical between two
        # builds of the same records: the result is a property of the input, not
        # of when it was computed.
        records = [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        record_set = make_set(records)
        first = build_evaluation(
            record_set=record_set, feed=feed, include_reference_context=False
        )
        second = build_evaluation(
            record_set=record_set, feed=feed, include_reference_context=False
        )
        a, b = dict(first.artifact), dict(second.artifact)
        a.pop("generated_at"), b.pop("generated_at")
        assert a == b

    def test_the_artifact_records_the_protocol_and_checksums(self, feed):
        outcome = build_evaluation(
            record_set=None, feed=feed, include_reference_context=False
        )
        artifact = outcome.artifact
        assert artifact["protocol"]["version"] == LLM_PROTOCOL.version
        assert artifact["dataset_checksums"] == artifact["protocol"]["dataset_checksums"]


class TestCommandLine:
    def test_main_writes_a_blocked_artifact_with_no_records(self, tmp_path):
        from tradingagents.llm_evaluation.evaluate import main

        out = tmp_path / "eval.json"
        code = main(["--out", str(out), "--no-reference-context"])
        assert code == 0
        payload = json.loads(out.read_text())
        assert payload["status"] == STATUS_BLOCKED
        assert payload["llm_results"] is None

    def test_main_replays_a_supplied_record_set(self, tmp_path, feed):
        from tradingagents.llm_evaluation.evaluate import main

        record_set = make_set(
            [make_record(feed, session="2022-03-06", action="BUY", rating="Buy")]
        )
        records_path = record_set.save(tmp_path / "records.json")
        out = tmp_path / "eval.json"
        code = main(
            ["--records", str(records_path), "--out", str(out), "--no-reference-context"],
            feed=feed,
        )
        assert code == 0
        payload = json.loads(out.read_text())
        assert payload["status"] == STATUS_ESTABLISHED_WITH_FINDINGS
        assert payload["genuine_inference"] is False
        assert payload["llm_results"] is not None
