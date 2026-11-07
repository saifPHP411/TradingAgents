"""Phase 9.5: reproducible price input validation for the smoke capture path.

Phase 9's first genuine Groq run returned a real decision but recorded
``bar_count: 0`` and ``data_snapshot_id: feed:unavailable``. The inference was
real; the *input* was not reproducible. These tests lock down the fix and, just
as importantly, the refusals:

- a session the frozen dataset covers yields a non-empty window that reaches the
  record, and the record names a snapshot that verifies against its own hash;
- a session the dataset does *not* cover yields no snapshot and a report that is
  explicitly not experiment-ready — a genuine inference is not enough;
- a snapshot round-trips through disk and its content hash is recomputed on read,
  so a tampered file is refused;
- the snapshot hash and the record's window hash are the *same* function, so a
  snapshot that does not match the record it claims to anchor is caught;
- the smoke CLI's default session is clamped to the frozen data, so the "today is
  past the dataset" bug that produced ``feed:unavailable`` cannot recur;
- a fixture (injected client) is never labelled ``llm_inference``, even when the
  snapshot is reproducible.

No test here builds an LLM client or reaches the network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.backtesting import MarketDataFeed
from tradingagents.llm_evaluation import (
    SOURCE_LLM_INFERENCE,
    DecisionRecordSet,
    PriceSnapshot,
    SnapshotError,
    build_record,
    smoke as smoke_mod,
    snapshot_for_session,
)
from tradingagents.signals import action_for_rating

SESSIONS = [f"2022-03-{d:02d}" for d in range(1, 15)]
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
    return MarketDataFeed({"SPY": _frame()})


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.llm = _FakeLLM(content)

    def get_llm(self):
        return self.llm


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return type("Response", (), {"content": self._content})()


def _fixture_factory(content: str):
    client = _FakeClient(content)
    return (lambda **kwargs: client), client


# ---------------------------------------------------------------------------
# Snapshot construction and content verification
# ---------------------------------------------------------------------------


class TestPriceSnapshot:
    def test_it_names_the_symbol_window_and_hash(self, feed):
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        assert snapshot.symbol == "SPY"
        assert snapshot.window_start == "2022-03-01"
        assert snapshot.window_end == "2022-03-06"
        assert snapshot.bar_count == 6
        assert snapshot.snapshot_id.startswith("spy-2022-03-01..2022-03-06-")
        # The id carries the first 12 characters of the content hash.
        assert snapshot.content_hash[:12] in snapshot.snapshot_id

    def test_the_hash_is_the_same_function_the_record_uses(self, feed):
        from tradingagents.llm_evaluation import window_hash

        frame = feed.history("SPY", "2022-03-06")
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        assert snapshot.content_hash == window_hash(frame)
        assert snapshot.recompute_hash() == snapshot.content_hash

    def test_an_empty_window_is_refused_not_snapshotted(self):
        empty = MarketDataFeed({"SPY": _frame()})
        with pytest.raises(SnapshotError):
            snapshot_for_session(empty, "SPY", "2000-01-01", source="test")

    def test_a_symbol_the_feed_lacks_is_refused(self, feed):
        with pytest.raises(SnapshotError):
            snapshot_for_session(feed, "TSLA", "2022-03-06", source="test")

    def test_a_window_that_does_not_end_at_the_session_is_refused(self, feed):
        # The frame's last bar is 2022-03-14, so a snapshot ending at 2022-03-20
        # cannot exist: the feed has no bar on the requested session.
        with pytest.raises(SnapshotError):
            snapshot_for_session(feed, "SPY", "2022-03-20", source="test")

    def test_it_round_trips_through_disk(self, feed, tmp_path):
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        path = snapshot.save(tmp_path / "spy.json")
        reloaded = PriceSnapshot.load(path)
        assert reloaded.snapshot_id == snapshot.snapshot_id
        assert reloaded.content_hash == snapshot.content_hash
        assert reloaded.bars == snapshot.bars

    def test_a_tampered_bar_fails_verification_on_load(self, feed, tmp_path):
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        path = snapshot.save(tmp_path / "spy.json")
        payload = json.loads(path.read_text())
        payload["bars"][0]["close"] = payload["bars"][0]["close"] + 1.0
        path.write_text(json.dumps(payload))
        with pytest.raises(SnapshotError):
            PriceSnapshot.load(path)

    def test_an_unsupported_schema_version_is_refused(self, feed, tmp_path):
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        path = snapshot.save(tmp_path / "spy.json")
        payload = json.loads(path.read_text())
        payload["schema_version"] = 999
        path.write_text(json.dumps(payload))
        with pytest.raises(SnapshotError):
            PriceSnapshot.load(path)

    def test_a_non_finite_bar_is_refused_at_construction(self):
        bad = _frame()
        bad.iloc[2, bad.columns.get_loc("close")] = float("nan")
        # The feed drops the unusable row at construction, so the snapshot sees a
        # shorter, valid window rather than a NaN bar.
        feed = MarketDataFeed({"SPY": bad})
        snapshot = snapshot_for_session(feed, "SPY", "2022-03-14", source="test")
        assert all(bar["close"] == bar["close"] for bar in snapshot.bars)


# ---------------------------------------------------------------------------
# The window reaches the record, and the record names the snapshot
# ---------------------------------------------------------------------------


class TestWindowReachesTheRecord:
    def test_a_covered_session_produces_a_non_empty_window(self, feed):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        record, reproducible = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=snapshot.to_frame(),
            protocol=DEFAULT_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
            snapshot_id=snapshot.snapshot_id,
        )
        assert reproducible is True
        assert record.bar_count == 6
        assert record.data_snapshot_id == snapshot.snapshot_id
        assert not record.data_snapshot_id.endswith("unavailable")
        assert snapshot.matches_record(record)

    def test_the_record_hash_equals_the_snapshot_hash(self, feed):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        record, _ = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=snapshot.to_frame(),
            protocol=DEFAULT_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
            snapshot_id=snapshot.snapshot_id,
        )
        assert record.data_window_hash == snapshot.content_hash

    def test_a_reloaded_snapshot_still_matches_the_record(self, feed, tmp_path):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        snapshot = snapshot_for_session(feed, "SPY", "2022-03-06", source="test")
        path = snapshot.save(tmp_path / "spy.json")
        reloaded = PriceSnapshot.load(path)
        record, _ = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=reloaded.to_frame(),
            protocol=DEFAULT_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
            snapshot_id=reloaded.snapshot_id,
        )
        assert reloaded.matches_record(record)

    def test_a_snapshot_from_a_different_window_does_not_match(self, feed):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        early = snapshot_for_session(feed, "SPY", "2022-03-03", source="test")
        record, _ = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=feed.history("SPY", "2022-03-06"),
            protocol=DEFAULT_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
        )
        assert not early.matches_record(record)


# ---------------------------------------------------------------------------
# Session resolution: the frozen-data clamp that fixes feed:unavailable
# ---------------------------------------------------------------------------


class TestSessionResolution:
    def test_the_default_session_is_the_last_frozen_session(self, feed):
        assert smoke_mod._resolve_session(feed, None) == "2022-03-14"

    def test_an_explicit_session_is_used_verbatim(self, feed):
        assert smoke_mod._resolve_session(feed, "2022-03-06") == "2022-03-06"

    def test_a_missing_feed_falls_back_to_today(self):
        resolved = smoke_mod._resolve_session(None, None)
        assert len(resolved) == 10 and resolved.count("-") == 2

    def test_the_production_default_lands_inside_the_frozen_dataset(self):
        """The bug that produced feed:unavailable was a default past the data.

        Resolving the real frozen feed's default session must yield a date the
        feed actually covers, so the window can end at the decision date.
        """
        from tradingagents.baseline.data import load_feed
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        real = load_feed(DEFAULT_PROTOCOL)
        session = smoke_mod._resolve_session(real, None)
        assert session in real.calendar()
        assert len(real.history("SPY", session)) > 0


# ---------------------------------------------------------------------------
# End-to-end through run_smoke, with an injected client (offline)
# ---------------------------------------------------------------------------


class TestSmokeSnapshotIntegration:
    def test_a_covered_session_reports_experiment_ready(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, client = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        report = smoke_mod.run_smoke(
            provider="groq", feed=feed, client_factory=factory, session="2022-03-06"
        )
        assert report.succeeded is True
        assert report.window_reproducible is True
        assert report.experiment_ready is True
        assert report.data_snapshot_id is not None
        assert not report.data_snapshot_id.endswith("unavailable")
        assert client.llm.calls == 1

    def test_a_fixture_with_a_window_is_still_not_genuine(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, _ = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        report = smoke_mod.run_smoke(
            provider="groq", feed=feed, client_factory=factory, session="2022-03-06"
        )
        assert report.genuine is False
        assert report.mode == "MOCK_TEST"

    def test_the_written_snapshot_verifies_and_matches_the_record(
        self, monkeypatch, feed, tmp_path
    ):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, _ = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        snap_path = tmp_path / "spy.json"
        rec_path = tmp_path / "records.json"
        report = smoke_mod.run_smoke(
            provider="groq",
            feed=feed,
            client_factory=factory,
            session="2022-03-06",
            snapshot_out=snap_path,
            record_out=rec_path,
        )
        snapshot = PriceSnapshot.load(snap_path)
        record_set = DecisionRecordSet.load(rec_path)
        record = record_set.decisions[0]
        assert snapshot.bar_count == 6
        assert snapshot.matches_record(record)
        assert record.data_snapshot_id == snapshot.snapshot_id
        assert report.data_snapshot_path == str(snap_path)

    def test_a_session_past_the_data_is_not_experiment_ready(self, monkeypatch, feed):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        factory, _ = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        report = smoke_mod.run_smoke(
            provider="groq", feed=feed, client_factory=factory, session="2022-03-20"
        )
        assert report.succeeded is True
        assert report.window_reproducible is False
        assert report.experiment_ready is False
        assert any("cannot anchor" in n for n in report.notes)

    def test_no_feed_means_not_experiment_ready(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "g" * 32)
        monkeypatch.setattr(smoke_mod, "_resolve_feed", lambda feed: (None, "no feed"))
        factory, _ = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        report = smoke_mod.run_smoke(
            provider="groq", client_factory=factory, session="2022-03-06"
        )
        assert report.succeeded is True
        assert report.experiment_ready is False
        assert report.data_snapshot_id is None

    def test_the_report_is_secret_free_and_names_no_credential(self, monkeypatch, feed):
        secret = "g" * 40
        monkeypatch.setenv("GROQ_API_KEY", secret)
        factory, _ = _fixture_factory("**Rating**: Buy\n**Executive Summary**: go")
        report = smoke_mod.run_smoke(
            provider="groq", feed=feed, client_factory=factory, session="2022-03-06"
        )
        blob = json.dumps(report.to_dict())
        assert secret not in blob


# ---------------------------------------------------------------------------
# The honest gate: absence of data must block an experiment claim
# ---------------------------------------------------------------------------


class TestExperimentGate:
    def test_a_record_with_no_window_is_not_reproducible(self):
        from tradingagents.llm_evaluation import DEFAULT_PROTOCOL

        record, reproducible = build_record(
            ticker="SPY",
            session="2022-03-06",
            rating="Buy",
            action=action_for_rating("Buy"),
            frame=None,
            protocol=DEFAULT_PROTOCOL,
            source=SOURCE_LLM_INFERENCE,
            provider="groq",
            model_name="openai/gpt-oss-20b",
        )
        assert reproducible is False
        assert record.bar_count == 0
        assert record.data_snapshot_id.endswith("unavailable")

    def test_snapshot_for_session_refuses_when_the_symbol_is_absent(self, feed):
        with pytest.raises(SnapshotError):
            snapshot_for_session(feed, "NVDA", "2022-03-06", source="test")

    def test_snapshot_for_session_refuses_without_a_feed(self):
        with pytest.raises(SnapshotError):
            snapshot_for_session(None, "SPY", "2022-03-06", source="test")


# ---------------------------------------------------------------------------
# CLI: the snapshot command is offline and verifies on demand
# ---------------------------------------------------------------------------


class TestSnapshotCli:
    def test_it_builds_and_verifies_a_snapshot_offline(self, tmp_path, capsys):
        out = tmp_path / "spy.json"
        code = smoke_mod_snapshot_main(["--symbol", "SPY", "--session", "2024-12-30", "--out", str(out)])
        assert code == 0
        assert out.exists()
        snapshot = PriceSnapshot.load(out)
        assert snapshot.bar_count > 1000
        assert snapshot.window_end == "2024-12-30"

        code = smoke_mod_snapshot_main(["--verify", str(out)])
        assert code == 0
        assert "OK" in capsys.readouterr().out

    def test_a_tampered_snapshot_fails_cli_verification(self, tmp_path):
        out = tmp_path / "spy.json"
        smoke_mod_snapshot_main(["--symbol", "SPY", "--session", "2024-12-30", "--out", str(out)])
        payload = json.loads(out.read_text())
        payload["bars"][0]["close"] += 5.0
        out.write_text(json.dumps(payload))
        with pytest.raises(SnapshotError):
            smoke_mod_snapshot_main(["--verify", str(out)])


def smoke_mod_snapshot_main(argv):
    from tradingagents.llm_evaluation.snapshot import _main

    return _main(argv)
