"""Phase 12: restart recovery of a *vendor-backed* paper experiment.

Phase 10 proved a restart is idempotent when the caller hands the restored
session the same price source it started with. That missed the failure this
phase exists to catch: a **vendor** source fetches once at construction, so a
resume that rebuilds the source from scratch either re-fetches (a second,
quota-consuming call — and on the free tier, a failure) or, worse, silently
resumes on a different dataset.

The fix persists a frame-backed source's validated bars and provenance in the
experiment state, so ``PaperExperiment.load(root)`` rebuilds the *same* source
without any network call. A source that cannot be persisted (a quote-only
source) is refused rather than swapped for the frozen dataset.

Every test here runs offline: the one vendor test drives a mocked HTTP
transport, so no socket is opened and no key is needed.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from tradingagents.paper_experiment import (
    ExperimentConfig,
    PaperExperiment,
    PriceSourceError,
    StaticPriceSource,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.providers import AlphaVantageDailySource
from tradingagents.paper_experiment.sources import (
    FramePriceSource,
    SourceProvenance,
    frame_to_records,
    records_to_frame,
)

TODAY = date(2024, 3, 1)


# ---------------------------------------------------------------------------
# Fixtures (kept local so this module is self-contained)
# ---------------------------------------------------------------------------


def _frame(periods: int = 24, *, start: float = 100.0, step: float = 1.0, volume: float = 1e6):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [volume] * periods,
        },
        index=index,
    )


def _config(**overrides):
    base = {
        "symbols": ("SPY",),
        "decision_frequency_sessions": 1,
        "evaluation_lag_sessions": 2,
    }
    base.update(overrides)
    return ExperimentConfig.from_protocol(**base)


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


def _daily_body(sessions):
    series = {}
    for i, session in enumerate(sessions):
        base = 100.0 + i
        series[session] = {
            "1. open": f"{base:.4f}",
            "2. high": f"{base + 2:.4f}",
            "3. low": f"{base - 1:.4f}",
            "4. close": f"{base + 1:.4f}",
            "5. volume": "1000000",
        }
    return json.dumps(
        {
            "Meta Data": {
                "2. Symbol": "IBM",
                "3. Last Refreshed": sessions[-1],
                "5. Time Zone": "US/Eastern",
            },
            "Time Series (Daily)": series,
        }
    )


def _counting_transport(body):
    calls = {"n": 0}

    def send(*, url, params, headers, timeout):
        calls["n"] += 1
        from tradingagents.paper_experiment.providers.base import HttpResponse

        return HttpResponse(200, body)

    send.calls = calls
    return send


# ---------------------------------------------------------------------------
# The state persists the source's bars, and nothing secret
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSourceIsPersistedInState:
    def test_state_carries_a_restorable_source_block(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame(periods=6)})
        experiment = PaperExperiment(
            source=source, config=_config(), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.save_state()
        state = json.loads((tmp_path / "exp" / "session_state.json").read_text())
        restore = state["source"]["restore"]
        assert restore["frames"]["SPY"], "the bars must be stored"
        assert restore["provenance"]["name"] == "static_test"

    def test_the_persisted_state_holds_no_credential(self, tmp_path, monkeypatch):
        # A credential in the environment must never reach the state file.
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "SECRET-VALUE-1234567890")
        source = StaticPriceSource({"SPY": _frame(periods=4)})
        experiment = PaperExperiment(
            source=source, config=_config(), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.process_session(source.sessions()[0])
        raw = (tmp_path / "exp" / "session_state.json").read_text()
        assert "SECRET-VALUE-1234567890" not in raw


# ---------------------------------------------------------------------------
# Restart WITHOUT handing load() a source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRestartWithoutExplicitSource:
    def _run(self, tmp_path, *, restart_at=None):
        source = StaticPriceSource({"SPY": _frame()})
        experiment = PaperExperiment(
            source=source, config=_config(), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        sessions = source.sessions()
        for index, session in enumerate(sessions):
            if restart_at is not None and index == restart_at:
                experiment.save_state()
                # No source= argument: the bars must come from the state.
                experiment = PaperExperiment.load(
                    tmp_path / "exp", inference=StubInference(rating="Buy")
                )
            experiment.process_session(session)
        return experiment, source

    def test_the_source_is_rebuilt_from_the_persisted_bars(self, tmp_path):
        source = StaticPriceSource({"SPY": _frame(periods=8)})
        experiment = PaperExperiment(
            source=source, config=_config(), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.process_session(source.sessions()[0])
        restored = PaperExperiment.load(tmp_path / "exp")
        assert restored.source.sessions() == source.sessions()
        assert restored.source.name == source.name

    def test_straight_and_restarted_runs_agree_on_accounting(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        assert _accounting(straight) == _accounting(restarted)

    def test_straight_and_restarted_runs_agree_on_events(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        assert _event_semantics(straight) == _event_semantics(restarted)

    def test_a_restart_without_a_source_does_not_duplicate_a_fill(self, tmp_path):
        straight, _ = self._run(tmp_path / "a")
        restarted, _ = self._run(tmp_path / "b", restart_at=10)
        straight_fills = [
            e for e in straight.session.ledger.events if e.type.value == "ORDER_FILLED"
        ]
        restarted_fills = [
            e for e in restarted.session.ledger.events if e.type.value == "ORDER_FILLED"
        ]
        assert len(straight_fills) == len(restarted_fills)

    def test_a_restart_without_a_source_does_not_duplicate_the_journal(self, tmp_path):
        _, _ = self._run(tmp_path / "b", restart_at=10)
        restarted = PaperExperiment.load(tmp_path / "b" / "exp")
        ids = [e.event_id for e in restarted.journal.entries]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# A vendor source restarts without a second fetch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVendorSourceRestart:
    def test_a_vendor_run_restarts_without_refetching(self, tmp_path):
        sessions = ["2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29"]
        transport = _counting_transport(_daily_body(sessions))
        source = AlphaVantageDailySource(
            "IBM", api_key="test-key-not-real", transport=transport, today=TODAY
        )
        assert transport.calls["n"] == 1, "the source fetches exactly once"

        experiment = PaperExperiment(
            source=source, config=_config(symbols=("IBM",)), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.process_session(sessions[0])
        experiment.save_state()

        restored = PaperExperiment.load(tmp_path / "exp", inference=StubInference(rating="Buy"))
        # No new HTTP call: the bars came from the state.
        assert transport.calls["n"] == 1
        assert restored.source.sessions() == source.sessions()
        assert restored.source.kind == "vendor_daily"

    def test_a_vendor_run_keeps_its_provenance_across_a_restart(self, tmp_path):
        sessions = ["2024-02-26", "2024-02-27", "2024-02-28"]
        source = AlphaVantageDailySource(
            "IBM", api_key="test-key-not-real",
            transport=_counting_transport(_daily_body(sessions)), today=TODAY,
        )
        experiment = PaperExperiment(
            source=source, config=_config(symbols=("IBM",)), root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        experiment.save_state()

        restored = PaperExperiment.load(tmp_path / "exp")
        # The bar is recent relative to TODAY, so the source stays current-market;
        # what matters is that the recorded provenance survives the restart.
        assert restored.source.is_current_market is True
        assert restored.source.provenance.last_bar_date == "2024-02-28"
        assert restored.source.provenance.kind == "vendor_daily"

    def test_a_vendor_restart_matches_a_straight_run(self, tmp_path):
        sessions = ["2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01"]

        def build(root):
            source = AlphaVantageDailySource(
                "IBM", api_key="test-key-not-real",
                transport=_counting_transport(_daily_body(sessions)), today=TODAY,
            )
            experiment = PaperExperiment(
                source=source, config=_config(symbols=("IBM",)), root=root,
                inference=StubInference(rating="Buy"),
            )
            experiment.ensure_created()
            return experiment, source

        straight, src_a = build(tmp_path / "a")
        for session in src_a.sessions():
            straight.process_session(session)

        restarted, src_b = build(tmp_path / "b")
        restarted.process_session(src_b.sessions()[0])
        restarted.save_state()
        restarted = PaperExperiment.load(tmp_path / "b", inference=StubInference(rating="Buy"))
        for session in src_b.sessions()[1:]:
            restarted.process_session(session)

        assert _accounting(straight) == _accounting(restarted)
        assert _event_semantics(straight) == _event_semantics(restarted)


# ---------------------------------------------------------------------------
# Multiple symbols
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiSymbolRestartWithoutSource:
    def test_multiple_symbols_survive_a_restart_without_a_source(self, tmp_path):
        frames = {"SPY": _frame(start=100.0), "QQQ": _frame(start=200.0)}
        source = StaticPriceSource(frames)
        config = _config(symbols=("SPY", "QQQ"))
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        sessions = source.sessions()
        for session in sessions[:6]:
            experiment.process_session(session)
        experiment.save_state()

        restored = PaperExperiment.load(tmp_path / "exp", inference=StubInference(rating="Buy"))
        assert set(restored.source.sessions()) == set(sessions)
        for session in sessions[6:]:
            restored.process_session(session)
        assert _accounting(restored)["positions"].keys() <= {"SPY", "QQQ"}


# ---------------------------------------------------------------------------
# Frame serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameSerialization:
    def test_a_frame_round_trips_through_records(self):
        frame = _frame(periods=5)
        rebuilt = records_to_frame(frame_to_records(frame))
        assert list(rebuilt.index) == list(frame.index)
        assert list(rebuilt["close"]) == list(frame["close"])

    def test_an_empty_record_list_is_refused(self):
        with pytest.raises(PriceSourceError):
            records_to_frame([])

    def test_a_corrupted_frame_is_refused_on_restore(self):
        payload = {
            "name": "corrupt",
            "kind": "static",
            "provenance": SourceProvenance(
                name="corrupt", kind="static", is_current_market=False
            ).to_dict(),
            "frames": {"SPY": [{"date": "2024-01-01", "open": float("nan"), "high": 3.0,
                                "low": 1.0, "close": 2.0, "volume": 1.0}]},
        }
        with pytest.raises(PriceSourceError):
            FramePriceSource.from_dict(payload)

    def test_a_source_without_frames_is_refused(self):
        with pytest.raises(PriceSourceError):
            FramePriceSource.from_dict({"name": "empty", "frames": {}})


# ---------------------------------------------------------------------------
# A source that cannot be persisted is refused, not swapped
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnpersistableSourceRefused:
    def test_load_refuses_when_the_state_has_no_restorable_bars(self, tmp_path):
        root = tmp_path / "exp"
        root.mkdir()
        state = {
            "schema_version": 1,
            "experiment_id": "x",
            "created_at": "2024-01-01T00:00:00+00:00",
            "config": _config().to_dict(),
            "config_hash": _config().config_hash(),
            "source": {"name": "finnhub_quote", "kind": "vendor_quote"},
            "processed_sessions": [],
            "decisions": {},
            "session": None,
        }
        (root / "session_state.json").write_text(json.dumps(state))
        from tradingagents.paper_experiment.experiment import ExperimentError

        with pytest.raises(ExperimentError, match="no restorable price bars"):
            PaperExperiment.load(root)


@pytest.mark.unit
class TestFrozenWindowGuard:
    """A frozen run reloads the committed dataset; a changed dataset is refused."""

    def _fake(self, sessions):
        class _Src:
            def sessions(self):
                return tuple(sessions)

        return _Src()

    def test_a_matching_window_is_accepted(self):
        from tradingagents.paper_experiment.__main__ import _check_frozen_window

        _check_frozen_window(
            self._fake(["2024-01-01", "2024-01-02"]),
            {"dataset_start": "2024-01-01", "dataset_end": "2024-01-02"},
        )

    def test_a_changed_window_is_refused(self):
        from tradingagents.paper_experiment.__main__ import RunnerError, _check_frozen_window

        with pytest.raises(RunnerError, match="different dataset"):
            _check_frozen_window(
                self._fake(["2024-01-01", "2024-06-30"]),
                {"dataset_start": "2024-01-01", "dataset_end": "2024-01-02"},
            )
