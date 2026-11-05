"""The TJR validation harness: it measures the strategy, it does not change it.

These tests pin the harness's own contract: that it runs under the paper account's
caps, that its evaluation cache is exact (a cached run equals a cold one), that the
chronological split is a real partition, and that the look-ahead probe would catch a
decision that changed when later bars appeared. The strategy rules are pinned
elsewhere (``test_tjr_rules.py``); nothing here relaxes them.

Every frame is hand-built, so no test touches the network. Real frames are fetched
only by ``tradingagents.tjr.fetch_data`` and read by the study CLI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tjr_helpers import BULLISH_BARS, SESSION, frame_from_bars, one_minute_from_five
from tradingagents.execution.costs import CostModel
from tradingagents.risk import paper_parity_limits
from tradingagents.tjr.frames import session_open_utc
from tradingagents.tjr.run import load_series
from tradingagents.tjr.validation import (
    DEFAULT_SCENARIOS,
    audit_frames,
    lookahead_probe,
    run_scenario,
    run_study,
    split_sessions,
)

pytestmark = pytest.mark.unit

# A second session, so a split has two halves to cut between.
SESSION_2 = "2026-10-30"

_WIN = BULLISH_BARS[:15] + [
    (100.6, 110.0, 100.55, 109.0),
    (109.0, 110.0, 108.0, 109.5),
]


def _write_frames(data_dir, symbols, sessions=(SESSION,)):
    for symbol in symbols:
        for session in sessions:
            five = frame_from_bars(session, _WIN)
            one = one_minute_from_five(five)
            suffix = "" if session == SESSION else f"-{session}"
            one.to_csv(data_dir / f"{symbol}-1m{suffix}.csv")
            five.to_csv(data_dir / f"{symbol}-5m{suffix}.csv")


def _write_multi_session(data_dir, symbols):
    """Two sessions in one file per symbol, the second shifted a day later."""
    for symbol in symbols:
        one_parts = []
        five_parts = []
        for session in (SESSION, SESSION_2):
            five = frame_from_bars(session, _WIN)
            one = one_minute_from_five(five)
            one_parts.append(one)
            five_parts.append(five)
        pd.concat(five_parts).to_csv(data_dir / f"{symbol}-5m.csv")
        pd.concat(one_parts).to_csv(data_dir / f"{symbol}-1m.csv")


# ---------------------------------------------------------------------------
# Risk parity: the caps are the paper account's, applied to the whole book
# ---------------------------------------------------------------------------


def test_the_study_uses_the_paper_parity_limits(tmp_path):
    _write_multi_session(tmp_path, ("A", "B"))
    payload = run_study(tmp_path, symbols=("A", "B"))
    limits = payload["risk_limits"]
    assert limits["source"] == "tradingagents.risk.paper_parity.paper_parity_limits()"
    assert limits["max_order_notional"] == 25.0
    assert limits["max_open_positions"] == 5
    assert limits["max_daily_orders"] == 10
    assert limits["allow_short"] is False
    assert limits["fractional_shares"] is True
    assert "portfolio" in limits["applied_to"]
    # The limits object itself is the shared one, not a local copy.
    assert paper_parity_limits().max_order_value == 25.0


# ---------------------------------------------------------------------------
# The cache is exact
# ---------------------------------------------------------------------------


def test_a_cached_run_equals_a_cold_run(tmp_path):
    _write_multi_session(tmp_path, ("A", "B"))
    series = load_series(tmp_path, ("A", "B"))
    costs = CostModel(commission_rate=0.0005, half_spread_bps=1.0, slippage_bps=2.0)
    cold = run_scenario(series, ("A", "B"), costs=costs, evaluation_cache={})
    shared: dict = {}
    run_scenario(series, ("A",), costs=costs, evaluation_cache=shared)  # warm the cache
    warm = run_scenario(series, ("A", "B"), costs=costs, evaluation_cache=shared)
    assert warm["trades_detail"] == cold["trades_detail"]
    assert warm["metrics"]["final_equity"] == cold["metrics"]["final_equity"]


# ---------------------------------------------------------------------------
# Cost scenarios
# ---------------------------------------------------------------------------


def test_costs_reduce_net_pnl_monotonically(tmp_path):
    _write_multi_session(tmp_path, ("A", "B"))
    series = load_series(tmp_path, ("A", "B"))
    cache: dict = {}
    nets = {}
    for scenario in DEFAULT_SCENARIOS:
        result = run_scenario(series, ("A", "B"), costs=scenario.costs, evaluation_cache=cache)
        nets[scenario.name] = result["net_pnl"]
    assert nets["zero"] >= nets["project"] >= nets["high"]


# ---------------------------------------------------------------------------
# The split is a real partition, and the view restricts to it
# ---------------------------------------------------------------------------


def test_split_sessions_is_chronological_and_disjoint():
    sessions = [f"2026-10-{d:02d}" for d in range(1, 11)]
    dev, test = split_sessions(sessions, fraction=0.5)
    assert dev + test == sorted(sessions)
    assert set(dev).isdisjoint(test)
    assert max(dev) < min(test)


def test_period_view_restricts_to_the_sessions(tmp_path):
    _write_multi_session(tmp_path, ("A",))
    payload = run_study(tmp_path, symbols=("A",), split_fraction=0.5)
    dev = payload["split"]["development"]
    test = payload["split"]["test"]
    assert dev["sessions"] + test["sessions"] == payload["window"]["sessions"]
    assert dev["end"] < test["start"]


# ---------------------------------------------------------------------------
# The data-quality audit reports what it sees
# ---------------------------------------------------------------------------


def test_the_audit_reports_sessions_and_agreement(tmp_path):
    _write_multi_session(tmp_path, ("A", "B"))
    audit = audit_frames(tmp_path, ("A", "B"))
    assert audit["symbols"]["A"]["present"] is True
    assert audit["symbols"]["A"]["one_minute"]["sessions"] == 2
    assert audit["symbols"]["A"]["five_minute"]["sessions"] == 2
    assert audit["symbols"]["A"]["session_sets_match"] is True
    assert audit["symbols"]["A"]["duplicate_timestamps"]["one_minute"] == 0
    assert audit["checks"]["session_alignment_across_symbols"] is True
    # The 5m bars are built from the same 1m bars here, so they agree exactly.
    compared = audit["checks"]["one_minute_vs_five_minute"]["A"]["compared_bars"]
    assert compared > 0


# ---------------------------------------------------------------------------
# Look-ahead probe
# ---------------------------------------------------------------------------


def test_the_lookahead_probe_finds_no_change_when_later_bars_are_dropped(tmp_path):
    _write_frames(tmp_path, ("A",))
    cut = (session_open_utc(SESSION) + pd.Timedelta(hours=3)).isoformat()
    probe = lookahead_probe(tmp_path, symbol="A", session=SESSION, cut=cut)
    assert probe["identical"] is True
    assert probe["full_entries_at_or_before_cut"] == probe["truncated_entries_at_or_before_cut"]


def test_the_lookahead_probe_detects_a_decision_that_needs_a_later_bar(tmp_path):
    # A cut before the setup completes: the full run still enters later, but no
    # entry is attributed to the truncated feed, so the probe shows the decision
    # did not exist yet rather than inventing one.
    _write_frames(tmp_path, ("A",))
    early = (session_open_utc(SESSION) + pd.Timedelta(minutes=30)).isoformat()
    probe = lookahead_probe(tmp_path, symbol="A", session=SESSION, cut=early)
    assert probe["full_entries_total"] > 0  # the setup does complete later in the day
    assert probe["full_entries_at_or_before_cut"] == 0
    assert probe["truncated_entries_at_or_before_cut"] == 0
    assert probe["identical"] is True


# ---------------------------------------------------------------------------
# The study payload is complete
# ---------------------------------------------------------------------------


def test_the_lookahead_probe_is_exact_over_a_multi_session_feed(tmp_path):
    # Two sessions, so the truncated feed still has the 1h/4h level sources it had
    # before the cut. If truncation dropped those sources, a decision would change
    # and the probe would report a false positive; this pins that it does not.
    _write_multi_session(tmp_path, ("A",))
    cut = (session_open_utc(SESSION_2) + pd.Timedelta(hours=3)).isoformat()
    probe = lookahead_probe(tmp_path, symbol="A", session=SESSION_2, cut=cut)
    assert probe["full_entries_at_or_before_cut"] > 0
    assert probe["identical"] is True


def test_the_study_reports_every_section(tmp_path):
    _write_multi_session(tmp_path, ("A", "B"))
    payload = run_study(tmp_path, symbols=("A", "B"))
    assert set(payload["portfolio"]) == {s.name for s in DEFAULT_SCENARIOS}
    assert set(payload["per_symbol_standalone"]) == {"A", "B"}
    assert payload["window"]["sessions"] == 2
    assert payload["data_audit"]["symbols"]["A"]["present"] is True
    assert payload["evaluation_cache_entries"] > 0
