"""The TJR CLI paths, exercised offline on frames written to a temporary directory.

The fetch is the only network path and is not called here — the suite refuses
network. What these tests pin is the offline half: that ``run`` reads exactly the
frames a fetch would have written, resamples the 5m series from the 1m one when no
5m file exists, and produces a report whose numbers are the ledger's.
"""

from __future__ import annotations

import json

import pytest

from tests.tjr_helpers import BULLISH_BARS, SESSION, frame_from_bars, one_minute_from_five
from tradingagents.tjr.run import build_run, main

pytestmark = pytest.mark.unit


def _write_frames(data_dir, symbols, bars):
    five = frame_from_bars(SESSION, bars)
    one = one_minute_from_five(five)
    for symbol in symbols:
        one.to_csv(data_dir / f"{symbol}-1m.csv")
        five.to_csv(data_dir / f"{symbol}-5m.csv")
    return five, one


def test_build_run_reads_the_frames_and_returns_a_report(tmp_path):
    _write_frames(tmp_path, ("X",), BULLISH_BARS[:15] + [(100.6, 110.0, 100.55, 109.0), (109.0, 110.0, 108.0, 109.5)])
    payload = build_run(tmp_path, ("X",))
    assert payload["sessions"] == 1
    assert payload["symbols"] == ["X"]
    assert payload["trades"]
    assert payload["trades"][0]["reason_code"] == "READY"


def test_build_run_resamples_5m_from_1m_when_no_5m_file_exists(tmp_path):
    five = frame_from_bars(SESSION, BULLISH_BARS)
    one_minute_from_five(five).to_csv(tmp_path / "X-1m.csv")
    # No X-5m.parquet: the run must build it from the 1m series, not fail.
    payload = build_run(tmp_path, ("X",))
    assert payload["sessions"] == 1


def test_build_run_refuses_a_missing_frame(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_run(tmp_path, ("X",))


def test_the_run_cli_writes_json(tmp_path, capsys):
    _write_frames(tmp_path, ("X",), BULLISH_BARS)
    out = tmp_path / "report.json"
    code = main(["--symbols", "X", "--data", str(tmp_path), "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["symbols"] == ["X"]
    assert "reason_counts" in payload


def test_a_flat_market_produces_a_report_with_no_trades(tmp_path):
    flat = [
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 100.3, 100.0, 100.2),
        (100.2, 100.4, 100.1, 100.3),
        (100.3, 100.5, 100.2, 100.4),
    ]
    _write_frames(tmp_path, ("X",), flat)
    payload = build_run(tmp_path, ("X",))
    assert payload["trades"] == []
    assert payload["reason_counts"]
    # Undefined metrics are None, not 0.0 — a flat run has no Sharpe ratio.
    assert payload["metrics"]["sharpe_ratio"] is None
