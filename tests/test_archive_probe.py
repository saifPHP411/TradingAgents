"""Offline tests for real-source ZIP prefix probing."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, date, datetime

import pytest

from tradingagents.crypto.backtest.archive_probe import inspect_zip_prefix


def zip_prefix(*, symbol="XRPUSDT", kind="snapshot", method=zipfile.ZIP_DEFLATED):
    timestamp = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000)
    event = {
        "ts": timestamp, "type": kind,
        "data": {"s": symbol, "b": [["1.01", "12"]], "a": [["1.02", "11"]]},
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=method) as bundle:
        bundle.writestr("2026-10-01_XRPUSDT_ob200.data", json.dumps(event) + "\n")
    return stream.getvalue()[:512 * 1024]


@pytest.mark.parametrize("method", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_probe_accepts_real_zip_container_with_complete_jsonl_prefix(method):
    report = inspect_zip_prefix(zip_prefix(method=method), symbol="XRPUSDT", day=date(2026, 10, 1))
    assert report["sampled_complete_records"] == 1
    assert report["sampled_snapshots"] == 1
    assert report["symbol"] == "XRPUSDT"
    assert report["first_ts_ms"] == 1790812800000


def test_probe_rejects_incorrect_symbol_and_date():
    with pytest.raises(ValueError, match="symbol mismatch"):
        inspect_zip_prefix(zip_prefix(), symbol="LTCUSDT", day=date(2026, 10, 1))
    with pytest.raises(ValueError, match="date differs"):
        inspect_zip_prefix(zip_prefix(), symbol="XRPUSDT", day=date(2026, 10, 2))


def test_probe_rejects_html_error_as_zip():
    with pytest.raises(ValueError, match="ZIP"):
        inspect_zip_prefix(b"<html>forbidden</html>", symbol="XRPUSDT", day=date(2026, 10, 1))


def test_probe_rejects_partial_header():
    with pytest.raises(ValueError, match="ZIP"):
        inspect_zip_prefix(zip_prefix()[:10], symbol="XRPUSDT", day=date(2026, 10, 1))
