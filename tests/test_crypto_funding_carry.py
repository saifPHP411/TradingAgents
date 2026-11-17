import csv
import io
import json
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
import requests

from tradingagents.crypto.backtest import funding_carry
from tradingagents.crypto.backtest.funding_carry import (
    _archive_funding_history,
    _archive_futures_klines,
    _get_json,
    _parse_funding_archive,
    _parse_futures_kline_archive,
    run_study,
    simulate_symbol,
)


def _fixture(next_rate=0.001, basis_change=0.0):
    t0 = 1_800_000_000_000
    t1 = t0 + 8 * 3_600_000
    t2 = t1 + 8 * 3_600_000
    return {
        "spot_1h": {
            str(t0 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t1 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t2 + 3_600_000): {"open": 100.0, "close": 100.0},
        },
        "perp_1h": {
            str(t0 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t1 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t2 + 3_600_000): {"open": 100.0 - basis_change, "close": 100.0},
        },
        "funding": [
            {"timestamp": t0, "rate": 0.001},
            {"timestamp": t1, "rate": next_rate},
            {"timestamp": t2, "rate": next_rate},
        ],
    }


def _zip_klines(rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        content = io.StringIO()
        writer = csv.writer(content)
        writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)
        archive.writestr("BTCUSDT-1h.csv", content.getvalue())
    return buffer.getvalue()


def _zip_funding(rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        content = io.StringIO()
        writer = csv.writer(content)
        writer.writerow(["calc_time", "funding_interval_hours", "last_funding_rate"])
        writer.writerows(rows)
        archive.writestr("BTCUSDT-fundingRate.csv", content.getvalue())
    return buffer.getvalue()


class _Response:
    def __init__(self, *, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = ""
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_funding_carry_uses_lagged_signal_and_next_realized_payment():
    result = simulate_symbol(
        "BTCUSDT", _fixture(next_rate=0.001),
        annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trades"] == 1
    assert result["trade_log"][0]["observed_funding_rate"] == 0.001
    assert result["trade_log"][0]["realized_next_funding_rate"] == 0.001
    assert result["trade_log"][0]["funding_events_held"] == 2
    assert result["trade_log"][0]["funding_return_pct"] > 0


def test_funding_signal_does_not_use_current_future_funding_to_enter():
    data = _fixture(next_rate=0.01)
    data["funding"][0]["rate"] = 0.0
    result = simulate_symbol(
        "BTCUSDT", data, annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trades"] == 1
    assert result["trade_log"][0]["observed_funding_rate"] == 0.01


def test_funding_costs_charge_four_legs_and_can_reverse_small_carry():
    result = simulate_symbol(
        "BTCUSDT", _fixture(next_rate=0.0001),
        annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 10, "half_spread_bps_per_fill": 1,
               "slippage_bps_per_fill": 2},
    )
    # Consecutive qualifying settlements are one continuous hedge, not
    # separate trades that repeatedly pay entry/exit costs.
    assert result["trades"] == 1
    assert result["trade_log"][0]["cost_pct"] == 0.52
    assert result["trade_log"][0]["net_return_pct"] < 0


def test_funding_basis_change_is_included_in_pnl():
    data = _fixture(next_rate=0.001, basis_change=1.0)
    result = simulate_symbol(
        "BTCUSDT", data, annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trade_log"][0]["basis_return_pct"] == 1.0


def test_run_study_reports_realistic_and_conservative_and_no_demo():
    result = run_study({"data_source": "synthetic", "symbols": {"BTCUSDT": _fixture()}})
    assert set(result["scenarios"]) == {"realistic", "conservative"}
    assert result["decision"] == "research_only_no_demo_approval"


def test_get_json_tries_next_host_after_forbidden_response():
    session = Mock()
    session.get.side_effect = [
        _Response(status_code=403),
        _Response(payload=[{"ok": True}]),
    ]

    payload = _get_json(
        session, ("https://blocked.example", "https://mirror.example"),
        "/history", {"symbol": "BTCUSDT"},
    )

    assert payload == [{"ok": True}]
    assert session.get.call_count == 2


def test_get_json_reports_all_hosts_when_every_response_is_invalid():
    first = _Response(payload=ValueError("not JSON"))
    first.text = "<html>region blocked</html>"
    first.headers = {"Content-Type": "text/html"}
    second = _Response(status_code=451)
    session = Mock()
    session.get.side_effect = [first, second]

    with pytest.raises(RuntimeError) as error:
        _get_json(
            session, ("https://host-one.example", "https://host-two.example"),
            "/fapi/v1/klines", {},
        )

    message = str(error.value)
    assert "host-one.example" in message
    assert "host-two.example" in message
    assert "non-JSON response" in message
    assert "HTTP 451" in message


def test_parse_futures_archive_filters_range_and_skips_header():
    start = 1_800_000_000_000
    rows = [
        [start - 3_600_000, "99", "101", "98", "100", "1"],
        [start, "100", "102", "99", "101", "2"],
        [start + 3_600_000, "101", "103", "100", "102", "3"],
    ]

    parsed = _parse_futures_kline_archive(
        _zip_klines(rows), start_ms=start, end_ms=start + 3_600_000
    )

    assert parsed == {start: {"open": 100.0, "close": 101.0}}


def test_archive_fallback_loads_completed_month_and_daily_current_month():
    now = datetime(2026, 11, 17, 12, tzinfo=UTC)
    start_dt = datetime(2026, 10, 31, 0, tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    class ArchiveSession:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout):
            self.urls.append(url)
            filename = url.rsplit("/", 1)[-1]
            if filename.endswith("-2026-10.zip"):
                first = datetime(2026, 10, 31, 0, tzinfo=UTC)
                last = datetime(2026, 11, 1, 0, tzinfo=UTC)
            else:
                day = datetime.strptime(filename.removesuffix(".zip"), "BTCUSDT-1h-%Y-%m-%d")
                first = datetime(day.year, day.month, day.day, tzinfo=UTC)
                last = first + timedelta(days=1)
            rows = []
            cursor = first
            while cursor < last:
                timestamp = int(cursor.timestamp() * 1000)
                rows.append([timestamp, "100", "102", "99", "101", "1"])
                cursor += timedelta(hours=1)
            return _Response(content=_zip_klines(rows))

    session = ArchiveSession()
    history, metadata = _archive_futures_klines(
        session, "BTCUSDT", start_ms, end_ms, now=now
    )

    assert len(history) == 24 * 17
    assert metadata["source"] == "binance_public_data_archive"
    assert metadata["downloaded_files"] == 17
    assert any("/monthly/klines/" in url for url in session.urls)
    assert any("/daily/klines/" in url for url in session.urls)
    assert max(history) < int(datetime(2026, 11, 17, 0, tzinfo=UTC).timestamp() * 1000)


def test_main_writes_diagnostic_artifact_when_market_data_is_unavailable(
    monkeypatch, tmp_path
):
    output = tmp_path / "funding-carry.json"
    monkeypatch.setattr(
        funding_carry, "fetch_market_history",
        Mock(side_effect=RuntimeError("public futures endpoint blocked")),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["funding_carry.py", "--days", "180", "--out", str(output)],
    )

    assert funding_carry.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "data_unavailable"
    assert report["decision"] == "no_strategy_result_no_demo_approval"
    assert "public futures endpoint blocked" in report["error"]



def test_parse_funding_archive_reads_header_and_filters_requested_range():
    start = 1_800_000_000_000
    payload = _zip_funding([
        [start - 1, 8, 0.0001],
        [start, 8, 0.0002],
        [start + 8 * 3_600_000, 4, -0.0001],
    ])

    parsed = _parse_funding_archive(
        payload, start_ms=start, end_ms=start + 8 * 3_600_000
    )

    assert parsed == [
        {"timestamp": float(start), "rate": 0.0002, "interval_hours": 8.0}
    ]


def test_funding_archive_fallback_uses_only_completed_months():
    now = datetime(2026, 11, 17, 12, tzinfo=UTC)
    start = datetime(2026, 10, 1, 0, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    rows = []
    cursor = start
    archive_end = datetime(2026, 11, 1, 0, tzinfo=UTC)
    while cursor < archive_end:
        rows.append([int(cursor.timestamp() * 1000), 8, 0.0001])
        cursor += timedelta(hours=8)

    class FundingArchiveSession:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout):
            self.urls.append(url)
            return _Response(content=_zip_funding(rows))

    session = FundingArchiveSession()
    history, metadata = _archive_funding_history(
        session, "BTCUSDT", start_ms, end_ms, now=now
    )

    assert len(history) == 93
    assert metadata["source"] == "binance_public_data_archive_monthly"
    assert metadata["current_month_excluded"] is True
    assert metadata["downloaded_files"] == 1
    assert len(session.urls) == 1
    assert "/monthly/fundingRate/BTCUSDT/" in session.urls[0]
    assert max(int(item["timestamp"]) for item in history) < int(archive_end.timestamp() * 1000)


def test_run_study_preserves_actual_data_coverage_and_archive_sources():
    data = _fixture()
    data.update({
        "coverage": {
            "spot_bars": 12,
            "perp_bars": 12,
            "funding_events": 3,
            "spot_first": 1_800_000_000_000,
            "spot_last": 1_800_043_200_000,
            "perp_first": 1_800_000_000_000,
            "perp_last": 1_800_043_200_000,
            "funding_first": 1_800_000_000_000,
            "funding_last": 1_800_057_600_000,
        },
        "perp_1h_source": "binance_public_data_archive",
        "funding_source": "binance_public_data_archive_monthly",
        "perp_archive": {"downloaded_files": 2},
        "funding_archive": {"downloaded_files": 1, "current_month_excluded": True},
    })
    report = run_study({
        "data_source": "Binance public archive",
        "requested_start_utc": "2026-10-31T00:00:00+00:00",
        "requested_end_utc": "2026-11-17T00:00:00+00:00",
        "symbols": {"BTCUSDT": data},
    })

    coverage = report["data_coverage"]["BTCUSDT"]
    assert coverage["spot_hourly_bars"] == 12
    assert coverage["perpetual_hourly_bars"] == 12
    assert coverage["funding_events"] == 3
    assert coverage["perpetual_source"] == "binance_public_data_archive"
    assert coverage["funding_source"] == "binance_public_data_archive_monthly"
    assert coverage["funding_archive"]["current_month_excluded"] is True
    assert coverage["spot_first_utc"] == "2027-01-15T08:00:00+00:00"
