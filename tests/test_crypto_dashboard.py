"""The dashboard's crypto view: real backend data, and nothing else.

The dashboard is the system's control room, so these tests pin the properties
that make it trustworthy:

1. every number it shows comes from the backend — a fake engine, so the assertions
   are exact and no socket is touched;
2. a value the backend does not have is reported as ``None`` (the page renders
   "N/A"), never as a zero or a placeholder price;
3. a stale price is reported as **stale**, not as a current price;
4. a symbol with no TJR evaluation is ``NO SIGNAL``, not a BUY or a SELL;
5. the worker's balances come from the worker's own recorded status, so the
   dashboard never computes a P/L of its own;
6. no secret appears anywhere in the view.
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest

from tradingagents.paper_experiment.dashboard.crypto_view import (
    CRYPTO_VIEW_SCHEMA_VERSION,
    CryptoViewService,
)

SYMBOL = "BTC/USDT"
OTHER = "ETH/USDT"

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeEngine:
    """A crypto engine shaped like the real one, with a fixed snapshot."""

    def __init__(self, snapshots, *, describe=None, stream=None, error=None):
        self._snapshots = snapshots
        self._describe = describe or {
            "crypto_data_provider": "binance_public",
            "crypto_data_status": "ok",
            "crypto_data_last_refresh_at": NOW.isoformat(),
            "crypto_data_last_error": None,
            "crypto_data_requires_key": False,
        }
        self._stream = stream
        self._error = error

    def describe(self):
        return dict(self._describe)

    def stream_status(self):
        return self._stream or {"state": "stopped", "detail": "no stream is running"}

    def snapshots(self):
        if self._error is not None:
            raise RuntimeError(self._error)
        return dict(self._snapshots)


def _snapshot(symbol, *, price=100.0, fresh=True, age=1.0):
    return {
        "symbol": symbol,
        "price": price,
        "fresh": fresh,
        "age_seconds": age,
        "status": "ok" if fresh else "stale",
        "detail": None if fresh else "the newest event is 300s old",
        "ticker": {
            "bid": price - 0.5,
            "ask": price + 0.5,
            "spread": 1.0,
            "change_24h": 2.0,
            "change_percent_24h": 0.02,
            "volume_24h": 1_234.5,
            "high_24h": 110.0,
            "low_24h": 90.0,
        },
        "closed": {
            "1m": {
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": price,
                "volume": 5.0,
                "open_time": "2026-01-01T11:59:00+00:00",
                "close_time": "2026-01-01T11:59:59.999000+00:00",
            }
        },
        "forming": {
            "5m": {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
                "open_time": "2026-01-01T12:00:00+00:00",
                "close_time": "2026-01-01T12:04:59.999000+00:00",
            }
        },
        "newest_event_at": NOW.isoformat(),
    }


def _view(engine=None, *, symbols=(SYMBOL,), worker=None, evaluations=None):
    return CryptoViewService(
        engine=engine,
        symbols=symbols,
        worker_status=(lambda: worker) if worker is not None else None,
        strategy_evaluations=(lambda: evaluations) if evaluations is not None else None,
        clock=lambda: NOW,
    )


@pytest.mark.unit
class TestCryptoMarketView:
    def test_the_row_comes_from_the_engine(self):
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL, price=123.5)}))
        market = view.market()
        assert market["status"] == "ok"
        assert market["count"] == 1
        row = market["symbols"][0]
        assert row["symbol"] == SYMBOL
        assert row["price"] == 123.5
        assert row["bid"] == 123.0
        assert row["ask"] == 124.0
        assert row["spread"] == 1.0
        assert row["change_percent_24h"] == 0.02
        assert row["volume_24h"] == 1_234.5
        assert row["high_24h"] == 110.0
        assert row["low_24h"] == 90.0
        assert row["fresh"] is True

    def test_a_closed_candle_is_shown_and_a_forming_one_is_kept_apart(self):
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL)}))
        row = view.market()["symbols"][0]
        assert row["closed_candles"]["1m"]["close"] == 100.0
        # The forming bar is display-only: it is never in the closed map.
        assert "5m" not in row["closed_candles"]
        assert row["forming_candles"]["5m"]["close"] == 100.0

    def test_a_stale_price_is_reported_as_stale_not_as_current(self):
        engine = _FakeEngine({SYMBOL: _snapshot(SYMBOL, fresh=False, age=300.0)})
        view = _view(engine)
        market = view.market()
        assert market["status"] == "stale"
        row = market["symbols"][0]
        assert row["fresh"] is False
        assert row["status"] == "stale"
        assert row["age_seconds"] == 300.0

    def test_a_symbol_the_engine_does_not_know_is_no_data_not_zero(self):
        view = _view(_FakeEngine({}), symbols=(SYMBOL, OTHER))
        market = view.market()
        assert market["status"] == "no_data"
        for row in market["symbols"]:
            assert row["price"] is None
            assert row["status"] == "no_data"
            assert row["volume_24h"] is None

    def test_an_engine_failure_is_reported_as_an_error(self):
        view = _view(_FakeEngine({}, error="venue unreachable"))
        market = view.market()
        assert market["status"] == "error"
        assert "venue unreachable" in market["detail"]
        assert market["symbols"] == []

    def test_no_engine_reports_not_started(self):
        market = _view(None).market()
        assert market["status"] == "not_started"
        assert market["symbols"] == []

    def test_the_view_names_the_provider_and_whether_it_needs_a_key(self):
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL)}))
        described = view.describe()
        assert described["crypto_data_provider"] == "binance_public"
        assert described["crypto_data_requires_key"] is False
        assert described["crypto_data_stream"]["state"] == "stopped"


@pytest.mark.unit
class TestCryptoStrategyView:
    def test_a_completed_setup_shows_its_levels(self):
        evaluations = {
            SYMBOL: {
                "stage": "ENTRY",
                "reason_code": "TJR_ENTRY",
                "detail": "1m BOS confirmed the entry",
                "direction": "long",
                "entry": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "risk_reward": 2.0,
                "evaluated_at": NOW.isoformat(),
                "actionable": True,
            }
        }
        view = _view(_FakeEngine({}), evaluations=evaluations)
        strategy = view.strategy()
        row = strategy["symbols"][0]
        assert row["stage"] == "ENTRY"
        assert row["direction"] == "long"
        assert row["entry"] == 100.0
        assert row["stop_loss"] == 98.0
        assert row["take_profit"] == 104.0
        assert row["risk_reward"] == 2.0
        assert row["actionable"] is True

    def test_a_symbol_with_no_evaluation_is_no_signal(self):
        view = _view(_FakeEngine({}), evaluations={})
        row = view.strategy()["symbols"][0]
        assert row["stage"] == "NO_SIGNAL"
        assert row["actionable"] is False
        assert row["entry"] is None
        assert row["stop_loss"] is None
        assert row["take_profit"] is None

    def test_an_incomplete_setup_keeps_its_stage_and_reason(self):
        evaluations = {
            SYMBOL: {
                "stage": "SWEEP",
                "reason_code": "NO_5M_REVERSAL",
                "detail": "the 5m close did not reverse the sweep",
                "actionable": False,
            }
        }
        view = _view(_FakeEngine({}), evaluations=evaluations)
        row = view.strategy()["symbols"][0]
        assert row["stage"] == "SWEEP"
        assert row["reason_code"] == "NO_5M_REVERSAL"
        assert row["actionable"] is False


@pytest.mark.unit
class TestCryptoHealthView:
    def test_the_balances_come_from_the_worker_not_from_the_view(self):
        worker = {
            "status": "RUNNING",
            "account": {"equity": 10_250.0, "cash": 5_000.0},
            "broker_status": "connected",
            "market_data_provider": "binance_snapshot",
            "market_data_status": "ok",
            "decision_source_kind": "crypto",
            "last_cycle_at": NOW.isoformat(),
            "error_count": 0,
        }
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL)}), worker=worker)
        health = view.health()
        assert health["worker_status"] == "RUNNING"
        assert health["broker_status"] == "connected"
        assert health["decision_source_kind"] == "crypto"
        assert health["market_data_provider"] == "binance_snapshot"
        assert health["market_data_status"] == "ok"
        assert health["error_count"] == 0
        # The view carries no balance of its own; the dashboard reads the
        # worker's own figures for the money.
        assert "equity" not in health

    def test_a_worker_failure_is_reported_not_hidden(self):
        def boom():
            raise RuntimeError("worker state unreadable")

        view = CryptoViewService(engine=None, worker_status=boom, clock=lambda: NOW)
        health = view.health()
        assert health["worker_status"] == "ERROR"
        assert "worker state unreadable" in health["worker_last_error"]

    def test_no_worker_reports_no_status(self):
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL)}))
        health = view.health()
        assert health["worker_status"] is None
        assert health["broker_status"] is None

    def test_the_health_view_carries_no_secret(self):
        worker = {"status": "RUNNING", "api_key": "AKIA-SECRET", "secret": "hunter2"}
        view = _view(_FakeEngine({}), worker=worker)
        rendered = str(view.health())
        assert "AKIA-SECRET" not in rendered
        assert "hunter2" not in rendered


@pytest.mark.unit
class TestCryptoSnapshot:
    def test_the_snapshot_carries_the_schema_version(self):
        view = _view(_FakeEngine({SYMBOL: _snapshot(SYMBOL)}))
        snapshot = view.snapshot()
        assert snapshot["schema_version"] == CRYPTO_VIEW_SCHEMA_VERSION
        assert set(snapshot) == {"schema_version", "market", "strategy", "health"}

    def test_the_symbol_list_is_canonicalised(self):
        view = _view(_FakeEngine({}), symbols=("BTC/USDT",))
        assert view.symbols == ("BTC/USDT",)


# ---------------------------------------------------------------------------
# The HTTP surface: the endpoints the page actually calls
# ---------------------------------------------------------------------------

_LOOPBACK_CONNECT, _LOOPBACK_LOOKUP = socket.socket.connect, socket.getaddrinfo


@pytest.fixture
def allow_loopback(monkeypatch):
    """Permit loopback sockets only; every other destination is refused.

    The suite's autouse guard blocks the network, so a test that binds a
    loopback server needs this narrow exception — and only for 127.0.0.1.
    """

    def _refuse(*args, **kwargs):
        raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (
            _LOOPBACK_CONNECT if address[0] == "127.0.0.1" else _refuse
        )(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (
            _LOOPBACK_LOOKUP if host == "127.0.0.1" else _refuse
        )(host, *a, **k),
    )


@pytest.mark.unit
class TestCryptoHttpEndpoints:
    def _serve(self, tmp_path, *, engine=None, symbols=()):
        import threading
        from http.server import ThreadingHTTPServer

        from tradingagents.paper_experiment.dashboard.server import Dashboard, make_handler

        dashboard = Dashboard(
            tmp_path,
            token="t",
            allow_control=False,
            stub=True,
            crypto_engine=engine,
            crypto_symbols=symbols,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _get(self, port, path):
        import json
        import urllib.request

        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
        request.add_header("Authorization", "Bearer t")
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_the_crypto_endpoint_serves_the_real_view(self, tmp_path, allow_loopback):
        engine = _FakeEngine({SYMBOL: _snapshot(SYMBOL, price=42.0)})
        server, thread = self._serve(tmp_path, engine=engine, symbols=(SYMBOL,))
        try:
            body = self._get(server.server_address[1], "/api/crypto")
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert body["schema_version"] == CRYPTO_VIEW_SCHEMA_VERSION
        assert body["market"]["symbols"][0]["price"] == 42.0
        assert body["strategy"]["symbols"][0]["stage"] == "NO_SIGNAL"
        assert body["health"]["data_provider"] == "binance_public"

    def test_the_market_endpoint_serves_only_the_market(self, tmp_path, allow_loopback):
        engine = _FakeEngine({SYMBOL: _snapshot(SYMBOL)})
        server, thread = self._serve(tmp_path, engine=engine, symbols=(SYMBOL,))
        try:
            body = self._get(server.server_address[1], "/api/crypto/market")
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert body["count"] == 1
        assert body["symbols"][0]["symbol"] == SYMBOL
        assert "health" not in body

    def test_without_an_engine_the_endpoint_reports_not_started(self, tmp_path, allow_loopback):
        server, thread = self._serve(tmp_path)
        try:
            body = self._get(server.server_address[1], "/api/crypto/market")
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert body["status"] == "not_started"
        assert body["symbols"] == []

    def test_the_endpoint_needs_the_token(self, tmp_path, allow_loopback):
        import urllib.error
        import urllib.request

        server, thread = self._serve(tmp_path)
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/crypto"
            )
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(request, timeout=5)
        finally:
            server.shutdown()
            thread.join(timeout=3)
        assert err.value.code == 401
