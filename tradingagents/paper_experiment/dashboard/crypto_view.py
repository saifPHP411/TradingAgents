"""The dashboard's crypto view: live market, TJR stage and system health.

The dashboard is the system's control room, so every number it shows must come
from a real source. This module is the read-only *crypto* half of that: it reads
the live crypto engine (Binance public market data), the paper worker's own
recorded state, and the TJR rule chain's evaluation — and it never computes a
balance, a P/L or a price of its own.

Three rules shape it:

- **No invented numbers.** A value the source does not have is ``None``, and the
  view reports ``None``. The dashboard renders "N/A", never a zero and never a
  placeholder price.
- **Freshness is part of the value.** A price older than the engine's freshness
  bound is reported as *stale* with its age, not as a current price.
- **No secrets.** The view reports provider names, statuses and timestamps — never
  a key, a token or a credential.

The engine is injectable, so the whole view is exercisable offline: a test drives
it with a fake engine and asserts the numbers it reports.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from tradingagents.crypto.symbols import parse_pair

CRYPTO_VIEW_SCHEMA_VERSION = 1
"""The on-disk/in-flight shape of the crypto view."""


class CryptoViewError(RuntimeError):
    """The crypto view could not be built."""


class CryptoViewService:
    """A read-only view over the live crypto engine and the paper worker.

    The engine is the single source of truth for market data: the same
    :class:`~tradingagents.crypto.engine.CryptoMarketDataEngine` the worker and
    the strategy read, so the dashboard cannot show a different price from the
    one an order was sized against.

    ``worker_status`` is a callable returning the worker's recorded status. It is
    optional: with no worker (paper trading off) the view reports the *absence* of
    a worker rather than fabricating an account.

    ``strategy_evaluations`` is a callable returning, per symbol, the TJR
    evaluation the rule chain last produced. It is optional for the same reason.
    """

    def __init__(
        self,
        *,
        engine: Any | None = None,
        symbols: tuple[str, ...] = (),
        worker_status: Callable[[], dict[str, Any]] | None = None,
        strategy_evaluations: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._engine = engine
        self.symbols = tuple(parse_pair(s).canonical for s in symbols)
        self._worker_status = worker_status
        self._strategy_evaluations = strategy_evaluations
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self.last_refresh_at: str | None = None
        self.last_error: str | None = None

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """A secret-free identity of the crypto data path, for a status view."""
        if self._engine is None:
            return {
                "crypto_data_provider": None,
                "crypto_data_status": "not_started",
                "crypto_data_detail": "no crypto market-data engine is configured",
            }
        described: dict[str, Any] = {}
        describe = getattr(self._engine, "describe", None)
        if callable(describe):
            try:
                described = describe()
            except Exception:  # noqa: BLE001 - a status view must never raise
                described = {}
        stream_status = getattr(self._engine, "stream_status", None)
        if callable(stream_status):
            try:
                described["crypto_data_stream"] = stream_status()
            except Exception:  # noqa: BLE001 - a status view must never raise
                described["crypto_data_stream"] = {"state": "unknown"}
        return described

    # -- live market ------------------------------------------------------

    def market(self) -> dict[str, Any]:
        """The live market view: one row per symbol, from the engine.

        Every field is the engine's own value. A symbol the engine does not know
        is reported as an unknown symbol, not as a zero-priced row.
        """
        engine = self._engine
        if engine is None:
            return {
                "symbols": [],
                "count": 0,
                "status": "not_started",
                "detail": "no crypto market-data engine is configured",
                "last_refresh_at": self.last_refresh_at,
            }
        try:
            snapshots = engine.snapshots()
        except Exception as exc:  # noqa: BLE001 - a source failure is reported
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "symbols": [],
                "count": 0,
                "status": "error",
                "detail": self.last_error,
                "last_refresh_at": self.last_refresh_at,
            }
        rows = [self._market_row(symbol, snapshots.get(symbol)) for symbol in self.symbols]
        with self._lock:
            self.last_refresh_at = self._clock().isoformat()
            self.last_error = None
        # The status describes the *data*, not just the row count: a market whose
        # only rows are stale is stale, one with no rows at all has no data, and a
        # market where some symbols are fresh and some are not is partial.
        # "ok" is reserved for a market where every row is fresh.
        if not rows:
            status = "no_data"
        elif all(row["status"] == "ok" for row in rows):
            status = "ok"
        elif any(row["status"] == "ok" for row in rows):
            status = "partial"
        elif any(row["status"] == "no_data" for row in rows):
            status = "no_data"
        else:
            status = "stale"
        fresh = sum(1 for row in rows if row["fresh"])
        return {
            "symbols": rows,
            "count": len(rows),
            "fresh_count": fresh,
            "status": status,
            "detail": None if status == "ok" else f"{fresh} of {len(rows)} symbols are fresh",
            "last_refresh_at": self.last_refresh_at,
        }

    def _market_row(self, symbol: str, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        """One symbol's live row, normalized to the dashboard's vocabulary."""
        if not snapshot:
            return {
                "symbol": symbol,
                "price": None,
                "fresh": False,
                "age_seconds": None,
                "status": "no_data",
                "detail": "the engine holds no data for this symbol",
                "bid": None,
                "ask": None,
                "spread": None,
                "change_24h": None,
                "change_percent_24h": None,
                "volume_24h": None,
                "high_24h": None,
                "low_24h": None,
                "closed_candles": {},
                "forming_candles": {},
                "newest_event_at": None,
            }
        ticker = snapshot.get("ticker") or {}
        closed = snapshot.get("closed") or {}
        forming = snapshot.get("forming") or {}
        return {
            "symbol": snapshot.get("symbol", symbol),
            "price": snapshot.get("price"),
            "fresh": bool(snapshot.get("fresh")),
            "age_seconds": snapshot.get("age_seconds"),
            "status": "ok" if snapshot.get("fresh") else "stale",
            "detail": snapshot.get("detail"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "spread": ticker.get("spread"),
            "change_24h": ticker.get("change_24h"),
            "change_percent_24h": ticker.get("change_percent_24h"),
            "volume_24h": ticker.get("volume_24h"),
            "high_24h": ticker.get("high_24h"),
            "low_24h": ticker.get("low_24h"),
            "closed_candles": {
                interval: _candle_row(candle) for interval, candle in closed.items()
            },
            "forming_candles": {
                interval: _candle_row(candle) for interval, candle in forming.items()
            },
            "newest_event_at": snapshot.get("newest_event_at"),
        }

    # -- TJR strategy -----------------------------------------------------

    def strategy(self) -> dict[str, Any]:
        """The TJR stage per symbol, from the rule chain's own evaluation.

        A symbol with no evaluation is reported as ``NO SIGNAL`` with the reason
        the chain gave — never as a BUY or SELL that did not complete. Every
        configured symbol gets a row even when no evaluation source is wired, so
        the dashboard shows the pair it is watching and says plainly that the
        chain has not evaluated it.
        """
        if self._strategy_evaluations is None:
            return {
                "symbols": [_strategy_row(symbol, None) for symbol in self.symbols],
                "count": len(self.symbols),
                "status": "not_started",
                "detail": "no TJR evaluation source is configured",
            }
        try:
            evaluations = self._strategy_evaluations()
        except Exception as exc:  # noqa: BLE001 - a source failure is reported
            return {
                "symbols": [_strategy_row(symbol, None) for symbol in self.symbols],
                "count": len(self.symbols),
                "status": "error",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        rows = []
        for symbol in self.symbols:
            evaluation = (evaluations or {}).get(symbol)
            rows.append(_strategy_row(symbol, evaluation))
        return {
            "symbols": rows,
            "count": len(rows),
            "status": "ok",
            "detail": None,
        }

    # -- system health ----------------------------------------------------

    def health(self) -> dict[str, Any]:
        """The system's own health: data path, worker, broker and errors.

        The worker and broker figures come from the worker's *recorded* status,
        so this view never calls the broker and never places an order.
        """
        worker: dict[str, Any] = {}
        if self._worker_status is not None:
            try:
                worker = self._worker_status() or {}
            except Exception as exc:  # noqa: BLE001 - a status view must never raise
                worker = {"status": "ERROR", "last_error": f"{type(exc).__name__}: {exc}"}
        data = self.describe()
        market = self.market()
        return {
            "schema_version": CRYPTO_VIEW_SCHEMA_VERSION,
            "data_provider": data.get("crypto_data_provider"),
            "data_status": data.get("crypto_data_status", "not_started"),
            "data_last_refresh_at": data.get("crypto_data_last_refresh_at"),
            "data_last_error": data.get("crypto_data_last_error"),
            "stream": data.get("crypto_data_stream"),
            "worker_status": worker.get("status"),
            "worker_last_error": worker.get("last_error"),
            "decision_source_kind": worker.get("decision_source_kind"),
            "market_data_provider": worker.get("market_data_provider"),
            "market_data_status": worker.get("market_data_status"),
            "broker_status": worker.get("broker_status"),
            "last_cycle_at": worker.get("last_cycle_at"),
            "positions_checked_at": worker.get("positions_checked_at"),
            "last_order_sync_at": worker.get("last_order_sync_at"),
            "error_count": worker.get("error_count"),
            "last_error": worker.get("last_error"),
            "market_status": market.get("status"),
            "market_detail": market.get("detail"),
            "fresh_symbols": market.get("fresh_count"),
            "total_symbols": market.get("count"),
            # Whether the venues' own listings have been read, and which pairs
            # they refused. The registries are read when the paper worker is
            # built, so before that this is an explicit "not validated yet" —
            # never an implied "everything is fine".
            "symbol_validation": worker.get("symbol_validation"),
            "observed_at": self._clock().isoformat(),
        }

    # -- the whole view ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything the crypto dashboard needs, in one read."""
        return {
            "schema_version": CRYPTO_VIEW_SCHEMA_VERSION,
            "market": self.market(),
            "strategy": self.strategy(),
            "health": self.health(),
        }


def _candle_row(candle: dict[str, Any] | None) -> dict[str, Any] | None:
    """A candle's display row, keeping the *closed* flag explicit.

    The engine keeps a forming bar apart from the closed ones, so the view can
    show the forming candle without ever letting it be read as a decision input.
    """
    if not isinstance(candle, dict):
        return None
    return {
        "open": candle.get("open"),
        "high": candle.get("high"),
        "low": candle.get("low"),
        "close": candle.get("close"),
        "volume": candle.get("volume"),
        "open_time": candle.get("open_time"),
        "close_time": candle.get("close_time"),
    }


def _strategy_row(symbol: str, evaluation: dict[str, Any] | None) -> dict[str, Any]:
    """One symbol's TJR stage row.

    The fields are the chain's own: the stage it reached, the reason it stopped,
    the direction it found and the levels it derived. A missing evaluation is a
    ``NO SIGNAL`` with no reason invented for it.
    """
    if not isinstance(evaluation, dict):
        return {
            "symbol": symbol,
            "stage": "NO_SIGNAL",
            "reason_code": None,
            "detail": None,
            "direction": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
            "evaluated_at": None,
            "actionable": False,
        }
    entry = evaluation.get("entry")
    stop = evaluation.get("stop_loss")
    target = evaluation.get("take_profit")
    return {
        "symbol": symbol,
        "stage": evaluation.get("stage") or "NO_SIGNAL",
        "reason_code": evaluation.get("reason_code"),
        "detail": evaluation.get("detail"),
        "direction": evaluation.get("direction"),
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": evaluation.get("risk_reward"),
        "evaluated_at": evaluation.get("evaluated_at"),
        "actionable": bool(evaluation.get("actionable")),
    }


__all__ = [
    "CRYPTO_VIEW_SCHEMA_VERSION",
    "CryptoViewError",
    "CryptoViewService",
]
