"""A small, phone-friendly HTTP dashboard for a paper experiment.

This is a **control and monitoring surface for the simulation**, not a trading
surface. It serves:

- ``GET /`` — a single self-contained HTML page (no external assets) sized for a
  phone;
- ``GET /healthz`` — an unauthenticated liveness probe for Railway;
- ``GET /api/summary``, ``/api/decisions``, ``/api/journal`` — read-only views of
  the persisted experiment;
- ``POST /api/run`` — *optionally* advance the simulation by N sessions.

Every ``/api`` route requires a bearer token. The ``POST /api/run`` route is
additionally gated behind ``--allow-control`` **and** a confirmation header, and
is refused while another run is in progress. There is no route that places a
broker order, and none can be added here without an explicit code change: the
only action available drives ``PaperExperiment.process_session``, which is
simulation-only.

The server uses only the standard library, so it adds no dependency to a project
whose runtime is deliberately small.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tradingagents.risk.paper_parity import WHOLE_SHARES

from ...crypto.symbols import parse_pair
from ..broker import (
    banner_for_provider,
    DECISION_SOURCE_CHOICES,
    DECISION_SOURCE_HISTORY,
    DEFAULT_MAX_PRICE_DEVIATION,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_CHOICES,
    MARKET_DATA_CHOICES,
    MAX_PRICE_DEVIATION_ENV,
    PROVIDER_BAR_OPEN,
    BrokerError,
    BrokerNotConfiguredError,
    BybitDemoBroker,
    BybitBrokerNotConfiguredError,
    EXECUTION_PROVIDER_BYBIT_DEMO,
)
from ..crypto_symbol_config import (
    DEFAULT_MARKET_DATA_SYMBOLS,
    parse_symbol_list,
)
from ..experiment import ExperimentConfig, PaperExperiment
from ..market_data_wiring import (
    DECISION_SOURCE_ENV,
    EXECUTION_PROVIDER_ENV,
    FRACTIONAL_SHARES_ENV,
    MARKET_DATA_ENV,
    SNAPSHOT_MAX_AGE_ENV,
    MarketDataWiringError,
    describe_decision_source,
    describe_execution_provider,
    describe_provider,
    market_data_factory_for,
    resolve_execution_provider,
    resolve_paper_settings,
    resolve_settings,
)
from ..providers.base import ProviderNotConfiguredError
from .auth import DashboardAuthError, check_bearer, require_bindable, resolve_token
from .crypto_view import CryptoViewService
from .paper_control import PaperTradingController, PaperTradingDisabled, _endpoint_for_provider
from .state import DashboardStateError, ExperimentView

DEFAULT_MAX_SESSIONS_PER_CALL = 50
"""The most sessions one control call may advance, so a request stays bounded."""

CONTROL_CONFIRM_HEADER = "X-Control-Confirm"

PAPER_CONFIRM_HEADER = "X-Paper-Confirm"
"""The explicit confirmation header a paper start/stop must carry.

Distinct from the run-confirm header on purpose: a stray "advance the simulation"
request must not be able to start the broker-backed worker. Starting the worker is
the one dashboard action that can lead to a broker order, so it requires its own
deliberate confirmation.
"""

ROOT_ENV = "TRADINGAGENTS_EXPERIMENT_ROOT"
"""Environment variable the module entry point reads ``--root`` from."""

HEALTH_PATH = "/healthz"
"""The unauthenticated liveness probe.

The deployment's health check names this path, so it is a named constant rather
than a literal in the handler: a rename must not silently break the platform's
probe.
"""

CRYPTO_SYMBOLS_ENV = "TRADINGAGENTS_CRYPTO_SYMBOLS"
"""Environment variable naming the crypto pairs the live section shows.

Space- or comma-separated, e.g. ``"BTC/USDT ETH/USDT"``. It selects what the
dashboard *displays*; it never decides what can be traded — the venue's own
listing and the risk engine do that.
"""

EXPERIMENT_SYMBOLS_ENV = "TRADINGAGENTS_EXPERIMENT_SYMBOLS"
"""Environment variable naming the experiment symbols (canonical crypto pairs).

Space- or comma-separated, e.g. ``"BTC/USDT ETH/USDT"``. Used to create the
experiment idempotently on first startup.
"""

DEFAULT_CRYPTO_SYMBOLS = DEFAULT_MARKET_DATA_SYMBOLS
"""The default live-section pairs: the liquid majors in Binance's USDT pairs.

Every pair here is a *request*. It is validated against both venue registries
before it is used, so a pair one venue does not serve is reported and excluded
rather than assumed. The list is deliberately **not** mixed: the market-data
venue quotes USDT pairs and the execution venue quotes USD pairs, and a single
list cannot be correct for both.
"""


def _experiment_symbols(crypto_symbols: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The symbols a *new* experiment is created with.

    ``TRADINGAGENTS_EXPERIMENT_SYMBOLS`` wins when it is set, because the
    experiment's persisted pair list is the deployment's own choice and must not
    be silently replaced by the dashboard's display list. The display list is the
    fallback, and the default list is the last resort.

    A symbol that is not a pair is a loud error: a malformed entry in the
    deployment's configuration must be reported, not persisted and not skipped.
    """
    configured = os.environ.get(EXPERIMENT_SYMBOLS_ENV, "").strip()
    if configured:
        return parse_symbol_list(configured)
    if crypto_symbols:
        return parse_symbol_list(tuple(crypto_symbols))
    return parse_symbol_list(DEFAULT_CRYPTO_SYMBOLS)


def _ensure_experiment(root: Path, *, crypto_symbols: tuple[str, ...] = ()) -> None:
    """Idempotently create a crypto experiment if one doesn't exist.

    This is called at dashboard startup to guarantee that the experiment
    directory has a valid `session_state.json` before any dashboard component
    tries to read it. It never overwrites an existing experiment, never resets
    state, never auto-starts the worker, and never submits orders.

    The symbol list comes from ``TRADINGAGENTS_EXPERIMENT_SYMBOLS`` when it is
    set, and from the *live-section* list (``crypto_symbols``) otherwise, so the
    experiment's persisted pairs and the dashboard's displayed pairs agree by
    default.

    The list is a **request**, and only its *syntax* is checked here. The
    venues' own listings are **not** read at this point, deliberately: reading
    them would make the dashboard's startup depend on both exchanges being
    reachable, and a deployment that cannot reach an exchange must still come
    up and say so. Tradability is therefore established later, when the paper
    worker is built (see
    :meth:`~.paper_control.PaperTradingController._build_worker`), which reads
    both venue registries and rejects every pair a venue will not trade with
    that venue's own reason. The result is reported in the controller's
    ``symbol_validation`` status, so a pair one venue does not serve is shown
    as rejected rather than silently missing.

    Args:
        root: Experiment root directory.
        crypto_symbols: Canonical crypto pairs to use if creating a new experiment.
                       If empty, falls back to DEFAULT_CRYPTO_SYMBOLS.

    Raises:
        RuntimeError: If experiment creation fails for any reason.
    """
    root = Path(root)
    state_path = root / "session_state.json"
    if state_path.exists():
        # Experiment already exists - do nothing, preserve existing state
        return

    # No experiment exists - create one using symbols from env or defaults
    symbols = _experiment_symbols(crypto_symbols)

    # Validate symbols are well-formed pairs
    for sym in symbols:
        try:
            parse_pair(sym)
        except Exception as exc:
            raise RuntimeError(f"Invalid crypto symbol {sym!r}: {exc}") from exc

    # Build a frozen-baseline source (no network call) for the experiment
    # Use the same frozen dataset the CLI uses
    from tradingagents.baseline.data import load_feed
    from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL

    from ..sources import FrozenPriceSource

    try:
        feed = load_feed(DEFAULT_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(f"the frozen dataset is unavailable: {exc}") from exc

    # Use the frozen baseline as the data source - it's local and requires no API
    source = FrozenPriceSource(feed, name="frozen_baseline")

    # Build experiment config
    config = ExperimentConfig.from_protocol(
        symbols=symbols,
        provider="frozen",
        model="frozen",
    )

    # Create the experiment
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=root,
        inference=None,  # No inference needed for frozen baseline
    )

    try:
        experiment.ensure_created()
        experiment.save_state()
    except Exception as exc:
        # Clean up any partial state
        if state_path.exists():
            state_path.unlink()
        journal_path = root / "journal.json"
        if journal_path.exists():
            journal_path.unlink()
        raise RuntimeError(f"Failed to create crypto experiment: {exc}") from exc


class Dashboard:
    """The dashboard's configuration and shared state, one per server."""

    def __init__(
        self,
        root: str | Path,
        *,
        token: str | None = None,
        allow_control: bool = False,
        max_sessions_per_call: int = DEFAULT_MAX_SESSIONS_PER_CALL,
        stub: bool = False,
        allow_paper_trading: bool = False,
        paper_order_notional: float = 25.0,
        broker_factory=None,
        market_data_factory=None,
        paper_whole_shares: bool = WHOLE_SHARES,
        paper_market_data_provider: str = PROVIDER_BAR_OPEN,
        paper_market_data_max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
        paper_decision_source: str = DECISION_SOURCE_HISTORY,
        paper_max_price_deviation: float = DEFAULT_MAX_PRICE_DEVIATION,
        paper_execution_provider: str = EXECUTION_PROVIDER_ALPACA_PAPER,
        crypto_engine=None,
        crypto_symbols: tuple[str, ...] = (),
        refresh_policy=None,
        refresh_fetcher=None,
    ) -> None:
        self.root = Path(root)
        self.token = resolve_token(token)
        self.allow_control = allow_control
        self.max_sessions_per_call = max_sessions_per_call
        self.stub = stub
        self._lock = threading.Lock()
        self._run_in_progress = False
        self.last_run: dict[str, Any] | None = None
        self.refresh_policy = refresh_policy
        self.refresh_fetcher = refresh_fetcher
        self.paper = PaperTradingController(
            self.root,
            enabled=allow_paper_trading,
            order_notional=paper_order_notional,
            stub=stub,
            broker_factory=broker_factory,
            market_data_factory=market_data_factory,
            whole_shares=paper_whole_shares,
            market_data_provider=paper_market_data_provider,
            market_data_max_age_seconds=paper_market_data_max_age_seconds,
            decision_source_kind=paper_decision_source,
            max_price_deviation=paper_max_price_deviation,
            # The execution broker. Alpaca Paper is the default; Bybit Demo is
            # an explicit opt-in that never falls back to Alpaca.
            execution_provider=paper_execution_provider,
        )
        # The crypto half of the dashboard: a read-only view over the live crypto
        # engine (Binance public market data) and the worker's recorded status.
        # The engine is injected, so the dashboard reads the *same* engine the
        # strategy and the worker read, and a test can drive it offline.
        self.crypto = CryptoViewService(
            engine=crypto_engine,
            symbols=tuple(crypto_symbols),
            worker_status=lambda: self.paper.status(),
            strategy_evaluations=lambda: self.paper.strategy_evaluations(),
        )

    def view(self) -> ExperimentView:
        return ExperimentView(self.root)

    # -- read-only engine views ------------------------------------------

    def _internal_positions(self) -> dict[str, dict[str, Any]]:
        """The experiment's simulation-ledger positions, or ``{}`` when unreadable.

        Read from the report so the dashboard and the CLI cannot disagree. A
        failure to read is an empty mapping, never a fabricated position.
        """
        try:
            summary = self.view().summary()
        except Exception:  # noqa: BLE001 - a missing experiment is not a crash here
            return {}
        positions = ((summary.get("book") or {}).get("positions")) or {}
        return {k: v for k, v in positions.items() if isinstance(v, dict)}

    def engine(self) -> dict[str, Any]:
        """The separated engine states, from the worker's own recorded status.

        Read-only and build-free: it reads the persisted worker state, so polling
        it does not construct a worker or touch the broker.
        """
        from .engine_view import engine_view

        return engine_view(self.paper.status())

    def positions(self) -> dict[str, Any]:
        """Open positions, broker-sourced, reconciled against the simulation book."""
        from .engine_view import positions_view

        status = self.paper.status()

        # A persisted worker snapshot can outlive the worker process. Refresh
        # public Bybit Spot quotes for display without starting the worker,
        # touching private account endpoints, or placing orders.
        positions = status.get("positions") or {}
        if self.paper.execution_provider == "bybit_demo" and positions:
            try:
                from tradingagents.crypto.bybit import BybitPublicClient

                symbols = [
                    str(row.get("venue_symbol") or symbol).replace("/", "")
                    for symbol, row in positions.items()
                    if isinstance(row, dict)
                ]
                tickers = BybitPublicClient().tickers(symbols)
                for symbol, row in positions.items():
                    if not isinstance(row, dict):
                        continue
                    venue_symbol = str(row.get("venue_symbol") or symbol).replace("/", "")
                    ticker = tickers.get(venue_symbol, {})
                    if isinstance(ticker, dict):
                        price = ticker.get("lastPrice") or ticker.get("last")
                        try:
                            price = float(price)
                        except (TypeError, ValueError):
                            price = None
                        if price is not None and price > 0:
                            row["mark_price"] = price
            except Exception:
                # Keep the last known snapshot; the UI will continue to show
                # N/A rather than fabricate a quote if the public feed is down.
                pass

        rows = positions_view(status, self._internal_positions())
        return {
            "positions": rows,
            "count": len(rows),
            "positions_checked_at": status.get("positions_checked_at"),
            "positions_at": status.get("positions_at"),
            "banner": banner_for_provider(self.paper.execution_provider),
            "live": False,
        }

    def trades(self) -> dict[str, Any]:
        """The trade journal: submitted orders and settled trades, kept apart."""
        from .engine_view import trades_view

        status = self.paper.status()
        try:
            decisions = self.view().decisions()
        except Exception:  # noqa: BLE001 - no experiment yet is an empty journal
            decisions = []
        result = trades_view(decisions, status)
        result["banner"] = banner_for_provider(self.paper.execution_provider)
        result["live"] = False
        return result

    # -- control ----------------------------------------------------------

    def run_sessions(self, sessions: int) -> dict[str, Any]:
        """Advance the simulation by ``sessions`` sessions. Simulation-only.

        Serialised with a lock: a second call while one is running is refused
        rather than run concurrently, because two writers to one experiment
        directory would race on the state file.

        The experiment is loaded with a real decision seam built from the
        *persisted* config, so a session advanced here decides through the same
        provider a terminal ``run`` would. Without one every decision would be
        recorded as blocked — and, because a blocked session is still marked
        processed, that would silently burn the session. The provider reads its
        own credential; this method neither reads nor logs one.
        """
        if not self.allow_control:
            raise PermissionError(
                "control is disabled; start the dashboard with --allow-control to "
                "enable advancing the simulation"
            )
        if sessions < 0:
            raise ValueError("sessions must not be negative")
        bounded = min(sessions, self.max_sessions_per_call)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a run is already in progress")
        try:
            self._run_in_progress = True
            from ..runner import advance, build_inference, pending_sessions

            experiment = self.view().load()
            experiment.inference = build_inference(experiment.config, stub=self.stub)
            remaining_before = len(pending_sessions(experiment))
            advanced = advance(
                experiment,
                sessions=bounded,
                refresh_policy=self.refresh_policy,
                fetcher=self.refresh_fetcher,
            )
            outcomes = advanced.outcomes
            refresh = advanced.refresh
            result = {
                "processed": [o.session for o in outcomes],
                "processed_count": len(outcomes),
                "decisions": [o.decision_id for o in outcomes if o.decision_id],
                "decision_count": sum(1 for o in outcomes if o.decision_id),
                "no_decision_reasons": sorted(
                    {
                        o.decision_skip_reason
                        for o in outcomes
                        if o.decision_skip_reason is not None
                    }
                ),
                "remaining_before": remaining_before,
                "remaining_after": len(pending_sessions(experiment)),
                "bounded_to": bounded,
                "refresh": refresh.to_dict() if refresh is not None else None,
            }
            self.last_run = result
            return result
        finally:
            self._run_in_progress = False
            self._lock.release()


def make_handler(dashboard: Dashboard):
    """Build a request handler bound to ``dashboard``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "TradingAgentsDashboard/1.0"

        # -- helpers --------------------------------------------------

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, code: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            if check_bearer(self.headers.get("Authorization"), dashboard.token):
                return True
            self._send_json(401, {"error": "unauthorized"})
            return False

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Default logging writes the request line, which can include a query
            # string; the token is only ever in a header, but keep the log terse.
            return

        # -- routes ---------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            path = urlsplit(self.path).path
            if path == HEALTH_PATH:
                self._send_json(200, {"status": "ok"})
                return
            if path == "/":
                self._send_html(200, INDEX_HTML)
                return
            if not self._authed():
                return
            try:
                if path == "/api/summary":
                    self._send_json(200, dashboard.view().summary())
                elif path == "/api/decisions":
                    self._send_json(200, {"decisions": dashboard.view().decisions()})
                elif path == "/api/journal":
                    query = parse_qs(urlsplit(self.path).query)
                    limit = int((query.get("limit") or ["50"])[0])
                    self._send_json(
                        200, {"entries": dashboard.view().journal_tail(limit=limit)}
                    )
                elif path == "/api/control/status":
                    self._send_json(
                        200,
                        {
                            "control_enabled": dashboard.allow_control,
                            "run_in_progress": dashboard._run_in_progress,
                            "max_sessions_per_call": dashboard.max_sessions_per_call,
                            "last_run": dashboard.last_run,
                        },
                    )
                elif path == "/api/paper/status":
                    self._send_json(200, dashboard.paper.status())
                elif path == "/api/paper/account-check":
                    if dashboard.paper.execution_provider != EXECUTION_PROVIDER_BYBIT_DEMO:
                        self._send_json(409, {
                            "ok": False,
                            "error": "wrong_execution_provider",
                            "message": "This read-only check is available only when Bybit Demo is selected.",
                        })
                        return
                    try:
                        # A fresh, stateless broker instance: connect() performs only
                        # GET /v5/account/wallet-balance. No worker, order, or state
                        # store is constructed or started by this diagnostic route.
                        account = BybitDemoBroker().connect()
                        self._send_json(200, {
                            "ok": True,
                            "provider": EXECUTION_PROVIDER_BYBIT_DEMO,
                            "endpoint": "https://api-demo.bybit.com",
                            "account": {
                                "account_type": account.account_type,
                                "currency": account.currency,
                                "equity": account.equity,
                                "available_balance": account.available_balance,
                            },
                        })
                    except BybitBrokerNotConfiguredError:
                        self._send_json(503, {
                            "ok": False,
                            "error": "credentials_not_configured",
                            "message": "Bybit Demo API credentials are not configured.",
                        })
                    except Exception as exc:  # noqa: BLE001 - return secret-free diagnosis
                        import re
                        message = str(exc)
                        match = re.search(r"retCode (\d+)", message)
                        code = int(match.group(1)) if match else None
                        diagnosis = "broker_or_transport_error"
                        if code == 10004:
                            diagnosis = "signature_error"
                        elif code == 10005:
                            diagnosis = "api_key_permission_denied"
                        elif code == 10003:
                            diagnosis = "api_key_invalid_or_wrong_environment"
                        elif code == 10010:
                            diagnosis = "ip_restriction"
                        elif code == 10002:
                            diagnosis = "timestamp_or_recv_window"
                        elif "HTTP 401" in message or "HTTP 403" in message:
                            diagnosis = "http_authentication_or_permission_rejected"
                        self._send_json(502, {
                            "ok": False,
                            "error": diagnosis,
                            "bybit_ret_code": code,
                            "error_type": type(exc).__name__,
                        })
                elif path == "/api/engine":
                    self._send_json(200, dashboard.engine())
                elif path == "/api/positions":
                    self._send_json(200, dashboard.positions())
                elif path == "/api/trades":
                    self._send_json(200, dashboard.trades())
                elif path == "/api/crypto/market":
                    self._send_json(200, dashboard.crypto.market())
                elif path == "/api/crypto/strategy":
                    self._send_json(200, dashboard.crypto.strategy())
                elif path == "/api/crypto/health":
                    self._send_json(200, dashboard.crypto.health())
                elif path == "/api/crypto":
                    self._send_json(200, dashboard.crypto.snapshot())
                else:
                    self._send_json(404, {"error": "not found"})
            except DashboardStateError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surface, never crash the server
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            path = urlsplit(self.path).path
            if path in ("/api/paper/start", "/api/paper/stop"):
                self._handle_paper_control(path)
                return
            if path != "/api/run":
                self._send_json(404, {"error": "not found"})
                return
            if not self._authed():
                return
            # A deliberate, explicit header so a stray form post or a CSRF-style
            # request cannot trigger a run.
            if (self.headers.get(CONTROL_CONFIRM_HEADER) or "").lower() != "yes":
                self._send_json(
                    403,
                    {
                        "error": "confirmation header required",
                        "hint": f"send {CONTROL_CONFIRM_HEADER}: yes",
                    },
                )
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
                sessions = int(payload.get("sessions", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                self._send_json(400, {"error": "body must be JSON with an integer 'sessions'"})
                return
            try:
                result = dashboard.run_sessions(sessions)
                self._send_json(200, result)
            except PermissionError as exc:
                self._send_json(403, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
            except DashboardStateError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        # -- paper trading control ------------------------------------

        def _handle_paper_control(self, path: str) -> None:
            """Start or stop the Alpaca **paper** worker. Auth + confirmation required.

            The endpoint cannot change the broker endpoint: the worker builds its
            broker from the server's own configuration, and the broker's gate
            refuses any non-paper host. No credential is accepted from the body or
            any header — the broker reads its own environment variables.
            """
            if not self._authed():
                return
            if (self.headers.get(PAPER_CONFIRM_HEADER) or "").lower() != "yes":
                self._send_json(
                    403,
                    {
                        "error": "confirmation header required",
                        "hint": f"send {PAPER_CONFIRM_HEADER}: yes",
                    },
                )
                return
            try:
                if path == "/api/paper/start":
                    result = dashboard.paper.start()
                    self._send_json(200, {"started": True, **result})
                else:
                    result = dashboard.paper.stop()
                    self._send_json(200, {"stopped": True, **result})
            except PaperTradingDisabled as exc:
                self._send_json(403, {"error": str(exc)})
            except (BrokerNotConfiguredError, ProviderNotConfiguredError) as exc:
                # A missing credential is a deployment misconfiguration, not a
                # fault in the request or in the server. It is reported as 503
                # with the same secret-free detail the worker recorded, so the
                # operator sees exactly which variable to set.
                self._send_json(
                    503,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "reason": "the paper venue's credentials are not configured",
                    },
                )
            except BrokerError as exc:
                # A preflight/broker failure is a safe, secret-free 502; the
                # worker has recorded ERROR and is not running.
                self._send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler



def _crypto_symbols(args) -> tuple[str, ...]:
    """The crypto pairs the live section shows, from the flag or the environment.

    A pair that the venue does not list is *not* dropped here: the engine reports
    it, so an operator sees which pair failed instead of a silently shorter list.
    A symbol that is not a pair at all is a loud error, never a silently ignored
    entry.
    """
    raw = getattr(args, "crypto_symbols", None)
    if raw is None:
        raw = os.environ.get(CRYPTO_SYMBOLS_ENV, "")
    return parse_symbol_list(raw)


def _resolve_execution_provider(args) -> str:
    """Which broker the paper worker places orders with.

    The flag wins, then the environment variable, then the default
    (``alpaca_paper``). A bad value is a loud error: a typo must not silently
    trade somewhere the operator did not choose.
    """
    raw = getattr(args, "paper_execution_provider", None)
    if raw is None:
        raw = os.environ.get(EXECUTION_PROVIDER_ENV, "").strip()
    if raw == "":
        return EXECUTION_PROVIDER_ALPACA_PAPER
    return resolve_execution_provider(raw)


def _resolve_max_price_deviation(args) -> float:
    """The largest gap tolerated between the decision price and the execution price.

    The flag wins, then the environment variable, then the default. A value that
    is not a finite, non-negative fraction is a loud error: a misconfigured guard
    must not silently become "refuse everything" or "refuse nothing".
    """
    raw = getattr(args, "paper_max_price_deviation", None)
    if raw is None:
        raw = os.environ.get(MAX_PRICE_DEVIATION_ENV, "").strip()
    if raw == "" or raw is None:
        return DEFAULT_MAX_PRICE_DEVIATION
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise MarketDataWiringError(
            f"{MAX_PRICE_DEVIATION_ENV} must be a number (a fraction, e.g. 0.05): {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise MarketDataWiringError(
            f"{MAX_PRICE_DEVIATION_ENV} must be a finite, non-negative fraction: {raw!r}"
        )
    return value


def _market_data_factory_for(choice: str, *, max_age_seconds: float | None = None):
    """The market-data factory the CLI's ``--paper-market-data`` selects.

    Delegates to :func:`~..market_data_wiring.market_data_factory_for`, so this
    entrypoint and the ``serve`` subcommand select the identical seam. ``bar_open``
    returns ``None``, which keeps the controller's historical default (the
    experiment's latest bar open). ``alpaca_snapshot`` returns a factory that
    builds a current-price source from Alpaca market data, with the configured
    freshness limit. A factory that cannot be built raises when the worker is
    constructed, so a missing credential is reported rather than silently replaced
    by a stale price.
    """
    return market_data_factory_for(choice, max_age_seconds=max_age_seconds)


def _build_crypto_engine(symbols: tuple[str, ...]):
    """The live crypto engine, with its REST refresh thread started.

    The engine reads Binance's **public** market data — no key, no registration —
    and the WebSocket tape is started alongside it, so the dashboard's live section
    is fed by a real feed. The refresh thread is a daemon: it dies with the
    process and never keeps it alive.

    A failure to reach the venue is *not* fatal: the engine records the error and
    the dashboard reports it as an error state, which is the honest outcome.
    """
    from tradingagents.crypto.engine import CryptoMarketDataEngine

    engine = CryptoMarketDataEngine(symbols=symbols)
    # A failure to reach the venue is not fatal: the engine records the error and
    # the dashboard reports it as an error state, which is the honest outcome.
    with contextlib.suppress(Exception):
        engine.refresh_tickers()
        for interval in engine.intervals:
            engine.refresh_candles(interval)
    # A stream failure leaves the REST path working.
    with contextlib.suppress(Exception):
        engine.start_stream()
    _start_crypto_refresh(engine)
    return engine


def _start_crypto_refresh(engine, *, interval_seconds: float = 15.0) -> threading.Thread:
    """A daemon thread that keeps the crypto engine's REST snapshot current.

    The WebSocket tape carries the price; this thread refreshes the 24-hour
    ticker and the candle windows, which the tape does not carry. It is a daemon,
    so it never blocks shutdown.
    """

    def _loop() -> None:
        while True:
            try:
                engine.refresh_tickers()
                for interval in engine.intervals:
                    engine.refresh_candles(interval)
            except Exception:  # noqa: BLE001 - the engine records the failure
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="crypto-refresh", daemon=True)
    thread.start()
    return thread


def build_parser() -> argparse.ArgumentParser:
    """The module entry point's argument parser.

    Built by a named function so a deployment's *defaults* can be inspected
    without starting a server: the safety-critical defaults — paper trading off,
    the live crypto engine off, the risk cap at $25 — are properties of the
    deployment, and a test must be able to read them.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.paper_experiment.dashboard.server",
        description=(
            "Paper-experiment dashboard. Simulation only: no broker order is ever "
            "placed. Read-only unless --allow-control is passed."
        ),
    )
    parser.add_argument(
        "--root",
        default=os.environ.get(ROOT_ENV) or None,
        help=f"experiment directory (default: ${ROOT_ENV})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("TRADINGAGENTS_DASHBOARD_HOST", "127.0.0.1"),
        help="interface to bind (default: $TRADINGAGENTS_DASHBOARD_HOST, else loopback)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port to listen on (default: $PORT, else 8000)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "dashboard token (default: $TRADINGAGENTS_DASHBOARD_TOKEN). Required "
            "for a non-loopback bind; never logged."
        ),
    )
    parser.add_argument(
        "--allow-control",
        action="store_true",
        help="enable the advance-the-simulation action (off by default)",
    )
    parser.add_argument(
        "--allow-paper-trading",
        action="store_true",
        help=(
            "enable the START/STOP control for the Alpaca PAPER worker (off by "
            "default). Paper only: the broker endpoint is fixed to Alpaca's paper "
            "host and every live endpoint is refused. Requires "
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET in the environment."
        ),
    )
    parser.add_argument(
        "--paper-order-notional",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_PAPER_ORDER_NOTIONAL", "25") or 25),
        help="target cash per paper order (default: $TRADINGAGENTS_PAPER_ORDER_NOTIONAL, else 25)",
    )
    parser.add_argument(
        "--paper-market-data",
        choices=MARKET_DATA_CHOICES,
        default=None,
        help=(
            "how the paper worker reads a current price (default: "
            f"${MARKET_DATA_ENV}, else bar_open, the experiment's latest bar open). "
            "'alpaca_snapshot' reads a genuinely current price from Alpaca market "
            "data (feed=iex) and requires ALPACA_PAPER_API_KEY / "
            "ALPACA_PAPER_API_SECRET. 'binance_snapshot' reads a genuinely current "
            "CRYPTO price from Binance's public market data (no API key, no "
            "registration). These change only the price seam, never the decision "
            "source or the strategy."
        ),
    )
    parser.add_argument(
        "--paper-snapshot-max-age-seconds",
        type=float,
        default=None,
        help=(
            "the maximum age of a snapshot price before it is refused, in seconds "
            f"(default: ${SNAPSHOT_MAX_AGE_ENV}, else "
            f"{DEFAULT_MAX_SNAPSHOT_AGE_SECONDS:.0f}). A stale snapshot raises and no "
            "order is sized from it."
        ),
    )
    parser.add_argument(
        "--paper-decision-source",
        choices=DECISION_SOURCE_CHOICES,
        default=None,
        help=(
            "which decisions the paper worker acts on (default: "
            f"${DECISION_SOURCE_ENV}, else history). 'history' is the existing paper "
            "experiment's recorded, historical decisions, bounded by its sessions (it "
            "reports HISTORY_EXHAUSTED once they are consumed). 'current_market' is an "
            "INDEPENDENT decision taken live over current Alpaca market data, gated on "
            "Alpaca's market clock; it is not blocked by a consumed history, and needs "
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET and an LLM credential. "
            "'tjr' is the deterministic TJR rule chain (no model) over current Alpaca "
            "INTRADAY 1m/5m bars. 'crypto' is that same unchanged TJR rule chain over "
            "current Binance public INTRADAY 1m/5m crypto bars (no API key), gated on "
            "a 24/7 clock; pair it with --paper-market-data binance_snapshot."
        ),
    )
    parser.add_argument(
        "--paper-execution-provider",
        choices=EXECUTION_PROVIDER_CHOICES,
        default=None,
        help=(
            "which broker the paper worker places orders with (default: "
            f"${EXECUTION_PROVIDER_ENV}, else alpaca_paper). 'alpaca_paper' is the "
            "existing Alpaca Paper path and stays the default. 'bybit_demo' is the "
            "Bybit DEMO Spot broker (crypto Spot only, demo trading, long-only); it "
            "needs BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET and never falls back "
            "to Alpaca when they are missing."
        ),
    )
    parser.add_argument(
        "--paper-max-price-deviation",
        type=float,
        default=None,
        help=(
            "the largest relative gap tolerated between the decision price and the "
            "execution venue's price before the order is refused (default: "
            f"${MAX_PRICE_DEVIATION_ENV}, else {DEFAULT_MAX_PRICE_DEVIATION * 100:.1f}%%). "
            "A larger gap means the two venues are quoting different markets or one "
            "is stale; the order is refused rather than sized from the wrong price. "
            "A fraction, not a percentage: 0.05 means five percent."
        ),
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        default=None,
        help=(
            "the crypto pairs the live market section shows, e.g. BTC/USD ETH/USD "
            "(default: $TRADINGAGENTS_CRYPTO_SYMBOLS, else none). The list is "
            "operator-controlled and each pair must be listed by the venue; an "
            "unsupported pair is reported, never substituted."
        ),
    )
    parser.add_argument(
        "--crypto-live",
        action="store_true",
        default=False,
        help=(
            "start the live crypto market-data engine (Binance PUBLIC market data: "
            "no API key, no registration) so the dashboard's live section shows a "
            "real feed. Off by default: with it off, the crypto section reports "
            "'not started' instead of showing data it does not have."
        ),
    )
    parser.add_argument(
        "--paper-fractional-shares",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "allow the paper worker to size a fractional quantity (default: "
            f"${FRACTIONAL_SHARES_ENV}, else fractional, the one parity constant). "
            "Needed for a high-priced symbol such as SPY under a small notional, "
            "where whole shares would floor to zero. The account must support "
            "fractional trading, or the broker will reject the order; the $25 "
            "notional cap and long-only rule are unchanged."
        ),
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "use the non-genuine stub decision source (offline demonstration); "
            "decisions are labelled synthetic_fixture"
        ),
    )
    parser.add_argument(
        "--auto-advance",
        type=int,
        default=int(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_SESSIONS", "0") or 0),
        help=(
            "advance this many sessions every interval, in-process (default: "
            "$TRADINGAGENTS_AUTO_ADVANCE_SESSIONS, else 0 = monitor only). "
            "Requires --allow-control."
        ),
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_INTERVAL", "60") or 60),
        help="seconds between auto-advance ticks (default: 60)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard as ``python -m ...dashboard.server``.

    A thin wrapper over :func:`serve` for a platform (Railway, Docker) that runs
    a module directly rather than the ``serve`` subcommand. ``--root`` falls back
    to ``TRADINGAGENTS_EXPERIMENT_ROOT`` and the port to ``$PORT``, so the start
    command can be as short as ``python -m ...dashboard.server``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.root:
        parser.error(f"an experiment directory is required: pass --root or set {ROOT_ENV}")
    port = args.port if args.port is not None else int(os.environ.get("PORT", "8000"))
    try:
        provider, max_age, whole_shares = resolve_settings(
            provider=args.paper_market_data,
            max_age=args.paper_snapshot_max_age_seconds,
            fractional=args.paper_fractional_shares,
        )
        decision_source = resolve_paper_settings(
            decision_source=args.paper_decision_source,
        )[3]
        execution_provider = _resolve_execution_provider(args)
        max_price_deviation = _resolve_max_price_deviation(args)
    except MarketDataWiringError as exc:
        parser.error(str(exc))
    print(
        f"paper market data: {describe_provider(provider)}"
        + (
            f"; snapshot max age {max_age:.0f}s"
            if provider == "alpaca_snapshot" and max_age is not None
            else ""
        ),
    )
    print(f"paper decision source: {describe_decision_source(decision_source)}")
    print(
        f"paper execution provider: {describe_execution_provider(execution_provider)}",
    )
    # The endpoint the provider resolves to. The provider name alone does not say
    # which host receives the order, and a mismatch between the two is exactly
    # what made a Bybit selection appear to trade on Alpaca. Secret-free: a URL.
    print(
        "paper execution endpoint: "
        f"{_endpoint_for_provider(execution_provider)}",
    )
    crypto_symbols = _crypto_symbols(args)
    try:
        serve(
            args.root,
            host=args.host,
            port=port,
            token=args.token,
            allow_control=args.allow_control,
            stub=args.stub,
            auto_advance=args.auto_advance,
            auto_interval=args.auto_interval,
            allow_paper_trading=args.allow_paper_trading,
            paper_order_notional=args.paper_order_notional,
            market_data_factory=_market_data_factory_for(provider, max_age_seconds=max_age),
            paper_whole_shares=whole_shares,
            paper_market_data_provider=provider,
            paper_market_data_max_age_seconds=max_age,
            paper_decision_source=decision_source,
            paper_max_price_deviation=max_price_deviation,
            paper_execution_provider=execution_provider,
            crypto_symbols=crypto_symbols,
            crypto_live=bool(getattr(args, "crypto_live", False)),
        )
    except DashboardAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def serve(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    allow_control: bool = False,
    max_sessions_per_call: int = DEFAULT_MAX_SESSIONS_PER_CALL,
    stub: bool = False,
    auto_advance: int = 0,
    auto_interval: float = 60.0,
    allow_paper_trading: bool = False,
    paper_order_notional: float = 25.0,
    broker_factory=None,
    market_data_factory=None,
    paper_whole_shares: bool = WHOLE_SHARES,
    paper_market_data_provider: str = PROVIDER_BAR_OPEN,
    paper_market_data_max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    paper_decision_source: str = DECISION_SOURCE_HISTORY,
    paper_max_price_deviation: float = DEFAULT_MAX_PRICE_DEVIATION,
    paper_execution_provider: str = EXECUTION_PROVIDER_ALPACA_PAPER,
    crypto_symbols: tuple[str, ...] = (),
    crypto_live: bool = False,
) -> None:
    """Run the dashboard until interrupted.

    Refuses to bind a non-loopback host with no token, so the control surface is
    never served unauthenticated on a public interface.

    ``auto_advance`` turns the dashboard into the *engine* as well as the monitor:
    when it is a positive N, a background thread advances N unprocessed sessions
    every ``auto_interval`` seconds, through the same locked ``run_sessions`` the
    Run button uses. One process, one writer, so it cannot race the state file. It
    is off by default (``0``), which leaves the server a pure monitor.

    ``allow_paper_trading`` enables the START/STOP control for the configured
    paper execution provider. It is off by default, so the dashboard cannot place a paper order
    unless the operator opts in; it never enables live trading.

    ``crypto_symbols`` selects the crypto pairs the live market section shows.
    ``crypto_live`` starts the Binance **public** market-data engine (no key, no
    registration) so the dashboard's live section is fed by a real feed rather
    than by an empty engine. It is off by default: with no symbols and no live
    flag the crypto section reports *not started* instead of inventing data.
    """
    # Ensure crypto experiment exists (idempotent, safe to call on every startup)
    _ensure_experiment(root, crypto_symbols=crypto_symbols)

    resolved = resolve_token(token)
    require_bindable(host, resolved)
    crypto_engine = None
    if crypto_live and crypto_symbols:
        crypto_engine = _build_crypto_engine(crypto_symbols)
    dashboard = Dashboard(
        root,
        token=resolved,
        allow_control=allow_control,
        max_sessions_per_call=max_sessions_per_call,
        stub=stub,
        allow_paper_trading=allow_paper_trading,
        paper_order_notional=paper_order_notional,
        broker_factory=broker_factory,
        market_data_factory=market_data_factory,
        paper_whole_shares=paper_whole_shares,
        paper_market_data_provider=paper_market_data_provider,
        paper_market_data_max_age_seconds=paper_market_data_max_age_seconds,
        paper_decision_source=paper_decision_source,
        paper_max_price_deviation=paper_max_price_deviation,
        paper_execution_provider=paper_execution_provider,
        crypto_engine=crypto_engine,
        crypto_symbols=tuple(crypto_symbols),
    )
    httpd = ThreadingHTTPServer((host, port), make_handler(dashboard))
    httpd.daemon_threads = True
    print(f"dashboard on http://{host}:{port}  (root: {Path(root).resolve()})")
    print(f"  auth: {'token required' if resolved else 'loopback only, no token'}")
    print(f"  control: {'enabled' if allow_control else 'read-only'}")
    print(
        "  paper trading: "
        + (
            f"enabled ({banner_for_provider(paper_execution_provider)}, order notional {paper_order_notional:g})"
            if allow_paper_trading
            else "disabled (pass --allow-paper-trading to enable)"
        )
    )
    if auto_advance:
        if not allow_control:
            print("  auto-advance: refused (needs --allow-control)", flush=True)
        else:
            _start_auto_advance(dashboard, auto_advance, auto_interval)
            print(
                f"  auto-advance: {auto_advance} session(s) every {auto_interval:g}s",
                flush=True,
            )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the paper worker's loop before the process exits, so a shutdown
        # never leaves a cycle mid-flight and never places a new order. This is
        # best-effort: a stop failure must not mask the server's own shutdown.
        try:
            dashboard.paper.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            print(f"  paper trading: stop on shutdown failed: {type(exc).__name__}", flush=True)
        httpd.server_close()


def _reasons_suffix(reasons: object) -> str:
    """Name why advanced sessions took no decision, so a tick that looks idle explains itself."""
    if not reasons:
        return ""
    return f" (no decision: {', '.join(reasons)})"


def _start_auto_advance(dashboard: Dashboard, sessions: int, interval: float) -> threading.Thread:
    """Advance the simulation on a timer, in-process, behind the control lock.

    A daemon thread so it never keeps the process alive. Each tick logs one line,
    so an operator can see from the platform's logs that sessions are progressing.
    A tick that finds nothing to do, or that loses the lock to a manual Run, says
    so rather than failing silently.
    """

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                result = dashboard.run_sessions(sessions)
            except PermissionError:
                print("[auto] control is disabled; stopping auto-advance", flush=True)
                return
            except RuntimeError:
                print("[auto] a run is already in progress; skipping this tick", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001 - a tick failure must not kill the thread
                print(f"[auto] tick failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            done = result["processed"]
            refresh = result.get("refresh")
            line = refresh.get("summary") if isinstance(refresh, dict) else None
            if not done:
                print(
                    "[auto] no unprocessed sessions remain; nothing to advance"
                    + (f" | {line}" if line else ""),
                    flush=True,
                )
                continue
            print(
                f"[auto] advanced {result['processed_count']} session(s): "
                f"{done[0]}..{done[-1]}; {result['remaining_after']} remaining; "
                f"{result.get('decision_count', 0)} decision(s)"
                + _reasons_suffix(result.get("no_decision_reasons"))
                + (f" | {line}" if line else ""),
                flush=True,
            )

    thread = threading.Thread(target=loop, name="auto-advance", daemon=True)
    thread.start()
    return thread


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>TradingAgents · Paper</title>
<style>
  :root{
    --bg:#0a0e13; --card:#141c25; --line:#212c38; --line2:#2b3846;
    --fg:#e8eef5; --mut:#8ba0b5; --mut2:#63788d;
    --acc:#4ea1ff; --ok:#37d67a; --warn:#ffc74d; --bad:#ff6b6b;
    --radius:14px; --radius-sm:10px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;
    padding:env(safe-area-inset-top) 0 calc(env(safe-area-inset-bottom) + 8px)}
  .wrap{max-width:1180px;margin:0 auto;padding:0 16px}
  header{position:sticky;top:0;z-index:20;background:rgba(10,14,19,.86);
    backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
  .hrow{display:flex;gap:14px;align-items:center;padding:12px 0}
  h1{font-size:17px;margin:0;letter-spacing:-.01em}
  .sub{color:var(--mut2);font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    margin-top:2px;word-break:break-all}
  .grow{flex:1;min-width:0}
  .badge{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;
    font-size:12px;font-weight:700;white-space:nowrap;border:1px solid transparent}
  .b-ok{background:rgba(55,214,122,.14);color:var(--ok);border-color:rgba(55,214,122,.3)}
  .b-warn{background:rgba(255,199,77,.14);color:var(--warn);border-color:rgba(255,199,77,.3)}
  .b-bad{background:rgba(255,107,107,.14);color:var(--bad);border-color:rgba(255,107,107,.3)}
  .b-mut{background:rgba(139,160,181,.14);color:var(--mut);border-color:rgba(139,160,181,.28)}
  .dot{width:7px;height:7px;border-radius:50%;background:currentColor;display:inline-block}
  nav{display:flex;gap:4px;overflow-x:auto;padding:0 0 8px;scrollbar-width:none}
  nav::-webkit-scrollbar{display:none}
  nav button{flex:0 0 auto;background:transparent;color:var(--mut);border:1px solid transparent;
    padding:7px 13px;border-radius:999px;font-size:13.5px;font-weight:600;cursor:pointer;
    font-family:inherit;transition:all .12s}
  nav button:hover{color:var(--fg)}
  nav button.on{color:var(--fg);border-color:var(--line2);background:var(--card)}
  section{display:none;animation:fade .18s ease-out}
  section.on{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
  .grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
  .grid.two{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:14px 16px;margin:0}
  .card h2{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
    margin:0 0 12px;font-weight:700;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .card h2 .spacer{flex:1}
  .card h2 .tag{text-transform:none;letter-spacing:0;font-weight:600;font-size:11px;color:var(--mut2)}
  .stat{display:flex;flex-direction:column;gap:3px}
  .stat .label{color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
  .stat .value{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .stat .value.sm{font-size:16px}
  .stat .hint{color:var(--mut2);font-size:11.5px}
  .stats{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
  .kv{display:grid;grid-template-columns:1fr auto;gap:7px 14px;font-size:14px}
  .kv .k{color:var(--mut);min-width:0}
  .kv .v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;word-break:break-word}
  .kv .full{grid-column:1/3}
  .pos{color:var(--ok)} .neg{color:var(--bad)} .warnc{color:var(--warn)}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--mut2);
    word-break:break-all}
  .pill{display:inline-block;padding:2px 8px;border-radius:7px;font-size:11.5px;font-weight:700;
    letter-spacing:.02em;white-space:nowrap}
  .s-pending{background:rgba(139,160,181,.16);color:var(--mut)}
  .s-executed{background:rgba(78,161,255,.16);color:var(--acc)}
  .s-evaluated{background:rgba(55,214,122,.16);color:var(--ok)}
  .s-blocked{background:rgba(255,107,107,.16);color:var(--bad)}
  .s-unevaluable{background:rgba(255,199,77,.16);color:var(--warn)}
  .s-filled,.s-accepted,.s-new,.s-partially_filled{background:rgba(78,161,255,.16);color:var(--acc)}
  .s-canceled,.s-expired,.s-rejected{background:rgba(255,107,107,.16);color:var(--bad)}
  .state-RUNNING{background:rgba(55,214,122,.16);color:var(--ok)}
  .state-STARTING,.state-STOPPING{background:rgba(255,199,77,.16);color:var(--warn)}
  .state-STOPPED,.state-DISABLED{background:rgba(139,160,181,.16);color:var(--mut)}
  .state-ERROR{background:rgba(255,107,107,.16);color:var(--bad)}
  ul.clean{list-style:none;margin:0;padding:0}
  ul.clean li{border-bottom:1px solid var(--line);padding:11px 0;font-size:14px}
  ul.clean li:last-child{border-bottom:0}
  ul.clean li:first-child{padding-top:0}
  .err{background:rgba(255,107,107,.1);border:1px solid rgba(255,107,107,.35);
    border-radius:var(--radius);padding:11px 14px;margin:12px 0;font-size:14px;display:none}
  button.act{background:var(--acc);color:#04121f;border:0;padding:10px 16px;border-radius:var(--radius-sm);
    font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;transition:filter .12s}
  button.act:hover{filter:brightness(1.08)}
  button.act[disabled]{opacity:.4;cursor:not-allowed}
  button.act.stop{background:var(--bad);color:#1a0505}
  button.act.ghost{background:transparent;color:var(--fg);border:1px solid var(--line2)}
  input{background:#0b1118;color:var(--fg);border:1px solid var(--line2);border-radius:var(--radius-sm);
    padding:10px 12px;font-size:15px;width:100%;font-family:inherit}
  .muted{color:var(--mut);font-size:12.5px}
  .hide{display:none}
  .banner{background:linear-gradient(90deg,rgba(255,107,107,.1),rgba(255,107,107,.04));
    border:1px solid rgba(255,107,107,.35);color:#ffb9b9;border-radius:var(--radius);
    padding:8px 14px;margin:12px 0;font-size:12.5px;font-weight:700;letter-spacing:.05em;
    display:flex;align-items:center;gap:9px}
  .flag{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:var(--radius-sm);
    background:#0b1118;border:1px solid var(--line);font-size:13px}
  .flag .dot{flex:0 0 auto}
  .flags{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
  .flag.ok{border-color:rgba(55,214,122,.3)}
  .flag .name{flex:1;font-weight:600}
  .flag .state{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
  .flag.ok .state{color:var(--ok)} .flag.off .state{color:var(--mut2)}
  .timeline{display:flex;flex-direction:column;gap:2px}
  .step{display:flex;gap:11px;align-items:flex-start;padding:8px 0}
  .step .mark{flex:0 0 auto;width:22px;height:22px;border-radius:50%;display:flex;
    align-items:center;justify-content:center;font-size:11px;font-weight:800;
    background:var(--line);color:var(--mut2)}
  .step.done .mark{background:rgba(55,214,122,.18);color:var(--ok)}
  .step.wait .mark{background:rgba(255,199,77,.16);color:var(--warn)}
  .step .body{flex:1;min-width:0}
  .step .title{font-weight:600;font-size:13.5px}
  .step .meta{font-size:12px;color:var(--mut2);margin-top:1px;word-break:break-word}
  .tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;
    padding:8px 10px;border-bottom:1px solid var(--line2);font-weight:700;white-space:nowrap}
  td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;
    font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:0}
  td.num{text-align:right;white-space:nowrap}
  .empty{padding:18px 0;text-align:center;color:var(--mut2);font-size:13.5px}
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{background:#0b1118;border:1px solid var(--line);border-radius:8px;padding:5px 9px;
    font-size:12px;display:flex;gap:7px;align-items:baseline}
  .chip .ck{color:var(--mut2);font-size:11px}
  .chip .cv{font-weight:700;font-variant-numeric:tabular-nums}
  .chip.s-buy{background:rgba(55,214,122,.16);border-color:rgba(55,214,122,.35)}
  .chip.s-buy .cv{color:var(--ok)}
  .chip.s-sell{background:rgba(255,107,107,.16);border-color:rgba(255,107,107,.35)}
  .chip.s-sell .cv{color:var(--bad)}
  .chip.s-hold{background:rgba(255,199,77,.14);border-color:rgba(255,199,77,.30)}
  .chip.s-hold .cv{color:var(--warn)}
  .sec-title{font-size:12.5px;font-weight:700;color:var(--mut);text-transform:uppercase;
    letter-spacing:.06em;margin:22px 0 10px}
  .sec-title:first-child{margin-top:6px}
  @media (max-width:600px){
    .stat .value{font-size:19px}
    .hrow{padding:10px 0}
    h1{font-size:15.5px}
  }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="hrow">
      <div class="grow">
        <h1>TradingAgents · Paper</h1>
        <div class="sub" id="expid">loading…</div>
      </div>
      <span id="status" class="badge b-mut"><span class="dot"></span>…</span>
    </div>
    <nav id="nav" class="hide">
      <button data-t="overview" class="on">Overview</button>
      <button data-t="crypto">Crypto</button>
      <button data-t="engine">Engine</button>
      <button data-t="decisions">Decisions</button>
      <button data-t="positions">Positions</button>
      <button data-t="trades">Trades</button>
      <button data-t="journal">Journal</button>
      <button data-t="control">Control</button>
    </nav>
  </div>
</header>

<div class="wrap">
  <div id="err" class="err"></div>

  <div id="livewarn" class="banner"><span class="dot"></span>PAPER TRADING — NOT LIVE</div>

  <div id="auth" class="card">
    <h2>Access token</h2>
    <input id="token" type="password" placeholder="paste dashboard token" autocomplete="off">
    <div style="display:flex;gap:10px;align-items:center;margin-top:10px">
      <button class="act" onclick="saveToken()">Connect</button>
      <span class="muted">stored only in this browser</span>
    </div>
  </div>

  <main id="main" class="hide">
    <section id="overview" class="on">
      <div class="sec-title">Live engine</div>
      <div class="card">
        <h2>Is the system actually working? <span class="spacer"></span>
          <span class="tag" id="ov-updated">—</span></h2>
        <div id="ov-states" class="flags"></div>
        <p class="muted" id="ov-verdict" style="margin:12px 0 0"></p>
      </div>
      <div class="grid" style="margin-top:12px">
        <div class="card"><h2>Account (paper broker)</h2><div id="ov-account" class="stats"></div>
          <div id="ov-account-note" class="muted" style="margin-top:10px"></div></div>
        <div class="card"><h2>Activity</h2><div id="ov-activity" class="stats"></div></div>
      </div>
      <div class="sec-title">Simulation book <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">— the backtest ledger, separate from the paper account above</span></div>
      <div class="grid">
        <div class="card"><h2>Book</h2><div id="bookkv" class="kv"></div></div>
        <div class="card"><h2>Performance</h2><div id="perf" class="kv"></div></div>
      </div>
      <div class="grid two" style="margin-top:12px">
        <div class="card"><h2>Market data</h2><div id="market" class="kv"></div></div>
        <div class="card"><h2>Dataset</h2><div id="data" class="kv"></div></div>
      </div>
      <div id="warncard" class="card" style="display:none;margin-top:12px">
        <h2>Warnings</h2><ul class="clean" id="warnings"></ul>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Readiness</h2><div id="readiness"></div>
      </div>
    </section>

    <section id="crypto">
      <div class="card">
        <h2>Live crypto market <span class="spacer"></span>
          <span class="tag" id="cx-status">—</span></h2>
        <p class="muted" id="cx-detail" style="margin:0 0 12px"></p>
        <div id="cx-market"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>TJR strategy <span class="spacer"></span>
          <span class="tag" id="cx-strategy-status">—</span></h2>
        <p class="muted" style="margin:0 0 12px">
          The stage the unchanged TJR rule chain reached. A BUY or SELL appears only
          when every condition completed; otherwise the row shows the stage and the
          reason it stopped.
        </p>
        <div id="cx-strategy"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Symbol validation</h2>
        <p class="muted" style="margin:0 0 12px">
          Which pairs the market-data venue and the execution venue each accept.
          The venues' own listings are read when the paper worker starts, so a
          rejected pair is shown with the venue's reason instead of silently
          missing.
        </p>
        <div id="cx-symbols"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>System health</h2>
        <div id="cx-health" class="kv"></div>
      </div>
    </section>

    <section id="engine">
      <div class="grid two">
        <div class="card">
          <h2>Worker</h2>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
            <span id="pstate" class="pill state-STOPPED">…</span>
            <span class="grow"></span>
            <button class="act" id="pstart" onclick="doPaperStart()">START PAPER TRADING</button>
            <button class="act stop" id="pstop" onclick="doPaperStop()">STOP PAPER TRADING</button>
          </div>
          <p class="muted" id="pnote" style="margin:0 0 12px"></p>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 12px">
            <button class="act" id="paccountcheck" onclick="doBybitAccountCheck()">READ BYBIT DEMO BALANCE ONLY</button>
            <span class="muted" id="paccountcheck-result">No account check has been run.</span>
          </div>
          <div id="eng-worker" class="kv"></div>
        </div>
        <div class="card">
          <h2>Pipeline <span class="spacer"></span><span class="tag" id="eng-cycle">—</span></h2>
          <div id="eng-pipeline" class="timeline"></div>
          <p class="muted" id="eng-reason" style="margin:12px 0 0"></p>
        </div>
      </div>
      <div class="grid two" style="margin-top:12px">
        <div class="card"><h2>Price seam</h2><div id="eng-market" class="kv"></div>
          <p class="muted" id="eng-markethint" style="margin:10px 0 0"></p></div>
        <div class="card"><h2>Heartbeat &amp; safety</h2><div id="eng-risk" class="kv"></div></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Decision statistics <span class="spacer"></span>
          <span class="tag" id="eng-stats-note">—</span></h2>
        <div id="eng-stats"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Exit policy <span class="tag">stop-loss / take-profit, recorded from the strategy</span></h2>
        <div id="eng-exits"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Last cycle <span class="spacer"></span>
          <button class="act ghost" onclick="refreshEngine()">Refresh</button></h2>
        <div id="eng-lasterr" class="err" style="display:none"></div>
        <div id="eng-cycle-json" class="mono">—</div>
      </div>
    </section>

    <section id="decisions">
      <div class="card">
        <h2>Decision cadence <span class="spacer"></span><span class="tag" id="dec-cadence"></span></h2>
        <p class="muted" id="dec-cadence-note" style="margin:0"></p>
      </div>
      <div id="declist" style="margin-top:12px"></div>
    </section>

    <section id="positions">
      <div class="card">
        <h2>Open positions <span class="spacer"></span><span class="tag" id="pos-note"></span></h2>
        <div id="pos-body"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Recorded exit levels</h2>
        <div id="pos-exits"></div>
      </div>
    </section>

    <section id="trades">
      <div class="card">
        <h2>Orders <span class="spacer"></span><span class="tag" id="ord-note"></span></h2>
        <div id="ord-body"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Filled executions <span class="spacer"></span><span class="tag" id="exec-note"></span></h2>
        <div id="exec-body"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Settled trades <span class="spacer"></span><span class="tag" id="trade-note"></span></h2>
        <div id="trade-body"></div>
      </div>
    </section>

    <section id="journal">
      <div class="card">
        <h2>Journal (latest 100) <span class="spacer"></span>
          <span class="tag">append-only event log</span></h2>
        <div id="jlist"></div>
      </div>
    </section>

    <section id="control">
      <div class="card">
        <h2>Advance simulation</h2>
        <p class="muted" id="ctlstate"></p>
        <div style="display:flex;gap:10px;align-items:center">
          <input id="nsessions" type="number" min="0" value="10" style="max-width:120px">
          <button class="act" id="runbtn" onclick="doRun()">Run sessions</button>
        </div>
        <p class="muted" style="margin-bottom:0">Simulation only. No broker order is ever placed by
          this control.</p>
      </div>
      <div class="card" style="margin-top:12px">
        <h2>Last run</h2><div id="lastrun" class="mono">—</div>
      </div>
    </section>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
let TOKEN = localStorage.getItem("ta_token") || "";
let timer = null;
let cryptoTimer = null;
let LAST = {};

function esc(v){
  if(v===null||v===undefined) return "N/A";
  return String(v).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
const money = (v) => (v===null||v===undefined||isNaN(v)) ? "N/A"
  : Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const pct = (v) => (v===null||v===undefined||isNaN(v)) ? "N/A" : (Number(v)*100).toFixed(2)+"%";
// Some feeds (Binance's priceChangePercent, for one) report a change as a
// percentage already -- 0.091 means 0.091%, not 9.1%. Scaling that with pct()
// would print it 100x too large, so it gets its own formatter.
const pctAlready = (v) => (v===null||v===undefined||isNaN(v)) ? "N/A" : Number(v).toFixed(2)+"%";
const num = (v,d) => (v===null||v===undefined||isNaN(v)) ? "N/A" : Number(v).toFixed(d===undefined?2:d);
const when = (v) => v ? new Date(v).toLocaleString() : "—";
const rel = (v) => {
  if(!v) return "never";
  const s = (Date.now() - new Date(v).getTime())/1000;
  if(isNaN(s)) return "—";
  if(s < 60) return Math.max(0,Math.round(s))+"s ago";
  if(s < 3600) return Math.round(s/60)+"m ago";
  if(s < 86400) return Math.round(s/3600)+"h ago";
  return Math.round(s/86400)+"d ago";
};
const cls = (v) => (v>0?"pos":(v<0?"neg":""));
const sgn = (v) => (v===null||v===undefined||isNaN(v)) ? "N/A" : (v>0?"+":"")+money(v);
const stat = (label,value,extra) => `<div class="stat"><div class="label">${esc(label)}</div>
  <div class="value ${(extra&&extra.sm)?"sm":""} ${(extra&&extra.cls)||""}">${value}</div>
  ${(extra&&extra.hint)?`<div class="hint">${esc(extra.hint)}</div>`:""}</div>`;
const kv = (obj,fmt) => Object.entries(obj).map(([k,v]) =>
  `<div class="k">${esc(k)}</div><div class="v">${fmt?fmt(k,v):(v===null||v===undefined?"N/A":esc(v))}</div>`
).join("");
const full = (t) => t ? `<div class="k full mono" style="color:var(--mut2)">${esc(t)}</div>` : "";
const empty = (t) => `<div class="empty">${esc(t)}</div>`;
const flag = (ok,label,state) => `<div class="flag ${ok?"ok":"off"}"><span class="dot"></span>
  <span class="name">${esc(label)}</span><span class="state">${esc(state)}</span></div>`;

function showErr(msg){
  const e = $("err");
  if(!msg){ e.style.display="none"; return; }
  e.textContent = msg; e.style.display="block";
}
function saveToken(){
  TOKEN = $("token").value.trim();
  localStorage.setItem("ta_token", TOKEN);
  boot();
}
async function api(path, opts){
  const o = Object.assign({headers:{}}, opts||{});
  o.headers["Authorization"] = "Bearer " + TOKEN;
  const r = await fetch(path, o);
  if(r.status === 401){ throw new Error("unauthorized — check the token"); }
  if(!r.ok){
    let m = "HTTP " + r.status;
    try{ const j = await r.json(); if(j.error) m = j.error; }catch(e){}
    throw new Error(m);
  }
  return r.json();
}

// -- Overview ---------------------------------------------------------------
function renderOverview(s){
  $("expid").textContent = s.experiment_id + "  ·  " + s.strategy_id;
  const rd = s.readiness || {};
  const rdCls = /READY/i.test(rd.status)?"b-ok":(/BLOCKED/i.test(rd.status)?"b-bad":"b-warn");
  $("status").className = "badge " + rdCls;
  $("status").innerHTML = `<span class="dot"></span>${esc(rd.status||"—")}`;
  $("readiness").innerHTML = `<span class="badge ${rdCls}">${esc(rd.status)}</span>` +
    ((rd.blockers&&rd.blockers.length)
      ? '<ul class="clean">'+rd.blockers.map(b=>`<li>${esc(b)}</li>`).join("")+"</ul>"
      : '<p class="muted" style="margin:10px 0 0">No blockers.</p>');

  const c = s.counts || {};
  $("ov-activity").innerHTML =
    stat("Sessions processed", esc(c.sessions_processed), {hint:"last "+(s.last_session_processed||"—")}) +
    stat("Decisions", esc(c.decisions_total)) +
    stat("Genuine inferences", esc(c.genuine_inferences), {hint:"model-produced, not blocked"}) +
    stat("Orders submitted", esc((s.orders||{}).submitted)) +
    stat("Filled orders", esc(c.filled_orders)) +
    stat("Evaluated", esc(c.evaluated));

  const cad = s.cadence || {};
  $("dec-cadence").textContent = "every " + (cad.decision_frequency_sessions||"?") + " session(s)";
  $("dec-cadence-note").textContent = cad.max_decisions_reached
    ? `max_decisions (${cad.max_decisions}) reached — no later session decides.`
    : (cad.next_decision_session
        ? `Next decision due at ${cad.next_decision_session}. ${cad.decisions_taken||0}/${cad.max_decisions||"?"} decisions taken.`
        : "No further decision session in the loaded data.");

  $("bookkv").innerHTML = kv({
    "Initial cash": money(s.book.initial_cash),
    "Cash": money(s.book.cash),
    "Market value": money(s.book.market_value),
    "Equity": money(s.book.equity),
    "Realized P&L": money(s.book.realized_pnl),
    "Unrealized P&L": money(s.book.unrealized_pnl),
    "Total fees": money(s.book.total_fees),
    "Resting orders": s.book.resting_orders,
  });
  const p = s.performance || {};
  $("perf").innerHTML = kv({
    "Total return": pct(p.total_return),
    "Max drawdown": pct(p.max_drawdown),
    "Sharpe": p.sharpe_ratio===null||p.sharpe_ratio===undefined?"N/A":p.sharpe_ratio.toFixed(2),
    "Closed outcomes": ((p.sample||{}).closed_outcomes),
    "Sample limited": p.sample_limited?"yes":"no",
  }) + full(p.sample_limited ? "Too few closed outcomes for reliable metrics." : "");

  const f = s.freshness || {};
  const ageText = (f.data_age_days===null||f.data_age_days===undefined)?"unknown":(f.data_age_days+" day(s)");
  const stCls = (f.market_status==="stale"||f.market_status==="data_source_unavailable")?"b-bad":
    (f.market_status==="up_to_date"?"b-ok":"b-mut");
  $("market").innerHTML =
    `<div class="k full" style="margin-bottom:8px"><span class="badge ${stCls}">${esc(f.market_status||"—")}</span></div>` +
    kv({
      "Latest available bar": f.latest_available_session||"—",
      "Last source bar": f.last_source_bar||"—",
      "Last processed session": f.last_processed_session||"—",
      "Data age": ageText,
      "Pending sessions": f.pending_sessions===undefined?"—":f.pending_sessions,
      "Source health": f.source_status==="failed"?"FAILING":"ok",
      "Last refresh": f.last_refresh_at||"never",
      "Fetches today": (f.fetches_today===undefined?"—":f.fetches_today)+" / "+(f.fetch_quota_per_day===undefined?"—":f.fetch_quota_per_day),
    }) +
    (f.source_status==="failed"
      ? `<div class="k full" style="color:var(--bad);margin-top:8px">Data source health: FAILING (${esc(f.source_failure_kind||"unknown")}) — ${esc(f.source_status_detail||f.reason||"")}</div>`
      : "") +
    full(f.reason||"") +
    full('This is the historical experiment dataset — the newest daily session is a past bar, not a live tick. It is separate from the price the paper worker sizes from; see "Paper price source".');

  $("data").innerHTML = kv({
    "Source": s.data.source_name,
    "Kind": s.data.source_kind,
    "Current market": s.data.is_current_market?"yes":"no",
    "Dataset": (s.data.dataset_start||"?")+" → "+(s.data.dataset_end||"?"),
    "Sessions available": s.data.sessions_available,
    "Last processed": s.last_session_processed||"—",
  }) + full(s.data.note||"");

  const warn = s.warnings || [];
  $("warncard").style.display = warn.length?"block":"none";
  $("warnings").innerHTML = warn.map(w=>`<li style="color:var(--warn)">${esc(w)}</li>`).join("");
}

// -- Engine -----------------------------------------------------------------
function paperState(status){
  const known = ["STOPPED","STARTING","RUNNING","STOPPING","ERROR","DISABLED"];
  const s = known.includes(status) ? status : "STOPPED";
  const el = $("pstate");
  el.textContent = status || "STOPPED";
  el.className = "pill state-" + s;
  $("pstart").disabled = status==="RUNNING" || status==="STARTING";
  $("pstop").disabled = status==="STOPPED" || status==="STOPPING";
}

function renderEngine(e){
  const w = e.worker || {};
  paperState(w.status);
  $("pnote").textContent = w.endpoint
    ? `Mode ${e.mode} · endpoint ${w.endpoint} · paper-only ${w.paper_endpoint_only?"yes":"NO"}`
    : (e.no_trade_reason || "paper trading is not configured on this dashboard");
  $("ov-updated").textContent = "updated " + rel(e.timestamps && e.timestamps.last_cycle_at);

  const st = e.states || {};
  const mdStatus = (e.market_data && e.market_data.status) || "unknown";
  $("ov-states").innerHTML =
    flag(st.worker_running, "Worker loop running", st.worker_running?"running":"stopped") +
    flag(st.market_data_available, "Market data available", st.market_data_available?"ok":mdStatus) +
    flag(st.strategy_evaluating, "Strategy evaluating", st.strategy_evaluating?("last "+rel(e.timestamps&&e.timestamps.last_strategy_evaluation_at)):"never") +
    flag(st.decision_generated, "Decision generated", st.decision_generated?"yes":"none yet") +
    flag(st.order_submitted, "Order submitted", st.order_submitted?"yes":"none") +
    flag(st.order_filled, "Order filled", st.order_filled?"yes":"none");

  const parts = [];
  if(!st.worker_running) parts.push("The worker is not running, so nothing is being analysed or traded.");
  else if(!st.market_data_available) parts.push("The worker is running but its price seam reports no usable price.");
  else if(!st.strategy_evaluating) parts.push("The worker is running and priced, but the strategy has not been evaluated yet.");
  else if(!st.order_submitted) parts.push("The strategy is being evaluated but has not produced a tradeable decision yet.");
  else parts.push("The strategy is evaluating and has produced orders.");
  if(e.no_trade_reason) parts.push("Reason no order was placed: " + e.no_trade_reason);
  $("ov-verdict").textContent = parts.join(" ");

  const hb = e.heartbeat || {};
  $("eng-worker").innerHTML = kv({
    "Status": w.status||"—",
    "Loop thread": w.loop_running?"alive":"not running",
    "Strategy": w.strategy_id||"—",
    "Endpoint": w.endpoint||"—",
    "Paper-only": w.paper_endpoint_only?"yes":"NO",
    "Started": when(e.timestamps&&e.timestamps.worker_started_at),
    "Stopped": (e.timestamps&&e.timestamps.worker_stopped_at)?when(e.timestamps.worker_stopped_at):"—",
    "Cycles completed": hb.cycles_completed===undefined?"—":hb.cycles_completed,
    "Poll interval": hb.poll_interval_seconds===undefined?"—":(hb.poll_interval_seconds+"s"),
    "Next cycle": when(hb.next_cycle_at),
    "Last cycle": rel(hb.last_cycle_at),
  });

  const ts = e.timestamps || {};
  const steps = [
    ["Market data", st.market_data_available,
      ((e.market_data&&e.market_data.provider)||"unknown") + " · " + mdStatus, null],
    ["Strategy evaluated", st.strategy_evaluating,
      ts.last_strategy_evaluation_at ? when(ts.last_strategy_evaluation_at) : "never", null],
    ["Decision generated", st.decision_generated,
      e.no_trade_reason || "a decision is in hand", null],
    ["Order submitted", st.order_submitted,
      st.order_submitted ? ("last " + rel(ts.last_order_submitted_at)) : "none yet", null],
    ["Order filled", st.order_filled,
      st.order_filled ? ((ts.last_fill_symbol||"?") + " · " + rel(ts.last_fill_at)) : "none yet", null],
  ];
  $("eng-pipeline").innerHTML = steps.map(([t,ok,meta],i)=>{
    const klass = ok ? "done" : (i===0 ? "wait" : "");
    return `<div class="step ${klass}"><div class="mark">${ok?"✓":(i+1)}</div><div class="body">
      <div class="title">${esc(t)}</div><div class="meta">${esc(meta)}</div></div></div>`;
  }).join("");
  $("eng-reason").innerHTML = (()=>{
    const b = e.blocker;
    if (b) {
      const cls = b.operator_action_required ? "b-bad" : "b-warn";
      return `<span class="badge ${cls}">${esc(b.code)}</span> ` +
        (b.detail ? esc(b.detail) : "") +
        (b.operator_action_required ? " <strong>— operator action may be required.</strong>" : "");
    }
    if (e.no_trade_reason) return "No order: " + esc(e.no_trade_reason);
    if (st.order_submitted) return "The last cycle placed an order.";
    return "";
  })();
  $("eng-cycle").textContent = "last cycle " + rel(ts.last_cycle_at);

  const md = e.market_data || {};
  $("eng-market").innerHTML = kv({
    "Provider": md.provider||md.provider_configured||"unknown",
    "Kind": md.kind||"—",
    "Status": md.status||"not started",
    "Last price": md.last_price===null||md.last_price===undefined?"—":money(md.last_price),
    "Observed at": when(md.last_observed_at),
    "Age": md.age_seconds===null||md.age_seconds===undefined?"—":(num(md.age_seconds,1)+"s"),
    "Max age": md.max_age_seconds===null||md.max_age_seconds===undefined?"—":(md.max_age_seconds+"s"),
  }) + full(md.status_detail||"");
  $("eng-markethint").textContent = md.kind === "current"
    ? "Paper price source: a current Alpaca market-data snapshot (feed=iex), refused when stale. This is the only price the paper worker sizes from."
    : "Paper price source: the experiment's historical daily-bar open for its latest session — a past-session price, not a live quote. This is the only price the paper worker sizes from; the dashboard's Dataset panel describes the same historical series.";

  const rl = LAST.risk || {};
  $("eng-risk").innerHTML = kv({
    "Last position sync": rel(ts.last_position_sync_at),
    "Last account read": rel(ts.last_account_at),
    "Max order notional": money(rl.max_order_notional),
    "Max open positions": rl.max_open_positions,
    "Max daily orders": rl.max_daily_orders,
    "Shorting": rl.allow_short?"ENABLED":"disabled",
    "Share sizing": rl.whole_shares?"whole shares only":"fractional allowed",
  });
  $("eng-lasterr").style.display = e.last_error?"block":"none";
  $("eng-lasterr").textContent = e.last_error||"";
  $("eng-cycle-json").textContent = LAST.lastCycle ? JSON.stringify(LAST.lastCycle, null, 2) : "—";

  const stops = LAST.stops || {};
  $("eng-exits").innerHTML = Object.keys(stops).length
    ? '<ul class="clean">'+Object.entries(stops).map(([s,v])=>`<li><b>${esc(s)}</b>
        <div class="mono">stop ${money(v.stop_loss)} · target ${money(v.take_profit)} · decision ${esc(v.decision_id||"—")}</div></li>`).join("")+"</ul>"
    : empty("No exit levels recorded. Stops and targets come only from decisions that carried them.");

  renderDecisionStats(e.statistics);
}

function renderDecisionStats(stats){
  const s = stats || {};
  const strat = s.strategy;
  const work = s.worker;
  $("eng-stats-note").textContent = s.last_evaluated_at
    ? "last evaluation " + rel(s.last_evaluated_at)
    : "no evaluation yet";
  if(!strat && !work){
    $("eng-stats").innerHTML = empty("No decision statistics yet. They appear after the strategy is evaluated.");
    return;
  }
  const parts = [];
  if(strat){
    const sides = strat.side_counts || {};
    const evals = strat.evaluations;
    parts.push(`<h3 style="margin:0 0 8px">Strategy ratings
      <span class="tag">${evals===undefined?"—":evals} evaluation(s)</span></h3>`);
    parts.push('<div class="chips">'+
      statChip("BUY", sides.buy, "s-buy") +
      statChip("SELL", sides.sell, "s-sell") +
      statChip("HOLD", sides.hold, "s-hold") + "</div>");
    if(strat.rating_counts && Object.keys(strat.rating_counts).length){
      parts.push('<div class="mono" style="margin-top:6px">by rating: '+
        Object.entries(strat.rating_counts).map(([k,v])=>`${esc(k)} ${v}`).join(" · ")+"</div>");
    }
    if(strat.last_hold_reason){
      parts.push(`<p class="muted" style="margin:8px 0 0">Last HOLD reason: “${esc(strat.last_hold_reason)}”</p>`);
    }
  }
  if(work){
    const byCode = work.no_trade_by_code || {};
    const codes = Object.keys(byCode);
    parts.push(`<h3 style="margin:14px 0 8px">Worker outcomes
      <span class="tag">${work.decisions_settled===undefined?"—":work.decisions_settled} settled</span></h3>`);
    parts.push('<div class="chips">'+
      statChip("Orders produced", work.orders_produced, "s-buy") +
      statChip("No-trade", work.no_trade_total, "s-hold") +
      statChip("Risk rejected", work.risk_rejections, "s-sell") + "</div>");
    if(work.orders_by_side && Object.keys(work.orders_by_side).length){
      parts.push('<div class="mono" style="margin-top:6px">orders by side: '+
        Object.entries(work.orders_by_side).map(([k,v])=>`${esc(k)} ${v}`).join(" · ")+"</div>");
    }
    parts.push('<div style="margin-top:8px">'+(codes.length
      ? '<div class="kv">'+codes.map(c=>`<div class="k">${esc(c)}</div><div class="v">${byCode[c]}</div>`).join("")+'</div>'
      : '<p class="muted" style="margin:0">No no-trade outcomes recorded.</p>')+'</div>');
  }
  $("eng-stats").innerHTML = parts.join("");
}

function statChip(label, value, cls){
  const n = (value===null||value===undefined) ? "—" : value;
  return `<span class="chip ${cls||""}"><span class="ck">${esc(label)}</span><span class="cv">${esc(String(n))}</span></span>`;
}

async function refreshEngine(){
  try{
    const [p, e] = await Promise.all([api("/api/paper/status"), api("/api/engine")]);
    LAST.risk = p.risk_limits;
    LAST.stops = p.stops;
    LAST.lastCycle = p.last_cycle;
    renderEngine(e);
  }catch(err){ showErr(err.message); }
}
async function doPaperStart(){
  $("pstart").disabled = true;
  try{ await api("/api/paper/start", {method:"POST", headers:{"X-Paper-Confirm":"yes"}}); }
  catch(e){ showErr(e.message); }
  finally{ refreshEngine(); }
}
async function doPaperStop(){
  $("pstop").disabled = true;
  try{ await api("/api/paper/stop", {method:"POST", headers:{"X-Paper-Confirm":"yes"}}); }
  catch(e){ showErr(e.message); }
  finally{ refreshEngine(); }
}
async function doBybitAccountCheck(){
  const button = $("paccountcheck");
  const result = $("paccountcheck-result");
  button.disabled = true;
  result.textContent = "Reading Bybit Demo wallet balance…";
  try{
    const data = await api("/api/paper/account-check");
    const a = data.account || {};
    result.textContent = "SUCCESS · " + a.currency + " equity: " +
      money(a.equity) + " · available: " + money(a.available_balance);
    result.className = "muted";
  }catch(e){
    result.textContent = "FAILED · " + e.message;
    result.className = "muted";
    showErr(e.message);
  }finally{
    button.disabled = false;
  }
}

// -- Decisions --------------------------------------------------------------
function indicatorChips(ind){
  if(!ind) return "";
  const pick = ["rsi","macd_hist","sma_trend","atr_pct","volume_ratio","realized_vol",
    "return_5d","return_20d","range_position","boll_percent_b"];
  const shown = pick.filter(k=>ind[k]!==null&&ind[k]!==undefined);
  if(!shown.length) return "";
  return '<div class="chips" style="margin-top:10px">'+shown
    .map(k=>`<span class="chip"><span class="ck">${esc(k)}</span><span class="cv">${num(ind[k],3)}</span></span>`).join("")+"</div>";
}
async function refreshDecisions(){
  try{
    const d = await api("/api/decisions");
    const list = d.decisions || [];
    let liveHtml = "";
    try{
      const live = await api("/api/crypto/strategy");
      const rows = live.symbols || [];
      liveHtml = '<div class="card"><h2>Live strategy evaluations <span class="spacer"></span><span class="tag">'+esc(live.status||"—")+'</span></h2>' +
        (rows.length ? rows.map(x=>`<div class="card" style="margin-top:10px">
          <h2>${esc(x.symbol)} <span class="spacer"></span><span class="pill ${x.actionable?"state-OK":"state-STOPPED"}">${esc(x.stage||"NO_SIGNAL")}</span>
          ${x.direction?`<span class="tag">${esc(x.direction)}</span>`:""}</h2>
          <div class="kv">
            <div class="k">Reason</div><div class="v">${esc(x.reason_code||"—")}</div>
            <div class="k">Detail</div><div class="v">${esc(x.detail||"—")}</div>
            <div class="k">Entry</div><div class="v">${x.entry===null||x.entry===undefined?"N/A":num(x.entry,4)}</div>
            <div class="k">Stop loss</div><div class="v">${x.stop_loss===null||x.stop_loss===undefined?"N/A":num(x.stop_loss,4)}</div>
            <div class="k">Take profit</div><div class="v">${x.take_profit===null||x.take_profit===undefined?"N/A":num(x.take_profit,4)}</div>
            <div class="k">Risk / reward</div><div class="v">${x.risk_reward===null||x.risk_reward===undefined?"N/A":num(x.risk_reward,2)}</div>
            <div class="k">Evaluated at</div><div class="v">${esc(when(x.evaluated_at))}</div>
          </div></div>`).join("") : '<p class="muted">No live strategy evaluation has been recorded yet.</p>') +
        '</div>';
    }catch(e){
      liveHtml = '<div class="card"><h2>Live strategy evaluations</h2><p class="muted">'+esc(e.message)+'</p></div>';
    }
    const historyHtml = list.length
      ? '<div class="sec-title">Historical experiment decisions</div>' + list.map(x=>`<div class="card">
        <h2>${esc(x.symbol)} <span class="spacer"></span>
          <span class="tag">${esc(x.decision_session)}</span>
          <span class="pill s-${esc(x.state)}">${esc(x.state)}</span></h2>
        <div class="kv">
          <div class="k">Action</div><div class="v">${esc(x.action||"—")}${x.rating?" · "+esc(x.rating):""}</div>
          <div class="k">Confidence</div><div class="v">${x.confidence===null||x.confidence===undefined?"N/A":num(x.confidence,2)}</div>
          <div class="k">Provenance</div><div class="v">${esc(x.provider||"—")}/${esc(x.model||"—")}</div>
          <div class="k">Source</div><div class="v">${esc(x.source||"—")}</div>
          <div class="k">Window</div><div class="v">${esc(x.window_start||"—")} → ${esc(x.window_end||"—")} (${x.indicator_bars===null||x.indicator_bars===undefined?"?":x.indicator_bars} bars)</div>
          <div class="k">Proposed stop / target</div><div class="v">${money(x.proposed_stop_loss)} / ${money(x.proposed_take_profit)}</div>
          <div class="k">Execution session</div><div class="v">${esc(x.execution_session||"—")}</div>
          <div class="k">Order</div><div class="v">${esc(x.order_status||"—")} · filled ${num(x.filled_quantity,4)}${x.average_fill_price?" @ "+money(x.average_fill_price):""} · fees ${money(x.fees)}</div>
        </div>
        ${x.reasoning_excerpt?`<p class="muted" style="margin:10px 0 0">“${esc(x.reasoning_excerpt)}”</p>`:""}
        ${x.rejection_reason?`<p class="muted warnc" style="margin:8px 0 0">Refused: ${esc(x.rejection_reason)}</p>`:""}
        ${indicatorChips(x.indicators)}
        ${x.evaluation?`<div class="mono" style="margin-top:8px">outcome: net ${money(x.evaluation.net_pnl)}${(x.evaluation.return_pct!==null&&x.evaluation.return_pct!==undefined)?" ("+pct(x.evaluation.return_pct)+")":""} · exit ${esc(x.evaluation.evaluation_session||"—")}</div>`:""}
        ${x.record?"":'<div class="mono" style="margin-top:8px">no journal record found for this decision — provenance unavailable</div>'}
      </div>`).join("")
      : '<div class="sec-title">Historical experiment decisions</div><div class="card">'+empty("No historical experiment decisions yet.")+'</div>';
    $("declist").innerHTML = liveHtml + historyHtml;
  }catch(e){ showErr(e.message); }
}

// -- Positions --------------------------------------------------------------
async function refreshPositions(){
  try{
    const [pos, p] = await Promise.all([api("/api/positions"), api("/api/paper/status")]);
    LAST.stops = p.stops;
    const rows = pos.positions || [];
    $("pos-note").textContent = "synced " + rel(pos.positions_checked_at);
    $("pos-body").innerHTML = rows.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Symbol</th><th class="num">Qty</th><th class="num">Entry</th>
        <th class="num">Current</th><th class="num">Value</th><th class="num">uP&L</th>
        <th class="num">Stop</th><th class="num">Target</th><th>Ledger</th></tr></thead>
      <tbody>${rows.map(x=>`<tr>
        <td><b>${esc(x.symbol)}</b><div class="mono">${esc(x.side)}</div></td>
        <td class="num">${num(x.quantity,4)}</td>
        <td class="num">${money(x.entry_price)}</td>
        <td class="num">${money(x.current_price)}</td>
        <td class="num">${money(x.market_value)}</td>
        <td class="num ${cls(x.unrealized_pnl)}">${sgn(x.unrealized_pnl)}</td>
        <td class="num">${money(x.stop_loss)}</td>
        <td class="num">${money(x.take_profit)}</td>
        <td>${x.reconciled?'<span class="pill s-evaluated">matched</span>':'<span class="pill s-unevaluable">broker only</span>'}</td>
      </tr>`).join("")}</tbody></table></div>`
      : empty("No open positions on the paper account.");

    const stops = LAST.stops || {};
    $("pos-exits").innerHTML = Object.keys(stops).length
      ? '<ul class="clean">'+Object.entries(stops).map(([s,v])=>`<li><b>${esc(s)}</b>
          <div class="mono">stop ${money(v.stop_loss)} · target ${money(v.take_profit)} · decision ${esc(v.decision_id||"—")}</div></li>`).join("")+"</ul>"
      : empty("No recorded exit levels.");
  }catch(e){ showErr(e.message); }
}

// -- Trades -----------------------------------------------------------------
async function refreshTrades(){
  try{
    const t = await api("/api/trades");
    const orders = t.orders || [];
    $("ord-note").textContent = `${t.submitted_orders} submitted · ${t.filled_orders} filled`;
    $("ord-body").innerHTML = orders.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Submitted</th><th>Symbol</th><th>Side</th><th class="num">Qty</th>
        <th class="num">Filled</th><th class="num">Remaining</th><th class="num">Avg price</th>
        <th>Status</th></tr></thead><tbody>${orders.map(o=>`<tr>
        <td class="mono">${esc(o.submitted_at||"—")}</td>
        <td><b>${esc(o.symbol)}</b></td><td>${esc(o.side)}</td>
        <td class="num">${num(o.quantity,4)}</td>
        <td class="num">${num(o.filled_quantity,4)}</td>
        <td class="num">${num(o.remaining_quantity,4)}</td>
        <td class="num">${money(o.average_fill_price)}</td>
        <td><span class="pill s-${esc(o.status||"")}">${esc(o.status||"—")}</span></td>
      </tr>`).join("")}</tbody></table></div>`
      : empty("No orders submitted by the worker.");

    const executions = t.executions || [];
    $("exec-note").textContent = (t.filled_executions ?? executions.length) + " fills";
    $("exec-body").innerHTML = executions.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Order time</th><th>Symbol</th><th>Side</th><th class="num">Filled qty</th>
        <th class="num">Average fill</th><th>Order status</th><th>Broker order</th></tr></thead><tbody>${executions.map(x=>`<tr>
        <td class="mono">${esc(x.order_timestamp||"—")}</td>
        <td><b>${esc(x.symbol)}</b></td><td>${esc(x.side)}</td>
        <td class="num">${num(x.filled_quantity,6)}</td>
        <td class="num">${money(x.average_fill_price)}</td>
        <td><span class="pill s-${esc(x.order_status||"")}">${esc(x.order_status||"—")}</span></td>
        <td class="mono">${esc(x.broker_order_id||x.client_order_id||"—")}</td>
      </tr>`).join("")}</tbody></table></div>`
      : empty("No filled executions yet. Submitted or accepted orders are not fills.");

    const trades = t.trades || [];
    $("trade-note").textContent = t.settled_trades + " settled";
    $("trade-body").innerHTML = trades.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Entry</th><th>Symbol</th><th>Side</th><th class="num">Qty</th>
        <th class="num">Entry px</th><th class="num">Exit px</th><th class="num">Fees</th>
        <th class="num">Realized</th><th>Exit reason</th></tr></thead><tbody>${trades.map(x=>`<tr>
        <td class="mono">${esc(x.entry_session||"—")}</td>
        <td><b>${esc(x.symbol)}</b></td><td>${esc(x.side||"—")}</td>
        <td class="num">${num(x.quantity,4)}</td>
        <td class="num">${money(x.entry_price)}</td>
        <td class="num">${money(x.exit_price)}</td>
        <td class="num">${money(x.fees)}</td>
        <td class="num ${cls(x.realized_pnl)}">${sgn(x.realized_pnl)}</td>
        <td>${esc(x.exit_reason||"—")}</td>
      </tr>`).join("")}</tbody></table></div>`
      : empty("No settled trades yet.");
  }catch(e){ showErr(e.message); }
}

// -- Journal ----------------------------------------------------------------
async function refreshJournal(){
  try{
    const j = await api("/api/journal?limit=100");
    const entries = j.entries || [];
    $("jlist").innerHTML = entries.length
      ? '<ul class="clean">'+entries.map(x=>`<li>
          <b>#${esc(x.seq)}</b> ${esc(x.kind)} <span class="muted">${esc(x.timestamp)}</span>
          <div class="mono">${esc(x.event_id)}</div></li>`).join("")+"</ul>"
      : empty("Journal empty.");
  }catch(e){ showErr(e.message); }
}

// -- Control ----------------------------------------------------------------
async function refreshControl(){
  try{
    const c = await api("/api/control/status");
    $("ctlstate").textContent = c.control_enabled
      ? `Control is enabled. Max ${c.max_sessions_per_call} sessions per call.`
      : "Control is disabled (read-only dashboard). Restart with --allow-control to enable.";
    $("runbtn").disabled = !c.control_enabled || c.run_in_progress;
    $("lastrun").textContent = c.last_run ? JSON.stringify(c.last_run, null, 2) : "—";
  }catch(e){ showErr(e.message); }
}
async function doRun(){
  const n = parseInt($("nsessions").value || "0", 10);
  $("runbtn").disabled = true;
  try{
    const r = await api("/api/run", {method:"POST",
      headers:{"Content-Type":"application/json", "X-Control-Confirm":"yes"},
      body: JSON.stringify({sessions:n})});
    $("lastrun").textContent = JSON.stringify(r, null, 2);
    await refresh();
  }catch(e){ showErr(e.message); }
  finally{ $("runbtn").disabled = false; }
}

// -- Crypto -----------------------------------------------------------------
function cxCandleCell(c){
  if(!c) return "N/A";
  return `${num(c.close,4)} <span class="muted">(${esc(c.close_time||"")})</span>`;
}
function renderCryptoMarket(m){
  const rows = (m && m.symbols) || [];
  $("cx-status").textContent = (m && m.status) ? m.status : "not_started";
  $("cx-detail").textContent = (m && m.detail)
    ? m.detail
    : (rows.length ? "Live Binance public market data." : "No crypto market data is configured.");
  if(!rows.length){
    $("cx-market").innerHTML = `<div class="card">${empty(
      m && m.status === "error" ? ("Crypto market data error: " + (m.detail||"unknown"))
                                 : "No crypto symbols are configured.")}</div>`;
    return;
  }
  $("cx-market").innerHTML = `<div class="tablewrap"><table>
    <thead><tr><th>Symbol</th><th class="num">Price</th><th class="num">24h %</th>
      <th class="num">24h Volume</th><th class="num">Bid</th><th class="num">Ask</th>
      <th class="num">Spread</th><th class="num">24h High</th><th class="num">24h Low</th>
      <th class="num">1m close</th><th class="num">5m close</th><th>Freshness</th></tr></thead>
    <tbody>${rows.map(r=>`<tr>
      <td><b>${esc(r.symbol)}</b><div class="mono">${esc(r.status)}</div></td>
      <td class="num">${r.price===null||r.price===undefined?"N/A":num(r.price,4)}</td>
      <td class="num ${cls(r.change_percent_24h)}">${r.change_percent_24h===null||r.change_percent_24h===undefined?"N/A":pctAlready(r.change_percent_24h)}</td>
      <td class="num">${r.volume_24h===null||r.volume_24h===undefined?"N/A":Number(r.volume_24h).toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td class="num">${r.bid===null||r.bid===undefined?"N/A":num(r.bid,4)}</td>
      <td class="num">${r.ask===null||r.ask===undefined?"N/A":num(r.ask,4)}</td>
      <td class="num">${r.spread===null||r.spread===undefined?"N/A":num(r.spread,6)}</td>
      <td class="num">${r.high_24h===null||r.high_24h===undefined?"N/A":num(r.high_24h,4)}</td>
      <td class="num">${r.low_24h===null||r.low_24h===undefined?"N/A":num(r.low_24h,4)}</td>
      <td class="num">${cxCandleCell((r.closed_candles||{})["1m"])}</td>
      <td class="num">${cxCandleCell((r.closed_candles||{})["5m"])}</td>
      <td>${r.fresh?`<span class="pill state-OK">live</span><div class="muted">${esc(rel(r.newest_event_at))}</div>`
                  :`<span class="pill state-ERROR">stale</span><div class="muted">${esc(r.detail||"")}</div>`}</td>
    </tr>`).join("")}</tbody></table></div>`;
}
function renderCryptoStrategy(s){
  const rows = (s && s.symbols) || [];
  $("cx-strategy-status").textContent = (s && s.status) ? s.status : "not_started";
  if(!rows.length){
    $("cx-strategy").innerHTML = `<div class="card">${empty(
      s && s.status === "error" ? ("TJR evaluation error: " + (s.detail||"unknown"))
                                 : "No TJR evaluation is available yet.")}</div>`;
    return;
  }
  $("cx-strategy").innerHTML = rows.map(r=>`<div class="card">
    <h2>${esc(r.symbol)} <span class="spacer"></span>
      <span class="pill ${r.actionable?"state-OK":"state-STOPPED"}">${esc(r.stage||"NO_SIGNAL")}</span>
      ${r.direction?`<span class="tag">${esc(r.direction)}</span>`:""}</h2>
    <div class="kv">
      <div class="k">Stage</div><div class="v">${esc(r.stage||"NO_SIGNAL")}</div>
      <div class="k">Reason</div><div class="v">${esc(r.reason_code||"—")}</div>
      <div class="k">Detail</div><div class="v">${esc(r.detail||"—")}</div>
      <div class="k">Entry</div><div class="v">${r.entry===null||r.entry===undefined?"N/A":num(r.entry,4)}</div>
      <div class="k">Stop loss</div><div class="v">${r.stop_loss===null||r.stop_loss===undefined?"N/A":num(r.stop_loss,4)}</div>
      <div class="k">Take profit</div><div class="v">${r.take_profit===null||r.take_profit===undefined?"N/A":num(r.take_profit,4)}</div>
      <div class="k">Risk / reward</div><div class="v">${r.risk_reward===null||r.risk_reward===undefined?"N/A":num(r.risk_reward,2)}</div>
      <div class="k">Decision time</div><div class="v">${esc(when(r.evaluated_at))}</div>
    </div>
  </div>`).join("");
}
function renderCryptoHealth(h){
  const sv = h.symbol_validation || null;
  const rows = {
    "Data provider": h.data_provider,
    "Data status": h.data_status,
    "Last successful update": h.data_last_refresh_at,
    "Data latency (s)": h.market_status,
    "Fresh symbols": (h.fresh_symbols===null||h.fresh_symbols===undefined)
      ? null : `${h.fresh_symbols} / ${h.total_symbols}`,
    "Worker status": h.worker_status,
    "Broker status": h.broker_status,
    "Decision source": h.decision_source_kind,
    "Market data seam": h.market_data_provider,
    "Market data status": h.market_data_status,
    "Last cycle": h.last_cycle_at,
    "Last order sync": h.last_order_sync_at,
    "Error count": h.error_count,
    "Last error": h.last_error,
    "Stream state": h.stream ? (h.stream.state || "unknown") : null,
    "Symbol validation": sv ? (sv.state || "unknown") : null,
    "Observed at": h.observed_at,
  };
  $("cx-health").innerHTML = kv(rows, (k,v)=>{
    if(v===null||v===undefined) return "N/A";
    if(k==="Last successful update"||k==="Last cycle"||k==="Last order sync"||k==="Observed at") return esc(when(v));
    return esc(v);
  });
  // The rejected pairs, each with the venue's own reason. A pair that was
  // dropped must be visible with its reason — never silently missing.
  const svBox = $("cx-symbols");
  if(!svBox) return;
  if(!sv){
    svBox.innerHTML = `<div class="muted">Symbol validation: not run yet.</div>`;
    return;
  }
  const rejected = Array.isArray(sv.rejected) ? sv.rejected : [];
  const accepted = Array.isArray(sv.accepted) ? sv.accepted : [];
  let html = `<div class="muted">${esc(sv.state||"unknown")} — ${esc(sv.detail||"")}</div>`;
  if(accepted.length){
    html += `<div class="mono" style="margin-top:6px">Accepted: ${esc(accepted.join(", "))}</div>`;
  }
  if(rejected.length){
    html += `<table class="tbl" style="margin-top:8px"><thead><tr><th>Symbol</th><th>Code</th><th>Reason</th></tr></thead><tbody>`;
    for(const r of rejected){
      html += `<tr><td><b>${esc(r.symbol)}</b></td><td class="mono">${esc(r.code||"")}</td><td>${esc(r.reason||"")}</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  svBox.innerHTML = html;
}
async function refreshCrypto(){
  try{
    const c = await api("/api/crypto");
    renderCryptoMarket(c.market);
    renderCryptoStrategy(c.strategy);
    renderCryptoHealth(c.health);
  }catch(e){ showErr(e.message); }
}

// -- nav / boot -------------------------------------------------------------
document.querySelectorAll("nav button").forEach(b=>{
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on", x===b));
    document.querySelectorAll("section").forEach(s=>s.classList.toggle("on", s.id===b.dataset.t));
    if(b.dataset.t==="crypto") refreshCrypto();
    if(b.dataset.t==="engine") refreshEngine();
    if(b.dataset.t==="decisions") refreshDecisions();
    if(b.dataset.t==="positions") refreshPositions();
    if(b.dataset.t==="trades") refreshTrades();
    if(b.dataset.t==="journal") refreshJournal();
    if(b.dataset.t==="control") refreshControl();
  };
});

async function refresh(){
  try{
    const s = await api("/api/summary");
    showErr("");
    $("auth").classList.add("hide");
    $("main").classList.remove("hide");
    $("nav").classList.remove("hide");
    renderOverview(s);
    await refreshEngine();
    await refreshOverviewAccount();
  }catch(e){
    if(/unauthorized/i.test(e.message)){
      $("auth").classList.remove("hide");
      $("main").classList.add("hide");
      $("nav").classList.add("hide");
    }
    showErr(e.message);
  }
}

async function refreshOverviewAccount(){
  try{
    const p = await api("/api/paper/status");
    const a = p.account || {};
    const eq = a.equity;
    const base = p.initial_equity;
    const daily = (eq!==null&&eq!==undefined&&base!==null&&base!==undefined) ? (eq-base) : null;
    const dailyPct = (daily!==null&&base) ? daily/base : null;
    $("ov-account").innerHTML =
      stat("Equity", money(eq), {hint:a.currency||""}) +
      stat("Cash", money(a.cash)) +
      stat("Buying power", money(a.buying_power)) +
      stat("Daily P&L", sgn(daily), {cls:cls(daily), hint:dailyPct===null?"no baseline":pct(dailyPct)+" today"}) +
      stat("Positions", esc(Object.keys(p.positions||{}).length)) +
      stat("Open orders", esc(Object.keys(p.orders||{}).length));
    $("ov-account-note").textContent = (a.status?("Account "+a.status+". "):"") +
      ((a.restrictions&&a.restrictions.length)?("Restrictions: "+a.restrictions.join(", ")+". "):"") +
      (base===null||base===undefined?"Daily P&L baseline not yet recorded (the worker has not read the account today).":
        ("Daily baseline "+money(base)+" set "+esc(p.day||"")+"."));
  }catch(e){ /* overview already surfaces the error */ }
}

function boot(){
  if(!TOKEN){ $("auth").classList.remove("hide"); return; }
  refresh();
  if(timer) clearInterval(timer);
  timer = setInterval(refresh, 10000);
  // The crypto market is live, so it polls faster than the 10-second summary
  // refresh. The poll is the same read-only endpoint; it places no order.
  if(cryptoTimer) clearInterval(cryptoTimer);
  cryptoTimer = setInterval(refreshCrypto, 3000);
}
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
