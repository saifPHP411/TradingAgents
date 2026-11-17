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
    DECISION_SOURCE_CHOICES,
    DECISION_SOURCE_HISTORY,
    DEFAULT_MAX_PRICE_DEVIATION,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_BYBIT_DEMO,
    EXECUTION_PROVIDER_CHOICES,
    MARKET_DATA_CHOICES,
    MAX_PRICE_DEVIATION_ENV,
    PROVIDER_BAR_OPEN,
    BrokerError,
    BrokerNotConfiguredError,
    BybitBrokerNotConfiguredError,
    BybitDemoBroker,
    banner_for_provider,
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
from .chart import DEFAULT_CHART_INTERVAL, DEFAULT_CHART_LIMIT, CryptoChartService
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

AUTO_START_PERPETUAL_STUDY_ENV = "TRADINGAGENTS_AUTO_START_PERPETUAL_STUDY"
PERPETUAL_ONLY_ENV = "TRADINGAGENTS_PERPETUAL_ONLY"
PERPETUAL_CHART_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
PERPETUAL_POSITION_FILE = "perpetual_open_position.json"


def _perpetual_only_enabled() -> bool:
    return os.environ.get(PERPETUAL_ONLY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

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


def _maybe_auto_start_perpetual_study(dashboard: Any, root: str | Path) -> None:
    """Optionally start the read-only Bybit study once on a dedicated research service."""
    enabled = os.environ.get(AUTO_START_PERPETUAL_STUDY_ENV, "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    report_path = Path(root) / "perpetual_ema_walkforward.json"
    if report_path.is_file():
        print("  perpetual study auto-start: skipped (saved report already exists)", flush=True)
        return
    status = dashboard.start_perpetual_study()
    print(
        f"  perpetual study auto-start: {status.get('status', 'unknown')}",
        flush=True,
    )


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
        chart_service: CryptoChartService | None = None,
        refresh_policy=None,
        refresh_fetcher=None,
        perpetual_client_factory=None,
        perpetual_only_mode: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.token = resolve_token(token)
        self.allow_control = allow_control
        self.perpetual_only_mode = _perpetual_only_enabled() if perpetual_only_mode is None else bool(perpetual_only_mode)
        self._perpetual_client_factory = perpetual_client_factory
        self.max_sessions_per_call = max_sessions_per_call
        self.stub = stub
        self._lock = threading.Lock()
        self._run_in_progress = False
        self.last_run: dict[str, Any] | None = None
        self.refresh_policy = refresh_policy
        self.refresh_fetcher = refresh_fetcher
        self.paper = PaperTradingController(
            self.root,
            enabled=allow_paper_trading and not self.perpetual_only_mode,
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
        # Historical candles load lazily and are cached; this view is read-only.
        self.chart_service = chart_service or CryptoChartService()
        # Independent read-only Bybit linear-perpetual research job. This never
        # starts a broker worker and never submits an exchange order.
        self._perpetual_study_lock = threading.Lock()
        self._perpetual_study_thread: threading.Thread | None = None
        self._perpetual_study_status: dict[str, Any] = {"status": "idle"}

    def perpetual_study_status(self) -> dict[str, Any]:
        with self._perpetual_study_lock:
            status = dict(self._perpetual_study_status)
        if status.get("status") != "idle":
            return status

        # A completed report lives on the Railway volume. Recover its status after
        # a container restart instead of showing "idle" beside a finished study.
        report_path = self.root / "perpetual_ema_walkforward.json"
        if not report_path.is_file():
            return status
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = {}
            for key, value in report["reports"].items():
                holdout = value["untouched_final_holdout"]
                summary[key] = {
                    "win_rate_pct": holdout["win_rate_pct"],
                    "return_pct": holdout["equal_weight_return_pct"],
                    "profit_factor": holdout["profit_factor"],
                    "trades": holdout["trades"],
                    "liquidations": holdout["liquidations"],
                    "passed": value["acceptance_gate"]["passed"],
                }
        except (OSError, ValueError, KeyError, TypeError):
            return status
        return {
            "status": "completed",
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(report_path.stat().st_mtime)
            ),
            "orders_submitted": False,
            "demo_ready": False,
            "report_file": report_path.name,
            "summary": summary,
        }

    def start_perpetual_study(self) -> dict[str, Any]:
        """Start a bounded public-data 1x/2x walk-forward study in the background."""
        with self._perpetual_study_lock:
            if self._perpetual_study_thread is not None and self._perpetual_study_thread.is_alive():
                return dict(self._perpetual_study_status)
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._perpetual_study_status = {
                "status": "running",
                "started_at": started_at,
                "orders_submitted": False,
                "demo_ready": False,
            }

            def run() -> None:
                try:
                    from types import SimpleNamespace

                    from tradingagents.crypto.backtest.perpetual_ema_walkforward import run_study

                    args = SimpleNamespace(
                        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                        data_source="bybit",
                        base_url="https://api.bybit.com",
                        days=730,
                        equity=1_000.0,
                        leverage=[1.0, 2.0],
                        warmup_bars=260,
                        test_bars=200,
                        holdout_fraction=0.20,
                    )
                    report = run_study(args)
                    report_path = self.root / "perpetual_ema_walkforward.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    summary = {}
                    for key, value in report["reports"].items():
                        holdout = value["untouched_final_holdout"]
                        summary[key] = {
                            "win_rate_pct": holdout["win_rate_pct"],
                            "return_pct": holdout["equal_weight_return_pct"],
                            "profit_factor": holdout["profit_factor"],
                            "trades": holdout["trades"],
                            "liquidations": holdout["liquidations"],
                            "passed": value["acceptance_gate"]["passed"],
                        }
                    with self._perpetual_study_lock:
                        self._perpetual_study_status = {
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "orders_submitted": False,
                            "demo_ready": False,
                            "report_file": report_path.name,
                            "summary": summary,
                        }
                except Exception as exc:  # noqa: BLE001 - report a failed research job, never crash the dashboard
                    with self._perpetual_study_lock:
                        self._perpetual_study_status = {
                            "status": "error",
                            "started_at": started_at,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "orders_submitted": False,
                            "demo_ready": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }

            self._perpetual_study_thread = threading.Thread(
                target=run, name="perpetual-walkforward-study", daemon=True
            )
            self._perpetual_study_thread.start()
            return dict(self._perpetual_study_status)

    def _perpetual_position_view(self, symbol: str) -> tuple[str, dict[str, Any] | None]:
        """Read a validated futures-position snapshot written by a perpetual worker.

        No broker is connected in the research deployment. Until a futures worker
        writes this file, the view is explicitly unavailable rather than flat or
        inferred from Spot positions.
        """
        path = self.root / PERPETUAL_POSITION_FILE
        if not path.is_file():
            return "unavailable", None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid", None
        if not isinstance(raw, dict):
            return "invalid", None
        state = str(raw.get("state", "")).strip().upper()
        raw_symbol = str(raw.get("symbol", "")).strip().upper()
        if state == "FLAT":
            return "flat", None
        if state != "OPEN" or raw_symbol != symbol:
            return "invalid", None
        side = str(raw.get("side", "")).strip().lower()
        if side not in {"long", "short"}:
            return "invalid", None

        def positive(name: str, *, optional: bool = False):
            value = raw.get(name)
            if optional and value is None:
                return None
            if isinstance(value, bool):
                raise ValueError(name)
            parsed = float(value)
            if not math.isfinite(parsed) or parsed <= 0:
                raise ValueError(name)
            return parsed

        try:
            position = {
                "symbol": symbol,
                "side": side,
                "quantity": positive("quantity"),
                "leverage": positive("leverage"),
                "entry_price": positive("entry_price"),
                "mark_price": positive("mark_price", optional=True),
                "stop_loss": positive("stop_loss", optional=True),
                "take_profit": positive("take_profit", optional=True),
                "liquidation_price": positive("liquidation_price", optional=True),
                "unrealized_pnl": raw.get("unrealized_pnl"),
                "updated_at": raw.get("updated_at"),
            }
            pnl = position["unrealized_pnl"]
            if pnl is not None:
                if isinstance(pnl, bool) or not math.isfinite(float(pnl)):
                    raise ValueError("unrealized_pnl")
                position["unrealized_pnl"] = float(pnl)
        except (TypeError, ValueError, OverflowError):
            return "invalid", None
        return "open", position

    def perpetual_chart(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Read completed Bybit linear-perpetual candles and a real worker position.

        Only the four supported USDT linear contracts are exposed. This endpoint is
        public market data only; it never authenticates or sends an exchange order.
        """
        market = str(symbol).strip().upper()
        if market not in PERPETUAL_CHART_SYMBOLS:
            raise ValueError("unsupported linear perpetual chart symbol")
        if self._perpetual_client_factory is None:
            from tradingagents.crypto.bybit_perpetual import BybitLinearPublicClient
            client = BybitLinearPublicClient()
        else:
            client = self._perpetual_client_factory()
        instrument = client.instrument_info(market)
        if (
            instrument.status != "Trading"
            or instrument.contract_type != "LinearPerpetual"
            or instrument.quote_coin != "USDT"
            or instrument.settle_coin != "USDT"
        ):
            raise ValueError("symbol is not an active USDT linear perpetual")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - 160 * 4 * 60 * 60 * 1000
        frame = client.klines(market, start_ms=start_ms, end_ms=end_ms)
        if frame is None or frame.empty:
            raise RuntimeError("Bybit returned no completed 4-hour candles")
        candles = []
        for stamp, row in frame.tail(160).iterrows():
            values = [float(row[name]) for name in ("open", "high", "low", "close")]
            if not all(math.isfinite(value) and value > 0 for value in values):
                continue
            candles.append({
                "time": stamp.isoformat(),
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": max(0.0, float(row.get("volume", 0.0))),
            })
        if not candles:
            raise RuntimeError("Bybit returned no valid completed candles")
        position_status, position = self._perpetual_position_view(market)
        return {
            "status": "ok",
            "venue": "Bybit V5 public linear perpetual market data",
            "category": "linear",
            "symbol": market,
            "timeframe": "4h",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candles": candles,
            "position_status": position_status,
            "position": position,
            "position_source": (
                PERPETUAL_POSITION_FILE if position_status in {"open", "flat"} else None
            ),
            "orders_submitted": False,
        }

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
        if self.perpetual_only_mode:
            return {"positions": [], "count": 0, "positions_checked_at": None, "positions_at": None,
                    "quote_checked_at": None, "quote_refresh_status": "disabled",
                    "banner": "Perpetual research only — Spot execution is disabled", "live": False}
        from .engine_view import positions_view

        status = self.paper.status()

        # A persisted worker snapshot can outlive the worker process. Refresh
        # public Bybit Spot quotes for display without starting the worker,
        # touching private account endpoints, or placing orders.
        positions = status.get("positions") or {}
        quote_checked_at = None
        quote_refresh_status = "not_applicable"
        if self.paper.execution_provider == "bybit_demo" and positions:
            quote_refresh_status = "error"
            try:
                from tradingagents.crypto.bybit import BybitPublicClient

                symbols = [
                    str(row.get("venue_symbol") or symbol).replace("/", "")
                    for symbol, row in positions.items()
                    if isinstance(row, dict)
                ]
                tickers = BybitPublicClient().tickers(symbols)
                quote_checked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                updated_quotes = 0
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
                            updated_quotes += 1
                quote_refresh_status = "ok" if updated_quotes else "no_quote"
            except Exception:
                # Keep the last known broker snapshot; never label its timestamp
                # as a fresh broker sync merely because a public quote was fetched.
                quote_refresh_status = "error"

        rows = positions_view(status, self._internal_positions())
        return {
            "positions": rows,
            "count": len(rows),
            "positions_checked_at": status.get("positions_checked_at"),
            "positions_at": status.get("positions_at"),
            "quote_checked_at": quote_checked_at,
            "quote_refresh_status": quote_refresh_status,
            "banner": banner_for_provider(self.paper.execution_provider),
            "live": False,
        }

    def chart(self, symbol: str, interval: str = DEFAULT_CHART_INTERVAL) -> dict[str, Any]:
        """A bounded public Spot chart for a symbol configured on this dashboard."""
        from tradingagents.crypto.symbols import parse_pair

        canonical = parse_pair(symbol).canonical
        if canonical not in set(self.crypto.symbols):
            raise ValueError("symbol is not configured for this dashboard")
        return self.chart_service.get(canonical, interval, DEFAULT_CHART_LIMIT)

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
        if self.perpetual_only_mode:
            raise PermissionError("Spot/equity simulation controls are disabled in perpetual-only mode")
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
                elif path == "/api/crypto/chart":
                    query = parse_qs(urlsplit(self.path).query)
                    default_symbol = dashboard.crypto.symbols[0] if dashboard.crypto.symbols else ""
                    symbol = (query.get("symbol") or [default_symbol])[0]
                    interval = (query.get("interval") or [DEFAULT_CHART_INTERVAL])[0]
                    self._send_json(200, dashboard.chart(symbol, interval))
                elif path == "/api/perpetual/study/status":
                    self._send_json(200, dashboard.perpetual_study_status())
                elif path == "/api/perpetual/chart":
                    if not self._authed():
                        return
                    symbol = parse_qs(urlsplit(self.path).query).get("symbol", ["BTCUSDT"])[0]
                    try:
                        self._send_json(200, dashboard.perpetual_chart(symbol))
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                    except Exception as exc:  # noqa: BLE001 - public data errors are surfaced without credentials
                        self._send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
                else:
                    self._send_json(404, {"error": "not found"})
            except DashboardStateError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surface, never crash the server
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            path = urlsplit(self.path).path
            if path == "/api/perpetual/study/start":
                if not self._authed():
                    return
                if (self.headers.get("X-Perpetual-Study-Confirm") or "").lower() != "yes":
                    self._send_json(403, {"error": "explicit research confirmation required"})
                    return
                self._send_json(202, dashboard.start_perpetual_study())
                return
            if path in ("/api/paper/start", "/api/paper/stop"):
                self._handle_paper_control(path)
                return
            if path != "/api/run":
                self._send_json(404, {"error": "not found"})
                return
            if not self._authed():
                return
            if dashboard.perpetual_only_mode:
                self._send_json(410, {"error": "Spot/equity simulation controls are disabled in perpetual-only mode"})
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
            if dashboard.perpetual_only_mode:
                self._send_json(410, {"error": "Spot paper execution is disabled in perpetual-only mode"})
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
    perpetual_only_mode = _perpetual_only_enabled()
    # The perpetual-only deployment must not create or refresh the legacy Spot/equity experiment.
    if not perpetual_only_mode:
        _ensure_experiment(root, crypto_symbols=crypto_symbols)

    resolved = resolve_token(token)
    require_bindable(host, resolved)
    crypto_engine = None
    if crypto_live and crypto_symbols and not perpetual_only_mode:
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
        perpetual_only_mode=perpetual_only_mode,
    )
    _maybe_auto_start_perpetual_study(dashboard, root)
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
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark"><title>TradingAgents · Perpetual Lab</title>
<style>
:root{color-scheme:dark;--bg:#070b16;--panel:#111a2b;--line:#24334c;--muted:#91a2bd;--text:#edf3ff;--cyan:#61d9e8;--green:#3fd6a2;--red:#ff718f;--amber:#ffc46a;--blue:#8ea5ff}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:radial-gradient(ellipse at 12% 0%,rgba(84,105,255,.16),transparent 38%),#070b16;color:var(--text);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
body{padding:env(safe-area-inset-top) 0 calc(env(safe-area-inset-bottom) + 20px)}button,input,select{font:inherit}button{cursor:pointer}
.shell{width:min(1440px,100%);margin:auto;padding:0 22px}.top{position:sticky;top:0;z-index:5;background:rgba(7,11,22,.9);backdrop-filter:blur(16px);border-bottom:1px solid rgba(142,165,255,.17)}
.toprow{display:flex;align-items:center;gap:12px;min-height:66px}.brand{font-weight:850;letter-spacing:.1em;text-transform:uppercase}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:.15em}.push{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:#101b2d;color:var(--muted);border-radius:999px;padding:5px 9px;font-size:10px;font-weight:800;letter-spacing:.05em;white-space:nowrap}.safe{color:var(--green);border-color:rgba(63,214,162,.3)}.warn{color:var(--amber);border-color:rgba(255,196,106,.3)}
.hero{padding:24px 0 16px;display:flex;justify-content:space-between;align-items:end;gap:16px}.kicker{font-size:10px;color:var(--cyan);font-weight:800;letter-spacing:.2em}h1{font-size:clamp(26px,4vw,40px);line-height:1.08;margin:8px 0;letter-spacing:-.04em}.hero p{color:var(--muted);max-width:720px;margin:0;font-size:13px}
.actions{display:flex;gap:8px;flex-wrap:wrap}.control,button.action{border:1px solid var(--line);background:#121d31;color:var(--text);border-radius:11px;padding:10px 12px;min-height:41px}button.action.primary{background:linear-gradient(110deg,#76a3ff,#4ad5df);border:0;color:#071324;font-weight:850}button:disabled{opacity:.48;cursor:wait}select.control{min-width:135px}
.notice{padding:11px 13px;border:1px solid rgba(255,196,106,.24);background:rgba(255,196,106,.07);border-radius:13px;color:#e8d4ad;font-size:12px;margin-bottom:14px}.notice strong{color:#ffe2ad}
.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(280px,.8fr);gap:14px;align-items:start}.panel{background:linear-gradient(145deg,rgba(18,28,48,.97),rgba(10,17,31,.98));border:1px solid var(--line);border-radius:17px;overflow:hidden;box-shadow:0 20px 50px rgba(0,0,0,.2)}
.panelhead{display:flex;align-items:center;gap:12px;padding:14px 16px;border-bottom:1px solid rgba(142,165,255,.12)}.panelhead h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.12em}.sub{font-size:11px;color:var(--muted)}
.canvaswrap{position:relative;padding:8px 9px 5px}#chart{display:block;width:100%;height:440px;touch-action:pan-y}.chart-empty{position:absolute;inset:8px 9px 5px;display:none;align-items:center;justify-content:center;text-align:center;padding:26px;color:var(--muted);background:rgba(7,12,23,.6);border-radius:12px;pointer-events:none}
.legend{display:flex;gap:12px;flex-wrap:wrap;padding:0 16px 13px;color:var(--muted);font-size:10px}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:17px;height:2px;background:var(--cyan)}.entry{background:var(--blue)}.stop{background:var(--red)}.target{background:var(--green)}.liq{background:var(--amber)}
.side{display:grid;gap:13px}.pad{padding:15px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric{padding:10px;border:1px solid rgba(142,165,255,.12);border-radius:12px;background:rgba(8,15,29,.55);min-width:0}.metric label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;font-size:18px;margin-top:3px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.metric small{font-size:10px;color:var(--muted)}
.position{display:grid;gap:7px}.position-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.position-head b{font-size:14px}.level{display:grid;grid-template-columns:1fr auto;gap:12px;padding:7px 0;border-bottom:1px solid rgba(142,165,255,.1);font-variant-numeric:tabular-nums}.level span:first-child{color:var(--muted)}.empty-state{padding:13px;border:1px dashed rgba(142,165,255,.24);border-radius:12px;color:var(--muted);font-size:12px}
.study-status{color:var(--muted);font-size:12px;margin:10px 0}.study-grid{display:grid;gap:8px}.study-row{padding:10px;border-radius:11px;background:rgba(8,15,29,.62);border:1px solid rgba(142,165,255,.12)}.study-row strong{display:block;font-size:12px;margin-bottom:4px}.study-row p{margin:0;color:var(--muted);font-size:11px;line-height:1.6}.gate-pass{color:var(--green)}.gate-fail{color:var(--red)}
#error{display:none;margin:12px 0;padding:10px 13px;border:1px solid rgba(255,113,143,.34);background:rgba(255,113,143,.08);border-radius:12px;color:#ffafc0;font-size:12px;white-space:pre-wrap}.tokenbox{display:flex;gap:8px;max-width:430px}.tokenbox input{min-width:0;flex:1;background:#081120;border:1px solid var(--line);border-radius:10px;padding:10px 12px;color:var(--text)}.muted{color:var(--muted)}.small{font-size:11px}.footer{margin-top:14px;color:#7586a2;font-size:10px;text-align:center}
@media(max-width:900px){.layout{grid-template-columns:minmax(0,1fr)}.side{grid-template-columns:repeat(2,minmax(0,1fr))}.side .wide{grid-column:1/-1}}
@media(max-width:620px){.shell{padding:0 12px}.toprow{min-height:59px}.brand{font-size:13px}.hero{display:block;padding:18px 0 13px}.hero .actions{margin-top:13px}.actions{width:100%}.actions>*{flex:1}.layout{gap:10px}.side{grid-template-columns:1fr}.side .wide{grid-column:auto}.panelhead,.pad{padding:12px}#chart{height:335px}.metric b{font-size:16px}}
@media(max-width:360px){.shell{padding:0 9px}.pill{font-size:9px;padding:5px 6px}#chart{height:295px}.tokenbox{flex-direction:column}.tokenbox button{width:100%}}
</style></head>
<body>
<header class="top"><div class="shell toprow"><div class="brand">TradingAgents<small>Linear perpetual research lab</small></div><div class="push"></div><span class="pill safe">● RESEARCH ONLY</span><span class="pill warn">ORDERS DISABLED</span></div></header>
<main class="shell">
<section class="hero"><div><div class="kicker">BYBIT · LINEAR USDT PERPETUALS · ISOLATED RESEARCH</div><h1>Perpetuals, with evidence.</h1><p>One focused workspace for leveraged strategy research, honest after-cost metrics and a clean chart. No Spot engine, no fabricated positions and no exchange orders.</p></div><div class="actions"><select id="symbol" class="control" aria-label="Perpetual symbol"><option value="BTCUSDT">BTC / USDT</option><option value="ETHUSDT">ETH / USDT</option><option value="SOLUSDT">SOL / USDT</option><option value="BNBUSDT">BNB / USDT</option></select><button id="refresh" class="action" type="button">Refresh chart</button></div></section>
<div class="notice"><strong>Leverage is not a profitability signal.</strong> Research leverage is capped at 2x and defaults to 1x. Simulated returns are estimates, not proof of future results or Bybit execution readiness.</div>
<div id="error"></div>
<section id="auth" class="panel pad" style="display:none;margin-bottom:14px"><div class="panelhead" style="padding:0 0 12px;border:0"><h2>Dashboard access</h2></div><div class="tokenbox"><input id="token" type="password" autocomplete="off" placeholder="Dashboard access token"><button id="connect" class="action primary" type="button">Connect</button></div><p class="muted small">Token is stored in this browser only.</p></section>
<section class="layout" id="main" style="display:none">
<div class="panel"><div class="panelhead"><div><h2 id="chart-title">BTCUSDT · 4H</h2><div class="sub" id="chart-sub">Bybit public linear perpetual data</div></div><div class="push"></div><span class="pill" id="market-state">CONNECTING</span></div><div class="canvaswrap"><canvas id="chart" aria-label="Bybit 4-hour candlestick chart"></canvas><div id="chart-empty" class="chart-empty"></div></div><div class="legend"><span><i class="swatch"></i>Candles</span><span><i class="swatch entry"></i>Entry</span><span><i class="swatch stop"></i>Stop loss</span><span><i class="swatch target"></i>Take profit</span><span><i class="swatch liq"></i>Est. liquidation</span></div></div>
<aside class="side">
<section class="panel pad"><div class="panelhead" style="padding:0 0 12px;border:0"><h2>Market snapshot</h2></div><div class="metrics"><div class="metric"><label>Last close</label><b id="last-price">—</b><small id="last-time">—</small></div><div class="metric"><label>4H candles</label><b id="bars">—</b><small>Completed only</small></div><div class="metric"><label>Instrument</label><b id="instrument">—</b><small>USDT linear</small></div><div class="metric"><label>Data updated</label><b id="updated">—</b><small>Bybit public endpoint</small></div></div></section>
<section class="panel pad"><div class="position-head"><h2 style="font-size:12px;letter-spacing:.12em;margin:0">Open futures position</h2><span id="position-badge" class="pill">UNKNOWN</span></div><div id="position-body" class="empty-state" style="margin-top:11px">Checking position source…</div></section>
<section class="panel pad wide"><div class="position-head"><h2 style="font-size:12px;letter-spacing:.12em;margin:0">Walk-forward study</h2><span id="study-badge" class="pill">IDLE</span></div><p id="study-status" class="study-status">A study compares 1x and 2x across realistic, conservative and stress cost scenarios.</p><button id="run-study" class="action primary" type="button" style="width:100%">Run perpetual research</button><div id="study-results" class="study-grid" style="margin-top:12px"></div></section>
</aside></section>
<div class="footer">Bybit public market data · completed 4-hour candles · no authenticated account · no orders submitted</div>
</main>
<script>
(function(){
"use strict";
var $=function(id){return document.getElementById(id);},TOKEN=localStorage.getItem("ta_token")||"",DATA=null,STUDY_TIMER=null,CHART_TIMER=null;
function fmt(v,d){if(v===null||v===undefined||!Number.isFinite(Number(v)))return "—";return Number(v).toLocaleString(undefined,{maximumFractionDigits:d===undefined?2:d});}
function esc(v){return String(v===null||v===undefined?"":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
function showError(s){var el=$("error");el.style.display=s?"block":"none";el.textContent=s||"";}
async function api(path,opts){opts=opts||{};opts.headers=Object.assign({},opts.headers||{},{Authorization:"Bearer "+TOKEN});var r=await fetch(path,opts),j={};try{j=await r.json();}catch(e){}if(r.status===401)throw new Error("Access token is invalid or missing.");if(!r.ok)throw new Error(j.error||("HTTP "+r.status));return j;}
function connect(){TOKEN=$("token").value.trim();if(!TOKEN){showError("Enter the dashboard token.");return;}localStorage.setItem("ta_token",TOKEN);boot();}
function boot(){ $("auth").style.display="none";$("main").style.display="grid";showError("");refreshChart();refreshStudy();if(CHART_TIMER)clearInterval(CHART_TIMER);if(STUDY_TIMER)clearInterval(STUDY_TIMER);CHART_TIMER=setInterval(refreshChart,60000);STUDY_TIMER=setInterval(refreshStudy,10000);}
function drawChart(payload){
 var canvas=$("chart"),r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(250,r.width),h=Math.max(260,r.height);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);var ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 var candles=payload.candles||[];if(!candles.length){$("chart-empty").style.display="flex";$("chart-empty").textContent="No completed Bybit candles available.";return;}$("chart-empty").style.display="none";
 var p=payload.position,levels=[];if(p){if(p.entry_price)levels.push({name:"ENTRY",v:Number(p.entry_price),c:"#8ea5ff"});if(p.stop_loss)levels.push({name:"STOP",v:Number(p.stop_loss),c:"#ff718f"});if(p.take_profit)levels.push({name:"TARGET",v:Number(p.take_profit),c:"#3fd6a2"});if(p.liquidation_price)levels.push({name:"EST.LIQ",v:Number(p.liquidation_price),c:"#ffc46a"});}
 var vals=[];candles.forEach(function(c){vals.push(Number(c.high),Number(c.low));});levels.forEach(function(x){vals.push(x.v);});vals=vals.filter(Number.isFinite);var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals),span=hi-lo;if(!Number.isFinite(span)||span<=0)span=Math.max(1,Math.abs(hi)*.01);lo-=span*.08;hi+=span*.08;
 var left=10,right=91,top=18,bottom=35,pb=h-52,ph=pb-top,pw=w-left-right,step=pw/candles.length;function y(v){return top+(hi-v)/(hi-lo)*ph;}
 ctx.font="10px ui-monospace,monospace";ctx.strokeStyle="rgba(145,162,189,.16)";ctx.fillStyle="#91a2bd";ctx.textAlign="left";
 for(var i=0;i<=5;i++){var yy=top+ph*i/5;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.fillText(fmt(hi-(hi-lo)*i/5,2),w-right+7,yy+3);}
 var maxVol=Math.max.apply(null,candles.map(function(c){return Number(c.volume)||0;}))||1;
 candles.forEach(function(c,i){var x=left+step*(i+.5),o=Number(c.open),cl=Number(c.close),hh=Number(c.high),ll=Number(c.low);if(![o,cl,hh,ll].every(Number.isFinite))return;var up=cl>=o,col=up?"#3fd6a2":"#ff718f",bw=Math.max(2,Math.min(10,step*.62)),yo=y(o),yc=y(cl);ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(hh));ctx.lineTo(x,y(ll));ctx.stroke();ctx.fillRect(x-bw/2,Math.min(yo,yc),bw,Math.max(1,Math.abs(yc-yo)));var vol=Math.max(0,Number(c.volume)||0);ctx.globalAlpha=.28;ctx.fillRect(x-bw/2,pb+6,bw,22*vol/maxVol);ctx.globalAlpha=1;});
 ctx.fillStyle="#91a2bd";ctx.textAlign="center";[0,Math.floor(candles.length/2),candles.length-1].forEach(function(i){var t=new Date(candles[i].time);ctx.fillText(isNaN(t.getTime())?"":t.toLocaleDateString(undefined,{month:"short",day:"numeric"}),left+step*(i+.5),h-10);});
 var lastY=-999;levels.sort(function(a,b){return b.v-a.v;}).forEach(function(l){var yy=y(l.v);if(yy<top||yy>pb)return;ctx.save();ctx.strokeStyle=l.c;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.restore();var ly=Math.max(top+9,Math.min(pb-9,yy));if(ly-lastY<19)ly=Math.min(pb-9,lastY+19);lastY=ly;ctx.fillStyle="rgba(7,13,25,.96)";ctx.fillRect(w-right+3,ly-9,right-5,18);ctx.fillStyle=l.c;ctx.textAlign="left";ctx.textBaseline="middle";ctx.font="9px ui-monospace,monospace";ctx.fillText(l.name+" "+fmt(l.v,2),w-right+7,ly);});
 if(p&&p.mark_price){ctx.strokeStyle="#61d9e8";ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(left,y(Number(p.mark_price)));ctx.lineTo(w-right,y(Number(p.mark_price)));ctx.stroke();ctx.setLineDash([]);}
}
function renderPosition(d){var s=d.position_status||"unavailable",p=d.position,b=$("position-badge"),el=$("position-body");b.textContent=s.toUpperCase();b.className="pill "+(s==="open"?"safe":s==="flat"?"":"warn");
 if(s==="unavailable"){el.className="empty-state";el.textContent="No authenticated Bybit Futures position feed is connected. The dashboard will not infer a position from Spot data or invent entry levels. Market chart only.";return;}
 if(s==="invalid"){el.className="empty-state";el.textContent="Position snapshot failed validation and is hidden.";return;}
 if(s==="flat"||!p){el.className="empty-state";el.textContent="The connected futures source reports no open position.";return;}
 el.className="position";el.innerHTML='<div class="position-head"><b>'+esc(p.symbol)+' · '+esc(p.side.toUpperCase())+'</b><span class="pill safe">'+fmt(p.leverage,1)+'x</span></div>'+
 [['Quantity',fmt(p.quantity,6)],['Entry',fmt(p.entry_price,4)],['Mark',fmt(p.mark_price,4)],['Unrealized P&L',fmt(p.unrealized_pnl,3)],['Stop loss',fmt(p.stop_loss,4)],['Take profit',fmt(p.take_profit,4)],['Estimated liquidation',fmt(p.liquidation_price,4)]].map(function(x){return '<div class="level"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>';}).join("")+
 '<div class="muted small">Snapshot: '+esc(p.updated_at||"unknown")+'</div>';
}
async function refreshChart(){var symbol=$("symbol").value;$("chart-title").textContent=symbol+" · 4H";$("market-state").textContent="LOADING";try{var d=await api("/api/perpetual/chart?symbol="+encodeURIComponent(symbol));DATA=d;var bars=d.candles||[],last=bars[bars.length-1]||{};$("market-state").textContent="LIVE DATA";$("market-state").className="pill safe";$("last-price").textContent=fmt(last.close,4);$("last-time").textContent=last.time?new Date(last.time).toLocaleString():"—";$("bars").textContent=String(bars.length);$("instrument").textContent=d.category==="linear"?"USDT LINEAR":"—";$("updated").textContent=d.updated_at?new Date(d.updated_at).toLocaleTimeString():"—";$("chart-sub").textContent="Bybit public · completed candles only";renderPosition(d);drawChart(d);showError("");}catch(e){$("market-state").textContent="UNAVAILABLE";$("market-state").className="pill warn";$("chart-empty").style.display="flex";$("chart-empty").textContent=e.message;$("position-badge").textContent="UNKNOWN";$("position-body").className="empty-state";$("position-body").textContent="Position status unavailable.";showError(e.message);}}
function renderStudy(s){var b=$("study-badge"),st=$("study-status"),box=$("study-results"),btn=$("run-study");b.textContent=String(s.status||"idle").toUpperCase();b.className="pill "+(s.status==="completed"?"safe":s.status==="error"?"warn":"");btn.disabled=s.status==="running";btn.textContent=s.status==="running"?"RESEARCH RUNNING…":"Run perpetual research";
 if(s.status==="running"){st.textContent="Fetching public Bybit history and calculating 1x/2x results. No exchange orders are submitted.";return;}
 if(s.status==="error"){st.textContent="Research failed: "+(s.error||s.error_type||"unknown error");box.innerHTML="";return;}
 if(s.status!=="completed"){st.textContent="No completed study report loaded. Start a research run to calculate trade statistics.";box.innerHTML="";return;}
 st.textContent="Completed study · orders submitted: no · Demo approval: no. Holdout statistics are historical research results, not a guarantee.";
 box.innerHTML=Object.entries(s.summary||{}).map(function(x){var m=x[1];return '<div class="study-row"><strong>'+esc(x[0])+' <span class="'+(m.passed?"gate-pass":"gate-fail")+'">'+(m.passed?"GATE PASSED":"GATE NOT PASSED")+'</span></strong><p>Win rate: '+(m.win_rate_pct===null?"—":fmt(m.win_rate_pct,1)+"%")+' · Return: '+(m.return_pct===null?"—":fmt(m.return_pct,2)+"%")+' · PF: '+esc(m.profit_factor)+' · Trades: '+esc(m.trades)+' · Liquidations: '+esc(m.liquidations)+'</p></div>';}).join("");
}
async function refreshStudy(){try{renderStudy(await api("/api/perpetual/study/status"));}catch(e){$("study-status").textContent="Study status unavailable: "+e.message;}}
async function runStudy(){var b=$("run-study");b.disabled=true;try{renderStudy(await api("/api/perpetual/study/start",{method:"POST",headers:{"X-Perpetual-Study-Confirm":"yes"}}));setTimeout(refreshStudy,1200);}catch(e){showError(e.message);}finally{setTimeout(refreshStudy,1500);}}
$("connect").addEventListener("click",connect);$("token").addEventListener("keydown",function(e){if(e.key==="Enter")connect();});$("symbol").addEventListener("change",refreshChart);$("refresh").addEventListener("click",refreshChart);$("run-study").addEventListener("click",runStudy);window.addEventListener("resize",function(){if(DATA)drawChart(DATA);});if(TOKEN)boot();else $("auth").style.display="block";
})();
</script>
</body></html>
"""


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
