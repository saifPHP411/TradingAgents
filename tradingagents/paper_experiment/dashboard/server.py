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
AUTO_START_PERPETUAL_LAB_ENV = "TRADINGAGENTS_AUTO_START_PERPETUAL_LAB"
AUTO_START_PERPETUAL_SHADOW_ENV = "TRADINGAGENTS_AUTO_START_PERPETUAL_SHADOW"
PERPETUAL_LAB_REPORT_FILE = "perpetual_strategy_lab.json"
PERPETUAL_SHADOW_POLL_SECONDS = 900
PERPETUAL_ONLY_ENV = "TRADINGAGENTS_PERPETUAL_ONLY"
PERPETUAL_CHART_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
PERPETUAL_POSITION_FILE = "perpetual_open_position.json"
PERPETUAL_ACCOUNT_BASELINE_ENV = "TRADINGAGENTS_INITIAL_CAPITAL_USDT"
PERPETUAL_ACCOUNT_BASELINE_FILE = "perpetual_account_baseline.json"
PERPETUAL_ACCOUNT_HISTORY_FILE = "perpetual_account_history.json"


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
    """Keep the legacy fixed-EMA runner disabled after its holdout has been inspected."""
    enabled = os.environ.get(AUTO_START_PERPETUAL_STUDY_ENV, "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        print(
            "  legacy perpetual holdout auto-start: disabled; use the development-only strategy lab",
            flush=True,
        )


def _maybe_auto_start_perpetual_lab(dashboard: Any, root: str | Path) -> None:
    """Optionally start the development-only candidate tournament once."""
    enabled = os.environ.get(AUTO_START_PERPETUAL_LAB_ENV, "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    report_path = Path(root) / PERPETUAL_LAB_REPORT_FILE
    if report_path.is_file():
        print("  perpetual strategy lab auto-start: skipped (saved report already exists)", flush=True)
        return
    status = dashboard.start_perpetual_lab()
    print(
        f"  perpetual strategy lab auto-start: {status.get('status', 'unknown')}",
        flush=True,
    )


def _maybe_auto_start_perpetual_shadow(dashboard: Any, root: str | Path) -> None:
    """Start the persistent read-only forward evaluator when explicitly enabled."""
    enabled = os.environ.get(AUTO_START_PERPETUAL_SHADOW_ENV, "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    status = dashboard.start_perpetual_shadow()
    print(
        f"  perpetual shadow forward auto-start: {status.get('status', 'unknown')}",
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
        perpetual_account_client_factory=None,
        perpetual_only_mode: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.token = resolve_token(token)
        self.allow_control = allow_control
        self.perpetual_only_mode = _perpetual_only_enabled() if perpetual_only_mode is None else bool(perpetual_only_mode)
        self._perpetual_client_factory = perpetual_client_factory
        self._perpetual_account_client_factory = perpetual_account_client_factory
        self.max_sessions_per_call = max_sessions_per_call
        self.stub = stub
        self._lock = threading.Lock()
        self._run_in_progress = False
        self.last_run: dict[str, Any] | None = None
        self.refresh_policy = refresh_policy
        self.refresh_fetcher = refresh_fetcher
        if self.perpetual_only_mode:
            # Do not instantiate the legacy Spot/equity or paper-trading stack at
            # all in this deployment. Perpetual routes own their own public
            # market-data and optional read-only Demo position clients.
            self.paper = None
            self.crypto = None
            self.chart_service = None
        else:
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
            self.crypto = CryptoViewService(
                engine=crypto_engine,
                symbols=tuple(crypto_symbols),
                worker_status=lambda: self.paper.status(),
                strategy_evaluations=lambda: self.paper.strategy_evaluations(),
            )
            self.chart_service = chart_service or CryptoChartService()
        # Independent read-only Bybit linear-perpetual research job. This never
        # starts a broker worker and never submits an exchange order.
        self._perpetual_study_lock = threading.Lock()
        self._perpetual_study_thread: threading.Thread | None = None
        self._perpetual_study_status: dict[str, Any] = {"status": "idle"}
        self._perpetual_lab_lock = threading.Lock()
        self._perpetual_lab_thread: threading.Thread | None = None
        self._perpetual_lab_status: dict[str, Any] = {"status": "idle"}
        self._perpetual_shadow_lock = threading.Lock()
        self._perpetual_shadow_thread: threading.Thread | None = None
        self._perpetual_shadow_stop = threading.Event()
        self._perpetual_shadow_status: dict[str, Any] = {"status": "idle"}
        self._perpetual_account_lock = threading.Lock()

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
        """Start the one-time historical holdout study; never overwrite a consumed holdout."""
        report_path = self.root / "perpetual_ema_walkforward.json"
        if report_path.is_file():
            return {
                **self.perpetual_study_status(),
                "rerun_blocked": True,
                "reason": (
                    "The historical holdout has already been evaluated. Re-running "
                    "the same rolling window would not create independent evidence."
                ),
            }
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

    def perpetual_lab_status(self) -> dict[str, Any]:
        """Recover the saved development report using the current acceptance rules."""
        with self._perpetual_lab_lock:
            status = dict(self._perpetual_lab_status)
        if status.get("status") != "idle":
            return status
        report_path = self.root / PERPETUAL_LAB_REPORT_FILE
        if not report_path.is_file():
            return status
        try:
            from tradingagents.crypto.backtest.perpetual_strategy_lab import (
                PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES,
                TARGET_WIN_RATE_PCT,
                acceptance_gate,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            grouped = []
            exclusions = PREVIOUS_HOLDOUT_DISQUALIFIED_CANDIDATES
            for item in report.get("rankings", []):
                candidate = item["candidate"]
                leverage = float(item["leverage"])
                scenario_rows = {}
                current_scenario_passes = []
                for scenario in ("realistic_estimate", "conservative", "stress"):
                    key = f"{candidate}/{leverage:g}x/{scenario}"
                    row = report["reports"][key]
                    metrics = row["development_only"]
                    # Saved reports may have been generated under an older gate.
                    # Recalculate from the immutable metrics, never trust stale
                    # pass/fail flags when displaying current eligibility.
                    current_gate = acceptance_gate(metrics, scenario)
                    current_scenario_passes.append(current_gate["passed"])
                    scenario_rows[scenario] = {
                        "win_rate_pct": metrics.get("win_rate_pct"),
                        "return_pct": metrics.get("equal_weight_compounded_return_pct"),
                        "expectancy_per_trade": metrics.get("expectancy_per_trade"),
                        "profit_factor": metrics.get("profit_factor"),
                        "trades": metrics.get("trades"),
                        "positive_folds": metrics.get("positive_folds"),
                        "fold_count": metrics.get("fold_count"),
                        "positive_symbols": metrics.get("positive_symbols"),
                        "symbol_count": metrics.get("symbol_count"),
                        "liquidations": metrics.get("liquidations"),
                        "passed": current_gate["passed"],
                        "acceptance_gate": current_gate,
                    }
                rejection = exclusions.get(candidate)
                all_current_gates_pass = all(current_scenario_passes)
                grouped.append({
                    **item,
                    "development_gate_passed_all_costs": all_current_gates_pass,
                    "previous_holdout_rejected": rejection is not None,
                    "previous_holdout_rejection_reason": rejection,
                    "selection_eligible": all_current_gates_pass and rejection is None,
                    "scenarios": scenario_rows,
                })
            eligible = [row for row in grouped if row["selection_eligible"]]
            eligible.sort(
                key=lambda row: (
                    row.get("minimum_scenario_profit_factor") or -math.inf,
                    row.get("conservative_return_pct") or -math.inf,
                ),
                reverse=True,
            )
            best_candidate = eligible[0] if eligible else None
        except (OSError, ValueError, KeyError, TypeError):
            return {"status": "error", "error": "saved perpetual strategy lab report is invalid"}
        return {
            "status": "completed",
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(report_path.stat().st_mtime)
            ),
            "orders_submitted": False,
            "demo_ready": False,
            "final_holdout_evaluated": False,
            "previously_inspected_holdout_reused": False,
            "excluded_previous_holdout_start": report.get("excluded_previous_holdout_start"),
            "target_win_rate_pct": TARGET_WIN_RATE_PCT,
            "best_candidate": best_candidate,
            "candidate_disqualifications": [
                {"candidate": candidate, "reason": reason}
                for candidate, reason in exclusions.items()
            ],
            "selection_rule": (
                f"A development candidate is eligible only when all three modeled cost gates "
                f"pass the {TARGET_WIN_RATE_PCT:g}% net-of-cost win-rate target and all other "
                "risk/robustness requirements, and no previously inspected independent "
                "holdout has rejected that same fixed strategy. No eligible candidate means "
                "no selection."
            ),
            "results": grouped,
            "report_file": report_path.name,
            "rerun_blocked": True,
        }

    def start_perpetual_lab(self) -> dict[str, Any]:
        """Run fixed candidates strictly before the previously inspected holdout."""
        if not self.perpetual_only_mode:
            return {
                "status": "error",
                "error": "The development-only strategy lab requires perpetual-only mode.",
                "orders_submitted": False,
                "demo_ready": False,
            }
        report_path = self.root / PERPETUAL_LAB_REPORT_FILE
        if report_path.is_file():
            return {
                **self.perpetual_lab_status(),
                "rerun_blocked": True,
                "reason": (
                    "This development sample has already been used for candidate selection; "
                    "reruns would increase selection bias."
                ),
            }
        prior_path = self.root / "perpetual_ema_walkforward.json"
        try:
            prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
            cutoff = prior_report["final_holdout_start"]
            if not isinstance(cutoff, str) or not cutoff:
                raise ValueError("missing holdout cutoff")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {
                "status": "error",
                "error": (
                    "The saved historical EMA report and final_holdout_start cutoff are required. "
                    "The lab refuses to guess the consumed holdout boundary."
                ),
                "orders_submitted": False,
                "demo_ready": False,
            }

        with self._perpetual_lab_lock:
            if self._perpetual_lab_thread is not None and self._perpetual_lab_thread.is_alive():
                return dict(self._perpetual_lab_status)
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._perpetual_lab_status = {
                "status": "running",
                "started_at": started_at,
                "orders_submitted": False,
                "demo_ready": False,
                "final_holdout_evaluated": False,
                "excluded_previous_holdout_start": cutoff,
            }

            def run() -> None:
                try:
                    from types import SimpleNamespace

                    from tradingagents.crypto.backtest.perpetual_strategy_lab import run_lab

                    args = SimpleNamespace(
                        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                        days=730,
                        equity=1_000.0,
                        leverage=[1.0, 2.0],
                        warmup_bars=260,
                        test_bars=200,
                        exclude_after=cutoff,
                    )
                    report = run_lab(args)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    best = report.get("best_candidate")
                    print(
                        "perpetual strategy lab completed: "
                        f"best_candidate={best.get('candidate') if best else 'NONE'} "
                        f"leverage={best.get('leverage') if best else 'NONE'} "
                        f"development_folds={report.get('development_folds')} "
                        "previous_holdout_reused=False final_holdout_evaluated=False "
                        "orders_submitted=False demo_ready=False",
                        flush=True,
                    )
                    # Emit reproducible aggregate Win Rate / Return / PF results
                    # to platform logs so the report can be audited without
                    # reading dashboard credentials or exposing account secrets.
                    for ranking in report.get("rankings", []):
                        candidate = ranking["candidate"]
                        leverage = float(ranking["leverage"])
                        for scenario in ("realistic_estimate", "conservative", "stress"):
                            key = f"{candidate}/{leverage:g}x/{scenario}"
                            item = report["reports"][key]
                            metrics = item["development_only"]
                            gate = item["acceptance_gate"]
                            print(
                                "perpetual strategy result: "
                                f"candidate={candidate} leverage={leverage:g}x "
                                f"scenario={scenario} "
                                f"win_rate_pct={metrics.get('win_rate_pct')} "
                                f"return_pct={metrics.get('equal_weight_compounded_return_pct')} "
                                f"profit_factor={metrics.get('profit_factor')} "
                                f"trades={metrics.get('trades')} "
                                f"positive_folds={metrics.get('positive_folds')}/"
                                f"{metrics.get('fold_count')} "
                                f"positive_symbols={metrics.get('positive_symbols')}/"
                                f"{metrics.get('symbol_count')} "
                                f"liquidations={metrics.get('liquidations')} "
                                f"gate_passed={gate.get('passed')}",
                                flush=True,
                            )
                    with self._perpetual_lab_lock:
                        self._perpetual_lab_status = {"status": "idle"}
                    recovered = self.perpetual_lab_status()
                    with self._perpetual_lab_lock:
                        self._perpetual_lab_status = {
                            **recovered,
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                except Exception as exc:  # noqa: BLE001 - fail this research run, not the dashboard
                    with self._perpetual_lab_lock:
                        self._perpetual_lab_status = {
                            "status": "error",
                            "started_at": started_at,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "orders_submitted": False,
                            "demo_ready": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }

            self._perpetual_lab_thread = threading.Thread(
                target=run, name="perpetual-strategy-lab", daemon=True
            )
            self._perpetual_lab_thread.start()
            return dict(self._perpetual_lab_status)

    def perpetual_shadow_status(self) -> dict[str, Any]:
        """Return persisted forward-test results plus any current worker error."""
        from tradingagents.crypto.backtest.perpetual_shadow_forward import read_shadow_status

        status = read_shadow_status(self.root)
        with self._perpetual_shadow_lock:
            worker = dict(self._perpetual_shadow_status)
        if worker.get("status") in {"starting", "error"}:
            status["status"] = worker["status"]
            if worker.get("error"):
                status["last_error"] = worker["error"]
            if worker.get("started_at"):
                status.setdefault("worker_started_at", worker["started_at"])
        return status

    def start_perpetual_shadow(self) -> dict[str, Any]:
        """Run the public-data shadow evaluator; this path never constructs a broker."""
        if not self.perpetual_only_mode:
            return {
                "status": "error",
                "error": "The shadow forward evaluator requires perpetual-only mode.",
                "orders_submitted": False,
                "execution_enabled": False,
                "demo_ready": False,
            }
        with self._perpetual_shadow_lock:
            if (
                self._perpetual_shadow_thread is not None
                and self._perpetual_shadow_thread.is_alive()
            ):
                return dict(self._perpetual_shadow_status)
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._perpetual_shadow_stop.clear()
            self._perpetual_shadow_status = {
                "status": "starting",
                "started_at": started_at,
                "orders_submitted": False,
                "execution_enabled": False,
                "demo_ready": False,
            }

            def run() -> None:
                from tradingagents.crypto.backtest.perpetual_shadow_forward import (
                    run_shadow_cycle,
                )

                while not self._perpetual_shadow_stop.is_set():
                    try:
                        result = run_shadow_cycle(self.root)
                        with self._perpetual_shadow_lock:
                            self._perpetual_shadow_status = dict(result)
                        print(
                            "perpetual shadow forward cycle: "
                            f"status={result.get('status')} "
                            f"bars={result.get('forward_bars', 0)} "
                            f"rows={len(result.get('results', []))} "
                            "orders_submitted=False execution_enabled=False",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - public-data failures do not stop the dashboard
                        error = f"{type(exc).__name__}: {str(exc)[:240]}"
                        with self._perpetual_shadow_lock:
                            self._perpetual_shadow_status = {
                                "status": "error",
                                "started_at": started_at,
                                "error": error,
                                "orders_submitted": False,
                                "execution_enabled": False,
                                "demo_ready": False,
                            }
                        print(
                            "perpetual shadow forward cycle failed: "
                            f"{type(exc).__name__}; retrying after poll interval",
                            flush=True,
                        )
                    self._perpetual_shadow_stop.wait(PERPETUAL_SHADOW_POLL_SECONDS)

            self._perpetual_shadow_thread = threading.Thread(
                target=run, name="perpetual-shadow-forward", daemon=True
            )
            self._perpetual_shadow_thread.start()
            return dict(self._perpetual_shadow_status)

    def _perpetual_position_view(self, symbol: str) -> tuple[str, dict[str, Any] | None, str | None]:
        """Read a genuine Bybit Demo linear position, falling back to a validated local snapshot.

        The live account path is signed and GET-only. Missing credentials mean
        "unavailable", never "flat"; Spot positions are never consulted.
        """
        from tradingagents.crypto.bybit_perpetual_account import (
            BybitLinearDemoPositionClient,
            BybitPerpetualAccountNotConfigured,
        )
        try:
            account_client = (
                self._perpetual_account_client_factory()
                if self._perpetual_account_client_factory is not None
                else BybitLinearDemoPositionClient()
            )
            result = account_client.get_position(symbol)
            status = result.get("status")
            position = result.get("position")
            if status == "flat" and position is None:
                return "flat", None, "validated futures position source"
            if status == "open" and isinstance(position, dict):
                position["position_source"] = "Bybit Demo read-only position endpoint"
                return "open", position, position.get("position_source")
            return "invalid", None, None
        except BybitPerpetualAccountNotConfigured:
            # With no credentials, a worker may still publish a validated local
            # snapshot. A failed signed request, by contrast, must not be hidden
            # by an older cached position.
            pass
        except Exception:  # noqa: BLE001 - account details/secrets never reach the page
            return "error", None, None

        path = self.root / PERPETUAL_POSITION_FILE
        if not path.is_file():
            return "unavailable", None, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid", None, None
        if not isinstance(raw, dict):
            return "invalid", None, None
        state = str(raw.get("state", "")).strip().upper()
        raw_symbol = str(raw.get("symbol", "")).strip().upper()
        if raw_symbol != symbol:
            return "invalid", None, None
        if state == "FLAT":
            return "flat", None, "validated futures position source"
        if state != "OPEN":
            return "invalid", None, None
        side = str(raw.get("side", "")).strip().lower()
        if side not in {"long", "short"}:
            return "invalid", None, None

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
                "position_source": "validated local futures snapshot",
            }
            pnl = position["unrealized_pnl"]
            if pnl is not None:
                if isinstance(pnl, bool) or not math.isfinite(float(pnl)):
                    raise ValueError("unrealized_pnl")
                position["unrealized_pnl"] = float(pnl)
        except (TypeError, ValueError, OverflowError):
            return "invalid", None, None
        return "open", position, position.get("position_source")

    def perpetual_chart(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Read completed Bybit perpetual candles and an optional read-only Demo position.

        Only the four supported USDT linear contracts are exposed. Market candles
        are public; a position is read through an authenticated GET-only Demo client
        when its credential pair is configured. No order endpoint is called.
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
            volume = float(row.get("volume", 0.0))
            if not math.isfinite(volume) or volume < 0:
                raise RuntimeError("Bybit returned an invalid candle volume")
            candles.append({
                "time": stamp.isoformat(),
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": volume,
            })
        if not candles:
            raise RuntimeError("Bybit returned no valid completed candles")
        position_status, position, position_source = self._perpetual_position_view(market)
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
            "position_source": position_source,
            "orders_submitted": False,
        }

    def perpetual_account(self) -> dict[str, Any]:
        """Return verified Bybit Demo wallet, recent closed PnL and tracked equity.

        All exchange reads are signed GET requests. The baseline is either an
        explicit operator-supplied initial capital or the first verified equity
        snapshot. External deposits/withdrawals are not silently treated as PnL.
        """
        if not self.perpetual_only_mode:
            return {
                "status": "disabled",
                "detail": "The futures account view is available only in perpetual-only mode.",
                "orders_submitted": False,
            }
        from tradingagents.crypto.bybit_perpetual_account import (
            BybitLinearDemoPositionClient,
            BybitPerpetualAccountNotConfigured,
        )
        with self._perpetual_account_lock:
            try:
                client = (
                    self._perpetual_account_client_factory()
                    if self._perpetual_account_client_factory is not None
                    else BybitLinearDemoPositionClient()
                )
                wallet = client.get_wallet_balance()
                if wallet.get("status") != "ok":
                    raise ValueError("Bybit wallet payload did not pass validation")
                equity = float(wallet["total_equity_usd"])
                if not math.isfinite(equity) or equity < 0:
                    raise ValueError("Bybit returned an invalid account equity")
                try:
                    closed = client.get_closed_pnl(limit=100)
                    if closed.get("status") != "ok":
                        raise ValueError("closed-PnL payload did not pass validation")
                    closed_error_type = None
                except Exception as exc:  # noqa: BLE001 - wallet data remains independently useful
                    closed = {"status": "error", "records": [], "window": None,
                              "next_page_available": False}
                    closed_error_type = type(exc).__name__
            except BybitPerpetualAccountNotConfigured:
                return {
                    "status": "not_configured",
                    "detail": "Read-only Bybit Demo wallet credentials are not configured.",
                    "orders_submitted": False,
                }
            except Exception as exc:  # noqa: BLE001 - never expose credentials or response bodies
                return {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "detail": "Bybit Demo account data is unavailable or failed validation; balances are hidden.",
                    "orders_submitted": False,
                }

            baseline_env = os.environ.get(PERPETUAL_ACCOUNT_BASELINE_ENV, "").strip()
            baseline_source = "first_verified_snapshot"
            baseline: float | None = None
            baseline_at: str | None = None
            if baseline_env:
                try:
                    baseline = float(baseline_env)
                except ValueError:
                    return {
                        "status": "error",
                        "error_type": "InvalidInitialCapital",
                        "detail": f"{PERPETUAL_ACCOUNT_BASELINE_ENV} must be a positive USDT amount.",
                        "orders_submitted": False,
                    }
                if not math.isfinite(baseline) or baseline <= 0:
                    return {
                        "status": "error",
                        "error_type": "InvalidInitialCapital",
                        "detail": f"{PERPETUAL_ACCOUNT_BASELINE_ENV} must be a positive USDT amount.",
                        "orders_submitted": False,
                    }
                baseline_source = "configured_initial_capital"
                baseline_at = "operator-configured"
            else:
                baseline_path = self.root / PERPETUAL_ACCOUNT_BASELINE_FILE
                if baseline_path.is_file():
                    try:
                        saved = json.loads(baseline_path.read_text(encoding="utf-8"))
                        baseline = float(saved["initial_capital_usd"])
                        baseline_at = str(saved["baseline_at"])
                        if not math.isfinite(baseline) or baseline <= 0:
                            raise ValueError("invalid saved initial capital")
                    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                        return {
                            "status": "error",
                            "error_type": "InvalidBaseline",
                            "detail": "Saved initial-capital baseline is invalid; it was not overwritten.",
                            "orders_submitted": False,
                        }
                else:
                    if equity <= 0:
                        return {
                            "status": "error",
                            "error_type": "ZeroInitialEquity",
                            "detail": "The first verified wallet equity is zero; set the actual original capital with TRADINGAGENTS_INITIAL_CAPITAL_USDT before calculating a return.",
                            "orders_submitted": False,
                        }
                    baseline = equity
                    baseline_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    baseline_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = baseline_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps({
                            "initial_capital_usd": baseline,
                            "baseline_at": baseline_at,
                            "source": baseline_source,
                        }, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary, baseline_path)

            assert baseline is not None
            now_ms = int(time.time() * 1000)
            pnl = equity - baseline
            return_pct = pnl / baseline * 100.0
            history_path = self.root / PERPETUAL_ACCOUNT_HISTORY_FILE
            history_error = None
            try:
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
                if not isinstance(history, list):
                    raise ValueError("history is not a list")
                if not history or now_ms - int(history[-1].get("timestamp_ms", 0)) >= 60_000:
                    history.append({
                        "timestamp_ms": now_ms,
                        "equity_usd": equity,
                        "wallet_balance_usd": wallet.get("total_wallet_balance_usd"),
                        "perpetual_unrealized_pnl_usd": wallet.get("total_perpetual_unrealized_pnl_usd"),
                    })
                    history = history[-1000:]
                    temporary = history_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(history, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary, history_path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
                history = []
                history_error = "Equity history could not be read or persisted."

            records = closed.get("records", []) if closed.get("status") == "ok" else []
            cumulative = 0.0
            closed_curve = []
            for record in records:
                cumulative += float(record["closed_pnl"])
                closed_curve.append({
                    "timestamp_ms": record["created_time_ms"],
                    "cumulative_closed_pnl": cumulative,
                    "trade_pnl": record["closed_pnl"],
                    "symbol": record["symbol"],
                })
            return {
                "status": "ok",
                "account": wallet,
                "initial_capital_usd": baseline,
                "initial_capital_source": baseline_source,
                "initial_capital_at": baseline_at,
                "net_pnl_since_initial_capital_usd": pnl,
                "return_since_initial_capital_pct": return_pct,
                "equity_history": history,
                "closed_pnl_curve": closed_curve,
                "closed_trades": records,
                "closed_pnl_status": closed.get("status", "error"),
                "closed_pnl_error_type": closed_error_type,
                "closed_pnl_window": closed.get("window"),
                "closed_pnl_next_page_available": bool(closed.get("next_page_available")),
                "equity_history_note": history_error,
                "capital_flow_adjusted": False,
                "capital_flow_note": (
                    "Return is equity versus the configured or first-observed baseline. "
                    "Deposits and withdrawals are not yet reconciled; configure the original "
                    "capital explicitly and exclude external transfers before treating this "
                    "percentage as an investment return."
                ),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "Bybit Demo read-only wallet and linear closed-PnL endpoints",
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
                self._send_html(200, PERPETUAL_INDEX_HTML if dashboard.perpetual_only_mode else INDEX_HTML)
                return
            if not self._authed():
                return
            if dashboard.perpetual_only_mode and path.startswith("/api/") and path not in {
                "/api/perpetual/study/status", "/api/perpetual/chart",
                "/api/perpetual/lab/status", "/api/perpetual/account",
                "/api/perpetual/shadow/status"
            }:
                self._send_json(410, {"error": "Spot/equity API is disabled in perpetual-only mode"})
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
                elif path == "/api/perpetual/lab/status":
                    self._send_json(200, dashboard.perpetual_lab_status())
                elif path == "/api/perpetual/account":
                    self._send_json(200, dashboard.perpetual_account())
                elif path == "/api/perpetual/shadow/status":
                    self._send_json(200, dashboard.perpetual_shadow_status())
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
            if path == "/api/perpetual/lab/start":
                if not self._authed():
                    return
                if not dashboard.perpetual_only_mode:
                    self._send_json(409, {"error": "perpetual-only mode is required for the strategy lab"})
                    return
                if (self.headers.get("X-Perpetual-Lab-Confirm") or "").lower() != "yes":
                    self._send_json(403, {"error": "explicit strategy-lab confirmation required"})
                    return
                self._send_json(202, dashboard.start_perpetual_lab())
                return
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
    perpetual_only_mode = _perpetual_only_enabled()
    if perpetual_only_mode:
        # Avoid even resolving or logging Spot/paper providers on the futures-only service.
        provider = PROVIDER_BAR_OPEN
        max_age = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS
        whole_shares = WHOLE_SHARES
        decision_source = DECISION_SOURCE_HISTORY
        execution_provider = EXECUTION_PROVIDER_ALPACA_PAPER
        max_price_deviation = DEFAULT_MAX_PRICE_DEVIATION
        crypto_symbols = ()
        print("perpetual-only mode: Spot/equity providers and paper settings skipped", flush=True)
    else:
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
            market_data_factory=(
                None if perpetual_only_mode
                else _market_data_factory_for(provider, max_age_seconds=max_age)
            ),
            paper_whole_shares=whole_shares,
            paper_market_data_provider=provider,
            paper_market_data_max_age_seconds=max_age,
            paper_decision_source=decision_source,
            paper_max_price_deviation=max_price_deviation,
            paper_execution_provider=execution_provider,
            crypto_symbols=crypto_symbols,
            crypto_live=bool(getattr(args, "crypto_live", False)) and not perpetual_only_mode,
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
    _maybe_auto_start_perpetual_lab(dashboard, root)
    _maybe_auto_start_perpetual_shadow(dashboard, root)
    httpd = ThreadingHTTPServer((host, port), make_handler(dashboard))
    httpd.daemon_threads = True
    print(f"dashboard on http://{host}:{port}  (root: {Path(root).resolve()})")
    print(f"  auth: {'token required' if resolved else 'loopback only, no token'}")
    print(f"  control: {'read-only' if perpetual_only_mode else ('enabled' if allow_control else 'read-only')}")
    if perpetual_only_mode:
        print("  mode: linear USDT perpetual research only; Spot/paper stack not initialized", flush=True)
    else:
        print(
            "  paper trading: "
            + (
                f"enabled ({banner_for_provider(paper_execution_provider)}, order notional {paper_order_notional:g})"
                if allow_paper_trading
                else "disabled (pass --allow-paper-trading to enable)"
            )
        )
    if auto_advance and perpetual_only_mode:
        print("  auto-advance: disabled in perpetual-only mode", flush=True)
    elif auto_advance:
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
        if dashboard.paper is not None:
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
    $("pos-note").textContent = "broker snapshot " + rel(pos.positions_checked_at) + (pos.quote_checked_at ? " · price checked " + rel(pos.quote_checked_at) : (pos.quote_refresh_status === "error" ? " · price refresh failed" : ""));
    $("pos-body").innerHTML = rows.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Symbol</th><th class="num">Qty</th><th class="num">Entry</th>
        <th class="num">Current</th><th class="num">Value</th><th class="num">uP&L</th>
        <th class="num">Stop</th><th class="num">Target</th><th>Protection</th><th>Ledger</th></tr></thead>
      <tbody>${rows.map(x=>`<tr>
        <td><b>${esc(x.symbol)}</b><div class="mono">${esc(x.side)}</div></td>
        <td class="num">${num(x.quantity,4)}</td>
        <td class="num">${money(x.entry_price)}</td>
        <td class="num">${money(x.current_price)}</td>
        <td class="num">${money(x.market_value)}</td>
        <td class="num ${cls(x.unrealized_pnl)}">${sgn(x.unrealized_pnl)}</td>
        <td class="num">${money(x.stop_loss)}</td>
        <td class="num">${money(x.take_profit)}</td>
        <td title="${esc(x.protection_reason || "")}">${x.protected?'<span class="pill s-evaluated">verified</span>':'<span class="pill s-unevaluable">not verified</span>'}<div class="mono">${esc(x.protection_state || "UNKNOWN")}</div></td>
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
    $("livewarn").textContent = s.banner || "PAPER TRADING — NOT LIVE";
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

PERPETUAL_INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark"><title>TradingAgents · Perpetual Lab</title>
<style>
:root{color-scheme:dark;--bg:#10070c;--panel:#24131b;--line:#51303f;--muted:#c4aab5;--text:#fff3f7;--cyan:#d58aa6;--green:#59d69d;--red:#ff7f98;--amber:#e9b86f;--blue:#c78ba7}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:radial-gradient(ellipse at 12% 0%,rgba(128,28,64,.24),transparent 38%),#10070c;color:var(--text);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
body{padding:env(safe-area-inset-top) 0 calc(env(safe-area-inset-bottom) + 20px)}button,input,select{font:inherit}button{cursor:pointer}
.shell{width:min(1440px,100%);margin:auto;padding:0 22px}.top{position:sticky;top:0;z-index:5;background:rgba(16,7,12,.94);backdrop-filter:blur(16px);border-bottom:1px solid rgba(194,120,149,.22)}
.toprow{display:flex;align-items:center;gap:12px;min-height:66px}.brand{font-weight:850;letter-spacing:.1em;text-transform:uppercase}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:.15em}.push{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:#2a1821;color:var(--muted);border-radius:999px;padding:5px 9px;font-size:10px;font-weight:800;letter-spacing:.05em;white-space:nowrap}.safe{color:var(--green);border-color:rgba(89,214,157,.34)}.warn{color:var(--amber);border-color:rgba(233,184,111,.34)}
.hero{padding:24px 0 16px;display:flex;justify-content:space-between;align-items:end;gap:16px}.kicker{font-size:10px;color:var(--cyan);font-weight:800;letter-spacing:.2em}h1{font-size:clamp(26px,4vw,40px);line-height:1.08;margin:8px 0;letter-spacing:-.04em}.hero p{color:var(--muted);max-width:720px;margin:0;font-size:13px}
.actions{display:flex;gap:8px;flex-wrap:wrap}.control,button.action{border:1px solid var(--line);background:#351c29;color:var(--text);border-radius:11px;padding:10px 12px;min-height:41px}button.action.primary{background:linear-gradient(110deg,#8e254b,#c66b8d);border:0;color:#fff3f7;font-weight:850}button:disabled{opacity:.48;cursor:wait}select.control{min-width:135px}
.notice{padding:11px 13px;border:1px solid rgba(233,184,111,.28);background:rgba(233,184,111,.09);border-radius:13px;color:#f0d5de;font-size:12px;margin-bottom:14px}.notice strong{color:#ffe7ed}
.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(280px,.8fr);gap:14px;align-items:start}.panel{background:linear-gradient(145deg,rgba(46,22,32,.98),rgba(22,10,15,.98));border:1px solid var(--line);border-radius:17px;overflow:hidden;box-shadow:0 20px 50px rgba(0,0,0,.2)}
.panelhead{display:flex;align-items:center;gap:12px;padding:14px 16px;border-bottom:1px solid rgba(194,120,149,.16)}.panelhead h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.12em}.sub{font-size:11px;color:var(--muted)}
.canvaswrap{position:relative;padding:8px 9px 5px}#chart{display:block;width:100%;height:440px;touch-action:pan-y}.chart-empty{position:absolute;inset:8px 9px 5px;display:none;align-items:center;justify-content:center;text-align:center;padding:26px;color:var(--muted);background:rgba(17,8,13,.78);border-radius:12px;pointer-events:none}
.legend{display:flex;gap:12px;flex-wrap:wrap;padding:0 16px 13px;color:var(--muted);font-size:10px}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:17px;height:2px;background:var(--cyan)}.entry{background:var(--blue)}.stop{background:var(--red)}.target{background:var(--green)}.liq{background:var(--amber)}
.side{display:grid;gap:13px}.pad{padding:15px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric{padding:10px;border:1px solid rgba(194,120,149,.16);border-radius:12px;background:rgba(28,12,19,.72);min-width:0}.metric label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;font-size:18px;margin-top:3px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.metric small{font-size:10px;color:var(--muted)}
.position{display:grid;gap:7px}.position-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.position-head b{font-size:14px}.level{display:grid;grid-template-columns:1fr auto;gap:12px;padding:7px 0;border-bottom:1px solid rgba(194,120,149,.14);font-variant-numeric:tabular-nums}.level span:first-child{color:var(--muted)}.empty-state{padding:13px;border:1px dashed rgba(194,120,149,.28);border-radius:12px;color:var(--muted);font-size:12px}
.study-status{color:var(--muted);font-size:12px;margin:10px 0}.study-grid{display:grid;gap:8px}.study-row{padding:10px;border-radius:11px;background:rgba(28,12,19,.78);border:1px solid rgba(194,120,149,.16)}.study-row strong{display:block;font-size:12px;margin-bottom:4px}.study-row p{margin:0;color:var(--muted);font-size:11px;line-height:1.6}.gate-pass{color:var(--green)}.gate-fail{color:var(--red)}
#error{display:none;margin:12px 0;padding:10px 13px;border:1px solid rgba(255,127,152,.38);background:rgba(255,127,152,.10);border-radius:12px;color:#ffb6c7;font-size:12px;white-space:pre-wrap}.tokenbox{display:flex;gap:8px;max-width:430px}.tokenbox input{min-width:0;flex:1;background:#160b11;border:1px solid var(--line);border-radius:10px;padding:10px 12px;color:var(--text)}.muted{color:var(--muted)}.small{font-size:11px}.footer{margin-top:14px;color:#a88998;font-size:10px;text-align:center}
.account-panel{grid-column:1/-1;background:linear-gradient(135deg,rgba(58,19,38,.98),rgba(24,10,17,.99));border-color:rgba(194,120,149,.25)}
.account-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}
.account-chart{min-width:0;padding:12px;border:1px solid rgba(194,120,149,.16);border-radius:13px;background:rgba(18,8,13,.62)}
.account-chart h3{font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 7px;color:#f1dbe3}
.account-chart canvas{display:block;width:100%;height:190px}
.account-meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
.account-note{font-size:11px;color:var(--muted);line-height:1.6;margin:10px 0 0}
.account-table{overflow:auto;margin-top:12px}
.account-table table{border-collapse:collapse;width:100%;min-width:620px;font-size:11px;font-variant-numeric:tabular-nums}
.account-table th,.account-table td{padding:8px 9px;border-bottom:1px solid rgba(194,120,149,.13);text-align:right;white-space:nowrap}
.account-table th:first-child,.account-table td:first-child{text-align:left}
.account-table th{color:var(--muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase}
.metric .positive{color:var(--green)}.metric .negative{color:var(--red)}
.account-heading{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.account-heading h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0}
@media(max-width:900px){.layout{grid-template-columns:minmax(0,1fr)}.side{grid-template-columns:repeat(2,minmax(0,1fr))}.side .wide{grid-column:1/-1}}
@media(max-width:620px){.shell{padding:0 12px}.toprow{min-height:59px}.brand{font-size:13px}.hero{display:block;padding:18px 0 13px}.hero .actions{margin-top:13px}.actions{width:100%}.actions>*{flex:1}.layout{gap:10px}.side{grid-template-columns:1fr}.side .wide{grid-column:auto}.panelhead,.pad{padding:12px}#chart{height:335px}.metric b{font-size:16px}.account-grid{grid-template-columns:1fr}.account-chart canvas{height:170px}.account-panel{padding:12px}}
@media(max-width:360px){.shell{padding:0 9px}.pill{font-size:9px;padding:5px 6px}#chart{height:295px}.tokenbox{flex-direction:column}.tokenbox button{width:100%}.metric{padding:8px}.metric b{font-size:15px}}
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
<section class="panel pad account-panel">
<div class="account-heading"><h2>Demo account & capital performance</h2><div class="push"></div><span id="account-status" class="pill warn">ACCOUNT UNKNOWN</span><button id="refresh-account" class="action" type="button">Refresh account</button></div>
<div id="account-metrics" class="metrics" style="grid-template-columns:repeat(auto-fit,minmax(145px,1fr));margin-top:13px"><div class="empty-state">Waiting for authenticated read-only Bybit Demo wallet data…</div></div>
<div class="account-grid">
<div class="account-chart"><h3>Tracked account equity</h3><canvas id="equity-chart" aria-label="Account equity history"></canvas><div id="equity-chart-note" class="muted small">History begins when monitoring first connects.</div></div>
<div class="account-chart"><h3>Realized P&amp;L · recent 7 days</h3><canvas id="realized-chart" aria-label="Cumulative realized profit and loss chart"></canvas><div id="realized-chart-note" class="muted small">Closed-position records returned by Bybit.</div></div>
</div>
<div class="account-meta"><span id="account-updated" class="muted small">Updated: —</span><span id="account-baseline" class="muted small">Initial capital: —</span></div>
<p id="account-note" class="account-note">Balances and performance are shown only after a successful authenticated Bybit Demo read. No orders are sent.</p>
<div class="account-table" id="closed-trades"><div class="empty-state">Closed trades will appear here after account data is available.</div></div>
</section>
<div class="panel"><div class="panelhead"><div><h2 id="chart-title">BTCUSDT · 4H</h2><div class="sub" id="chart-sub">Bybit public linear perpetual data</div></div><div class="push"></div><span class="pill" id="market-state">CONNECTING</span></div><div class="canvaswrap"><canvas id="chart" aria-label="Bybit 4-hour candlestick chart"></canvas><div id="chart-empty" class="chart-empty"></div></div><div class="legend"><span><i class="swatch"></i>Candles</span><span><i class="swatch entry"></i>Entry</span><span><i class="swatch stop"></i>Stop loss</span><span><i class="swatch target"></i>Take profit</span><span><i class="swatch liq"></i>Est. liquidation</span></div></div>
<aside class="side">
<section class="panel pad"><div class="panelhead" style="padding:0 0 12px;border:0"><h2>Market snapshot</h2></div><div class="metrics"><div class="metric"><label>Last close</label><b id="last-price">—</b><small id="last-time">—</small></div><div class="metric"><label>4H candles</label><b id="bars">—</b><small>Completed only</small></div><div class="metric"><label>Instrument</label><b id="instrument">—</b><small>USDT linear</small></div><div class="metric"><label>Data updated</label><b id="updated">—</b><small>Bybit public endpoint</small></div></div></section>
<section class="panel pad"><div class="position-head"><h2 style="font-size:12px;letter-spacing:.12em;margin:0">Open futures position</h2><span id="position-badge" class="pill">UNKNOWN</span></div><div id="position-body" class="empty-state" style="margin-top:11px">Checking position source…</div></section>
<section class="panel pad wide"><div class="position-head"><h2 style="font-size:12px;letter-spacing:.12em;margin:0">Shadow Forward Test</h2><span id="shadow-badge" class="pill warn">STARTING</span></div><p id="shadow-status" class="study-status">Monitoring future completed Bybit 4h candles. Five fixed candidates · 1x / 2x · realistic, conservative and stress costs. No orders.</p><div id="shadow-results" class="study-grid" style="margin-top:12px"><div class="empty-state">Waiting for the first untouched forward candles.</div></div><p id="shadow-updated" class="muted small" style="margin-top:10px">The forward boundary is persisted on the research volume.</p></section>
<section class="panel pad wide"><div class="position-head"><h2 style="font-size:12px;letter-spacing:.12em;margin:0">Strategy tournament</h2><span id="study-badge" class="pill">IDLE</span></div><p id="study-status" class="study-status">Five fixed EMA and Donchian families · 1x / 2x · three cost models · development folds only.</p><button id="run-study" class="action primary" type="button" style="width:100%">Run strategy tournament</button><div id="study-results" class="study-grid" style="margin-top:12px"></div></section>
</aside></section>
<div class="footer">Bybit USDT linear perpetuals · authenticated Demo account is read-only · no Spot engine · no orders submitted</div>
</main>
<script>
(function(){
"use strict";
var $=function(id){return document.getElementById(id);},TOKEN=localStorage.getItem("ta_token")||"",DATA=null,ACCOUNT_DATA=null,STUDY_TIMER=null,CHART_TIMER=null,ACCOUNT_TIMER=null,SHADOW_TIMER=null;
function fmt(v,d){if(v===null||v===undefined||!Number.isFinite(Number(v)))return "—";return Number(v).toLocaleString(undefined,{maximumFractionDigits:d===undefined?2:d});}
function esc(v){return String(v===null||v===undefined?"":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
function showError(s){var el=$("error");el.style.display=s?"block":"none";el.textContent=s||"";}
async function api(path,opts){opts=opts||{};opts.headers=Object.assign({},opts.headers||{},{Authorization:"Bearer "+TOKEN});var r=await fetch(path,opts),j={};try{j=await r.json();}catch(e){}if(r.status===401)throw new Error("Access token is invalid or missing.");if(!r.ok)throw new Error(j.error||("HTTP "+r.status));return j;}
function connect(){TOKEN=$("token").value.trim();if(!TOKEN){showError("Enter the dashboard token.");return;}localStorage.setItem("ta_token",TOKEN);boot();}
function boot(){ $("auth").style.display="none";$("main").style.display="grid";showError("");refreshChart();refreshStudy();refreshAccount();refreshShadow();if(CHART_TIMER)clearInterval(CHART_TIMER);if(STUDY_TIMER)clearInterval(STUDY_TIMER);if(ACCOUNT_TIMER)clearInterval(ACCOUNT_TIMER);if(SHADOW_TIMER)clearInterval(SHADOW_TIMER);CHART_TIMER=setInterval(refreshChart,60000);STUDY_TIMER=setInterval(refreshStudy,10000);ACCOUNT_TIMER=setInterval(refreshAccount,60000);SHADOW_TIMER=setInterval(refreshShadow,60000);}
function drawChart(payload){
 var canvas=$("chart"),r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(250,r.width),h=Math.max(260,r.height);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);var ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 var candles=payload.candles||[];if(!candles.length){$("chart-empty").style.display="flex";$("chart-empty").textContent="No completed Bybit candles available.";return;}$("chart-empty").style.display="none";
 var p=payload.position,levels=[];if(p){if(p.entry_price)levels.push({name:"ENTRY",v:Number(p.entry_price),c:"#c78ba7"});if(p.stop_loss)levels.push({name:"STOP",v:Number(p.stop_loss),c:"#ff7f98"});if(p.take_profit)levels.push({name:"TARGET",v:Number(p.take_profit),c:"#59d69d"});if(p.liquidation_price)levels.push({name:"LIQUIDATION",v:Number(p.liquidation_price),c:"#e9b86f"});}
 var vals=[];candles.forEach(function(c){vals.push(Number(c.high),Number(c.low));});levels.forEach(function(x){vals.push(x.v);});vals=vals.filter(Number.isFinite);var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals),span=hi-lo;if(!Number.isFinite(span)||span<=0)span=Math.max(1,Math.abs(hi)*.01);lo-=span*.08;hi+=span*.08;
 var left=10,right=91,top=18,bottom=35,pb=h-52,ph=pb-top,pw=w-left-right,step=pw/candles.length;function y(v){return top+(hi-v)/(hi-lo)*ph;}
 ctx.font="10px ui-monospace,monospace";ctx.strokeStyle="rgba(196,170,181,.18)";ctx.fillStyle="#c4aab5";ctx.textAlign="left";
 for(var i=0;i<=5;i++){var yy=top+ph*i/5;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.fillText(fmt(hi-(hi-lo)*i/5,2),w-right+7,yy+3);}
 var maxVol=Math.max.apply(null,candles.map(function(c){return Number(c.volume)||0;}))||1;
 candles.forEach(function(c,i){var x=left+step*(i+.5),o=Number(c.open),cl=Number(c.close),hh=Number(c.high),ll=Number(c.low);if(![o,cl,hh,ll].every(Number.isFinite))return;var up=cl>=o,col=up?"#59d69d":"#ff7f98",bw=Math.max(2,Math.min(10,step*.62)),yo=y(o),yc=y(cl);ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(hh));ctx.lineTo(x,y(ll));ctx.stroke();ctx.fillRect(x-bw/2,Math.min(yo,yc),bw,Math.max(1,Math.abs(yc-yo)));var vol=Math.max(0,Number(c.volume)||0);ctx.globalAlpha=.28;ctx.fillRect(x-bw/2,pb+6,bw,22*vol/maxVol);ctx.globalAlpha=1;});
 ctx.fillStyle="#c4aab5";ctx.textAlign="center";[0,Math.floor(candles.length/2),candles.length-1].forEach(function(i){var t=new Date(candles[i].time);ctx.fillText(isNaN(t.getTime())?"":t.toLocaleDateString(undefined,{month:"short",day:"numeric"}),left+step*(i+.5),h-10);});
 var lastY=-999;levels.sort(function(a,b){return b.v-a.v;}).forEach(function(l){var yy=y(l.v);if(yy<top||yy>pb)return;ctx.save();ctx.strokeStyle=l.c;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.restore();var ly=Math.max(top+9,Math.min(pb-9,yy));if(ly-lastY<19)ly=Math.min(pb-9,lastY+19);lastY=ly;ctx.fillStyle="rgba(17,8,13,.98)";ctx.fillRect(w-right+3,ly-9,right-5,18);ctx.fillStyle=l.c;ctx.textAlign="left";ctx.textBaseline="middle";ctx.font="9px ui-monospace,monospace";ctx.fillText(l.name+" "+fmt(l.v,2),w-right+7,ly);});
 if(p&&p.mark_price){ctx.strokeStyle="#d58aa6";ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(left,y(Number(p.mark_price)));ctx.lineTo(w-right,y(Number(p.mark_price)));ctx.stroke();ctx.setLineDash([]);}
}
function renderPosition(d){var s=d.position_status||"unavailable",p=d.position,b=$("position-badge"),el=$("position-body");b.textContent=s.toUpperCase();b.className="pill "+(s==="open"?"safe":s==="flat"?"":"warn");
 if(s==="unavailable"){el.className="empty-state";el.textContent="No authenticated Bybit Futures position feed is connected. The dashboard will not infer a position from Spot data or invent entry levels. Market chart only.";return;}
 if(s==="invalid"){el.className="empty-state";el.textContent="Position snapshot failed validation and is hidden.";return;}
 if(s==="error"){el.className="empty-state";el.textContent="The Bybit Demo position read failed. Position status is unknown, not flat; check read-only API permissions and connection.";return;}
 if(s==="flat"){el.className="empty-state";el.textContent="The connected futures source reports no open position.";return;}
 if(!p){el.className="empty-state";el.textContent="No verified open position is available.";return;}
 el.className="position";el.innerHTML='<div class="position-head"><b>'+esc(p.symbol)+' · '+esc(p.side.toUpperCase())+'</b><span class="pill safe">'+fmt(p.leverage,1)+'x</span></div>'+
 [['Quantity',fmt(p.quantity,6)],['Entry',fmt(p.entry_price,4)],['Mark',fmt(p.mark_price,4)],['Unrealized P&L',fmt(p.unrealized_pnl,3)],['Stop loss',fmt(p.stop_loss,4)],['Take profit',fmt(p.take_profit,4)],['Liquidation price',fmt(p.liquidation_price,4)]].map(function(x){return '<div class="level"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>';}).join("")+
 '<div class="muted small">Snapshot: '+esc(p.updated_at||"unknown")+'</div>';
}
async function refreshChart(){var symbol=$("symbol").value;$("chart-title").textContent=symbol+" · 4H";$("market-state").textContent="LOADING";try{var d=await api("/api/perpetual/chart?symbol="+encodeURIComponent(symbol));DATA=d;var bars=d.candles||[],last=bars[bars.length-1]||{};$("market-state").textContent="LIVE DATA";$("market-state").className="pill safe";$("last-price").textContent=fmt(last.close,4);$("last-time").textContent=last.time?new Date(last.time).toLocaleString():"—";$("bars").textContent=String(bars.length);$("instrument").textContent=d.category==="linear"?"USDT LINEAR":"—";$("updated").textContent=d.updated_at?new Date(d.updated_at).toLocaleTimeString():"—";$("chart-sub").textContent="Bybit public · completed candles only";renderPosition(d);drawChart(d);showError("");}catch(e){$("market-state").textContent="UNAVAILABLE";$("market-state").className="pill warn";$("chart-empty").style.display="flex";$("chart-empty").textContent=e.message;$("position-badge").textContent="UNKNOWN";$("position-body").className="empty-state";$("position-body").textContent="Position status unavailable.";showError(e.message);}}
function drawSeries(canvasId,points,valueKey,color,baseline){
 var canvas=$(canvasId),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(220,rect.width),h=Math.max(130,rect.height);
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);var ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 var vals=(points||[]).map(function(p){return Number(p[valueKey]);}).filter(Number.isFinite);
 if(baseline!==null&&baseline!==undefined&&Number.isFinite(Number(baseline)))vals.push(Number(baseline));
 if(!vals.length){ctx.fillStyle="#c4aab5";ctx.font="12px system-ui";ctx.textAlign="center";ctx.fillText("No history recorded yet",w/2,h/2);return;}
 var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals),span=hi-lo;if(span<=0)span=Math.max(1,Math.abs(hi)*.02);lo-=span*.12;hi+=span*.12;
 var left=12,right=12,top=12,bottom=23,pw=w-left-right,ph=h-top-bottom;
 ctx.font="9px ui-monospace,monospace";ctx.fillStyle="#c4aab5";ctx.strokeStyle="rgba(196,170,181,.15)";
 for(var i=0;i<=3;i++){var gridY=top+ph*i/3;ctx.beginPath();ctx.moveTo(left,gridY);ctx.lineTo(w-right,gridY);ctx.stroke();ctx.textAlign="left";ctx.fillText(fmt(hi-(hi-lo)*i/3,2),left+2,gridY-3);}
 function y(v){return top+(hi-v)/(hi-lo)*ph;}
 if(baseline!==null&&baseline!==undefined&&Number.isFinite(Number(baseline))){ctx.save();ctx.strokeStyle="rgba(233,184,111,.8)";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(left,y(Number(baseline)));ctx.lineTo(w-right,y(Number(baseline)));ctx.stroke();ctx.restore();}
 if(points&&points.length){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.forEach(function(p,i){var x=left+(points.length===1?pw/2:pw*i/(points.length-1)),v=Number(p[valueKey]);if(!Number.isFinite(v))return;if(i===0)ctx.moveTo(x,y(v));else ctx.lineTo(x,y(v));});ctx.stroke();var last=Number(points[points.length-1][valueKey]);if(Number.isFinite(last)){ctx.fillStyle=color;ctx.beginPath();ctx.arc(w-right-1,y(last),3.2,0,Math.PI*2);ctx.fill();}}
 ctx.fillStyle="#c4aab5";ctx.textAlign="left";ctx.fillText(points&&points.length?new Date(points[0].timestamp_ms).toLocaleDateString(): "",left,h-5);ctx.textAlign="right";ctx.fillText(points&&points.length?new Date(points[points.length-1].timestamp_ms).toLocaleDateString():"",w-right,h-5);
}
function accountMetric(label,value,hint,kind){return '<div class="metric"><label>'+esc(label)+'</label><b class="'+(kind||"")+'">'+value+'</b><small>'+esc(hint||"")+'</small></div>';}
function renderAccount(d){
 ACCOUNT_DATA=d;var badge=$("account-status"),metrics=$("account-metrics"),note=$("account-note");
 badge.textContent=String(d.status||"unknown").replace(/_/g," ").toUpperCase();badge.className="pill "+(d.status==="ok"?"safe":"warn");
 if(d.status!=="ok"){
  metrics.innerHTML='<div class="empty-state" style="grid-column:1/-1">'+esc(d.detail||"Account data is unavailable. Balances are intentionally hidden.")+'</div>';
  $("equity-chart-note").textContent="No verified account equity history.";
  $("realized-chart-note").textContent="No verified closed-PnL history.";
  $("closed-trades").innerHTML='<div class="empty-state">Closed trades are unavailable until the Bybit Demo read succeeds.</div>';
  $("account-updated").textContent="Updated: —";$("account-baseline").textContent="Initial capital: —";
  note.textContent="The dashboard never treats an API error as a zero balance or a flat account. Check Bybit Demo read-only Account and Contract-position permissions.";
  drawSeries("equity-chart",[],"equity_usd","#d58aa6",null);drawSeries("realized-chart",[],"cumulative_closed_pnl","#59d69d",0);return;
 }
 var a=d.account||{},trades=d.closed_trades||[],realized=trades.reduce(function(sum,t){return sum+Number(t.closed_pnl||0);},0);
 var marginUse=(a.total_initial_margin_usd!==null&&a.total_initial_margin_usd!==undefined&&Number(a.total_equity_usd)>0)?Number(a.total_initial_margin_usd)/Number(a.total_equity_usd)*100:null;
 var pnl=Number(d.net_pnl_since_initial_capital_usd),ret=Number(d.return_since_initial_capital_pct);
 metrics.innerHTML=
 accountMetric("Account equity",fmt(a.total_equity_usd)+" USD","Bybit Unified account")+
 accountMetric("Wallet balance",fmt(a.total_wallet_balance_usd)+" USD","Excludes unrealized P&L")+
 accountMetric("Available margin",fmt(a.total_available_balance_usd)+" USD","Bybit reported available balance")+
 accountMetric("Margin balance",fmt(a.total_margin_balance_usd)+" USD","Wallet + perpetual UPL")+
 accountMetric("Perpetual unrealized P&L",fmt(a.total_perpetual_unrealized_pnl_usd)+" USD","All linear / futures UPL",Number(a.total_perpetual_unrealized_pnl_usd)>0?"positive":Number(a.total_perpetual_unrealized_pnl_usd)<0?"negative":"")+
 accountMetric("Initial capital",fmt(d.initial_capital_usd)+" USD",d.initial_capital_source==="configured_initial_capital"?"Operator configured":"First verified snapshot")+
 accountMetric("Net P&L since baseline",(pnl>0?"+":"")+fmt(pnl)+" USD","Equity minus baseline",pnl>0?"positive":pnl<0?"negative":"")+
 accountMetric("Return since baseline",(ret>0?"+":"")+fmt(ret,2)+"%","External flows not adjusted",ret>0?"positive":ret<0?"negative":"")+
 accountMetric("Realized P&L · 7d",(realized>0?"+":"")+fmt(realized)+" USD",trades.length+" closed records",realized>0?"positive":realized<0?"negative":"")+
 accountMetric("Initial margin ratio",marginUse===null?"—":fmt(marginUse,2)+"%","Initial margin / account equity");
 var history=d.equity_history||[],curve=d.closed_pnl_curve||[];
 drawSeries("equity-chart",history,"equity_usd","#d58aa6",Number(d.initial_capital_usd));
 drawSeries("realized-chart",curve,"cumulative_closed_pnl",curve.length&&Number(curve[curve.length-1].cumulative_closed_pnl)<0?"#ff7f98":"#59d69d",0);
 $("equity-chart-note").textContent=history.length+" saved snapshots · tracking starts when this dashboard first reads the wallet.";
 $("realized-chart-note").textContent=(d.closed_pnl_window||"Recent records")+" · cumulative closed P&L from returned records only.";
 $("account-updated").textContent="Updated: "+(d.updated_at?new Date(d.updated_at).toLocaleString():"—");
 $("account-baseline").textContent="Baseline: "+fmt(d.initial_capital_usd)+" USD · "+esc(d.initial_capital_at||"");
 note.textContent=(d.capital_flow_note||"")+(d.equity_history_note?(" "+d.equity_history_note):"");
 var recent=trades.slice().sort(function(x,y){return y.created_time_ms-x.created_time_ms;}).slice(0,8);
 $("closed-trades").innerHTML=recent.length?'<table><thead><tr><th>Closed</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Closed P&L</th><th>Fees</th></tr></thead><tbody>'+recent.map(function(t){var fees=Number(t.open_fee||0)+Number(t.close_fee||0),v=Number(t.closed_pnl);return '<tr><td>'+esc(new Date(t.created_time_ms).toLocaleString())+'</td><td><b>'+esc(t.symbol)+'</b></td><td>'+esc(t.side)+'</td><td>'+fmt(t.quantity,5)+'</td><td>'+fmt(t.entry_price,4)+'</td><td>'+fmt(t.exit_price,4)+'</td><td class="'+(v>0?"positive":v<0?"negative":"")+'">'+(v>0?"+":"")+fmt(v,3)+'</td><td>'+fmt(fees,4)+'</td></tr>';}).join("")+'</tbody></table>':'<div class="empty-state">No closed linear-perpetual records were returned for the recent 7-day window.</div>';
}
async function refreshAccount(){try{renderAccount(await api("/api/perpetual/account"));}catch(e){renderAccount({status:"error",detail:e.message});}}
function renderShadow(s){var b=$("shadow-badge"),st=$("shadow-status"),box=$("shadow-results"),updated=$("shadow-updated");var state=String(s.status||"idle");b.textContent=state.replace(/_/g," ").toUpperCase();b.className="pill "+(state==="running"?"safe":state==="error"?"warn":"");var detail=s.detail||"All five fixed candidates are being tracked in parallel; no candidate is selected.";if(state==="error")detail="Market-data cycle failed; the saved forward boundary is preserved. "+(s.last_error||s.error||"The worker will retry.");if(s.forward_start)detail+=" Forward start: "+s.forward_start+".";if(s.forward_bars!==undefined)detail+=" Future completed 4h bars: "+s.forward_bars+".";if(s.cycles!==undefined)detail+=" Completed scoring cycles: "+s.cycles+".";st.textContent=detail;updated.textContent="Last completed candle: "+(s.last_bar_time||"waiting")+" · Last update: "+(s.last_updated_at||s.last_checked_at||"—")+" · Simulated equity: 1,000 USD per symbol; not account capital.";var rows=s.results||[];if(!rows.length){box.innerHTML='<div class="empty-state">'+esc(s.detail||"Waiting for untouched future candles; no result has been scored yet.")+'</div>';return;}box.innerHTML=rows.map(function(r){var wr=r.win_rate_pct===null?"—":fmt(r.win_rate_pct,1)+"%",pf=r.profit_factor===null?"—":String(r.profit_factor),ret=r.estimated_equal_weight_return_pct===null?"—":fmt(r.estimated_equal_weight_return_pct,3)+"%";return '<div class="study-row"><strong>'+esc(r.candidate)+' · '+fmt(r.leverage,1)+'x · '+esc(r.scenario)+'</strong><p>Closed trades: '+esc(r.closed_trades)+' · WR: '+wr+' · PF: '+esc(pf)+'</p><p>Realized net P&amp;L: '+fmt(r.realized_net_pnl_usd,3)+' USD · estimated open P&amp;L after exit costs: '+fmt(r.estimated_open_pnl_after_exit_costs_usd,3)+' USD</p><p>Mean simulated return: '+ret+' · Max symbol drawdown: '+fmt(r.max_symbol_drawdown_pct,2)+'% · Open positions: '+esc(r.open_positions)+' · Liquidations: '+esc(r.liquidations)+'</p></div>';}).join("");}
async function refreshShadow(){try{renderShadow(await api("/api/perpetual/shadow/status"));}catch(e){$("shadow-status").textContent="Shadow forward status unavailable: "+e.message;}}
function renderStudy(s){var b=$("study-badge"),st=$("study-status"),box=$("study-results"),btn=$("run-study");b.textContent=String(s.status||"idle").toUpperCase();b.className="pill "+(s.status==="completed"?"safe":s.status==="error"?"warn":"");btn.disabled=s.status==="running"||s.status==="completed";btn.textContent=s.status==="running"?"TOURNAMENT RUNNING…":s.status==="completed"?"LAB COMPLETE · RERUN LOCKED":"Run strategy tournament";
 if(s.status==="running"){st.textContent="Comparing fixed EMA and Donchian candidates only on time-ordered development folds before the already-inspected holdout. No exchange orders are submitted.";return;}
 if(s.status==="error"){st.textContent="Research failed: "+(s.error||s.error_type||"unknown error");box.innerHTML="";return;}
 if(s.status!=="completed"){st.textContent="No tournament report loaded. Start the fixed-candidate development study.";box.innerHTML="";return;}
 var best=s.best_candidate,exclusions=s.candidate_disqualifications||[];
 if(best){st.textContent="Development shortlist: "+best.candidate+" at "+fmt(best.leverage,1)+"x. This is still not independent final validation.";}
 else if(exclusions.length){st.textContent="NO ELIGIBLE STRATEGY. EMA14/EMA50 was rejected by the previously inspected independent holdout at 1x and 2x. Every other candidate failed at least one cost or consistency gate. No strategy is approved for trading.";}
 else{st.textContent="No candidate passed all three after-cost development gates. No strategy is selected; execution remains disabled.";}
 box.innerHTML=(s.results||[]).map(function(row){var scenarios=row.scenarios||{},r=scenarios.realistic_estimate||{},c=scenarios.conservative||{},x=scenarios.stress||{},rejected=!!row.previous_holdout_rejected,ok=!!row.selection_eligible,label=rejected?"REJECTED BY PRIOR HOLDOUT":(ok?"ELIGIBLE":"NOT SELECTED"),reason=rejected?'<p class="gate-fail">'+esc(row.previous_holdout_rejection_reason)+'</p>':"";function m(v){return "WR "+(v.win_rate_pct===null?"—":fmt(v.win_rate_pct,1)+"%")+" · R "+(v.return_pct===null?"—":fmt(v.return_pct,2)+"%")+" · PF "+esc(v.profit_factor)+" · N "+esc(v.trades);}return '<div class="study-row"><strong>'+esc(row.candidate)+' · '+fmt(row.leverage,1)+'x <span class="'+(ok?"gate-pass":"gate-fail")+'">'+label+'</span></strong>'+reason+'<p>Realistic: '+m(r)+'</p><p>Conservative: '+m(c)+'</p><p>Stress: '+m(x)+'</p><p>Consistent folds: '+esc(c.positive_folds)+'/'+esc(c.fold_count)+' · Positive symbols: '+esc(c.positive_symbols)+'/'+esc(c.symbol_count)+' · Liquidations: '+esc(c.liquidations)+'</p></div>';}).join("");
}
async function refreshStudy(){try{renderStudy(await api("/api/perpetual/lab/status"));}catch(e){$("study-status").textContent="Lab status unavailable: "+e.message;}}
async function runStudy(){var b=$("run-study");b.disabled=true;try{renderStudy(await api("/api/perpetual/lab/start",{method:"POST",headers:{"X-Perpetual-Lab-Confirm":"yes"}}));setTimeout(refreshStudy,1200);}catch(e){showError(e.message);}finally{setTimeout(refreshStudy,1500);}}
$("connect").addEventListener("click",connect);$("token").addEventListener("keydown",function(e){if(e.key==="Enter")connect();});$("symbol").addEventListener("change",refreshChart);$("refresh").addEventListener("click",refreshChart);$("refresh-account").addEventListener("click",refreshAccount);$("run-study").addEventListener("click",runStudy);window.addEventListener("resize",function(){if(DATA)drawChart(DATA);if(ACCOUNT_DATA)renderAccount(ACCOUNT_DATA);});if(TOKEN)boot();else $("auth").style.display="block";
})();
</script>
</body></html>
"""


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
