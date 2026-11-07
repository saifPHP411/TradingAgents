"""Dashboard-side control of the Alpaca **paper** trading worker.

This is the glue that lets an authenticated dashboard operator start and stop the
broker-backed paper worker. It is the only place the dashboard reaches the
broker, and it is **off unless explicitly enabled**: a dashboard started without
``--allow-paper-trading`` reports the feature as disabled and refuses every
start/stop, so the default dashboard cannot place a paper order.

Safety properties, each pinned by a test:

- the controller builds the worker with the **paper** endpoint only; the broker's
  own gate refuses a live host, and no argument here can supply another endpoint;
- a start runs the worker's preflight (broker connection, account status, buying
  power, market data, risk config) before anything is placed, and a failed
  preflight leaves the worker in ``ERROR``, never ``RUNNING``;
- a second worker cannot be started while one is running;
- stop is graceful — it closes no position and cancels no order;
- the worker's state is read back as ``STOPPED`` after a restart.

Credentials are never read here. The worker's default broker factory reads
``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET`` from the environment; the
dashboard never sees them.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tradingagents.risk.paper_parity import WHOLE_SHARES

from ..broker import (
    BANNER,
    DECISION_SOURCE_CRYPTO,
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_HISTORY,
    DECISION_SOURCE_TJR,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    PROVIDER_BAR_OPEN,
    STATUS_STOPPED,
    AlpacaPaperBroker,
    PaperTradingConfig,
    PaperTradingWorker,
    SourceMarketData,
    StaticMarketData,
    WorkerStateStore,
    worker_status_view,
)
from ..symbol_mapping import (
    SymbolMapping,
    map_binance_to_alpaca,
)


class PaperTradingDisabled(RuntimeError):
    """The dashboard was not started with paper trading enabled."""


def _book_snapshot_from_worker(worker: Any) -> Any:
    """A read-only TJR book snapshot mirroring the worker's broker account.

    The RiskEngine gate must size against the *real* paper book, not a fabricated
    one, so this reads the worker's own last-synced account and positions. Nothing
    here writes or places an order. Cash/equity come from the account snapshot;
    the daily entry count comes from the broker's own counter when a broker exists
    (the same counter the broker enforces ``max_daily_orders`` with), else 0 —
    which is honest: with no broker there is no order history to count, and the
    broker's own limit is the backstop.
    """
    from ..tjr_source import TjrBookSnapshot

    state = getattr(worker, "state", None)
    account = getattr(state, "account", None) or {}
    cash = _as_finite(account.get("cash"))
    if cash is None:
        cash = 0.0
    positions: dict[str, float] = {}
    for symbol, record in (getattr(state, "positions", None) or {}).items():
        if not isinstance(record, dict):
            continue
        quantity = _as_finite(record.get("quantity"))
        if quantity is not None and quantity > 0:
            positions[str(symbol).upper()] = float(quantity)
    broker = getattr(worker, "_broker", None)
    daily_orders = 0
    if broker is not None:
        counter = getattr(broker, "daily_orders", None)
        if callable(counter):
            try:
                daily_orders = int(counter())
            except Exception:  # noqa: BLE001 - an unreadable counter is not fatal
                daily_orders = 0
    return TjrBookSnapshot(
        cash=float(cash),
        positions=positions,
        daily_orders=max(daily_orders, 0),
        day_start_equity=_as_finite(getattr(state, "initial_equity", None)),
    )


def _as_finite(value: Any) -> float | None:
    """``value`` as a finite float, or ``None``. A NaN never passes."""
    import math

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


class PaperTradingController:
    """Owns one paper-trading worker for one experiment directory.

    The worker is built lazily on the first ``start``, so a read-only status view
    performs no request and needs no credential.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = False,
        order_notional: float = 25.0,
        stub: bool = False,
        broker_factory: Callable[[], AlpacaPaperBroker] | None = None,
        worker_factory: Callable[..., PaperTradingWorker] | None = None,
        market_data_factory: Callable[[tuple[str, ...]], Any] | None = None,
        whole_shares: bool = WHOLE_SHARES,
        market_data_provider: str = PROVIDER_BAR_OPEN,
        market_data_max_age_seconds: float | None = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
        decision_source_kind: str = DECISION_SOURCE_HISTORY,
        current_source_factory: Callable[..., Any] | None = None,
        market_hours_gate: Callable[[], Any] | None = None,
        decision_data_factory: Callable[..., Any] | None = None,
        tjr_source_factory: Callable[..., Any] | None = None,
        tjr_data_factory: Callable[..., Any] | None = None,
        crypto_source_factory: Callable[..., Any] | None = None,
        crypto_data_factory: Callable[..., Any] | None = None,
        crypto_market_hours_gate: Callable[[], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.order_notional = order_notional
        self.stub = stub
        # Sizing follows the one parity constant (WHOLE_SHARES). Fractional sizing
        # is the default so a high-priced symbol under a small notional (e.g. SPY
        # at ~580 under $25) still buys a positive quantity instead of flooring to
        # zero, matching what the TJR backtest enforces.
        self.whole_shares = whole_shares
        self._broker_factory = broker_factory
        self._worker_factory = worker_factory or PaperTradingWorker
        # How the worker reads a *current* price. ``None`` keeps the historical
        # default (the experiment's latest bar open), so existing behaviour is
        # unchanged unless an operator opts into a snapshot source explicitly.
        self._market_data_factory = market_data_factory
        # The *configured* provider, recorded so a read-only status view can name
        # it without building a worker. ``market_data_factory`` is the seam itself;
        # this is the label, and the two are set together by the entrypoints.
        self.market_data_provider = market_data_provider
        self.market_data_max_age_seconds = market_data_max_age_seconds
        # Which decision source the worker draws on. ``history`` is the default, so
        # the existing historical path is unchanged. ``current_market`` is opt-in
        # and builds the independent current-market source instead.
        self.decision_source_kind = decision_source_kind
        # Seams the current-market path needs, injectable so a test drives them
        # offline. ``None`` means "build the real one" (Alpaca clock, Alpaca bars).
        self._current_source_factory = current_source_factory
        self._market_hours_gate = market_hours_gate
        self._decision_data_factory = decision_data_factory
        self._tjr_source_factory = tjr_source_factory
        self._tjr_data_factory = tjr_data_factory
        self._crypto_source_factory = crypto_source_factory
        self._crypto_data_factory = crypto_data_factory
        self._crypto_market_hours_gate = crypto_market_hours_gate
        self._worker: PaperTradingWorker | None = None
        self._lock = threading.Lock()

    # -- worker construction ---------------------------------------------

    def _worker_config(self) -> PaperTradingConfig:
        """The config the worker runs with, from the persisted experiment config.

        Shared by :meth:`_build_worker` and the read-only :meth:`status`, so the
        status view reports the same limits the worker would enforce without
        building — and therefore without freezing — a worker.
        """
        from ..experiment import PaperExperiment

        experiment = PaperExperiment.load(self.root, inference=None)
        return PaperTradingConfig(
            strategy_id=experiment.config.strategy_id,
            order_notional=self.order_notional,
            whole_shares=self.whole_shares,
            state_path=self.root / "worker_state.json",
            market_data_provider=self.market_data_provider,
            market_data_max_age_seconds=self.market_data_max_age_seconds,
            decision_source=self.decision_source_kind,
        )

    def _reload_experiment(self):
        """Re-read the experiment state from disk, for the decision source.

        Read-only: it loads the persisted experiment (no inference, so nothing is
        decided) and returns it, so the source sees decisions another writer has
        recorded since the worker was built. Loading never mutates the experiment.
        """
        from ..experiment import PaperExperiment

        return PaperExperiment.load(self.root, inference=None)

    def _build_worker(self) -> PaperTradingWorker:
        from ..experiment import PaperExperiment

        experiment = PaperExperiment.load(self.root, inference=None)

        # Build symbol mappings for crypto TJR path
        # Market data comes from Binance (USDT-quoted), execution on Alpaca (USD-quoted)
        symbol_mappings: tuple[SymbolMapping, ...] = ()
        if self.decision_source_kind == DECISION_SOURCE_CRYPTO:
            # For crypto paper trading, Alpaca execution registry is REQUIRED.
            # Missing credentials, network errors, malformed responses, and
            # unavailable registry must fail closed - never interpret an unavailable
            # registry as proof that a symbol is tradable.

            # Build Binance instrument registry from exchange info
            from ..crypto_intraday_data import BinancePublicClient
            binance_client = BinancePublicClient()
            binance_registry = binance_client.exchange_info()

            # Get Alpaca registry - REQUIRED for crypto paper execution
            from ..alpaca_paper import AlpacaPaperBroker
            broker = AlpacaPaperBroker(
                base_url=PaperTradingConfig().base_url,
                limits=PaperTradingConfig().limits,
            )
            broker.connect()
            alpaca_assets = broker.list_assets()
            # Build a simple registry from assets
            from datetime import UTC, datetime

            from ..crypto.instruments import InstrumentRegistry, TradingRules
            alpaca_rules = {}
            for asset in alpaca_assets:
                if asset.get("class") == "crypto" and asset.get("tradable"):
                    symbol = asset.get("symbol", "")
                    if "/" in symbol:
                        alpaca_rules[symbol] = TradingRules(
                            symbol=symbol,
                            tradable=True,
                            min_qty=float(asset.get("min_order_size", "0.00001")),
                            step_qty=float(asset.get("qty_increment", "0.00001")),
                            min_notional=float(asset.get("min_notional", "1")),
                            price_increment=float(asset.get("price_increment", "0.01")),
                            status="TRADING",
                        )
            alpaca_registry = InstrumentRegistry(
                rules=alpaca_rules,
                venue="alpaca_paper",
                fetched_at=datetime.now(UTC).isoformat(),
            )

            # Build mappings from experiment config symbols
            mappings, rejected = map_binance_to_alpaca(
                canonical_symbols=list(experiment.config.symbols),
                binance_registry=binance_registry,
                alpaca_registry=alpaca_registry,
            )
            if rejected:
                # Log rejected symbols but continue with valid ones
                import logging
                logger = logging.getLogger(__name__)
                for sym, reason in rejected.items():
                    logger.warning(f"Symbol {sym} rejected for paper trading: {reason}")

            symbol_mappings = tuple(mappings)
            if not symbol_mappings:
                raise RuntimeError(
                    "No valid symbol mappings for crypto paper trading. "
                    "All configured symbols were rejected. Check Alpaca credentials "
                    "and ensure symbols are tradable on both Binance and Alpaca Paper."
                )

        # Build the decision source from the persisted config, so the worker acts
        # on the same decisions the experiment recorded.
        config = PaperTradingConfig(
            strategy_id=experiment.config.strategy_id,
            order_notional=self.order_notional,
            whole_shares=self.whole_shares,
            state_path=self.root / "worker_state.json",
            market_data_provider=self.market_data_provider,
            market_data_max_age_seconds=self.market_data_max_age_seconds,
            decision_source=self.decision_source_kind,
            symbol_mappings=symbol_mappings,
        )
        source = experiment.source
        sessions = source.sessions()
        market = self._build_market_data(experiment, source, sessions)
        # The worker needs the decision source at construction, but the current
        # source's idempotency filter consults the worker's own durable
        # processed-decision set. A late-bound holder breaks that cycle, exactly as
        # the historical source does.
        holder: dict[str, PaperTradingWorker] = {}
        decision_source = self._build_decision_source(experiment, holder)
        kwargs: dict[str, Any] = {
            "config": config,
            "decision_source": decision_source,
            "market_data": market,
        }
        if self._broker_factory is not None:
            kwargs["broker_factory"] = self._broker_factory
        worker = self._worker_factory(**kwargs)
        holder["worker"] = worker
        return worker

    def _build_decision_source(self, experiment, holder: dict[str, PaperTradingWorker]):
        """The worker's decision source, from the configured kind.

        ``history`` (the default) builds the existing ``ExperimentDecisionSource``
        over the experiment's recorded decisions, unchanged. ``current_market``
        builds the independent ``CurrentMarketDecisionSource``, which fetches live
        Alpaca data and is never blocked by a consumed historical history. Both
        satisfy the same ``DecisionSource`` protocol. They differ in what "already
        done" means, though: the historical source filters against the worker's
        processed-decision set (a refused historical decision is deliberately not
        retried), while the current source filters against the order-only set, so a
        current decision that produced no order is re-offered rather than mistaken
        for a consumed session.
        """
        if self.decision_source_kind == DECISION_SOURCE_CURRENT_MARKET:
            return self._build_current_source(experiment, holder)
        if self.decision_source_kind == DECISION_SOURCE_TJR:
            return self._build_tjr_source(experiment, holder)
        if self.decision_source_kind == DECISION_SOURCE_CRYPTO:
            return self._build_crypto_source(experiment, holder)
        return self._build_history_source(experiment, holder)

    def _build_history_source(self, experiment, holder: dict[str, PaperTradingWorker]):
        """The existing historical decision source, unchanged.

        It re-reads the experiment state from disk on every decision, so a decision
        produced *after* this worker was built — by an autonomous ``auto-advance``
        progressing the experiment, for example — is still discovered. It also
        filters against the worker's own durable processed-decision set, wired
        through a late-bound holder so the source (which the worker needs at
        construction) can consult the worker's state once it exists.
        """
        from ..paper_trading import ExperimentDecisionSource

        return ExperimentDecisionSource(
            experiment,
            order_notional=self.order_notional,
            reload=self._reload_experiment,
            is_consumed=lambda decision_id: holder["worker"]._is_processed(decision_id),
        )

    def _build_current_source(self, experiment, holder: dict[str, PaperTradingWorker]):
        """The independent current-market decision source.

        Its decision data comes from Alpaca's market-data host, its market-hours
        gate from Alpaca's paper clock, and its strategy from the *same*
        ``ProviderInference`` seam the historical path uses (built from the
        persisted config). Its per-session memory lives in the source's own state
        file — never the experiment's — so it is not a second writer to the
        historical experiment directory. A test may inject any of the three seams
        through the controller, so the path is exercisable offline.
        """
        from ..current_source import CurrentMarketDecisionSource
        from ..runner import build_inference

        inference = build_inference(experiment.config, stub=self.stub)
        market_hours = self._build_market_hours()
        decision_data = self._build_decision_data()
        kwargs: dict[str, Any] = {
            "symbols": tuple(experiment.config.symbols),
            "inference": inference,
            "market_data": decision_data,
            "market_hours": market_hours,
            "order_notional": self.order_notional,
            "strategy_name": experiment.config.strategy_id,
            "is_consumed": lambda decision_id: holder["worker"].has_order_for(decision_id),
            "state_path": self.root / "current_source_state.json",
        }
        if self._current_source_factory is not None:
            return self._current_source_factory(**kwargs)
        return CurrentMarketDecisionSource(**kwargs)

    def _build_tjr_source(self, experiment, holder: dict[str, PaperTradingWorker]):
        """The independent TJR rule-chain decision source.

        Its decision data is current *intraday* (1m/5m) Alpaca market data, its
        market-hours gate is Alpaca's paper clock, and its strategy is the
        deterministic TJR rule chain — no model is consulted. The RiskEngine gate
        and the book it sizes against are wired here: the book is a read-only
        mirror of the worker's own broker account/positions, and the price the
        RiskEngine sizes against is the worker's execution seam, so the live path
        sizes at the same price the backtest would fill at. Its per-session memory
        lives in the source's own state file — never the experiment's. A test may
        inject any of the seams through the controller, so the path is exercisable
        offline.
        """
        from ..tjr_source import TjrBookSnapshot, TjrDecisionSource

        market_hours = self._build_market_hours()
        data = self._build_tjr_data()

        def _book() -> TjrBookSnapshot:
            worker = holder.get("worker")
            if worker is None:
                return TjrBookSnapshot(0.0)
            return _book_snapshot_from_worker(worker)

        def _price(symbol: str) -> float:
            worker = holder.get("worker")
            if worker is None:
                return 0.0
            try:
                return float(worker.market_data.price(symbol))
            except Exception:  # noqa: BLE001 - an unpriceable entry is no decision
                # 0.0 is not a price: the source falls back to the newest closed 1m
                # close and, failing that, the RiskEngine refuses to size. A market-
                # data failure must never raise out of the decision source.
                return 0.0

        kwargs: dict[str, Any] = {
            "symbols": tuple(experiment.config.symbols),
            "data": data,
            "market_hours": market_hours,
            "order_notional": self.order_notional,
            "strategy_name": experiment.config.strategy_id,
            "book_provider": _book,
            "price_provider": _price,
            "is_consumed": lambda decision_id: holder["worker"].has_order_for(decision_id),
            "state_path": self.root / "tjr_source_state.json",
        }
        if self._tjr_source_factory is not None:
            return self._tjr_source_factory(**kwargs)
        return TjrDecisionSource(**kwargs)

    def _build_tjr_data(self):
        """The TJR decision-data seam: an injected one, else Alpaca intraday bars."""
        if self._tjr_data_factory is not None:
            return self._tjr_data_factory()
        from ..tjr_intraday_data import AlpacaIntradayBars

        return AlpacaIntradayBars()

    def _build_crypto_source(self, experiment, holder: dict[str, PaperTradingWorker]):
        """The deterministic TJR rule chain over current **crypto** intraday bars.

        This is the *same* ``TjrDecisionSource`` class the ``tjr`` path builds — the
        rule chain is identical, so there is exactly one strategy — but wired with
        the crypto seams: Binance public 1m/5m bars (no key) for the decision window
        and a 24/7 crypto clock for the session. The RiskEngine gate and the book it
        sizes against are the same read-only mirror of the worker's broker
        account/positions, and the price it sizes at is the worker's own execution
        seam (``--paper-market-data binance_snapshot``). Its per-session memory lives
        in ``crypto_source_state.json`` — never the experiment's state and never the
        equity TJR source's — so it is not a second writer to the experiment
        directory. A test may inject the seams, so the path is exercisable offline.
        """
        from ..tjr_source import TjrBookSnapshot, TjrDecisionSource

        market_hours = self._build_crypto_market_hours()
        data = self._build_crypto_data()

        def _book() -> TjrBookSnapshot:
            worker = holder.get("worker")
            if worker is None:
                return TjrBookSnapshot(0.0)
            return _book_snapshot_from_worker(worker)

        def _price(symbol: str) -> float:
            worker = holder.get("worker")
            if worker is None:
                return 0.0
            try:
                return float(worker.market_data.price(symbol))
            except Exception:  # noqa: BLE001 - an unpriceable entry is no decision
                return 0.0

        kwargs: dict[str, Any] = {
            "symbols": tuple(experiment.config.symbols),
            "data": data,
            "market_hours": market_hours,
            "order_notional": self.order_notional,
            "strategy_name": experiment.config.strategy_id,
            "book_provider": _book,
            "price_provider": _price,
            "is_consumed": lambda decision_id: holder["worker"].has_order_for(decision_id),
            "state_path": self.root / "crypto_source_state.json",
        }
        if self._crypto_source_factory is not None:
            return self._crypto_source_factory(**kwargs)
        return TjrDecisionSource(**kwargs)

    def _build_crypto_data(self):
        """The crypto decision-data seam: an injected one, else Binance 1m/5m bars."""
        if self._crypto_data_factory is not None:
            return self._crypto_data_factory()
        from ..crypto_intraday_data import CryptoIntradayBars

        return CryptoIntradayBars()

    def _build_crypto_market_hours(self):
        """The crypto market-hours gate: an injected one, else the 24/7 clock."""
        if self._crypto_market_hours_gate is not None:
            return self._crypto_market_hours_gate()
        from ..crypto_market_hours import CryptoMarketHours

        return CryptoMarketHours()

    def _build_market_hours(self):
        """The market-hours gate: an injected one, else Alpaca's paper clock."""
        if self._market_hours_gate is not None:
            return self._market_hours_gate()
        from ..market_hours import AlpacaMarketClock

        return AlpacaMarketClock()

    def _build_decision_data(self):
        """The current decision-data seam: an injected one, else Alpaca bars."""
        if self._decision_data_factory is not None:
            return self._decision_data_factory()
        from ..current_market_data import AlpacaCurrentBars

        return AlpacaCurrentBars()

    def _build_market_data(self, experiment, source, sessions):
        """The worker's current-price seam.

        With no factory configured the behaviour is the historical default: the
        experiment's latest bar open. A factory lets an operator opt into a
        *current* price (Alpaca's snapshot) without changing the decision source or
        the strategy. A factory that raises leaves the worker without a price
        source, which is reported rather than papered over.
        """
        if self._market_data_factory is None:
            return (
                SourceMarketData(source, sessions[-1]) if sessions else StaticMarketData({})
            )
        return self._market_data_factory(tuple(experiment.config.symbols))

    def _ensure_worker(self) -> PaperTradingWorker:
        if self._worker is None:
            self._worker = self._build_worker()
        return self._worker

    # -- control ----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A secret-free status view. Never raises for a disabled controller.

        Read-only and **build-free**: it never constructs a worker (which would
        construct the decision source and could freeze a live view). It reports a
        live worker's state when one exists, and otherwise reads the persisted
        worker state from disk as ``STOPPED``. That is what makes a dashboard that
        polls ``status`` every few seconds leave the worker unbuilt until an
        operator presses START.
        """
        base: dict[str, Any] = {
            "banner": BANNER,
            "enabled": self.enabled,
            "mode": "ALPACA_PAPER",
            "live": False,
        }
        if not self.enabled:
            base.update({"status": "DISABLED", "reason": "paper trading is not enabled"})
            return base
        with self._lock:
            worker = self._worker
        if worker is not None:
            base.update(worker.status())
            return base
        try:
            config = self._worker_config()
        except Exception as exc:  # noqa: BLE001 - a build failure is reported
            base.update({"status": "ERROR", "last_error": f"{type(exc).__name__}: {exc}"})
            return base
        state = WorkerStateStore(self.root / "worker_state.json").load()
        # Nothing is running (no worker object exists), so the persisted state is
        # reported as STOPPED, never as a stale RUNNING from a previous process.
        state.status = STATUS_STOPPED
        base.update(worker_status_view(state=state, config=config))
        return base

    def start(self) -> dict[str, Any]:
        """Preflight, then start the worker. Places no order by itself."""
        if not self.enabled:
            raise PaperTradingDisabled(
                "paper trading is not enabled; start the dashboard with "
                "--allow-paper-trading to enable it"
            )
        with self._lock:
            worker = self._ensure_worker()
            return worker.start()

    def stop(self) -> dict[str, Any]:
        """Stop the worker gracefully. Closes no position, cancels no order.

        Signals the worker's loop, waits for the in-flight cycle to finish, and
        leaves the state ``STOPPED``. Never builds a worker just to stop one.
        """
        if not self.enabled:
            raise PaperTradingDisabled(
                "paper trading is not enabled; start the dashboard with "
                "--allow-paper-trading to enable it"
            )
        with self._lock:
            worker = self._worker
        if worker is None:
            return {"status": "STOPPED", "stopped": False, "reason": "no worker was started"}
        with self._lock:
            return worker.stop()

    def shutdown(self) -> None:
        """Best-effort stop for process shutdown. Never raises, never builds a worker.

        Only touches a worker that was already started, so an idle dashboard's
        shutdown performs no request and needs no credential.
        """
        worker = self._worker
        if worker is None:
            return
        with contextlib.suppress(Exception):
            worker.stop()

    def run_once(self) -> dict[str, Any]:
        """Process at most one decision (used by tests and an explicit CLI tick)."""
        if not self.enabled:
            raise PaperTradingDisabled("paper trading is not enabled")
        with self._lock:
            worker = self._ensure_worker()
            return worker.run_once().to_dict()
