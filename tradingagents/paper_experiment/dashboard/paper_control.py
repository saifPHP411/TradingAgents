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
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.crypto.tjr_settings import crypto_tjr_settings
from tradingagents.risk.paper_parity import WHOLE_SHARES

from ..broker import (
    DECISION_SOURCE_CRYPTO,
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_HISTORY,
    DECISION_SOURCE_TJR,
    DEFAULT_MAX_PRICE_DEVIATION,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_BYBIT_DEMO,
    PROVIDER_BAR_OPEN,
    STATUS_STOPPED,
    AlpacaPaperBroker,
    PaperTradingConfig,
    PaperTradingWorker,
    SourceMarketData,
    StaticMarketData,
    WorkerStateStore,
    banner_for_provider,
    mode_for_provider,
    worker_status_view,
)
from ..symbol_mapping import (
    SymbolMapping,
)


class PaperTradingDisabled(RuntimeError):
    """The dashboard was not started with paper trading enabled."""


def _endpoint_for_provider(execution_provider: str) -> str:
    """The endpoint the given provider trades on.

    A status view has to name the venue, and it has to name the *right* one.
    The endpoint follows the provider: a Bybit selection resolves to Bybit's
    demo host and an Alpaca selection to Alpaca's paper host. It is derived
    from the same rule the config applies, so the view and the worker cannot
    disagree about which host receives the order.

    Secret-free by construction: it is a public URL.
    """
    if execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO:
        from ..broker.bybit_demo import DEMO_BASE_URL

        return DEMO_BASE_URL
    from ..broker.alpaca_paper import PAPER_BASE_URL

    return PAPER_BASE_URL


def _strategy_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map a decision source's persisted record onto the strategy row.

    The source records what the chain concluded in its own vocabulary (``side``,
    ``stop``, ``target``, ``reference_price``); the dashboard row uses the
    trading vocabulary (``direction``, ``stop_loss``, ``take_profit``, ``entry``).
    The mapping is one way and loses nothing: a record with no side is a
    no-trade, and its levels stay ``None`` rather than being invented.

    The risk/reward is derived here rather than stored, because the source
    records the two levels and not their ratio. It is ``None`` when either level
    is missing or the risk is not positive, so an undefined ratio is never
    reported as zero.
    """
    side = record.get("side")
    direction = None
    if side == "buy":
        direction = "long"
    elif side == "sell":
        direction = "short"

    entry = record.get("reference_price")
    stop = record.get("stop")
    target = record.get("target")
    risk_reward = None
    if entry is not None and stop is not None and target is not None:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            risk_reward = reward / risk

    return {
        "stage": record.get("stage") or "NO_SIGNAL",
        "reason_code": record.get("reason_code"),
        "detail": record.get("detail"),
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": risk_reward,
        "evaluated_at": record.get("evaluated_at"),
        "actionable": side is not None,
    }


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
        max_price_deviation: float = DEFAULT_MAX_PRICE_DEVIATION,
        execution_provider: str = EXECUTION_PROVIDER_ALPACA_PAPER,
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
        # The largest permitted gap between the decision price (market-data venue)
        # and the execution price (execution venue). Beyond it, no order is placed.
        self.max_price_deviation = max_price_deviation
        # Which execution broker the worker places orders with. Alpaca Paper is
        # the default, so the existing path is unchanged; ``bybit_demo`` is an
        # explicit opt-in. It is recorded here so a read-only status view can
        # name the configured provider without building a worker.
        self.execution_provider = execution_provider
        # Broker-owned durable state must never be shared across execution venues.
        # Keep Alpaca's historical paths unchanged, but put Bybit Demo state in
        # its own namespace so an existing Railway Volume cannot make old Alpaca
        # orders, positions, fills, account snapshots or protections look like
        # Bybit state (or vice versa).
        #
        # The experiment directory itself remains shared: it contains strategy
        # inputs/decisions, not broker account or execution state.
        # The resolved symbol configuration, once the worker has been built. It
        # records which symbols were accepted on both venues and which were
        # rejected, so a status view can report both without re-reading the
        # registries (which would be a network call).
        self.symbol_configuration: Any | None = None
        # The *validation state* itself: whether both venue registries have been
        # checked yet, when, and why not. The registries are read when the
        # worker is built (deferred, so the dashboard's startup does not depend
        # on either exchange being reachable), so before that point the honest
        # answer is "not validated yet" — not "everything is fine".
        self.symbol_validation: dict[str, Any] = {
            "state": "NOT_VALIDATED",
            "detail": (
                "the venue registries are read when the paper worker starts; "
                "no symbol has been accepted or rejected yet"
            ),
            "validated_at": None,
            "requested": [],
            "accepted": [],
            "rejected": [],
        }
        self._worker: PaperTradingWorker | None = None
        self._lock = threading.Lock()

    # -- provider-scoped durable state ----------------------------------

    @property
    def _provider_state_root(self) -> Path:
        """Directory for state owned by the selected execution venue.

        Alpaca retains the original root-level paths for backward compatibility.
        Bybit Demo gets a fresh namespace; no Alpaca state is copied or migrated.
        """
        if self.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO:
            return self.root / "execution_state" / "bybit_demo"
        return self.root

    def _worker_state_path(self) -> Path:
        """The selected venue's durable broker/worker state file."""
        return self._provider_state_root / "worker_state.json"

    def _source_state_path(self, filename: str) -> Path:
        """Decision-source memory scoped to the selected execution venue."""
        return self._provider_state_root / filename

    # -- worker construction ---------------------------------------------

    def _load_experiment(self):
        """Load the persisted experiment with its price source restored.

        The source is rebuilt by the runner's shared helper, which handles both
        cases honestly: a frame-backed source resumes on the *same* persisted
        bars (no second vendor call), and the frozen dataset reloads from disk
        with its recorded window checked. A source that can persist neither is
        refused rather than silently swapped for another.

        Loading without a source fails for the frozen dataset — it carries no
        persisted bars — which would make the paper status an error even though
        the experiment is perfectly resumable.
        """
        from ..__main__ import _restored_source
        from ..experiment import PaperExperiment

        return PaperExperiment.load(self.root, source=_restored_source(self.root), inference=None)

    def _worker_config(self) -> PaperTradingConfig:
        """The config the worker runs with, from the persisted experiment config.

        Shared by :meth:`_build_worker` and the read-only :meth:`status`, so the
        status view reports the same limits the worker would enforce without
        building — and therefore without freezing — a worker.
        """
        experiment = self._load_experiment()
        return PaperTradingConfig(
            strategy_id=experiment.config.strategy_id,
            order_notional=self.order_notional,
            whole_shares=self.whole_shares,
            state_path=self._worker_state_path(),
            market_data_provider=self.market_data_provider,
            market_data_max_age_seconds=self.market_data_max_age_seconds,
            decision_source=self.decision_source_kind,
            # The same asset class the worker itself runs with, so the status
            # view reports the settings the worker enforces. Alpaca's crypto
            # orders take only 'gtc'/'ioc'; the equity path keeps 'day'.
            crypto=self.decision_source_kind == DECISION_SOURCE_CRYPTO,
            # The execution broker the worker places orders with. The status
            # view and the worker must agree on it, or the view reports
            # settings the worker does not enforce.
            execution_provider=self.execution_provider,
        )

    def _reload_experiment(self):
        """Re-read the experiment state from disk, for the decision source.

        Read-only: it loads the persisted experiment (no inference, so nothing is
        decided) and returns it, so the source sees decisions another writer has
        recorded since the worker was built. Loading never mutates the experiment.
        """
        return self._load_experiment()

    def _build_worker(self) -> PaperTradingWorker:
        experiment = self._load_experiment()

        # Build symbol mappings for crypto TJR path
        # Market data comes from Binance (USDT-quoted), execution on Alpaca (USD-quoted)
        symbol_mappings: tuple[SymbolMapping, ...] = ()
        execution_price_seam = None
        if self.decision_source_kind == DECISION_SOURCE_CRYPTO:
            # For crypto paper trading, BOTH venue registries are required.
            # Missing credentials, network errors, malformed responses, and an
            # unavailable registry must fail closed - never interpret an unavailable
            # registry as proof that a symbol is tradable.

            # Build Binance instrument registry from exchange info
            from ..crypto_intraday_data import BinancePublicClient
            try:
                binance_client = BinancePublicClient()
                binance_registry = binance_client.exchange_info()
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                self.symbol_validation = {
                    "state": "MARKET_DATA_REGISTRY_UNAVAILABLE",
                    "detail": (
                        "the market-data venue's instrument registry could not be "
                        f"read: {type(exc).__name__}: {exc}"
                    ),
                    "validated_at": datetime.now(UTC).isoformat(),
                    "requested": list(experiment.config.symbols),
                    "accepted": [],
                    "rejected": [],
                }
                raise

            # The execution venue's own registry - REQUIRED for crypto paper
            # execution. It is read from the *selected* venue's demo/paper host
            # through that venue's endpoint gate, so a live host can never be
            # reached from here. It used to read Alpaca's registry through a
            # freshly built default config, so a Bybit selection still consulted
            # Alpaca's listing and accepted pairs Alpaca alone served.
            if self.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO:
                from ..crypto_venue import read_bybit_execution_registry

                read_execution_registry = read_bybit_execution_registry
                execution_base_url = _endpoint_for_provider(self.execution_provider)
            else:
                from ..crypto_venue import read_alpaca_execution_registry

                read_execution_registry = read_alpaca_execution_registry
                execution_base_url = _endpoint_for_provider(self.execution_provider)

            try:
                execution_registry = read_execution_registry(
                    base_url=execution_base_url
                )
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                self.symbol_validation = {
                    "state": "EXECUTION_REGISTRY_UNAVAILABLE",
                    "detail": (
                        "the execution venue's instrument registry could not be "
                        f"read: {type(exc).__name__}: {exc}"
                    ),
                    "validated_at": datetime.now(UTC).isoformat(),
                    "requested": list(experiment.config.symbols),
                    "accepted": [],
                    "rejected": [],
                }
                raise

            # Validate every experiment symbol against BOTH venue registries. A
            # symbol only one venue serves is rejected with that venue's reason and
            # is never substituted for a different pair.
            from ..crypto_symbol_config import resolve_symbol_configuration

            configuration = resolve_symbol_configuration(
                list(experiment.config.symbols),
                market_data_registry=binance_registry,
                execution_registry=execution_registry,
            )
            if configuration.rejected:
                import logging
                logger = logging.getLogger(__name__)
                for rejection in configuration.rejected:
                    logger.warning(
                        "Symbol %s rejected for crypto paper trading: %s",
                        rejection.symbol,
                        rejection.reason,
                    )

            symbol_mappings = tuple(configuration.mappings)
            self.symbol_configuration = configuration
            self.symbol_validation = {
                "state": "VALIDATED",
                "detail": (
                    f"{len(configuration.accepted)} accepted, "
                    f"{len(configuration.rejected)} rejected"
                ),
                "validated_at": datetime.now(UTC).isoformat(),
                "requested": list(configuration.requested),
                "accepted": list(configuration.accepted),
                "rejected": [r.to_dict() for r in configuration.rejected],
            }
            if not symbol_mappings:
                raise RuntimeError(
                    "No valid symbol mappings for crypto paper trading. "
                    "All configured symbols were rejected. Check the "
                    f"{self.execution_provider} credentials and ensure symbols are "
                    "tradable on both the market-data venue and the execution venue."
                )

            # The execution-side price seam. The order must be sized from a price
            # of the pair that will actually trade, read from the *execution*
            # venue itself. On Alpaca the strategy analyses Binance's BTC/USDT and
            # the order is placed on Alpaca's BTC/USD; on Bybit both sides carry
            # the same BASE/USDT pair. The seam follows the selected venue.
            if self.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO:
                from ..broker import BybitDemoExecutionPrice

                execution_price_seam = BybitDemoExecutionPrice(
                    [m.execution_symbol for m in symbol_mappings],
                    max_age_seconds=self.market_data_max_age_seconds,
                )
            else:
                from ..broker import AlpacaCryptoExecutionPrice

                execution_price_seam = AlpacaCryptoExecutionPrice(
                    [m.execution_symbol for m in symbol_mappings],
                    max_age_seconds=self.market_data_max_age_seconds,
                )

        # Build the decision source from the persisted config, so the worker acts
        # on the same decisions the experiment recorded.
        # The crypto decision source takes its own symbol list — the accepted
        # symbols only — so a rejected symbol never reaches the TJR chain or
        # the execution path.
        config = PaperTradingConfig(
            strategy_id=experiment.config.strategy_id,
            order_notional=self.order_notional,
            whole_shares=self.whole_shares,
            state_path=self._worker_state_path(),
            market_data_provider=self.market_data_provider,
            market_data_max_age_seconds=self.market_data_max_age_seconds,
            decision_source=self.decision_source_kind,
            symbol_mappings=symbol_mappings,
            execution_price_seam=execution_price_seam,
            max_price_deviation=self.max_price_deviation,
            # Alpaca's crypto orders accept only 'gtc' and 'ioc'; the equity
            # vocabulary's 'day' is refused with HTTP 422. The crypto branch
            # above is the only wiring that trades crypto, so this flag is set
            # there and only there.
            crypto=self.decision_source_kind == DECISION_SOURCE_CRYPTO,
            # The execution broker the worker places orders with. This MUST be
            # the provider the operator selected: the config resolves the
            # endpoint from it (a Bybit selection gets Bybit's demo host, never
            # Alpaca's) and the worker builds the broker from it. Omitting it
            # left the config on its Alpaca default, so a Bybit selection
            # silently traded on Alpaca Paper while the start log named Bybit.
            execution_provider=self.execution_provider,
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
            "state_path": self._source_state_path("current_source_state.json"),
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
            "state_path": self._source_state_path("tjr_source_state.json"),
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

        def _execution_price(symbol: str) -> float:
            """The execution venue's price for the pair the order will trade.

            The crypto path analyses Binance's ``XRP/USDT`` and trades Alpaca's
            ``XRP/USD``. The account is funded in USD and every risk cap is
            denominated in that currency, so the risk engine must size the entry
            at the **execution** venue's price. Sizing it at the market-data
            venue's price sized a USD order in USDT terms.

            The canonical symbol is translated to its execution pair through the
            worker's own validated symbol map — the same map the worker uses to
            place the order — so the price is read for the instrument the order
            will actually be placed on, and an unmapped symbol has no price.
            """
            worker = holder.get("worker")
            if worker is None:
                return 0.0
            symbol_map = getattr(worker, "_symbol_map", None) or {}
            mapping = symbol_map.get(symbol)
            if mapping is None:
                return 0.0
            seam = getattr(worker.config, "execution_price_seam", None)
            if seam is None:
                return 0.0
            try:
                return float(seam.price(mapping.execution_symbol))
            except Exception:  # noqa: BLE001 - an unpriceable entry is no decision
                return 0.0

        # Use ONLY accepted symbols from the validated configuration.
        # Rejected symbols must not reach the TJR decision source.
        crypto_symbols = tuple(
            self.symbol_configuration.accepted
            if self.symbol_configuration is not None
            else experiment.config.symbols
        )

        kwargs: dict[str, Any] = {
            "symbols": crypto_symbols,
            "data": data,
            "market_hours": market_hours,
            "order_notional": self.order_notional,
            "strategy_name": experiment.config.strategy_id,
            "book_provider": _book,
            "price_provider": _price,
            "execution_price_provider": _execution_price,
            "is_consumed": lambda decision_id: holder["worker"].has_order_for(decision_id),
            "state_path": self._source_state_path("crypto_source_state.json"),
            # The same settings the crypto backtest measured. Without this the
            # live path would use the equity 15:30 ET entry cut and stop taking
            # entries eight hours before the session ends, so the backtest's
            # numbers would describe a system that does not exist.
            "settings": crypto_tjr_settings(),
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
        # For crypto path, use ONLY accepted symbols from validated configuration.
        if self.decision_source_kind == DECISION_SOURCE_CRYPTO:
            symbols = tuple(
                self.symbol_configuration.accepted
                if self.symbol_configuration is not None
                else experiment.config.symbols
            )
        else:
            symbols = tuple(experiment.config.symbols)
        return self._market_data_factory(symbols)

    def _ensure_worker(self) -> PaperTradingWorker:
        if self._worker is None:
            self._worker = self._build_worker()
        return self._worker

    # -- control ----------------------------------------------------------

    def strategy_evaluations(self) -> dict[str, Any]:
        """The TJR evaluation per symbol, from the decision source's own state.

        Read from the source's persisted records, so the dashboard's strategy
        section shows what the chain actually concluded — the stage it reached,
        the reason it stopped and the levels it derived — without building the
        worker and without evaluating the chain a second time.

        The evaluation is **not** re-run here: a second evaluation would fetch a
        second window and could disagree with the one the worker acted on. The
        persisted record is the chain's own answer, so it is the only honest
        source for the view.

        Returns ``{}`` when the source has not evaluated anything yet, which the
        view reports as "no evaluation" rather than as a no-signal with a reason
        invented for it.
        """
        state_path = self._source_state_path(self._source_state_filename())
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        records = payload.get("records")
        if not isinstance(records, dict):
            return {}
        out: dict[str, Any] = {}
        for symbol, record in records.items():
            if isinstance(record, dict):
                out[str(symbol)] = _strategy_row_from_record(record)
        return out

    def _source_state_filename(self) -> str:
        """The decision source's own state file, per the configured source kind.

        Each source keeps its memory in its own file so it is not a second writer
        to the experiment directory. The view reads the file of the source the
        worker is actually configured to use, so it cannot show another source's
        evaluation.
        """
        if self.decision_source_kind == DECISION_SOURCE_CRYPTO:
            return "crypto_source_state.json"
        if self.decision_source_kind == DECISION_SOURCE_TJR:
            return "tjr_source_state.json"
        if self.decision_source_kind == DECISION_SOURCE_CURRENT_MARKET:
            return "current_source_state.json"
        return "session_state.json"

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
            # The banner and mode name the broker this controller was
            # configured with. They were hardcoded to Alpaca, so a Bybit
            # selection displayed "ALPACA PAPER — NOT LIVE" — a label that
            # contradicts the endpoint the worker actually trades on.
            "banner": banner_for_provider(self.execution_provider),
            "enabled": self.enabled,
            "mode": mode_for_provider(self.execution_provider),
            "live": False,
            # The configured provider and the endpoint that provider resolves
            # to. Both are secret-free: a URL and a label.
            "execution_provider": self.execution_provider,
            "endpoint": _endpoint_for_provider(self.execution_provider),
            # The symbol validation state, reported whether or not a worker was
            # ever built. It says whether the venues' own listings have been
            # read and which symbols they refused — an explicit "not validated
            # yet" rather than a silent absence.
            "symbol_validation": dict(self.symbol_validation),
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
        state = WorkerStateStore(self._worker_state_path()).load()
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
