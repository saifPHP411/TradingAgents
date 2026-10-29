"""A start/stop-controllable paper-trading worker that places Alpaca **paper** orders.

This is the one place in the project that turns a *strategy decision* into a
broker order. It is deliberately narrow, and it is the only module that passes
``allow_orders=True`` to the broker. Nothing here runs on its own: the worker is
``STOPPED`` until an authenticated caller explicitly starts it, and after a
restart it is ``STOPPED`` again, never ``RUNNING``.

The path, stated once:

    decision source -> risk checks -> AlpacaPaperBroker -> Alpaca Paper API
                    -> order status / fills -> durable worker state

Design rules, each of which a test pins:

- **Paper only.** The broker's own endpoint gate is the single chokepoint: only
  ``https://paper-api.alpaca.markets`` is accepted and every live host is refused
  by :func:`~.alpaca_paper.assert_paper_endpoint`. This module never constructs a
  broker with any other base URL, and there is no setting that turns paper into
  live.
- **No order on start.** ``start()`` runs preflight checks and sets the state to
  ``RUNNING``; it places no order. An order is placed only when the decision
  source yields a decision *and* the local risk checks pass.
- **Idempotent.** Each decision maps to a deterministic ``client_order_id``
  derived from the decision id, symbol and side. A re-run, a timeout, or a
  restart reconciles by that id rather than submitting a second order.
- **Fail closed.** A missing or unconfirmed account, an unavailable market-data
  source, a buying-power shortfall, a breached cap, or a would-be short all refuse
  the order *before* it reaches the broker — the broker is never relied on to
  reject it.
- **Graceful stop.** ``stop()`` refuses further cycles and stops the loop; it does
  not close positions and does not cancel open orders. An order already in flight
  is reported with its status, not abandoned silently.

Secrets never appear here: the broker reads ``ALPACA_PAPER_API_KEY`` /
``ALPACA_PAPER_API_SECRET`` from the environment itself, and this module never
reads, logs, or persists a credential.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .alpaca_paper import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
    BrokerOrder,
    BrokerRejectedError,
    PaperRiskLimits,
    PaperRiskRejected,
    assert_paper_endpoint,
)

WORKER_STATE_SCHEMA_VERSION = 1
"""The on-disk worker state format. Bumped when a reader must know the shape changed."""

# -- the closed state vocabulary -------------------------------------------------

STATUS_STOPPED = "STOPPED"
STATUS_STARTING = "STARTING"
STATUS_RUNNING = "RUNNING"
STATUS_STOPPING = "STOPPING"
STATUS_ERROR = "ERROR"

WORKER_STATUSES = (
    STATUS_STOPPED,
    STATUS_STARTING,
    STATUS_RUNNING,
    STATUS_STOPPING,
    STATUS_ERROR,
)

BANNER = "ALPACA PAPER — NOT LIVE"
"""Shown wherever the worker's state is surfaced, so no view can imply live trading."""


class WorkerError(RuntimeError):
    """The worker was driven in a way it cannot support."""


class PreflightError(WorkerError):
    """A start-time check failed, so the worker must not run."""

    def __init__(self, check: str, detail: str) -> None:
        self.check = check
        self.detail = detail
        super().__init__(f"preflight check {check!r} failed: {detail}")


class MarketDataError(WorkerError):
    """The market-data source could not supply a usable price."""


# ---------------------------------------------------------------------------
# Decision and market-data seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingDecision:
    """A strategy decision the worker may turn into one paper order.

    ``notional`` is the target cash amount to trade; the worker converts it to a
    share quantity from the *current* price, so sizing never depends on a stale
    price. ``quantity`` may be given instead (then ``notional`` is ignored).
    """

    decision_id: str
    symbol: str
    side: str  # "buy" or "sell"
    notional: float | None = None
    quantity: float | None = None
    rationale: str = ""
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise WorkerError("a decision must carry a decision_id")
        if not self.symbol:
            raise WorkerError("a decision must carry a symbol")
        if self.side.lower() not in ("buy", "sell"):
            raise WorkerError(f"side must be buy or sell, got {self.side!r}")
        if self.notional is None and self.quantity is None:
            raise WorkerError("a decision must carry a notional or a quantity")
        for value in (self.notional, self.quantity, self.limit_price):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise WorkerError(
                    f"decision figures must be finite and positive, got {value!r}"
                )

    def client_order_id(self, *, strategy_id: str) -> str:
        """A deterministic, broker-safe id for this decision.

        Deterministic so a re-run, a timeout, or a restart produces the *same* id
        and the broker's own idempotency (and this worker's reconcile path) refuse
        a duplicate rather than placing a second order.
        """
        material = f"{strategy_id}:{self.decision_id}:{self.symbol.upper()}:{self.side.lower()}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
        return f"pt-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "limit_price": self.limit_price,
        }


class DecisionSource(Protocol):
    """Yields the next strategy decision, or ``None`` when there is nothing to do.

    A source must be *deterministic in its ordering* and must not itself place an
    order. The worker treats a decision it has already processed as done, so a
    source that yields the same decision twice is safe.
    """

    def next_decision(self) -> TradingDecision | None:  # pragma: no cover - protocol
        ...


class MarketData(Protocol):
    """Supplies a current price for a symbol, and reports whether it is usable."""

    def price(self, symbol: str) -> float:  # pragma: no cover - protocol
        ...

    def available(self) -> bool:  # pragma: no cover - protocol
        ...


@dataclass
class StaticMarketData:
    """A fixed price map, for tests and offline demonstration."""

    prices: dict[str, float] = field(default_factory=dict)

    def available(self) -> bool:
        return bool(self.prices)

    def price(self, symbol: str) -> float:
        value = self.prices.get(symbol.upper())
        if value is None:
            raise MarketDataError(f"no market data for {symbol!r}")
        if not math.isfinite(value) or value <= 0:
            raise MarketDataError(f"a usable price is required for {symbol!r}, got {value!r}")
        return float(value)


@dataclass
class SourceMarketData:
    """Prices from a price source's bar for a session — the "current" bar to trade.

    ``source`` is any object with ``bars(symbol, session)`` returning a bar with an
    ``open`` (the same :class:`~tradingagents.paper_experiment.sources.PriceSource`
    the experiment uses). This is the *latest available* bar for the session, which
    is the honest meaning of "current market data" for a daily-bar experiment. A
    live intraday quote provider can implement the same ``price``/``available``
    seam instead.
    """

    source: Any
    session: str

    def available(self) -> bool:
        try:
            return self.source.covers(self.session)
        except Exception:  # noqa: BLE001 - an unusable source is "not available"
            return False

    def price(self, symbol: str) -> float:
        try:
            bar = self.source.bars(symbol.upper(), self.session)
        except Exception as exc:  # noqa: BLE001 - a source failure is surfaced
            raise MarketDataError(f"the price source failed for {symbol!r}: {exc}") from None
        if bar is None:
            raise MarketDataError(f"no bar for {symbol!r} at session {self.session}")
        price = float(bar.open)
        if not math.isfinite(price) or price <= 0:
            raise MarketDataError(f"the bar for {symbol!r} carries no usable price: {price!r}")
        return price


@dataclass
class ListDecisionSource:
    """Yields a fixed, ordered list of decisions, one per call, then ``None``.

    The order is the list's order, so a re-run yields the same decisions in the
    same sequence — which is what makes the worker's per-decision idempotency
    meaningful. The source holds a cursor; :meth:`reset` rewinds it.
    """

    decisions: list[TradingDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = 0

    def next_decision(self) -> TradingDecision | None:
        if self._index >= len(self.decisions):
            return None
        decision = self.decisions[self._index]
        self._index += 1
        return decision

    def reset(self) -> None:
        self._index = 0


@dataclass
class PaperTradingConfig:
    """The worker's configuration. Safe by default; every limit is explicit.

    ``order_notional`` is the target cash size of one order and is capped by
    ``limits.max_order_notional``; the two are checked together, so a large
    ``order_notional`` cannot silently exceed the notional cap.
    """

    strategy_id: str = "alpaca-paper"
    order_notional: float = 25.0
    """Target cash per order. Deliberately small, so a 50 USD paper account is not
    drained by one order."""
    whole_shares: bool = True
    """When ``True``, size to whole shares (Alpaca supports fractional, but whole
    shares keep a tiny account's behaviour obvious)."""
    base_url: str = PAPER_BASE_URL
    limits: PaperRiskLimits = field(default_factory=PaperRiskLimits)
    state_path: str | os.PathLike | None = None
    poll_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not (self.order_notional > 0):
            raise WorkerError("order_notional must be positive")
        if self.order_notional > self.limits.max_order_notional:
            raise WorkerError(
                f"order_notional {self.order_notional} exceeds the notional cap "
                f"{self.limits.max_order_notional}; lower one of them"
            )
        if self.poll_interval_seconds <= 0:
            raise WorkerError("poll_interval_seconds must be positive")
        # The endpoint is validated at configuration time, so a live host is
        # refused before a broker is ever constructed.
        assert_paper_endpoint(self.base_url)


# ---------------------------------------------------------------------------
# Durable worker state
# ---------------------------------------------------------------------------


class WorkerStateError(RuntimeError):
    """The persisted worker state is malformed, truncated, or a newer version."""


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".worker-state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass
class WorkerState:
    """Everything the worker needs to resume without duplicating an order.

    ``status`` is persisted but is always read back as ``STOPPED`` by
    :meth:`PaperTradingWorker.load`: a restart must never resume trading on its
    own. What *is* preserved is the order memory — every ``client_order_id`` the
    worker has seen, its last known broker status, the applied-fill set, the
    positions last read, and the last processed decision id — so a resumed worker
    reconciles instead of re-submitting.
    """

    schema_version: int = WORKER_STATE_SCHEMA_VERSION
    strategy_id: str = "alpaca-paper"
    status: str = STATUS_STOPPED
    last_processed_decision_id: str | None = None
    last_error: str | None = None
    last_cycle_at: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    orders: dict[str, dict] = field(default_factory=dict)
    applied_fills: set[str] = field(default_factory=set)
    positions: dict[str, dict] = field(default_factory=dict)
    positions_at: str | None = None
    account: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "last_processed_decision_id": self.last_processed_decision_id,
            "last_error": self.last_error,
            "last_cycle_at": self.last_cycle_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "orders": {k: dict(v) for k, v in sorted(self.orders.items())},
            "applied_fills": sorted(self.applied_fills),
            "positions": {k: dict(v) for k, v in sorted(self.positions.items())},
            "positions_at": self.positions_at,
            "account": dict(self.account),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> WorkerState:
        if not isinstance(payload, dict):
            raise WorkerStateError("the worker state must be a JSON object")
        version = payload.get("schema_version")
        if version != WORKER_STATE_SCHEMA_VERSION:
            raise WorkerStateError(
                f"unsupported worker state schema_version {version!r}; this build reads "
                f"version {WORKER_STATE_SCHEMA_VERSION}"
            )
        status = payload.get("status", STATUS_STOPPED)
        if status not in WORKER_STATUSES:
            raise WorkerStateError(f"unknown worker status {status!r}")
        orders = payload.get("orders") or {}
        if not isinstance(orders, dict):
            raise WorkerStateError("worker state 'orders' must be a mapping")
        return cls(
            schema_version=version,
            strategy_id=payload.get("strategy_id", "alpaca-paper"),
            status=status,
            last_processed_decision_id=payload.get("last_processed_decision_id"),
            last_error=payload.get("last_error"),
            last_cycle_at=payload.get("last_cycle_at"),
            started_at=payload.get("started_at"),
            stopped_at=payload.get("stopped_at"),
            orders={k: dict(v) for k, v in orders.items()},
            applied_fills=set(payload.get("applied_fills") or []),
            positions={k: dict(v) for k, v in (payload.get("positions") or {}).items()},
            positions_at=payload.get("positions_at"),
            account=dict(payload.get("account") or {}),
        )


class WorkerStateStore:
    """Loads and saves :class:`WorkerState`, atomically and version-checked."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def load(self) -> WorkerState:
        if not self.path.exists():
            return WorkerState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkerStateError(f"worker state {self.path} is not usable: {exc}") from exc
        return WorkerState.from_dict(payload)

    def save(self, state: WorkerState) -> Path:
        _atomic_write(self.path, json.dumps(state.to_dict(), indent=2) + "\n")
        return self.path


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


@dataclass
class CycleResult:
    """What one :meth:`PaperTradingWorker.run_once` produced."""

    ran: bool
    decision_id: str | None = None
    client_order_id: str | None = None
    order: BrokerOrder | None = None
    skipped_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "decision_id": self.decision_id,
            "client_order_id": self.client_order_id,
            "order": self.order.to_dict() if self.order is not None else None,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
        }


class PaperTradingWorker:
    """A start/stop paper-trading worker over one Alpaca paper broker.

    The broker is created lazily by :meth:`preflight` (so construction performs no
    request and needs no credential), through a factory that defaults to the real
    :class:`AlpacaPaperBroker`. A test injects a factory that returns a
    broker with a mocked transport.
    """

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        decision_source: DecisionSource,
        market_data: MarketData,
        broker_factory: Callable[[], AlpacaPaperBroker] | None = None,
        root: str | os.PathLike | None = None,
    ) -> None:
        self.config = config
        self.decision_source = decision_source
        self.market_data = market_data
        self._broker_factory = broker_factory or self._default_broker_factory
        state_path = config.state_path or (
            (Path(root) / "worker_state.json") if root is not None else None
        )
        self._store = WorkerStateStore(state_path) if state_path else None
        self._broker: AlpacaPaperBroker | None = None
        # A restart reads the state back as STOPPED, never RUNNING.
        self.state = self._store.load() if self._store else WorkerState()
        self.state.status = STATUS_STOPPED
        self.state.strategy_id = config.strategy_id
        self.last_cycle: CycleResult | None = None

    # -- construction helpers --------------------------------------------

    def _default_broker_factory(self) -> AlpacaPaperBroker:
        """Build the real paper broker from the environment's credentials.

        ``state_path`` is the same directory as the worker state, so the broker's
        own order memory and the worker's live side by side. Credentials are read
        by the broker from ``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET``;
        this factory never touches them.
        """
        state_path = self._store.path.with_name("broker_state.json") if self._store else None
        return AlpacaPaperBroker(
            base_url=self.config.base_url,
            limits=self.config.limits,
            state_path=state_path,
        )

    def _save(self) -> None:
        if self._store:
            self._store.save(self.state)

    # -- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A secret-free view of the worker, for the dashboard and the CLI."""
        return {
            "banner": BANNER,
            "mode": "ALPACA_PAPER",
            "live": False,
            "status": self.state.status,
            "strategy_id": self.state.strategy_id,
            "endpoint": self.config.base_url,
            "paper_endpoint_only": self.config.base_url == PAPER_BASE_URL,
            "last_processed_decision_id": self.state.last_processed_decision_id,
            "last_error": self.state.last_error,
            "last_cycle_at": self.state.last_cycle_at,
            "started_at": self.state.started_at,
            "stopped_at": self.state.stopped_at,
            "orders": {k: dict(v) for k, v in sorted(self.state.orders.items())},
            "positions": {k: dict(v) for k, v in sorted(self.state.positions.items())},
            "positions_at": self.state.positions_at,
            "account": dict(self.state.account),
            "last_cycle": self.last_cycle.to_dict() if self.last_cycle else None,
            "risk_limits": {
                "max_order_notional": self.config.limits.max_order_notional,
                "max_open_positions": self.config.limits.max_open_positions,
                "max_daily_orders": self.config.limits.max_daily_orders,
                "allow_short": self.config.limits.allow_short,
                "order_notional": self.config.order_notional,
            },
        }

    # -- lifecycle --------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        """Run every start-time check. Returns the checks; raises on the first failure.

        Checks, in order: the broker connects (paper endpoint + credentials), the
        account status is usable, buying power is positive, the market-data source
        is available, and the risk configuration is valid. None of these places an
        order.
        """
        checks: list[dict[str, Any]] = []

        broker = self._broker_factory()
        account = broker.connect()  # raises on a bad endpoint/credential/timeout
        checks.append({"check": "broker_connection", "ok": True, "detail": "connected"})

        if account.status.upper() != "ACTIVE":
            raise PreflightError("account_status", f"account status is {account.status!r}")
        if account.trading_blocked or account.account_blocked:
            raise PreflightError(
                "account_status", f"account restrictions: {', '.join(account.restrictions())}"
            )
        checks.append({"check": "account_status", "ok": True, "detail": account.status})

        if not (account.buying_power > 0):
            raise PreflightError(
                "buying_power", f"buying power is {account.buying_power!r}"
            )
        checks.append(
            {"check": "buying_power", "ok": True, "detail": f"{account.buying_power:.2f}"}
        )

        if not self.market_data.available():
            raise PreflightError("market_data", "the market-data source reports no data")
        checks.append({"check": "market_data", "ok": True, "detail": "available"})

        # Constructing the limits already validated them; re-check the relationship
        # between the order size and the cap so a misconfiguration cannot start.
        if self.config.order_notional > self.config.limits.max_order_notional:
            raise PreflightError(
                "risk_config",
                f"order_notional {self.config.order_notional} exceeds the cap "
                f"{self.config.limits.max_order_notional}",
            )
        checks.append({"check": "risk_config", "ok": True, "detail": "valid"})

        # Read positions so a later sell is checked against a known holding.
        broker.positions(refresh=True)

        self._broker = broker
        self.state.account = {
            "account_id": account.account_id,
            "status": account.status,
            "cash": account.cash,
            "equity": account.equity,
            "buying_power": account.buying_power,
            "currency": account.currency,
            "restrictions": account.restrictions(),
        }
        self._save()
        return {"checks": checks, "account": self.state.account}

    def start(self) -> dict[str, Any]:
        """Run preflight, then mark the worker ``RUNNING``. Places no order.

        Refuses if the worker is already running (one worker per state file), and
        refuses if preflight fails — setting ``ERROR`` with the failing check
        recorded, never ``RUNNING``.
        """
        if self.state.status in (STATUS_RUNNING, STATUS_STARTING):
            raise WorkerError("a worker is already running; stop it before starting another")
        self.state.status = STATUS_STARTING
        self.state.last_error = None
        self._save()
        try:
            result = self.preflight()
        except (BrokerError, PreflightError, WorkerError) as exc:
            self.state.status = STATUS_ERROR
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self._save()
            raise
        self.state.status = STATUS_RUNNING
        self.state.started_at = datetime.now(UTC).isoformat()
        self.state.stopped_at = None
        self._save()
        return {"status": self.state.status, **result}

    def stop(self) -> dict[str, Any]:
        """Stop gracefully: refuse further cycles. Closes no position, cancels no order.

        An order already submitted is left alone (it belongs to the broker now);
        its last known status stays in :meth:`status`, so a caller can see it.
        """
        if self.state.status in (STATUS_STOPPED, STATUS_STOPPING):
            self.state.status = STATUS_STOPPED
            self._save()
            return self.status()
        self.state.status = STATUS_STOPPING
        self._save()
        self.state.status = STATUS_STOPPED
        self.state.stopped_at = datetime.now(UTC).isoformat()
        self._save()
        return self.status()

    # -- one cycle --------------------------------------------------------

    def run_once(self) -> CycleResult:
        """Process at most one decision, placing at most one order.

        Refuses unless ``RUNNING``. A decision already processed is skipped
        (idempotent). A duplicate ``client_order_id`` is reconciled by the broker
        rather than re-submitted. A transient market-data failure is recorded and
        the worker stays ``RUNNING``; the cycle simply does nothing.
        """
        if self.state.status != STATUS_RUNNING:
            raise WorkerError(
                f"the worker is {self.state.status}; it must be RUNNING to process a "
                "decision (call start() first)"
            )
        broker = self._broker
        if broker is None:  # pragma: no cover - start() always sets it
            raise WorkerError("the worker has no broker; call start() first")

        self.state.last_cycle_at = datetime.now(UTC).isoformat()
        decision = self.decision_source.next_decision()
        if decision is None:
            self._save()
            result = CycleResult(ran=False, skipped_reason="no decision is due")
            self.last_cycle = result
            return result

        if decision.decision_id == self.state.last_processed_decision_id:
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                skipped_reason="the decision has already been processed",
            )
            self.last_cycle = result
            return result

        client_order_id = decision.client_order_id(strategy_id=self.config.strategy_id)
        # A decision whose order the worker already knows is not re-submitted.
        if client_order_id in self.state.orders:
            order = self._reconcile(client_order_id)
            self.state.last_processed_decision_id = decision.decision_id
            self._record_order(order)
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                order=order,
                skipped_reason="the order for this decision already exists",
            )
            self.last_cycle = result
            return result

        try:
            reference_price = self.market_data.price(decision.symbol)
        except MarketDataError as exc:
            self.state.last_error = f"MarketDataError: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="market data was unavailable",
            )
            self.last_cycle = result
            return result

        try:
            quantity = self._size(decision, reference_price)
        except WorkerError as exc:
            self.state.last_error = f"WorkerError: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                error=str(exc),
                skipped_reason="the order could not be sized",
            )
            self.last_cycle = result
            return result

        try:
            order = broker.submit_order(
                symbol=decision.symbol,
                quantity=quantity,
                side=decision.side,
                reference_price=reference_price,
                client_order_id=client_order_id,
                order_type="limit" if decision.limit_price else "market",
                allow_orders=True,
                refresh=True,
            )
        except (PaperRiskRejected, BrokerRejectedError, BrokerError) as exc:
            # A local refusal or a broker rejection is recorded, not hidden, and the
            # worker stays RUNNING so the next cycle can act.
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self._save()
            result = CycleResult(
                ran=False,
                decision_id=decision.decision_id,
                client_order_id=client_order_id,
                error=str(exc),
                skipped_reason="the order was refused",
            )
            self.last_cycle = result
            return result

        self.state.last_processed_decision_id = decision.decision_id
        self.state.last_error = None
        self._record_order(order)
        result = CycleResult(
            ran=True,
            decision_id=decision.decision_id,
            client_order_id=client_order_id,
            order=order,
        )
        self.last_cycle = result
        return result

    # -- helpers ----------------------------------------------------------

    def _size(self, decision: TradingDecision, reference_price: float) -> float:
        """Convert a decision into a share quantity, sized from the current price."""
        if decision.quantity is not None:
            return float(decision.quantity)
        assert decision.notional is not None  # __post_init__ guarantees one
        raw = decision.notional / reference_price
        quantity = math.floor(raw) if self.config.whole_shares else round(raw, 6)
        if quantity <= 0:
            raise WorkerError(
                f"the notional {decision.notional} buys no shares at price "
                f"{reference_price}; the order would be empty"
            )
        return float(quantity)

    def _reconcile(self, client_order_id: str) -> BrokerOrder:
        broker = self._broker
        if broker is None:  # pragma: no cover
            raise WorkerError("the worker has no broker")
        return broker._reconcile(client_order_id)

    def _record_order(self, order: BrokerOrder) -> None:
        self.state.orders[order.client_order_id] = {
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "filled_avg_price": order.filled_avg_price,
            "status": order.status,
            "submitted_at": order.submitted_at,
        }
        self._save()

    def sync_positions(self) -> dict[str, dict]:
        """Refresh the broker's positions into the worker state (read-only)."""
        broker = self._broker
        if broker is None:
            raise WorkerError("the worker has no broker; start it first")
        positions = broker.positions(refresh=True)
        self.state.positions = {s: p.to_dict() for s, p in positions.items()}
        self.state.positions_at = datetime.now(UTC).isoformat()
        self._save()
        return dict(self.state.positions)

    @classmethod
    def load(cls, path: str | os.PathLike) -> WorkerState:
        """Load the persisted worker state, always as ``STOPPED``.

        A restart must never resume trading by itself: the state is returned with
        its order memory intact but its status forced to ``STOPPED``.
        """
        state = WorkerStateStore(path).load()
        state.status = STATUS_STOPPED
        return state
