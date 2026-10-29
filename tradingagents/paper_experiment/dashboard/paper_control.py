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

from ..broker import (
    BANNER,
    AlpacaPaperBroker,
    PaperTradingConfig,
    PaperTradingWorker,
    SourceMarketData,
    StaticMarketData,
)


class PaperTradingDisabled(RuntimeError):
    """The dashboard was not started with paper trading enabled."""


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
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.order_notional = order_notional
        self.stub = stub
        self._broker_factory = broker_factory
        self._worker_factory = worker_factory or PaperTradingWorker
        self._worker: PaperTradingWorker | None = None
        self._lock = threading.Lock()

    # -- worker construction ---------------------------------------------

    def _build_worker(self) -> PaperTradingWorker:
        from ..experiment import PaperExperiment
        from ..paper_trading import ExperimentDecisionSource

        experiment = PaperExperiment.load(self.root, inference=None)
        # Build the decision source from the persisted config, so the worker acts
        # on the same decisions the experiment recorded.
        config = PaperTradingConfig(
            strategy_id=experiment.config.strategy_id,
            order_notional=self.order_notional,
            state_path=self.root / "worker_state.json",
        )
        source = experiment.source
        sessions = source.sessions()
        market = (
            SourceMarketData(source, sessions[-1]) if sessions else StaticMarketData({})
        )
        decision_source = ExperimentDecisionSource(experiment, order_notional=self.order_notional)
        kwargs: dict[str, Any] = {
            "config": config,
            "decision_source": decision_source,
            "market_data": market,
        }
        if self._broker_factory is not None:
            kwargs["broker_factory"] = self._broker_factory
        return self._worker_factory(**kwargs)

    def _ensure_worker(self) -> PaperTradingWorker:
        if self._worker is None:
            self._worker = self._build_worker()
        return self._worker

    # -- control ----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A secret-free status view. Never raises for a disabled controller."""
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
            try:
                worker = self._ensure_worker()
            except Exception as exc:  # noqa: BLE001 - a build failure is reported
                base.update(
                    {
                        "status": "ERROR",
                        "last_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return base
            base.update(worker.status())
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
