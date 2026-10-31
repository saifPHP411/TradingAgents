"""A paper-only broker seam, kept deliberately apart from the simulator.

The internal paper experiment already has a portfolio and a ledger. Alpaca Paper
Trading has its *own* simulated broker account. Wiring the two together naively
would double-count: one fill would move both the internal ledger and the broker
account, and two equity curves would disagree about the same order.

So the architecture is explicit, not merged:

- the internal simulator (``execution/simulator.py``, ``execution/paper.py``)
  stays the only thing that fills against a bar and writes the internal ledger;
- :class:`~.alpaca_paper.AlpacaPaperBroker` is a **separate, broker-backed** path
  with its **own** account state (``BrokerAccountState``) and its own execution
  records. It never writes the internal ledger, and the internal ledger never
  reads it.

Nothing in this package runs unless a caller constructs a broker and passes
``allow_orders=True``. Selecting a broker is not a live path: the endpoint is
hard-validated to Alpaca's *paper* host, and every live endpoint is refused.
"""

from __future__ import annotations

from .alpaca_paper import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerAccount,
    BrokerError,
    BrokerNotConfiguredError,
    BrokerOrder,
    BrokerPosition,
    BrokerRejectedError,
    LiveEndpointRefusedError,
    PaperRiskLimits,
    PaperRiskRejected,
    assert_paper_endpoint,
    broker_status,
)
from .state import BrokerAccountState, BrokerStateError, BrokerStateStore
from .trading import (
    BANNER,
    STATUS_ERROR,
    STATUS_RUNNING,
    STATUS_STARTING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    WORKER_STATE_SCHEMA_VERSION,
    CycleResult,
    DecisionSource,
    ListDecisionSource,
    MarketData,
    MarketDataError,
    PaperTradingConfig,
    PaperTradingWorker,
    PreflightError,
    SourceMarketData,
    StaticMarketData,
    TradingDecision,
    WorkerError,
    WorkerState,
    WorkerStateError,
    WorkerStateStore,
    worker_status_view,
)

__all__ = [
    "BANNER",
    "PAPER_BASE_URL",
    "STATUS_ERROR",
    "STATUS_RUNNING",
    "STATUS_STARTING",
    "STATUS_STOPPED",
    "STATUS_STOPPING",
    "WORKER_STATE_SCHEMA_VERSION",
    "AlpacaPaperBroker",
    "BrokerAccount",
    "BrokerAccountState",
    "BrokerError",
    "BrokerNotConfiguredError",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerRejectedError",
    "BrokerStateError",
    "BrokerStateStore",
    "CycleResult",
    "DecisionSource",
    "ListDecisionSource",
    "LiveEndpointRefusedError",
    "MarketData",
    "MarketDataError",
    "PaperRiskLimits",
    "PaperRiskRejected",
    "PaperTradingConfig",
    "PaperTradingWorker",
    "PreflightError",
    "SourceMarketData",
    "StaticMarketData",
    "TradingDecision",
    "WorkerError",
    "WorkerState",
    "WorkerStateError",
    "WorkerStateStore",
    "assert_paper_endpoint",
    "broker_status",
    "worker_status_view",
]
