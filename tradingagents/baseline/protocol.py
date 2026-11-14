"""The official baseline protocol: one frozen, versioned, reproducible spec.

Every value that shapes a baseline run lives here, and nothing that shapes a run
lives anywhere else. The protocol is a frozen dataclass so a run cannot be nudged
after the fact, and it carries its own version so an artifact records which spec
produced it.

Two kinds of split are defined, and they answer different questions:

- **Development / Validation / Test** (a single chronological partition) is the
  headline protocol. The strategy's *form* is fixed before the run (see
  ``reference.py``); the development period is what a human may look at while
  choosing that form, validation is the first honest read, and test is opened
  once. This is the split the published headline comes from.
- **Walk-forward folds** roll a train/validate/test window across the whole
  calendar, and are the stronger evidence: the test windows are disjoint and
  contiguous, and the selection runs on validation only. Both are reported.

The periods below were chosen before any result was seen. The test window is a
contiguous block at the end of the data, and it is not touched to select
anything: there is exactly one candidate strategy, so there is no selection to
make on the test window.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from tradingagents.execution import CostModel
from tradingagents.risk import RiskLimits

# Bump when anything that changes a result changes. An artifact records this so a
# comparison across versions is explicit rather than silent.
PROTOCOL_VERSION = "1.0.0"

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class BaselineProtocol:
    """The frozen specification of the official baseline run."""

    version: str = PROTOCOL_VERSION

    # --- Universe and data --------------------------------------------------
    universe: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD")
    benchmark: str = "SPY"
    """The instrument the strategy is compared against, buy-and-hold."""
    start: str = "2015-01-02"
    end: str = "2024-12-30"
    timeframe: str = "1 day"
    """Daily bars. The feed is daily; there is no intraday path."""

    # --- Capital and sizing -------------------------------------------------
    initial_cash: float = 100_000.0
    currency: str = "USD"
    position_size: float = 0.20
    """Fraction of equity targeted per entry. With five symbols this is a full book."""
    leverage: float = 1.0
    """None is supported: exposure is capped at 1.0 and cash never goes negative."""
    holding_period_days: int = 40
    """Sessions an entry is held before the scheduled exit. A time-based exit,
    fixed a priori, so the baseline does not depend on a tuned exit."""

    # --- Costs --------------------------------------------------------------
    commission_rate: float = 0.0005
    """5 bps of traded notional per fill: a realistic retail-equity commission."""
    half_spread_bps: float = 1.0
    """Half the bid/ask spread, 1 bp, applied against the trade."""
    slippage_bps: float = 2.0
    """2 bps of adverse slippage per fill."""
    max_volume_participation: float = 0.05
    """A market order may take at most 5% of a session's volume."""

    # --- Risk ---------------------------------------------------------------
    max_position_size: float = 0.25
    max_portfolio_exposure: float = 0.95
    max_open_positions: int = 5
    max_daily_loss: float | None = None
    max_drawdown: float | None = None

    # --- Execution assumptions ---------------------------------------------
    order_type: str = "MARKET"
    execute_on_next_open: bool = True
    """The look-ahead guard. A decision on session T fills at T+1's open."""
    use_stops: bool = True

    # --- Timing -------------------------------------------------------------
    signal_timing: str = "close"
    """A signal is computed from bars through session T's close and acted on at T+1's open."""
    rebalance_frequency: str = "daily"
    """The strategy is asked every session; it acts only when its rule fires."""

    # --- Splits -------------------------------------------------------------
    dev_end: str = "2018-12-31"
    validation_start: str = "2019-01-01"
    validation_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2024-12-30"
    wf_train_sessions: int = 504
    """Two years of training, so the reference rule warms up inside a fold."""
    wf_validate_sessions: int = 252
    wf_test_sessions: int = 252

    # --- Determinism --------------------------------------------------------
    seed: int = 20260101
    """Reserved. The reference strategy and the engine are deterministic, so no
    random number is drawn; the seed is recorded so an artifact names one."""

    limits: RiskLimits = field(default=None)  # type: ignore[assignment]
    costs: CostModel = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Build the nested value objects from the scalar fields, so the scalars
        # are the single source and the objects cannot drift from them.
        object.__setattr__(
            self,
            "limits",
            RiskLimits(
                max_position_size=self.max_position_size,
                max_portfolio_exposure=self.max_portfolio_exposure,
                max_open_positions=self.max_open_positions,
                max_daily_loss=self.max_daily_loss,
                max_drawdown=self.max_drawdown,
            ),
        )
        object.__setattr__(
            self,
            "costs",
            CostModel(
                commission_rate=self.commission_rate,
                half_spread_bps=self.half_spread_bps,
                slippage_bps=self.slippage_bps,
            ),
        )
        if not (self.dev_end < self.validation_start <= self.validation_end < self.test_start):
            raise ValueError("the dev/validation/test periods must be ordered and disjoint")
        if self.test_end < self.test_start:
            raise ValueError("the test window ends before it starts")

    def split_periods(self) -> tuple[tuple[str, str, str], ...]:
        """The three chronological periods as ``(label, start, end)``."""
        return (
            ("development", self.start, self.dev_end),
            ("validation", self.validation_start, self.validation_end),
            ("test", self.test_start, self.test_end),
        )

    def dataset_checksums(self) -> dict[str, str]:
        """SHA-256 of each frozen data file, for the artifact's provenance block."""
        sums: dict[str, str] = {}
        for symbol in self.universe:
            path = DATA_DIR / f"{symbol}.csv"
            if path.exists():
                sums[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        return sums

    def to_dict(self) -> dict:
        """A JSON-safe mapping of every field that shapes a run."""
        return {
            "version": self.version,
            "universe": list(self.universe),
            "benchmark": self.benchmark,
            "start": self.start,
            "end": self.end,
            "timeframe": self.timeframe,
            "initial_cash": self.initial_cash,
            "currency": self.currency,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "holding_period_days": self.holding_period_days,
            "costs": self.costs.model_dump(),
            "max_volume_participation": self.max_volume_participation,
            "risk_limits": self.limits.model_dump(),
            "order_type": self.order_type,
            "execute_on_next_open": self.execute_on_next_open,
            "use_stops": self.use_stops,
            "signal_timing": self.signal_timing,
            "rebalance_frequency": self.rebalance_frequency,
            "splits": {
                "development": [self.start, self.dev_end],
                "validation": [self.validation_start, self.validation_end],
                "test": [self.test_start, self.test_end],
            },
            "walk_forward": {
                "train_sessions": self.wf_train_sessions,
                "validate_sessions": self.wf_validate_sessions,
                "test_sessions": self.wf_test_sessions,
            },
            "seed": self.seed,
            "dataset_checksums": self.dataset_checksums(),
        }


DEFAULT_PROTOCOL = BaselineProtocol()
