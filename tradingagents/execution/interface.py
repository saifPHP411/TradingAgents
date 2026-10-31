"""The execution interface, and the two modes that implement it.

The purpose of this module is to make an accidental live order *impossible*
rather than merely discouraged. Three mechanisms do that:

1. ``ExecutionMode`` has exactly two members, ``BACKTEST`` and ``PAPER``. There
   is no ``LIVE`` to select, and ``resolve_mode`` refuses any string that is not
   one of the two, so no configuration value and no environment variable can
   produce a live path.
2. ``ExecutionEngine`` is a ``Protocol``. It has no network dependency, and the
   simulator that satisfies it takes prices as arguments rather than fetching
   them, so there is nothing for a live implementation to hook into without
   deliberately writing a new class.
3. ``assert_simulation_only`` is called by the simulator's constructor, so even
   a subclass cannot be constructed in a mode that is not simulation.

A paper mode exists as a distinct mode, not as a second code path: it uses the
same simulator, with the same fills and the same costs. The difference is that a
paper run is expected to be driven by a live clock and live quotes supplied by
its caller, whereas a backtest is driven by historical bars. That is a
difference in the caller, not in the engine, which is exactly the point — the
accounting cannot diverge between "pretend" and "real" fills.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from tradingagents.execution.orders import Order


class ExecutionMode(StrEnum):
    """The only two ways this system may execute. There is deliberately no LIVE."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"


class LiveTradingNotSupportedError(RuntimeError):
    """Raised when something asks for a live execution mode.

    This is a hard error, not a warning: the system is a research and simulation
    tool, and a mode that could reach a venue must fail loudly at the point of
    the request rather than degrade to a simulation silently.
    """


# Names that a user might reasonably try, all of which must fail. Kept explicit
# rather than matching a pattern, so the error message can name what was asked.
_LIVE_ALIASES = frozenset({
    "live", "real", "production", "prod", "broker", "exchange", "mainnet",
})


def resolve_mode(value: object) -> ExecutionMode:
    """Coerce ``value`` to a simulation mode, refusing anything live.

    Accepts an :class:`ExecutionMode`, or a string naming one. A string naming a
    live mode raises :class:`LiveTradingNotSupportedError`; any other string
    raises ``ValueError``. ``None`` is not accepted: defaulting an unspecified
    mode would hide which mode ran, and the default would be wrong half the time.
    """
    if isinstance(value, ExecutionMode):
        return value
    text = str(value).strip().lower()
    if text in _LIVE_ALIASES:
        raise LiveTradingNotSupportedError(
            f"execution mode {value!r} is not supported: TradingAgents is a research and "
            "simulation framework and never places live orders. Use BACKTEST or PAPER."
        )
    try:
        return ExecutionMode(text.upper())
    except ValueError as exc:
        raise ValueError(
            f"execution mode must be BACKTEST or PAPER, got {value!r}"
        ) from exc


def assert_simulation_only(mode: ExecutionMode) -> None:
    """Fail unless ``mode`` is a simulation mode. The last line of defence."""
    if mode not in (ExecutionMode.BACKTEST, ExecutionMode.PAPER):
        raise LiveTradingNotSupportedError(f"{mode!r} is not a simulation mode")


@runtime_checkable
class ExecutionEngine(Protocol):
    """What the ledger's driver requires of an execution engine.

    Deliberately tiny, and free of any market-data or network concern: the
    engine is handed an order and the price context it should fill against, and
    it answers with what happened. A paper engine and a backtest engine differ
    only in where that context comes from.
    """

    mode: ExecutionMode

    def submit(self, order: Order, *, market: MarketContext) -> Order:  # pragma: no cover - protocol
        """Attempt to execute ``order`` and return it with its outcome recorded."""
        ...


class MarketContext:
    """The price and liquidity facts an order is filled against.

    Passed explicitly rather than fetched, so the engine has no way to reach a
    vendor and a test can supply a bar by hand. ``is_open`` is required rather
    than inferred from a missing price, because "the market is closed" and "I
    have no price" are different answers and must be reported differently.

    ``low`` and ``high`` are the bar's intraday range when the caller has it. A
    LIMIT order needs them to decide whether the market traded to its limit: a
    single reference price can only say "the market is at X", not "the market
    passed through X". They are optional, and a context without them falls back
    to the reference price, so a caller that has only one price is not forced to
    invent a range. A range does not prove the order would have been filled in
    the real queue; it says the price traded, which is the most a daily bar can
    honestly say.
    """

    __slots__ = (
        "symbol", "price", "timestamp", "is_open", "average_daily_volume", "low", "high",
    )

    def __init__(
        self,
        *,
        symbol: str,
        price: float | None,
        timestamp,
        is_open: bool = True,
        average_daily_volume: float | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp
        self.is_open = is_open
        self.average_daily_volume = average_daily_volume
        self.low = low
        self.high = high
