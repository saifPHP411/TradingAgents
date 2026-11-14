"""A read-only window onto a :class:`~tradingagents.portfolio.ledger.Ledger`.

The strategy that a backtest runs is given the book for context — how much cash
is free, what is held — and must not be able to move it. Orders are the engine's
business. Handing over the live :class:`Ledger` broke that guarantee: a strategy
could write ``context.ledger.cash = 10_000_000`` or call ``apply_fill`` directly
and fabricate a result no order ever produced.

:class:`LedgerView` closes that surface. It exposes the read accessors and
nothing else, and every position it returns is a copy, so mutating what the
strategy receives cannot reach the real book. It is a view, not a snapshot: it
reads the live ledger, so the figures the strategy sees are the current ones.

This is not a security boundary — Python has no private attributes, and a
determined caller can still reach the wrapped object. It removes the accidental
and the obvious path, which is what the documented guarantee promises.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.portfolio.ledger import Holding, Ledger, LedgerSnapshot


class LedgerView:
    """Read-only access to a ledger's figures, with no way to change them."""

    __slots__ = ("__ledger",)

    def __init__(self, ledger: Ledger) -> None:
        object.__setattr__(self, "_LedgerView__ledger", ledger)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "a LedgerView is read-only; a strategy may read the book but not move it"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            "a LedgerView is read-only; a strategy may read the book but not move it"
        )

    # --- Aggregates ---------------------------------------------------------

    @property
    def initial_cash(self) -> float:
        return self.__ledger.initial_cash

    @property
    def currency(self) -> str:
        return self.__ledger.currency

    @property
    def cash(self) -> float:
        return self.__ledger.cash

    @property
    def market_value(self) -> float:
        return self.__ledger.market_value

    @property
    def equity(self) -> float:
        return self.__ledger.equity

    @property
    def total_exposure(self) -> float:
        return self.__ledger.total_exposure

    @property
    def realized_pnl(self) -> float:
        return self.__ledger.realized_pnl

    @property
    def net_realized_pnl(self) -> float:
        return self.__ledger.net_realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self.__ledger.unrealized_pnl

    @property
    def total_fees(self) -> float:
        return self.__ledger.total_fees

    @property
    def total_pnl(self) -> float:
        return self.__ledger.total_pnl

    @property
    def available_cash(self) -> float:
        """Cash free to spend. Equals cash: this book has no margin."""
        return self.__ledger.cash

    @property
    def buying_power(self) -> float:
        return self.__ledger.cash

    # --- Positions ----------------------------------------------------------

    def position_quantity(self, symbol: str) -> float:
        return self.__ledger.position_quantity(symbol)

    def average_entry_price(self, symbol: str) -> float:
        return self.__ledger.average_entry_price(symbol)

    def holding(self, symbol: str) -> Holding:
        """A copy of the holding, so mutating it cannot reach the book."""
        return replace(self.__ledger.holding(symbol))

    def open_positions(self) -> dict[str, Holding]:
        """Copies of the open holdings, so mutating them cannot reach the book."""
        return {s: replace(h) for s, h in self.__ledger.open_positions().items()}

    @property
    def holdings(self) -> dict[str, Holding]:
        """Copies of every holding, open or closed."""
        return {s: replace(h) for s, h in self.__ledger.holdings.items()}

    @property
    def events(self) -> tuple:
        """The event log so far, as an immutable tuple."""
        return tuple(self.__ledger.events)

    def snapshot(self, timestamp: datetime | None = None) -> LedgerSnapshot:
        return self.__ledger.snapshot(timestamp)
