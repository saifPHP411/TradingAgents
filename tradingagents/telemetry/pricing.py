"""Pricing for LLM calls: a table you supply, not a table we guess.

Prices change, and a hardcoded price that has gone stale is worse than no price:
it produces a confident cost figure that is wrong. So this module holds no
built-in rates. A caller supplies the rates it knows, and any call whose model is
not in the table records a ``None`` cost rather than a fabricated one.

That also keeps the module free of anything that could be mistaken for
credentials: a price is a number, and there is nothing here to leak.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model, in US dollars.

    Per million rather than per token because that is how providers quote them,
    and converting at the edge avoids a factor-of-a-million error in the middle
    of a calculation.
    """

    input_per_million: float
    output_per_million: float

    def cost(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_million
            + output_tokens / 1_000_000 * self.output_per_million
        )


class PriceTable:
    """A lookup from ``(provider, model)`` to :class:`ModelPrice`.

    Keys are matched case-insensitively. An unknown model is not an error — the
    call still happened and its tokens are still counted — so :meth:`cost`
    returns ``None`` and the record says the cost is unknown.
    """

    def __init__(self, prices: dict[tuple[str, str], ModelPrice] | None = None) -> None:
        self._prices: dict[tuple[str, str], ModelPrice] = {}
        for (provider, model), price in (prices or {}).items():
            self.set(provider, model, price)

    def set(self, provider: str, model: str, price: ModelPrice) -> None:
        self._prices[self._key(provider, model)] = price

    def get(self, provider: str, model: str) -> ModelPrice | None:
        return self._prices.get(self._key(provider, model))

    def cost(self, *, provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """The cost of a call, or ``None`` when the model's price is not known."""
        price = self.get(provider, model)
        if price is None:
            return None
        return price.cost(input_tokens=input_tokens, output_tokens=output_tokens)

    @staticmethod
    def _key(provider: str, model: str) -> tuple[str, str]:
        return (provider.strip().lower(), model.strip().lower())

    def __contains__(self, item: tuple[str, str]) -> bool:
        return self._key(*item) in self._prices

    def __len__(self) -> int:
        return len(self._prices)


EMPTY_PRICE_TABLE = PriceTable()
"""A table with no prices, so every cost is ``None``. The honest default."""
