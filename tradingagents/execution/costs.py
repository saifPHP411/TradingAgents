"""Realistic, configurable execution costs.

A backtest without costs measures a strategy nobody can trade. These are the
three costs a simulated fill must carry, and they are applied *in the ledger*
rather than reported alongside it, so a "gross" result is not available by
accident:

- **Commission**: a per-fill cost, either a flat fee or a per-unit rate, with a
  configurable minimum.
- **Spread**: the difference between the quote mid and the price actually paid.
  A buy lifts the offer, a sell hits the bid, so the spread is always a cost and
  never a rebate.
- **Slippage**: the adverse move between the decision price and the fill price.
  Modelled as a fraction of the mid, scaled by participation when volume is
  known, so a large order in a thin name is penalised more than a small one.

Fees are deterministic given the inputs: no randomness anywhere, so the same
simulation replays to the same figures. A stochastic slippage model would make
the ledger unreproducible, which defeats the point of testing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Side(StrEnum):
    """Which way a fill moves the market, and therefore which way it pays the spread."""

    BUY = "BUY"
    SELL = "SELL"


class CostModel(BaseModel):
    """The configurable cost schedule for a simulation.

    Defaults are zero for every rate, so a run that does not opt into costs is
    not silently charged a fee it did not configure. A simulation that wants
    realistic costs sets them explicitly and gets the same numbers every time.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    commission_flat: float = Field(
        default=0.0, ge=0.0, description="Flat fee per fill, in the account currency"
    )
    commission_per_unit: float = Field(
        default=0.0, ge=0.0, description="Fee per unit traded, in the account currency"
    )
    commission_min: float = Field(
        default=0.0, ge=0.0, description="Minimum commission per fill, applied to whichever rate is used"
    )
    commission_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Commission as a fraction of the traded notional, e.g. 0.0005 for 5 bps",
    )

    half_spread_bps: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Half the bid/ask spread in basis points. Applied against the trade, "
            "so a buy pays the offer and a sell receives the bid."
        ),
    )
    slippage_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Base adverse price move in basis points, before any volume scaling",
    )
    slippage_volume_factor: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Extra basis points of slippage per 1% of average daily volume the order "
            "represents. Zero disables participation scaling."
        ),
    )
    max_slippage_bps: float = Field(
        default=500.0,
        ge=0.0,
        description="Ceiling on total slippage, so a degenerate input cannot price a fill at zero",
    )

    maker_fee_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Maker rebate/fee in basis points, for limit orders that add liquidity",
    )
    taker_fee_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Taker fee in basis points, for market orders that remove liquidity",
    )

    @model_validator(mode="after")
    def _commission_min_is_reachable(self) -> CostModel:
        """A minimum above every possible commission would make the rate meaningless."""
        if self.commission_min > 0 and not (
            self.commission_flat or self.commission_per_unit or self.commission_rate
        ):
            raise ValueError(
                "commission_min is set but no commission rate is configured; "
                "set commission_flat, commission_per_unit or commission_rate"
            )
        return self

    def commission(self, *, quantity: float, price: float) -> float:
        """The commission for a fill of ``abs(quantity)`` units at ``price``.

        The three rates are additive (flat + per-unit + notional), which lets a
        schedule combine a ticket fee with a bps rate, as real brokers do. The
        minimum applies to the sum.
        """
        units = abs(quantity)
        notional = units * price
        total = self.commission_flat + self.commission_per_unit * units + self.commission_rate * notional
        return max(total, self.commission_min) if units else 0.0

    def spread_cost(self, *, quantity: float, price: float) -> float:
        """The half-spread paid on a fill, in the account currency.

        Always a cost: the sign of the adverse move is handled by
        :meth:`fill_price`, so this is the magnitude of what the spread took.
        """
        return abs(quantity) * price * (self.half_spread_bps / 10_000.0)

    def slippage_bps_for(self, *, quantity: float, price: float, average_daily_volume: float | None) -> float:
        """Total adverse slippage in basis points for this fill.

        Scales with participation when a volume is known: an order that is 5% of
        a day's volume pays ``5 * slippage_volume_factor`` extra bps. Without a
        volume there is nothing to scale against, so only the base rate applies.
        """
        bps = self.slippage_bps
        if average_daily_volume and average_daily_volume > 0:
            participation_pct = abs(quantity) / average_daily_volume * 100.0
            bps += participation_pct * self.slippage_volume_factor
        return min(bps, self.max_slippage_bps)

    def fill_price(
        self,
        *,
        side: Side,
        reference_price: float,
        quantity: float,
        average_daily_volume: float | None = None,
    ) -> tuple[float, float, float]:
        """The price a fill gets, plus the spread and slippage it paid.

        ``reference_price`` is the quote mid the decision was made against. The
        return is ``(fill_price, spread_cost, slippage_cost)``; the fill price is
        the mid moved adversely by half the spread and by slippage, so a buy
        always fills at or above the mid and a sell at or below it.
        """
        if reference_price <= 0:
            raise ValueError(f"reference_price must be positive, got {reference_price}")
        if not math.isfinite(reference_price) or not math.isfinite(quantity):
            raise ValueError(
                f"reference_price and quantity must be finite, got {reference_price!r} and {quantity!r}"
            )
        adverse_bps = self.half_spread_bps + self.slippage_bps_for(
            quantity=quantity, price=reference_price, average_daily_volume=average_daily_volume
        )
        direction = 1.0 if side is Side.BUY else -1.0
        fill = reference_price * (1.0 + direction * adverse_bps / 10_000.0)
        if not math.isfinite(fill) or fill <= 0:
            raise ValueError(
                f"costs priced {side.value} fill at {fill}; reduce the configured spread/slippage"
            )
        spread = self.spread_cost(quantity=quantity, price=reference_price)
        slippage = abs(quantity) * reference_price * (
            self.slippage_bps_for(quantity=quantity, price=reference_price,
                                  average_daily_volume=average_daily_volume) / 10_000.0
        )
        return fill, spread, slippage


# A zero-cost schedule, for tests and for a run that only wants the ledger's
# arithmetic without pretending the market charged anything.
NO_COSTS = CostModel()


@dataclass(frozen=True)
class CostBreakdown:
    """What one fill cost, itemised, so the report can show where money went."""

    commission: float
    spread: float
    slippage: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage
