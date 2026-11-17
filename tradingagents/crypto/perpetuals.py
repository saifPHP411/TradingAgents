"""Conservative primitives for linear USDT perpetual research.

This module contains no exchange-order methods. It models only linear USDT-settled
contracts, isolated margin and one-way positions; inverse contracts, cross margin
and hedge mode are intentionally unsupported in this first implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import decimal
import math


MAX_RESEARCH_LEVERAGE = 2.0


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class LinearPerpetualInstrument:
    """Current Bybit V5 rules needed before sizing a linear perpetual order."""

    symbol: str
    base_coin: str
    quote_coin: str
    settle_coin: str
    status: str
    contract_type: str
    min_quantity: float
    max_market_quantity: float
    quantity_step: float
    min_notional: float
    tick_size: float
    min_leverage: float
    max_leverage: float
    leverage_step: float
    funding_interval_minutes: int

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.upper():
            raise ValueError("symbol must be a non-empty uppercase Bybit symbol")
        if self.quote_coin.upper() != "USDT" or self.settle_coin.upper() != "USDT":
            raise ValueError("only linear USDT-quoted and USDT-settled contracts are supported")
        if not self.base_coin:
            raise ValueError("base_coin must be present")
        if self.status != "Trading":
            raise ValueError("instrument must have Trading status")
        if self.contract_type != "LinearPerpetual":
            raise ValueError("only linear perpetual contracts are supported")
        for name in (
            "min_quantity", "max_market_quantity", "quantity_step", "min_notional",
            "tick_size", "min_leverage", "max_leverage", "leverage_step",
        ):
            _finite_positive(getattr(self, name), name)
        if self.max_market_quantity < self.min_quantity:
            raise ValueError("max_market_quantity is below min_quantity")
        if self.max_leverage < self.min_leverage:
            raise ValueError("max_leverage is below min_leverage")
        if (
            isinstance(self.funding_interval_minutes, bool)
            or not isinstance(self.funding_interval_minutes, int)
            or self.funding_interval_minutes < 1
        ):
            raise ValueError("funding_interval_minutes must be a positive integer")


@dataclass(frozen=True)
class PerpetualRiskSettings:
    """Hard limits for the isolated, one-way research simulator."""

    leverage: float = 1.0
    max_leverage: float = MAX_RESEARCH_LEVERAGE
    risk_fraction: float = 0.005
    max_margin_fraction: float = 0.10
    max_notional_fraction: float = 0.20
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.005
    liquidation_buffer_fraction: float = 0.02
    max_daily_loss_fraction: float = 0.03
    max_drawdown_fraction: float = 0.10
    max_consecutive_losses: int = 3

    def __post_init__(self) -> None:
        for name in (
            "leverage", "max_leverage", "risk_fraction", "max_margin_fraction",
            "max_notional_fraction", "maintenance_margin_rate", "liquidation_fee_rate",
            "liquidation_buffer_fraction", "max_daily_loss_fraction", "max_drawdown_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 1.0 <= self.leverage <= self.max_leverage <= MAX_RESEARCH_LEVERAGE:
            raise ValueError("research leverage must be between 1x and the hard 2x ceiling")
        if not 0 < self.risk_fraction <= 0.01:
            raise ValueError("risk_fraction must be in (0, 0.01]")
        if not 0 < self.max_margin_fraction <= 0.20:
            raise ValueError("max_margin_fraction must be in (0, 0.20]")
        if not 0 < self.max_notional_fraction <= 0.40:
            raise ValueError("max_notional_fraction must be in (0, 0.40]")
        if not 0 < self.maintenance_margin_rate < 0.10:
            raise ValueError("maintenance_margin_rate must be in (0, 0.10)")
        if not 0 <= self.liquidation_fee_rate <= 0.05:
            raise ValueError("liquidation_fee_rate must be in [0, 0.05]")
        if not 0 < self.liquidation_buffer_fraction <= 0.10:
            raise ValueError("liquidation_buffer_fraction must be in (0, 0.10]")
        if not 0 < self.max_daily_loss_fraction <= 0.05:
            raise ValueError("max_daily_loss_fraction must be in (0, 0.05]")
        if not 0 < self.max_drawdown_fraction <= 0.20:
            raise ValueError("max_drawdown_fraction must be in (0, 0.20]")
        if (
            isinstance(self.max_consecutive_losses, bool)
            or not isinstance(self.max_consecutive_losses, int)
            or not 1 <= self.max_consecutive_losses <= 10
        ):
            raise ValueError("max_consecutive_losses must be an integer in [1, 10]")


@dataclass(frozen=True)
class PositionSize:
    quantity: float
    notional: float
    initial_margin: float
    risk_budget: float
    modeled_stop_risk: float
    rejection_reason: str | None = None

    @property
    def approved(self) -> bool:
        return self.quantity > 0 and self.rejection_reason is None


def round_quantity_down(quantity: float, quantity_step: float) -> float:
    """Floor to the venue's decimal quantity step without binary-float rounding up."""
    quantity = _finite_positive(quantity, "quantity")
    quantity_step = _finite_positive(quantity_step, "quantity_step")
    units = (decimal.Decimal(str(quantity)) / decimal.Decimal(str(quantity_step))).to_integral_value(
        rounding=decimal.ROUND_DOWN
    )
    return float(units * decimal.Decimal(str(quantity_step)))


def size_linear_position(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    direction: int,
    cost_rate_per_fill: float,
    instrument: LinearPerpetualInstrument,
    risk: PerpetualRiskSettings,
) -> PositionSize:
    """Size a position under both stop-risk and isolated initial-margin caps.

    ``cost_rate_per_fill`` includes one side's commission, half-spread and slippage.
    Stop-risk includes estimated entry and stop-exit costs; it is not just the
    leverage-adjusted notional and never increases the approved size.
    """
    equity = _finite_positive(equity, "equity")
    entry_price = _finite_positive(entry_price, "entry_price")
    stop_price = _finite_positive(stop_price, "stop_price")
    if isinstance(cost_rate_per_fill, bool) or not math.isfinite(cost_rate_per_fill) or cost_rate_per_fill < 0:
        raise ValueError("cost_rate_per_fill must be finite and non-negative")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 for long or -1 for short")
    if (direction > 0 and stop_price >= entry_price) or (direction < 0 and stop_price <= entry_price):
        return PositionSize(0, 0, 0, equity * risk.risk_fraction, 0, "stop_on_wrong_side")
    if risk.leverage < instrument.min_leverage or risk.leverage > instrument.max_leverage:
        return PositionSize(0, 0, 0, equity * risk.risk_fraction, 0, "leverage_outside_instrument_limits")
    leverage_steps = (risk.leverage - instrument.min_leverage) / instrument.leverage_step
    if not math.isclose(leverage_steps, round(leverage_steps), rel_tol=0, abs_tol=1e-8):
        return PositionSize(0, 0, 0, equity * risk.risk_fraction, 0, "invalid_leverage_step")

    stop_distance = abs(entry_price - stop_price)
    per_unit_risk = stop_distance + (entry_price + stop_price) * cost_rate_per_fill
    if not math.isfinite(per_unit_risk) or per_unit_risk <= 0:
        return PositionSize(0, 0, 0, equity * risk.risk_fraction, 0, "invalid_stop_risk")
    risk_budget = equity * risk.risk_fraction
    risk_quantity = risk_budget / per_unit_risk
    margin_notional_cap = equity * risk.max_margin_fraction * risk.leverage
    notional_cap = equity * risk.max_notional_fraction
    cap_notional = min(margin_notional_cap, notional_cap)
    raw_quantity = min(risk_quantity, cap_notional / entry_price, instrument.max_market_quantity)
    quantity = round_quantity_down(raw_quantity, instrument.quantity_step) if raw_quantity > 0 else 0.0
    notional = quantity * entry_price
    margin = notional / risk.leverage if quantity > 0 else 0.0
    modeled_risk = quantity * per_unit_risk

    rejection = None
    if quantity + 1e-12 < instrument.min_quantity:
        rejection = "below_minimum_quantity"
    elif notional + 1e-9 < instrument.min_notional:
        rejection = "below_minimum_notional"
    elif modeled_risk > risk_budget + 1e-8:
        rejection = "risk_budget_exceeded_after_rounding"
    elif margin > equity * risk.max_margin_fraction + 1e-8:
        rejection = "initial_margin_limit_exceeded"
    if rejection:
        return PositionSize(0, 0, 0, risk_budget, 0, rejection)
    return PositionSize(quantity, notional, margin, risk_budget, modeled_risk)


def estimate_isolated_liquidation_price(
    *,
    entry_price: float,
    leverage: float,
    maintenance_margin_rate: float,
    direction: int,
) -> float:
    """Approximate linear isolated liquidation price, not Bybit's exact tier formula.

    Bybit's actual value depends on risk tier, maintenance-margin deductions,
    account settings and fees. This conservative simulator therefore requires a
    current instrument and risk-tier review before any Demo integration.
    """
    entry_price = _finite_positive(entry_price, "entry_price")
    if isinstance(leverage, bool) or not math.isfinite(leverage) or leverage < 1:
        raise ValueError("leverage must be finite and >= 1")
    if (
        isinstance(maintenance_margin_rate, bool)
        or not math.isfinite(maintenance_margin_rate)
        or not 0 < maintenance_margin_rate < 0.10
    ):
        raise ValueError("maintenance_margin_rate must be in (0, 0.10)")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 for long or -1 for short")
    factor = 1.0 - 1.0 / leverage + maintenance_margin_rate
    if direction < 0:
        factor = 1.0 + 1.0 / leverage - maintenance_margin_rate
    price = entry_price * factor
    if not math.isfinite(price) or price <= 0:
        raise ValueError("liquidation price is not positive")
    return price


def linear_funding_cashflow(
    *, direction: int, quantity: float, mark_price: float, funding_rate: float
) -> float:
    """Cash paid (+negative) or received (+positive) at a linear funding event."""
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 for long or -1 for short")
    quantity = _finite_positive(quantity, "quantity")
    mark_price = _finite_positive(mark_price, "mark_price")
    if isinstance(funding_rate, bool) or not math.isfinite(funding_rate):
        raise ValueError("funding_rate must be finite")
    if abs(funding_rate) > 0.05:
        raise ValueError("funding_rate exceeds the 5% sanity limit")
    return -direction * quantity * mark_price * funding_rate
