"""Mandatory trade protection: the gate between a strategy idea and an order.

A trade is only worth taking when it can be *managed*. This module is the rule
that says so, and it is deliberately separate from both the strategy and the
broker:

- the **strategy** (TJR) decides *what* to trade and derives the levels from
  market structure — the invalidation swing for the stop, the next draw on
  liquidity for the target;
- this module decides whether those levels are **usable**: are they on the right
  side, do they describe a sane risk/reward, is there a stop *and* a target;
- the **broker** places the order and holds the position.

The rule this module enforces is absolute: **a trade with no protectable stop
is not taken.** The refusal happens locally, before any order reaches the venue,
and it names the reason from a closed vocabulary so a dashboard can tell an
operator exactly what was missing.

Why this is a separate module and not a check inside the worker: the same rule
has to hold on every path that can place an entry — the crypto TJR source, the
current-market source, the historical replay — and a rule that is duplicated
into each of them will eventually be missing from one. One validator, called by
the one place that submits an entry order, is the only arrangement where the
rule cannot drift.

The venue's own limits matter too. Alpaca's crypto API supports only ``market``,
``limit`` and ``stop_limit`` orders — there is no bracket order and no OCO for
crypto — so the venue cannot hold a stop and a target as one resting order.
The protection is therefore a *monitor*: the worker watches the market and
sends the exit itself when a level is crossed. That is a real constraint, and
it is why "protected" cannot mean "a level was recorded": it has to mean "the
monitor is running and the level it watches is executable".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# -- the closed vocabulary of protection failures ---------------------------------

NO_VALID_STOP_LOSS = "NO_VALID_STOP_LOSS"
"""No usable stop: missing, non-numeric, non-finite, non-positive, or on the
wrong side of the entry."""

NO_VALID_TAKE_PROFIT = "NO_VALID_TAKE_PROFIT"
"""No usable target: missing, non-numeric, non-finite, non-positive, or on the
wrong side of the entry."""

INVALID_RISK_REWARD = "INVALID_RISK_REWARD"
"""The target's reward is not at least the configured multiple of the risk."""

INVALID_TARGET_ORDER = "INVALID_TARGET_ORDER"
"""The targets are not strictly increasing (for a long) away from the entry."""

PROTECTION_UNAVAILABLE = "PROTECTION_UNAVAILABLE"
"""The levels are individually fine, but the position could not be protected:
the venue offers no usable order type for the exit, or the monitor could not
be armed."""

PROTECTION_REASON_CODES = (
    NO_VALID_STOP_LOSS,
    NO_VALID_TAKE_PROFIT,
    INVALID_RISK_REWARD,
    INVALID_TARGET_ORDER,
    PROTECTION_UNAVAILABLE,
)
"""Every reason a protection check can fail, from a closed set. A caller may
iterate this vocabulary; it never grows a prose-only reason."""


def _finite_positive(value: Any) -> float | None:
    """A finite, strictly positive number, or ``None``.

    NaN is the dangerous case: every comparison against it is false, so a NaN
    level passes a naive ``> 0`` guard and then poisons every arithmetic step
    downstream. ``math.isfinite`` is checked explicitly for exactly that reason.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


@dataclass(frozen=True)
class ProtectionPlan:
    """A validated, protectable set of levels for one position.

    ``entry``/``stop_loss``/``targets`` are the levels the strategy derived, in
    the market's own prices. ``targets`` is ordered and may hold one, two or
    three levels — TJR only claims the ones the market actually supports, so a
    single target is the normal case and an empty list is never produced by a
    successful validation.

    ``risk`` and ``reward`` are per-unit distances; ``risk_reward`` is their
    ratio. ``quantity`` is the size the plan implies, but it is the **risk
    engine's** size: this plan never overrides it, it only records what the
    risk engine was asked to respect.
    """

    entry: float
    stop_loss: float
    targets: tuple[float, ...]
    risk: float
    reward: float
    risk_reward: float
    side: str = "long"
    reason_code: str | None = None
    detail: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    """Where each level came from, when the strategy said. Purely an audit
    trail: the dashboard shows it and nothing branches on it."""

    @property
    def take_profit(self) -> float:
        """The first target. The one the exit policy watches when there is only
        one."""
        return self.targets[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "targets": list(self.targets),
            "take_profit": self.targets[0],
            "risk": self.risk,
            "reward": self.reward,
            "risk_reward": self.risk_reward,
            "side": self.side,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "sources": dict(self.sources),
        }


@dataclass(frozen=True)
class ProtectionRefusal:
    """A refused trade, with the closed-vocabulary reason."""

    reason_code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"reason_code": self.reason_code, "detail": self.detail}


def validate_protection(
    *,
    entry: Any,
    stop_loss: Any,
    targets: Any = None,
    take_profit: Any = None,
    side: str = "long",
    min_reward_risk: float = 1.0,
    max_targets: int = 3,
    sources: dict[str, str] | None = None,
) -> ProtectionPlan | ProtectionRefusal:
    """Whether these levels describe a trade that can be protected.

    The checks, in order, each with its own named reason:

    1. the **entry** is a finite positive price — without one, "the right side"
       is meaningless;
    2. the **stop** is a finite positive price;
    3. the **targets** are finite positive prices;
    4. the stop and every target sit on the **correct side** of the entry — a
       long's stop is below its entry and its targets are above it;
    5. the targets are **ordered** away from the entry;
    6. the nearest target's reward is at least ``min_reward_risk`` times the
       risk.

    A long is the only side this project trades (``allow_short`` is False
    everywhere), but the function keeps the short branch explicit rather than
    assuming it, so a future caller cannot accidentally read a long's stop as
    a short's.

    Returns a :class:`ProtectionPlan` when the trade is protectable, and a
    :class:`ProtectionRefusal` otherwise. It never raises for a bad input: a
    bad input is a refusal, which is what the caller wants to report.
    """
    side = side.lower()
    if side not in ("long", "short"):
        return ProtectionRefusal(
            PROTECTION_UNAVAILABLE, f"the side {side!r} is not a tradable direction"
        )

    entry_price = _finite_positive(entry)
    if entry_price is None:
        return ProtectionRefusal(
            NO_VALID_STOP_LOSS,
            "the entry price is missing or is not a finite positive number, so no "
            "stop can be placed on the correct side of it",
        )

    stop = _finite_positive(stop_loss)
    if stop is None:
        return ProtectionRefusal(
            NO_VALID_STOP_LOSS,
            "the strategy proposed no stop loss, or the stop is not a finite "
            "positive price; a trade without a stop cannot be protected",
        )

    raw_targets: list[Any]
    if targets is not None:
        raw_targets = list(targets)
    elif take_profit is not None:
        raw_targets = [take_profit]
    else:
        raw_targets = []
    if not raw_targets:
        return ProtectionRefusal(
            NO_VALID_TAKE_PROFIT,
            "the strategy proposed no take profit, or the target is not a finite "
            "positive price; a trade without a target has no defined exit",
        )

    clean_targets: list[float] = []
    for candidate in raw_targets:
        value = _finite_positive(candidate)
        if value is None:
            return ProtectionRefusal(
                NO_VALID_TAKE_PROFIT,
                "one of the proposed targets is not a finite positive price, so "
                "the exit it names cannot be placed",
            )
        clean_targets.append(value)
    clean_targets = sorted(set(clean_targets))
    if len(clean_targets) > max_targets:
        clean_targets = clean_targets[:max_targets]

    if side == "long":
        if stop >= entry_price:
            return ProtectionRefusal(
                NO_VALID_STOP_LOSS,
                f"the stop {stop:g} is not below the entry {entry_price:g}; a long's "
                "stop must sit on the losing side of its entry",
            )
        wrong_side = [level for level in clean_targets if level <= entry_price]
        if wrong_side:
            return ProtectionRefusal(
                NO_VALID_TAKE_PROFIT,
                f"the target {wrong_side[0]:g} is not above the entry "
                f"{entry_price:g}; a long's target must sit on the winning side "
                "of its entry",
            )
    else:
        if stop <= entry_price:
            return ProtectionRefusal(
                NO_VALID_STOP_LOSS,
                f"the stop {stop:g} is not above the entry {entry_price:g}; a short's "
                "stop must sit on the losing side of its entry",
            )
        wrong_side = [level for level in clean_targets if level >= entry_price]
        if wrong_side:
            return ProtectionRefusal(
                NO_VALID_TAKE_PROFIT,
                f"the target {wrong_side[0]:g} is not below the entry "
                f"{entry_price:g}; a short's target must sit on the winning side "
                "of its entry",
            )

    risk = abs(entry_price - stop)
    if risk <= 0:
        return ProtectionRefusal(
            NO_VALID_STOP_LOSS,
            "the stop is on the entry itself, so the trade carries no risk to "
            "manage",
        )
    reward = abs(clean_targets[0] - entry_price)
    risk_reward = reward / risk if risk > 0 else 0.0
    if risk_reward < min_reward_risk:
        return ProtectionRefusal(
            INVALID_RISK_REWARD,
            f"the nearest target offers {risk_reward:.2f}R against a required "
            f"{min_reward_risk:.2f}R, so the trade's reward does not justify its risk",
        )

    return ProtectionPlan(
        entry=entry_price,
        stop_loss=stop,
        targets=tuple(clean_targets),
        risk=risk,
        reward=reward,
        risk_reward=risk_reward,
        side=side,
        sources=dict(sources or {}),
    )


def size_for_stop_distance(
    *,
    entry: float,
    stop_loss: float,
    max_risk: float,
    max_notional: float,
    fraction: float = 1.0,
) -> float:
    """The quantity whose **risk** fits ``max_risk``, bounded by the notional.

    The size of a trade is a function of how far away its stop is: the same
    dollar risk buys more units when the stop is close and fewer when it is
    far. Sizing from a fixed notional alone ignores the stop entirely, which
    is how a 2% stop and a 20% stop end up carrying the same risk.

    The result is the **smaller** of the two bounds, so neither the risk cap
    nor the notional cap can be exceeded by the other:

    - the risk bound: ``max_risk / (entry - stop)`` units;
    - the notional bound: ``max_notional / entry`` units.

    ``fraction`` scales both down (never up), so a caller can take a fraction
    of the risk budget. The result is never negative, and a stop at the entry
    yields 0.0 — there is no risk to size, so there is no position.
    """
    distance = abs(float(entry) - float(stop_loss))
    if not math.isfinite(distance) or distance <= 0:
        return 0.0
    risk_units = (float(max_risk) * float(fraction)) / distance
    notional_units = (float(max_notional) * float(fraction)) / float(entry)
    if not math.isfinite(risk_units) or not math.isfinite(notional_units):
        return 0.0
    return max(0.0, min(risk_units, notional_units))


__all__ = [
    "INVALID_RISK_REWARD",
    "INVALID_TARGET_ORDER",
    "NO_VALID_STOP_LOSS",
    "NO_VALID_TAKE_PROFIT",
    "PROTECTION_REASON_CODES",
    "PROTECTION_UNAVAILABLE",
    "ProtectionPlan",
    "ProtectionRefusal",
    "size_for_stop_distance",
    "validate_protection",
]
