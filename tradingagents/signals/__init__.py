"""Signal normalization: the AI brain's decision as a validated engine input.

``contract`` holds the validated :class:`TradingSignal` and the rating→action
mapping. Everything downstream of it (risk, execution, ledger) consumes only
signals, never agent prose.
"""

from tradingagents.signals.contract import (
    OrderIntent,
    SignalAction,
    SignalError,
    SignalTimeInForce,
    TradingSignal,
    action_for_rating,
    confidence_for_rating,
    signal_from_decision,
)

__all__ = [
    "OrderIntent",
    "SignalAction",
    "SignalError",
    "SignalTimeInForce",
    "TradingSignal",
    "action_for_rating",
    "confidence_for_rating",
    "signal_from_decision",
]
