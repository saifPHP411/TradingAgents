"""Turn a completed TJR evaluation into the engine's validated signal.

The rule chain lives in :mod:`tradingagents.tjr.rules`; this module is only the
adapter to :class:`~tradingagents.signals.contract.TradingSignal`, so the signal the
engine sees is built the same way every other strategy's is — validated, long-only
by construction, with no size requested (the risk engine sizes it).

A completed setup becomes a BUY with its stop and target attached. Anything short
of complete becomes a HOLD, which the engine records and then ignores: TJR's whole
point is that most of the day is a no-trade, and the reason is carried on the
evaluation rather than smuggled into a trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.signals.contract import (
    SignalAction,
    TradingSignal,
    signal_from_decision,
)

from .frames import IntradaySeries
from .rules import (
    TjrEvaluation,
    TjrSettings,
    evaluate,
)


@dataclass
class TjrStrategy:
    """The TJR strategy over one symbol set, evaluated as of an instant.

    Holds the per-symbol 1m and 5m series (and optional 1h/4h for levels) and the
    settings. :meth:`signal` runs the chain for one symbol at one instant and maps
    the outcome onto a signal; :meth:`evaluate` returns the full evaluation for the
    audit trail.
    """

    one_minute: dict[str, IntradaySeries]
    five_minute: dict[str, IntradaySeries]
    hourly: dict[str, IntradaySeries] | None = None
    four_hour: dict[str, IntradaySeries] | None = None
    settings: TjrSettings = TjrSettings()
    allow_short: bool = False

    def evaluate(self, symbol: str, session: str, as_of: pd.Timestamp) -> TjrEvaluation:
        return evaluate(
            symbol=symbol,
            session=session,
            as_of=as_of,
            series_5m=self.five_minute[symbol],
            series_1m=self.one_minute[symbol],
            hourly=(self.hourly or {}).get(symbol),
            four_hour=(self.four_hour or {}).get(symbol),
            settings=self.settings,
            allow_short=self.allow_short,
        )

    def signal(self, symbol: str, session: str, as_of: pd.Timestamp) -> tuple[TradingSignal, TjrEvaluation]:
        """The signal and the evaluation it came from, for one symbol and instant."""
        evaluation = self.evaluate(symbol, session, as_of)
        return _signal_from_evaluation(evaluation), evaluation


def _signal_from_evaluation(evaluation: TjrEvaluation) -> TradingSignal:
    """Map a completed evaluation to a BUY, and anything else to a HOLD.

    The HOLD carries the stage and reason code in its reasoning text so a report can
    show *why* the strategy stood aside; it carries no levels and no size, which the
    contract enforces.
    """
    if evaluation.is_ready and evaluation.side == "long" and evaluation.entry is not None:
        signal = signal_from_decision(
            symbol=evaluation.symbol,
            rating="Buy",
            as_of_date=evaluation.session,
            timestamp=evaluation.as_of,
            entry_price=evaluation.entry,
            stop_loss=evaluation.stop,
            take_profit=evaluation.target,
            holding_period_days=1,
            source="tjr",
            reasoning=evaluation.detail,
        )
        assert signal is not None  # "Buy" always maps to an action
        return signal
    return TradingSignal(
        symbol=evaluation.symbol,
        action=SignalAction.HOLD,
        confidence=0.0,
        timestamp=evaluation.as_of,
        as_of_date=evaluation.session,
        rating="Hold",
        source="tjr",
        reasoning=f"{evaluation.stage}:{evaluation.reason_code}:{evaluation.detail}",
    )
