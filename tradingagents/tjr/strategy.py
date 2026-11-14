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
from .sessions import SessionPolicy


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
    session_policy: SessionPolicy | None = None
    """The session clock the chain reads. ``None`` uses the 5m series' own policy,
    so an equity caller need not pass it and a crypto caller gets its policy from
    the series it built."""

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
            session_policy=self.session_policy,
        )

    def signal(
        self, symbol: str, session: str, as_of: pd.Timestamp
    ) -> tuple[TradingSignal, TjrEvaluation]:
        """The signal and the evaluation it came from, for one symbol and instant."""
        evaluation = self.evaluate(symbol, session, as_of)
        return _signal_from_evaluation(evaluation), evaluation


class MemoTjrStrategy(TjrStrategy):
    """A :class:`TjrStrategy` that caches its pure evaluations for one study run.

    The rule chain is a pure function of the bars closed by an instant *and of
    the settings it is run with*, so its answer for a ``(symbol, session,
    instant)`` is identical across the scenarios of a study that share those
    settings. Memoizing it is what makes a multi-scenario study of ten symbols
    finish in minutes rather than hours, and the cache is exact — a miss calls
    the real function — so a cached run and a cold run produce identical trades.

    The cache key carries the **settings and the session policy** as well as the
    instant. The rule chain reads both: a different swing lookback finds
    different swings, a different minimum R:R rejects a different setup, and a
    different session policy labels and bounds a different session. A key that
    omitted them would hand a sensitivity variant the *baseline's* answer and
    report a difference that the variation never produced.

    The cache key carries no book state because none of the rule chain reads the
    book; memoizing anything that did would be wrong, so nothing that does is
    memoized here.
    """

    def __init__(self, *, cache: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cache = cache

    def _cache_key(self, symbol: str, session: str, as_of: pd.Timestamp) -> tuple:
        """The identity of an evaluation: the instant *and* what it was run with."""
        return (
            symbol,
            session,
            pd.Timestamp(as_of).isoformat(),
            self.settings,
            self.session_policy,
            self.allow_short,
        )

    def evaluate(self, symbol: str, session: str, as_of: pd.Timestamp) -> TjrEvaluation:
        key = self._cache_key(symbol, session, as_of)
        hit = self._cache.get(key)
        if hit is None:
            hit = super().evaluate(symbol, session, as_of)
            self._cache[key] = hit
        return hit


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
            targets=list(evaluation.targets),
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
