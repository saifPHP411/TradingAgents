"""The official reference strategy and benchmark.

This module fixes the *form* of the strategy before any result is seen, so the
baseline measures one pre-registered rule rather than the best of many. It is
deliberately simple and conventional: the point of the baseline is the protocol
and the honest measurement, not a clever signal.

The reference rule is a long-only trend filter:

    Hold a long position in a symbol while its fast simple moving average is
    above its slow simple moving average.

    Entry: on the first session a symbol is flat and its fast SMA is above its
    slow SMA, buy ``protocol.position_size`` of equity in it.
    Exit:  the engine's fixed ``holding_period_days`` time exit (and the
    signal's stop/target, when set), so the exit is not a second tuned rule.

The moving-average lengths are the conventional 20 and 100. They were chosen
from convention, not from this data, and no length is tuned on any window. The
shorter pair is deliberate: a walk-forward test window is 126 sessions, and a
200-session warmup would leave too few sessions inside a test window to trade at
all, which would make the out-of-sample fold an empty measurement. 20/100 warms
up inside a fold. The same object is used unchanged in development, validation,
test and every walk-forward fold — which is what makes the walk-forward selection
honest: there is exactly one candidate, so there is nothing to overfit.

A strategy is handed a point-in-time ``StrategyContext``. It reads only
``context.history(symbol)``, which is sliced at the current session by
construction, so it cannot see a future bar even if it tried.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.signals import SignalAction, TradingSignal

# Conventional lengths, fixed a priori. Not tuned on the baseline data. Shorter
# than the textbook 50/200 so a 126-session walk-forward test window can warm up
# and still trade; see the module docstring.
FAST_WINDOW = 20
SLOW_WINDOW = 100


@dataclass(frozen=True)
class SmaCrossover:
    """Long-only trend filter: long while the fast SMA is above the slow SMA."""

    position_size: float = 0.20
    fast: int = FAST_WINDOW
    slow: int = SLOW_WINDOW
    stop_loss_pct: float | None = 0.20
    """A 20% protective stop below the entry close, a conventional risk control."""
    name: str = "sma_crossover_20_100"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")
        if not 0.0 < self.position_size <= 1.0:
            raise ValueError("position_size must be in (0, 1]")

    def __call__(self, context) -> TradingSignal | None:
        """One signal, or ``None``. Deterministic: symbols are tried in sorted order.

        The engine takes a single signal per session, so this returns the first
        symbol (in the feed's sorted order) that is flat and in an uptrend. That
        keeps the choice deterministic and free of any hidden priority.
        """
        for symbol in context.symbols:
            if context.position(symbol) > 0:
                continue
            if not self._in_uptrend(context, symbol):
                continue
            close = context.close(symbol)
            if close is None:
                continue
            stop = None if self.stop_loss_pct is None else round(close * (1 - self.stop_loss_pct), 4)
            return TradingSignal(
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=0.6,
                timestamp=context.clock.at,
                as_of_date=context.session,
                position_size=self.position_size,
                stop_loss=stop,
                reasoning=(
                    f"{symbol}: {self.fast}-session SMA above {self.slow}-session SMA "
                    f"as of {context.session}"
                ),
            )
        return None

    def _in_uptrend(self, context, symbol: str) -> bool:
        """Whether ``symbol``'s fast SMA is above its slow SMA through this session.

        Uses only ``context.history``, which is cut at the session, so the last
        close it can see is the session's own close.
        """
        history = context.history(symbol)
        closes = history["close"]
        if len(closes) < self.slow:
            return False
        fast = float(closes.iloc[-self.fast:].mean())
        slow = float(closes.iloc[-self.slow:].mean())
        return fast > slow


def benchmark_equity(
    feed, symbol: str, *, initial_cash: float, sessions: list[str]
) -> pd.Series:
    """Buy-and-hold equity for ``symbol`` over ``sessions``.

    Buys at the first session's close and marks to each subsequent close, so it
    is comparable session by session with the strategy's own equity curve. A
    session with no bar for the symbol carries the last close forward, which is
    the same convention the backtest uses when marking a held position.
    """
    from tradingagents.performance.metrics import buy_and_hold_equity

    closes = feed.history(symbol, sessions[-1])["close"]
    closes = closes.reindex(pd.to_datetime(sessions)).ffill()
    closes.index = pd.Index(sessions, name="session")
    return buy_and_hold_equity(closes, initial_capital=initial_cash)


def align_benchmark(benchmark: pd.Series, sessions: list[str]) -> pd.Series:
    """Reindex a benchmark curve onto the strategy's own sessions, forward-filled."""
    return benchmark.reindex(sessions).ffill().bfill()
