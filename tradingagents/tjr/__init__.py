"""TJR: a faithful, testable implementation of TJR's day-trading strategy.

The strategy is a sequence of conditions on 1-minute and 5-minute bars — a liquidity
sweep, a 5-minute reversal confirmation, a 5-minute continuation confluence, a
1-minute entry confirmation — and this package implements exactly that, with every
rule checkable in code and every "no trade" explained by a reason code.

The two modules that matter most:

- :mod:`tradingagents.tjr.rules` is the strategy itself, a pure function of the bars
  visible at an instant.
- :mod:`tradingagents.tjr.engine` is the intraday, shared-book backtest that runs it
  over the real session clock, reusing the project's ledger, risk engine and cost
  model rather than inventing a second one.

Real data comes from the data layer (``dataflows.vendors.yahoo.intraday``); a run
never fabricates a bar, and a window the vendor will not serve is reported, not
filled in.
"""

from .engine import TjrBacktester, TjrConfig, TjrResult, TjrTrade
from .frames import IntradaySeries, resample
from .report import TjrReport, build_report
from .rules import (
    TJR_REASON_CODES,
    TjrEvaluation,
    TjrSettings,
    evaluate,
)
from .strategy import TjrStrategy

__all__ = [
    "TJR_REASON_CODES",
    "IntradaySeries",
    "TjrBacktester",
    "TjrConfig",
    "TjrEvaluation",
    "TjrReport",
    "TjrResult",
    "TjrSettings",
    "TjrStrategy",
    "TjrTrade",
    "build_report",
    "evaluate",
    "resample",
]
