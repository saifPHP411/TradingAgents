"""The intraday TJR backtest: a shared book over the session's 5-minute clock.

The daily engine (:mod:`tradingagents.backtesting.engine`) cannot run this strategy.
It marks every bar to a midnight date, so it has no 5-minute structure, no session
open, and no way to say "the 10:00 bar closed". TJR is entirely made of those
things, so it needs its own loop. Everything *around* the loop is reused, because
none of it is time-frame specific: the same :class:`~tradingagents.portfolio.Ledger`,
the same :class:`~tradingagents.risk.RiskEngine`, the same cost model, and the same
:class:`~tradingagents.signals.contract.TradingSignal`.

The loop, and the one rule that shapes it:

    for each 5m bar-close instant t, in order:
        exits first: a stop, a target, or the session close
        then entries: evaluate TJR on bars closed at t, size and approve via risk,
                      and fill at the *open of the next 5m bar*

Filling at the next bar's open is the same look-ahead guard the daily engine uses:
the decision was made from bars that had closed at ``t``, so filling at ``t``'s own
price would trade on a price the decision already saw. The next bar's open is the
first price a trader could have got. The next bar's *range* is never read — only its
open — so an entry cannot use information from inside the bar it fills in.

Exits are checked on each subsequent 5m bar's high/low. When a bar covers both the
stop and the target, the stop is assumed first: a bar cannot say which came first,
and assuming the loss is the conservative choice.

The book is shared across all symbols, and the ten symbols' events are merged into
one timeline, so two names competing for the same cash or the same position slot do
so in a fixed, reproducible order (by timestamp, then symbol).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from tradingagents.execution.costs import CostModel
from tradingagents.portfolio import Ledger
from tradingagents.risk import RiskEngine, RiskLimits
from tradingagents.signals.contract import SignalAction, TradingSignal

from .frames import (
    SESSION_TZ,
    IntradaySeries,
    session_close_utc,
    session_open_utc,
)
from .rules import TjrEvaluation, TjrSettings
from .strategy import TjrStrategy


@dataclass
class TjrTrade:
    """One closed round trip, with the ledger's own net P&L."""

    symbol: str
    session: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    exit_reason: str
    fees: float
    gross_pnl: float
    net_pnl: float
    reason_code: str = ""
    detail: str = ""

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost else 0.0


@dataclass
class TjrConfig:
    """Everything that shapes a TJR run, so the run is reproducible from it alone."""

    initial_cash: float = 100_000.0
    currency: str = "USD"
    risk_limits: RiskLimits = field(default_factory=lambda: RiskLimits(max_open_positions=5))
    costs: CostModel = field(default_factory=CostModel)
    """Fees and slippage. A default of zero is honest: it reports a frictionless
    figure, and the caller sets the schedule the study is meant to use."""
    settings: TjrSettings = field(default_factory=TjrSettings)
    allow_short: bool = False
    """Kept False: the book is long-only, matching the live configuration."""
    flat_by_session_close: bool = True
    """Close any open position on the session's last bar. TJR is a day strategy."""

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {self.initial_cash}")


@dataclass
class TjrResult:
    """What a TJR run produced."""

    run_id: str
    sessions: list[str] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    """(instant ISO, equity) at each marked 5m bar-close."""
    trades: list[TjrTrade] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    unfilled: list[dict] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)
    per_symbol: dict[str, dict] = field(default_factory=dict)
    ledger: Ledger | None = None
    config: TjrConfig | None = None
    initial_cash: float = 0.0
    symbols: tuple[str, ...] = ()

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.initial_cash

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(
            [value for _, value in self.equity_curve],
            index=pd.Index([stamp for stamp, _ in self.equity_curve], name="instant"),
            dtype=float,
        )


@dataclass
class _OpenPosition:
    symbol: str
    session: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    stop: float
    target: float
    fees: float
    evaluation: TjrEvaluation


class TjrBacktester:
    """Runs TJR over a set of symbols, moving one ledger through the session clock."""

    def __init__(
        self,
        *,
        one_minute: Mapping[str, IntradaySeries],
        five_minute: Mapping[str, IntradaySeries],
        hourly: Mapping[str, IntradaySeries] | None = None,
        four_hour: Mapping[str, IntradaySeries] | None = None,
        config: TjrConfig | None = None,
        run_id: str = "tjr",
    ) -> None:
        self.config = config or TjrConfig()
        self.one_minute = {s.upper(): v for s, v in one_minute.items()}
        self.five_minute = {s.upper(): v for s, v in five_minute.items()}
        self.hourly = None if hourly is None else {s.upper(): v for s, v in hourly.items()}
        self.four_hour = None if four_hour is None else {s.upper(): v for s, v in four_hour.items()}
        self.symbols = tuple(sorted(self.five_minute))
        missing = [s for s in self.symbols if s not in self.one_minute]
        if missing:
            raise ValueError(f"no 1-minute series for {missing}")
        self.strategy = TjrStrategy(
            one_minute=self.one_minute,
            five_minute=self.five_minute,
            hourly=self.hourly,
            four_hour=self.four_hour,
            settings=self.config.settings,
            allow_short=self.config.allow_short,
        )
        self.ledger = Ledger(initial_cash=self.config.initial_cash, currency=self.config.currency)
        self.risk = RiskEngine(self.config.risk_limits)
        self.result = TjrResult(
            run_id=run_id, ledger=self.ledger, config=self.config,
            initial_cash=self.config.initial_cash, symbols=self.symbols,
        )
        self._open: dict[str, _OpenPosition] = {}
        self._last_close: dict[str, float] = {}
        self._attempted: dict[tuple[str, str], bool] = {}
        self._equity_peak = self.config.initial_cash

    # --- The loop -----------------------------------------------------------

    def run(self) -> TjrResult:
        for session in self._sessions():
            self._run_session(session)
        return self.result

    def _sessions(self) -> list[str]:
        days: set[str] = set()
        for series in self.five_minute.values():
            local = series.frame.index.tz_convert(SESSION_TZ)
            days |= set(local.strftime("%Y-%m-%d"))
        return sorted(days)

    def _run_session(self, session: str) -> None:
        self.result.sessions.append(session)
        day_start_equity = self.ledger.equity
        instants = self._timeline(session)
        last_instant = instants[-1] if instants else None
        for instant in instants:
            self._process_exits(session, instant)
            self._process_entries(session, instant, day_start_equity)
            # Flat at the session close, or at the last bar the data actually
            # carries. A day strategy must not carry a position overnight, and a
            # feed that stops short of 16:00 is still the end of the day's trading.
            if self.config.flat_by_session_close and (
                instant >= session_close_utc(session) or instant == last_instant
            ):
                self._close_all(session, instant)
            self._mark(instant)

    def _timeline(self, session: str) -> list[pd.Timestamp]:
        """Every 5m bar-close instant in the session, ascending, de-duplicated.

        A 5m bar that starts at ``t`` closes at ``t + 5m``; the timeline holds the
        close instants, because that is when the bar's information is usable.
        """
        open_utc = session_open_utc(session)
        close_utc = session_close_utc(session)
        instants: set[pd.Timestamp] = set()
        for series in self.five_minute.values():
            local = series.frame.index.tz_convert(SESSION_TZ)
            in_session = (local.strftime("%Y-%m-%d") == session)
            for start in series.frame.index[in_session]:
                close = start + pd.Timedelta(minutes=series.bar_minutes)
                if open_utc < close <= close_utc:
                    instants.add(close)
        return sorted(instants)

    def _bar_closing_at(self, series: IntradaySeries, instant: pd.Timestamp):
        start = instant - pd.Timedelta(minutes=series.bar_minutes)
        frame = series.frame
        if start not in frame.index:
            return None
        return frame.loc[start]

    def _bar_starting_at(self, series: IntradaySeries, instant: pd.Timestamp):
        frame = series.frame
        if instant not in frame.index:
            return None
        return frame.loc[instant]

    # --- Exits --------------------------------------------------------------

    def _process_exits(self, session: str, instant: pd.Timestamp) -> None:
        for symbol in sorted(self._open):
            position = self._open[symbol]
            row = self._bar_closing_at(self.five_minute[symbol], instant)
            if row is None:
                continue
            if position.stop is not None and float(row["low"]) <= position.stop:
                self._close_position(symbol, position, instant, position.stop, "stop")
            elif position.target is not None and float(row["high"]) >= position.target:
                self._close_position(symbol, position, instant, position.target, "target")

    def _close_all(self, session: str, instant: pd.Timestamp) -> None:
        if not self.config.flat_by_session_close:
            return
        for symbol in sorted(self._open):
            position = self._open[symbol]
            row = self._bar_closing_at(self.five_minute[symbol], instant)
            if row is None:
                continue
            self._close_position(symbol, position, instant, float(row["close"]), "session_close")

    def _close_position(
        self, symbol: str, position: _OpenPosition, instant: pd.Timestamp, price: float, reason: str
    ) -> None:
        fill_price = self._apply_slippage(price, side="sell")
        fee = self._fee(quantity=position.quantity, price=fill_price)
        entry_notional = position.entry_price * position.quantity
        exit_notional = fill_price * position.quantity
        self.ledger.apply_fill(
            symbol=symbol,
            quantity=-position.quantity,
            price=fill_price,
            fee=fee,
            timestamp=instant.to_pydatetime(),
            reason=f"tjr_{reason}",
            order_id=f"tjr-{symbol}-{instant.isoformat()}-exit",
        )
        gross = exit_notional - entry_notional
        net = gross - position.fees - fee
        self.result.trades.append(
            TjrTrade(
                symbol=symbol,
                session=position.session,
                side="long",
                quantity=position.quantity,
                entry_price=position.entry_price,
                exit_price=fill_price,
                entry_time=position.entry_time.isoformat(),
                exit_time=instant.isoformat(),
                exit_reason=reason,
                fees=position.fees + fee,
                gross_pnl=gross,
                net_pnl=net,
                reason_code=position.evaluation.reason_code,
                detail=position.evaluation.detail,
            )
        )
        self._record_symbol(symbol, net)
        del self._open[symbol]

    # --- Entries ------------------------------------------------------------

    def _process_entries(self, session: str, instant: pd.Timestamp, day_start_equity: float) -> None:
        for symbol in self.symbols:
            if symbol in self._open:
                continue
            if self._attempted.get((symbol, session)):
                continue
            signal, evaluation = self.strategy.signal(symbol, session, instant)
            self.result.evaluations.append(evaluation.to_dict())
            self.result.reason_counts[evaluation.reason_code] = (
                self.result.reason_counts.get(evaluation.reason_code, 0) + 1
            )
            if signal.action is not SignalAction.BUY:
                continue
            self._attempted[(symbol, session)] = True
            self._try_enter(symbol, session, instant, signal, evaluation, day_start_equity)

    def _try_enter(
        self,
        symbol: str,
        session: str,
        instant: pd.Timestamp,
        signal: TradingSignal,
        evaluation: TjrEvaluation,
        day_start_equity: float,
    ) -> None:
        next_bar = self._bar_starting_at(self.five_minute[symbol], instant)
        if next_bar is None:
            self.result.unfilled.append(
                {"symbol": symbol, "session": session, "time": instant.isoformat(),
                 "reason": "no next bar to execute in"}
            )
            return
        reference = float(next_bar["open"])
        decision = self.risk.evaluate(
            signal,
            self.ledger,
            price=reference,
            equity_peak=self._equity_peak,
            day_start_equity=day_start_equity,
        )
        if not decision.approved or decision.quantity <= 0:
            self.result.rejections.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "time": instant.isoformat(),
                    "reasons": [r.value for r in decision.reasons],
                    "detail": decision.explain(),
                }
            )
            return
        quantity = self._affordable_quantity(decision.quantity, reference)
        if quantity <= 0:
            self.result.rejections.append(
                {"symbol": symbol, "session": session, "time": instant.isoformat(),
                 "reasons": ["INSUFFICIENT_CASH"], "detail": "cash cannot cover the order"}
            )
            return
        fill_price = self._apply_slippage(reference, side="buy")
        fee = self._fee(quantity=quantity, price=fill_price)
        try:
            self.ledger.apply_fill(
                symbol=symbol,
                quantity=quantity,
                price=fill_price,
                fee=fee,
                timestamp=instant.to_pydatetime(),
                reason="tjr_entry",
                order_id=f"tjr-{symbol}-{instant.isoformat()}-entry",
            )
        except Exception as exc:  # noqa: BLE001 - a ledger refusal is a rejection, not a crash
            self.result.rejections.append(
                {"symbol": symbol, "session": session, "time": instant.isoformat(),
                 "reasons": ["LEDGER_REFUSED"], "detail": type(exc).__name__}
            )
            return
        self._open[symbol] = _OpenPosition(
            symbol=symbol,
            session=session,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=instant,
            stop=evaluation.stop if evaluation.stop is not None else 0.0,
            target=evaluation.target if evaluation.target is not None else 0.0,
            fees=fee,
            evaluation=evaluation,
        )

    # --- Marking ------------------------------------------------------------

    def _mark(self, instant: pd.Timestamp) -> None:
        for symbol in self.symbols:
            row = self._bar_closing_at(self.five_minute[symbol], instant)
            if row is not None:
                self._last_close[symbol] = float(row["close"])
        if self._last_close:
            self.ledger.mark_many(self._last_close)
        equity = self.ledger.equity
        self._equity_peak = max(self._equity_peak, equity)
        self.result.equity_curve.append((instant.isoformat(), equity))

    # --- Cost helpers -------------------------------------------------------

    def _apply_slippage(self, price: float, *, side: str) -> float:
        """Worsen ``price`` by the spread and slippage the cost model configures."""
        bps = (self.config.costs.half_spread_bps + self.config.costs.slippage_bps) / 10_000.0
        if side == "buy":
            return price * (1.0 + bps)
        return price * (1.0 - bps)

    def _fee(self, *, quantity: float, price: float) -> float:
        return self.config.costs.fee_for_fill(quantity=quantity, price=price, is_maker=False)

    def _affordable_quantity(self, quantity: float, price: float) -> float:
        affordable = self.config.costs.affordable_notional(cash=self.ledger.cash, price=price)
        if affordable <= 0:
            return 0.0
        max_quantity = affordable / price
        return min(quantity, max_quantity)

    def _record_symbol(self, symbol: str, net_pnl: float) -> None:
        bucket = self.result.per_symbol.setdefault(
            symbol, {"trades": 0, "net_pnl": 0.0, "wins": 0, "losses": 0}
        )
        bucket["trades"] += 1
        bucket["net_pnl"] += net_pnl
        if net_pnl > 0:
            bucket["wins"] += 1
        elif net_pnl < 0:
            bucket["losses"] += 1
