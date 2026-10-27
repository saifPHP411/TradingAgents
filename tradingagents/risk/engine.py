"""The deterministic risk engine: the only thing that decides whether a trade is allowed.

The AI brain proposes; this module disposes. That split is the entire reason the
risk engine exists, and it is why nothing here reads model text, calls a model,
or accepts a size the model chose. Every limit is a number in a config, checked
in code, and the answer is one of a closed set of verdicts.

Why this cannot be a prompt. A prompt is a suggestion to a stochastic process.
Asked to "keep positions under 10% of equity", a model will usually comply and
sometimes not, and the failures are the expensive ones — the oversized position,
the short it decided was a hedge, the fifteenth name in a twelve-name book.
Constraints that must hold have to be arithmetic.

Long-only is enforced three ways, none of which can be prompted away:

1. :class:`~tradingagents.signals.contract.SignalAction` has no SHORT, so a short
   cannot be *expressed*.
2. :meth:`RiskEngine.evaluate` refuses any non-positive requested quantity, so a
   negative size cannot slip through as a sign error.
3. The engine refuses a sell that would leave a negative position, so even a
   valid SELL cannot open a short through the back door.

Sizing is the engine's job. When a signal carries a ``position_size`` the engine
treats it as a request and clamps it to ``max_position_size``; when it carries
none the engine sizes from ``default_position_size``. The model's opinion of the
size moves the order only within limits it cannot change.

Drawdown and daily loss are checked against the book's own history rather than a
running counter, so they cannot drift out of sync with the ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.portfolio.ledger import Ledger
from tradingagents.signals.contract import SignalAction, TradingSignal


class RiskVerdict(StrEnum):
    """The engine's answer. Two outcomes, so a caller cannot half-apply a rule."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RiskReason(StrEnum):
    """Why a trade was refused. A closed set, so rejections are analysable."""

    NOT_ACTIONABLE = "NOT_ACTIONABLE"
    NON_POSITIVE_QUANTITY = "NON_POSITIVE_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    MIN_ORDER_VALUE = "MIN_ORDER_VALUE"
    MAX_ORDER_VALUE = "MAX_ORDER_VALUE"
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_PORTFOLIO_EXPOSURE = "MAX_PORTFOLIO_EXPOSURE"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_POSITION = "INSUFFICIENT_POSITION"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    NO_EQUITY = "NO_EQUITY"


class RiskLimits(BaseModel):
    """The configured limits. Every one is a number, and every one is checked.

    A limit of ``None`` disables that check, which is deliberate: a caller
    running a research simulation without exposure caps should be able to say so
    explicitly rather than by setting a limit to a huge number and hoping.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_position_size: float | None = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="Most equity one position may represent, as a fraction",
    )
    max_portfolio_exposure: float | None = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Most equity all positions together may represent, as a fraction",
    )
    max_open_positions: int | None = Field(
        default=10, ge=1, description="Most distinct open positions at once"
    )
    max_daily_loss: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Most of the day's opening equity that may be lost in one day, as a "
            "fraction. None disables the check."
        ),
    )
    max_drawdown: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Most peak-to-trough decline tolerated, as a fraction. None disables it.",
    )
    max_order_value: float | None = Field(
        default=None, gt=0.0, description="Largest single order notional. None disables it."
    )
    min_order_value: float | None = Field(
        default=None, ge=0.0, description="Smallest single order notional. None disables it."
    )
    default_position_size: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of equity used when a signal requests no size. Kept at or "
            "below max_position_size by the validator."
        ),
    )
    long_only: bool = Field(
        default=True,
        description=(
            "Whether shorts are forbidden. The engine only implements long-only, so "
            "setting this False is a configuration error rather than a behaviour change."
        ),
    )

    @model_validator(mode="after")
    def _limits_are_coherent(self) -> RiskLimits:
        if not self.long_only:
            raise ValueError(
                "this engine only implements long-only enforcement; set long_only=True"
            )
        if (
            self.max_position_size is not None
            and self.default_position_size > self.max_position_size
        ):
            raise ValueError(
                f"default_position_size ({self.default_position_size}) must not exceed "
                f"max_position_size ({self.max_position_size})"
            )
        if (
            self.min_order_value is not None
            and self.max_order_value is not None
            and self.min_order_value > self.max_order_value
        ):
            raise ValueError(
                f"min_order_value ({self.min_order_value}) exceeds "
                f"max_order_value ({self.max_order_value})"
            )
        return self


@dataclass
class RiskDecision:
    """The engine's verdict on one signal, with the reasoning a report can quote."""

    verdict: RiskVerdict
    symbol: str
    action: SignalAction
    quantity: float = 0.0
    """Units approved. Zero for a rejection, and zero for an approved HOLD."""
    reference_price: float | None = None
    notional: float = 0.0
    reasons: list[RiskReason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Human-readable detail, one entry per reason. Never model text."""

    @property
    def approved(self) -> bool:
        return self.verdict is RiskVerdict.APPROVED

    @property
    def is_order(self) -> bool:
        """Whether this decision should produce an order at all."""
        return self.approved and self.quantity > 0

    def explain(self) -> str:
        """One line summarising the verdict, for a log or a report."""
        if self.approved:
            if self.quantity <= 0:
                return f"{self.symbol}: {self.action.value} approved as no-op"
            return (
                f"{self.symbol}: {self.action.value} approved, {self.quantity:.6g} units "
                f"at {self.reference_price} ({self.notional:,.2f})"
            )
        return f"{self.symbol}: {self.action.value} rejected ({', '.join(r.value for r in self.reasons)})"


class RiskEngine:
    """Applies :class:`RiskLimits` to a :class:`TradingSignal` against a ledger.

    Stateless apart from its limits: everything it needs to know about the book
    it reads from the ledger it is handed, so two engines with the same limits
    give the same verdict for the same book. That is what makes the enforcement
    testable.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    # --- Public API ---------------------------------------------------------

    def evaluate(
        self,
        signal: TradingSignal,
        ledger: Ledger,
        *,
        price: float | None,
        equity_peak: float | None = None,
        day_start_equity: float | None = None,
    ) -> RiskDecision:
        """Decide whether ``signal`` may become an order, and at what size.

        ``price`` is the reference price the order would be priced against. A
        non-positive or missing price is a rejection, because a risk limit
        expressed in equity cannot be checked without one.

        ``equity_peak`` and ``day_start_equity`` let the caller supply the
        history the drawdown and daily-loss checks need. When they are absent
        those two checks cannot fire, and the decision says so in its notes
        rather than silently passing.
        """
        decision = RiskDecision(
            verdict=RiskVerdict.APPROVED, symbol=signal.symbol, action=signal.action
        )

        if signal.action is SignalAction.HOLD:
            decision.reasons.append(RiskReason.NOT_ACTIONABLE)
            decision.notes.append("the signal is a HOLD; no order is produced")
            return self._reject(decision)

        if price is None or price <= 0:
            decision.reasons.append(RiskReason.INVALID_PRICE)
            decision.notes.append(f"no usable reference price ({price}) to size against")
            return self._reject(decision)

        decision.reference_price = float(price)
        equity = ledger.equity
        if equity <= 0:
            decision.reasons.append(RiskReason.NO_EQUITY)
            decision.notes.append(f"equity is {equity:,.2f}; there is nothing to size against")
            return self._reject(decision)

        # Book-level checks first: a book already over a limit must not accept a
        # new position, whatever the individual order would look like.
        self._check_book_limits(decision, ledger, equity, equity_peak, day_start_equity)
        if decision.reasons:
            return self._reject(decision)

        quantity = self._size(signal, ledger, price, equity, decision)
        if decision.reasons:
            return self._reject(decision)

        if signal.action is SignalAction.SELL:
            self._check_sell(decision, ledger, signal.symbol, quantity)
            if decision.reasons:
                return self._reject(decision)

        decision.quantity = quantity
        decision.notional = quantity * price
        return decision

    # --- Book-level limits --------------------------------------------------

    def _check_book_limits(
        self,
        decision: RiskDecision,
        ledger: Ledger,
        equity: float,
        equity_peak: float | None,
        day_start_equity: float | None,
    ) -> None:
        """The limits that depend on the whole book, not on this order."""
        limits = self.limits

        if limits.max_drawdown is not None:
            if equity_peak is None:
                decision.notes.append("max_drawdown not checked: no equity peak supplied")
            elif equity_peak > 0:
                drawdown = (equity_peak - equity) / equity_peak
                if drawdown >= limits.max_drawdown:
                    decision.reasons.append(RiskReason.MAX_DRAWDOWN)
                    decision.notes.append(
                        f"drawdown {drawdown:.2%} has reached the {limits.max_drawdown:.2%} limit"
                    )

        if limits.max_daily_loss is not None:
            if day_start_equity is None:
                decision.notes.append("max_daily_loss not checked: no day-start equity supplied")
            elif day_start_equity > 0:
                loss = (day_start_equity - equity) / day_start_equity
                if loss >= limits.max_daily_loss:
                    decision.reasons.append(RiskReason.MAX_DAILY_LOSS)
                    decision.notes.append(
                        f"the day is down {loss:.2%}, at or past the {limits.max_daily_loss:.2%} limit"
                    )

    # --- Sizing -------------------------------------------------------------

    def _size(
        self,
        signal: TradingSignal,
        ledger: Ledger,
        price: float,
        equity: float,
        decision: RiskDecision,
    ) -> float:
        """The quantity to trade, after clamping the request to every size limit.

        Returns 0.0 and records reasons when nothing may be traded. A clamp that
        reduces the request is recorded as a note, so a report can show that the
        model asked for more than it got.
        """
        limits = self.limits
        if signal.action is SignalAction.SELL:
            return self._size_sell(signal, ledger, price, decision)

        requested = signal.position_size if signal.position_size is not None else limits.default_position_size
        if signal.position_size is not None and limits.max_position_size is not None \
                and signal.position_size > limits.max_position_size:
            decision.notes.append(
                f"requested size {signal.position_size:.2%} clamped to the "
                f"{limits.max_position_size:.2%} position limit"
            )
        requested_value = min(
            requested,
            limits.max_position_size if limits.max_position_size is not None else requested,
        ) * equity

        # Every cap bounds the position the order would *leave behind*, not the
        # order itself. Capping the order alone would let a strategy that buys
        # 10% five times hold 50% of a name, which is the opposite of a position
        # limit. So each cap is converted into headroom above the holding.
        held = ledger.position_quantity(signal.symbol)
        held_value = held * price
        value = requested_value

        position_headroom = self._position_headroom_value(ledger, equity, held_value, decision)
        if decision.reasons:
            return 0.0
        if position_headroom is not None:
            if requested_value > position_headroom:
                decision.notes.append(
                    f"size reduced to keep {signal.symbol} within the "
                    f"{limits.max_position_size:.2%} position limit"
                )
            value = min(value, position_headroom)

        exposure_headroom = self._exposure_headroom_value(ledger, equity, held_value, decision)
        if decision.reasons:
            return 0.0
        if exposure_headroom is not None:
            value = min(value, exposure_headroom)

        cash = ledger.cash
        if value > cash:
            decision.notes.append(
                f"size capped by available cash: {value:,.2f} requested, {cash:,.2f} free"
            )
            value = cash

        self._check_new_position_count(decision, ledger, signal.symbol)

        return self._apply_value_limits(value, price, decision)

    def _position_headroom_value(
        self, ledger: Ledger, equity: float, held_value: float, decision: RiskDecision
    ) -> float | None:
        """Value that may be added without taking this symbol past its position cap.

        ``None`` when no cap is configured. Zero (recorded as a rejection) when
        the holding is already at or above the cap, which happens when a position
        appreciates into its limit or when a cap is tightened mid-run.
        """
        cap = self.limits.max_position_size
        if cap is None:
            return None
        headroom = cap * equity - held_value
        if headroom <= 0:
            decision.reasons.append(RiskReason.MAX_POSITION_SIZE)
            decision.notes.append(
                f"the {self.limits.max_position_size:.2%} position limit leaves no room to add"
            )
            return 0.0
        return headroom

    def _size_sell(
        self, signal: TradingSignal, ledger: Ledger, price: float, decision: RiskDecision
    ) -> float:
        """A SELL trades what is held, clamped by the requested fraction of it."""
        held = ledger.position_quantity(signal.symbol)
        if held <= 0:
            decision.reasons.append(RiskReason.INSUFFICIENT_POSITION)
            decision.notes.append(
                f"no position in {signal.symbol} to sell; this book is long-only and "
                "will not open a short"
            )
            return 0.0
        requested = signal.position_size
        if requested is None:
            return held
        return min(held, held * requested) if requested < 1.0 else held

    def _exposure_headroom_value(
        self, ledger: Ledger, equity: float, held_value: float, decision: RiskDecision
    ) -> float | None:
        """Value that may still be added without breaching the exposure cap.

        ``None`` when no cap is configured. The current market value of this
        symbol is excluded from the base, because the cap applies to the book
        *after* the order and the order replaces part of that value rather than
        adding to it. Zero, recorded as a rejection, when the book is already at
        the cap.
        """
        if self.limits.max_portfolio_exposure is None:
            return None
        cap_value = self.limits.max_portfolio_exposure * equity
        current = ledger.market_value - held_value
        headroom = cap_value - current
        if headroom <= 0:
            decision.reasons.append(RiskReason.MAX_PORTFOLIO_EXPOSURE)
            decision.notes.append(
                f"the book is at the {self.limits.max_portfolio_exposure:.2%} exposure "
                "limit; there is no room for another position"
            )
            return 0.0
        return headroom

    def _check_new_position_count(self, decision: RiskDecision, ledger: Ledger, symbol: str) -> None:
        """A new name must not take the book past its position count."""
        cap = self.limits.max_open_positions
        if cap is None:
            return
        open_positions = ledger.open_positions()
        if symbol not in open_positions and len(open_positions) >= cap:
            decision.reasons.append(RiskReason.MAX_OPEN_POSITIONS)
            decision.notes.append(
                f"the book already holds {len(open_positions)} positions, at the {cap} limit"
            )

    def _apply_value_limits(self, target_value: float, price: float, decision: RiskDecision) -> float:
        """Convert a target notional to units, enforcing the order-value band.

        The band is applied to the notional, then the notional is converted, so a
        min/max order value is respected exactly rather than approximately.
        """
        limits = self.limits
        value = target_value
        if limits.max_order_value is not None and value > limits.max_order_value:
            decision.notes.append(
                f"order capped to max_order_value {limits.max_order_value:,.2f} from {value:,.2f}"
            )
            value = limits.max_order_value
        if limits.min_order_value is not None and 0 < value < limits.min_order_value:
            decision.reasons.append(RiskReason.MIN_ORDER_VALUE)
            decision.notes.append(
                f"order value {value:,.2f} is below min_order_value {limits.min_order_value:,.2f}"
            )
            return 0.0
        if value <= 0:
            decision.reasons.append(RiskReason.MAX_ORDER_VALUE)
            decision.notes.append("no value is available for this order after applying the limits")
            return 0.0
        return value / price

    # --- Sell-side checks ---------------------------------------------------

    def _check_sell(
        self, decision: RiskDecision, ledger: Ledger, symbol: str, quantity: float
    ) -> None:
        """A sell must not exceed the holding, which is how a short would appear."""
        held = ledger.position_quantity(symbol)
        if quantity > held + 1e-9:
            decision.reasons.append(RiskReason.SHORT_NOT_ALLOWED)
            decision.notes.append(
                f"selling {quantity:.6g} {symbol} would take the position negative "
                f"(holding {held:.6g}); short positions are not allowed"
            )

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _reject(decision: RiskDecision) -> RiskDecision:
        decision.verdict = RiskVerdict.REJECTED
        decision.quantity = 0.0
        decision.notional = 0.0
        return decision

    def evaluate_quantity(
        self,
        *,
        symbol: str,
        action: SignalAction,
        quantity: float,
        ledger: Ledger,
        price: float | None,
    ) -> RiskDecision:
        """Check an explicit quantity, for a caller that has already sized the order.

        This is the guard for a hand-built order: a non-positive quantity is
        refused here, so a negative size cannot reach the execution engine even
        if a caller bypasses :func:`~tradingagents.signals.contract.signal_from_decision`.
        """
        decision = RiskDecision(verdict=RiskVerdict.APPROVED, symbol=symbol, action=action)
        if action is SignalAction.HOLD:
            decision.reasons.append(RiskReason.NOT_ACTIONABLE)
            decision.notes.append("a HOLD produces no order")
            return self._reject(decision)
        if quantity is None or quantity <= 0:
            decision.reasons.append(RiskReason.NON_POSITIVE_QUANTITY)
            decision.notes.append(
                f"quantity must be positive, got {quantity}; direction is carried by the action, "
                "never by a sign"
            )
            return self._reject(decision)
        if price is None or price <= 0:
            decision.reasons.append(RiskReason.INVALID_PRICE)
            decision.notes.append(f"no usable reference price ({price})")
            return self._reject(decision)
        if symbol not in ledger.holdings and action is SignalAction.SELL:
            decision.reasons.append(RiskReason.UNKNOWN_SYMBOL)
            decision.notes.append(f"no position in {symbol} to sell")
            return self._reject(decision)
        if action is SignalAction.SELL:
            self._check_sell(decision, ledger, symbol, quantity)
            if decision.reasons:
                return self._reject(decision)
        decision.quantity = quantity
        decision.reference_price = float(price)
        decision.notional = quantity * price
        return decision

    # --- Fill-time re-check ------------------------------------------------

    def review_fill(
        self,
        *,
        symbol: str,
        action: SignalAction,
        quantity: float,
        ledger: Ledger,
        fill_price: float,
        max_notional: float | None = None,
    ) -> RiskDecision:
        """Re-check the position and exposure caps at the price a fill will get.

        Sizing happens when the decision is taken, against the price the decision
        saw. The fill happens later, at a price that can have moved — in a
        backtest it is the next session's open, which can gap. A cap satisfied at
        the decision price can therefore be breached by the fill: a position sized
        at 20% of equity on a 100 close is worth 40% when the fill opens at 200.
        The engine is event-driven, so the check and the fill are necessarily one
        session apart, and re-checking here is the only place that closes the gap.

        The re-check is the same arithmetic as :meth:`evaluate`, applied to
        ``fill_price`` instead of the decision price, so it enforces the caps on
        the position the fill actually creates rather than the one the decision
        intended. When a cap is exceeded it returns the *largest* quantity that
        keeps every cap, approved and noted, rather than refusing outright: a
        smaller fill is still a fill, and reducing is what the sizing path already
        does when a request is too large. Only when no positive quantity remains
        does it reject.

        It never touches the ledger. It computes and returns, so the caller
        applies the result or refuses it, and nothing is deducted for a quantity
        that is not executed.

        A SELL needs no cap re-check: it reduces the position and the exposure, so
        a gap can only make it smaller. Its holding constraint is the ledger's,
        and is enforced where the fill is applied.
        """
        decision = RiskDecision(verdict=RiskVerdict.APPROVED, symbol=symbol, action=action)

        if quantity is None or not math.isfinite(quantity) or quantity <= 0:
            decision.reasons.append(RiskReason.NON_POSITIVE_QUANTITY)
            decision.notes.append(
                f"fill quantity {quantity!r} is not a positive finite number"
            )
            return self._reject(decision)
        if fill_price is None or not math.isfinite(fill_price) or fill_price <= 0:
            decision.reasons.append(RiskReason.INVALID_PRICE)
            decision.notes.append(
                f"fill price {fill_price!r} is not a positive finite number"
            )
            return self._reject(decision)
        if max_notional is not None and (not math.isfinite(max_notional) or max_notional < 0):
            decision.reasons.append(RiskReason.INVALID_PRICE)
            decision.notes.append(
                f"max_notional {max_notional!r} is not a finite, non-negative number"
            )
            return self._reject(decision)

        decision.reference_price = float(fill_price)

        if action is SignalAction.SELL:
            # No cap re-check: a sell reduces the position and the exposure, so a
            # gap can only make it smaller. Its holding constraint is the
            # ledger's, enforced where the fill is applied, and the caller caps
            # the size to the holding before submitting, so a sell cannot oversell
            # through this path.
            decision.quantity = quantity
            decision.notional = quantity * fill_price
            return decision

        equity = ledger.equity
        if equity <= 0:
            decision.reasons.append(RiskReason.NO_EQUITY)
            decision.notes.append(
                f"equity is {equity:,.2f} at fill time; there is nothing to size against"
            )
            return self._reject(decision)

        limits = self.limits
        held = ledger.position_quantity(symbol)
        holding = ledger.holdings.get(symbol)
        # The symbol's own mark, so the rest of the book is valued without it and
        # its value can be recomputed at the fill price.
        mark = (
            holding.last_price
            if holding is not None and holding.last_price is not None
            else fill_price
        )
        other_positions_value = ledger.market_value - held * mark

        allowed = quantity
        binding: list[RiskReason] = []

        def cap_quantity(headroom_value: float, reason: RiskReason) -> None:
            """Tighten ``allowed`` to fit ``headroom_value`` of additional value."""
            nonlocal allowed
            max_quantity = headroom_value / fill_price
            if max_quantity < allowed - 1e-12:
                if reason not in binding:
                    binding.append(reason)
                allowed = max(max_quantity, 0.0)

        if limits.max_position_size is not None:
            # The value the position would hold *after* the fill.
            headroom = limits.max_position_size * equity - held * fill_price
            cap_quantity(headroom, RiskReason.MAX_POSITION_SIZE)

        if limits.max_portfolio_exposure is not None:
            headroom = (
                limits.max_portfolio_exposure * equity
                - other_positions_value
                - held * fill_price
            )
            cap_quantity(headroom, RiskReason.MAX_PORTFOLIO_EXPOSURE)

        if limits.max_order_value is not None:
            cap_quantity(limits.max_order_value, RiskReason.MAX_ORDER_VALUE)

        # The cash still available at fill time, as a notional the caller has
        # already made fee-aware. A size was checked against cash when the order
        # was created, but the fill price can gap and the fee is added at the
        # fill, so the request can exceed what the cash covers. Reducing to the
        # affordable notional is the honest outcome; refusing would lose a fill
        # the cash could still pay for.
        if max_notional is not None:
            cap_quantity(max_notional, RiskReason.INSUFFICIENT_CASH)

        if limits.max_open_positions is not None and held <= 0:
            open_positions = ledger.open_positions()
            if len(open_positions) >= limits.max_open_positions:
                decision.reasons.append(RiskReason.MAX_OPEN_POSITIONS)
                decision.notes.append(
                    f"the book already holds {len(open_positions)} positions at fill time, "
                    f"at the {limits.max_open_positions} limit"
                )
                return self._reject(decision)

        if allowed <= 0:
            decision.reasons.extend(binding or [RiskReason.MAX_POSITION_SIZE])
            decision.notes.append(
                "no positive quantity keeps the fill within the limits at the fill price "
                f"{fill_price:g}; the position is already at or above a cap"
            )
            return self._reject(decision)

        notional = allowed * fill_price
        if limits.min_order_value is not None and notional < limits.min_order_value:
            decision.reasons.append(RiskReason.MIN_ORDER_VALUE)
            decision.notes.append(
                f"the reduced fill value {notional:,.2f} is below min_order_value "
                f"{limits.min_order_value:,.2f}"
            )
            return self._reject(decision)

        if allowed < quantity - 1e-12:
            # Record which cap bound, so the reduction is auditable and a report
            # can say why the filled size differs from the requested one.
            decision.reasons.extend(binding)
            decision.notes.append(
                f"fill size reduced from {quantity:.6g} to {allowed:.6g} to keep the position "
                f"within the limits at the fill price {fill_price:g}"
            )
        decision.quantity = allowed
        decision.notional = notional
        return decision


def drawdown_from_peak(equity: float, peak: float) -> float:
    """Peak-to-trough decline as a positive fraction; 0 when at or above the peak."""
    if peak <= 0:
        return 0.0
    return max((peak - equity) / peak, 0.0)


def same_day(before: datetime, after: datetime) -> bool:
    """Whether two instants fall on the same calendar day (UTC)."""
    return before.date() == after.date()


def utc_date(value: datetime) -> date:
    """The UTC calendar date of an instant."""
    return value.date()
