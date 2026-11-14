"""The TJR rule chain, as a pure function of the bars visible at an instant.

This is the strategy itself, expressed exactly as the source video states it, and
nothing else. The chain, in order, and every step is required:

1. **Draws on liquidity** — mark session, 1h, 4h and swing highs/lows
   (:mod:`tradingagents.tjr.levels`).
2. **Liquidity sweep** — price takes a level out and *closes back inside* it. Not
   every breakout is a sweep: the close back through the level is what makes it one.
3. **5-minute reversal confirmation** — a 5m break of structure or inverse fair
   value gap, in the direction *against* the sweep.
4. **5-minute continuation confluence** — price retraces into a 5m fair value gap or
   the 50% equilibrium of the sweep leg. The reversal signal alone is not an entry.
5. **1-minute entry confirmation** — a 1m break of structure or inverse fair value
   gap in the trade's direction. If it does not complete, the answer is NO TRADE.
6. **Management** — stop beyond the invalidation swing, target the next draw on
   liquidity.

Nothing here consults a model. Every condition is arithmetic on OHLC bars, so the
signal is reproducible and can be argued with. The evaluation returns the *stage it
reached* and a reason code from a closed vocabulary, so "no trade" is always
explained rather than silent.

Assumptions the video leaves open are named in the dataclass defaults and repeated
in the final report; none of them is presented as a verbatim rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .frames import IntradaySeries
from .levels import (
    LiquidityLevel,
    liquidity_levels,
    running_extreme_before,
)
from .sessions import SessionPolicy
from .structure import (
    break_of_structure,
    equilibrium,
    fair_value_gaps,
    inverse_fair_value_gap,
    last_swing,
    reset_frame_cache,
    swing_points,
)

# The closed vocabulary of reasons a TJR evaluation produced no entry.
TJR_NO_SETUP = "NO_SETUP"
TJR_NO_SWEEP = "NO_SWEEP"
TJR_NO_5M_CONFIRMATION = "NO_5M_CONFIRMATION"
TJR_NO_CONTINUATION = "NO_CONTINUATION"
TJR_NO_1M_ENTRY = "NO_1M_ENTRY"
TJR_INVALID_GEOMETRY = "INVALID_GEOMETRY"
TJR_SHORT_SETUP_LONG_ONLY = "SHORT_SETUP_LONG_ONLY"
TJR_OUTSIDE_SESSION = "OUTSIDE_SESSION"
TJR_ENTRY_WINDOW_CLOSED = "ENTRY_WINDOW_CLOSED"
TJR_READY = "READY"

TJR_REASON_CODES: tuple[str, ...] = (
    TJR_NO_SETUP,
    TJR_NO_SWEEP,
    TJR_NO_5M_CONFIRMATION,
    TJR_NO_CONTINUATION,
    TJR_NO_1M_ENTRY,
    TJR_INVALID_GEOMETRY,
    TJR_SHORT_SETUP_LONG_ONLY,
    TJR_OUTSIDE_SESSION,
    TJR_ENTRY_WINDOW_CLOSED,
    TJR_READY,
)

# The stage reached, oldest first. A later stage implies every earlier one passed.
STAGE_NONE = "NONE"
STAGE_SWEPT = "SWEPT"
STAGE_CONFIRMED = "CONFIRMED"
STAGE_IN_ZONE = "IN_ZONE"
STAGE_READY = "READY"


@dataclass(frozen=True)
class Sweep:
    """A liquidity level taken out and closed back through, in one 5m bar."""

    level: LiquidityLevel
    bar: pd.Timestamp
    extreme: float
    direction: str  # "bearish" (a high swept) or "bullish" (a low swept)

    def to_dict(self) -> dict:
        return {
            "level": self.level.to_dict(),
            "bar": self.bar.isoformat(),
            "extreme": round(self.extreme, 6),
            "direction": self.direction,
        }


@dataclass
class TjrEvaluation:
    """What the rule chain concluded at one instant.

    ``side`` is ``"long"`` / ``"short"`` only when the chain completed; ``stage``
    says how far it got; ``reason_code`` says why it stopped. ``entry``/``stop``/
    ``target`` are populated only for a completed setup.
    """

    symbol: str
    session: str
    as_of: pd.Timestamp
    side: str | None
    stage: str
    reason_code: str
    detail: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    targets: list[float] = field(default_factory=list)
    """Every target the market supports, nearest first.

    ``target`` is the first of these, kept as its own field because it is what
    the strategy has always exposed. ``targets`` is the whole ladder, so a
    caller can manage a position in stages without re-deriving the levels.

    The ladder is **not** padded: it holds only the levels the market actually
    supports, so it can be empty (no setup), hold one target, or hold several.
    """
    sweep: Sweep | None = None
    confirmation: str | None = None
    zone: tuple[float, float] | None = None
    levels: list[LiquidityLevel] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.stage == STAGE_READY and self.side is not None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "session": self.session,
            "as_of": self.as_of.isoformat(),
            "side": self.side,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "entry": None if self.entry is None else round(self.entry, 6),
            "stop": None if self.stop is None else round(self.stop, 6),
            "target": None if self.target is None else round(self.target, 6),
            "targets": [round(value, 6) for value in self.targets],
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
            "confirmation": self.confirmation,
            "zone": None if self.zone is None else [round(self.zone[0], 6), round(self.zone[1], 6)],
            "levels": [level.to_dict() for level in self.levels],
        }


@dataclass(frozen=True)
class TjrSettings:
    """The knobs the video leaves open. Every default is a documented choice."""

    swing_lookback: int = 2
    """Bars on each side of a fractal swing. Two is the common 5m definition."""

    sweep_lookback_bars: int = 78
    """How many recent 5m bars may hold the sweep. 78 is a full regular session
    (6.5 hours), so a sweep made in the morning remains the day's setup until a
    later sweep replaces it — TJR treats the session's sweep as the day's idea,
    not a signal that expires after an hour."""

    equal_tolerance: float = 0.0005
    """Relative tolerance for calling two swings equal highs/lows (5 bps)."""

    stop_buffer: float = 0.0005
    """Stop sits this fraction beyond the invalidation swing."""

    min_reward_risk: float = 1.0
    """A setup whose nearest draw is closer than this multiple is not taken."""

    fallback_target_r: float = 2.0
    """When no draw sits beyond the entry, target this many times the risk."""

    latest_entry_time: tuple[int, int] = (15, 30)
    """No new entries at or after this ET time. The last 30 minutes are left alone."""

    require_index_alignment: bool = False
    """TJR's ES/NQ alignment filter. Off by default: the universe is 10 stocks and
    the rule is about index futures, which this study does not fetch. Documented,
    not silently dropped."""

    def __post_init__(self) -> None:
        if self.swing_lookback < 1:
            raise ValueError("swing_lookback must be at least 1")
        if self.sweep_lookback_bars < 1:
            raise ValueError("sweep_lookback_bars must be at least 1")
        if self.min_reward_risk <= 0:
            raise ValueError("min_reward_risk must be positive")


def _local_time(stamp: pd.Timestamp, policy: SessionPolicy) -> pd.Timestamp:
    return pd.Timestamp(stamp).tz_convert(policy.timezone)


def _after_entry_window(as_of: pd.Timestamp, settings: TjrSettings, policy: SessionPolicy) -> bool:
    """Whether a new entry is barred because the session's entry window has closed.

    On the equity policy this is the 15:30 ET cut the strategy has always used. On a
    24/7 crypto policy there is no session close, so the same *local clock* cut
    applies to the UTC day (or the policy's own timezone); a policy with no entry
    cut would need a new, documented knob rather than a silent change, so the cut is
    kept and stated.
    """
    local = _local_time(as_of, policy)
    hour, minute = settings.latest_entry_time
    return (local.hour, local.minute) >= (hour, minute)


def find_sweeps(
    frame_5m: pd.DataFrame,
    levels: list[LiquidityLevel],
    *,
    as_of: pd.Timestamp,
    settings: TjrSettings,
) -> list[Sweep]:
    """Every sweep in the recent window, most recent first.

    A bearish sweep takes a *high* out: the bar's high is above the level and its
    close is back below. A bullish sweep takes a *low* out: the low is below the
    level and the close is back above. Two kinds of level qualify:

    - a **static** level (a previous session, 1h or 4h extreme, or a confirmed
      swing), which must already have formed before the bar; and
    - the **running session extreme before the bar** — the current session's own
      high or low as it stood *before* this bar. This is what lets the first sweep
      of a session's own range be caught: the level is the prior extreme, not the
      bar's own print.

    All of them are returned rather than only the latest, because a session can
    sweep several levels and only one of them may go on to confirm. The caller
    chooses; picking the latest blindly would let a small sweep made during a
    retrace replace the sweep that is actually setting up.
    """
    if frame_5m.empty:
        return []
    window = frame_5m.tail(settings.sweep_lookback_bars)
    running_high = running_extreme_before(window, kind="high")
    running_low = running_extreme_before(window, kind="low")
    found: list[Sweep] = []
    for stamp, row in window.iterrows():
        # Candidates are typed by kind: a *high* level can only be taken out by a
        # bearish sweep, a *low* level only by a bullish one. Mixing them would let
        # an ordinary down bar "sweep" a low that price is simply falling through.
        highs: list[tuple[float, str]] = []
        lows: list[tuple[float, str]] = []
        for level in levels:
            if level.formed_at <= stamp:
                (highs if level.kind == "high" else lows).append((level.price, level.label))
        prior_high = running_high.loc[stamp]
        prior_low = running_low.loc[stamp]
        if prior_high == prior_high:  # not NaN
            highs.append((float(prior_high), "session_high"))
        if prior_low == prior_low:
            lows.append((float(prior_low), "session_low"))
        for price, label in highs:
            if row["high"] > price and row["close"] < price:
                found.append(
                    Sweep(
                        level=_synthetic_level(label, "high", price, stamp),
                        bar=stamp,
                        extreme=float(row["high"]),
                        direction="bearish",
                    )
                )
        for price, label in lows:
            if row["low"] < price and row["close"] > price:
                found.append(
                    Sweep(
                        level=_synthetic_level(label, "low", price, stamp),
                        bar=stamp,
                        extreme=float(row["low"]),
                        direction="bullish",
                    )
                )
    found.sort(key=lambda sweep: sweep.bar, reverse=True)
    return found


def find_sweep(
    frame_5m: pd.DataFrame,
    levels: list[LiquidityLevel],
    *,
    as_of: pd.Timestamp,
    settings: TjrSettings,
) -> Sweep | None:
    """The most recent sweep, or ``None``. See :func:`find_sweeps`."""
    sweeps = find_sweeps(frame_5m, levels, as_of=as_of, settings=settings)
    return sweeps[0] if sweeps else None


def _synthetic_level(label: str, kind: str, price: float, stamp: pd.Timestamp) -> LiquidityLevel:
    """A level object for a running extreme, for a sweep's provenance."""
    return LiquidityLevel(label=label, kind=kind, price=float(price), formed_at=stamp)


def _confirmation(
    frame_5m: pd.DataFrame, sweep: Sweep, *, as_of: pd.Timestamp, settings: TjrSettings
) -> tuple[str | None, pd.Timestamp | None]:
    """The 5m reversal confirmation after the sweep: BOS or IFVG, or neither.

    The break-of-structure reference is the most recent swing *as it stood at the
    sweep bar*, fixed there. Recomputing it per bar would let a swing that confirms
    later move the reference and un-confirm a setup that had already confirmed — the
    structure being broken is the one the sweep came from, not whatever has formed
    since. The inverse-gap test is anchored the same way, to flips at or after the
    sweep.
    """
    direction = "down" if sweep.direction == "bearish" else "up"
    swings = swing_points(frame_5m, lookback=settings.swing_lookback)
    reference = last_swing(swings, "high" if direction == "up" else "low", at=sweep.bar)
    if reference is not None:
        later = frame_5m[frame_5m.index > sweep.bar]
        if direction == "up":
            breaks = later[later["close"] > reference.price]
        else:
            breaks = later[later["close"] < reference.price]
        if len(breaks):
            return "BOS", breaks.index[0]
    ifvg = inverse_fair_value_gap(frame_5m, direction=direction, since=sweep.bar)
    if ifvg is not None:
        return "IFVG", ifvg.flipped_at
    return None, None


def _continuation_zone(
    frame_5m: pd.DataFrame,
    sweep: Sweep,
    *,
    confirmed_at: pd.Timestamp,
    settings: TjrSettings,
) -> tuple[tuple[float, float] | None, str | None]:
    """The 5m continuation zone: a fair value gap, or the sweep leg's equilibrium.

    The leg runs from the sweep bar to the **confirmation bar** — the impulse that
    produced the break — and is frozen there. Freezing it matters: the equilibrium
    of a leg that is still growing moves with every bar, and price is always near
    the recent midpoint, which would make "price is in the zone" true almost
    immediately. The frozen leg's 50% is the equilibrium TJR names, and the first
    gap on that leg is the FVG zone. The FVG is preferred when one exists — TJR's own
    examples all enter on a gap fill — and the equilibrium is the fallback when the
    leg left no imbalance to retrace into.
    """
    leg = frame_5m[(frame_5m.index >= sweep.bar) & (frame_5m.index <= confirmed_at)]
    if leg.empty:
        return None, None
    # The leg's origin is the sweep bar's *close* — where price closed back inside the
    # level, i.e. where the reversal is established — not the sweep's extreme. Using
    # the extreme would put the 50% far below the recovery and demand an unrealistically
    # deep retrace.
    origin = float(frame_5m.loc[sweep.bar, "close"])
    if sweep.direction == "bearish":
        end = float(leg["low"].min())
        leg_low, leg_high = end, origin
        gap_kind = "bear"
    else:
        end = float(leg["high"].max())
        leg_low, leg_high = origin, end
        gap_kind = "bull"
    if leg_high <= leg_low:
        return None, None
    gaps = fair_value_gaps(leg, kind=gap_kind)
    if gaps:
        narrowest = min(gaps, key=lambda g: g.high - g.low)
        return (narrowest.low, narrowest.high), "fvg"
    return (equilibrium(leg_low, leg_high),) * 2, "equilibrium"


def _after_bar(frame: pd.DataFrame, stamp: pd.Timestamp) -> pd.Timestamp:
    """The index of the bar after ``stamp``, or ``stamp`` itself if it is the last.

    Used so a retrace must happen on a *later* bar than the one that confirmed the
    setup: the confirmation bar's own range is part of the impulse, not a retrace
    into it.
    """
    later = frame.index[frame.index > stamp]
    return later[0] if len(later) else stamp


def _retrace_at(
    frame_5m: pd.DataFrame, zone: tuple[float, float], *, since: pd.Timestamp
) -> pd.Timestamp | None:
    """The first bar at or after ``since`` whose range overlaps ``zone``, or ``None``."""
    later = frame_5m[frame_5m.index >= since]
    if later.empty:
        return None
    low, high = zone
    overlaps = (later["low"] <= high) & (later["high"] >= low)
    hits = later.index[overlaps]
    return hits[0] if len(hits) else None


def _entry_trigger(
    frame_1m: pd.DataFrame, *, direction: str, since: pd.Timestamp, settings: TjrSettings
) -> str | None:
    """The 1m entry confirmation: BOS or IFVG in the trade direction."""
    if frame_1m.empty:
        return None
    swings = swing_points(frame_1m, lookback=settings.swing_lookback)
    if break_of_structure(frame_1m, swings, direction=direction, since=since) is not None:
        return "BOS"
    if inverse_fair_value_gap(frame_1m, direction=direction, since=since) is not None:
        return "IFVG"
    return None


def _levels_and_stop(
    frame_5m: pd.DataFrame, sweep: Sweep, *, settings: TjrSettings
) -> tuple[float, float]:
    """The invalidation stop for a sweep setup, and the leg extreme it sits beyond.

    TJR places the stop beyond the swing that would invalidate the idea — for a
    sweep setup that is the sweep's own extreme, widened by a small buffer so a
    marginal re-test does not clip it.
    """
    if sweep.direction == "bearish":
        return sweep.extreme * (1 + settings.stop_buffer), sweep.extreme
    return sweep.extreme * (1 - settings.stop_buffer), sweep.extreme


def evaluate(
    *,
    symbol: str,
    session: str,
    as_of: pd.Timestamp,
    series_5m: IntradaySeries,
    series_1m: IntradaySeries,
    hourly: IntradaySeries | None = None,
    four_hour: IntradaySeries | None = None,
    settings: TjrSettings | None = None,
    allow_short: bool = False,
    session_policy: SessionPolicy | None = None,
) -> TjrEvaluation:
    """Run the whole TJR chain at ``as_of`` and report the outcome.

    Pure: it reads only bars that had closed by ``as_of``, computes every level and
    structure fresh, and returns the stage reached with a reason code. It never
    reads a later bar, so calling it in a loop reproduces the same decision for the
    same instant and a re-run is deterministic.

    A session can sweep more than one level, so every recent sweep is tried and the
    most *progressed* setup wins — a completed entry beats a bare sweep, whatever
    their order. Ties go to the more recent sweep. This is what keeps a small sweep
    made during a retrace from replacing the sweep that is actually setting up.

    ``session_policy`` labels the session day and bounds its window. It defaults to
    the series' own policy (the US equity session for an equity series, the crypto
    policy for a crypto one), so a caller that built its series with a policy need
    not pass it again. It changes *which* bars a session's structure is read from,
    never whether a rule fires.
    """
    settings = settings or TjrSettings()
    policy = session_policy or series_5m.session
    reset_frame_cache()
    try:
        return _evaluate_inner(
            symbol=symbol,
            session=session,
            as_of=as_of,
            series_5m=series_5m,
            series_1m=series_1m,
            hourly=hourly,
            four_hour=four_hour,
            settings=settings,
            allow_short=allow_short,
            policy=policy,
        )
    finally:
        reset_frame_cache()


def _evaluate_inner(
    *,
    symbol: str,
    session: str,
    as_of: pd.Timestamp,
    series_5m: IntradaySeries,
    series_1m: IntradaySeries,
    hourly: IntradaySeries | None,
    four_hour: IntradaySeries | None,
    settings: TjrSettings,
    allow_short: bool,
    policy: SessionPolicy,
) -> TjrEvaluation:
    frame_5m = series_5m.session_bars(session, as_of)
    frame_1m = series_1m.session_bars(session, as_of)
    if frame_5m.empty or frame_1m.empty:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_NONE,
            reason_code=TJR_OUTSIDE_SESSION,
            detail="no in-session bars had closed yet",
        )
    if _after_entry_window(as_of, settings, policy):
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_NONE,
            reason_code=TJR_ENTRY_WINDOW_CLOSED,
            detail="the entry window for this session has closed",
        )

    levels = liquidity_levels(
        session=session,
        as_of=as_of,
        base=series_5m,
        hourly=hourly,
        four_hour=four_hour,
        swing_lookback=settings.swing_lookback,
        equal_tolerance=settings.equal_tolerance,
    )
    sweeps = find_sweeps(frame_5m, levels, as_of=as_of, settings=settings)
    if not sweeps:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_NONE,
            reason_code=TJR_NO_SWEEP,
            detail="no level was swept and closed back through",
            levels=levels,
        )
    # Ties in progress go to the earlier (more-established) sweep, so a sweep made
    # while price chops around a retrace cannot displace the one that is setting up.
    order = {id(sweep): index for index, sweep in enumerate(sweeps)}
    evaluations = [
        _evaluate_sweep(
            symbol=symbol,
            session=session,
            as_of=as_of,
            sweep=sweep,
            frame_5m=frame_5m,
            frame_1m=frame_1m,
            levels=levels,
            settings=settings,
            allow_short=allow_short,
        )
        for sweep in sweeps
    ]
    evaluations.sort(key=lambda item: (-_STAGE_RANK[item.stage], order[id(item.sweep)]))
    return evaluations[0]


_STAGE_RANK: dict[str, int] = {
    STAGE_NONE: 0,
    STAGE_SWEPT: 1,
    STAGE_CONFIRMED: 2,
    STAGE_IN_ZONE: 3,
    STAGE_READY: 4,
}


def _evaluate_sweep(
    *,
    symbol: str,
    session: str,
    as_of: pd.Timestamp,
    sweep: Sweep,
    frame_5m: pd.DataFrame,
    frame_1m: pd.DataFrame,
    levels: list[LiquidityLevel],
    settings: TjrSettings,
    allow_short: bool,
) -> TjrEvaluation:
    """Take one candidate sweep through confirmation, retrace and entry."""
    confirmation, confirmed_at = _confirmation(frame_5m, sweep, as_of=as_of, settings=settings)
    if confirmation is None or confirmed_at is None:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_SWEPT,
            reason_code=TJR_NO_5M_CONFIRMATION,
            detail="the sweep has no 5m BOS or IFVG yet",
            sweep=sweep,
            levels=levels,
        )

    zone, zone_label = _continuation_zone(
        frame_5m, sweep, confirmed_at=confirmed_at, settings=settings
    )
    if zone is None:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_CONFIRMED,
            reason_code=TJR_NO_CONTINUATION,
            detail="no 5m gap or equilibrium to retrace into",
            sweep=sweep,
            confirmation=confirmation,
            levels=levels,
        )
    retrace = _retrace_at(frame_5m, zone, since=_after_bar(frame_5m, confirmed_at))
    if retrace is None:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_CONFIRMED,
            reason_code=TJR_NO_CONTINUATION,
            detail=f"price has not retraced into the {zone_label} zone yet",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )

    direction = "down" if sweep.direction == "bearish" else "up"
    trigger = _entry_trigger(frame_1m, direction=direction, since=retrace, settings=settings)
    if trigger is None:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_IN_ZONE,
            reason_code=TJR_NO_1M_ENTRY,
            detail="price is in the zone but the 1m confirmation has not printed",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )

    side = "short" if sweep.direction == "bearish" else "long"
    if side == "short" and not allow_short:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_READY,
            reason_code=TJR_SHORT_SETUP_LONG_ONLY,
            detail="a complete bearish setup, but the book is long-only",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )

    stop, _ = _levels_and_stop(frame_5m, sweep, settings=settings)
    entry = float(frame_1m["close"].iloc[-1])
    # A long needs its stop below the entry and a short above it. Real bars can put
    # the sweep's extreme on the wrong side of the entry: price makes a new low
    # *after* the sweep, so a long's stop would sit above the price it is protecting.
    # That is not a wide stop, it is an inverted one, and it must never become an
    # order — the contract refuses it, and the strategy must refuse it first.
    if (side == "long" and stop >= entry) or (side == "short" and stop <= entry):
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_IN_ZONE,
            reason_code=TJR_INVALID_GEOMETRY,
            detail="the invalidation stop is on the wrong side of the entry",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )
    impulse = _impulse_extreme(frame_5m, sweep, confirmed_at)
    ladder = _target_ladder(
        side,
        entry=entry,
        stop=stop,
        levels=levels,
        settings=settings,
        formed_by=sweep.bar,
        impulse=impulse,
    )
    target = ladder[0] if ladder else None
    if target is None:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_IN_ZONE,
            reason_code=TJR_NO_CONTINUATION,
            detail="no draw on liquidity beyond the entry to target",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0 or reward / risk < settings.min_reward_risk:
        return TjrEvaluation(
            symbol=symbol,
            session=session,
            as_of=as_of,
            side=None,
            stage=STAGE_IN_ZONE,
            reason_code=TJR_NO_CONTINUATION,
            detail=f"the nearest draw gives only {reward / risk if risk else 0:.2f}R",
            sweep=sweep,
            confirmation=confirmation,
            zone=zone,
            levels=levels,
        )

    return TjrEvaluation(
        symbol=symbol,
        session=session,
        as_of=as_of,
        side=side,
        stage=STAGE_READY,
        reason_code=TJR_READY,
        detail=f"{sweep.direction} sweep confirmed by 5m {confirmation}, 1m {trigger}",
        entry=entry,
        stop=stop,
        target=target,
        targets=list(ladder),
        sweep=sweep,
        confirmation=confirmation,
        zone=zone,
        levels=levels,
    )


def _impulse_extreme(frame_5m: pd.DataFrame, sweep: Sweep, confirmed_at: pd.Timestamp) -> float:
    """The extreme the impulse reached by the confirmation bar.

    The target has to sit beyond this, or it is a level the move already consumed
    rather than a draw price is still being pulled toward.
    """
    leg = frame_5m[(frame_5m.index >= sweep.bar) & (frame_5m.index <= confirmed_at)]
    if sweep.direction == "bearish":
        return float(leg["low"].min()) if not leg.empty else sweep.extreme
    return float(leg["high"].max()) if not leg.empty else sweep.extreme


def _target(
    side: str,
    *,
    entry: float,
    stop: float,
    levels: list[LiquidityLevel],
    settings: TjrSettings,
    formed_by: pd.Timestamp,
    impulse: float,
) -> float | None:
    """The exit: the next draw on liquidity beyond the entry, or a fallback R.

    For a long, the nearest level above the entry; for a short, the nearest below.
    Only levels that predate the sweep count, and the level must sit beyond the
    impulse's own extreme, so the entry's move cannot become its own target. When no
    such draw exists, a fixed multiple of the risk is used, so a setup is never taken
    without a defined exit.
    """
    ladder = _target_ladder(
        side,
        entry=entry,
        stop=stop,
        levels=levels,
        settings=settings,
        formed_by=formed_by,
        impulse=impulse,
    )
    return ladder[0] if ladder else None


def _target_ladder(
    side: str,
    *,
    entry: float,
    stop: float,
    levels: list[LiquidityLevel],
    settings: TjrSettings,
    formed_by: pd.Timestamp,
    impulse: float,
    max_targets: int = 3,
) -> list[float]:
    """The ladder of draws on liquidity beyond the entry, nearest first.

    The same rule as :func:`_target`, applied repeatedly: each level must sit
    beyond the **previous** one, so the ladder is strictly ordered away from
    the entry and no two targets are the same price.

    How many targets there are is the **market's** decision, not the
    strategy's. A setup with one draw on liquidity has one target; a setup
    with a draw, a higher swing high and an equal-high cluster above it has
    three. The ladder is never padded: if only one level supports a target,
    there is one target, and the fallback R is used only when **no** draw
    exists at all.

    The fallback is not repeated. When no draw exists, one target is derived
    from the risk multiple; inventing a second and third target at fixed R
    multiples would be a fabricated ladder, not market structure.
    """
    risk = abs(entry - stop)
    eligible = [level for level in levels if level.formed_at <= formed_by]
    if side == "long":
        beyond = sorted(
            (level for level in eligible if level.price > max(entry, impulse)),
            key=lambda level: level.price,
        )
        ladder: list[float] = []
        last = max(entry, impulse)
        for level in beyond:
            if level.price <= last:
                continue
            ladder.append(level.price)
            last = level.price
            if len(ladder) >= max_targets:
                break
        if ladder:
            return ladder
        return [entry + settings.fallback_target_r * risk] if risk > 0 else []
    beyond = sorted(
        (level for level in eligible if level.price < min(entry, impulse)),
        key=lambda level: level.price,
        reverse=True,
    )
    ladder = []
    last = min(entry, impulse)
    for level in beyond:
        if level.price >= last:
            continue
        ladder.append(level.price)
        last = level.price
        if len(ladder) >= max_targets:
            break
    if ladder:
        return ladder
    return [entry - settings.fallback_target_r * risk] if risk > 0 else []
