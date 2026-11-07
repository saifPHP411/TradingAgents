"""The session clock a TJR run reads, made explicit and pluggable.

TJR's rule chain is a sequence of arithmetic checks on intraday bars, but several
of its inputs are *session* quantities: the day a bar belongs to, the running
session high/low, the previous session's high/low, and the last instant a new
entry may be taken. On a US equity those are fixed (09:30–16:00 ET, Monday to
Friday). On a 24/7 crypto market there is no opening bell, so "the session" has to
be *defined* rather than assumed.

This module makes that definition an object — :class:`SessionPolicy` — instead of
module-level constants, so the same unchanged rule chain can run on equities and
on crypto. The equity policy reproduces the original behaviour exactly (the
``America/New_York`` calendar day and the 09:30–16:00 regular session), so every
existing TJR result is unchanged; the crypto policies define the sessions a
24/7 market actually has.

A policy answers four questions, and nothing else:

- which session a UTC instant belongs to (:meth:`SessionPolicy.session_of`);
- the UTC bounds of a named session (:meth:`SessionPolicy.session_bounds`);
- whether a bar-start instant is inside a session's window
  (:meth:`SessionPolicy.contains`);
- the local instant a coarse bar is bucketed from, so a 1h/4h level is anchored
  on the session rather than on the top of the UTC hour
  (:meth:`SessionPolicy.anchor`).

Nothing here decides a trade. A policy only labels time, so changing it can move
*which* bars a session's structure is built from, never whether a rule fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

Window = tuple[tuple[int, int], tuple[int, int]]
"""A trading window as ``((start_hour, start_minute), (end_hour, end_minute))``.

``(24, 0)`` as an end means midnight at the end of the local day, i.e. a window
that runs to the day's close, which is how a 24/7 session is written.
"""


@dataclass(frozen=True)
class SessionPolicy:
    """How a strategy's sessions are labelled and bounded in time.

    ``timezone`` is the zone whose calendar day names a session and whose clock the
    windows are written in. ``windows`` is the ordered set of trading windows inside
    that day; a bar-start is *in* the session when it falls in any window. An empty
    ``windows`` means the whole local day is the session (a 24/7 market).

    ``label`` is a short, stable name for the policy, carried into reports and the
    dashboard so a run states which session definition it used.
    """

    label: str
    timezone: str
    windows: tuple[Window, ...] = ()
    """Trading windows in ``timezone``. Empty means the whole local day."""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a session policy needs a label")
        for start, end in self.windows:
            if start >= end:
                raise ValueError(f"window {start} does not start before {end}")

    # -- naming -----------------------------------------------------------

    def session_of(self, stamp: pd.Timestamp) -> str:
        """The ``YYYY-MM-DD`` session label a UTC instant belongs to.

        The local calendar day, so a bar just after local midnight belongs to the
        new day even though it is still the previous UTC day. For equities this is
        the behaviour the strategy has always had; for a UTC crypto policy it is
        the UTC day.
        """
        local = pd.Timestamp(stamp).tz_convert(self.timezone)
        return local.strftime("%Y-%m-%d")

    # -- bounds -----------------------------------------------------------

    def session_bounds(self, session: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The session's first window open and last window close, in UTC.

        A 24/7 policy spans the whole local day: midnight to the next midnight.
        """
        day = pd.Timestamp(session, tz=self.timezone)
        if not self.windows:
            return day.tz_convert("UTC"), (day + pd.Timedelta(days=1)).tz_convert("UTC")
        first = self.windows[0][0]
        last = self.windows[-1][1]
        open_local = day + pd.Timedelta(hours=first[0], minutes=first[1])
        if last == (24, 0):
            close_local = day + pd.Timedelta(days=1)
        else:
            close_local = day + pd.Timedelta(hours=last[0], minutes=last[1])
        return open_local.tz_convert("UTC"), close_local.tz_convert("UTC")

    def session_open_utc(self, session: str) -> pd.Timestamp:
        return self.session_bounds(session)[0]

    def session_close_utc(self, session: str) -> pd.Timestamp:
        return self.session_bounds(session)[1]

    # -- membership -------------------------------------------------------

    def contains(self, stamp: pd.Timestamp, session: str) -> bool:
        """Whether a UTC bar-start instant is inside ``session``'s windows."""
        local = pd.Timestamp(stamp).tz_convert(self.timezone)
        if local.strftime("%Y-%m-%d") != session:
            return False
        if not self.windows:
            return True
        clock = local.hour * 60 + local.minute
        return any(
            start[0] * 60 + start[1] <= clock < end[0] * 60 + end[1] for start, end in self.windows
        )

    def index_mask(self, index: pd.DatetimeIndex, session: str) -> pd.Series:
        """A boolean mask of which bar-starts in ``index`` fall inside ``session``.

        The vectorized form of :meth:`contains`, for the hot path where a whole
        frame is filtered at every instant. Same answer, one pass, no per-bar Python.
        """
        local = index.tz_convert(self.timezone)
        mask = pd.Series(local.strftime("%Y-%m-%d") == session, index=index)
        if not self.windows:
            return mask
        clock = pd.Series(local.hour * 60 + local.minute, index=index)
        window_mask = pd.Series(False, index=index)
        for start, end in self.windows:
            window_mask |= (clock >= start[0] * 60 + start[1]) & (clock < end[0] * 60 + end[1])
        return mask & window_mask

    def anchor(self, session: str) -> pd.Timestamp:
        """The local instant a coarse bar is bucketed from.

        Resampling anchors on the session open, so a 4h level is "the first 4h of
        the session" rather than a bucket aligned to the UTC clock. A 24/7 policy
        anchors on local midnight.
        """
        day = pd.Timestamp(session, tz=self.timezone)
        if not self.windows:
            return day
        first = self.windows[0][0]
        return day + pd.Timedelta(hours=first[0], minutes=first[1])

    def local(self, stamp: pd.Timestamp) -> pd.Timestamp:
        """``stamp`` in the policy's timezone, for a clock-time comparison."""
        return pd.Timestamp(stamp).tz_convert(self.timezone)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "timezone": self.timezone,
            "windows": [[list(start), list(end)] for start, end in self.windows],
        }


# The US equity regular session: the 09:30–16:00 ET window on the New York
# calendar day. This is the policy the original TJR implementation hard-coded, so
# it is the default everywhere and no existing result moves.
EQUITY_SESSION = SessionPolicy(
    label="us_equity", timezone="America/New_York", windows=(((9, 30), (16, 0)),)
)

# Named crypto session windows, in UTC, for a study that wants London / New York /
# Asia structure instead of a flat 24h day. These are the conventional FX-hours
# windows, stated explicitly rather than assumed.
LONDON_WINDOW: Window = ((7, 0), (16, 0))
NEW_YORK_WINDOW: Window = ((13, 0), (21, 0))
ASIA_WINDOW: Window = ((0, 0), (9, 0))


def utc_day_session() -> SessionPolicy:
    """The whole UTC calendar day as one session — the crypto 24/7 default.

    Crypto never closes, so there is no opening bell to mark; the honest default
    is the UTC day, which is the one boundary every exchange and every vendor
    agrees on. It is *not* presented as a market open — it is the period a day's
    structure is measured over.
    """
    return SessionPolicy(label="utc_day", timezone="UTC", windows=())


def crypto_named_session() -> SessionPolicy:
    """A crypto policy built from the London / New York / Asia windows.

    A bar counts as in-session when it falls in any of the three windows, so the
    session spans the liquidity hours and the quiet hours between them are
    excluded from the session's structure. Used for the session-sensitivity study,
    never as the default.
    """
    return SessionPolicy(
        label="crypto_london_ny_asia",
        timezone="UTC",
        windows=(ASIA_WINDOW, LONDON_WINDOW, NEW_YORK_WINDOW),
    )


def policy_named(name: str) -> SessionPolicy:
    """A session policy by its short name, or raise.

    The closed vocabulary a CLI flag or a config selects from, so an unknown name
    is a loud error rather than a silent fall-back to a different session clock.
    """
    table = {
        "us_equity": EQUITY_SESSION,
        "utc_day": utc_day_session(),
        "crypto_named": crypto_named_session(),
    }
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"unknown session policy {name!r}; choose one of {', '.join(sorted(table))}"
        ) from None


__all__ = [
    "ASIA_WINDOW",
    "EQUITY_SESSION",
    "LONDON_WINDOW",
    "NEW_YORK_WINDOW",
    "SessionPolicy",
    "Window",
    "crypto_named_session",
    "policy_named",
    "utc_day_session",
]
