"""An explicit, honest freshness view of a paper experiment's market data.

The reported failure this module answers is a system that cannot tell an operator
*why* nothing is happening, and that could imply its data is current when it is
not. A daily vendor source is not a live price feed: it has a newest *session*, and
between sessions there is legitimately nothing new to process. That is not a
fault, and it must not be reported as one — nor must a source whose newest bar has
gone stale be allowed to read as "current".

The view is derived only from facts already recorded (the source provenance, the
refresh sidecar, and the processed-session set); it fetches nothing. The statuses
are a closed vocabulary, and each names the *data* condition rather than claiming
knowledge of an exchange clock the system does not have:

- ``historical`` — the source is a frozen/static dataset, not the current market.
- ``no_data`` — the source carries no sessions at all.
- ``stale`` — a freshness rule exists and the newest bar is older than it.
- ``up_to_date`` — the newest available session has been processed.
- ``pending_sessions`` — newer sessions are available but not yet processed.

``data_age_days`` is measured against a supplied reference date (never a hidden
clock in the arithmetic), so a test can pin it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

MARKET_STATUS_HISTORICAL = "historical"
MARKET_STATUS_NO_DATA = "no_data"
MARKET_STATUS_STALE = "stale"
MARKET_STATUS_UP_TO_DATE = "up_to_date"
MARKET_STATUS_PENDING = "pending_sessions"

MARKET_STATUSES = (
    MARKET_STATUS_HISTORICAL,
    MARKET_STATUS_NO_DATA,
    MARKET_STATUS_STALE,
    MARKET_STATUS_UP_TO_DATE,
    MARKET_STATUS_PENDING,
)


def _age_days(latest: str | None, reference: date) -> int | None:
    if latest is None:
        return None
    try:
        bar = date.fromisoformat(str(latest)[:10])
    except ValueError:
        return None
    return (reference - bar).days


@dataclass(frozen=True)
class MarketFreshness:
    """The freshness facts a report or dashboard shows, with no live fetch."""

    market_status: str
    reason: str
    latest_available_session: str | None
    last_source_bar: str | None
    last_processed_session: str | None
    last_refresh_at: str | None
    next_refresh_allowed_at: str | None
    data_age_days: int | None
    pending_sessions: int
    is_stale: bool
    fetches_today: int
    fetch_quota_per_day: int
    reference_date: str

    def to_dict(self) -> dict:
        return {
            "market_status": self.market_status,
            "reason": self.reason,
            "latest_available_session": self.latest_available_session,
            "last_source_bar": self.last_source_bar,
            "last_processed_session": self.last_processed_session,
            "last_refresh_at": self.last_refresh_at,
            "next_refresh_allowed_at": self.next_refresh_allowed_at,
            "data_age_days": self.data_age_days,
            "pending_sessions": self.pending_sessions,
            "is_stale": self.is_stale,
            "fetches_today": self.fetches_today,
            "fetch_quota_per_day": self.fetch_quota_per_day,
            "reference_date": self.reference_date,
        }


def market_freshness(experiment, *, policy=None, now: datetime | None = None) -> MarketFreshness:
    """Build the freshness view for ``experiment`` without contacting any source.

    ``policy`` supplies the refresh throttle so ``next_refresh_allowed_at`` can be
    computed; when omitted the default :class:`RefreshPolicy` is used, which is the
    same default the runner applies.
    """
    from .refresh import RefreshPolicy, read_refresh_state
    from .sources import bar_is_fresh

    if now is None:
        now = datetime.now(UTC)
    policy = policy or RefreshPolicy()
    state = read_refresh_state(experiment.root)

    sessions = list(experiment.source.sessions())
    latest_available = sessions[-1] if sessions else None
    last_source_bar = state.last_bar_date or latest_available
    processed = sorted(experiment.processed_sessions)
    last_processed = processed[-1] if processed else None
    pending = sum(1 for s in sessions if s not in experiment.processed_sessions)

    provenance = getattr(experiment.source, "provenance", None)
    max_stale = getattr(provenance, "max_stale_days", None)
    is_current_market = bool(getattr(experiment.source, "is_current_market", False))
    age = _age_days(latest_available, now.date())

    next_allowed = None
    if state.last_attempt_at:
        try:
            elapsed = (now - datetime.fromisoformat(state.last_attempt_at)).total_seconds()
        except ValueError:
            elapsed = policy.min_interval_seconds
        if elapsed < policy.min_interval_seconds:
            from datetime import timedelta

            next_allowed = (
                datetime.fromisoformat(state.last_attempt_at)
                + timedelta(seconds=policy.min_interval_seconds)
            ).isoformat()

    # Decide the status in priority order, so the most important fact wins.
    if not sessions:
        status, reason = MARKET_STATUS_NO_DATA, "the price source has no sessions"
        is_stale = False
    elif not is_current_market and max_stale is None:
        status, reason = (
            MARKET_STATUS_HISTORICAL,
            "the source is a historical/reproducible dataset, not the current market",
        )
        is_stale = False
    elif max_stale is not None and not bar_is_fresh(
        latest_available, reference=now.date(), max_stale_days=max_stale
    ):
        status = MARKET_STATUS_STALE
        reason = (
            f"the newest bar {latest_available} is {age} day(s) old, beyond the "
            f"source's {max_stale} day freshness limit"
        )
        is_stale = True
    elif pending:
        status = MARKET_STATUS_PENDING
        reason = f"{pending} newer session(s) available and not yet processed"
        is_stale = False
    else:
        status = MARKET_STATUS_UP_TO_DATE
        reason = (
            f"latest available session {latest_available} has been processed; "
            "no newer market session is available"
        )
        is_stale = False

    return MarketFreshness(
        market_status=status,
        reason=reason,
        latest_available_session=latest_available,
        last_source_bar=last_source_bar,
        last_processed_session=last_processed,
        last_refresh_at=state.last_success_at or state.last_attempt_at,
        next_refresh_allowed_at=next_allowed,
        data_age_days=age,
        pending_sessions=pending,
        is_stale=is_stale,
        fetches_today=state.fetches_today,
        fetch_quota_per_day=policy.max_fetches_per_day,
        reference_date=now.date().isoformat(),
    )
