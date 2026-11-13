"""Appending new vendor sessions to a running experiment, in place.

A vendor daily source fetches once and freezes its frame. That is what makes a
resume cheap and reproducible, but it also means the experiment's session set
never grows on its own: once every session of the original fetch is processed,
``pending_sessions`` is empty for good and the engine has nothing to advance. This
module is the missing step — it asks the vendor for its current daily series and
*appends* any session strictly newer than the experiment already holds.

The rules are chosen so a refresh can never corrupt the accounting or the run's
identity:

- **Same experiment.** The bars are merged into the existing source and the
  experiment is saved through its normal ``save_state``, so ``experiment_id``,
  ``created_at``, ``processed_sessions``, ``decisions`` and the session book are
  all preserved. The journal is appended to, never rewritten. Nothing is deleted.
- **Append-only, forward.** Only sessions strictly newer than the source's current
  last session are added. A backfilled older session is ignored rather than
  inserted, because inserting a past session would let the engine settle the book
  against a bar it has already passed — exactly the reordering ``process_session``
  refuses. A session the experiment has already processed is never re-added.
- **No silent restatement.** A session present in both the old and the new frame
  must carry identical OHLCV. If the vendor restates a bar the run already used,
  the refresh is refused (``conflict``) rather than allowed to change a number the
  ledger has already booked.
- **Quota-aware.** The free Alpha Vantage plan allows 25 requests/day, so a fetch
  is throttled by a minimum interval and a per-UTC-day cap. The counters live in a
  small sidecar file (``data_refresh.json``), so the recovery file is only ever
  grown, never rewritten for bookkeeping.
- **Never fatal.** A quota notice, a stale feed, a network error or a malformed
  body is recorded as a status and returned, not raised: a refresh failure must
  stop new sessions from arriving, not crash the engine that is already running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from .sources import (
    BAR_COLUMNS,
    FramePriceSource,
    PriceSource,
    PriceSourceError,
    _validate_frame,
    bar_is_fresh,
)

REFRESH_STATE_FILENAME = "data_refresh.json"
REFRESH_SCHEMA_VERSION = 1

# Closed vocabulary of refresh outcomes. A caller branches on these, never on prose.
STATUS_DISABLED = "disabled"
STATUS_NOT_REFRESHABLE = "not_refreshable"
STATUS_THROTTLED = "throttled"
STATUS_QUOTA_GUARD = "quota_guard"
STATUS_APPENDED = "appended"
STATUS_NO_NEW_SESSIONS = "no_new_sessions"
STATUS_UNAVAILABLE = "unavailable"
STATUS_AUTH_FAILED = "auth_failed"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_CONFLICT = "conflict"
# Recorded in the sidecar by `start`, not returned by a refresh: the initial vendor
# fetch, which counts against the daily quota just as a refresh does.
STATUS_INITIAL_FETCH = "initial_fetch"

REFRESH_STATUSES = frozenset(
    {
        STATUS_DISABLED,
        STATUS_NOT_REFRESHABLE,
        STATUS_THROTTLED,
        STATUS_QUOTA_GUARD,
        STATUS_APPENDED,
        STATUS_NO_NEW_SESSIONS,
        STATUS_UNAVAILABLE,
        STATUS_AUTH_FAILED,
        STATUS_NOT_CONFIGURED,
        STATUS_CONFLICT,
        STATUS_INITIAL_FETCH,
    }
)

# A skip is a tick that decided *not* to spend a vendor call. It is transient and
# must never overwrite the last real outcome: an operator who fixed a key would
# otherwise see a bare "throttled" forever and never learn the fetch still fails.
SKIP_STATUSES = frozenset({STATUS_THROTTLED, STATUS_QUOTA_GUARD})

# The outcomes that mean "the source could not serve new data". This is the
# umbrella the dashboard and freshness view report as DATA_SOURCE_UNAVAILABLE, so a
# failing provider is never confused with "no newer bar exists".
SOURCE_FAILURE_STATUSES = frozenset(
    {STATUS_UNAVAILABLE, STATUS_AUTH_FAILED, STATUS_NOT_CONFIGURED}
)

# The finer reason behind a failure, so an authentication fault, a missing key, a
# vendor rate limit and a transport error are never collapsed into one label.
FAILURE_KIND_AUTH = "auth"
FAILURE_KIND_NOT_CONFIGURED = "not_configured"
FAILURE_KIND_RATE_LIMITED = "rate_limited"
FAILURE_KIND_STALE = "stale"
FAILURE_KIND_TRANSPORT = "transport"
FAILURE_KIND_MALFORMED = "malformed"
FAILURE_KIND_UNSUPPORTED = "unsupported"

FAILURE_KINDS = frozenset(
    {
        FAILURE_KIND_AUTH,
        FAILURE_KIND_NOT_CONFIGURED,
        FAILURE_KIND_RATE_LIMITED,
        FAILURE_KIND_STALE,
        FAILURE_KIND_TRANSPORT,
        FAILURE_KIND_MALFORMED,
        FAILURE_KIND_UNSUPPORTED,
    }
)

# The source kind a refresh can extend: a vendor's historical daily bars. A frozen
# dataset or a test fixture has no vendor to ask; a quote-only source has no bar
# history to append to.
REFRESHABLE_KINDS = ("vendor_daily",)


@dataclass(frozen=True)
class RefreshPolicy:
    """How often a refresh may spend a vendor call.

    Defaults suit Alpha Vantage's free tier (25 requests/day): at most one call an
    hour and at most 24 calls a UTC day, so an experiment can refresh daily bars
    without exhausting the quota on a retry storm.
    """

    enabled: bool = True
    min_interval_seconds: float = 3600.0
    max_fetches_per_day: int = 24


@dataclass
class RefreshState:
    """The durable refresh bookkeeping, kept out of the recovery file."""

    day: str | None = None
    fetches_today: int = 0
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_status: str | None = None
    last_bar_date: str | None = None
    last_new_session: str | None = None
    last_new_sessions: int = 0
    last_error: str | None = None
    # The finer reason behind the last failure (see FAILURE_KINDS). Kept distinct
    # from last_status so an operator sees "auth" and not just "unavailable".
    last_failure_kind: str | None = None
    # The last tick that *skipped* a fetch (throttled/quota). Recorded separately so
    # a skip never overwrites the last real outcome above.
    last_skip: str | None = None
    last_skip_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": REFRESH_SCHEMA_VERSION,
            "day": self.day,
            "fetches_today": self.fetches_today,
            "last_attempt_at": self.last_attempt_at,
            "last_success_at": self.last_success_at,
            "last_status": self.last_status,
            "last_bar_date": self.last_bar_date,
            "last_new_session": self.last_new_session,
            "last_new_sessions": self.last_new_sessions,
            "last_error": self.last_error,
            "last_failure_kind": self.last_failure_kind,
            "last_skip": self.last_skip,
            "last_skip_at": self.last_skip_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> RefreshState:
        """Rebuild from a stored payload, ignoring unknown keys (additive)."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})

    @classmethod
    def load(cls, path: Path) -> RefreshState:
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # A missing or unreadable sidecar is not fatal: refresh simply starts
            # from a clean slate, and the throttle is conservative as a result.
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


@dataclass
class RefreshOutcome:
    """What a refresh attempt did, in terms a caller can log and branch on."""

    status: str
    attempted: bool
    new_sessions: tuple[str, ...] = ()
    last_bar_date: str | None = None
    last_processed_session: str | None = None
    message: str | None = None
    fetches_today: int = 0
    max_fetches_per_day: int = 0
    failure_kind: str | None = None
    source_status: str = "ok"
    state: RefreshState = field(default_factory=RefreshState)

    def summary_line(self) -> str:
        """One compact, secret-free line: last data date, newest session, status.

        When this tick skipped a fetch, the line carries the *last real outcome*
        too, so a bare ``status=throttled`` can never hide a standing failure.
        """
        line = (
            f"[data] source_last_bar={self.last_bar_date or '?'} "
            f"last_new_session={self.last_new_session or 'none'} "
            f"last_processed={self.last_processed_session or 'none'} "
            f"status={self.status} "
            f"new={len(self.new_sessions)} "
            f"fetches_today={self.fetches_today}/{self.max_fetches_per_day}"
        )
        if self.status in SKIP_STATUSES and self.state.last_status:
            line += (
                f" last_outcome={self.state.last_status}"
                f" last_error={self.state.last_error or 'none'}"
            )
        elif self.failure_kind:
            line += f" failure_kind={self.failure_kind}"
        return line

    def to_dict(self) -> dict:
        """A JSON-safe form, for a dashboard response or the persisted last run."""
        return {
            "status": self.status,
            "attempted": self.attempted,
            "new_sessions": list(self.new_sessions),
            "last_bar_date": self.last_bar_date,
            "last_processed_session": self.last_processed_session,
            "message": self.message,
            "fetches_today": self.fetches_today,
            "max_fetches_per_day": self.max_fetches_per_day,
            "failure_kind": self.failure_kind,
            "source_status": self.source_status,
            # The last *real* outcome, so a dashboard reading a skip still sees the
            # standing failure behind it.
            "last_outcome": self.state.last_status,
            "last_error": self.state.last_error,
            "last_failure_kind": self.state.last_failure_kind,
            "summary": self.summary_line(),
        }

    @property
    def last_new_session(self) -> str | None:
        return self.new_sessions[-1] if self.new_sessions else None


def _source_symbol(source: PriceSource) -> str | None:
    """The symbol a source carries, from its provenance or its single frame.

    A vendor daily source records its symbol in provenance; a frame-backed source
    without one still carries exactly one symbol, so the frame key is a safe
    fallback. ``None`` means the symbol cannot be determined, and a refresh is
    refused rather than guessed.
    """
    provenance = getattr(source, "provenance", None)
    symbol = getattr(provenance, "symbol", None)
    if symbol:
        return symbol
    frames = getattr(source, "_frames", None)
    if isinstance(frames, dict) and len(frames) == 1:
        return next(iter(frames))
    return None


def _latest_processed(experiment) -> str | None:
    sessions = getattr(experiment, "processed_sessions", None) or set()
    return max(sessions) if sessions else None


def _default_fetcher(experiment, *, today: date | None = None) -> PriceSource:
    """Rebuild the vendor daily source the experiment was started on.

    The provider is chosen from the source's own provenance, so a refresh rebuilds
    the *same* vendor the run started on: an Alpaca experiment is refreshed from
    Alpaca, an Alpha Vantage experiment from Alpha Vantage. There is no
    cross-provider fallback — a source this module cannot rebuild is refused rather
    than refreshed from a different vendor. Each provider reads its own credential
    from the environment; this module never touches a key.
    """
    source = experiment.source
    symbol = _source_symbol(source)
    if not symbol:
        raise PriceSourceError(
            f"the source {source.name!r} does not record a symbol, so it cannot be refreshed"
        )
    provider = _provider_for(source)
    if provider == "alpaca_iex":
        from .providers.alpaca_iex import AlpacaIexDailySource

        return AlpacaIexDailySource(symbol, name=source.name, today=today)
    if provider == "alpha_vantage":
        from .providers.alpha_vantage import AlphaVantageDailySource

        return AlphaVantageDailySource(symbol, name=source.name, today=today)
    raise PriceSourceError(
        f"the source {source.name!r} (kind {getattr(source, 'kind', 'unknown')!r}) is "
        "not a refreshable vendor daily source; refusing to refresh it from a "
        "different provider"
    )


def _provider_for(source: PriceSource) -> str | None:
    """The vendor a source came from, from its provenance name.

    The single reading of "which vendor is this", so a refresh cannot rebuild a
    source from the wrong provider. ``None`` means the vendor is unknown, and a
    refresh is refused rather than guessed.
    """
    name = (getattr(source, "name", "") or "").lower()
    provenance = getattr(source, "provenance", None)
    provenance_name = (getattr(provenance, "name", "") or "").lower()
    haystack = f"{name} {provenance_name}"
    if "alpaca" in haystack:
        return "alpaca_iex"
    if "alpha_vantage" in haystack or "alphavantage" in haystack:
        return "alpha_vantage"
    return None


def _merge_frames(
    old: pd.DataFrame, new: pd.DataFrame, *, tolerance: float = 1e-9
) -> tuple[pd.DataFrame, tuple[str, ...], str | None]:
    """Append ``new`` sessions after ``old``, refusing a restated bar.

    Returns ``(merged_frame, new_session_labels, conflict)``. ``conflict`` is a
    short reason when an existing session's OHLCV disagrees, in which case the
    caller must refuse the whole refresh.
    """
    old_labels = {str(index)[:10]: index for index in old.index}
    additions: list[dict] = []
    new_labels: list[str] = []
    last_old = str(old.index[-1])[:10]

    for index, row in new.iterrows():
        label = str(index)[:10]
        if label in old_labels:
            existing = old.loc[old_labels[label]]
            for column in BAR_COLUMNS:
                if column not in old.columns or column not in new.columns:
                    continue
                a, b = float(existing[column]), float(row[column])
                if abs(a - b) > tolerance:
                    return old, (), (
                        f"session {label} changed {column} from {a} to {b}; refusing to "
                        "restate a bar the run has already used"
                    )
            continue
        # Append-only and forward: a session at or before the current last session
        # is a backfill. Inserting it would let the engine settle against a bar it
        # has already passed, so it is ignored rather than reordered in.
        if label <= last_old:
            continue
        additions.append(
            {
                "session": pd.Timestamp(index),
                **{column: row[column] for column in BAR_COLUMNS},
            }
        )
        new_labels.append(label)

    if not additions:
        return old, (), None

    appended = pd.DataFrame(additions).set_index("session")
    appended.index.name = old.index.name
    merged = pd.concat([old[list(BAR_COLUMNS)], appended[list(BAR_COLUMNS)]]).sort_index()
    merged.index.name = old.index.name
    return merged, tuple(new_labels), None


def _rebuild_source(
    experiment, symbol: str, merged: pd.DataFrame, *, now: datetime | None = None
) -> PriceSource:
    """A source of the same kind/name as the experiment's, over the merged bars."""
    old_source = experiment.source
    old_provenance = getattr(old_source, "provenance", None)
    last_bar = str(merged.index[-1])[:10]
    now = datetime.now(UTC)

    provenance = old_provenance
    if old_provenance is not None:
        provenance = replace(
            old_provenance,
            fetched_at=now.isoformat(),
            last_bar_date=last_bar,
            dataset_end=last_bar,
            is_current_market=bar_is_fresh(
                last_bar,
                reference=now.date(),
                max_stale_days=old_provenance.max_stale_days,
            ),
        )
    rebuilt = FramePriceSource(
        {symbol: merged},
        name=old_source.name,
        provenance=provenance,
    )
    kind = getattr(old_source, "kind", None)
    if kind:
        rebuilt.kind = kind
    return rebuilt


def _failure_kind(exc: BaseException) -> str:
    """Map a fetch failure to the finer reason the operator needs.

    Authentication, a missing credential, a vendor rate limit, a stale feed and a
    transport error are deliberately distinct: collapsing them is exactly the bug
    this classification exists to prevent.
    """
    from .providers.base import (
        ProviderAuthError,
        ProviderNotConfiguredError,
        ProviderRateLimitedError,
        ProviderStaleDataError,
        ProviderUnsupportedError,
    )

    if isinstance(exc, ProviderAuthError):
        return FAILURE_KIND_AUTH
    if isinstance(exc, ProviderNotConfiguredError):
        return FAILURE_KIND_NOT_CONFIGURED
    if isinstance(exc, ProviderRateLimitedError):
        return FAILURE_KIND_RATE_LIMITED
    if isinstance(exc, ProviderStaleDataError):
        return FAILURE_KIND_STALE
    if isinstance(exc, ProviderUnsupportedError):
        return FAILURE_KIND_UNSUPPORTED
    # A transport failure (timeout, DNS, connection reset) or a malformed body.
    return FAILURE_KIND_TRANSPORT


def _status_for_failure(kind: str) -> str:
    """The refresh status a failure kind maps to (a closed vocabulary)."""
    if kind == FAILURE_KIND_AUTH:
        return STATUS_AUTH_FAILED
    if kind == FAILURE_KIND_NOT_CONFIGURED:
        return STATUS_NOT_CONFIGURED
    return STATUS_UNAVAILABLE


def refresh_source(
    experiment,
    *,
    policy: RefreshPolicy | None = None,
    fetcher=None,
    now: datetime | None = None,
    state_path: Path | None = None,
) -> RefreshOutcome:
    """Append the vendor's newest sessions to ``experiment`` in place.

    Best-effort and never fatal: every failure is returned as a status. On success
    the experiment's source is replaced with one over the merged bars and the
    experiment is saved through ``save_state`` (so identity and the book are kept).
    """
    policy = policy or RefreshPolicy()
    now = now or datetime.now(UTC)
    root = Path(experiment.root)
    state_file = Path(state_path) if state_path is not None else root / REFRESH_STATE_FILENAME
    state = RefreshState.load(state_file)
    last_processed = _latest_processed(experiment)

    def finish(status: str, *, attempted: bool, skip: bool = False, **kwargs) -> RefreshOutcome:
        if skip:
            # A skip decided *not* to fetch, so it must not advance the throttle
            # clock, count against the quota, or overwrite the last real outcome.
            # Otherwise the throttle is self-perpetuating and a fixed credential
            # would never be retried.
            state.last_skip = status
            state.last_skip_at = now.isoformat()
        else:
            state.last_status = status
            state.last_attempt_at = now.isoformat()
            if attempted:
                state.day = now.date().isoformat()
                state.fetches_today += 1
        failure_kind = kwargs.pop("failure_kind", None)
        # The effective source health: a skip reports the last real outcome's
        # health, so a throttled tick still shows the standing auth failure.
        effective = state.last_status if skip else status
        outcome = RefreshOutcome(
            status=status,
            attempted=attempted,
            last_processed_session=last_processed,
            last_bar_date=state.last_bar_date,
            fetches_today=state.fetches_today,
            max_fetches_per_day=policy.max_fetches_per_day,
            failure_kind=failure_kind or (state.last_failure_kind if skip else None),
            source_status="failed" if effective in SOURCE_FAILURE_STATUSES else "ok",
            state=state,
            **kwargs,
        )
        # A source with no vendor to ask (a frozen dataset, a fixture) has nothing
        # to record, so no sidecar is written for it — a read-only monitor of a
        # frozen run stays read-only.
        if status not in (STATUS_DISABLED, STATUS_NOT_REFRESHABLE):
            state.save(state_file)
        return outcome

    if not policy.enabled:
        return finish(STATUS_DISABLED, attempted=False)

    kind = getattr(experiment.source, "kind", None)
    if kind not in REFRESHABLE_KINDS:
        return finish(STATUS_NOT_REFRESHABLE, attempted=False)

    # Resolve the symbol *before* any fetch: a source whose symbol cannot be
    # determined (a multi-symbol frame) must be refused without spending a call.
    symbol = _source_symbol(experiment.source)
    if symbol is None:
        return finish(
            STATUS_NOT_REFRESHABLE,
            attempted=False,
            message=(
                f"the source {experiment.source.name!r} does not record a single "
                "symbol, so it cannot be refreshed"
            ),
        )

    # A new UTC day resets the daily counter.
    if state.day != now.date().isoformat():
        state.day = now.date().isoformat()
        state.fetches_today = 0

    if state.fetches_today >= policy.max_fetches_per_day:
        return finish(STATUS_QUOTA_GUARD, attempted=False, skip=True)

    if state.last_attempt_at:
        try:
            elapsed = (now - datetime.fromisoformat(state.last_attempt_at)).total_seconds()
        except ValueError:
            elapsed = policy.min_interval_seconds
        if elapsed < policy.min_interval_seconds:
            return finish(STATUS_THROTTLED, attempted=False, skip=True)

    try:
        fresh = (fetcher or _default_fetcher)(experiment, today=now.date())
    except Exception as exc:  # noqa: BLE001 - a fetch failure must not stop the engine
        reason = f"{type(exc).__name__}: {exc}"
        state.last_error = reason
        failure_kind = _failure_kind(exc)
        state.last_failure_kind = failure_kind
        return finish(
            _status_for_failure(failure_kind),
            attempted=True,
            message=reason,
            failure_kind=failure_kind,
        )

    try:
        fresh_frame = fresh.frame(symbol)
    except PriceSourceError as exc:
        state.last_error = str(exc)
        failure_kind = _failure_kind(exc)
        state.last_failure_kind = failure_kind
        return finish(
            _status_for_failure(failure_kind),
            attempted=True,
            message=state.last_error,
            failure_kind=failure_kind,
        )

    old_frame = experiment.source.frame(symbol)
    merged, new_labels, conflict = _merge_frames(old_frame, fresh_frame)
    if conflict:
        state.last_error = conflict
        state.last_failure_kind = FAILURE_KIND_MALFORMED
        return finish(
            STATUS_CONFLICT,
            attempted=True,
            message=conflict,
            failure_kind=FAILURE_KIND_MALFORMED,
        )
    if not new_labels:
        state.last_bar_date = str(merged.index[-1])[:10]
        state.last_new_sessions = 0
        state.last_success_at = now.isoformat()
        state.last_error = None
        state.last_failure_kind = None
        return finish(STATUS_NO_NEW_SESSIONS, attempted=True)

    # Validate the merged frame before it becomes the experiment's data.
    try:
        validated = _validate_frame(merged, symbol, getattr(experiment.source, "kind", "vendor_daily"))
    except PriceSourceError as exc:
        state.last_error = str(exc)
        state.last_failure_kind = FAILURE_KIND_MALFORMED
        return finish(
            STATUS_UNAVAILABLE,
            attempted=True,
            message=state.last_error,
            failure_kind=FAILURE_KIND_MALFORMED,
        )

    experiment.source = _rebuild_source(experiment, symbol, validated, now=now)
    experiment.save_state()

    state.last_bar_date = new_labels[-1]
    state.last_new_session = new_labels[-1]
    state.last_new_sessions = len(new_labels)
    state.last_success_at = now.isoformat()
    state.last_error = None
    state.last_failure_kind = None
    return finish(STATUS_APPENDED, attempted=True, new_sessions=new_labels)


def read_refresh_state(root: str | Path) -> RefreshState:
    """The recorded refresh bookkeeping, for a status view or a log line."""
    return RefreshState.load(Path(root) / REFRESH_STATE_FILENAME)
