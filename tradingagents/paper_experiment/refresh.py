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
        STATUS_CONFLICT,
        STATUS_INITIAL_FETCH,
    }
)

# The source kinds a refresh can extend: a vendor's historical daily bars. A
# frozen dataset or a test fixture has no vendor to ask; a quote-only source has
# no bar history to append to.
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
    state: RefreshState = field(default_factory=RefreshState)

    def summary_line(self) -> str:
        """One compact, secret-free line: last data date, newest session, status."""
        return (
            f"[data] source_last_bar={self.last_bar_date or '?'} "
            f"last_new_session={self.last_new_session or 'none'} "
            f"last_processed={self.last_processed_session or 'none'} "
            f"status={self.status} "
            f"new={len(self.new_sessions)} "
            f"fetches_today={self.fetches_today}/{self.max_fetches_per_day}"
        )

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

    Only the Alpha Vantage daily source is refreshable today, and it reads its own
    credential from the environment (this module never touches a key). A source
    whose provenance does not name a symbol cannot be refreshed.
    """
    from .providers.alpha_vantage import AlphaVantageDailySource

    source = experiment.source
    symbol = _source_symbol(source)
    if not symbol:
        raise PriceSourceError(
            f"the source {source.name!r} does not record a symbol, so it cannot be refreshed"
        )
    return AlphaVantageDailySource(symbol, name=source.name, today=today)


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


def _rebuild_source(experiment, symbol: str, merged: pd.DataFrame) -> PriceSource:
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

    def finish(status: str, *, attempted: bool, **kwargs) -> RefreshOutcome:
        state.last_status = status
        state.last_attempt_at = now.isoformat()
        if attempted:
            state.day = now.date().isoformat()
            state.fetches_today += 1
        outcome = RefreshOutcome(
            status=status,
            attempted=attempted,
            last_processed_session=last_processed,
            last_bar_date=state.last_bar_date,
            fetches_today=state.fetches_today,
            max_fetches_per_day=policy.max_fetches_per_day,
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
        return finish(STATUS_QUOTA_GUARD, attempted=False)

    if state.last_attempt_at:
        try:
            elapsed = (now - datetime.fromisoformat(state.last_attempt_at)).total_seconds()
        except ValueError:
            elapsed = policy.min_interval_seconds
        if elapsed < policy.min_interval_seconds:
            return finish(STATUS_THROTTLED, attempted=False)

    try:
        fresh = (fetcher or _default_fetcher)(experiment, today=now.date())
    except Exception as exc:  # noqa: BLE001 - a fetch failure must not stop the engine
        state.last_error = f"{type(exc).__name__}: {exc}"
        return finish(STATUS_UNAVAILABLE, attempted=True, message=state.last_error)

    try:
        fresh_frame = fresh.frame(symbol)
    except PriceSourceError as exc:
        state.last_error = str(exc)
        return finish(STATUS_UNAVAILABLE, attempted=True, message=state.last_error)

    old_frame = experiment.source.frame(symbol)
    merged, new_labels, conflict = _merge_frames(old_frame, fresh_frame)
    if conflict:
        state.last_error = conflict
        return finish(STATUS_CONFLICT, attempted=True, message=conflict)
    if not new_labels:
        state.last_bar_date = str(merged.index[-1])[:10]
        state.last_new_sessions = 0
        state.last_success_at = now.isoformat()
        state.last_error = None
        return finish(STATUS_NO_NEW_SESSIONS, attempted=True)

    # Validate the merged frame before it becomes the experiment's data.
    try:
        validated = _validate_frame(merged, symbol, getattr(experiment.source, "kind", "vendor_daily"))
    except PriceSourceError as exc:
        state.last_error = str(exc)
        return finish(STATUS_UNAVAILABLE, attempted=True, message=state.last_error)

    experiment.source = _rebuild_source(experiment, symbol, validated)
    experiment.save_state()

    state.last_bar_date = new_labels[-1]
    state.last_new_session = new_labels[-1]
    state.last_new_sessions = len(new_labels)
    state.last_success_at = now.isoformat()
    state.last_error = None
    return finish(STATUS_APPENDED, attempted=True, new_sessions=new_labels)


def read_refresh_state(root: str | Path) -> RefreshState:
    """The recorded refresh bookkeeping, for a status view or a log line."""
    return RefreshState.load(Path(root) / REFRESH_STATE_FILENAME)
