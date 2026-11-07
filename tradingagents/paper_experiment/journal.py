"""The durable, append-only, tamper-evident journal for a paper experiment.

A prospective paper experiment accumulates decisions over time. The journal is
the record of *what happened and when* — it is not a reporting snapshot, and it
is not the paper session's recovery file. The three are kept apart on purpose:

- the **journal** (this module) is the experiment's own history: an append-only,
  hash-chained log of sessions, decisions, executions and evaluations;
- the **paper session state** (``execution/paper.py``) is the recovery format for
  the book — the ledger, resting orders and processed-signal set;
- a **reporting snapshot** is a read-only view for a human.

Two properties make the journal auditable rather than merely written:

- **Append-only and hash-chained.** Every entry carries the hash of the entry
  before it, so removing, reordering or editing any entry breaks the chain and
  :meth:`ExperimentJournal.verify` raises instead of trusting a tampered file.
- **Atomic and idempotent.** A write replaces the whole file through a temporary
  file and ``os.replace``, so a crash cannot leave a half-written line. An entry
  with an ``event_id`` already present is not appended twice: re-running a step
  after a restart cannot duplicate an event.

No credential reaches the journal: every payload is scanned by
:func:`~tradingagents.llm_evaluation.records.assert_secret_free` before it is
written, so a key cannot be serialised into the history.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.llm_evaluation.records import assert_secret_free

JOURNAL_SCHEMA_VERSION = 1
"""The on-disk journal format. Bumped when a reader must know the shape changed."""

# The closed vocabulary of journal entry kinds. A new behaviour adds a kind here
# rather than writing free-form text, so the history stays queryable and a typo
# is a construction error.
KIND_EXPERIMENT_CREATED = "experiment_created"
KIND_SESSION_PROCESSED = "session_processed"
KIND_DECISION_RECORDED = "decision_recorded"
KIND_DECISION_BLOCKED = "decision_blocked"
KIND_ORDER_EXECUTED = "order_executed"
KIND_DECISION_EVALUATED = "decision_evaluated"
KIND_EXPERIMENT_HALTED = "experiment_halted"
KIND_STATE_SAVED = "state_saved"

JOURNAL_KINDS = frozenset(
    {
        KIND_EXPERIMENT_CREATED,
        KIND_SESSION_PROCESSED,
        KIND_DECISION_RECORDED,
        KIND_DECISION_BLOCKED,
        KIND_ORDER_EXECUTED,
        KIND_DECISION_EVALUATED,
        KIND_EXPERIMENT_HALTED,
        KIND_STATE_SAVED,
    }
)

GENESIS_HASH = "0" * 64


class JournalError(RuntimeError):
    """The journal is malformed, tampered with, or could not be written."""


def _canonical(payload: Any) -> str:
    # Strict: a payload that is not JSON-safe is a bug, and it should fail here
    # with a clear error rather than being coerced by ``default=str`` and then
    # failing later at write time (or, worse, hashing differently from the file).
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _compute_hash(
    *, seq: int, event_id: str, timestamp: str, kind: str, payload: dict, prev_hash: str
) -> str:
    body = _canonical(
        {
            "seq": seq,
            "event_id": event_id,
            "timestamp": timestamp,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
        }
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JournalEntry:
    """One immutable line of the journal, with the hash that chains it."""

    seq: int
    event_id: str
    timestamp: str
    kind: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> JournalEntry:
        if not isinstance(raw, dict):
            raise JournalError(f"a journal entry must be a mapping, got {type(raw).__name__}")
        required = {"seq", "event_id", "timestamp", "kind", "payload", "prev_hash", "entry_hash"}
        missing = required - set(raw)
        if missing:
            raise JournalError(f"journal entry is missing {sorted(missing)}")
        if raw["kind"] not in JOURNAL_KINDS:
            raise JournalError(f"unknown journal entry kind {raw['kind']!r}")
        if not isinstance(raw["payload"], dict):
            raise JournalError("a journal entry payload must be a mapping")
        return cls(
            seq=int(raw["seq"]),
            event_id=str(raw["event_id"]),
            timestamp=str(raw["timestamp"]),
            kind=str(raw["kind"]),
            payload=dict(raw["payload"]),
            prev_hash=str(raw["prev_hash"]),
            entry_hash=str(raw["entry_hash"]),
        )


class ExperimentJournal:
    """An append-only, hash-chained log of one experiment's history.

    The file is a single JSON document (``{"schema_version": 1, "entries": [...]}``)
    rewritten atomically on every append. A journal is small — one line per
    session, decision and evaluation — so a full rewrite is cheap and buys a
    crash-safe write that a bare append does not.
    """

    def __init__(self, path: str | Path, entries: list[JournalEntry] | None = None) -> None:
        self.path = Path(path)
        self._entries: list[JournalEntry] = list(entries or [])
        self._by_id: dict[str, JournalEntry] = {e.event_id: e for e in self._entries}

    # -- reading ----------------------------------------------------------

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        return tuple(self._entries)

    @property
    def last_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def __len__(self) -> int:
        return len(self._entries)

    def has(self, event_id: str) -> bool:
        return event_id in self._by_id

    def get(self, event_id: str) -> JournalEntry | None:
        return self._by_id.get(event_id)

    def of_kind(self, kind: str) -> tuple[JournalEntry, ...]:
        return tuple(e for e in self._entries if e.kind == kind)

    # -- writing ----------------------------------------------------------

    def append(
        self, kind: str, payload: dict, *, event_id: str, timestamp: str | None = None
    ) -> JournalEntry:
        """Append one entry, or return the existing one if ``event_id`` repeats.

        Idempotent on ``event_id``: a step re-run after a restart finds its event
        already present and does not duplicate it. The payload is scanned for
        credentials before anything is written.
        """
        if kind not in JOURNAL_KINDS:
            raise JournalError(f"unknown journal entry kind {kind!r}")
        if not event_id:
            raise JournalError("a journal entry needs a non-empty event_id")
        existing = self._by_id.get(event_id)
        if existing is not None:
            return existing
        if not isinstance(payload, dict):
            raise JournalError("a journal entry payload must be a mapping")
        assert_secret_free(payload, path=f"journal[{kind}]")
        seq = len(self._entries) + 1
        prev_hash = self.last_hash
        timestamp = timestamp or _now()
        entry_hash = _compute_hash(
            seq=seq,
            event_id=event_id,
            timestamp=timestamp,
            kind=kind,
            payload=payload,
            prev_hash=prev_hash,
        )
        entry = JournalEntry(
            seq=seq,
            event_id=event_id,
            timestamp=timestamp,
            kind=kind,
            payload=dict(payload),
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        self._by_id[event_id] = entry
        self._write_atomic()
        return entry

    def _write_atomic(self) -> None:
        document = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "entries": [e.to_dict() for e in self._entries],
        }
        text = json.dumps(document, indent=2, sort_keys=False) + "\n"
        # Scan the whole document too: a payload assembled across entries cannot
        # slip a credential past the per-entry check.
        assert_secret_free(text, path="journal")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and rename, so a crash mid-write leaves
        # either the old file or the new one, never a truncated one.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".journal-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    # -- verification -----------------------------------------------------

    def verify(self) -> None:
        """Raise unless the chain is intact and the sequence is dense from 1."""
        prev = GENESIS_HASH
        for index, entry in enumerate(self._entries, start=1):
            if entry.seq != index:
                raise JournalError(
                    f"journal sequence is not dense: expected {index}, found {entry.seq}"
                )
            if entry.prev_hash != prev:
                raise JournalError(
                    f"journal chain is broken at seq {entry.seq}: entry links to "
                    f"{entry.prev_hash[:12]}…, previous entry hashes to {prev[:12]}…"
                )
            recomputed = _compute_hash(
                seq=entry.seq,
                event_id=entry.event_id,
                timestamp=entry.timestamp,
                kind=entry.kind,
                payload=entry.payload,
                prev_hash=entry.prev_hash,
            )
            if recomputed != entry.entry_hash:
                raise JournalError(
                    f"journal entry {entry.seq} has been altered: stored hash "
                    f"{entry.entry_hash[:12]}…, recomputed {recomputed[:12]}…"
                )
            prev = entry.entry_hash

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> ExperimentJournal:
        """Read a journal, verifying the chain before returning it."""
        source = Path(path)
        if not source.exists():
            return cls(source)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise JournalError(f"{source} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise JournalError(f"{source} could not be read: {exc}") from exc
        if not isinstance(raw, dict):
            raise JournalError("a journal file must be a mapping")
        version = raw.get("schema_version")
        if version != JOURNAL_SCHEMA_VERSION:
            raise JournalError(
                f"unsupported journal schema_version {version!r}; this build reads "
                f"version {JOURNAL_SCHEMA_VERSION}"
            )
        entries = [JournalEntry.from_dict(e) for e in raw.get("entries", [])]
        journal = cls(source, entries)
        if verify:
            journal.verify()
        return journal
