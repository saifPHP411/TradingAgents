"""Durable, versioned state for the broker-backed paper path.

This is the broker path's *own* recovery format. It is deliberately not the
internal paper session's ``session_state.json``: the two accounts are separate,
and mixing them would make a broker fill look like an internal fill.

The file is written atomically (temp file + ``os.replace``) and carries a
``schema_version`` so an older reader refuses a newer file rather than
misreading it. It records:

- the broker account snapshot (cash, equity, buying power) last confirmed by the
  broker, so a restart can compare it against the broker rather than assume it;
- every order the broker has accepted, keyed by the **client order id** the
  broker also knows, so a restart can reconcile a submission whose response was
  lost instead of re-submitting it;
- the set of client order ids whose fills have already been applied, so a fill
  cannot be counted twice across a restart.

No credential is ever written here: the account snapshot holds numbers, and the
order records hold ids, symbols and quantities — never a key.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BROKER_STATE_SCHEMA_VERSION = 1
"""The on-disk broker state format. Bumped when a reader must know the shape changed."""


class BrokerStateError(RuntimeError):
    """The persisted broker state is malformed, truncated, or a newer version."""


def _as_non_negative_int(value: Any, default: int) -> int:
    """Coerce a persisted counter, tolerating a missing or malformed value.

    An additive field must not make an otherwise readable state file unusable, so
    a bad counter falls back to the default rather than raising. A bool is not a
    counter, and a negative count is treated as the default.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _as_non_negative_float(value: Any, default: float = 0.0) -> float:
    """Coerce a persisted quantity, tolerating a missing or malformed value.

    The same rule as the counter: an additive field must not make an
    otherwise readable state file unusable. A quantity that is not a finite
    number, or that is negative, falls back to the default -- a quantity is
    never negative and never NaN.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0:
        return default
    return number


def _atomic_write(path: Path, text: str) -> None:
    import contextlib
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".broker-state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass
class BrokerAccountState:
    """The broker account figures last confirmed by the broker, plus order memory.

    ``confirmed`` is ``False`` until the broker has actually answered an account
    read. A state that has never been confirmed is not a claim about the account;
    it is a local cache that must be refreshed before it is trusted.
    """

    schema_version: int = BROKER_STATE_SCHEMA_VERSION
    endpoint: str = ""
    account_id: str | None = None
    cash: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    currency: str | None = None
    trading_blocked: bool | None = None
    account_blocked: bool | None = None
    confirmed: bool = False
    confirmed_at: str | None = None
    """Client order ids the broker has accepted, mapped to the last known state."""
    orders: dict[str, dict] = field(default_factory=dict)
    applied_fills: set[str] = field(default_factory=set)
    """The client order ids whose fill has been applied at least once.

    This set says "has this order been seen", which is not the same question
    as "how much of it has been accounted for". The quantity lives in
    ``applied_fill_quantities`` beside it.
    """
    applied_fill_quantities: dict[str, float] = field(default_factory=dict)
    """The cumulative filled quantity already applied, keyed by client order id.

    A client order id alone is not enough to decide whether a fill is new: an
    order that traded 0.4 and then traded 0.2 more carries the **same** id,
    and the second report is a new quantity that has to be accounted for. So
    the applied **cumulative** quantity is kept, and a report is new only
    when it carries more than what was applied before. The quantity, not the
    id, is what makes the accounting exactly-once.

    Kept beside the older ``applied_fills`` set, which stays the answer to
    "has this order been seen at all" and is what the rest of the project
    reads.
    """
    applied_execution_ids: dict[str, list[str]] = field(default_factory=dict)
    """The venue's execution ids already applied, keyed by client order id.

    An **execution id** names one trade at the venue. It is the finest unit
    the venue offers, so it is the unit of exactly-once accounting: the same
    execution id is never applied twice, no matter how many times the order
    is read back. The ids are kept per order so a restart does not re-apply
    a trade that was already applied.
    """
    positions: dict[str, dict] = field(default_factory=dict)
    """The broker's own positions, last read, keyed by symbol.

    Persisted so a restart can report what the account held, and so a worker can
    refuse a sell against a stale-but-non-empty position set rather than assume
    flat. It is a cache, not a claim: ``positions_at`` records when it was read.
    """
    positions_at: str | None = None
    daily_orders_count: int = 0
    daily_orders_date: str | None = None
    """Daily-order counter, persisted so it survives a restart.

    ``daily_orders_count`` is the number of orders the broker actually accepted
    on ``daily_orders_date`` (a UTC calendar day, ``YYYY-MM-DD``). Only an
    accepted order increments it: a locally refused or broker-rejected order does
    not, and a partial fill of one order is still one order. On read, a count
    whose date is not today is treated as zero, so a new day starts fresh with no
    explicit reset. Additive and optional: an older state file without these
    fields reads back as "no orders today", so the schema version does not change.
    """
    protection_order_pairs: dict[str, dict] = field(default_factory=dict)
    """Protection order pairs (SL/TP), keyed by the prefix used to create them.

    Each entry holds the client order ids of both legs, the symbol and the
    position quantity they cover, so the counterpart can be cancelled when one
    leg fills. Additive and optional: an older state file without this field
    reads back with no pairs, so the schema version does not change.
    """
    protection_levels: dict[str, dict] = field(default_factory=dict)
    """The levels a protection pair was created from, keyed by canonical symbol.

    Each entry holds the ``stop_loss`` and ``take_profit`` prices the pair
    covers. They are kept so that a **partial** fill can be re-protected: when
    one leg sells part of the position, the quantity that is still held needs
    a new guard, and the only levels that may be used for it are the ones the
    strategy derived — the broker never invents a level. Additive and
    optional: an older state file without this field reads back with no
    levels, so the schema version does not change.
    """
    protection_residuals: dict[str, dict] = field(default_factory=dict)
    """What happened to the quantity a partial protection fill left behind.

    Keyed by the prefix of the pair that partially filled. Each entry records
    the executed quantity the venue reported, the residual the venue still
    held, whether a replacement pair was created, whether the venue confirmed
    it, and why the outcome is what it is. It is the audit trail for the
    decision "the residual is protected" or "the residual is unprotected", so
    the reason survives a restart. Additive and optional: an older state file
    without this field reads back with no residuals, so the schema version
    does not change.
    """
    protection_cancel_outcomes: dict[str, dict] = field(default_factory=dict)
    """What the venue said when the counterpart of a protection pair was cancelled.

    Keyed by the prefix of the pair. Each entry records the counterpart's
    client order id and one of three outcomes: ``confirmed_terminal`` (the
    venue stated the order is finished), ``still_active`` (the venue still
    shows it working) or ``unknown`` (its fate could not be established).

    The record exists because the outcome **decides** whether a replacement
    pair may be created, and because a restart has to know why a position is
    unprotected. Additive and optional: an older state file without this field
    reads back with no outcomes, so the schema version does not change.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "endpoint": self.endpoint,
            "account_id": self.account_id,
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "currency": self.currency,
            "trading_blocked": self.trading_blocked,
            "account_blocked": self.account_blocked,
            "confirmed": self.confirmed,
            "confirmed_at": self.confirmed_at,
            "orders": {k: dict(v) for k, v in sorted(self.orders.items())},
            "applied_fills": sorted(self.applied_fills),
            "applied_fill_quantities": {
                k: float(v) for k, v in sorted(self.applied_fill_quantities.items())
            },
            "applied_execution_ids": {
                k: sorted(v) for k, v in sorted(self.applied_execution_ids.items())
            },
            "positions": {k: dict(v) for k, v in sorted(self.positions.items())},
            "positions_at": self.positions_at,
            "daily_orders_count": self.daily_orders_count,
            "daily_orders_date": self.daily_orders_date,
            "protection_order_pairs": {k: dict(v) for k, v in sorted(self.protection_order_pairs.items())},
            "protection_levels": {k: dict(v) for k, v in sorted(self.protection_levels.items())},
            "protection_residuals": {k: dict(v) for k, v in sorted(self.protection_residuals.items())},
            "protection_cancel_outcomes": {
                k: dict(v) for k, v in sorted(self.protection_cancel_outcomes.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BrokerAccountState:
        if not isinstance(payload, dict):
            raise BrokerStateError("the broker state must be a JSON object")
        version = payload.get("schema_version")
        if version != BROKER_STATE_SCHEMA_VERSION:
            raise BrokerStateError(
                f"unsupported broker state schema_version {version!r}; this build "
                f"reads version {BROKER_STATE_SCHEMA_VERSION}"
            )
        orders = payload.get("orders") or {}
        if not isinstance(orders, dict):
            raise BrokerStateError("broker state 'orders' must be a mapping")
        applied = payload.get("applied_fills") or []
        if not isinstance(applied, list):
            raise BrokerStateError("broker state 'applied_fills' must be a list")
        applied_fill_quantities = payload.get("applied_fill_quantities") or {}
        if not isinstance(applied_fill_quantities, dict):
            raise BrokerStateError(
                "broker state 'applied_fill_quantities' must be a mapping"
            )
        applied_execution_ids = payload.get("applied_execution_ids") or {}
        if not isinstance(applied_execution_ids, dict):
            raise BrokerStateError(
                "broker state 'applied_execution_ids' must be a mapping"
            )
        positions = payload.get("positions") or {}
        if not isinstance(positions, dict):
            raise BrokerStateError("broker state 'positions' must be a mapping")
        protection_pairs = payload.get("protection_order_pairs") or {}
        if not isinstance(protection_pairs, dict):
            raise BrokerStateError("broker state 'protection_order_pairs' must be a mapping")
        protection_levels = payload.get("protection_levels") or {}
        if not isinstance(protection_levels, dict):
            raise BrokerStateError("broker state 'protection_levels' must be a mapping")
        protection_residuals = payload.get("protection_residuals") or {}
        if not isinstance(protection_residuals, dict):
            raise BrokerStateError("broker state 'protection_residuals' must be a mapping")
        protection_cancel_outcomes = payload.get("protection_cancel_outcomes") or {}
        if not isinstance(protection_cancel_outcomes, dict):
            raise BrokerStateError(
                "broker state 'protection_cancel_outcomes' must be a mapping"
            )
        return cls(
            schema_version=version,
            endpoint=payload.get("endpoint", ""),
            account_id=payload.get("account_id"),
            cash=payload.get("cash"),
            equity=payload.get("equity"),
            buying_power=payload.get("buying_power"),
            currency=payload.get("currency"),
            trading_blocked=payload.get("trading_blocked"),
            account_blocked=payload.get("account_blocked"),
            confirmed=bool(payload.get("confirmed", False)),
            confirmed_at=payload.get("confirmed_at"),
            orders={k: dict(v) for k, v in orders.items()},
            applied_fills=set(applied),
            applied_fill_quantities={
                k: _as_non_negative_float(v)
                for k, v in applied_fill_quantities.items()
            },
            applied_execution_ids={
                k: [str(i) for i in v]
                for k, v in applied_execution_ids.items()
                if isinstance(v, list)
            },
            positions={k: dict(v) for k, v in positions.items()},
            positions_at=payload.get("positions_at"),
            daily_orders_count=_as_non_negative_int(payload.get("daily_orders_count"), 0),
            daily_orders_date=payload.get("daily_orders_date") or None,
            protection_order_pairs={k: dict(v) for k, v in protection_pairs.items()},
            protection_levels={k: dict(v) for k, v in protection_levels.items()},
            protection_residuals={k: dict(v) for k, v in protection_residuals.items()},
            protection_cancel_outcomes={
                k: dict(v) for k, v in protection_cancel_outcomes.items()
            },
        )


class BrokerStateStore:
    """Loads and saves :class:`BrokerAccountState` for one broker path.

    A missing file is not an error: it means no order has been placed yet, so an
    empty state is returned. A malformed or truncated file *is* an error — a
    recovery that guessed at a corrupt state would resume with a wrong idea of
    what the broker already knows, so it fails instead.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> BrokerAccountState:
        if not self.path.exists():
            return BrokerAccountState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BrokerStateError(f"broker state {self.path} is not usable: {exc}") from exc
        return BrokerAccountState.from_dict(payload)

    def save(self, state: BrokerAccountState) -> Path:
        _atomic_write(self.path, json.dumps(state.to_dict(), indent=2) + "\n")
        return self.path
