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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BROKER_STATE_SCHEMA_VERSION = 1
"""The on-disk broker state format. Bumped when a reader must know the shape changed."""


class BrokerStateError(RuntimeError):
    """The persisted broker state is malformed, truncated, or a newer version."""


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
    """Client order ids whose fill has already been applied to this state."""
    applied_fills: set[str] = field(default_factory=set)

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
