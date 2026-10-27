"""The ledger's event log: every state change recorded, in order, with a reason.

The ledger is auditable by replay, which is the only way to answer "why does the
book look like this?" after the fact. That requires the events to be a complete
and ordered account of every mutation: if a state change happens without an
event, replay diverges from the live book and the audit trail is a story rather
than a proof.

Events are immutable and appended, never rewritten. ``seq`` is a dense
monotonically increasing counter starting at 1, so a missing event is detectable
rather than invisible.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """The full vocabulary of ledger events.

    The order lifecycle events (``ORDER_*``) describe what was asked for; the
    position events (``POSITION_*``) describe what the book became; ``FEE_CHARGED``
    records a cost with no market side. Keeping them distinct means a rejected
    order is visible as a rejected order, not as an absence of a position event.
    """

    ORDER_CREATED = "ORDER_CREATED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_INCREASED = "POSITION_INCREASED"
    POSITION_REDUCED = "POSITION_REDUCED"
    POSITION_CLOSED = "POSITION_CLOSED"
    FEE_CHARGED = "FEE_CHARGED"


class LedgerEvent(BaseModel):
    """One immutable record of something that happened to the book."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=1, description="Dense sequence number, 1-based")
    timestamp: datetime = Field(description="When the event was recorded (UTC-aware)")
    type: EventType = Field(description="What happened")
    symbol: str | None = Field(default=None, description="Instrument, when the event concerns one")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific figures. Must contain no credentials and no free text from a model.",
    )
    reason: str = Field(
        default="",
        description="Short machine-ish explanation, e.g. why an order was rejected",
    )
