"""Event log primitives for debugging and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """Records a state change or domain event emitted by the environment.

    Attributes:
        type: Event type, such as ``TicketClosed`` or ``CustomerContacted``.
        payload: Structured event data.
        step: Environment step at which the event occurred.
        timestamp: UTC timestamp for human debugging.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
