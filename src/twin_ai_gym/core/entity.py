"""Entity primitives for the digital twin graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Entity:
    """Represents a typed node in a digital twin Knowledge Graph.

    Attributes:
        id: Stable unique identifier for the entity.
        type: Domain type, such as ``Customer``, ``Ticket``, or ``Employee``.
        attributes: Mutable state variables and metadata associated with the entity.
    """

    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Entity":
        """Return a deep-enough copy for snapshotting primitive state."""

        return Entity(id=self.id, type=self.type, attributes=dict(self.attributes))
