"""Relation primitives for the digital twin graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Relation:
    """Represents a typed directed edge between two entities.

    Attributes:
        source: Identifier of the source entity.
        type: Relation type, such as ``OWNS``, ``ASSIGNED_TO``, or ``AFFECTS``.
        target: Identifier of the target entity.
        attributes: Optional metadata for the edge.
    """

    source: str
    type: str
    target: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> tuple[str, str, str]:
        """Return the stable relation key used by the world state."""

        return (self.source, self.type, self.target)
