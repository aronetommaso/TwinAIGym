"""Observation policies for partial and full graph visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any

from twin_ai_gym.core.entity import Entity
from twin_ai_gym.core.relation import Relation
from twin_ai_gym.core.world import WorldState


@dataclass(slots=True)
class Observation:
    """Serializable graph observation returned to agents."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], Relation] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the observation to JSON-friendly primitives."""

        return {
            "entities": {
                entity_id: {"type": entity.type, "attributes": dict(entity.attributes)}
                for entity_id, entity in self.entities.items()
            },
            "relations": [
                {
                    "source": relation.source,
                    "type": relation.type,
                    "target": relation.target,
                    "attributes": dict(relation.attributes),
                }
                for relation in self.relations.values()
            ],
            "metrics": dict(self.metrics),
            "events": list(self.events),
        }


class ObservationPolicy:
    """Base class for converting world state into agent-visible observations."""

    def observe(self, state: WorldState, agent_id: str | None = None) -> Observation:
        """Return an observation for an agent."""

        raise NotImplementedError


class FullObservation(ObservationPolicy):
    """Observation policy that exposes the full graph state."""

    def observe(self, state: WorldState, agent_id: str | None = None) -> Observation:
        """Return every entity and relation in the world."""

        return Observation(
            entities={key: entity.copy() for key, entity in state.entities.items()},
            relations=dict(state.relations),
            events=[{"type": event.type, "payload": event.payload, "step": event.step} for event in state.events],
        )


class LocalSubgraphObservation(ObservationPolicy):
    """Observation policy that exposes a bounded neighborhood around an entity."""

    def __init__(self, root_entity_id: str | None = None, depth: int = 1) -> None:
        """Initialize the local subgraph policy.

        Args:
            root_entity_id: Default root entity if no agent ID is passed at observation time.
            depth: Maximum graph distance to include.
        """

        self.root_entity_id = root_entity_id
        self.depth = depth

    def observe(self, state: WorldState, agent_id: str | None = None) -> Observation:
        """Return a bounded local subgraph observation."""

        root = agent_id or self.root_entity_id
        if root is None:
            raise ValueError("LocalSubgraphObservation requires a root entity or agent ID.")
        if root not in state.entities:
            raise KeyError(f"Unknown observation root: {root}")

        seen = {root}
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        relation_ids: set[tuple[str, str, str]] = set()

        while queue:
            entity_id, distance = queue.popleft()
            if distance >= self.depth:
                continue
            for relation in state.relations.values():
                if relation.source == entity_id:
                    other = relation.target
                elif relation.target == entity_id:
                    other = relation.source
                else:
                    continue
                relation_ids.add(relation.id)
                if other not in seen:
                    seen.add(other)
                    queue.append((other, distance + 1))

        return Observation(
            entities={entity_id: state.entities[entity_id].copy() for entity_id in seen},
            relations={relation_id: state.relations[relation_id] for relation_id in relation_ids},
        )
