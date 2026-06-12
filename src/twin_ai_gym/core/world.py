"""Mutable graph state, snapshots, diffs, and rollback support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import copy
import random

from twin_ai_gym.core.entity import Entity
from twin_ai_gym.core.events import Event
from twin_ai_gym.core.relation import Relation


@dataclass(slots=True)
class WorldSnapshot:
    """Immutable snapshot of a world state.

    Attributes:
        entities: Entity mapping captured at snapshot time.
        relations: Relation mapping captured at snapshot time.
        events: Event log captured at snapshot time.
        step_count: Step counter captured at snapshot time.
        rng_state: Python random generator state for deterministic rollback.
    """

    entities: dict[str, Entity]
    relations: dict[tuple[str, str, str], Relation]
    events: list[Event]
    step_count: int
    rng_state: object


@dataclass(slots=True)
class StateDiff:
    """Describes changes between two world snapshots."""

    added_entities: dict[str, Entity] = field(default_factory=dict)
    removed_entities: dict[str, Entity] = field(default_factory=dict)
    changed_entities: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    added_relations: dict[tuple[str, str, str], Relation] = field(default_factory=dict)
    removed_relations: dict[tuple[str, str, str], Relation] = field(default_factory=dict)
    new_events: list[Event] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return whether the diff contains no changes."""

        return not any(
            [
                self.added_entities,
                self.removed_entities,
                self.changed_entities,
                self.added_relations,
                self.removed_relations,
                self.new_events,
            ]
        )

    def summary(self) -> list[str]:
        """Return a compact human-readable summary of state changes."""

        lines: list[str] = []
        for entity_id, entity in self.added_entities.items():
            lines.append(f"Entity added: {entity_id} ({entity.type})")
        for entity_id, entity in self.removed_entities.items():
            lines.append(f"Entity removed: {entity_id} ({entity.type})")
        for entity_id, changes in self.changed_entities.items():
            for key, (before, after) in changes.items():
                lines.append(f"{entity_id}.{key}: {before!r} -> {after!r}")
        for key in self.added_relations:
            lines.append(f"Relation added: {key[0]} -[{key[1]}]-> {key[2]}")
        for key in self.removed_relations:
            lines.append(f"Relation removed: {key[0]} -[{key[1]}]-> {key[2]}")
        for event in self.new_events:
            lines.append(f"Event: {event.type} {event.payload}")
        return lines


class WorldState:
    """Stateful Knowledge Graph backing a digital twin environment."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize an empty world.

        Args:
            seed: Optional deterministic seed for stochastic dynamics.
        """

        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str, str], Relation] = {}
        self.events: list[Event] = []
        self.step_count = 0
        self.rng = random.Random(seed)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the world.

        Args:
            entity: Entity to insert.

        Raises:
            ValueError: If an entity with the same ID already exists.
        """

        if entity.id in self.entities:
            raise ValueError(f"Entity already exists: {entity.id}")
        self.entities[entity.id] = entity

    def remove_entity(self, entity_id: str) -> Entity:
        """Remove an entity and all incident relations.

        Args:
            entity_id: Entity identifier.

        Returns:
            Removed entity.
        """

        entity = self.entities.pop(entity_id)
        for relation_id in list(self.relations):
            source, _, target = relation_id
            if source == entity_id or target == entity_id:
                del self.relations[relation_id]
        return entity

    def add_relation(self, relation: Relation) -> None:
        """Add a directed relation between existing entities."""

        if relation.source not in self.entities:
            raise KeyError(f"Unknown relation source: {relation.source}")
        if relation.target not in self.entities:
            raise KeyError(f"Unknown relation target: {relation.target}")
        self.relations[relation.id] = relation

    def remove_relation(self, relation_id: tuple[str, str, str]) -> Relation:
        """Remove a relation by key."""

        return self.relations.pop(relation_id)

    def get_entity(self, entity_id: str) -> Entity:
        """Return an entity by identifier."""

        return self.entities[entity_id]

    def find_entities(self, entity_type: str | None = None, **attributes: Any) -> list[Entity]:
        """Find entities matching a type and exact attribute filters."""

        results = []
        for entity in self.entities.values():
            if entity_type is not None and entity.type != entity_type:
                continue
            if all(entity.attributes.get(key) == value for key, value in attributes.items()):
                results.append(entity)
        return results

    def neighbors(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: str = "out",
    ) -> list[Entity]:
        """Return neighboring entities connected to an entity.

        Args:
            entity_id: Entity identifier.
            relation_type: Optional relation type filter.
            direction: ``"out"``, ``"in"``, or ``"both"``.
        """

        neighbors: list[Entity] = []
        for relation in self.relations.values():
            if relation_type is not None and relation.type != relation_type:
                continue
            if direction in {"out", "both"} and relation.source == entity_id:
                neighbors.append(self.entities[relation.target])
            if direction in {"in", "both"} and relation.target == entity_id:
                neighbors.append(self.entities[relation.source])
        return neighbors

    def emit(self, event_type: str, **payload: Any) -> Event:
        """Append an event to the world event log."""

        event = Event(type=event_type, payload=payload, step=self.step_count)
        self.events.append(event)
        return event

    def snapshot(self) -> WorldSnapshot:
        """Capture the current world state for rollback, replay, or diffing."""

        return WorldSnapshot(
            entities={key: entity.copy() for key, entity in self.entities.items()},
            relations=copy.deepcopy(self.relations),
            events=list(self.events),
            step_count=self.step_count,
            rng_state=self.rng.getstate(),
        )

    def rollback(self, snapshot: WorldSnapshot) -> None:
        """Restore the world to a previous snapshot."""

        self.entities = {key: entity.copy() for key, entity in snapshot.entities.items()}
        self.relations = copy.deepcopy(snapshot.relations)
        self.events = list(snapshot.events)
        self.step_count = snapshot.step_count
        self.rng.setstate(snapshot.rng_state)

    def diff(self, before: WorldSnapshot, after: WorldSnapshot | None = None) -> StateDiff:
        """Compute a diff between two snapshots.

        Args:
            before: Baseline snapshot.
            after: Optional target snapshot. Defaults to the current world state.
        """

        after = after or self.snapshot()
        diff = StateDiff()

        before_ids = set(before.entities)
        after_ids = set(after.entities)
        for entity_id in after_ids - before_ids:
            diff.added_entities[entity_id] = after.entities[entity_id]
        for entity_id in before_ids - after_ids:
            diff.removed_entities[entity_id] = before.entities[entity_id]
        for entity_id in before_ids & after_ids:
            before_entity = before.entities[entity_id]
            after_entity = after.entities[entity_id]
            changes = self._attribute_diff(before_entity.attributes, after_entity.attributes)
            if before_entity.type != after_entity.type:
                changes["type"] = (before_entity.type, after_entity.type)
            if changes:
                diff.changed_entities[entity_id] = changes

        before_relations = set(before.relations)
        after_relations = set(after.relations)
        for relation_id in after_relations - before_relations:
            diff.added_relations[relation_id] = after.relations[relation_id]
        for relation_id in before_relations - after_relations:
            diff.removed_relations[relation_id] = before.relations[relation_id]

        diff.new_events = after.events[len(before.events) :]
        return diff

    def clone(self) -> "WorldState":
        """Return a mutable clone of this world."""

        clone = WorldState()
        clone.rollback(self.snapshot())
        return clone

    @staticmethod
    def _attribute_diff(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """Compute shallow attribute changes for an entity."""

        changes: dict[str, tuple[Any, Any]] = {}
        for key in set(before) | set(after):
            old = before.get(key)
            new = after.get(key)
            if old != new:
                changes[key] = (old, new)
        return changes

    def entities_by_ids(self, entity_ids: Iterable[str]) -> dict[str, Entity]:
        """Return a subset of entities by identifiers."""

        return {entity_id: self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities}
