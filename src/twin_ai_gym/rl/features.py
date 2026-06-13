"""Fixed-size vector encoders for classical RL and optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.observation import Observation, ObservationPolicy
from twin_ai_gym.core.world import WorldState


@dataclass(frozen=True, slots=True)
class VectorObservation:
    """Fixed-size numeric observation with stable feature names."""

    values: tuple[float, ...]
    feature_names: tuple[str, ...]

    def to_list(self) -> list[float]:
        """Return values as a mutable list for RL libraries."""

        return list(self.values)

    def to_dict(self) -> dict[str, float]:
        """Return a feature-name mapping."""

        return dict(zip(self.feature_names, self.values, strict=True))


class VectorObservationEncoder(Protocol):
    """Protocol for converting graph observations into numeric feature vectors."""

    feature_names: tuple[str, ...]

    def encode(self, env: TwinEnv, observation: Observation | None = None) -> VectorObservation:
        """Encode a TwinAIGym environment state."""


class VectorObservationPolicy(ObservationPolicy):
    """Observation policy that stores vector features under ``observation.metrics``."""

    def __init__(self, encoder: VectorObservationEncoder) -> None:
        """Initialize the policy."""

        self.encoder = encoder

    def observe(self, state: WorldState, agent_id: str | None = None) -> Observation:
        """Return an observation carrying vector metadata.

        This policy is useful for agents that still consume ``Observation`` but
        want a fixed-size vector in ``observation.metrics["vector"]``. The
        Gymnasium wrapper can also call encoders directly.
        """

        observation = Observation(
            entities={key: entity.copy() for key, entity in state.entities.items()},
            relations=dict(state.relations),
            events=[{"type": event.type, "payload": event.payload, "step": event.step} for event in state.events],
        )
        return observation


class CustomerSupportVectorEncoder:
    """Numerical feature encoder for ``CustomerSupportWorld``."""

    feature_names = (
        "open_tickets_ratio",
        "closed_tickets_ratio",
        "escalated_tickets_ratio",
        "average_satisfaction",
        "sla_violations_ratio",
        "average_priority",
        "max_priority",
        "average_difficulty",
        "max_difficulty",
        "average_age_ratio",
        "sla_pressure",
        "refund_cost_ratio",
        "specialist_load_ratio",
        "prompt_injection_ratio",
        "episode_progress",
    )

    def encode(self, env: TwinEnv, observation: Observation | None = None) -> VectorObservation:
        """Encode support queues, risk, cost, and episode progress."""

        observation = observation or env.observe()
        customers = [entity for entity in observation.entities.values() if entity.type == "Customer"]
        tickets = [entity for entity in observation.entities.values() if entity.type == "Ticket"]
        teams = [entity for entity in observation.entities.values() if entity.type == "Team"]
        total_tickets = max(1, len(tickets))
        open_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "open"]
        closed_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "closed"]
        escalated_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "escalated"]
        priorities = [float(ticket.attributes.get("priority", 0.0)) for ticket in tickets]
        difficulties = [float(ticket.attributes.get("difficulty", 0.0)) for ticket in tickets]
        age_ratios = [
            min(
                2.0,
                float(ticket.attributes.get("age_hours", 0.0)) / max(1.0, float(ticket.attributes.get("sla_hours", 24.0))),
            )
            for ticket in tickets
        ]
        sla_pressure = [
            1.0
            if ticket.attributes.get("status") == "open"
            and float(ticket.attributes.get("age_hours", 0.0)) >= 0.8 * float(ticket.attributes.get("sla_hours", 24.0))
            else 0.0
            for ticket in tickets
        ]
        values = (
            len(open_tickets) / total_tickets,
            len(closed_tickets) / total_tickets,
            len(escalated_tickets) / total_tickets,
            _average([float(customer.attributes.get("satisfaction", 0.0)) for customer in customers]),
            len([ticket for ticket in tickets if ticket.attributes.get("sla_breached")]) / total_tickets,
            _average(priorities),
            max(priorities, default=0.0),
            _average(difficulties),
            max(difficulties, default=0.0),
            min(1.0, _average(age_ratios)),
            _average(sla_pressure),
            min(1.0, sum(float(ticket.attributes.get("refund_paid", 0.0)) for ticket in tickets) / 500.0),
            min(1.0, sum(float(team.attributes.get("load", 0.0)) for team in teams) / 10.0),
            len([ticket for ticket in tickets if ticket.attributes.get("risk") == "prompt_injection"]) / total_tickets,
            min(1.0, env.world.step_count / max(1, env.max_steps)),
        )
        return VectorObservation(values=values, feature_names=self.feature_names)


class BusinessVectorEncoder:
    """Numerical feature encoder for generated business-process worlds."""

    feature_names = (
        "open_items_ratio",
        "closed_items_ratio",
        "average_quality",
        "max_quality",
        "average_risk",
        "max_risk",
        "average_priority",
        "max_priority",
        "average_age_ratio",
        "cost_ratio",
        "owner_health",
        "episode_progress",
    )

    def encode(self, env: TwinEnv, observation: Observation | None = None) -> VectorObservation:
        """Encode generic business process health."""

        observation = observation or env.observe()
        items = [entity for entity in observation.entities.values() if entity.attributes.get("kind") == "work_item"]
        owners = [entity for entity in observation.entities.values() if entity.attributes.get("health") is not None]
        total_items = max(1, len(items))
        open_items = [item for item in items if item.attributes.get("status") == "open"]
        closed_items = [item for item in items if item.attributes.get("status") == "closed"]
        qualities = [float(item.attributes.get("quality", 0.0)) for item in items]
        risks = [float(item.attributes.get("risk", 0.0)) for item in items]
        priorities = [float(item.attributes.get("priority", 0.0)) for item in items]
        ages = [min(1.0, float(item.attributes.get("age", 0.0)) / 20.0) for item in items]
        values = (
            len(open_items) / total_items,
            len(closed_items) / total_items,
            _average(qualities),
            max(qualities, default=0.0),
            _average(risks),
            max(risks, default=0.0),
            _average(priorities),
            max(priorities, default=0.0),
            _average(ages),
            min(1.0, sum(float(item.attributes.get("cost", 0.0)) for item in items) / 1000.0),
            _average([float(owner.attributes.get("health", 0.0)) for owner in owners]),
            min(1.0, env.world.step_count / max(1, env.max_steps)),
        )
        return VectorObservation(values=values, feature_names=self.feature_names)


class GenericGraphVectorEncoder:
    """Fallback graph-statistics encoder for custom worlds."""

    feature_names = (
        "entity_count_ratio",
        "relation_count_ratio",
        "event_count_ratio",
        "unique_entity_types_ratio",
        "average_degree_ratio",
        "episode_progress",
    )

    def __init__(self, max_entities: int = 100, max_relations: int = 250, max_events: int = 100) -> None:
        """Initialize normalization constants."""

        self.max_entities = max_entities
        self.max_relations = max_relations
        self.max_events = max_events

    def encode(self, env: TwinEnv, observation: Observation | None = None) -> VectorObservation:
        """Encode graph shape for arbitrary environments."""

        observation = observation or env.observe()
        entity_count = len(observation.entities)
        relation_count = len(observation.relations)
        types = {entity.type for entity in observation.entities.values()}
        average_degree = relation_count / max(1, entity_count)
        values = (
            min(1.0, entity_count / self.max_entities),
            min(1.0, relation_count / self.max_relations),
            min(1.0, len(observation.events) / self.max_events),
            min(1.0, len(types) / 25.0),
            min(1.0, average_degree / 10.0),
            min(1.0, env.world.step_count / max(1, env.max_steps)),
        )
        return VectorObservation(values=values, feature_names=self.feature_names)


def _average(values: list[float]) -> float:
    """Return a safe average."""

    return sum(values) / len(values) if values else 0.0
