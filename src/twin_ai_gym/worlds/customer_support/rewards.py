"""Reward components for the customer support world."""

from __future__ import annotations

from twin_ai_gym.core.action import ActionResult
from twin_ai_gym.core.reward import RewardComponent
from twin_ai_gym.core.world import WorldSnapshot, WorldState


def _avg_satisfaction_from_entities(entities: dict) -> float:
    """Compute average customer satisfaction from entity mappings."""

    customers = [entity for entity in entities.values() if entity.type == "Customer"]
    if not customers:
        return 0.0
    return sum(float(customer.attributes.get("satisfaction", 0.0)) for customer in customers) / len(customers)


class SatisfactionReward(RewardComponent):
    """Reward improvements in average customer satisfaction."""

    name = "satisfaction"
    weight = 3.0

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        """Compute reward from satisfaction delta."""

        return _avg_satisfaction_from_entities(after.entities) - _avg_satisfaction_from_entities(before.entities)


class ResolutionReward(RewardComponent):
    """Reward tickets closed during the transition."""

    name = "resolution"
    weight = 1.0

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        """Compute reward from newly closed tickets."""

        before_closed = {
            entity_id
            for entity_id, entity in before.entities.items()
            if entity.type == "Ticket" and entity.attributes.get("status") == "closed"
        }
        after_closed = {
            entity_id
            for entity_id, entity in after.entities.items()
            if entity.type == "Ticket" and entity.attributes.get("status") == "closed"
        }
        return float(len(after_closed - before_closed))


class SlaPenaltyReward(RewardComponent):
    """Penalize SLA breaches introduced during a transition."""

    name = "sla_penalty"
    weight = 1.0

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        """Compute negative reward for newly breached tickets."""

        before_breached = {
            entity_id
            for entity_id, entity in before.entities.items()
            if entity.type == "Ticket" and entity.attributes.get("sla_breached")
        }
        after_breached = {
            entity_id
            for entity_id, entity in after.entities.items()
            if entity.type == "Ticket" and entity.attributes.get("sla_breached")
        }
        return -float(len(after_breached - before_breached))


class RefundCostReward(RewardComponent):
    """Penalize refund spend."""

    name = "refund_cost"
    weight = 0.01

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        """Compute negative reward from newly paid refunds."""

        total_before = sum(
            float(entity.attributes.get("refund_paid", 0.0))
            for entity in before.entities.values()
            if entity.type == "Ticket"
        )
        total_after = sum(
            float(entity.attributes.get("refund_paid", 0.0))
            for entity in after.entities.values()
            if entity.type == "Ticket"
        )
        return -(total_after - total_before)
