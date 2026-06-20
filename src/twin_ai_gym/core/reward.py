"""Reward components and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from twin_ai_gym.core.action import ActionResult
from twin_ai_gym.core.world import WorldSnapshot, WorldState


class RewardComponent:
    """Base class for computing a reward contribution from graph transitions."""

    name = "reward"
    weight = 1.0
    description = ""

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        """Return a scalar reward contribution."""

        raise NotImplementedError


@dataclass(slots=True)
class RewardBreakdown:
    """Reward result with component-level details."""

    total: float
    components: dict[str, float]
    attribution: dict[str, RewardAttribution] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardAttribution:
    """Auditable contribution of one business objective to scalar reward."""

    component: str
    raw_value: float
    weight: float
    contribution: float
    description: str = ""


class RewardAggregator:
    """Combines weighted reward components."""

    def __init__(self, components: list[RewardComponent] | None = None) -> None:
        """Initialize the aggregator."""

        self.components = components or []

    def add(self, component: RewardComponent) -> None:
        """Register a reward component."""

        self.components.append(component)

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> RewardBreakdown:
        """Compute total reward and component-level contributions."""

        values: dict[str, float] = {}
        attribution: dict[str, RewardAttribution] = {}
        total = 0.0
        if not action_result.valid:
            values["invalid_action"] = -1.0
            total -= 1.0
        for component in self.components:
            raw_value = component.compute(before, after, action_result)
            value = raw_value * component.weight
            values[component.name] = value
            attribution[component.name] = RewardAttribution(
                component=component.name,
                raw_value=raw_value,
                weight=component.weight,
                contribution=value,
                description=component.description or (component.__doc__ or "").strip().splitlines()[0],
            )
            total += value
        total -= action_result.cost
        if action_result.cost:
            values["action_cost"] = -action_result.cost
        return RewardBreakdown(total=total, components=values, attribution=attribution)
