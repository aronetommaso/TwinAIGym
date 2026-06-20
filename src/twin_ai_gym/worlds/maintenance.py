"""Causal maintenance world for graph-reasoning agent evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from twin_ai_gym.core.action import Action, ActionCommand, ActionParameter, ActionResult
from twin_ai_gym.core.dynamics import PropagationRule, RuleBasedTransitionModel
from twin_ai_gym.core.entity import Entity
from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.observation import FullObservation, ObservationPolicy
from twin_ai_gym.core.relation import Relation
from twin_ai_gym.core.reward import RewardAggregator, RewardComponent
from twin_ai_gym.core.world import WorldSnapshot, WorldState


class RepairComponent(Action):
    """Repair a named infrastructure component."""

    name = "repair_component"
    cost = 0.18
    parameters = {
        "component_id": ActionParameter(str, description="Component node to repair."),
    }

    def check_preconditions(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> str | None:
        component_id = str((parameters or {}).get("component_id", ""))
        entity = state.entities.get(component_id)
        if entity is None or entity.type != "Component":
            return f"Unknown component: {component_id}"
        if entity.attributes.get("operational"):
            return f"Component is already operational: {component_id}"
        return None

    def apply_effects(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        component = state.get_entity(str((parameters or {})["component_id"]))
        component.attributes.update(
            operational=True,
            health=1.0,
            incident_reported=False,
        )
        state.emit("ComponentRepaired", component_id=component.id)
        return {"component_id": component.id}


class RestartService(Action):
    """Restart a named service without repairing its dependencies."""

    name = "restart_service"
    cost = 0.08
    parameters = {
        "service_id": ActionParameter(str, description="Service node to restart."),
    }

    def check_preconditions(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> str | None:
        service_id = str((parameters or {}).get("service_id", ""))
        entity = state.entities.get(service_id)
        if entity is None or entity.type != "Service":
            return f"Unknown service: {service_id}"
        return None

    def apply_effects(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = state.get_entity(str((parameters or {})["service_id"]))
        service.attributes.update(operational=True, health=max(0.8, float(service.attributes["health"])))
        state.emit("ServiceRestarted", service_id=service.id)
        return {"service_id": service.id}


class RebalanceTeam(Action):
    """Reduce overload for a named owning team."""

    name = "rebalance_team"
    cost = 0.12
    parameters = {
        "team_id": ActionParameter(str, description="Team node to rebalance."),
    }

    def check_preconditions(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> str | None:
        team_id = str((parameters or {}).get("team_id", ""))
        entity = state.entities.get(team_id)
        if entity is None or entity.type != "Team":
            return f"Unknown team: {team_id}"
        return None

    def apply_effects(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        team = state.get_entity(str((parameters or {})["team_id"]))
        team.attributes.update(load=max(0.0, float(team.attributes["load"]) - 0.45), overloaded=False)
        state.emit("TeamRebalanced", team_id=team.id)
        return {"team_id": team.id}


class Wait(Action):
    """Advance the incident without intervention."""

    name = "wait"

    def apply_effects(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state.emit("TimeAdvanced")
        return {}


class AvailabilityReward(RewardComponent):
    """Reward restored graph nodes, weighted by business criticality."""

    name = "availability"
    weight = 2.0

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        return _weighted_availability(after.entities) - _weighted_availability(before.entities)


class HealthReward(RewardComponent):
    """Reward improvements in infrastructure health."""

    name = "health"
    weight = 0.5

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        return _average_health(after.entities) - _average_health(before.entities)


class OverloadPenalty(RewardComponent):
    """Penalize newly overloaded ownership teams."""

    name = "team_overload"
    weight = 0.5

    def compute(
        self,
        before: WorldSnapshot,
        after: WorldState,
        action_result: ActionResult,
    ) -> float:
        old = sum(
            bool(entity.attributes.get("overloaded"))
            for entity in before.entities.values()
            if entity.type == "Team"
        )
        new = sum(
            bool(entity.attributes.get("overloaded"))
            for entity in after.entities.values()
            if entity.type == "Team"
        )
        return float(old - new)


class MaintenanceWorld(TwinEnv):
    """Infrastructure incident world with delayed graph propagation."""

    def __init__(
        self,
        seed: int | None = None,
        observation: ObservationPolicy | None = None,
        max_steps: int = 8,
    ) -> None:
        world = WorldState(seed=seed)
        self._build_world(world)
        transition_model = RuleBasedTransitionModel(
            rules=[
                PropagationRule(
                    "dependency_failure",
                    _propagate_dependency_failures,
                    description="A service fails while any DEPENDS_ON target is unavailable.",
                ),
                PropagationRule(
                    "incident_ownership_load",
                    _propagate_owner_load,
                    description="A failed component increases load on its owning team.",
                ),
                PropagationRule(
                    "overload_degradation",
                    _propagate_overload,
                    description="Overloaded teams degrade the components they own.",
                ),
            ]
        )
        super().__init__(
            world=world,
            actions={
                "repair_component": RepairComponent(),
                "restart_service": RestartService(),
                "rebalance_team": RebalanceTeam(),
                "wait": Wait(),
            },
            reward=RewardAggregator([AvailabilityReward(), HealthReward(), OverloadPenalty()]),
            observation=observation or FullObservation(),
            transition_model=transition_model,
            max_steps=max_steps,
        )

    def metrics(self) -> dict[str, float]:
        """Return incident recovery metrics derived from the graph."""

        return {
            "weighted_availability": _weighted_availability(self.world.entities),
            "average_health": _average_health(self.world.entities),
            "overloaded_teams": float(
                sum(
                    bool(entity.attributes.get("overloaded"))
                    for entity in self.world.entities.values()
                    if entity.type == "Team"
                )
            ),
            "causal_rules_fired": float(
                sum(event.type == "CausalRuleFired" for event in self.world.events)
            ),
        }

    def score(self, total_reward: float, episodes: int, metrics: dict[str, float]) -> float:
        """Normalize final recovery quality to [0, 1]."""

        score = (
            0.7 * metrics.get("weighted_availability", 0.0)
            + 0.3 * metrics.get("average_health", 0.0)
            - 0.1 * min(1.0, metrics.get("overloaded_teams", 0.0))
        )
        return max(0.0, min(1.0, score))

    def is_done(self) -> bool:
        """Finish when all infrastructure is available and teams are stable."""

        infrastructure = [
            entity
            for entity in self.world.entities.values()
            if entity.type in {"Service", "Component"}
        ]
        teams = self.world.find_entities("Team")
        return all(item.attributes.get("operational") for item in infrastructure) and not any(
            team.attributes.get("overloaded") for team in teams
        )

    @staticmethod
    def _build_world(world: WorldState) -> None:
        cache_failed = world.rng.random() < 0.3
        world.add_entity(
            Entity(
                "service:api",
                "Service",
                {
                    "operational": False,
                    "health": round(0.25 + world.rng.random() * 0.2, 3),
                    "criticality": 1.0,
                },
            )
        )
        world.add_entity(
            Entity(
                "service:worker",
                "Service",
                {
                    "operational": True,
                    "health": round(0.65 + world.rng.random() * 0.2, 3),
                    "criticality": 0.7,
                },
            )
        )
        world.add_entity(
            Entity(
                "component:db",
                "Component",
                {
                    "operational": False,
                    "health": round(0.05 + world.rng.random() * 0.1, 3),
                    "criticality": 1.0,
                },
            )
        )
        world.add_entity(
            Entity(
                "component:cache",
                "Component",
                {
                    "operational": not cache_failed,
                    "health": (
                        round(0.08 + world.rng.random() * 0.12, 3)
                        if cache_failed
                        else round(0.82 + world.rng.random() * 0.15, 3)
                    ),
                    "criticality": 0.6,
                },
            )
        )
        world.add_entity(
            Entity(
                "team:platform",
                "Team",
                {"load": round(0.55 + world.rng.random() * 0.2, 3), "overloaded": False},
            )
        )
        world.add_relation(Relation("service:api", "DEPENDS_ON", "component:db"))
        world.add_relation(Relation("service:api", "DEPENDS_ON", "component:cache"))
        world.add_relation(Relation("service:worker", "DEPENDS_ON", "component:db"))
        world.add_relation(Relation("component:db", "OWNED_BY", "team:platform"))
        world.add_relation(Relation("component:cache", "OWNED_BY", "team:platform"))


def _propagate_dependency_failures(
    state: WorldState,
    result: ActionResult,
) -> dict[str, Any] | None:
    affected: list[str] = []
    for service in state.find_entities("Service"):
        dependencies = state.neighbors(service.id, "DEPENDS_ON", "out")
        if dependencies and any(not item.attributes.get("operational") for item in dependencies):
            if service.attributes.get("operational"):
                affected.append(service.id)
            service.attributes["operational"] = False
            service.attributes["health"] = max(0.0, float(service.attributes["health"]) - 0.12)
    return {"affected": affected} if affected else None


def _propagate_owner_load(
    state: WorldState,
    result: ActionResult,
) -> dict[str, Any] | None:
    loaded: list[str] = []
    for component in state.find_entities("Component"):
        if component.attributes.get("operational") or component.attributes.get("incident_reported"):
            continue
        component.attributes["incident_reported"] = True
        for team in state.neighbors(component.id, "OWNED_BY", "out"):
            team.attributes["load"] = min(1.0, float(team.attributes["load"]) + 0.35)
            team.attributes["overloaded"] = team.attributes["load"] >= 0.85
            loaded.append(team.id)
    return {"teams": loaded} if loaded else None


def _propagate_overload(
    state: WorldState,
    result: ActionResult,
) -> dict[str, Any] | None:
    degraded: list[str] = []
    for team in state.find_entities("Team"):
        if not team.attributes.get("overloaded"):
            continue
        for component in state.neighbors(team.id, "OWNED_BY", "in"):
            component.attributes["health"] = max(0.0, float(component.attributes["health"]) - 0.08)
            degraded.append(component.id)
    return {"components": degraded} if degraded else None


def _weighted_availability(entities: Mapping[str, Entity]) -> float:
    infrastructure = [
        entity for entity in entities.values() if entity.type in {"Service", "Component"}
    ]
    denominator = sum(float(item.attributes.get("criticality", 1.0)) for item in infrastructure)
    if not denominator:
        return 0.0
    return sum(
        float(item.attributes.get("criticality", 1.0))
        for item in infrastructure
        if item.attributes.get("operational")
    ) / denominator


def _average_health(entities: Mapping[str, Entity]) -> float:
    infrastructure = [
        entity for entity in entities.values() if entity.type in {"Service", "Component"}
    ]
    if not infrastructure:
        return 0.0
    return sum(float(item.attributes.get("health", 0.0)) for item in infrastructure) / len(
        infrastructure
    )


def graph_aware_maintenance_agent(observation: Any) -> ActionCommand:
    """Reference policy that follows graph dependencies to root causes."""

    failed_services = {
        entity.id
        for entity in observation.entities.values()
        if entity.type == "Service" and not entity.attributes.get("operational")
    }
    for relation in observation.relations.values():
        dependency = observation.entities.get(relation.target)
        if (
            relation.type == "DEPENDS_ON"
            and relation.source in failed_services
            and dependency is not None
            and not dependency.attributes.get("operational")
        ):
            return ActionCommand("repair_component", {"component_id": dependency.id})
    for relation in observation.relations.values():
        team = observation.entities.get(relation.target)
        if (
            relation.type == "OWNED_BY"
            and team is not None
            and team.attributes.get("overloaded")
        ):
            return ActionCommand("rebalance_team", {"team_id": team.id})
    if failed_services:
        return ActionCommand("restart_service", {"service_id": sorted(failed_services)[0]})
    return ActionCommand("wait")


def myopic_maintenance_agent(observation: Any) -> ActionCommand:
    """Baseline that treats the visible service symptom as the root cause."""

    for entity in observation.entities.values():
        if entity.type == "Service" and not entity.attributes.get("operational"):
            return ActionCommand("restart_service", {"service_id": entity.id})
    return ActionCommand("wait")
