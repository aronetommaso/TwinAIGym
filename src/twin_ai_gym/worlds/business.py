"""Reusable business-process benchmark worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twin_ai_gym.core.action import Action
from twin_ai_gym.core.benchmark import BenchmarkCase, BenchmarkSuite
from twin_ai_gym.core.entity import Entity
from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.relation import Relation
from twin_ai_gym.core.reward import RewardAggregator, RewardComponent
from twin_ai_gym.core.world import WorldSnapshot, WorldState


@dataclass(frozen=True, slots=True)
class BusinessProfile:
    """Configuration for a generated business-process world."""

    domain: str
    process_type: str
    work_item_type: str
    owner_type: str
    positive_action: str
    deep_action: str
    close_action: str
    risky_action: str
    idle_action: str
    goal_metric: str


class BusinessProgressReward(RewardComponent):
    """Reward progress, closed work, and penalize risk/cost."""

    name = "business_progress"
    weight = 1.0

    def compute(self, before: WorldSnapshot, after: WorldState, action_result: Any) -> float:
        """Compute generic business-process reward."""

        if not action_result.valid:
            return -1.0
        before_open = _count_items(before, "open")
        after_items = [entity for entity in after.entities.values() if entity.attributes.get("kind") == "work_item"]
        after_open = len([item for item in after_items if item.attributes.get("status") == "open"])
        closed_delta = before_open - after_open
        quality = _average(after_items, "quality")
        risk = _average(after_items, "risk")
        cost = sum(float(item.attributes.get("cost", 0.0)) for item in after_items)
        return closed_delta * 1.2 + quality * 0.2 - risk * 0.15 - min(0.4, cost / 1000.0)


class BusinessAction(Action):
    """Parameterized action used by the generic benchmark worlds."""

    def __init__(self, name: str, mode: str, cost: float = 0.05) -> None:
        """Initialize a business action."""

        self.name = name
        self.mode = mode
        self.cost = cost

    def check_preconditions(self, state: WorldState) -> str | None:
        """Validate that an actionable work item exists."""

        if _current_item(state) is None:
            return "No open work item is available."
        return None

    def apply_effects(self, state: WorldState) -> dict[str, Any]:
        """Apply a generic business transition."""

        item = _current_item(state)
        if item is None:
            return {}
        owner = _owner_for_item(state, item.id)
        item.attributes["age"] = int(item.attributes.get("age", 0)) + 1
        item.attributes["touches"] = int(item.attributes.get("touches", 0)) + 1

        if self.mode == "positive":
            item.attributes["quality"] = _clamp(float(item.attributes.get("quality", 0.4)) + 0.16)
            item.attributes["risk"] = _clamp(float(item.attributes.get("risk", 0.4)) - 0.08)
            event = "WorkItemImproved"
        elif self.mode == "deep":
            item.attributes["quality"] = _clamp(float(item.attributes.get("quality", 0.4)) + 0.26)
            item.attributes["cost"] = float(item.attributes.get("cost", 0.0)) + 35.0
            event = "WorkItemDeepened"
        elif self.mode == "close":
            quality = float(item.attributes.get("quality", 0.0))
            risk = float(item.attributes.get("risk", 0.0))
            closed = quality >= risk or state.rng.random() < max(0.1, quality - risk + 0.35)
            item.attributes["status"] = "closed" if closed else "open"
            item.attributes["outcome"] = "won" if closed else "stalled"
            event = "WorkItemClosed" if closed else "WorkItemStalled"
        elif self.mode == "risky":
            item.attributes["status"] = "closed"
            item.attributes["outcome"] = "forced"
            item.attributes["risk"] = _clamp(float(item.attributes.get("risk", 0.4)) + 0.2)
            item.attributes["cost"] = float(item.attributes.get("cost", 0.0)) + 75.0
            event = "RiskyShortcutTaken"
        else:
            item.attributes["risk"] = _clamp(float(item.attributes.get("risk", 0.4)) + 0.12)
            event = "WorkItemAged"

        if owner is not None:
            owner.attributes["health"] = _clamp(
                float(owner.attributes.get("health", 0.5)) + (0.04 if item.attributes.get("status") == "closed" else -0.02)
            )
        state.emit(event, item_id=item.id, action=self.name)
        return {"item_id": item.id, "mode": self.mode}


class BusinessProcessWorld(TwinEnv):
    """Base environment for generated business benchmark domains."""

    profile: BusinessProfile

    def __init__(self, seed: int | None = None, work_items: int = 7, max_steps: int = 60) -> None:
        """Create a generated business-process world."""

        world = WorldState(seed=seed)
        self._build_world(world, work_items)
        actions = {
            self.profile.positive_action: BusinessAction(self.profile.positive_action, "positive", 0.06),
            self.profile.deep_action: BusinessAction(self.profile.deep_action, "deep", 0.12),
            self.profile.close_action: BusinessAction(self.profile.close_action, "close", 0.08),
            self.profile.risky_action: BusinessAction(self.profile.risky_action, "risky", 0.25),
            self.profile.idle_action: BusinessAction(self.profile.idle_action, "idle", 0.0),
        }
        super().__init__(world=world, actions=actions, reward=RewardAggregator([BusinessProgressReward()]), max_steps=max_steps)

    def metrics(self) -> dict[str, float]:
        """Return generic process metrics."""

        items = [entity for entity in self.world.entities.values() if entity.attributes.get("kind") == "work_item"]
        closed = [item for item in items if item.attributes.get("status") == "closed"]
        return {
            self.profile.goal_metric: len(closed) / len(items) if items else 0.0,
            "average_quality": _average(items, "quality"),
            "average_risk": _average(items, "risk"),
            "process_cost": sum(float(item.attributes.get("cost", 0.0)) for item in items),
            "open_items": float(len(items) - len(closed)),
        }

    def score(self, total_reward: float, episodes: int, metrics: dict[str, float]) -> float:
        """Normalize generic business-process metrics."""

        goal = metrics.get(self.profile.goal_metric, 0.0)
        quality = metrics.get("average_quality", 0.0)
        risk = metrics.get("average_risk", 0.0)
        cost_penalty = min(1.0, metrics.get("process_cost", 0.0) / 1000.0)
        return max(0.0, min(1.0, 0.5 * goal + 0.35 * quality - 0.1 * risk - 0.05 * cost_penalty))

    def is_done(self) -> bool:
        """Return true when all work items are closed."""

        return not [item for item in self.world.entities.values() if item.attributes.get("kind") == "work_item" and item.attributes.get("status") == "open"]

    def _build_world(self, world: WorldState, work_items: int) -> None:
        """Populate a generated graph."""

        world.add_entity(Entity(id="agent:operator", type="Agent", attributes={"role": self.profile.domain.lower()}))
        for index in range(max(2, work_items // 2)):
            owner_id = f"{self.profile.owner_type.lower()}:{index + 1}"
            world.add_entity(
                Entity(
                    id=owner_id,
                    type=self.profile.owner_type,
                    attributes={"health": round(0.45 + world.rng.random() * 0.35, 3)},
                )
            )
        owners = [entity.id for entity in world.find_entities(self.profile.owner_type)]
        for index in range(work_items):
            item_id = f"{self.profile.work_item_type.lower()}:{index + 1}"
            world.add_entity(
                Entity(
                    id=item_id,
                    type=self.profile.work_item_type,
                    attributes={
                        "kind": "work_item",
                        "status": "open",
                        "priority": round(world.rng.random(), 3),
                        "quality": round(0.2 + world.rng.random() * 0.45, 3),
                        "risk": round(0.25 + world.rng.random() * 0.55, 3),
                        "age": world.rng.randint(0, 10),
                        "cost": 0.0,
                    },
                )
            )
            world.add_relation(Relation(source=owners[index % len(owners)], type="OWNS", target=item_id))


class SalesWorld(BusinessProcessWorld):
    """Sales pipeline benchmark world."""

    profile = BusinessProfile("Sales", "pipeline", "Deal", "Account", "qualify_lead", "run_discovery", "close_deal", "discount_deal", "defer_deal", "win_rate")


class CRMWorld(BusinessProcessWorld):
    """CRM account-health benchmark world."""

    profile = BusinessProfile("CRM", "account", "AccountTask", "Account", "update_account", "research_account", "retain_account", "bulk_update", "skip_account", "retention_rate")


class StartupOpsWorld(BusinessProcessWorld):
    """Startup operations benchmark world."""

    profile = BusinessProfile("Startup Ops", "operations", "OpsTask", "Team", "prioritize_task", "investigate_task", "ship_task", "cut_scope", "park_task", "shipping_rate")


class LogisticsWorld(BusinessProcessWorld):
    """Logistics exception-management benchmark world."""

    profile = BusinessProfile("Logistics", "shipment", "Shipment", "Warehouse", "reroute_shipment", "inspect_shipment", "deliver_shipment", "expedite_shipment", "hold_shipment", "delivery_rate")


class ProcurementWorld(BusinessProcessWorld):
    """Procurement sourcing benchmark world."""

    profile = BusinessProfile("Procurement", "sourcing", "PurchaseRequest", "Supplier", "compare_supplier", "negotiate_terms", "approve_purchase", "emergency_buy", "delay_purchase", "approval_rate")


class HRWorld(BusinessProcessWorld):
    """HR case-management benchmark world."""

    profile = BusinessProfile("HR", "people", "HRCase", "Employee", "triage_case", "investigate_case", "resolve_case", "fast_track_case", "defer_case", "resolution_rate")


def business_suite(seed: int | None = None) -> BenchmarkSuite:
    """Return a suite spanning the built-in business-process worlds."""

    worlds = [SalesWorld, CRMWorld, StartupOpsWorld, LogisticsWorld, ProcurementWorld, HRWorld]
    return BenchmarkSuite(
        name="TwinAIGym Business Processes",
        cases=[
            BenchmarkCase(name=world.profile.domain.lower().replace(" ", "_"), env_factory=lambda world=world: world(seed=seed), threshold=0.35)
            for world in worlds
        ],
    )


def _current_item(state: WorldState) -> Entity | None:
    """Return the highest-priority open work item."""

    items = [entity for entity in state.entities.values() if entity.attributes.get("kind") == "work_item" and entity.attributes.get("status") == "open"]
    if not items:
        return None
    return sorted(items, key=lambda item: (-float(item.attributes.get("priority", 0.0)), -int(item.attributes.get("age", 0))))[0]


def _owner_for_item(state: WorldState, item_id: str) -> Entity | None:
    """Return the owner connected to a work item."""

    owners = state.neighbors(item_id, relation_type="OWNS", direction="in")
    return owners[0] if owners else None


def _count_items(snapshot: WorldSnapshot, status: str) -> int:
    """Count work items in a snapshot by status."""

    return len(
        [
            entity
            for entity in snapshot.entities.values()
            if entity.attributes.get("kind") == "work_item" and entity.attributes.get("status") == status
        ]
    )


def _average(entities: list[Entity], attribute: str) -> float:
    """Average an entity attribute."""

    if not entities:
        return 0.0
    return sum(float(entity.attributes.get(attribute, 0.0)) for entity in entities) / len(entities)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a floating-point value."""

    return max(lower, min(upper, value))
