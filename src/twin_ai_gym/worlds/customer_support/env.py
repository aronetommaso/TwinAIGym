"""Customer support world implemented with the public TwinAIGym core API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twin_ai_gym.core.entity import Entity
from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.observation import FullObservation, ObservationPolicy
from twin_ai_gym.core.relation import Relation
from twin_ai_gym.core.reward import RewardAggregator
from twin_ai_gym.core.world import WorldState
from twin_ai_gym.worlds.customer_support.actions import (
    AskMoreInfoAction,
    EscalateTicketAction,
    IgnoreTicketAction,
    RefundCustomerAction,
    ReplyTicketAction,
)
from twin_ai_gym.worlds.customer_support.rewards import (
    RefundCostReward,
    ResolutionReward,
    SatisfactionReward,
    SlaPenaltyReward,
)


class CustomerSupportWorld(TwinEnv):
    """Ready-to-use digital twin for customer support agent evaluation.

    The environment models customers, support tickets, a support agent, and a
    specialist team. Actions mutate the graph and emit events; reward is computed
    from satisfaction, resolution rate, SLA violations, and refund cost.
    """

    def __init__(
        self,
        customers: int = 3,
        tickets: int = 5,
        seed: int | None = None,
        max_steps: int = 50,
        observation: ObservationPolicy | None = None,
    ) -> None:
        """Initialize a generated customer support world.

        Args:
            customers: Number of synthetic customers to create.
            tickets: Number of synthetic tickets to create.
            seed: Optional deterministic seed.
            max_steps: Maximum episode length.
            observation: Optional observation policy.
        """

        world = WorldState(seed=seed)
        self._build_default_world(world, customers=customers, tickets=tickets)
        actions = {
            "reply_ticket": ReplyTicketAction(),
            "escalate_ticket": EscalateTicketAction(),
            "refund_customer": RefundCustomerAction(),
            "ask_more_info": AskMoreInfoAction(),
            "ignore_ticket": IgnoreTicketAction(),
        }
        rewards = RewardAggregator(
            [
                SatisfactionReward(),
                ResolutionReward(),
                SlaPenaltyReward(),
                RefundCostReward(),
            ]
        )
        super().__init__(
            world=world,
            actions=actions,
            reward=rewards,
            observation=observation or FullObservation(),
            max_steps=max_steps,
        )

    def add_customer(
        self,
        customer_id: str,
        satisfaction: float = 0.6,
        lifetime_value: float = 500.0,
        segment: str = "standard",
    ) -> None:
        """Add a customer through a domain-native API."""

        self.world.add_entity(
            Entity(
                id=customer_id,
                type="Customer",
                attributes={
                    "satisfaction": satisfaction,
                    "lifetime_value": lifetime_value,
                    "segment": segment,
                },
            )
        )

    def add_ticket(
        self,
        ticket_id: str,
        customer_id: str,
        priority: float = 0.5,
        difficulty: float = 0.5,
        sla_hours: int = 24,
        refund_amount: float = 25.0,
    ) -> None:
        """Add a support ticket and link it to a customer."""

        self.world.add_entity(
            Entity(
                id=ticket_id,
                type="Ticket",
                attributes={
                    "status": "open",
                    "priority": priority,
                    "difficulty": difficulty,
                    "age_hours": 0,
                    "sla_hours": sla_hours,
                    "sla_breached": False,
                    "touches": 0,
                    "refund_amount": refund_amount,
                },
            )
        )
        self.world.add_relation(Relation(source=customer_id, type="OWNS_TICKET", target=ticket_id))

    def metrics(self) -> dict[str, float]:
        """Return high-level support metrics derived from the graph."""

        customers = self.world.find_entities("Customer")
        tickets = self.world.find_entities("Ticket")
        avg_satisfaction = (
            sum(float(customer.attributes.get("satisfaction", 0.0)) for customer in customers) / len(customers)
            if customers
            else 0.0
        )
        closed = [ticket for ticket in tickets if ticket.attributes.get("status") == "closed"]
        breached = [ticket for ticket in tickets if ticket.attributes.get("sla_breached")]
        refund_cost = sum(float(ticket.attributes.get("refund_paid", 0.0)) for ticket in tickets)
        return {
            "average_satisfaction": avg_satisfaction,
            "resolution_rate": len(closed) / len(tickets) if tickets else 0.0,
            "sla_violations": float(len(breached)),
            "refund_cost": refund_cost,
            "open_tickets": float(len([ticket for ticket in tickets if ticket.attributes.get("status") == "open"])),
        }

    def is_done(self) -> bool:
        """Return true when all tickets are closed or escalated."""

        actionable = [
            ticket
            for ticket in self.world.find_entities("Ticket")
            if ticket.attributes.get("status") in {"open", "escalated"}
        ]
        return len(actionable) == 0

    @classmethod
    def from_scenario(cls, path: str | Path, seed: int | None = None) -> "CustomerSupportWorld":
        """Create a world from a YAML scenario file.

        Args:
            path: YAML file with ``customers`` and ``tickets`` sections.
            seed: Optional deterministic seed.

        Raises:
            ImportError: If PyYAML is not installed.
        """

        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install twin-ai-gym[yaml] to load YAML scenarios.") from exc

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        env = cls(customers=0, tickets=0, seed=seed)
        for customer in data.get("customers", []):
            env.add_customer(**customer)
        for ticket in data.get("tickets", []):
            env.add_ticket(**ticket)
        env._initial_snapshot = env.world.snapshot()
        env.episode.initial_snapshot = env._initial_snapshot
        return env

    def _build_default_world(self, world: WorldState, customers: int, tickets: int) -> None:
        """Populate the generated demo world."""

        world.add_entity(
            Entity(
                id="agent:frontline",
                type="Agent",
                attributes={"role": "support_agent", "skill": 0.7},
            )
        )
        world.add_entity(
            Entity(
                id="team:specialist",
                type="Team",
                attributes={"role": "specialist", "load": 0},
            )
        )

        for index in range(customers):
            satisfaction = 0.45 + world.rng.random() * 0.35
            world.add_entity(
                Entity(
                    id=f"customer:{index + 1}",
                    type="Customer",
                    attributes={
                        "satisfaction": round(satisfaction, 3),
                        "lifetime_value": float(world.rng.randint(200, 1500)),
                        "segment": "premium" if index == 0 else "standard",
                    },
                )
            )

        if customers == 0:
            return

        for index in range(tickets):
            customer_id = f"customer:{(index % customers) + 1}"
            ticket_id = f"ticket:{index + 1}"
            world.add_entity(
                Entity(
                    id=ticket_id,
                    type="Ticket",
                    attributes={
                        "status": "open",
                        "priority": round(world.rng.random(), 3),
                        "difficulty": round(0.2 + world.rng.random() * 0.7, 3),
                        "age_hours": world.rng.randint(0, 18),
                        "sla_hours": 24,
                        "sla_breached": False,
                        "touches": 0,
                        "refund_amount": float(world.rng.choice([15, 25, 50, 100])),
                    },
                )
            )
            world.add_relation(Relation(source=customer_id, type="OWNS_TICKET", target=ticket_id))
