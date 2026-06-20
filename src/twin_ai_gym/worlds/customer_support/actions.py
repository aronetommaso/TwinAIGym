"""Domain actions for the customer support world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from twin_ai_gym.core.action import Action
from twin_ai_gym.core.world import WorldState


def _current_ticket(state: WorldState) -> Any | None:
    """Return the highest-priority open ticket."""

    open_tickets = state.find_entities("Ticket", status="open")
    if not open_tickets:
        return None
    return sorted(
        open_tickets,
        key=lambda ticket: (
            -float(ticket.attributes.get("priority", 0.0)),
            int(ticket.attributes.get("age_hours", 0)),
        ),
    )[0]


def _ticket_customer(state: WorldState, ticket_id: str) -> Any | None:
    """Return the customer who owns a ticket."""

    customers = state.neighbors(ticket_id, relation_type="OWNS_TICKET", direction="in")
    return customers[0] if customers else None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a floating-point value."""

    return max(lower, min(upper, value))


@dataclass(slots=True)
class ReplyTicketAction(Action):
    """Reply to the current open ticket and attempt a lightweight resolution."""

    name: str = "reply_ticket"
    cost: float = 0.05

    def check_preconditions(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> str | None:
        """Validate that an open ticket exists."""

        if _current_ticket(state) is None:
            return "No open ticket is available."
        return None

    def apply_effects(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply support reply dynamics."""

        ticket = _current_ticket(state)
        if ticket is None:
            return {}
        customer = _ticket_customer(state, ticket.id)
        difficulty = float(ticket.attributes.get("difficulty", 0.5))
        resolution_probability = _clamp(0.85 - difficulty * 0.45)
        resolved = state.rng.random() < resolution_probability
        ticket.attributes["age_hours"] = int(ticket.attributes.get("age_hours", 0)) + 1
        ticket.attributes["touches"] = int(ticket.attributes.get("touches", 0)) + 1
        if resolved:
            ticket.attributes["status"] = "closed"
            ticket.attributes["resolution"] = "answered"
            state.emit("TicketClosed", ticket_id=ticket.id, resolution="answered")
        else:
            ticket.attributes["status"] = "open"
            state.emit("TicketReplied", ticket_id=ticket.id, resolved=False)
        if customer is not None:
            delta = 0.12 if resolved else 0.03
            customer.attributes["satisfaction"] = _clamp(
                float(customer.attributes.get("satisfaction", 0.5)) + delta
            )
        return {"ticket_id": ticket.id, "resolved": resolved}


@dataclass(slots=True)
class EscalateTicketAction(Action):
    """Escalate the current ticket to an engineering or specialist team."""

    name: str = "escalate_ticket"
    cost: float = 0.2

    def check_preconditions(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> str | None:
        """Validate that an open ticket exists."""

        if _current_ticket(state) is None:
            return "No open ticket is available."
        specialists = state.find_entities("Team", role="specialist")
        if not specialists:
            return "No specialist team exists."
        return None

    def apply_effects(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply escalation dynamics."""

        ticket = _current_ticket(state)
        if ticket is None:
            return {}
        team = state.find_entities("Team", role="specialist")[0]
        customer = _ticket_customer(state, ticket.id)
        ticket.attributes["status"] = "escalated"
        ticket.attributes["assigned_team"] = team.id
        ticket.attributes["age_hours"] = int(ticket.attributes.get("age_hours", 0)) + 2
        team.attributes["load"] = int(team.attributes.get("load", 0)) + 1
        if customer is not None:
            customer.attributes["satisfaction"] = _clamp(
                float(customer.attributes.get("satisfaction", 0.5)) + 0.08
            )
        state.emit("TicketEscalated", ticket_id=ticket.id, team_id=team.id)
        return {"ticket_id": ticket.id, "team_id": team.id}


@dataclass(slots=True)
class RefundCustomerAction(Action):
    """Offer a refund to reduce customer churn risk at a direct cost."""

    name: str = "refund_customer"
    cost: float = 0.4

    def check_preconditions(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> str | None:
        """Validate that an open or escalated ticket exists."""

        tickets = state.find_entities("Ticket", status="open") + state.find_entities("Ticket", status="escalated")
        if not tickets:
            return "No refundable ticket is available."
        return None

    def apply_effects(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply refund dynamics."""

        tickets = state.find_entities("Ticket", status="open") + state.find_entities("Ticket", status="escalated")
        ticket = sorted(tickets, key=lambda item: -float(item.attributes.get("priority", 0.0)))[0]
        customer = _ticket_customer(state, ticket.id)
        refund_amount = float(ticket.attributes.get("refund_amount", 25.0))
        ticket.attributes["status"] = "closed"
        ticket.attributes["resolution"] = "refunded"
        ticket.attributes["refund_paid"] = refund_amount
        if customer is not None:
            customer.attributes["satisfaction"] = _clamp(
                float(customer.attributes.get("satisfaction", 0.5)) + 0.2
            )
            customer.attributes["lifetime_value"] = max(
                0.0,
                float(customer.attributes.get("lifetime_value", 0.0)) - refund_amount,
            )
        state.emit("RefundIssued", ticket_id=ticket.id, amount=refund_amount)
        state.emit("TicketClosed", ticket_id=ticket.id, resolution="refunded")
        return {"ticket_id": ticket.id, "refund_amount": refund_amount}


@dataclass(slots=True)
class AskMoreInfoAction(Action):
    """Ask the customer for more information when the issue is ambiguous."""

    name: str = "ask_more_info"
    cost: float = 0.08

    def check_preconditions(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> str | None:
        """Validate that an open ticket exists."""

        if _current_ticket(state) is None:
            return "No open ticket is available."
        return None

    def apply_effects(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply information request dynamics."""

        ticket = _current_ticket(state)
        if ticket is None:
            return {}
        customer = _ticket_customer(state, ticket.id)
        ticket.attributes["age_hours"] = int(ticket.attributes.get("age_hours", 0)) + 4
        ticket.attributes["info_requested"] = True
        ticket.attributes["difficulty"] = max(0.05, float(ticket.attributes.get("difficulty", 0.5)) - 0.15)
        if customer is not None:
            customer.attributes["satisfaction"] = _clamp(
                float(customer.attributes.get("satisfaction", 0.5)) - 0.02
            )
        state.emit("CustomerContacted", ticket_id=ticket.id, reason="more_info")
        return {"ticket_id": ticket.id}


@dataclass(slots=True)
class IgnoreTicketAction(Action):
    """Let time pass without acting on the current ticket."""

    name: str = "ignore_ticket"
    cost: float = 0.0

    def check_preconditions(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> str | None:
        """Validate that an open ticket exists."""

        if _current_ticket(state) is None:
            return "No open ticket is available."
        return None

    def apply_effects(self, state: WorldState, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply passive aging and satisfaction decay."""

        ticket = _current_ticket(state)
        if ticket is None:
            return {}
        customer = _ticket_customer(state, ticket.id)
        ticket.attributes["age_hours"] = int(ticket.attributes.get("age_hours", 0)) + 8
        ticket.attributes["sla_breached"] = ticket.attributes["age_hours"] > int(
            ticket.attributes.get("sla_hours", 24)
        )
        if customer is not None:
            penalty = 0.08 if ticket.attributes["sla_breached"] else 0.04
            customer.attributes["satisfaction"] = _clamp(
                float(customer.attributes.get("satisfaction", 0.5)) - penalty
            )
        state.emit("TicketIgnored", ticket_id=ticket.id, sla_breached=ticket.attributes["sla_breached"])
        return {"ticket_id": ticket.id, "sla_breached": ticket.attributes["sla_breached"]}
