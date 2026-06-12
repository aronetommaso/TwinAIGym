"""Pytest-style regression test for an AI agent."""

from twin_ai_gym import Observation
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def conservative_agent(observation: Observation) -> str:
    """Return a conservative customer support action."""

    tickets = [entity for entity in observation.entities.values() if entity.type == "Ticket"]
    open_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "open"]
    risky = [ticket for ticket in open_tickets if ticket.attributes.get("risk") == "prompt_injection"]
    if risky:
        return "escalate_ticket"
    difficult = [
        ticket
        for ticket in open_tickets
        if float(ticket.attributes.get("difficulty", 0.0)) > 0.7
    ]
    if difficult:
        return "ask_more_info"
    return "reply_ticket"


def test_customer_support_agent_regression() -> None:
    """Example CI assertion for an agent release."""

    env = CustomerSupportWorld.adversarial(seed=7)
    result = env.evaluate(conservative_agent, seed=7)

    assert result.score > 0.45
    assert result.metrics["sla_violations"] <= 5
