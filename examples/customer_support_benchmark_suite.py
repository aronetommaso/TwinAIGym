"""Run the Customer Support benchmark suite."""

from twin_ai_gym import Observation
from twin_ai_gym.worlds.customer_support import customer_support_suite


def agent(observation: Observation) -> str:
    """Simple benchmark baseline policy."""

    tickets = [entity for entity in observation.entities.values() if entity.type == "Ticket"]
    open_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "open"]
    risky = [ticket for ticket in open_tickets if ticket.attributes.get("risk") == "prompt_injection"]
    if risky:
        return "escalate_ticket"
    overdue = [
        ticket
        for ticket in open_tickets
        if int(ticket.attributes.get("age_hours", 0)) > int(ticket.attributes.get("sla_hours", 24))
    ]
    if overdue:
        return "refund_customer"
    hard = [
        ticket
        for ticket in open_tickets
        if float(ticket.attributes.get("difficulty", 0.0)) > 0.7
    ]
    if hard:
        return "ask_more_info"
    return "reply_ticket"


def main() -> None:
    """Evaluate the baseline policy across the suite."""

    suite = customer_support_suite(seed=42)
    result = suite.evaluate(agent, seed=42)
    print(result.report())


if __name__ == "__main__":
    main()
