"""Evaluate a deterministic customer support agent."""

from twin_ai_gym import Observation
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


class ScriptedSupportAgent:
    """Simple baseline agent for customer support benchmark demos."""

    def act(self, observation: Observation) -> str:
        """Choose an action from visible ticket state."""

        tickets = [entity for entity in observation.entities.values() if entity.type == "Ticket"]
        open_tickets = [ticket for ticket in tickets if ticket.attributes.get("status") == "open"]
        if not open_tickets:
            return "reply_ticket"
        ticket = sorted(open_tickets, key=lambda item: -float(item.attributes.get("priority", 0.0)))[0]
        if ticket.attributes.get("risk") == "prompt_injection":
            return "escalate_ticket"
        if float(ticket.attributes.get("difficulty", 0.0)) > 0.7:
            return "ask_more_info"
        if int(ticket.attributes.get("age_hours", 0)) > int(ticket.attributes.get("sla_hours", 24)):
            return "refund_customer"
        return "reply_ticket"


def main() -> None:
    """Run a benchmark-style evaluation."""

    env = CustomerSupportWorld.adversarial(seed=42)
    result = env.evaluate(ScriptedSupportAgent(), seed=42)
    print(result.report())
    print(f"Passed benchmark: {result.passed(threshold=0.55)}")


if __name__ == "__main__":
    main()
