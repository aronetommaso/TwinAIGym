"""Evaluate one baseline policy across all built-in business worlds."""

from twin_ai_gym import Observation
from twin_ai_gym.worlds import business_suite


class BusinessBaselineAgent:
    """Simple process policy that improves risky items and closes mature items."""

    def act(self, observation: Observation) -> str:
        """Choose a domain action from the visible process graph."""

        actions_by_type = {
            "Deal": ("qualify_lead", "run_discovery", "close_deal"),
            "AccountTask": ("update_account", "research_account", "retain_account"),
            "OpsTask": ("prioritize_task", "investigate_task", "ship_task"),
            "Shipment": ("reroute_shipment", "inspect_shipment", "deliver_shipment"),
            "PurchaseRequest": ("compare_supplier", "negotiate_terms", "approve_purchase"),
            "HRCase": ("triage_case", "investigate_case", "resolve_case"),
        }
        work_items = [
            entity
            for entity in observation.entities.values()
            if entity.attributes.get("kind") == "work_item" and entity.attributes.get("status") == "open"
        ]
        if not work_items:
            return "qualify_lead"
        item = sorted(work_items, key=lambda value: -float(value.attributes.get("priority", 0.0)))[0]
        positive, deep, close = actions_by_type[item.type]
        quality = float(item.attributes.get("quality", 0.0))
        risk = float(item.attributes.get("risk", 0.0))
        if quality > risk + 0.15:
            return close
        if risk > quality:
            return deep
        return positive


def main() -> None:
    """Run the business-process suite."""

    result = business_suite(seed=11).evaluate(BusinessBaselineAgent(), seed=11)
    print(result.report())


if __name__ == "__main__":
    main()
