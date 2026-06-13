"""Marketplace-style metadata for environment packages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentPackage:
    """Describes an installable or built-in TwinAIGym environment package."""

    name: str
    domain: str
    import_path: str
    description: str
    built_in: bool = True


def list_environment_packages() -> list[EnvironmentPackage]:
    """Return known built-in and planned marketplace environment packages."""

    return [
        EnvironmentPackage(
            name="twinaigym-customer-support",
            domain="Customer Support",
            import_path="twin_ai_gym.worlds.customer_support",
            description="Support tickets, SLA pressure, refunds, and escalation.",
        ),
        EnvironmentPackage(
            name="twinaigym-sales",
            domain="Sales",
            import_path="twin_ai_gym.worlds.business.SalesWorld",
            description="Pipeline qualification, follow-up, negotiation, and closing.",
        ),
        EnvironmentPackage(
            name="twinaigym-crm",
            domain="CRM",
            import_path="twin_ai_gym.worlds.business.CRMWorld",
            description="Account hygiene, lifecycle updates, churn risk, and retention.",
        ),
        EnvironmentPackage(
            name="twinaigym-banking",
            domain="Banking",
            import_path="twinaigym_banking",
            description="External package slot for regulated banking workflows.",
            built_in=False,
        ),
    ]
