"""Run the graph-reasoning benchmark used in the project README."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from twin_ai_gym import ActionCommand, compare_agents
from twin_ai_gym.worlds.maintenance import (
    MaintenanceWorld,
    graph_aware_maintenance_agent,
    myopic_maintenance_agent,
)


class RandomMaintenanceAgent:
    """Seeded valid-action baseline."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: Any) -> ActionCommand:
        choices = [ActionCommand("wait")]
        for entity in observation.entities.values():
            if entity.type == "Component" and not entity.attributes.get("operational"):
                choices.append(ActionCommand("repair_component", {"component_id": entity.id}))
            elif entity.type == "Service":
                choices.append(ActionCommand("restart_service", {"service_id": entity.id}))
            elif entity.type == "Team":
                choices.append(ActionCommand("rebalance_team", {"team_id": entity.id}))
        return self.rng.choice(choices)


def main() -> None:
    """Execute 30 paired seeds and persist machine-readable and Markdown reports."""

    result = compare_agents(
        name="Maintenance causal graph reasoning",
        env_factory=lambda seed: MaintenanceWorld(seed=seed),
        agents={
            "graph-aware": graph_aware_maintenance_agent,
            "myopic-service-restart": myopic_maintenance_agent,
            "random": RandomMaintenanceAgent(seed=2026),
        },
        seeds=range(30),
    )
    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "maintenance_results.md").write_text(result.report() + "\n", encoding="utf-8")
    payload = {
        "benchmark": result.name,
        "seeds": list(result.seeds),
        "agents": {
            name: {
                "mean_score": agent.mean_score,
                "score_std": agent.score_std,
                "score_ci95": agent.score_ci95,
                "termination_rate": agent.termination_rate,
                "scores": list(agent.scores),
                "rewards": list(agent.rewards),
                "steps": list(agent.steps),
            }
            for name, agent in result.agents.items()
        },
    }
    (output_dir / "maintenance_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.report())


if __name__ == "__main__":
    main()
