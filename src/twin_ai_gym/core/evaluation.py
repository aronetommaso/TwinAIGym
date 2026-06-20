"""Agent evaluation primitives for TwinAIGym benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from twin_ai_gym.core.action import ActionCommand
from twin_ai_gym.core.observation import Observation


class AgentLike(Protocol):
    """Protocol for agents that can act inside a TwinAIGym environment."""

    def act(self, observation: Observation) -> str | ActionCommand:
        """Return the next action name for an observation."""


AgentPolicy = AgentLike | Callable[[Observation], str | ActionCommand]


@dataclass(slots=True)
class EvaluationResult:
    """Summary produced after evaluating an agent in an environment.

    Attributes:
        score: Normalized score in the ``0.0`` to ``1.0`` range when possible.
        total_reward: Sum of rewards collected during the episode.
        steps: Number of executed environment steps.
        terminated: Whether the episode reached a terminal state.
        truncated: Whether the episode hit the step limit.
        metrics: Domain metrics reported by the environment.
        reward_components: Last known reward component values.
        failures: Validation failures or exceptions observed during evaluation.
    """

    score: float
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def passed(self, threshold: float = 0.85) -> bool:
        """Return whether the evaluation passes a benchmark threshold."""

        return self.score >= threshold and not self.failures

    def report(self) -> str:
        """Return a concise human-readable benchmark report."""

        lines = [
            f"Score: {self.score:.2%}",
            f"Total reward: {self.total_reward:.3f}",
            f"Steps: {self.steps}",
        ]
        for key, value in self.metrics.items():
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value:.3f}")
        if self.failures:
            lines.append("Failures:")
            lines.extend(f"- {failure}" for failure in self.failures)
        return "\n".join(lines)


def resolve_agent_action(agent: AgentPolicy, observation: Observation) -> str | ActionCommand:
    """Resolve the next action from a callable or object-style agent."""

    if callable(agent) and not hasattr(agent, "act"):
        return agent(observation)
    return agent.act(observation)  # type: ignore[union-attr]
