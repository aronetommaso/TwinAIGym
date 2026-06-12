"""Episode recording and replay utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from twin_ai_gym.core.action import ActionResult
from twin_ai_gym.core.world import WorldSnapshot, WorldState


@dataclass(slots=True)
class EpisodeStep:
    """Single transition recorded during an episode."""

    action: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    action_result: ActionResult


@dataclass(slots=True)
class Episode:
    """Recorded sequence of actions, rewards, diffs, and metadata."""

    initial_snapshot: WorldSnapshot
    steps: list[EpisodeStep] = field(default_factory=list)

    def append(
        self,
        action: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        action_result: ActionResult,
    ) -> None:
        """Append a transition to the episode."""

        self.steps.append(
            EpisodeStep(
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                action_result=action_result,
            )
        )

    def replay(self, world: WorldState) -> list[ActionResult]:
        """Replay state changes using stored snapshots and diffs.

        This method restores the initial state and returns the recorded action results.
        It does not re-run domain actions, which keeps replay deterministic even when the
        original actions had stochastic dynamics.
        """

        world.rollback(self.initial_snapshot)
        results = []
        for step in self.steps:
            if step.action_result.snapshot_before is not None:
                world.rollback(step.action_result.snapshot_before)
            results.append(step.action_result)
        return results
