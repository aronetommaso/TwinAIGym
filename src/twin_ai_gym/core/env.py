"""Gym-style environment wrapper for digital twin worlds."""

from __future__ import annotations

from typing import Any, Mapping

from twin_ai_gym.core.action import Action, ActionResult
from twin_ai_gym.core.episode import Episode
from twin_ai_gym.core.observation import FullObservation, Observation, ObservationPolicy
from twin_ai_gym.core.reward import RewardAggregator
from twin_ai_gym.core.world import WorldSnapshot, WorldState


class TwinEnv:
    """Minimal Gymnasium-inspired environment for graph-native worlds."""

    def __init__(
        self,
        world: WorldState,
        actions: Mapping[str, Action],
        reward: RewardAggregator | None = None,
        observation: ObservationPolicy | None = None,
        max_steps: int = 100,
    ) -> None:
        """Initialize the environment.

        Args:
            world: Mutable world state.
            actions: Mapping from action names to action instances.
            reward: Reward aggregator.
            observation: Observation policy.
            max_steps: Episode step limit.
        """

        self.world = world
        self.actions = dict(actions)
        self.reward = reward or RewardAggregator()
        self.observation = observation or FullObservation()
        self.max_steps = max_steps
        self._initial_snapshot = world.snapshot()
        self.episode = Episode(self._initial_snapshot)

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return the first observation."""

        self.world.rollback(self._initial_snapshot)
        if seed is not None:
            self.world.rng.seed(seed)
        self.episode = Episode(self.world.snapshot())
        return self.observe(), {"step": self.world.step_count}

    def step(
        self,
        action: str | Action | Mapping[str, str | Action],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply a single-agent or multi-agent action.

        Args:
            action: Action name, action object, or mapping of agent ID to action.
        """

        if isinstance(action, Mapping):
            return self._step_multi(action)
        return self._step_single(action)

    def observe(self, agent_id: str | None = None) -> Observation:
        """Return the current observation."""

        return self.observation.observe(self.world, agent_id=agent_id)

    def render(self) -> str:
        """Render a compact textual view of the world."""

        lines = [f"World step: {self.world.step_count}"]
        for entity in sorted(self.world.entities.values(), key=lambda item: item.id):
            lines.append(f"- {entity.id} ({entity.type}): {entity.attributes}")
        return "\n".join(lines)

    def snapshot(self) -> WorldSnapshot:
        """Return a snapshot of the current world."""

        return self.world.snapshot()

    def rollback(self, snapshot: WorldSnapshot) -> None:
        """Rollback the world to a previous snapshot."""

        self.world.rollback(snapshot)

    def _step_single(
        self,
        action: str | Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action and compute observation, reward, and info."""

        action_obj = self._resolve_action(action)
        result = action_obj.apply(self.world)
        before = result.snapshot_before
        if before is None:
            raise RuntimeError("Action did not produce a pre-action snapshot.")
        reward = self.reward.compute(before, self.world, result)
        terminated = self.is_done()
        truncated = self.world.step_count >= self.max_steps
        info = {
            "action": result.action_name,
            "valid": result.valid,
            "message": result.message,
            "metadata": result.metadata,
            "diff": result.diff,
            "reward_components": reward.components,
        }
        self.episode.append(result.action_name, reward.total, terminated, truncated, info, result)
        return self.observe(), reward.total, terminated, truncated, info

    def _step_multi(
        self,
        actions: Mapping[str, str | Action],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply multiple agent actions sequentially in deterministic key order."""

        total_reward = 0.0
        results: dict[str, ActionResult] = {}
        infos: dict[str, Any] = {}
        for agent_id in sorted(actions):
            observation, reward, terminated, truncated, info = self._step_single(actions[agent_id])
            total_reward += reward
            results[agent_id] = self.episode.steps[-1].action_result
            infos[agent_id] = info
            if terminated or truncated:
                break
        return observation, total_reward, self.is_done(), self.world.step_count >= self.max_steps, {
            "agents": infos,
            "results": results,
        }

    def _resolve_action(self, action: str | Action) -> Action:
        """Resolve an action name or action instance."""

        if isinstance(action, Action):
            return action
        try:
            return self.actions[action]
        except KeyError as exc:
            raise KeyError(f"Unknown action: {action}") from exc

    def is_done(self) -> bool:
        """Return whether the environment reached a terminal state."""

        return False
