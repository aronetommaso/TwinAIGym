"""Gym-style environment wrapper for digital twin worlds."""

from __future__ import annotations

from typing import Any, Mapping

from twin_ai_gym.core.action import Action, ActionCommand, ActionResult, ActionSpec
from twin_ai_gym.core.dynamics import DirectTransitionModel, TransitionModel
from twin_ai_gym.core.episode import Episode
from twin_ai_gym.core.evaluation import AgentPolicy, EvaluationResult, resolve_agent_action
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
        transition_model: TransitionModel | None = None,
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
        self.transition_model = transition_model or DirectTransitionModel()
        self.max_steps = max_steps
        self._initial_snapshot = world.snapshot()
        self.episode = Episode(self._initial_snapshot)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return the first observation.

        Args:
            seed: Optional deterministic seed for the episode.
            options: Optional Gymnasium-compatible reset options.
        """

        self.world.rollback(self._initial_snapshot)
        if seed is not None:
            self.world.rng.seed(seed)
            self.world.observation_rng.seed(seed + 1_000_003)
        self.episode = Episode(self.world.snapshot())
        return self.observe(), {
            "step": self.world.step_count,
            "options": options or {},
            "observability": self.observation.observability,
            "transition_model": type(self.transition_model).__name__,
        }

    def step(
        self,
        action: str | Action | ActionCommand | Mapping[str, str | Action | ActionCommand],
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

    @property
    def action_space(self) -> dict[str, ActionSpec]:
        """Return the explicit, machine-readable high-level action space."""

        return {name: action.spec() for name, action in self.actions.items()}

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

    def evaluate(
        self,
        agent: AgentPolicy,
        episodes: int = 1,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> EvaluationResult:
        """Evaluate an agent as a deterministic benchmark episode.

        Args:
            agent: Callable policy or object exposing ``act(observation)``.
            episodes: Number of episodes to run.
            max_steps: Optional evaluation step cap. Defaults to ``self.max_steps``.
            seed: Optional base seed. Episode index is added for repeatability.

        Returns:
            Aggregate evaluation result across all episodes.
        """

        total_reward = 0.0
        total_steps = 0
        failures: list[str] = []
        last_metrics: dict[str, float] = {}
        last_components: dict[str, float] = {}
        terminated = False
        truncated = False

        for episode_index in range(episodes):
            episode_seed = seed + episode_index if seed is not None else None
            observation, _ = self.reset(seed=episode_seed)
            episode_limit = max_steps or self.max_steps
            for _ in range(episode_limit):
                try:
                    action = resolve_agent_action(agent, observation)
                    observation, reward, terminated, truncated, info = self.step(action)
                except Exception as exc:  # noqa: BLE001 - benchmark failures should be reported.
                    failures.append(f"Episode {episode_index + 1}: {exc}")
                    break
                total_reward += reward
                total_steps += 1
                last_components = dict(info.get("reward_components", {}))
                if not info.get("valid", True):
                    failures.append(f"Episode {episode_index + 1}: invalid action {info.get('action')}")
                if terminated or truncated:
                    break
            last_metrics = self.metrics()

        score = self.score(total_reward=total_reward, episodes=episodes, metrics=last_metrics)
        return EvaluationResult(
            score=score,
            total_reward=total_reward,
            steps=total_steps,
            terminated=terminated,
            truncated=truncated,
            metrics=last_metrics,
            reward_components=last_components,
            failures=failures,
        )

    def metrics(self) -> dict[str, float]:
        """Return domain metrics for benchmark reports."""

        return {}

    def score(self, total_reward: float, episodes: int, metrics: dict[str, float]) -> float:
        """Normalize reward into a benchmark score.

        Subclasses should override this when they expose domain-specific metrics.
        """

        if episodes <= 0:
            return 0.0
        return max(0.0, min(1.0, total_reward / episodes))

    def _step_single(
        self,
        action: str | Action | ActionCommand,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action and compute observation, reward, and info."""

        action_obj, command = self._resolve_action(action)
        result = self.transition_model.transition(self.world, action_obj, command)
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
            "reward_attribution": reward.attribution,
            "transition_model": type(self.transition_model).__name__,
            "command": command,
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

    def _resolve_action(
        self,
        action: str | Action | ActionCommand,
    ) -> tuple[Action, ActionCommand]:
        """Resolve an action name or action instance."""

        if isinstance(action, Action):
            return action, ActionCommand(action.name)
        if isinstance(action, ActionCommand):
            try:
                return self.actions[action.name], action
            except KeyError as exc:
                raise KeyError(f"Unknown action: {action.name}") from exc
        try:
            return self.actions[action], ActionCommand(action)
        except KeyError as exc:
            raise KeyError(f"Unknown action: {action}") from exc

    def is_done(self) -> bool:
        """Return whether the environment reached a terminal state."""

        return False
