"""Trajectory collection and export for RL, imitation learning, and fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.evaluation import AgentPolicy, resolve_agent_action
from twin_ai_gym.rl.features import VectorObservationEncoder


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One transition suitable for RL datasets."""

    episode: int
    step: int
    observation: dict | list[float]
    action: str
    action_index: int
    reward: float
    next_observation: dict | list[float]
    terminated: bool
    truncated: bool
    score: float
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        """Return JSON-serializable transition data."""

        return {
            "episode": self.episode,
            "step": self.step,
            "observation": self.observation,
            "action": self.action,
            "action_index": self.action_index,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "score": self.score,
            "metrics": self.metrics,
        }


def collect_trajectories(
    env_factory: Callable[[], TwinEnv],
    agent: AgentPolicy,
    episodes: int,
    seed: int | None = None,
    encoder: VectorObservationEncoder | None = None,
) -> list[TrajectoryStep]:
    """Collect transitions by running an agent in fresh environments."""

    transitions: list[TrajectoryStep] = []
    for episode in range(episodes):
        env = env_factory()
        episode_seed = seed + episode if seed is not None else None
        observation, _ = env.reset(seed=episode_seed)
        for step in range(env.max_steps):
            encoded_observation = _encode(env, observation, encoder)
            action = resolve_agent_action(agent, observation)
            action_index = list(env.actions).index(action) if action in env.actions else -1
            next_observation, reward, terminated, truncated, _ = env.step(action)
            metrics = env.metrics()
            transitions.append(
                TrajectoryStep(
                    episode=episode,
                    step=step,
                    observation=encoded_observation,
                    action=action,
                    action_index=action_index,
                    reward=reward,
                    next_observation=_encode(env, next_observation, encoder),
                    terminated=terminated,
                    truncated=truncated,
                    score=env.score(reward, 1, metrics),
                    metrics=metrics,
                )
            )
            observation = next_observation
            if terminated or truncated:
                break
    return transitions


def export_trajectories_jsonl(transitions: list[TrajectoryStep], path: str | Path) -> None:
    """Write trajectories to JSONL."""

    with Path(path).open("w", encoding="utf-8") as file:
        for transition in transitions:
            file.write(json.dumps(transition.to_dict()) + "\n")


def _encode(env: TwinEnv, observation, encoder: VectorObservationEncoder | None) -> dict | list[float]:
    """Encode observations for export."""

    if encoder is None:
        return observation.to_dict()
    return encoder.encode(env, observation).to_list()
