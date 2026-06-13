"""Gymnasium compatibility wrapper for TwinAIGym environments."""

from __future__ import annotations

from typing import Any

from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.rl.features import VectorObservationEncoder

try:  # pragma: no cover - exercised only when Gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - default lightweight install path.
    gym = None
    spaces = None


class _FallbackEnv:
    """Minimal base class used when Gymnasium is not installed."""


class GymnasiumTwinEnvWrapper(_FallbackEnv if gym is None else gym.Env):
    """Expose a ``TwinEnv`` through the formal Gymnasium ``Env`` API.

    By default observations are JSON-friendly dictionaries. Pass a
    ``VectorObservationEncoder`` to emit fixed-size numeric vectors suitable for
    classical RL libraries. Actions are ``Discrete(n)`` indices when Gymnasium
    is available. String action names are also accepted for debugging.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env: TwinEnv, observation_encoder: VectorObservationEncoder | None = None) -> None:
        """Initialize the wrapper."""

        self.env = env
        self.observation_encoder = observation_encoder
        self.action_names = list(env.actions)
        self.action_space = spaces.Discrete(len(self.action_names)) if spaces is not None else None
        if spaces is not None and observation_encoder is not None:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(observation_encoder.feature_names),),
                dtype=float,
            )
        else:
            self.observation_space = spaces.Dict({}) if spaces is not None else None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | list[float], dict[str, Any]]:
        """Reset the wrapped environment."""

        observation, info = self.env.reset(seed=seed, options=options)
        return self._encode(observation), self._augment_info(info)

    def step(self, action: int | str) -> tuple[dict[str, Any] | list[float], float, bool, bool, dict[str, Any]]:
        """Apply a Gymnasium action index or TwinAIGym action name."""

        action_name = self.action_names[action] if isinstance(action, int) else action
        observation, reward, terminated, truncated, info = self.env.step(action_name)
        info = self._augment_info(info)
        info["action_index"] = self.action_names.index(action_name) if action_name in self.action_names else -1
        return self._encode(observation), reward, terminated, truncated, info

    def render(self) -> str:
        """Render the wrapped environment."""

        return self.env.render()

    def close(self) -> None:
        """Close the wrapper."""

    def _encode(self, observation) -> dict[str, Any] | list[float]:
        """Encode observations as graph dictionaries or numeric vectors."""

        if self.observation_encoder is None:
            return observation.to_dict()
        return self.observation_encoder.encode(self.env, observation).to_list()

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Attach RL-friendly action and feature metadata."""

        enriched = dict(info)
        enriched["action_names"] = list(self.action_names)
        if self.observation_encoder is not None:
            enriched["feature_names"] = list(self.observation_encoder.feature_names)
        return enriched

def make_gymnasium_env(
    env: TwinEnv,
    observation_encoder: VectorObservationEncoder | None = None,
) -> GymnasiumTwinEnvWrapper:
    """Return a Gymnasium-compatible wrapper for a TwinAIGym environment."""

    return GymnasiumTwinEnvWrapper(env, observation_encoder=observation_encoder)
