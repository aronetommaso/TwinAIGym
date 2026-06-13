"""Gymnasium compatibility wrapper for TwinAIGym environments."""

from __future__ import annotations

from typing import Any

from twin_ai_gym.core.env import TwinEnv

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

    Observations are JSON-friendly dictionaries and actions are ``Discrete(n)``
    indices when Gymnasium is available. String action names are also accepted
    for convenience during debugging.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env: TwinEnv) -> None:
        """Initialize the wrapper."""

        self.env = env
        self.action_names = list(env.actions)
        self.action_space = spaces.Discrete(len(self.action_names)) if spaces is not None else None
        self.observation_space = spaces.Dict({}) if spaces is not None else None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the wrapped environment."""

        observation, info = self.env.reset(seed=seed, options=options)
        return observation.to_dict(), info

    def step(self, action: int | str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply a Gymnasium action index or TwinAIGym action name."""

        action_name = self.action_names[action] if isinstance(action, int) else action
        observation, reward, terminated, truncated, info = self.env.step(action_name)
        return observation.to_dict(), reward, terminated, truncated, info

    def render(self) -> str:
        """Render the wrapped environment."""

        return self.env.render()

    def close(self) -> None:
        """Close the wrapper."""


def make_gymnasium_env(env: TwinEnv) -> GymnasiumTwinEnvWrapper:
    """Return a Gymnasium-compatible wrapper for a TwinAIGym environment."""

    return GymnasiumTwinEnvWrapper(env)
