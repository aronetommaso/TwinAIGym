"""Pytest helpers for TwinAIGym benchmark assertions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.evaluation import AgentPolicy, EvaluationResult


def assert_agent_score(
    agent: AgentPolicy,
    env: TwinEnv,
    threshold: float = 0.85,
    episodes: int = 1,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate an agent and raise a pytest assertion with the benchmark report."""

    result = env.evaluate(agent, episodes=episodes, seed=seed)
    if not result.passed(threshold):
        pytest.fail(f"Agent score {result.score:.2%} is below threshold {threshold:.2%}\n{result.report()}")
    return result


@dataclass(slots=True)
class TwinAgentAssertion:
    """Fluent assertion helper exposed as the ``twinaigym`` pytest fixture."""

    def score_above(
        self,
        agent: AgentPolicy,
        env: TwinEnv,
        threshold: float = 0.85,
        episodes: int = 1,
        seed: int | None = None,
    ) -> EvaluationResult:
        """Assert that an agent scores at least ``threshold`` on an environment."""

        return assert_agent_score(agent, env, threshold=threshold, episodes=episodes, seed=seed)


@pytest.fixture
def twinaigym() -> TwinAgentAssertion:
    """Return the TwinAIGym pytest assertion helper."""

    return TwinAgentAssertion()
