"""Benchmark suites for customer support agents."""

from __future__ import annotations

from twin_ai_gym.core.benchmark import BenchmarkCase, BenchmarkSuite
from twin_ai_gym.worlds.customer_support.env import CustomerSupportWorld


def customer_support_suite(seed: int | None = None) -> BenchmarkSuite:
    """Return the default customer support benchmark suite."""

    return BenchmarkSuite(
        name="TwinAIGym Customer Support",
        cases=[
            BenchmarkCase(
                name="standard",
                env_factory=lambda: CustomerSupportWorld(seed=seed),
                threshold=0.55,
            ),
            BenchmarkCase(
                name="adversarial",
                env_factory=lambda: CustomerSupportWorld.adversarial(seed=seed),
                threshold=0.25,
            ),
        ],
    )
