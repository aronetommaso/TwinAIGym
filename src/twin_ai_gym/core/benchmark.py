"""Benchmark suite utilities for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from twin_ai_gym.core.env import TwinEnv
from twin_ai_gym.core.evaluation import AgentPolicy, EvaluationResult


@dataclass(slots=True)
class BenchmarkCase:
    """Single benchmark scenario for an agent.

    Attributes:
        name: Human-readable benchmark case name.
        env_factory: Callable that creates a fresh environment instance.
        threshold: Minimum normalized score required to pass.
    """

    name: str
    env_factory: Callable[[], TwinEnv]
    threshold: float = 0.85


@dataclass(slots=True)
class BenchmarkSuiteResult:
    """Aggregate result for a benchmark suite."""

    name: str
    cases: dict[str, EvaluationResult] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Return the average normalized score across benchmark cases."""

        if not self.cases:
            return 0.0
        return sum(result.score for result in self.cases.values()) / len(self.cases)

    def passed(self) -> bool:
        """Return whether every benchmark case passed its threshold."""

        return all(
            result.passed(self.thresholds.get(case_name, 0.85))
            for case_name, result in self.cases.items()
        )

    def report(self) -> str:
        """Return a concise suite-level benchmark report."""

        lines = [f"Benchmark suite: {self.name}", f"Average score: {self.score:.2%}"]
        for case_name, result in self.cases.items():
            threshold = self.thresholds.get(case_name, 0.85)
            status = "PASS" if result.passed(threshold) else "FAIL"
            lines.append(f"{case_name}: {result.score:.2%} ({status}, threshold={threshold:.2%})")
        return "\n".join(lines)


class BenchmarkSuite:
    """Collection of repeatable benchmark cases for an agent."""

    def __init__(self, name: str, cases: list[BenchmarkCase]) -> None:
        """Initialize a benchmark suite."""

        self.name = name
        self.cases = cases

    def evaluate(self, agent: AgentPolicy, seed: int | None = None) -> BenchmarkSuiteResult:
        """Evaluate an agent across all suite cases."""

        suite_result = BenchmarkSuiteResult(name=self.name)
        for index, case in enumerate(self.cases):
            env = case.env_factory()
            case_seed = seed + index if seed is not None else None
            suite_result.cases[case.name] = env.evaluate(agent, seed=case_seed)
            suite_result.thresholds[case.name] = case.threshold
        return suite_result
