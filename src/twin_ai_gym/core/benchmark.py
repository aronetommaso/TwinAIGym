"""Benchmark suite utilities for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import fmean, stdev
from typing import Callable, Mapping, Sequence

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


@dataclass(frozen=True, slots=True)
class RepeatedAgentResult:
    """Statistics for one policy evaluated over the same seed set."""

    agent: str
    scores: tuple[float, ...]
    rewards: tuple[float, ...]
    steps: tuple[int, ...]
    termination_rate: float

    @property
    def mean_score(self) -> float:
        """Return mean normalized score."""

        return fmean(self.scores) if self.scores else 0.0

    @property
    def score_std(self) -> float:
        """Return sample standard deviation."""

        return stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def score_ci95(self) -> float:
        """Return a normal-approximation 95% confidence half-width."""

        return 1.96 * self.score_std / sqrt(len(self.scores)) if self.scores else 0.0


@dataclass(slots=True)
class RepeatedBenchmarkResult:
    """Comparable multi-seed results for multiple policies."""

    name: str
    seeds: tuple[int, ...]
    agents: dict[str, RepeatedAgentResult]

    def report(self) -> str:
        """Render a paper-friendly compact table."""

        lines = [
            f"Repeated benchmark: {self.name}",
            f"Seeds: {len(self.seeds)}",
            "agent | mean score | std | 95% CI | termination | mean steps",
            "--- | ---: | ---: | ---: | ---: | ---:",
        ]
        for name, result in sorted(
            self.agents.items(),
            key=lambda item: item[1].mean_score,
            reverse=True,
        ):
            lines.append(
                f"{name} | {result.mean_score:.3f} | {result.score_std:.3f} | "
                f"+/- {result.score_ci95:.3f} | {result.termination_rate:.1%} | "
                f"{fmean(result.steps):.2f}"
            )
        return "\n".join(lines)


def compare_agents(
    name: str,
    env_factory: Callable[[int], TwinEnv],
    agents: Mapping[str, AgentPolicy],
    seeds: Sequence[int],
) -> RepeatedBenchmarkResult:
    """Evaluate policies on paired deterministic seeds."""

    seed_tuple = tuple(seeds)
    results: dict[str, RepeatedAgentResult] = {}
    for agent_name, agent in agents.items():
        evaluations = [
            env_factory(seed).evaluate(agent, seed=seed)
            for seed in seed_tuple
        ]
        results[agent_name] = RepeatedAgentResult(
            agent=agent_name,
            scores=tuple(result.score for result in evaluations),
            rewards=tuple(result.total_reward for result in evaluations),
            steps=tuple(result.steps for result in evaluations),
            termination_rate=(
                sum(result.terminated for result in evaluations) / len(evaluations)
                if evaluations
                else 0.0
            ),
        )
    return RepeatedBenchmarkResult(name=name, seeds=seed_tuple, agents=results)
