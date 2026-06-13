"""Metrics helpers for RL training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EpisodeMetric:
    """Metrics collected for one training or evaluation episode."""

    episode: int
    total_reward: float
    steps: int
    score: float
    terminated: bool
    truncated: bool
    epsilon: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate RL metrics across episodes."""

    episodes: int
    average_reward: float
    average_score: float
    average_steps: float
    best_score: float
    success_rate: float


def summarize_episode_metrics(metrics: list[EpisodeMetric], success_threshold: float = 0.55) -> EvaluationSummary:
    """Summarize per-episode RL metrics."""

    if not metrics:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    episodes = len(metrics)
    return EvaluationSummary(
        episodes=episodes,
        average_reward=sum(metric.total_reward for metric in metrics) / episodes,
        average_score=sum(metric.score for metric in metrics) / episodes,
        average_steps=sum(metric.steps for metric in metrics) / episodes,
        best_score=max(metric.score for metric in metrics),
        success_rate=sum(1 for metric in metrics if metric.score >= success_threshold) / episodes,
    )
