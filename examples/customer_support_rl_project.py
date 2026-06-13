"""Serious CPU-only RL project on the customer support digital twin.

This trains a tabular Q-learning agent over vectorized graph observations. It is
small enough to run without a GPU, but it follows the same project structure you
would use before swapping in DQN/PPO:

1. vectorize the mutable graph state;
2. discretize the vector for a tabular baseline;
3. train with epsilon-greedy exploration;
4. evaluate on held-out seeds;
5. export trajectories and an HTML training report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from twin_ai_gym.adapters import make_gymnasium_env
from twin_ai_gym.renderers import render_rl_training_report
from twin_ai_gym.rl import (
    CustomerSupportVectorEncoder,
    EpisodeMetric,
    collect_trajectories,
    export_trajectories_jsonl,
    summarize_episode_metrics,
)
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def make_env(seed: int | None = None):
    """Create a vectorized Gymnasium-compatible customer support environment."""

    return make_gymnasium_env(
        CustomerSupportWorld.adversarial(seed=seed),
        observation_encoder=CustomerSupportVectorEncoder(),
    )


@dataclass
class TabularQAgent:
    """Epsilon-greedy Q-learning agent over discretized vector observations."""

    action_names: list[str]
    bins: int = 5
    learning_rate: float = 0.22
    discount: float = 0.92
    epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.985
    q_values: dict[tuple[int, ...], list[float]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def act_index(self, observation: list[float], explore: bool = True) -> int:
        """Choose an action index."""

        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.action_names))
        state = self.discretize(observation)
        values = self.q_values.setdefault(state, [0.0 for _ in self.action_names])
        return max(range(len(values)), key=values.__getitem__)

    def act(self, observation) -> str:
        """TwinAIGym policy interface for trajectory export."""

        vector = _encode_policy_observation(observation)
        return self.action_names[self.act_index(vector, explore=False)]

    def learn(
        self,
        observation: list[float],
        action_index: int,
        reward: float,
        next_observation: list[float],
        done: bool,
    ) -> None:
        """Apply one Q-learning update."""

        state = self.discretize(observation)
        next_state = self.discretize(next_observation)
        values = self.q_values.setdefault(state, [0.0 for _ in self.action_names])
        next_values = self.q_values.setdefault(next_state, [0.0 for _ in self.action_names])
        target = reward if done else reward + self.discount * max(next_values)
        values[action_index] += self.learning_rate * (target - values[action_index])

    def decay_exploration(self) -> None:
        """Decay epsilon after each training episode."""

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def discretize(self, observation: list[float]) -> tuple[int, ...]:
        """Convert normalized continuous features into tabular bins."""

        return tuple(max(0, min(self.bins - 1, int(value * self.bins))) for value in observation)


class TrainedPolicy:
    """Bridge a trained Q-table back into the TwinAIGym ``act`` protocol."""

    def __init__(self, agent: TabularQAgent) -> None:
        """Initialize the policy."""

        self.agent = agent
        self.encoder = CustomerSupportVectorEncoder()

    def act(self, observation) -> str:
        """Choose the greedy action from a graph observation."""

        vector = _encode_policy_observation(observation)
        return self.agent.action_names[self.agent.act_index(vector, explore=False)]


def train(episodes: int = 220) -> tuple[TabularQAgent, list[EpisodeMetric]]:
    """Train a tabular Q-learning agent."""

    probe_env = make_env(seed=0)
    agent = TabularQAgent(action_names=list(probe_env.action_names))
    metrics: list[EpisodeMetric] = []

    for episode in range(episodes):
        env = make_env(seed=episode)
        observation, _ = env.reset(seed=episode)
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        for steps in range(1, env.env.max_steps + 1):
            action_index = agent.act_index(observation, explore=True)
            next_observation, reward, terminated, truncated, _ = env.step(action_index)
            done = terminated or truncated
            agent.learn(observation, action_index, reward, next_observation, done)
            total_reward += reward
            observation = next_observation
            if done:
                break

        score = env.env.score(total_reward, 1, env.env.metrics())
        metrics.append(
            EpisodeMetric(
                episode=episode,
                total_reward=total_reward,
                steps=steps,
                score=score,
                terminated=terminated,
                truncated=truncated,
                epsilon=agent.epsilon,
            )
        )
        agent.decay_exploration()

    return agent, metrics


def evaluate(agent: TabularQAgent, seeds: range = range(1000, 1030)) -> list[EpisodeMetric]:
    """Evaluate the greedy policy on held-out seeds."""

    metrics: list[EpisodeMetric] = []
    for index, seed in enumerate(seeds):
        env = make_env(seed=seed)
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        for steps in range(1, env.env.max_steps + 1):
            action_index = agent.act_index(observation, explore=False)
            observation, reward, terminated, truncated, _ = env.step(action_index)
            total_reward += reward
            if terminated or truncated:
                break

        metrics.append(
            EpisodeMetric(
                episode=index,
                total_reward=total_reward,
                steps=steps,
                score=env.env.score(total_reward, 1, env.env.metrics()),
                terminated=terminated,
                truncated=truncated,
                epsilon=None,
            )
        )
    return metrics


def export_eval_trajectories(agent: TabularQAgent) -> None:
    """Export greedy evaluation trajectories as JSONL."""

    class Policy:
        def __init__(self, agent: TabularQAgent) -> None:
            self.agent = agent
            self.encoder = CustomerSupportVectorEncoder()

        def act(self, observation) -> str:
            vector = _encode_policy_observation(observation)
            return self.agent.action_names[self.agent.act_index(vector, explore=False)]

    transitions = collect_trajectories(
        env_factory=lambda: CustomerSupportWorld.adversarial(seed=900),
        agent=Policy(agent),
        episodes=5,
        seed=900,
        encoder=CustomerSupportVectorEncoder(),
    )
    export_trajectories_jsonl(transitions, "customer_support_rl_trajectories.jsonl")


def main() -> None:
    """Train, evaluate, export, and visualize the RL project."""

    agent, training_metrics = train()
    evaluation_metrics = evaluate(agent)
    train_summary = summarize_episode_metrics(training_metrics)
    eval_summary = summarize_episode_metrics(evaluation_metrics)

    export_eval_trajectories(agent)
    render_rl_training_report(
        training_metrics,
        evaluation_metrics,
        path="customer_support_rl_report.html",
        title="Customer Support RL Optimization",
    )

    print("Training summary")
    print(f"- episodes: {train_summary.episodes}")
    print(f"- average score: {train_summary.average_score:.2%}")
    print(f"- best score: {train_summary.best_score:.2%}")
    print()
    print("Held-out evaluation summary")
    print(f"- episodes: {eval_summary.episodes}")
    print(f"- average score: {eval_summary.average_score:.2%}")
    print(f"- success rate: {eval_summary.success_rate:.2%}")
    print(f"- best score: {eval_summary.best_score:.2%}")
    print()
    print("Artifacts")
    print("- customer_support_rl_report.html")
    print("- customer_support_rl_trajectories.jsonl")


def _encode_policy_observation(observation) -> list[float]:
    """Encode a graph observation when only the policy API is available."""

    class _WorldView:
        step_count = 0

    class _EnvView:
        world = _WorldView()
        max_steps = 80

    return CustomerSupportVectorEncoder().encode(_EnvView(), observation).to_list()


if __name__ == "__main__":
    main()
