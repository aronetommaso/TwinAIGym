"""Controlled benchmark of state representation and causal dynamics.

The experiment generates one latent incident task and exposes it through
graph, relational-table, or nested-object observations. Transition semantics,
action vocabulary, success criteria, and ground-truth causes remain paired.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from math import sqrt
import os
from pathlib import Path
import random
import re
from statistics import fmean, stdev
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
import urllib.request


REPRESENTATIONS = ("graph_causal", "graph_no_propagation", "tabular", "object")
OBSERVABILITY = ("full", "partial", "noisy")
ACTIONS = ("inspect", "repair", "restart", "rebalance", "wait")


@dataclass(frozen=True, slots=True)
class IncidentTask:
    """Latent infrastructure incident shared by every experimental condition."""

    task_id: str
    services: tuple[str, ...]
    components: tuple[str, ...]
    teams: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    ownership: tuple[tuple[str, str], ...]
    failed_components: tuple[str, ...]
    initial_health: tuple[tuple[str, float], ...]
    team_load: tuple[tuple[str, float], ...]
    criticality: tuple[tuple[str, float], ...]
    max_steps: int = 12

    @property
    def ground_truth_causal_nodes(self) -> set[str]:
        """Return root causes and all services depending on them."""

        causes = set(self.failed_components)
        changed = True
        while changed:
            changed = False
            for service, dependency in self.dependencies:
                if dependency in causes and service not in causes:
                    causes.add(service)
                    changed = True
        return causes


@dataclass(frozen=True, slots=True)
class IncidentCommand:
    """Representation-independent action used by all policies."""

    action: str
    target: str | None = None


@dataclass(slots=True)
class IncidentObservation:
    """Serializable policy input for one representation condition."""

    representation: str
    observability: str
    payload: dict[str, Any]
    visible_ids: tuple[str, ...]
    candidate_actions: tuple[IncidentCommand, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready observation."""

        return {
            "representation": self.representation,
            "observability": self.observability,
            "state": self.payload,
            "visible_ids": list(self.visible_ids),
            "candidate_actions": [asdict(action) for action in self.candidate_actions],
        }


@dataclass(slots=True)
class IncidentState:
    """Mutable latent state used by all representation backends."""

    operational: dict[str, bool]
    health: dict[str, float]
    team_load: dict[str, float]
    overloaded: dict[str, bool]
    inspected: set[str] = field(default_factory=set)
    steps: int = 0


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """Auditable result of one benchmark transition."""

    valid: bool
    reward: float
    changed_nodes: frozenset[str]
    causal_nodes: frozenset[str]
    reward_attribution: Mapping[str, float]
    message: str = ""


@dataclass(frozen=True, slots=True)
class EpisodeMeasurement:
    """Metrics collected from one task-policy-condition trial."""

    task_id: str
    representation: str
    observability: str
    agent: str
    trial: int
    success: bool
    total_reward: float
    steps: int
    recovered_after_error: bool
    attribution_f1: float
    invalid_actions: int
    tokens: int
    cost_usd: float
    runtime_ms: float


@dataclass(frozen=True, slots=True)
class AggregateMeasurement:
    """Aggregate metrics for one condition and policy."""

    representation: str
    observability: str
    agent: str
    episodes: int
    task_success: float
    reward_mean: float
    reward_std: float
    consistency: float
    recovery_rate: float
    attribution_f1: float
    mean_steps: float
    invalid_action_rate: float
    tokens: int
    cost_usd: float
    runtime_ms: float


@dataclass(frozen=True, slots=True)
class ScalingMeasurement:
    """Serialization and transition cost at one problem size."""

    representation: str
    nodes: int
    relations: int
    observation_ms: float
    transition_ms: float
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """Paired effect of one representation against a reference condition."""

    reference: str
    comparison: str
    observability: str
    agent: str
    pairs: int
    success_delta: float
    reward_delta: float
    reward_ci95: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a controlled representation experiment."""

    tasks: int = 120
    trials: int = 3
    seed: int = 2026
    train_tasks: int = 300
    representations: tuple[str, ...] = REPRESENTATIONS
    observability: tuple[str, ...] = OBSERVABILITY
    include_llm: bool = True
    llm_tasks: int = 20
    output_dir: str = "benchmarks/representation_study"


class IncidentAgent(Protocol):
    """Policy interface used by the experimental runner."""

    name: str
    tokens: int
    cost_usd: float

    def reset(self, seed: int) -> None:
        """Reset policy-local state for one trial."""

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Choose one structured action."""


class IncidentSimulator:
    """Paired simulator with swappable representation and propagation."""

    def __init__(
        self,
        task: IncidentTask,
        representation: str,
        observability: str,
        seed: int,
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        if observability not in OBSERVABILITY:
            raise ValueError(f"Unknown observability: {observability}")
        self.task = task
        self.representation = representation
        self.observability = observability
        self.rng = random.Random(seed)
        failed = set(task.failed_components)
        health = dict(task.initial_health)
        operational = {node: node not in failed for node in task.components}
        operational.update({service: True for service in task.services})
        self.state = IncidentState(
            operational=operational,
            health=health,
            team_load=dict(task.team_load),
            overloaded={team: dict(task.team_load)[team] >= 0.85 for team in task.teams},
        )
        # All ablations start from the same incident state. The no-propagation
        # condition disables only consequences after agent actions.
        self._propagate(set(), apply=True)
        self.reported_services = {
            service for service in task.services if not self.state.operational[service]
        }

    @property
    def has_propagation(self) -> bool:
        """Return whether graph consequences are applied after actions."""

        return self.representation != "graph_no_propagation"

    def observe(self) -> IncidentObservation:
        """Materialize the latent state in the configured representation."""

        visible = self._visible_nodes()
        if self.representation.startswith("graph"):
            payload = self._graph_payload(visible)
        elif self.representation == "tabular":
            payload = self._tabular_payload(visible)
        else:
            payload = self._object_payload(visible)
        return IncidentObservation(
            representation=self.representation,
            observability=self.observability,
            payload=payload,
            visible_ids=tuple(sorted(visible)),
            candidate_actions=self._candidate_actions(visible),
        )

    def step(self, command: IncidentCommand) -> TransitionRecord:
        """Apply one action and, when enabled, propagate causal effects."""

        before_operational = dict(self.state.operational)
        before_health = dict(self.state.health)
        before_overload = dict(self.state.overloaded)
        changed: set[str] = set()
        valid, message = self._apply_command(command, changed)
        if valid:
            self.state.steps += 1
            causal = self._propagate(changed, apply=self.has_propagation)
        else:
            causal = set()
        reward_components = self._reward_components(
            before_operational,
            before_health,
            before_overload,
            valid,
        )
        reward = sum(reward_components.values())
        return TransitionRecord(
            valid=valid,
            reward=reward,
            changed_nodes=frozenset(changed),
            causal_nodes=frozenset(causal),
            reward_attribution=reward_components,
            message=message,
        )

    def success(self) -> bool:
        """Return whether every node is operational and every team stable."""

        return all(self.state.operational.values()) and not any(self.state.overloaded.values())

    def done(self) -> bool:
        """Return terminal or truncated state."""

        return self.success() or self.state.steps >= self.task.max_steps

    def _apply_command(self, command: IncidentCommand, changed: set[str]) -> tuple[bool, str]:
        target = command.target
        if command.action not in ACTIONS:
            return False, "unknown action"
        if command.action == "wait":
            return True, ""
        if target is None:
            return False, "missing target"
        all_nodes = set(self.task.services) | set(self.task.components) | set(self.task.teams)
        if target not in all_nodes:
            return False, "unknown target"
        if command.action == "inspect":
            self.state.inspected.add(target)
            return True, ""
        if command.action == "repair":
            if target not in self.task.components or self.state.operational[target]:
                return False, "target is not a failed component"
            self.state.operational[target] = True
            self.state.health[target] = 1.0
            changed.add(target)
            return True, ""
        if command.action == "restart":
            if target not in self.task.services:
                return False, "target is not a service"
            self.state.operational[target] = True
            self.state.health[target] = max(0.8, self.state.health[target])
            changed.add(target)
            return True, ""
        if command.action == "rebalance":
            if target not in self.task.teams:
                return False, "target is not a team"
            self.state.team_load[target] = max(0.0, self.state.team_load[target] - 0.55)
            self.state.overloaded[target] = False
            changed.add(target)
            return True, ""
        return False, "unsupported action"

    def _propagate(self, changed: set[str], apply: bool) -> set[str]:
        causal: set[str] = set()
        if not apply:
            return causal
        stable = False
        while not stable:
            stable = True
            for service, dependency in self.task.dependencies:
                expected = self.state.operational[dependency]
                if not expected and self.state.operational[service]:
                    self.state.operational[service] = False
                    self.state.health[service] = max(0.0, self.state.health[service] - 0.08)
                    causal.add(service)
                    changed.add(service)
                    stable = False
            for component, team in self.task.ownership:
                if not self.state.operational[component]:
                    load = min(1.0, self.state.team_load[team] + 0.12)
                    if load != self.state.team_load[team]:
                        self.state.team_load[team] = load
                        changed.add(team)
                    overloaded = load >= 0.85
                    if overloaded != self.state.overloaded[team]:
                        self.state.overloaded[team] = overloaded
                        changed.add(team)
                    causal.add(team)
        return causal

    def _reward_components(
        self,
        before_operational: Mapping[str, bool],
        before_health: Mapping[str, float],
        before_overload: Mapping[str, bool],
        valid: bool,
    ) -> dict[str, float]:
        availability = sum(
            dict(self.task.criticality)[node]
            * (float(self.state.operational[node]) - float(before_operational[node]))
            for node in before_operational
        )
        health = 0.15 * sum(
            self.state.health[node] - before_health[node] for node in before_health
        )
        overload = 0.4 * sum(
            float(before_overload[team]) - float(self.state.overloaded[team])
            for team in before_overload
        )
        return {
            "availability": availability,
            "health": health,
            "overload": overload,
            "invalid_action": 0.0 if valid else -0.5,
            "step_cost": -0.03 if valid else 0.0,
        }

    def _visible_nodes(self) -> set[str]:
        all_nodes = set(self.task.services) | set(self.task.components) | set(self.task.teams)
        if self.observability == "full":
            return all_nodes
        failed_services = self.reported_services | {
            service for service in self.task.services if not self.state.operational[service]
        }
        visible = failed_services | self.state.inspected
        for source, target in self.task.dependencies:
            if source in self.state.inspected or target in self.state.inspected:
                visible.update((source, target))
        for component, team in self.task.ownership:
            if component in self.state.inspected or team in self.state.inspected:
                visible.update((component, team))
        if self.observability == "partial":
            for source, target in self.task.dependencies:
                if source in failed_services:
                    visible.add(target)
            return visible
        for node in all_nodes:
            if node in failed_services or node in self.state.inspected or self.rng.random() >= 0.25:
                visible.add(node)
        return visible

    def _node_row(self, node: str) -> dict[str, Any]:
        node_type = (
            "service"
            if node in self.task.services
            else "component"
            if node in self.task.components
            else "team"
        )
        row: dict[str, Any] = {
            "id": node,
            "type": node_type,
            "inspected": node in self.state.inspected,
        }
        if node_type == "team":
            row.update(
                load=self.state.team_load[node],
                overloaded=self.state.overloaded[node],
            )
        else:
            health = self.state.health[node]
            if self.observability == "noisy":
                health = max(0.0, min(1.0, health + self.rng.uniform(-0.12, 0.12)))
            row.update(
                operational=self.state.operational[node],
                health=health,
                criticality=dict(self.task.criticality)[node],
            )
        return row

    def _graph_payload(self, visible: set[str]) -> dict[str, Any]:
        return {
            "nodes": [self._node_row(node) for node in sorted(visible)],
            "edges": [
                {"source": source, "type": "DEPENDS_ON", "target": target}
                for source, target in self.task.dependencies
                if source in visible and target in visible
            ]
            + [
                {"source": component, "type": "OWNED_BY", "target": team}
                for component, team in self.task.ownership
                if component in visible and team in visible
            ],
        }

    def _tabular_payload(self, visible: set[str]) -> dict[str, Any]:
        return {
            "services": [
                self._node_row(node) for node in self.task.services if node in visible
            ],
            "components": [
                self._node_row(node) for node in self.task.components if node in visible
            ],
            "teams": [self._node_row(node) for node in self.task.teams if node in visible],
            "service_dependencies": [
                {"service_id": source, "component_id": target}
                for source, target in self.task.dependencies
                if source in visible and target in visible
            ],
            "component_owners": [
                {"component_id": component, "team_id": team}
                for component, team in self.task.ownership
                if component in visible and team in visible
            ],
        }

    def _object_payload(self, visible: set[str]) -> dict[str, Any]:
        owners = dict(self.task.ownership)
        dependencies = defaultdict(list)
        for service, component in self.task.dependencies:
            dependencies[service].append(component)
        return {
            "services": [
                {
                    **self._node_row(service),
                    "dependencies": [
                        {
                            **self._node_row(component),
                            "owner": (
                                self._node_row(owners[component])
                                if owners[component] in visible
                                else {"id": owners[component], "hidden": True}
                            ),
                        }
                        for component in dependencies[service]
                        if component in visible
                    ],
                }
                for service in self.task.services
                if service in visible
            ],
            "component_registry": [
                {
                    **self._node_row(component),
                    "owner": (
                        self._node_row(owners[component])
                        if owners[component] in visible
                        else {"id": owners[component], "hidden": True}
                    ),
                }
                for component in self.task.components
                if component in visible
            ],
            "teams": [
                self._node_row(team) for team in self.task.teams if team in visible
            ],
        }

    def _candidate_actions(self, visible: set[str]) -> tuple[IncidentCommand, ...]:
        actions = [IncidentCommand("wait")]
        for node in sorted(visible):
            actions.append(IncidentCommand("inspect", node))
            if node in self.task.components:
                actions.append(IncidentCommand("repair", node))
            elif node in self.task.services:
                actions.append(IncidentCommand("restart", node))
            else:
                actions.append(IncidentCommand("rebalance", node))
        return tuple(actions)


class StructuralHeuristicAgent:
    """Representation-aware parser implementing one representation-neutral strategy."""

    name = "heuristic"

    def __init__(self) -> None:
        self.tokens = 0
        self.cost_usd = 0.0

    def reset(self, seed: int) -> None:
        """Reset has no state for the deterministic heuristic."""

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Repair failed dependencies, stabilize owners, then restart services."""

        nodes, dependencies, ownership = normalize_observation(observation)
        failed_services = {
            node_id
            for node_id, row in nodes.items()
            if row["type"] == "service" and not row.get("operational", True)
        }
        for service, component in dependencies:
            component_row = nodes.get(component)
            if (
                service in failed_services
                and component_row is not None
                and not component_row.get("operational", True)
            ):
                return IncidentCommand("repair", component)
        failed_components = sorted(
            node_id
            for node_id, row in nodes.items()
            if row["type"] == "component" and not row.get("operational", True)
        )
        if failed_components:
            return IncidentCommand("repair", failed_components[0])
        for component, team in ownership:
            team_row = nodes.get(team)
            if team_row is not None and team_row.get("overloaded"):
                return IncidentCommand("rebalance", team)
        if failed_services:
            uninspected_components = sorted(
                node_id
                for node_id, row in nodes.items()
                if row["type"] == "component" and not row.get("inspected")
            )
            if uninspected_components:
                return IncidentCommand("inspect", uninspected_components[0])
            return IncidentCommand("restart", sorted(failed_services)[0])
        candidates = [action for action in observation.candidate_actions if action.action != "wait"]
        return candidates[0] if candidates else IncidentCommand("wait")


class RandomIncidentAgent:
    """Seeded random baseline over currently visible structured actions."""

    name = "random"

    def __init__(self) -> None:
        self.rng = random.Random(0)
        self.tokens = 0
        self.cost_usd = 0.0

    def reset(self, seed: int) -> None:
        """Reset the random stream for paired trials."""

        self.rng.seed(seed)

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Choose uniformly from candidate actions."""

        return self.rng.choice(observation.candidate_actions)


@dataclass(slots=True)
class TabularQIncidentAgent:
    """Small dependency-free Q-learning baseline over aggregate observations."""

    name: str = "q_learning"
    bins: int = 4
    alpha: float = 0.22
    gamma: float = 0.92
    epsilon: float = 0.15
    q: dict[tuple[int, ...], list[float]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    tokens: int = 0
    cost_usd: float = 0.0

    def reset(self, seed: int) -> None:
        """Reset policy exploration randomness."""

        self.rng.seed(seed)

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Choose a strategy and ground it to a visible target."""

        strategy = self._choose(observation, explore=False)
        return ground_strategy(strategy, observation)

    def train(
        self,
        tasks: Sequence[IncidentTask],
        representation: str,
        observability: str,
        seed: int,
        epochs: int = 3,
    ) -> None:
        """Train on generated incidents for one experimental condition."""

        self.rng.seed(seed)
        for epoch in range(epochs):
            order = list(tasks)
            self.rng.shuffle(order)
            for index, task in enumerate(order):
                env = IncidentSimulator(
                    task,
                    representation,
                    observability,
                    seed + epoch * len(tasks) + index,
                )
                while not env.done():
                    observation = env.observe()
                    state = discretize_features(observation, self.bins)
                    action_index = self._choose_index(state, explore=True)
                    command = ground_strategy(ACTIONS[action_index], observation)
                    transition = env.step(command)
                    next_observation = env.observe()
                    next_state = discretize_features(next_observation, self.bins)
                    values = self.q.setdefault(state, [0.0] * len(ACTIONS))
                    next_values = self.q.setdefault(next_state, [0.0] * len(ACTIONS))
                    target = transition.reward
                    if not env.done():
                        target += self.gamma * max(next_values)
                    values[action_index] += self.alpha * (target - values[action_index])

    def _choose(self, observation: IncidentObservation, explore: bool) -> str:
        state = discretize_features(observation, self.bins)
        return ACTIONS[self._choose_index(state, explore)]

    def _choose_index(self, state: tuple[int, ...], explore: bool) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        values = self.q.setdefault(state, [0.0] * len(ACTIONS))
        return max(range(len(values)), key=values.__getitem__)


@dataclass(slots=True)
class OpenAICompatibleIncidentAgent:
    """Optional JSON tool-agent baseline with token and cost accounting."""

    strategy: str
    model: str = "gpt-4.1-mini"
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    input_cost_per_million: float = 0.40
    output_cost_per_million: float = 1.60
    name: str = field(init=False)
    tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self.name = f"llm_{self.strategy}"

    @property
    def available(self) -> bool:
        """Return whether an API key is configured."""

        return bool(os.getenv("OPENAI_API_KEY"))

    def reset(self, seed: int) -> None:
        """LLM calls are stateless across trials."""

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Request one strict JSON action from an OpenAI-compatible endpoint."""

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        prompts = {
            "direct": "Choose the best incident-response action.",
            "causal": "Trace dependencies and ownership before choosing an action.",
            "audit": "Choose an action that maximizes recovery and supports causal auditability.",
        }
        body = {
            "model": os.getenv("OPENAI_MODEL", self.model),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        prompts[self.strategy]
                        + ' Return JSON only: {"action": string, "target": string|null}.'
                    ),
                },
                {"role": "user", "content": json.dumps(observation.to_dict())},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        usage = result.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        self.tokens += input_tokens + output_tokens
        self.cost_usd += (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return IncidentCommand(str(parsed["action"]), parsed.get("target"))


@dataclass(slots=True)
class OllamaIncidentAgent:
    """Local Ollama JSON agent used for cost-free representation pilots."""

    model: str
    strategy: str
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    name: str = field(init=False)
    tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        safe_model = self.model.replace(":", "_")
        self.name = f"ollama_{safe_model}_{self.strategy}"

    def reset(self, seed: int) -> None:
        """Ollama requests are stateless for this benchmark."""

    def act(self, observation: IncidentObservation) -> IncidentCommand:
        """Choose one command using a local model with JSON-constrained output."""

        prompts = {
            "direct": "Choose the best incident-response action.",
            "causal": "Trace dependencies and ownership before choosing an action.",
            "audit": "Choose a recoverable action and preserve causal auditability.",
        }
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "seed": 0,
                "num_predict": 128,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        prompts[self.strategy]
                        + ' Return only {"action": string, "target": string|null}. '
                        + f"Allowed actions: {ACTIONS}."
                    ),
                },
                {"role": "user", "content": json.dumps(observation.to_dict())},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        self.tokens += int(result.get("prompt_eval_count", 0))
        self.tokens += int(result.get("eval_count", 0))
        content = result["message"]["content"]
        try:
            parsed = json.loads(content)
            return IncidentCommand(str(parsed.get("action", "wait")), parsed.get("target"))
        except json.JSONDecodeError:
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', content)
            target_match = re.search(r'"target"\s*:\s*(null|"[^"]*")', content)
            action = action_match.group(1) if action_match else "wait"
            target = None
            if target_match and target_match.group(1) != "null":
                target = target_match.group(1).strip('"')
            return IncidentCommand(action, target)


class RepresentationExperiment:
    """Run, aggregate, and persist the complete factorial experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.tasks = generate_tasks(config.tasks, config.seed)
        self.train_tasks = generate_tasks(config.train_tasks, config.seed + 100_000)

    def run(self) -> tuple[list[EpisodeMeasurement], list[AggregateMeasurement], list[str]]:
        """Execute all local agents and configured LLM agents."""

        measurements: list[EpisodeMeasurement] = []
        skipped: list[str] = []
        for representation in self.config.representations:
            for observability in self.config.observability:
                agents: list[IncidentAgent] = [
                    StructuralHeuristicAgent(),
                    RandomIncidentAgent(),
                ]
                q_agent = TabularQIncidentAgent()
                q_agent.train(
                    self.train_tasks,
                    representation,
                    observability,
                    self.config.seed,
                )
                agents.append(q_agent)
                if self.config.include_llm:
                    for strategy in ("direct", "causal", "audit"):
                        llm = OpenAICompatibleIncidentAgent(strategy)
                        if llm.available:
                            agents.append(llm)
                        else:
                            skipped.append(
                                f"{llm.name}/{representation}/{observability}: missing API key"
                            )
                for agent in agents:
                    task_limit = (
                        self.config.llm_tasks
                        if agent.name.startswith("llm_")
                        else None
                    )
                    measurements.extend(
                        self._run_agent(
                            agent,
                            representation,
                            observability,
                            task_limit=task_limit,
                        )
                    )
        aggregates = aggregate_measurements(measurements)
        return measurements, aggregates, skipped

    def save(
        self,
        measurements: Sequence[EpisodeMeasurement],
        aggregates: Sequence[AggregateMeasurement],
        skipped: Sequence[str],
    ) -> Path:
        """Persist raw JSONL, aggregate JSON, CSV, and Markdown report."""

        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "episodes.jsonl").write_text(
            "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in measurements),
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "aggregates": [asdict(item) for item in aggregates],
                    "skipped": list(skipped),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        headers = list(AggregateMeasurement.__dataclass_fields__)
        csv_lines = [",".join(headers)]
        for item in aggregates:
            row = asdict(item)
            csv_lines.append(",".join(str(row[header]) for header in headers))
        (output / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        scaling = run_scaling_benchmark(self.config.seed)
        contrasts = compute_paired_contrasts(measurements)
        (output / "scaling.json").write_text(
            json.dumps([asdict(item) for item in scaling], indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "contrasts.json").write_text(
            json.dumps([asdict(item) for item in contrasts], indent=2) + "\n",
            encoding="utf-8",
        )
        report = render_experiment_report(
            self.config,
            aggregates,
            skipped,
            scaling,
            contrasts,
        )
        report_path = output / "REPORT.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def _run_agent(
        self,
        agent: IncidentAgent,
        representation: str,
        observability: str,
        task_limit: int | None = None,
    ) -> list[EpisodeMeasurement]:
        rows: list[EpisodeMeasurement] = []
        tasks = self.tasks[:task_limit] if task_limit is not None else self.tasks
        for task_index, task in enumerate(tasks):
            for trial in range(self.config.trials):
                trial_seed = self.config.seed + task_index * 1009 + trial
                agent.reset(trial_seed)
                env = IncidentSimulator(task, representation, observability, trial_seed)
                started = perf_counter()
                total_reward = 0.0
                invalid = 0
                attributed_nodes: set[str] = set()
                forced_error = IncidentCommand("restart", task.services[0])
                first = env.step(forced_error)
                total_reward += first.reward
                error_had_effect = not env.success()
                while not env.done():
                    observation = env.observe()
                    command = agent.act(observation)
                    transition = env.step(command)
                    total_reward += transition.reward
                    invalid += int(not transition.valid)
                    attributed_nodes.update(transition.changed_nodes)
                    attributed_nodes.update(transition.causal_nodes)
                runtime_ms = (perf_counter() - started) * 1000
                rows.append(
                    EpisodeMeasurement(
                        task_id=task.task_id,
                        representation=representation,
                        observability=observability,
                        agent=agent.name,
                        trial=trial,
                        success=env.success(),
                        total_reward=total_reward,
                        steps=env.state.steps,
                        recovered_after_error=error_had_effect and env.success(),
                        attribution_f1=set_f1(
                            attributed_nodes,
                            task.ground_truth_causal_nodes,
                        ),
                        invalid_actions=invalid,
                        tokens=agent.tokens,
                        cost_usd=agent.cost_usd,
                        runtime_ms=runtime_ms,
                    )
                )
        return rows


def generate_tasks(count: int, seed: int) -> list[IncidentTask]:
    """Generate heterogeneous relational incidents with paired deterministic seeds."""

    tasks: list[IncidentTask] = []
    for index in range(count):
        rng = random.Random(seed + index * 7919)
        component_count = rng.randint(4, 9)
        service_count = rng.randint(3, 7)
        team_count = rng.randint(2, 4)
        components = tuple(f"component:{item}" for item in range(component_count))
        services = tuple(f"service:{item}" for item in range(service_count))
        teams = tuple(f"team:{item}" for item in range(team_count))
        ownership = tuple(
            (component, teams[rng.randrange(team_count)]) for component in components
        )
        dependencies: list[tuple[str, str]] = []
        for service in services:
            for component in rng.sample(
                components,
                k=rng.randint(1, min(3, component_count)),
            ):
                dependencies.append((service, component))
        covered_components = {component for _, component in dependencies}
        for component in components:
            if component not in covered_components:
                dependencies.append((rng.choice(services), component))
        failed_components = tuple(
            rng.sample(components, k=rng.randint(1, min(3, component_count)))
        )
        all_infrastructure = services + components
        health = tuple(
            (
                node,
                round(
                    rng.uniform(0.05, 0.25)
                    if node in failed_components
                    else rng.uniform(0.65, 1.0),
                    3,
                ),
            )
            for node in all_infrastructure
        )
        team_load = tuple((team, round(rng.uniform(0.35, 0.78), 3)) for team in teams)
        criticality = tuple((node, round(rng.uniform(0.4, 1.0), 3)) for node in all_infrastructure)
        tasks.append(
            IncidentTask(
                task_id=f"incident-{index:04d}",
                services=services,
                components=components,
                teams=teams,
                dependencies=tuple(dependencies),
                ownership=ownership,
                failed_components=failed_components,
                initial_health=health,
                team_load=team_load,
                criticality=criticality,
                max_steps=(
                    len(services)
                    + 2 * len(failed_components)
                    + len(teams)
                    + 7
                ),
            )
        )
    return tasks


def run_scaling_benchmark(
    seed: int,
    sizes: Sequence[int] = (10, 25, 50, 100),
    repeats: int = 40,
) -> list[ScalingMeasurement]:
    """Measure observation and transition overhead as graph size grows."""

    rows: list[ScalingMeasurement] = []
    for size in sizes:
        task = _generate_scaled_task(size, seed + size)
        relation_count = len(task.dependencies) + len(task.ownership)
        for representation in REPRESENTATIONS:
            observation_times: list[float] = []
            transition_times: list[float] = []
            payload_bytes = 0
            for repeat in range(repeats):
                env = IncidentSimulator(task, representation, "full", seed + repeat)
                started = perf_counter()
                observation = env.observe()
                observation_times.append((perf_counter() - started) * 1000)
                payload_bytes = len(json.dumps(observation.to_dict(), sort_keys=True))
                started = perf_counter()
                env.step(IncidentCommand("restart", task.services[0]))
                transition_times.append((perf_counter() - started) * 1000)
            rows.append(
                ScalingMeasurement(
                    representation=representation,
                    nodes=len(task.services) + len(task.components) + len(task.teams),
                    relations=relation_count,
                    observation_ms=fmean(observation_times),
                    transition_ms=fmean(transition_times),
                    payload_bytes=payload_bytes,
                )
            )
    return rows


def _generate_scaled_task(size: int, seed: int) -> IncidentTask:
    """Generate a connected incident with approximately ``size`` nodes."""

    rng = random.Random(seed)
    team_count = max(2, size // 20)
    service_count = max(3, size // 3)
    component_count = max(4, size - team_count - service_count)
    services = tuple(f"service:{index}" for index in range(service_count))
    components = tuple(f"component:{index}" for index in range(component_count))
    teams = tuple(f"team:{index}" for index in range(team_count))
    ownership = tuple(
        (component, teams[index % team_count])
        for index, component in enumerate(components)
    )
    dependencies = [
        (services[index % service_count], component)
        for index, component in enumerate(components)
    ]
    for service in services:
        dependencies.append((service, rng.choice(components)))
    failed = tuple(components[: max(1, component_count // 10)])
    infrastructure = services + components
    return IncidentTask(
        task_id=f"scale-{size}",
        services=services,
        components=components,
        teams=teams,
        dependencies=tuple(dict.fromkeys(dependencies)),
        ownership=ownership,
        failed_components=failed,
        initial_health=tuple(
            (node, 0.1 if node in failed else 0.9) for node in infrastructure
        ),
        team_load=tuple((team, 0.5) for team in teams),
        criticality=tuple((node, 1.0) for node in infrastructure),
        max_steps=size,
    )


def normalize_observation(
    observation: IncidentObservation,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Normalize graph, table, and object observations into equivalent facts."""

    payload = observation.payload
    nodes: dict[str, dict[str, Any]] = {}
    dependencies: list[tuple[str, str]] = []
    ownership: list[tuple[str, str]] = []
    if observation.representation.startswith("graph"):
        nodes = {row["id"]: row for row in payload["nodes"]}
        for edge in payload["edges"]:
            pair = (edge["source"], edge["target"])
            if edge["type"] == "DEPENDS_ON":
                dependencies.append(pair)
            else:
                ownership.append(pair)
    elif observation.representation == "tabular":
        for table in ("services", "components", "teams"):
            nodes.update({row["id"]: row for row in payload[table]})
        dependencies = [
            (row["service_id"], row["component_id"])
            for row in payload["service_dependencies"]
        ]
        ownership = [
            (row["component_id"], row["team_id"])
            for row in payload["component_owners"]
        ]
    else:
        for service in payload["services"]:
            nodes[service["id"]] = {
                key: value for key, value in service.items() if key != "dependencies"
            }
            for component in service["dependencies"]:
                owner = component.get("owner", {})
                nodes[component["id"]] = {
                    key: value for key, value in component.items() if key != "owner"
                }
                dependencies.append((service["id"], component["id"]))
                if owner and not owner.get("hidden"):
                    nodes[owner["id"]] = owner
                    ownership.append((component["id"], owner["id"]))
        for component in payload.get("component_registry", []):
            owner = component.get("owner", {})
            nodes[component["id"]] = {
                key: value for key, value in component.items() if key != "owner"
            }
            if owner and not owner.get("hidden"):
                nodes[owner["id"]] = owner
                pair = (component["id"], owner["id"])
                if pair not in ownership:
                    ownership.append(pair)
        for team in payload.get("teams", []):
            nodes[team["id"]] = team
        dependencies = list(dict.fromkeys(dependencies))
    return nodes, sorted(set(dependencies)), sorted(set(ownership))


def observation_features(observation: IncidentObservation) -> tuple[float, ...]:
    """Encode representation-neutral aggregate features for tabular RL."""

    nodes, dependencies, ownership = normalize_observation(observation)
    services = [row for row in nodes.values() if row["type"] == "service"]
    components = [row for row in nodes.values() if row["type"] == "component"]
    teams = [row for row in nodes.values() if row["type"] == "team"]
    failed_services = sum(not row.get("operational", True) for row in services)
    failed_components = sum(not row.get("operational", True) for row in components)
    overloaded = sum(bool(row.get("overloaded")) for row in teams)
    return (
        min(1.0, failed_services / max(1, len(services))),
        min(1.0, failed_components / max(1, len(components))),
        min(1.0, overloaded / max(1, len(teams))),
        min(1.0, len(dependencies) / max(1, len(services) * 3)),
        min(1.0, len(ownership) / max(1, len(components))),
        min(1.0, len(observation.visible_ids) / 20),
    )


def discretize_features(observation: IncidentObservation, bins: int) -> tuple[int, ...]:
    """Discretize normalized observation features."""

    return tuple(min(bins - 1, int(value * bins)) for value in observation_features(observation))


def ground_strategy(strategy: str, observation: IncidentObservation) -> IncidentCommand:
    """Ground a fixed RL action category to a visible entity."""

    nodes, dependencies, ownership = normalize_observation(observation)
    if strategy == "repair":
        failed = sorted(
            node_id
            for node_id, row in nodes.items()
            if row["type"] == "component" and not row.get("operational", True)
        )
        return IncidentCommand("repair", failed[0]) if failed else IncidentCommand("wait")
    if strategy == "restart":
        failed = sorted(
            node_id
            for node_id, row in nodes.items()
            if row["type"] == "service" and not row.get("operational", True)
        )
        return IncidentCommand("restart", failed[0]) if failed else IncidentCommand("wait")
    if strategy == "rebalance":
        overloaded = sorted(
            node_id
            for node_id, row in nodes.items()
            if row["type"] == "team" and row.get("overloaded")
        )
        return IncidentCommand("rebalance", overloaded[0]) if overloaded else IncidentCommand("wait")
    if strategy == "inspect":
        candidates = sorted(
            action.target
            for action in observation.candidate_actions
            if action.action == "inspect" and action.target is not None
        )
        return IncidentCommand("inspect", candidates[0]) if candidates else IncidentCommand("wait")
    return IncidentCommand("wait")


def set_f1(predicted: set[str], expected: set[str]) -> float:
    """Return set-level F1 attribution quality."""

    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def aggregate_measurements(
    rows: Sequence[EpisodeMeasurement],
) -> list[AggregateMeasurement]:
    """Aggregate raw trials by representation, observability, and agent."""

    groups: dict[tuple[str, str, str], list[EpisodeMeasurement]] = defaultdict(list)
    for row in rows:
        groups[(row.representation, row.observability, row.agent)].append(row)
    aggregates: list[AggregateMeasurement] = []
    for (representation, observability, agent), items in sorted(groups.items()):
        rewards = [item.total_reward for item in items]
        by_task: dict[str, list[bool]] = defaultdict(list)
        for item in items:
            by_task[item.task_id].append(item.success)
        consistency = fmean(
            1.0 if all(values) or not any(values) else 0.0 for values in by_task.values()
        )
        aggregates.append(
            AggregateMeasurement(
                representation=representation,
                observability=observability,
                agent=agent,
                episodes=len(items),
                task_success=fmean(float(item.success) for item in items),
                reward_mean=fmean(rewards),
                reward_std=stdev(rewards) if len(rewards) > 1 else 0.0,
                consistency=consistency,
                recovery_rate=fmean(float(item.recovered_after_error) for item in items),
                attribution_f1=fmean(item.attribution_f1 for item in items),
                mean_steps=fmean(item.steps for item in items),
                invalid_action_rate=sum(item.invalid_actions for item in items)
                / max(1, sum(item.steps for item in items)),
                tokens=max(item.tokens for item in items),
                cost_usd=max(item.cost_usd for item in items),
                runtime_ms=fmean(item.runtime_ms for item in items),
            )
        )
    return aggregates


def compute_paired_contrasts(
    rows: Sequence[EpisodeMeasurement],
) -> list[PairedContrast]:
    """Compute paired representation effects on identical task/trial runs."""

    index = {
        (
            row.representation,
            row.observability,
            row.agent,
            row.task_id,
            row.trial,
        ): row
        for row in rows
    }
    contrasts: list[PairedContrast] = []
    comparisons = (
        ("graph_causal", "tabular"),
        ("graph_causal", "object"),
        ("graph_causal", "graph_no_propagation"),
    )
    conditions = sorted({(row.observability, row.agent) for row in rows})
    for reference, comparison in comparisons:
        for observability, agent in conditions:
            paired: list[tuple[EpisodeMeasurement, EpisodeMeasurement]] = []
            for key, reference_row in index.items():
                representation, obs, row_agent, task_id, trial = key
                if representation != reference or obs != observability or row_agent != agent:
                    continue
                comparison_row = index.get(
                    (comparison, obs, row_agent, task_id, trial)
                )
                if comparison_row is not None:
                    paired.append((reference_row, comparison_row))
            if not paired:
                continue
            reward_deltas = [
                left.total_reward - right.total_reward for left, right in paired
            ]
            reward_std = stdev(reward_deltas) if len(reward_deltas) > 1 else 0.0
            contrasts.append(
                PairedContrast(
                    reference=reference,
                    comparison=comparison,
                    observability=observability,
                    agent=agent,
                    pairs=len(paired),
                    success_delta=fmean(
                        float(left.success) - float(right.success)
                        for left, right in paired
                    ),
                    reward_delta=fmean(reward_deltas),
                    reward_ci95=1.96 * reward_std / sqrt(len(reward_deltas)),
                )
            )
    return contrasts


def render_experiment_report(
    config: ExperimentConfig,
    aggregates: Sequence[AggregateMeasurement],
    skipped: Sequence[str],
    scaling: Sequence[ScalingMeasurement] = (),
    contrasts: Sequence[PairedContrast] = (),
) -> str:
    """Render a compact research report from aggregate measurements."""

    lines = [
        "# TwinAIGym Representation and Causality Study",
        "",
        f"- Latent tasks: {config.tasks}",
        f"- Trials per task: {config.trials}",
        f"- Training tasks for Q-learning: {config.train_tasks}",
        f"- Total evaluated episodes: {sum(item.episodes for item in aggregates)}",
        "",
        "## Results",
        "",
        "| Representation | Observation | Agent | Success | Reward | Consistency | "
        "Recovery | Attribution F1 | Steps | Invalid | Runtime ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(
        aggregates,
        key=lambda row: (
            row.representation,
            row.observability,
            -row.task_success,
            row.agent,
        ),
    ):
        lines.append(
            f"| {item.representation} | {item.observability} | {item.agent} | "
            f"{item.task_success:.1%} | {item.reward_mean:.3f} +/- "
            f"{1.96 * item.reward_std / sqrt(item.episodes):.3f} | "
            f"{item.consistency:.1%} | {item.recovery_rate:.1%} | "
            f"{item.attribution_f1:.3f} | {item.mean_steps:.2f} | "
            f"{item.invalid_action_rate:.2%} | {item.runtime_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Every condition is generated from the same latent tasks and action vocabulary.",
            "- `graph_no_propagation` is an ablation of dynamics, not merely serialization.",
            "- Tabular and object conditions retain causal dynamics; they isolate representation.",
            "- The forced first-step symptom restart measures recovery after a controlled error.",
            "- LLM conditions are omitted when no API key is configured; they must not be imputed.",
        ]
    )
    if scaling:
        lines.extend(
            [
                "",
                "## Scalability",
                "",
                "| Representation | Nodes | Relations | Observation ms | "
                "Transition ms | Payload bytes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in scaling:
            lines.append(
                f"| {item.representation} | {item.nodes} | {item.relations} | "
                f"{item.observation_ms:.3f} | {item.transition_ms:.3f} | "
                f"{item.payload_bytes} |"
            )
    if contrasts:
        lines.extend(
            [
                "",
                "## Paired contrasts",
                "",
                "Positive deltas favor the reference representation.",
                "",
                "| Reference | Comparison | Observation | Agent | Pairs | "
                "Success delta | Reward delta | 95% CI |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in contrasts:
            lines.append(
                f"| {item.reference} | {item.comparison} | "
                f"{item.observability} | {item.agent} | {item.pairs} | "
                f"{item.success_delta:+.1%} | {item.reward_delta:+.3f} | "
                f"+/- {item.reward_ci95:.3f} |"
            )
    if skipped:
        lines.extend(["", "## Skipped conditions", ""])
        lines.extend(f"- {item}" for item in skipped)
    return "\n".join(lines) + "\n"
