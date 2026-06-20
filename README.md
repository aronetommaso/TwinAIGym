# TwinAIGym

**TwinAIGym is a graph-native evaluation framework for autonomous agents.**

It helps you test agents before production by running them inside mutable business
simulations: customer support queues, CRM workflows, sales pipelines, logistics operations,
procurement flows, HR processes, and other enterprise scenarios where actions have delayed
consequences.

Its scientific thesis is:

> Representing enterprise environments as evolving knowledge graphs enables auditable state
> transitions, explainable reward attribution, and reproducible evaluation of agents under
> relational and delayed consequences.

The graph is not merely storage. Each environment defines typed actions, an explicit transition
model, causal propagation rules, an observation model, compositional reward components, and
replayable graph trajectories.

```python
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld

env = CustomerSupportWorld.adversarial(seed=42)
result = env.evaluate(my_agent, seed=42)

print(result.report())
assert result.score > 0.85
```

Or run a benchmark suite:

```python
from twin_ai_gym.worlds.customer_support import customer_support_suite

suite = customer_support_suite(seed=42)
result = suite.evaluate(my_agent, seed=42)

print(result.report())
```

## Why TwinAIGym?

Most agent evaluations are still static: prompt in, answer out, judge score. Real agents operate
in stateful systems. A bad support response can reopen a ticket tomorrow. A poor CRM action can
hurt revenue next week. A risky tool call can trigger an incident that only becomes visible after
several steps.

TwinAIGym is designed for that missing layer:

- Regression testing for agents in CI/CD.
- Benchmarking business-process agents across repeatable scenarios.
- Adversarial scenarios where agents break before production.
- Graph-native state, so business dependencies are explicit and inspectable.
- Diff, snapshot, rollback, and replay for debugging agent behavior.

## Formal Model

TwinAIGym environments implement a graph-valued MDP or POMDP:

```text
G(t+1) ~ P(G(t+1) | G(t), A(t))
O(t)   ~ Omega(O(t) | G(t))
R(t)    = sum_k weight(k) * reward_component(k)
```

- `WorldState` is the latent evolving graph.
- `ActionCommand` is a typed high-level action with validated parameters.
- `Action` is the graph operator that realizes the command.
- `TransitionModel` and named `PropagationRule` objects define the dynamics.
- `FullObservation`, `LocalSubgraphObservation`, and `NoisyObservation` distinguish MDP and
  POMDP evaluation.
- `StateDiff` makes every transition inspectable.
- `RewardAttribution` retains raw values, weights, and contributions behind the scalar reward.

See [the scientific formulation](docs/scientific_formulation.md).

## Included Worlds

`CustomerSupportWorld` models:

It models:

- Customers, tickets, agents, and specialist teams.
- Actions such as reply, escalate, refund, ask for more information, and ignore.
- Rewards for satisfaction, resolution rate, SLA behavior, and refund cost.
- Standard and adversarial scenarios.

`MaintenanceWorld` is the causal graph benchmark. Services depend on components, components are
owned by teams, incidents propagate through `DEPENDS_ON` and `OWNED_BY`, and symptom-level actions
fail until the root cause is repaired.

Generated benchmark worlds also cover Sales, CRM, Startup Operations, Logistics, Procurement,
and HR.

### Structured action example

```python
from twin_ai_gym import ActionCommand
from twin_ai_gym.worlds import MaintenanceWorld

env = MaintenanceWorld(seed=42)
observation, _ = env.reset()

observation, reward, terminated, truncated, info = env.step(
    ActionCommand("repair_component", {"component_id": "component:db"})
)

print(info["metadata"]["causal_rules"])
print(info["reward_attribution"])
print(info["diff"].summary())
```

Run the example:

```bash
python examples/customer_support_basic.py
```

Example output shape:

```text
Score: 56.40%
Total reward: 1.618
Steps: 13
Average Satisfaction: 0.774
Resolution Rate: 0.750
Sla Violations: 0.000
Refund Cost: 215.000
Passed benchmark: True
```

## Installation

For local development:

```bash
pip install -e ".[dev,yaml]"
```

For a minimal install:

```bash
pip install .
```

If you run examples without installing the package, set `PYTHONPATH` first:

```powershell
$env:PYTHONPATH = ".\src"
python examples/customer_support_basic.py
```

## Positioning

TwinAIGym is not another RL algorithm library. It provides the environments where agents can be
tested, compared, and improved.

Compared with generic Gym-style environments, TwinAIGym is business-process-first and
graph-native. Compared with static agent benchmarks, TwinAIGym evaluates consequences over time.
Compared with industrial simulator platforms, TwinAIGym aims to be small, Pythonic, composable,
and easy to extend.

## Roadmap

### Implemented

- Graph-native `TwinEnv` with `step`, `reset`, `evaluate`, snapshots, diffs, rollback,
  events, metrics, and replayable episodes.
- Customer support benchmark world with standard and adversarial scenarios.
- Explicit `TransitionModel` dynamics and named causal graph propagation rules.
- Typed `ActionCommand`/`ActionSpec` action spaces with parameter validation.
- MDP, local-subgraph, and seeded noisy POMDP observation policies.
- Component-level reward attribution with raw values and weights.
- Causal maintenance benchmark with graph-aware, myopic, and random baselines.
- Multi-seed comparison with standard deviation, confidence intervals, and termination rate.
- Benchmark suite API for evaluating one agent across multiple named cases.
- Formal Gymnasium-compatible adapter through `twin_ai_gym.adapters.make_gymnasium_env`.
- Pytest plugin and helper for threshold-based benchmark assertions.
- Lightweight adapters for LangChain, LangGraph, CrewAI, AutoGen, and PydanticAI-style
  agents.
- Additional generated business benchmark worlds: Sales, CRM, Startup Ops, Logistics,
  Procurement, and HR.
- Marketplace-style package metadata via `list_environment_packages()`, including built-in
  packages such as `twinaigym-sales` and external package slots such as `twinaigym-banking`.
- Interactive HTML graph/timeline renderer for debugging and replay inspection.
- Example workflows for business-suite evaluation, Gymnasium-style policy search, and
  LLM prompt training/evaluation with replay export.
- RL-ready vector observation encoders for customer support, generated business worlds,
  and generic graph-state environments.
- Gymnasium wrapper support for fixed-size numeric observations and discrete action
  spaces, suitable for classical RL baselines.
- Trajectory collection and JSONL export for RL, imitation learning, and fine-tuning
  datasets.
- RL training metrics and dependency-free HTML reports for score, reward, and held-out
  evaluation.
- CPU-only Q-learning project example for customer support optimization.

### Still Missing

- Direct integration examples for Stable-Baselines3, RLlib, and CleanRL.
- Supervised policy training examples from collected trajectories, including logistic
  regression, decision trees, and small neural policies.
- Fine-tuning dataset builders for LLM providers and local open-weight models.
- Robust LLM action parsing with JSON schema validation, retry logic, invalid-action
  recovery, and cost/token tracking.
- More domain-specific business worlds with richer rules instead of only generated generic
  process dynamics.
- Hosted leaderboard and externally submitted benchmark cards.
- Stronger CI coverage, package publishing workflow, and documentation for third-party
  environment packages.

## Design Principles

Worlds should expose domain-native APIs such as `add_customer()` or `add_ticket()` instead of
forcing users to manage low-level graph nodes and edges.

The Knowledge Graph is the evolving state. Typed commands select graph operators. The transition
model propagates consequences. Graph diffs explain what changed. Reward attribution explains why
it mattered. The benchmark is the reproducible contract.

## Benchmark

Run the paper-style causal benchmark:

```powershell
$env:PYTHONPATH = ".\src"
python examples\scientific_benchmark.py
```

Thirty paired seeds produce:

| Agent | Mean score | Std | 95% CI | Termination | Mean steps |
|---|---:|---:|---:|---:|---:|
| Graph-aware | 0.956 | 0.008 | +/- 0.003 | 100.0% | 2.60 |
| Random | 0.726 | 0.238 | +/- 0.085 | 43.3% | 7.20 |
| Myopic service restart | 0.068 | 0.045 | +/- 0.016 | 0.0% | 8.00 |

The graph-aware policy repairs the failed dependency before restarting the affected service. The
myopic policy repeatedly restarts the visible service, after which the causal transition model
immediately propagates the unresolved database outage back through `DEPENDS_ON`.

Machine-readable results are in
[`benchmarks/maintenance_results.json`](benchmarks/maintenance_results.json); the generated table
is in [`benchmarks/maintenance_results.md`](benchmarks/maintenance_results.md).

## Controlled Representation Study

TwinAIGym also ships a paired experiment over 120 latent tasks, four state/dynamics conditions,
three observation regimes, and three local policy baselines:

```powershell
$env:PYTHONPATH = ".\src"
python examples\representation_benchmark.py
```

The current offline run contains 12,960 evaluated episodes plus a 10–100 node scaling study. Its
most important result is deliberately conservative: graph, tabular, and object representations
produce identical outcomes for a symbolic heuristic when they expose equivalent facts. Causal
propagation, however, substantially changes policy success and reveals failures hidden by static
dynamics.

Three OpenAI-compatible LLM/tool-agent conditions are implemented with token and cost tracking.
They are reported as skipped when no API key is configured; missing model results are never
imputed.

See [the experimental protocol](docs/representation_study.md) and the generated
[study report](benchmarks/representation_study/REPORT.md).
