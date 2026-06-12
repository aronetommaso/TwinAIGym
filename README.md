# TwinAIGym

**TwinAIGym is an open-source testing and benchmarking platform for AI agents.**

It helps you test agents before production by running them inside mutable business
simulations: customer support queues, CRM workflows, sales pipelines, logistics operations,
procurement flows, HR processes, and other enterprise scenarios where actions have delayed
consequences.

The technical foundation is a graph-native digital twin. Each environment stores state as a
dynamic Knowledge Graph, applies domain actions, computes rewards from graph transitions, and
records snapshots, diffs, rollback data, events, metrics, and replayable episodes.

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

## Current MVP

The first included benchmark world is `CustomerSupportWorld`.

It models:

- Customers, tickets, agents, and specialist teams.
- Actions such as reply, escalate, refund, ask for more information, and ignore.
- Rewards for satisfaction, resolution rate, SLA behavior, and refund cost.
- Standard and adversarial scenarios.

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

- Formal Gymnasium adapter.
- Pytest plugin for `assert agent.score(env) > threshold` workflows.
- LangChain, LangGraph, CrewAI, AutoGen, and PydanticAI adapters.
- Additional benchmark worlds: Sales, CRM, Startup Ops, Logistics, Procurement, HR.
- Marketplace-style environment packages such as `twinaigym-sales` and `twinaigym-banking`.
- Interactive graph/timeline renderer for screenshots and debugging.

## Design Principles

Worlds should expose domain-native APIs such as `add_customer()` or `add_ticket()` instead of
forcing users to manage low-level graph nodes and edges.

The Knowledge Graph is the state. The transition rules are the simulator. The benchmark is the
contract that tells you whether an agent is ready to ship.
