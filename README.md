# TwinAIGym

TwinAIGym is a Python library for building graph-native digital twin environments where AI
agents can act, observe consequences, receive rewards, and be evaluated before deployment.

The project is intentionally shaped like a lightweight Gym-style environment library, but the
state is a dynamic Knowledge Graph rather than a flat dictionary. The core package provides:

- Entities, relations, and mutable world state.
- Snapshots, rollback, state diffs, event logs, and episode replay.
- Declarative actions with preconditions, effects, costs, and undo support.
- Observation policies for full-state and local-subgraph views.
- Reward components and aggregators computed from graph transitions.
- A concrete `CustomerSupportWorld` MVP implemented using the public core API.

## Quickstart

```python
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld

env = CustomerSupportWorld(seed=42)
observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step("reply_ticket")
print(reward)
print(info["diff"].summary())
```

## Installation

For editable local development:

```bash
pip install -e ".[dev,yaml]"
```

For a minimal install:

```bash
pip install .
```

## Design Principles

TwinAIGym is not a graph database wrapper and does not try to provide RL training algorithms.
It provides the environments in which agents can be trained or evaluated.

World authors should expose domain-native APIs such as `add_customer()` or `add_ticket()`
instead of asking users to manually manage nodes and edges.
