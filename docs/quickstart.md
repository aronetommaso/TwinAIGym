# Quickstart

TwinAIGym environments expose a Gymnasium-inspired interface:

```python
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld

env = CustomerSupportWorld(seed=42)
observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step("reply_ticket")

print(reward)
print(info["reward_components"])
print(info["diff"].summary())
```

The key difference from a standard Gym environment is that the state is a mutable
Knowledge Graph. A step can change entity attributes, relations, events, and metrics.

## Core Concepts

- `WorldState`: mutable graph state with entities, relations, events, snapshots, diffs, and rollback.
- `Action`: domain transition with preconditions, effects, cost, and undo support.
- `ObservationPolicy`: controls what part of the graph an agent can see.
- `RewardComponent`: computes reward from the transition between two graph states.
- `TwinEnv`: wraps a world into a reset/step/render interface.

## Determinism

Pass a seed to a world to make stochastic transitions reproducible:

```python
env = CustomerSupportWorld(seed=7)
```

Snapshots include the random generator state, so rollback restores stochastic behavior too.
