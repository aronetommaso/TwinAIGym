# Build Your Own World

Worlds should expose domain-native APIs and hide low-level graph manipulation from users.
The goal is to create environments where agents can be tested and benchmarked before they touch
production systems.
Prefer this:

```python
world.add_customer("customer:acme")
world.add_ticket("ticket:1", customer_id="customer:acme")
```

over this:

```python
world.add_entity(...)
world.add_relation(...)
```

The graph primitives are still available, but a good world feels like the domain it models.

## Minimal Structure

Create a package under `src/twin_ai_gym/worlds/my_world`:

```text
my_world/
  __init__.py
  actions.py
  rewards.py
  env.py
```

## Action Pattern

Actions encode the transition function of the digital twin.

```python
from twin_ai_gym.core.action import Action
from twin_ai_gym.core.world import WorldState


class ResolveTaskAction(Action):
    """Resolve the highest-priority task."""

    name = "resolve_task"
    cost = 0.1

    def check_preconditions(self, state: WorldState) -> str | None:
        if not state.find_entities("Task", status="open"):
            return "No open task is available."
        return None

    def apply_effects(self, state: WorldState) -> dict:
        task = state.find_entities("Task", status="open")[0]
        task.attributes["status"] = "done"
        state.emit("TaskResolved", task_id=task.id)
        return {"task_id": task.id}
```

## Reward Pattern

Rewards should be computed from the graph transition, not from a single isolated variable.

```python
from twin_ai_gym.core.reward import RewardComponent


class CompletionReward(RewardComponent):
    """Reward newly completed tasks."""

    name = "completion"
    weight = 1.0

    def compute(self, before, after, action_result) -> float:
        before_done = {
            entity_id
            for entity_id, entity in before.entities.items()
            if entity.type == "Task" and entity.attributes.get("status") == "done"
        }
        after_done = {
            entity_id
            for entity_id, entity in after.entities.items()
            if entity.type == "Task" and entity.attributes.get("status") == "done"
        }
        return float(len(after_done - before_done))
```

## Design Rule

If a world needs a capability, add it to the generic core only when it is useful across domains.
This keeps the framework general while letting each world stay expressive.

## Benchmark Rule

Every world should define:

- A repeatable standard scenario.
- At least one adversarial scenario.
- A compact `metrics()` method.
- A normalized `score()` method.
- One short example that can be copied into a CI regression test.
