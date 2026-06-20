"""Explicit transition models for graph-native environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from twin_ai_gym.core.action import Action, ActionCommand, ActionResult
from twin_ai_gym.core.world import StateDiff, WorldState


class TransitionModel(Protocol):
    """Defines the environment dynamics P(G_next | G, action)."""

    stochastic: bool

    def transition(
        self,
        state: WorldState,
        action: Action,
        command: ActionCommand,
    ) -> ActionResult:
        """Sample and apply one state transition."""


class DirectTransitionModel:
    """Default model where the selected graph operator owns all effects."""

    stochastic = False

    def transition(
        self,
        state: WorldState,
        action: Action,
        command: ActionCommand,
    ) -> ActionResult:
        """Apply the selected action operator."""

        return action.apply(state, command)


RuleEffect = Callable[[WorldState, ActionResult], dict[str, Any] | None]


@dataclass(slots=True)
class PropagationRule:
    """Named causal rule applied after a valid action."""

    name: str
    effect: RuleEffect
    probability: float = 1.0
    description: str = ""

    def apply(self, state: WorldState, result: ActionResult) -> dict[str, Any] | None:
        """Sample this rule and apply its graph effect."""

        if self.probability < 1.0 and state.rng.random() > self.probability:
            return None
        return self.effect(state, result)


@dataclass(slots=True)
class RuleBasedTransitionModel:
    """Action dynamics followed by explicit causal graph propagation."""

    rules: list[PropagationRule] = field(default_factory=list)
    stochastic: bool = True

    def transition(
        self,
        state: WorldState,
        action: Action,
        command: ActionCommand,
    ) -> ActionResult:
        """Apply an action and propagate its consequences through named rules."""

        result = action.apply(state, command)
        if not result.valid or result.snapshot_before is None:
            return result
        fired: list[dict[str, Any]] = []
        for rule in self.rules:
            metadata = rule.apply(state, result)
            if metadata is not None:
                fired.append({"rule": rule.name, **metadata})
                state.emit("CausalRuleFired", rule=rule.name, details=metadata)
        result.metadata["causal_rules"] = fired
        result.diff = state.diff(result.snapshot_before)
        return result


def changed_attributes(diff: StateDiff | None) -> set[tuple[str, str]]:
    """Return entity/attribute pairs changed by a transition."""

    if diff is None:
        return set()
    return {
        (entity_id, attribute)
        for entity_id, changes in diff.changed_entities.items()
        for attribute in changes
    }
