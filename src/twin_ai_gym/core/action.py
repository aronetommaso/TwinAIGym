"""Action primitives for graph transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from twin_ai_gym.core.world import StateDiff, WorldSnapshot, WorldState


@dataclass(slots=True)
class ActionResult:
    """Result returned after applying an action."""

    action_name: str
    valid: bool
    cost: float = 0.0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_before: WorldSnapshot | None = None
    diff: StateDiff | None = None


class Action:
    """Base class for domain actions.

    Subclasses should override ``check_preconditions`` and ``apply_effects``.
    The public ``apply`` method handles snapshots, diffs, costs, and undo data.
    """

    name = "action"
    cost = 0.0

    def is_valid(self, state: WorldState) -> bool:
        """Return whether the action can be applied to the current state."""

        return self.check_preconditions(state) is None

    def check_preconditions(self, state: WorldState) -> str | None:
        """Return an error message when the action is invalid."""

        return None

    def apply_effects(self, state: WorldState) -> dict[str, Any]:
        """Mutate the world state.

        Args:
            state: World state to mutate.

        Returns:
            Optional metadata describing the domain effect.
        """

        return {}

    def apply(self, state: WorldState) -> ActionResult:
        """Apply this action and return a structured result."""

        snapshot = state.snapshot()
        error = self.check_preconditions(state)
        if error is not None:
            return ActionResult(
                action_name=self.name,
                valid=False,
                cost=self.cost,
                message=error,
                snapshot_before=snapshot,
                diff=state.diff(snapshot),
            )
        metadata = self.apply_effects(state)
        state.step_count += 1
        return ActionResult(
            action_name=self.name,
            valid=True,
            cost=self.cost,
            metadata=metadata,
            snapshot_before=snapshot,
            diff=state.diff(snapshot),
        )

    def undo(self, state: WorldState, result: ActionResult) -> None:
        """Undo an applied action using the stored pre-action snapshot."""

        if result.snapshot_before is None:
            raise ValueError("Cannot undo an action result without a snapshot.")
        state.rollback(result.snapshot_before)
