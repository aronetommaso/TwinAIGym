"""Action primitives for graph transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    command: ActionCommand | None = None


@dataclass(frozen=True, slots=True)
class ActionParameter:
    """Schema entry for one high-level action parameter."""

    type: type
    required: bool = True
    default: Any = None
    choices: tuple[Any, ...] = ()
    description: str = ""

    def validate(self, name: str, value: Any) -> None:
        """Validate a concrete parameter value."""

        if not isinstance(value, self.type):
            raise TypeError(f"Action parameter {name!r} must be {self.type.__name__}.")
        if self.choices and value not in self.choices:
            raise ValueError(f"Action parameter {name!r} must be one of {self.choices!r}.")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Machine-readable definition of a reproducible high-level action."""

    name: str
    parameters: dict[str, ActionParameter] = field(default_factory=dict)
    description: str = ""
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """A high-level action plus validated, serializable parameters."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


class Action:
    """Base class for domain actions.

    Subclasses should override ``check_preconditions`` and ``apply_effects``.
    The public ``apply`` method handles snapshots, diffs, costs, and undo data.
    """

    name = "action"
    cost = 0.0
    parameters: dict[str, ActionParameter] = {}

    def spec(self) -> ActionSpec:
        """Return the public action-space schema for this operator."""

        return ActionSpec(
            name=self.name,
            parameters=dict(self.parameters),
            description=(self.__doc__ or "").strip().splitlines()[0],
            cost=self.cost,
        )

    def validate_command(self, command: ActionCommand) -> dict[str, Any]:
        """Validate a command and fill optional defaults."""

        if command.name != self.name:
            raise ValueError(f"Command {command.name!r} cannot be applied by {self.name!r}.")
        unknown = set(command.parameters) - set(self.parameters)
        if unknown:
            raise ValueError(f"Unknown parameters for {self.name!r}: {sorted(unknown)!r}")
        values: dict[str, Any] = {}
        for name, parameter in self.parameters.items():
            if name in command.parameters:
                value = command.parameters[name]
            elif parameter.required:
                raise ValueError(f"Missing required parameter {name!r} for {self.name!r}.")
            else:
                value = parameter.default
            parameter.validate(name, value)
            values[name] = value
        return values

    def is_valid(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return whether the action can be applied to the current state."""

        return self.check_preconditions(state, parameters) is None

    def check_preconditions(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Return an error message when the action is invalid."""

        return None

    def apply_effects(
        self,
        state: WorldState,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mutate the world state.

        Args:
            state: World state to mutate.

        Returns:
            Optional metadata describing the domain effect.
        """

        return {}

    def apply(
        self,
        state: WorldState,
        command: ActionCommand | None = None,
    ) -> ActionResult:
        """Apply this action and return a structured result."""

        snapshot = state.snapshot()
        command = command or ActionCommand(self.name)
        try:
            parameters = self.validate_command(command)
        except (TypeError, ValueError) as exc:
            return ActionResult(
                action_name=self.name,
                valid=False,
                cost=self.cost,
                message=str(exc),
                snapshot_before=snapshot,
                diff=state.diff(snapshot),
                command=command,
            )
        error = self.check_preconditions(state, parameters)
        if error is not None:
            return ActionResult(
                action_name=self.name,
                valid=False,
                cost=self.cost,
                message=error,
                snapshot_before=snapshot,
                diff=state.diff(snapshot),
                command=command,
            )
        metadata = self.apply_effects(state, parameters)
        state.step_count += 1
        return ActionResult(
            action_name=self.name,
            valid=True,
            cost=self.cost,
            metadata=metadata,
            snapshot_before=snapshot,
            diff=state.diff(snapshot),
            command=command,
        )

    def undo(self, state: WorldState, result: ActionResult) -> None:
        """Undo an applied action using the stored pre-action snapshot."""

        if result.snapshot_before is None:
            raise ValueError("Cannot undo an action result without a snapshot.")
        state.rollback(result.snapshot_before)
