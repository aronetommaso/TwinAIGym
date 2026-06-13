"""Small adapter layer for popular Python agent frameworks.

The adapters intentionally avoid hard imports. LangChain, LangGraph, CrewAI,
AutoGen, and PydanticAI objects differ by version and are often user-defined,
so TwinAIGym treats them as providers that can return text, dicts, or objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from twin_ai_gym.core.observation import Observation


def _extract_action(output: Any, actions: Iterable[str]) -> str:
    """Extract a valid TwinAIGym action name from framework output."""

    action_names = list(actions)
    if not action_names:
        raise ValueError("At least one action name is required.")
    if isinstance(output, str):
        lowered = output.lower()
        for action in action_names:
            if action.lower() in lowered:
                return action
        return output.strip()
    if isinstance(output, dict):
        for key in ("action", "tool", "name", "next_action"):
            value = output.get(key)
            if isinstance(value, str):
                return _extract_action(value, action_names)
        if "output" in output:
            return _extract_action(output["output"], action_names)
    for attr in ("action", "tool", "name", "output", "content"):
        if hasattr(output, attr):
            return _extract_action(getattr(output, attr), action_names)
    return str(output)


def _default_prompt(observation: Observation, actions: list[str]) -> dict[str, Any]:
    """Return a compact framework-friendly input payload."""

    return {
        "observation": observation.to_dict(),
        "actions": actions,
        "instruction": "Choose exactly one action from the actions list.",
    }


@dataclass(slots=True)
class AdapterAgent:
    """Wrap an arbitrary framework object behind TwinAIGym's ``act`` protocol."""

    runner: Any
    actions: list[str]
    input_builder: Callable[[Observation, list[str]], Any] = _default_prompt
    method: str | None = None

    def act(self, observation: Observation) -> str:
        """Run the wrapped agent and return a TwinAIGym action name."""

        payload = self.input_builder(observation, self.actions)
        output = self._call_runner(payload)
        return _extract_action(output, self.actions)

    def _call_runner(self, payload: Any) -> Any:
        """Call common framework execution methods in a stable order."""

        if self.method is not None:
            return getattr(self.runner, self.method)(payload)
        if callable(self.runner):
            return self.runner(payload)
        for method in ("invoke", "run", "predict", "kickoff", "generate_reply", "chat"):
            if hasattr(self.runner, method):
                return getattr(self.runner, method)(payload)
        raise TypeError("Runner must be callable or expose invoke/run/predict/kickoff/generate_reply/chat.")


def langchain_agent(runnable: Any, actions: Iterable[str]) -> AdapterAgent:
    """Adapt a LangChain Runnable or Chain."""

    return AdapterAgent(runner=runnable, actions=list(actions), method="invoke" if hasattr(runnable, "invoke") else None)


def langgraph_agent(graph: Any, actions: Iterable[str]) -> AdapterAgent:
    """Adapt a compiled LangGraph graph."""

    return AdapterAgent(runner=graph, actions=list(actions), method="invoke" if hasattr(graph, "invoke") else None)


def crewai_agent(crew_or_agent: Any, actions: Iterable[str]) -> AdapterAgent:
    """Adapt a CrewAI Crew or Agent-like object."""

    return AdapterAgent(
        runner=crew_or_agent,
        actions=list(actions),
        method="kickoff" if hasattr(crew_or_agent, "kickoff") else None,
    )


def autogen_agent(agent: Any, actions: Iterable[str]) -> AdapterAgent:
    """Adapt an AutoGen conversable agent."""

    return AdapterAgent(
        runner=agent,
        actions=list(actions),
        method="generate_reply" if hasattr(agent, "generate_reply") else None,
    )


def pydanticai_agent(agent: Any, actions: Iterable[str]) -> AdapterAgent:
    """Adapt a PydanticAI Agent."""

    return AdapterAgent(runner=agent, actions=list(actions), method="run_sync" if hasattr(agent, "run_sync") else None)
