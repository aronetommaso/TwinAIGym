"""Prompt-train and evaluate an LLM-style agent with TwinAIGym.

By default this runs offline with a deterministic heuristic LLM. Set
``OPENAI_API_KEY`` to evaluate a real OpenAI-compatible chat model.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from twin_ai_gym import Observation
from twin_ai_gym.renderers import render_graph_timeline
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld, customer_support_suite


SYSTEM_PROMPTS = [
    "You are a careful support operations agent. Choose one allowed action.",
    "Prioritize SLA, reduce risk, and avoid unnecessary refunds. Choose one allowed action.",
    "Resolve easy tickets, ask for more info on hard tickets, and escalate risky tickets.",
]


@dataclass(slots=True)
class LLMActionAgent:
    """LLM-backed agent that maps model text to an environment action."""

    prompt: str
    model: str = "offline-heuristic"

    def act(self, observation: Observation) -> str:
        """Return one action for a TwinAIGym observation."""

        actions = ["reply_ticket", "escalate_ticket", "refund_customer", "ask_more_info", "ignore_ticket"]
        payload = {"system": self.prompt, "observation": observation.to_dict(), "actions": actions}
        text = call_llm(payload, model=self.model)
        lowered = text.lower()
        for action in actions:
            if action in lowered:
                return action
        return "reply_ticket"


def call_llm(payload: dict[str, Any], model: str) -> str:
    """Call a real chat model when configured, otherwise use an offline heuristic."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        body = json.dumps(
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                "messages": [
                    {"role": "system", "content": payload["system"]},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])
    return offline_heuristic_llm(payload)


def offline_heuristic_llm(payload: dict[str, Any]) -> str:
    """Deterministic local stand-in for an LLM action response."""

    entities = payload["observation"]["entities"]
    tickets = [entity for entity in entities.values() if entity["type"] == "Ticket"]
    open_tickets = [ticket for ticket in tickets if ticket["attributes"].get("status") == "open"]
    if not open_tickets:
        return "reply_ticket"
    ticket = sorted(open_tickets, key=lambda value: -float(value["attributes"].get("priority", 0.0)))[0]
    attrs = ticket["attributes"]
    if attrs.get("risk") == "prompt_injection":
        return "escalate_ticket"
    if float(attrs.get("difficulty", 0.0)) > 0.72:
        return "ask_more_info"
    if int(attrs.get("age_hours", 0)) > int(attrs.get("sla_hours", 24)):
        return "refund_customer"
    return "reply_ticket"


def train_prompt() -> tuple[str, dict[str, float]]:
    """Select the best prompt on a small deterministic training split."""

    scores: dict[str, float] = {}
    for prompt in SYSTEM_PROMPTS:
        agent = LLMActionAgent(prompt=prompt)
        result = CustomerSupportWorld.adversarial(seed=21).evaluate(agent, episodes=3, seed=21)
        scores[prompt] = result.score
    best_prompt = max(scores, key=scores.get)
    return best_prompt, scores


def main() -> None:
    """Train, evaluate, and render a replay."""

    best_prompt, training_scores = train_prompt()
    agent = LLMActionAgent(prompt=best_prompt)
    suite_result = customer_support_suite(seed=100).evaluate(agent, seed=100)

    replay_env = CustomerSupportWorld.adversarial(seed=100)
    replay_env.evaluate(agent, seed=100)
    render_graph_timeline(replay_env, "llm_customer_support_replay.html", title="LLM Customer Support Replay")

    print("Training prompt scores:")
    for prompt, score in training_scores.items():
        print(f"- {score:.2%}: {prompt}")
    print()
    print("Best prompt:")
    print(best_prompt)
    print()
    print(suite_result.report())
    print("Replay written to llm_customer_support_replay.html")


if __name__ == "__main__":
    main()
