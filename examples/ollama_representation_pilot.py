"""Run a local-LLM first-action study over equivalent representations."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from twin_ai_gym.experiments.representation_benchmark import (
    IncidentSimulator,
    OllamaIncidentAgent,
    REPRESENTATIONS,
    StructuralHeuristicAgent,
    generate_tasks,
)


def main() -> None:
    """Compare three tool-agent prompts with one call per representation."""

    task = generate_tasks(1, 2026)[0]
    oracle = StructuralHeuristicAgent()
    agents = [
        OllamaIncidentAgent("qwen3:4b", "direct"),
        OllamaIncidentAgent("qwen3:4b", "causal"),
        OllamaIncidentAgent("qwen3:4b", "audit"),
    ]
    rows = []
    for representation in REPRESENTATIONS:
        env = IncidentSimulator(task, representation, "full", seed=2026)
        observation = env.observe()
        expected = oracle.act(observation)
        for agent in agents:
            started = perf_counter()
            predicted = agent.act(observation)
            runtime_ms = (perf_counter() - started) * 1000
            rows.append(
                {
                    "representation": representation,
                    "agent": agent.name,
                    "expected": asdict(expected),
                    "predicted": asdict(predicted),
                    "correct": predicted == expected,
                    "tokens": agent.tokens,
                    "runtime_ms": runtime_ms,
                }
            )

    output = Path("benchmarks/representation_study")
    output.mkdir(parents=True, exist_ok=True)
    (output / "ollama_pilot.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Ollama First-Action Representation Pilot",
        "",
        "This is a 12-call parser/reasoning smoke test, not an episode-level benchmark.",
        "",
        "| Representation | Agent | Correct | Expected | Predicted | Tokens | Runtime ms |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation']} | {row['agent']} | "
            f"{'yes' if row['correct'] else 'no'} | "
            f"`{row['expected']}` | `{row['predicted']}` | "
            f"{row['tokens']} | {row['runtime_ms']:.1f} |"
        )
    (output / "OLLAMA_PILOT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed {len(rows)} local-LLM decisions.")


if __name__ == "__main__":
    main()
