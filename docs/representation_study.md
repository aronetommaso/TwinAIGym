# Representation and Causality Study

This experiment tests two separate hypotheses:

1. **Representation hypothesis:** graph-native observations help agents solve relational tasks
   compared with equivalent relational tables or nested Python objects.
2. **Dynamics hypothesis:** explicit causal propagation exposes failures that simpler static
   environments hide.

These hypotheses must not be conflated. A graph can be convenient without improving policy
performance, while causal dynamics can make evaluation more realistic without making tasks easier.

## Controlled design

The generator creates 120 latent infrastructure incidents. Each contains:

- 3–7 services;
- 4–9 components;
- 2–4 owning teams;
- typed service dependencies and component ownership;
- 1–3 failed root components;
- heterogeneous health, load, and business criticality.

Every latent task is materialized in four conditions:

- `graph_causal`: graph observation and causal propagation;
- `graph_no_propagation`: identical initial graph, but post-action propagation disabled;
- `tabular`: equivalent normalized tables with foreign keys and causal propagation;
- `object`: equivalent nested Python-object serialization and causal propagation.

Each representation is evaluated with:

- full observation;
- partial observation with an explicit `inspect` action;
- noisy observation with node dropout and numeric perturbation.

The test suite verifies that full graph, tabular, and object observations normalize to exactly the
same facts. This prevents accidental information advantages.

## Agents

- `heuristic`: one structural strategy with representation-specific parsing only;
- `random`: seeded random policy over visible structured commands;
- `q_learning`: dependency-free tabular Q-learning, trained separately for every condition on
  300 held-out generated incidents;
- `llm_direct`, `llm_causal`, `llm_audit`: optional OpenAI-compatible JSON tool agents using three
  prompting strategies.

LLM runs are limited to 20 tasks by default because the full factorial study can require thousands
of paid model calls. Missing API conditions are reported as skipped and never replaced by synthetic
scores.

A separate `examples/ollama_representation_pilot.py` performs a 12-call first-action comparison
for local models. This is intentionally separated from episode-level scores: a slow local model
should not make the main offline experiment incomplete or silently produce partial results.

## Metrics

- task success;
- total compositional reward and confidence interval;
- consistency across three trials;
- recovery after a forced first-step symptom-level error;
- causal attribution F1 against latent ground truth;
- steps and invalid-action rate;
- token usage and estimated API cost;
- runtime, serialized payload size, and scaling from 10 to 100 nodes.

## Run

```powershell
$env:PYTHONPATH = ".\src"
python examples\representation_benchmark.py
```

To execute configured LLM agents:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "your-model"
python examples\representation_benchmark.py --llm-tasks 20
```

Artifacts are written to `benchmarks/representation_study/`.

## Current offline finding

The local 12,960-episode run does **not** establish a graph representation advantage. The
structural heuristic obtains identical results on graph, tabular, and object encodings when their
information is equivalent. This is the correct control result: serialization alone should not
improve a symbolic policy that normalizes all formats.

The experiment does establish that causal propagation changes benchmark difficulty. Random and
small Q-learning policies often appear stronger when propagation is disabled, because symptom-level
actions are not invalidated by unresolved dependencies. The graph-causal condition therefore
detects weaknesses hidden by a static transition model.

Testing whether raw graph serialization improves LLM/tool-agent reasoning remains pending until
the three configured model conditions are executed. The report marks these cells as skipped rather
than drawing a conclusion from absent data.
