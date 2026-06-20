# Scientific Formulation

TwinAIGym models an enterprise environment as a graph-valued controlled stochastic process:

\[
G_{t+1} \sim P_\theta(G_{t+1}\mid G_t, A_t), \qquad
O_t \sim \Omega_\phi(O_t\mid G_t), \qquad
R_t = \sum_k w_k r_k(G_t, A_t, G_{t+1}).
\]

The scientific claim is not that knowledge graphs are novel storage. The claim tested by the
included maintenance benchmark is:

> Evolving graph state makes relational and delayed consequences explicit, enabling reproducible
> evaluation of agent policies that reason about dependencies.

## First-class primitives

- `WorldState` is the latent evolving graph \(G_t\).
- `ActionCommand` is the typed high-level decision emitted by an agent.
- `Action` is the validated graph operator associated with that command.
- `TransitionModel` defines the dynamics and can apply named `PropagationRule` objects.
- `StateDiff` records the exact transition for audit and replay.
- `ObservationPolicy` defines \(\Omega\). `FullObservation` is an MDP view;
  `LocalSubgraphObservation` and `NoisyObservation` induce POMDP views.
- `RewardComponent` defines one measurable business objective. `RewardAttribution` preserves its
  raw value, weight, and weighted contribution.
- `compare_agents` evaluates policies on paired seeds and reports mean, sample standard deviation,
  confidence interval, termination rate, and episode length.

## Reproducibility contract

Action parameters are schema-validated before state mutation. Transition randomness and
observation noise use independent seeded random streams. Snapshots preserve both streams.
Every valid transition records the command, fired causal rules, graph diff, and component-level
reward attribution.

## Falsifiable benchmark

`MaintenanceWorld` contains services, components, and owning teams linked by `DEPENDS_ON` and
`OWNED_BY` relations. Restarting an unavailable service does not solve an outage while its database
dependency is down. The graph-aware reference policy repairs the failed dependency before
restarting the service; the myopic baseline repeatedly treats the visible symptom.

Run:

```bash
python examples/scientific_benchmark.py
```

The script evaluates graph-aware, myopic, and random policies on 30 paired seeds and writes JSON
and Markdown results under `benchmarks/`.
