"""Tests for the controlled state-representation experiment."""

from twin_ai_gym.experiments.representation_benchmark import (
    EpisodeMeasurement,
    IncidentCommand,
    IncidentSimulator,
    StructuralHeuristicAgent,
    compute_paired_contrasts,
    generate_tasks,
    normalize_observation,
    run_scaling_benchmark,
)


def test_generated_tasks_are_relational_and_reproducible() -> None:
    """Every generated component should participate in a dependency graph."""

    first = generate_tasks(8, 12)
    second = generate_tasks(8, 12)

    assert first == second
    for task in first:
        covered = {component for _, component in task.dependencies}
        assert covered == set(task.components)
        assert task.failed_components


def test_representations_expose_equivalent_full_information() -> None:
    """Graph, table, and object encodings should normalize to the same facts."""

    task = generate_tasks(1, 7)[0]
    normalized = []
    for representation in ("graph_causal", "tabular", "object"):
        env = IncidentSimulator(task, representation, "full", seed=7)
        normalized.append(normalize_observation(env.observe()))

    assert normalized[0] == normalized[1] == normalized[2]


def test_no_propagation_is_a_transition_ablation_only() -> None:
    """Causal and no-propagation variants must start from the same incident."""

    task = generate_tasks(1, 9)[0]
    causal = IncidentSimulator(task, "graph_causal", "full", seed=9)
    ablated = IncidentSimulator(task, "graph_no_propagation", "full", seed=9)

    assert causal.state.operational == ablated.state.operational
    service = next(
        source
        for source, component in task.dependencies
        if component in task.failed_components
    )
    causal.step(IncidentCommand("restart", service))
    ablated.step(IncidentCommand("restart", service))

    assert causal.state.operational[service] is False
    assert ablated.state.operational[service] is True

    partial = IncidentSimulator(task, "graph_no_propagation", "partial", seed=9)
    partial.step(IncidentCommand("restart", service))
    assert service in partial.observe().visible_ids


def test_inspection_expands_partial_observation() -> None:
    """Inspecting a component should reveal its ownership neighborhood."""

    task = generate_tasks(1, 15)[0]
    env = IncidentSimulator(task, "graph_causal", "partial", seed=15)
    before = env.observe()
    component = next(
        node for node in before.visible_ids if node in task.components
    )
    env.step(IncidentCommand("inspect", component))
    after = env.observe()
    owner = dict(task.ownership)[component]

    assert owner in after.visible_ids


def test_structural_heuristic_runs_on_every_representation() -> None:
    """The same decision strategy should be executable across all encodings."""

    task = generate_tasks(1, 23)[0]
    agent = StructuralHeuristicAgent()
    commands = []
    for representation in ("graph_causal", "tabular", "object"):
        env = IncidentSimulator(task, representation, "full", seed=23)
        commands.append(agent.act(env.observe()))

    assert commands[0] == commands[1] == commands[2]


def test_scaling_benchmark_covers_all_representations() -> None:
    """Scaling measurements should report payload and runtime for every backend."""

    rows = run_scaling_benchmark(3, sizes=(10,), repeats=2)

    assert {row.representation for row in rows} == {
        "graph_causal",
        "graph_no_propagation",
        "tabular",
        "object",
    }
    assert all(row.payload_bytes > 0 for row in rows)


def test_equivalent_representations_have_zero_heuristic_contrast() -> None:
    """Paired graph and table runs should agree for the normalized heuristic."""

    task = generate_tasks(1, 31)[0]
    agent = StructuralHeuristicAgent()
    rows = []
    for representation in ("graph_causal", "tabular"):
        env = IncidentSimulator(task, representation, "full", seed=31)
        total_reward = 0.0
        while not env.done():
            transition = env.step(agent.act(env.observe()))
            total_reward += transition.reward
        rows.append(
            EpisodeMeasurement(
                task_id=task.task_id,
                representation=representation,
                observability="full",
                agent=agent.name,
                trial=0,
                success=env.success(),
                total_reward=total_reward,
                steps=env.state.steps,
                recovered_after_error=False,
                attribution_f1=0.0,
                invalid_actions=0,
                tokens=0,
                cost_usd=0.0,
                runtime_ms=0.0,
            )
        )

    contrast = compute_paired_contrasts(rows)[0]
    assert contrast.success_delta == 0.0
    assert contrast.reward_delta == 0.0
