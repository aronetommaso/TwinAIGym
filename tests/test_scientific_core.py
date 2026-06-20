"""Tests for the graph-native scientific formulation."""

from twin_ai_gym import ActionCommand, NoisyObservation, compare_agents
from twin_ai_gym.worlds.maintenance import (
    MaintenanceWorld,
    graph_aware_maintenance_agent,
    myopic_maintenance_agent,
)


def test_structured_action_space_validates_parameters() -> None:
    """High-level actions should expose schemas and reject malformed commands."""

    env = MaintenanceWorld(seed=3)

    assert env.action_space["repair_component"].parameters["component_id"].type is str
    _, reward, _, _, info = env.step(ActionCommand("repair_component"))

    assert reward < 0.0
    assert info["valid"] is False
    assert "Missing required parameter" in info["message"]


def test_causal_transition_propagates_over_graph_relations() -> None:
    """Restarting a symptom should fail while its dependency remains down."""

    env = MaintenanceWorld(seed=3)
    _, _, _, _, info = env.step(
        ActionCommand("restart_service", {"service_id": "service:api"})
    )

    assert env.world.get_entity("service:api").attributes["operational"] is False
    assert any(
        item["rule"] == "dependency_failure"
        for item in info["metadata"]["causal_rules"]
    )
    assert "health" in info["diff"].changed_entities["service:api"]


def test_reward_is_auditable_by_component() -> None:
    """Every scalar contribution should retain raw value and weight."""

    env = MaintenanceWorld(seed=4)
    _, _, _, _, info = env.step(
        ActionCommand("repair_component", {"component_id": "component:db"})
    )

    availability = info["reward_attribution"]["availability"]
    assert availability.component == "availability"
    assert availability.contribution == availability.raw_value * availability.weight


def test_noisy_observation_does_not_change_transition_rng() -> None:
    """Observation noise must not perturb stochastic environment dynamics."""

    full = MaintenanceWorld(seed=9)
    noisy = MaintenanceWorld(
        seed=9,
        observation=NoisyObservation(entity_dropout=0.2, numeric_noise=0.1),
    )
    noisy.observe()
    noisy.observe()

    assert full.world.rng.getstate() == noisy.world.rng.getstate()


def test_repeated_benchmark_separates_graph_reasoning_from_myopic_policy() -> None:
    """The causal benchmark should detect a meaningful policy difference."""

    result = compare_agents(
        "maintenance",
        lambda seed: MaintenanceWorld(seed=seed),
        {
            "graph": graph_aware_maintenance_agent,
            "myopic": myopic_maintenance_agent,
        },
        range(5),
    )

    assert result.agents["graph"].mean_score > result.agents["myopic"].mean_score + 0.5
    assert result.agents["graph"].termination_rate == 1.0
