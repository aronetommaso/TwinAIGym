"""Tests for the customer support MVP world."""

from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def test_step_produces_diff_and_reward() -> None:
    """A domain action should mutate graph state and produce a reward."""

    env = CustomerSupportWorld(seed=7)
    _, _ = env.reset()

    _, reward, terminated, truncated, info = env.step("reply_ticket")

    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["valid"] is True
    assert info["diff"] is not None
    assert info["diff"].summary()


def test_snapshot_and_rollback_restore_state() -> None:
    """Rollback should restore graph attributes after an action."""

    env = CustomerSupportWorld(seed=7)
    snapshot = env.snapshot()
    before = env.render()

    env.step("ignore_ticket")
    env.rollback(snapshot)

    assert env.render() == before


def test_metrics_are_derived_from_graph() -> None:
    """Metrics should expose the world-level evaluation surface."""

    env = CustomerSupportWorld(seed=7)
    metrics = env.metrics()

    assert "average_satisfaction" in metrics
    assert "resolution_rate" in metrics
    assert "sla_violations" in metrics
