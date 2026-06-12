"""Tests for the customer support MVP world."""

from twin_ai_gym.worlds.customer_support import CustomerSupportWorld, customer_support_suite


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


def test_agent_evaluation_returns_benchmark_result() -> None:
    """Evaluation should produce an assertable benchmark result."""

    env = CustomerSupportWorld.adversarial(seed=7)

    def agent(_observation):
        return "reply_ticket"

    result = env.evaluate(agent, seed=7)

    assert 0.0 <= result.score <= 1.0
    assert result.steps > 0
    assert "Score:" in result.report()
    assert "resolution_rate" in result.metrics


def test_customer_support_benchmark_suite_runs() -> None:
    """Benchmark suites should evaluate an agent across named cases."""

    def agent(_observation):
        return "reply_ticket"

    result = customer_support_suite(seed=7).evaluate(agent, seed=7)

    assert "standard" in result.cases
    assert "adversarial" in result.cases
    assert 0.0 <= result.score <= 1.0
