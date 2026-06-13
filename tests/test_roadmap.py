"""Tests covering roadmap-level integrations."""

from twin_ai_gym.adapters import AdapterAgent, make_gymnasium_env
from twin_ai_gym.marketplace import list_environment_packages
from twin_ai_gym.renderers import render_graph_timeline, render_rl_training_report
from twin_ai_gym.rl import CustomerSupportVectorEncoder, EpisodeMetric, collect_trajectories
from twin_ai_gym.worlds import SalesWorld, business_suite
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def test_gymnasium_wrapper_accepts_discrete_actions() -> None:
    """The Gymnasium wrapper should map integer actions to action names."""

    wrapper = make_gymnasium_env(CustomerSupportWorld(seed=3))
    observation, info = wrapper.reset(seed=3)
    _, reward, _, _, step_info = wrapper.step(0)

    assert "entities" in observation
    assert info["step"] == 0
    assert isinstance(reward, float)
    assert step_info["action"] in wrapper.action_names


def test_vector_gymnasium_wrapper_returns_fixed_numeric_observation() -> None:
    """Vector encoders should make observations usable by classical RL."""

    encoder = CustomerSupportVectorEncoder()
    wrapper = make_gymnasium_env(CustomerSupportWorld(seed=3), observation_encoder=encoder)
    observation, info = wrapper.reset(seed=3)
    next_observation, _, _, _, step_info = wrapper.step(0)

    assert isinstance(observation, list)
    assert len(observation) == len(encoder.feature_names)
    assert len(next_observation) == len(encoder.feature_names)
    assert info["feature_names"] == list(encoder.feature_names)
    assert step_info["action_index"] == 0


def test_framework_adapter_extracts_action() -> None:
    """Framework adapters should normalize text outputs to actions."""

    agent = AdapterAgent(lambda _payload: "I choose escalate_ticket now.", ["reply_ticket", "escalate_ticket"])
    action = agent.act(CustomerSupportWorld(seed=3).observe())

    assert action == "escalate_ticket"


def test_business_world_and_suite_run() -> None:
    """Built-in business worlds should be benchmarkable."""

    env = SalesWorld(seed=5)

    def agent(_observation):
        return "run_discovery"

    result = env.evaluate(agent, seed=5)
    suite_result = business_suite(seed=5).evaluate(agent, seed=5)

    assert 0.0 <= result.score <= 1.0
    assert "sales" in suite_result.cases


def test_renderer_and_marketplace() -> None:
    """Renderer and package metadata should be available without optional deps."""

    env = CustomerSupportWorld(seed=4)
    env.step("reply_ticket")
    html = render_graph_timeline(env)
    packages = list_environment_packages()

    assert "TwinAIGym Replay" in html
    assert any(package.name == "twinaigym-sales" for package in packages)


def test_trajectory_collection_and_rl_report() -> None:
    """RL utilities should export vector trajectories and report HTML."""

    def agent(_observation):
        return "reply_ticket"

    transitions = collect_trajectories(
        env_factory=lambda: CustomerSupportWorld(seed=8),
        agent=agent,
        episodes=1,
        seed=8,
        encoder=CustomerSupportVectorEncoder(),
    )
    html = render_rl_training_report(
        [EpisodeMetric(0, 1.0, 2, 0.4, False, False, epsilon=1.0)],
        [EpisodeMetric(0, 2.0, 3, 0.6, True, False)],
    )

    assert transitions
    assert isinstance(transitions[0].observation, list)
    assert "TwinAIGym RL Training Report" in html
