"""Tests covering roadmap-level integrations."""

from twin_ai_gym.adapters import AdapterAgent, make_gymnasium_env
from twin_ai_gym.marketplace import list_environment_packages
from twin_ai_gym.renderers import render_graph_timeline
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
