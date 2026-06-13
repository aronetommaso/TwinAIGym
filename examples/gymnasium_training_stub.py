"""Train a tiny tabular policy through the Gymnasium-compatible adapter.

This example does not require Gymnasium to be installed because the wrapper also
accepts raw string actions. With Gymnasium installed, ``action_space`` is a
standard ``spaces.Discrete`` and this can be swapped into RLlib/SB3 loops.
"""

from twin_ai_gym.adapters import make_gymnasium_env
from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def main() -> None:
    """Run a minimal policy-search loop."""

    gym_env = make_gymnasium_env(CustomerSupportWorld.adversarial(seed=7))
    candidates = ["reply_ticket", "ask_more_info", "escalate_ticket", "refund_customer"]
    scores: dict[str, float] = {}
    for action in candidates:
        total = 0.0
        for seed in range(5):
            _, _ = gym_env.reset(seed=seed)
            done = False
            while not done:
                _, reward, terminated, truncated, _ = gym_env.step(action)
                total += reward
                done = terminated or truncated
        scores[action] = total

    best_action = max(scores, key=scores.get)
    print(f"Best constant action: {best_action}")
    print(scores)


if __name__ == "__main__":
    main()
