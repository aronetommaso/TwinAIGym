"""Run a deterministic customer support episode."""

from twin_ai_gym.worlds.customer_support import CustomerSupportWorld


def main() -> None:
    """Execute a small scripted episode."""

    env = CustomerSupportWorld(seed=42)
    observation, info = env.reset()
    print(f"Initial entities: {len(observation.entities)}")
    print(f"Initial info: {info}")

    for action in ["reply_ticket", "ask_more_info", "reply_ticket", "refund_customer"]:
        _, reward, terminated, truncated, info = env.step(action)
        print(f"\nAction: {action}")
        print(f"Reward: {reward:.3f}")
        print("Diff:")
        for line in info["diff"].summary():
            print(f"  - {line}")
        if terminated or truncated:
            break

    print("\nMetrics:")
    for key, value in env.metrics().items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
