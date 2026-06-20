"""Run the controlled TwinAIGym representation study."""

from __future__ import annotations

import argparse

from twin_ai_gym.experiments import ExperimentConfig, RepresentationExperiment


def main() -> None:
    """Run the factorial experiment and write all artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-tasks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-tasks", type=int, default=20)
    args = parser.parse_args()
    config = ExperimentConfig(
        tasks=args.tasks,
        trials=args.trials,
        train_tasks=args.train_tasks,
        seed=args.seed,
        include_llm=not args.no_llm,
        llm_tasks=args.llm_tasks,
    )
    experiment = RepresentationExperiment(config)
    measurements, aggregates, skipped = experiment.run()
    report = experiment.save(measurements, aggregates, skipped)
    print(f"Completed {len(measurements)} episodes.")
    print(f"Report: {report}")
    if skipped:
        print(f"Skipped {len(skipped)} unconfigured LLM conditions.")


if __name__ == "__main__":
    main()
