"""HTML visualizations for RL training runs."""

from __future__ import annotations

from html import escape
from pathlib import Path

from twin_ai_gym.rl.metrics import EpisodeMetric, summarize_episode_metrics


def render_rl_training_report(
    training_metrics: list[EpisodeMetric],
    evaluation_metrics: list[EpisodeMetric],
    path: str | Path | None = None,
    title: str = "TwinAIGym RL Training Report",
) -> str:
    """Render a dependency-free RL training report."""

    train_summary = summarize_episode_metrics(training_metrics)
    eval_summary = summarize_episode_metrics(evaluation_metrics)
    html = _html(title, training_metrics, evaluation_metrics, train_summary, eval_summary)
    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


def _html(title, training_metrics, evaluation_metrics, train_summary, eval_summary) -> str:
    """Build the report document."""

    train_points = _polyline(training_metrics, "score")
    reward_points = _polyline(training_metrics, "total_reward")
    eval_rows = "\n".join(
        f"<tr><td>{metric.episode}</td><td>{metric.score:.3f}</td><td>{metric.total_reward:.3f}</td><td>{metric.steps}</td></tr>"
        for metric in evaluation_metrics
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f6f7f4; color: #1e2420; }}
    header {{ background: #19352d; color: white; padding: 22px 28px; }}
    main {{ padding: 18px; display: grid; gap: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card, section {{ background: white; border: 1px solid #d9ded7; border-radius: 8px; padding: 16px; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .label {{ color: #5f6f67; font-size: 12px; text-transform: uppercase; }}
    svg {{ width: 100%; height: 260px; background: #fbfcfa; border: 1px solid #d9ded7; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #e5e8e3; padding: 8px; text-align: left; }}
    polyline {{ fill: none; stroke-width: 3; }}
  </style>
</head>
<body>
  <header><h1>{escape(title)}</h1></header>
  <main>
    <div class="grid">
      <div class="card"><div class="label">Train avg score</div><div class="metric">{train_summary.average_score:.2%}</div></div>
      <div class="card"><div class="label">Eval avg score</div><div class="metric">{eval_summary.average_score:.2%}</div></div>
      <div class="card"><div class="label">Eval success rate</div><div class="metric">{eval_summary.success_rate:.2%}</div></div>
      <div class="card"><div class="label">Best eval score</div><div class="metric">{eval_summary.best_score:.2%}</div></div>
    </div>
    <section>
      <h2>Training Score</h2>
      <svg viewBox="0 0 1000 260"><polyline points="{train_points}" stroke="#246b54"></polyline></svg>
    </section>
    <section>
      <h2>Training Reward</h2>
      <svg viewBox="0 0 1000 260"><polyline points="{reward_points}" stroke="#8a5a1d"></polyline></svg>
    </section>
    <section>
      <h2>Evaluation Episodes</h2>
      <table><thead><tr><th>Episode</th><th>Score</th><th>Reward</th><th>Steps</th></tr></thead><tbody>{eval_rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""


def _polyline(metrics: list[EpisodeMetric], attr: str) -> str:
    """Convert metric history to SVG points."""

    if not metrics:
        return ""
    values = [float(getattr(metric, attr)) for metric in metrics]
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    last = max(1, len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = 20 + 960 * index / last
        y = 240 - 220 * ((value - low) / span)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)
