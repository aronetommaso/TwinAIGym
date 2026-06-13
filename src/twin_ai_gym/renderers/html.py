"""Dependency-free HTML graph and timeline renderer."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from twin_ai_gym.core.env import TwinEnv


def render_graph_timeline(env: TwinEnv, path: str | Path | None = None, title: str = "TwinAIGym Replay") -> str:
    """Render the current graph and recorded episode timeline as interactive HTML."""

    nodes = [
        {
            "id": entity.id,
            "type": entity.type,
            "attributes": entity.attributes,
        }
        for entity in env.world.entities.values()
    ]
    edges = [
        {
            "source": relation.source,
            "type": relation.type,
            "target": relation.target,
            "attributes": relation.attributes,
        }
        for relation in env.world.relations.values()
    ]
    steps = [
        {
            "index": index + 1,
            "action": step.action,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "diff": step.info.get("diff").summary() if step.info.get("diff") is not None else [],
        }
        for index, step in enumerate(env.episode.steps)
    ]
    html = _html(title=title, nodes=nodes, edges=edges, steps=steps)
    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


def _html(title: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], steps: list[dict[str, Any]]) -> str:
    """Build the HTML document."""

    node_cards = "\n".join(
        f"<button class='node' data-node='{escape(str(node))}'>{escape(node['id'])}<span>{escape(node['type'])}</span></button>"
        for node in nodes
    )
    edge_rows = "\n".join(
        f"<li>{escape(edge['source'])} <b>{escape(edge['type'])}</b> {escape(edge['target'])}</li>"
        for edge in edges
    )
    step_rows = "\n".join(
        "<button class='step' data-diff='{}'>#{:02d} {} <span>{:+.3f}</span></button>".format(
            escape("\n".join(step["diff"])),
            step["index"],
            escape(step["action"]),
            step["reward"],
        )
        for step in steps
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f7f7f4; color: #20231f; }}
    header {{ padding: 20px 28px; background: #24352f; color: white; }}
    main {{ display: grid; grid-template-columns: 1.4fr .9fr; gap: 18px; padding: 18px; }}
    section {{ background: white; border: 1px solid #d9ded7; border-radius: 8px; padding: 16px; }}
    h1 {{ font-size: 22px; margin: 0; }}
    h2 {{ font-size: 15px; margin: 0 0 12px; }}
    .graph {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .node, .step {{ text-align: left; border: 1px solid #cfd8d1; border-radius: 8px; background: #fbfcfa; padding: 10px; cursor: pointer; }}
    .node:hover, .step:hover {{ border-color: #34785f; }}
    .node span, .step span {{ display: block; color: #617069; font-size: 12px; margin-top: 4px; }}
    .timeline {{ display: grid; gap: 8px; max-height: 360px; overflow: auto; }}
    pre {{ white-space: pre-wrap; background: #17211d; color: #e7f1ea; padding: 12px; border-radius: 8px; min-height: 130px; }}
    li {{ margin-bottom: 6px; }}
    @media (max-width: 760px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>{escape(title)}</h1></header>
  <main>
    <section><h2>Graph Nodes</h2><div class="graph">{node_cards}</div></section>
    <section><h2>Timeline</h2><div class="timeline">{step_rows or "<p>No episode steps yet.</p>"}</div><pre id="details">Select a node or timeline step.</pre></section>
    <section><h2>Relations</h2><ul>{edge_rows}</ul></section>
  </main>
  <script>
    const details = document.getElementById('details');
    document.querySelectorAll('.node').forEach((button) => button.onclick = () => details.textContent = button.dataset.node);
    document.querySelectorAll('.step').forEach((button) => button.onclick = () => details.textContent = button.dataset.diff || 'No diff');
  </script>
</body>
</html>"""
