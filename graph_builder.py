"""
graph_builder.py — ownership network PNG via NetworkX + matplotlib.

Same approach as FriendShore, deliberately: no graphviz system dependency,
just pip-installable libraries, so it works identically on Render.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend — no display on a server
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from risk_engine import jurisdiction_category  # noqa: E402

_COLORS = {
    "focal": "#4ea1ff",
    "adversary": "#ff5050",
    "secrecy": "#ffb24e",
    "normal": "#8a93a3",
}
_BG = "#10161d"


def build_graph_png(
    entities: list[dict],
    links: list[dict],
    focal_name: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    g = nx.DiGraph()
    colors_by_name: dict[str, str] = {}
    for e in entities:
        name = e["canonical_name"]
        if name == focal_name:
            color = _COLORS["focal"]
        else:
            cat = jurisdiction_category(e.get("jurisdiction"))
            color = _COLORS.get(cat or "normal", _COLORS["normal"])
        g.add_node(name)
        colors_by_name[name] = color

    edge_labels: dict[tuple[str, str], str] = {}
    for link in links:
        g.add_edge(link["owner"], link["owned"])
        pct = link.get("ownership_pct")
        edge_labels[(link["owner"], link["owned"])] = (
            f"{pct:g}%" if pct is not None else "?%"
        )
        for endpoint in (link["owner"], link["owned"]):
            colors_by_name.setdefault(endpoint, _COLORS["normal"])

    if not g.nodes:
        return

    node_colors = [colors_by_name[n] for n in g.nodes]
    pos = nx.spring_layout(g, seed=42, k=1.6)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#5a6675", arrows=True,
                           arrowsize=16, width=1.4, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=1700,
                           edgecolors="#222b35", linewidths=1.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="#e8edf2",
                            font_weight="bold")
    nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels,
                                 font_size=7, font_color="#c9d2dc",
                                 bbox={"facecolor": _BG, "edgecolor": "none"})
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
