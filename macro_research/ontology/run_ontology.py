"""
run_ontology.py - Main orchestrator for the research ontology.

Steps:
  1. Build empty graph (schema)
  2. Populate with research data (populate)
  3. Run inference rules (inference)
  4. Print XAI reasoning chains
  5. Export JSON-LD knowledge graph
  6. Save signal summary CSV
  7. Plot ontology graph visualization

Usage:
  python -m ontology.run_ontology              # active regime only
  python -m ontology.run_ontology --all        # all rules regardless of regime
  python -m ontology.run_ontology --no-plot    # skip visualization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, FIGURES_DIR
from ontology.schema import (
    build_empty_graph,
    EdgeType,
    NodeType,
    SignalType,
    SECTOR_NAMES,
    MACRO_NAMES,
    CAUSAL_MACRO_NAMES,
)
from ontology.populate import populate
from ontology.inference import (
    generate_signals,
    signals_to_dataframe,
    print_signals,
)


# ---------------------------------------------------------------------------
# JSON-LD export
# ---------------------------------------------------------------------------

def _node_to_jsonld(node_id: str, attrs: dict) -> dict:
    obj: dict = {"@id": node_id}
    obj.update(attrs)
    return obj


def export_jsonld(G: nx.DiGraph, path: Path) -> None:
    """Export graph as JSON-LD document."""
    nodes = []
    for nid, attrs in G.nodes(data=True):
        nodes.append(_node_to_jsonld(nid, attrs))

    edges = []
    for src, dst, attrs in G.edges(data=True):
        edges.append({"@from": src, "@to": dst, **attrs})

    doc = {
        "@context": {
            "sector":      "schema:sector",
            "macro":       "schema:macro",
            "SENSITIVE_TO": "schema:sensitivity",
            "CO_CRASH_WITH": "schema:tailDependence",
            "GENERATES":    "schema:generates",
        },
        "@graph": {
            "nodes": nodes,
            "edges": edges,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved JSON-LD: {path}")


# ---------------------------------------------------------------------------
# Graph visualization
# ---------------------------------------------------------------------------

_SIG_COLORS = {
    SignalType.OVERWEIGHT.value:  "#2E7D32",   # dark green
    SignalType.UNDERWEIGHT.value: "#C62828",   # dark red
    SignalType.HEDGE.value:       "#E65100",   # orange
    SignalType.MONITOR.value:     "#1565C0",   # blue
}

_NODE_COLORS = {
    NodeType.SECTOR.value:       "#BBDEFB",
    NodeType.MACRO_FACTOR.value: "#FFE082",
    NodeType.REGIME.value:       "#C8E6C9",
    NodeType.SIGNAL.value:       "#F3E5F5",
}


def _build_viz_subgraph(G: nx.DiGraph, signals) -> nx.DiGraph:
    """Build a simplified graph for visualization (no signal detail nodes)."""
    H = nx.DiGraph()

    # Sector, macro, and causal-macro nodes
    for n in list(SECTOR_NAMES.keys()) + list(MACRO_NAMES.keys()) + list(CAUSAL_MACRO_NAMES.keys()):
        if n in G.nodes:
            H.add_node(n, **G.nodes[n])

    # SENSITIVE_TO edges (only where |delta| > 0.003)
    for sec in SECTOR_NAMES:
        for mac in MACRO_NAMES:
            data = G.get_edge_data(sec, mac)
            if data and abs(float(data.get("delta", 0) or 0)) > 0.003:
                H.add_edge(sec, mac, **data)

    # CO_CRASH_WITH edges (only strong ones, lambda > 0.5)
    for s1, s2, data in G.edges(data=True):
        if (data.get("edge_type") == EdgeType.CO_CRASH_WITH.value
                and float(data.get("lambda_lower", 0) or 0) > 0.50):
            H.add_edge(s1, s2, **data)

    # TRANSMITS_TO edges (structural + emerging, score > 0.90)
    for s1, s2, data in G.edges(data=True):
        if (data.get("edge_type") == EdgeType.TRANSMITS_TO.value
                and float(data.get("score", 0) or 0) > 0.90):
            H.add_edge(s1, s2, **data)

    return H


def plot_ontology_graph(G: nx.DiGraph, signals, out_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Signal summary bar chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: signal table heatmap by sector/rule
    if signals:
        sig_df = signals_to_dataframe(signals)[["sector", "signal", "rule", "confidence"]]
        pivot  = sig_df.pivot_table(
            index="sector", columns="signal", values="confidence",
            aggfunc="max", fill_value=0.0
        )
        sig_cols = [c for c in [SignalType.OVERWEIGHT.value, SignalType.UNDERWEIGHT.value,
                                 SignalType.HEDGE.value, SignalType.MONITOR.value]
                    if c in pivot.columns]
        pivot = pivot[sig_cols]

        cmap_colors = [_SIG_COLORS.get(c, "#BDBDBD") for c in sig_cols]
        bar_width = 0.8 / len(sig_cols) if sig_cols else 0.2
        x = np.arange(len(pivot.index))

        ax = axes[0]
        for i, (col, color) in enumerate(zip(sig_cols, cmap_colors)):
            vals = pivot[col].values
            bars = ax.bar(x + i * bar_width - bar_width * len(sig_cols) / 2,
                          vals, bar_width, label=col, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, fontsize=8)
        ax.set_ylabel("Confidence", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("Ontology Signals: Confidence by Sector & Type", fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(0.75, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.text(len(pivot.index) - 0.3, 0.76, "0.75", fontsize=7, alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, "No signals generated", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=12)

    # Right: simplified graph visualization
    ax2 = axes[1]
    H   = _build_viz_subgraph(G, signals)

    # Sector positions in a circle; macro in a row at bottom; causal macro below that
    sector_list       = list(SECTOR_NAMES.keys())
    macro_list        = list(MACRO_NAMES.keys())
    causal_macro_list = [k for k in CAUSAL_MACRO_NAMES.keys() if k in H.nodes]

    pos: dict = {}
    for i, s in enumerate(sector_list):
        angle = 2 * np.pi * i / len(sector_list)
        pos[s] = (np.cos(angle) * 2.5, np.sin(angle) * 2.5)
    for i, m in enumerate(macro_list):
        pos[m] = (-3 + i * 2.0, -3.8)
    for i, m in enumerate(causal_macro_list):
        pos[m] = (-3 + i * 2.0, -5.6)

    node_colors = []
    for n in H.nodes:
        nt = H.nodes[n].get("node_type", "")
        node_colors.append(_NODE_COLORS.get(nt, "#EEEEEE"))

    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=350,
                           alpha=0.90, ax=ax2)
    nx.draw_networkx_labels(H, pos, font_size=5.5, ax=ax2)

    # Edge colors by type
    sens_edges     = [(u, v) for u, v, d in H.edges(data=True)
                      if d.get("edge_type") == EdgeType.SENSITIVE_TO.value]
    crash_edges    = [(u, v) for u, v, d in H.edges(data=True)
                      if d.get("edge_type") == EdgeType.CO_CRASH_WITH.value]
    transmit_edges = [(u, v) for u, v, d in H.edges(data=True)
                      if d.get("edge_type") == EdgeType.TRANSMITS_TO.value]

    nx.draw_networkx_edges(H, pos, edgelist=sens_edges,
                           edge_color="#1565C0", alpha=0.5, arrows=True,
                           arrowsize=10, width=0.8, ax=ax2,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(H, pos, edgelist=crash_edges,
                           edge_color="#C62828", alpha=0.5, arrows=False,
                           width=1.4, style="dashed", ax=ax2)
    nx.draw_networkx_edges(H, pos, edgelist=transmit_edges,
                           edge_color="#2E7D32", alpha=0.75, arrows=True,
                           arrowsize=14, width=1.8, ax=ax2,
                           connectionstyle="arc3,rad=0.15")

    # Signal overlay: color sector node ring
    sig_lookup: dict[str, str] = {}
    for sig in signals:
        if sig.sector not in sig_lookup or sig.confidence > signals[0].confidence:
            sig_lookup[sig.sector] = sig.signal_type.value

    if sig_lookup:
        ring_nodes  = [s for s in sector_list if s in sig_lookup]
        ring_colors = [_SIG_COLORS.get(sig_lookup[s], "#BDBDBD") for s in ring_nodes]
        ring_pos    = {s: pos[s] for s in ring_nodes}
        nx.draw_networkx_nodes(H, ring_pos, nodelist=ring_nodes,
                               node_color=ring_colors, node_size=500,
                               alpha=0.35, ax=ax2)

    legend_handles = [
        mpatches.Patch(color=_NODE_COLORS[NodeType.SECTOR.value],       label="Sector"),
        mpatches.Patch(color=_NODE_COLORS[NodeType.MACRO_FACTOR.value], label="Macro Factor"),
        mpatches.Patch(color="#1565C0", label="SENSITIVE_TO (|delta|>0.003)", alpha=0.5),
        mpatches.Patch(color="#C62828", label="CO_CRASH_WITH (lambda>0.5)",   alpha=0.5),
        mpatches.Patch(color="#2E7D32", label="TRANSMITS_TO (PCMCI, score>0.90)", alpha=0.7),
    ]
    for stype, color in _SIG_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=f"Signal: {stype}", alpha=0.5))

    ax2.legend(handles=legend_handles, fontsize=6, loc="lower right")
    ax2.set_title("Ontology Graph (filtered -- strong links only)", fontsize=10)
    ax2.axis("off")

    plt.suptitle("Macro-Sector Research Ontology -- Object-Link-Action",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_all: bool = False, no_plot: bool = False) -> None:
    print("\n" + "=" * 66)
    print("  MACRO-SECTOR RESEARCH ONTOLOGY  (Object-Link-Action)")
    print("=" * 66)

    # 1. Build graph
    print("\n[1] Building empty ontology graph ...")
    G = build_empty_graph()

    # 2. Populate
    print("\n[2] Populating graph from research data ...")
    active_regime = populate(G)

    # 3. Inference
    print(f"\n[3] Running inference rules  (regime={active_regime}, all={run_all}) ...")
    signals = generate_signals(G, active_regime, run_all_regimes=run_all)
    print(f"    Generated {len(signals)} signals across {len(SECTOR_NAMES)} sectors")

    # 4. Print XAI reasoning
    print("\n[4] XAI Reasoning Chains")
    print_signals(signals)

    # 5. Export JSON-LD
    print("\n[5] Exporting JSON-LD ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_jsonld(G, OUTPUT_DIR / "ontology_graph.jsonld")

    # 6. Save signal CSV
    if signals:
        sig_df = signals_to_dataframe(signals)
        sig_df.to_csv(OUTPUT_DIR / "ontology_signals.csv")
        print(f"  Saved: ontology_signals.csv  ({len(sig_df)} rows)")

        # Summary table
        print("\n[6] Signal Summary Table")
        print("-" * 66)
        summary = (sig_df[["sector", "signal", "rule", "confidence"]]
                   .sort_values("confidence", ascending=False)
                   .to_string(index=False))
        print(summary)
    else:
        print("\n  No signals generated -- check data files in output/")

    # 7. Visualization
    if not no_plot:
        print("\n[7] Plotting ontology graph ...")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_ontology_graph(G, signals, FIGURES_DIR / "ontology_graph.png")

    print("\n" + "=" * 66)
    print("  Done.")
    print("=" * 66 + "\n")
    return G, signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro-sector research ontology")
    parser.add_argument("--all",     action="store_true", help="Run all rules (ignore regime filter)")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph visualization")
    args = parser.parse_args()

    main(run_all=args.all, no_plot=args.no_plot)
