#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5. Annotated domain-size summary of the web presence taxonomy.

Input:
    Figure5.xlsx
    Required columns: hypernym, hyponym

Outputs:
    Figure5_final.png
    Figure5_final.pdf
    Figure5_final.svg
    Figure5_final.eps
    Figure5_final.tiff
    Figure5_final_source.csv

Notes:
    - This is a single-panel figure, so no "(a)" panel label is used.
    - Style is aligned with Figure 4: blue bars, orange count labels,
      teal example annotations, light blue-gray grid, clean axes.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ----------------------------
# Figure style helpers
# ----------------------------

DOUBLE_COL_MM = 183
BITMAP_DPI = 600
FIG_W_MM = DOUBLE_COL_MM
FIG_H_MM = 125

SCI_BLUE = "#4E79A7"
SCI_ORANGE = "#E07B39"
SCI_TEAL = "#59AFC2"
GRID_COLOR = "#E6ECF3"
TEXT_COLOR = "#222222"


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def lighten(color: str, amount: float = 0.3):
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)


def darken(color: str, amount: float = 0.2):
    c = mcolors.to_rgb(color)
    return tuple(x * (1 - amount) for x in c)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ----------------------------
# Data loading and taxonomy graph
# ----------------------------

def load_taxonomy(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[["hypernym", "hyponym"]].dropna().drop_duplicates()
    df["hypernym"] = df["hypernym"].astype(str).str.strip()
    df["hyponym"] = df["hyponym"].astype(str).str.strip()
    return df


def build_children(df: pd.DataFrame):
    children = defaultdict(list)
    edge_set = set()

    for parent, child in df[["hypernym", "hyponym"]].itertuples(index=False):
        if parent and child:
            if child not in children[parent]:
                children[parent].append(child)
            edge_set.add((parent, child))

    for parent in children:
        children[parent] = sorted(children[parent], key=str.casefold)

    return children, edge_set


def infer_root(df: pd.DataFrame, children) -> str:
    if "Homepage" in children:
        return "Homepage"

    all_parents = set(df["hypernym"])
    all_children = set(df["hyponym"])
    roots = sorted(all_parents - all_children)
    if not roots:
        raise ValueError("Could not infer taxonomy root.")
    return roots[0]


def subtree_nodes(children, start: str) -> set[str]:
    seen = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)

        for child in children.get(node, []):
            queue.append(child)

    return seen


def subtree_edge_count(children, edge_set, start: str) -> int:
    nodes = subtree_nodes(children, start)
    return sum(1 for parent, child in edge_set if parent in nodes and child in nodes)


def representative_concepts(children, start: str, max_items: int = 4) -> list[str]:
    reps = []

    # Direct children first.
    for child in children.get(start, []):
        if child not in reps:
            reps.append(child)
        if len(reps) >= max_items:
            return reps

    # Then one layer deeper.
    for child in children.get(start, []):
        for grandchild in children.get(child, []):
            if grandchild not in reps:
                reps.append(grandchild)
            if len(reps) >= max_items:
                return reps

    return reps


def summarize_domains(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    children, edge_set = build_children(df)
    root = infer_root(df, children)

    rows = []
    for domain in children[root]:
        rows.append(
            {
                "domain": domain,
                "n_relations": subtree_edge_count(children, edge_set, domain),
                "representative_concepts": "; ".join(
                    representative_concepts(children, domain, max_items=4)
                ),
            }
        )

    domain_df = (
        pd.DataFrame(rows)
        .sort_values("n_relations", ascending=True)
        .reset_index(drop=True)
    )

    unique_relations = len(df)
    n_domains = len(children[root])
    within_domain_sum = int(domain_df["n_relations"].sum())

    return domain_df, unique_relations, n_domains, within_domain_sum


# ----------------------------
# Plotting
# ----------------------------

def plot_figure5(domain_df: pd.DataFrame, unique_relations: int, output_stem: Path) -> None:
    bar_color = lighten(SCI_BLUE, 0.28)
    bar_edge = darken(SCI_BLUE, 0.16)
    annotation_color = darken(SCI_TEAL, 0.18)

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=BITMAP_DPI)
    ax = fig.add_subplot(111)

    ax.barh(
        domain_df["domain"],
        domain_df["n_relations"],
        color=bar_color,
        edgecolor=bar_edge,
        linewidth=0.45,
        label="Taxonomy relations",
    )

    ax.set_xlabel("Number of taxonomy relations")
    ax.set_ylabel("Functional domain")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = domain_df["n_relations"].max()
    ax.set_xlim(0, max_val * 2.35)

    for i, row in domain_df.iterrows():
        value = row["n_relations"]

        ax.text(
            value + max_val * 0.035,
            i,
            f"n = {value}",
            va="center",
            ha="left",
            fontsize=8,
            color=SCI_ORANGE,
        )

        examples = textwrap.fill(
            f"Examples: {row['representative_concepts']}",
            width=42,
        )
        ax.text(
            value + max_val * 0.22,
            i,
            examples,
            va="center",
            ha="left",
            fontsize=7.5,
            color=annotation_color,
        )

    # Single-panel figure: no "(a)" label.
    # No in-figure title/caption/footnote; these should be handled in manuscript text.
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)

    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.16, top=0.94)

    fig.savefig(output_stem.with_suffix(".png"), dpi=BITMAP_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".eps"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=BITMAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    taxonomy_path = Path("Figure5.xlsx")
    output_stem = Path("Figure5_final")

    domain_df, unique_relations, n_domains, within_domain_sum = summarize_domains(
        load_taxonomy(taxonomy_path)
    )

    # Save source data in descending order for readability.
    domain_df.sort_values("n_relations", ascending=False).to_csv(
        output_stem.with_name(output_stem.name + "_source.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_figure5(domain_df, unique_relations, output_stem)

    print("Created Figure 5 files:")
    for suffix in [".png", ".pdf", ".svg", ".eps", ".tiff"]:
        print(output_stem.with_suffix(suffix))
    print(output_stem.with_name(output_stem.name + "_source.csv"))
    print()
    print(f"Unique relations: {unique_relations}")
    print(f"First-level domains: {n_domains}")
    print(f"Within-domain relation sum: {within_domain_sum}")


if __name__ == "__main__":
    main()
