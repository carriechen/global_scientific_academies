from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.gridspec import GridSpec
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from scientometrics_plot_utils import (
    apply_style,
    DOUBLE_COL_MM,
    BITMAP_DPI,
    PRIMARY3,
    SCI_BLUE,
    SCI_TEAL,
    SCI_ORANGE,
    SCI_SEQ_CMAP,
    darken,
    lighten,
    mm_to_in,
)

MM_TO_INCH = 1 / 25.4
FIG_W_MM = DOUBLE_COL_MM
OUT_DPI = BITMAP_DPI


def find_missing_parents(pairs: set[tuple[str, str]], hypernym_map: dict[str, set[str]]) -> set[tuple[str, str]]:
    missing: set[tuple[str, str]] = set()
    for _, hyponym in pairs:
        current = hyponym
        visited: set[str] = set()
        while current in hypernym_map:
            if current in visited:
                break
            visited.add(current)
            parent = None
            for p in hypernym_map[current]:
                parent = p
                if (p, current) not in pairs:
                    missing.add((p, current))
            if parent is None:
                break
            current = parent
    return missing



def save_all(fig: plt.Figure, basepath: Path, raster_dpi: int = OUT_DPI) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(basepath.with_suffix(".png"), dpi=raster_dpi, **save_kwargs)
    fig.savefig(basepath.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(basepath.with_suffix(".eps"), **save_kwargs)
    fig.savefig(basepath.with_suffix(".tiff"), dpi=raster_dpi, pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs)



def format_val(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"



def relabel_clusters_left_to_right(linkage_matrix: np.ndarray, raw_clusters: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    leaves = sch.leaves_list(linkage_matrix)
    order: list[int] = []
    for idx in leaves:
        c = int(raw_clusters[idx])
        if c not in order:
            order.append(c)
    mapping = {raw: new for new, raw in enumerate(order, start=1)}
    relabeled = np.array([mapping[int(c)] for c in raw_clusters], dtype=int)
    return relabeled, mapping



def set_scientometrics_style() -> None:
    apply_style()
    mpl.rcParams.update({
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.transparent": False,
        "hatch.linewidth": 0.55,
    })
    sns.set_style(
        "whitegrid",
        {
            "axes.edgecolor": darken(SCI_BLUE, 0.30),
            "grid.color": "#E6ECF3",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        },
    )



def apply_box_hatches(ax: plt.Axes, hatches: list[str], edgecolor: str | None = None) -> None:
    edgecolor = edgecolor or darken(SCI_BLUE, 0.28)
    path_patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    for patch, hatch in zip(path_patches[: len(hatches)], hatches):
        patch.set_hatch(hatch)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(0.85)



def cluster_segments_from_leaves(linkage_matrix: np.ndarray, clusters: np.ndarray) -> list[tuple[int, int, int]]:
    leaves = sch.leaves_list(linkage_matrix)
    ordered_clusters = clusters[leaves]
    segments: list[tuple[int, int, int]] = []
    current = int(ordered_clusters[0])
    start = 0
    for i, c in enumerate(ordered_clusters):
        c = int(c)
        if c != current:
            segments.append((current, start, i - 1))
            current = c
            start = i
    segments.append((current, start, len(ordered_clusters) - 1))
    return segments



def add_cluster_strip(ax: plt.Axes, linkage_matrix: np.ndarray, clusters: np.ndarray, fill_map: dict[int, str], hatch_map: dict[int, str]) -> None:
    segments = cluster_segments_from_leaves(linkage_matrix, clusters)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    strip_y = -0.145
    strip_h = 0.065
    num_y = strip_y + strip_h / 2

    for cluster_id, start_idx, end_idx in segments:
        x0 = 10 * start_idx
        width = 10 * (end_idx - start_idx + 1)
        rect = Rectangle(
            (x0, strip_y),
            width,
            strip_h,
            transform=trans,
            clip_on=False,
            facecolor=fill_map[cluster_id],
            edgecolor=darken(fill_map[cluster_id], 0.28),
            linewidth=0.7,
            hatch=hatch_map[cluster_id],
        )
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            num_y,
            str(cluster_id),
            transform=trans,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=darken(SCI_BLUE, 0.32),
            clip_on=False,
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-hierarchy", default="Figure6.xlsx")
    parser.add_argument("--taxonomy", default="Figure5.xlsx")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--num-clusters", type=int, default=3)
    args = parser.parse_args()

    set_scientometrics_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    site_df = pd.read_excel(args.site_hierarchy)
    taxonomy_df = pd.read_excel(args.taxonomy)[["hypernym", "hyponym"]].dropna().drop_duplicates().reset_index(drop=True)

    hypernym_map: dict[str, set[str]] = defaultdict(set)
    for h, y in zip(taxonomy_df["hypernym"], taxonomy_df["hyponym"]):
        hypernym_map[str(y)].add(str(h))

    common_taxonomy_set = set(zip(taxonomy_df["hypernym"].astype(str), taxonomy_df["hyponym"].astype(str)))
    site_dict = (
        site_df.groupby("sitedomain")[["hypernym", "hyponym"]]
        .apply(lambda x: set(zip(x["hypernym"].astype(str), x["hyponym"].astype(str))))
        .to_dict()
    )

    enhanced_site_dict: dict[str, set[tuple[str, str]]] = {}
    similarity_results: list[dict[str, float | str | int]] = []
    for site, pairs in site_dict.items():
        missing_pairs = find_missing_parents(pairs, hypernym_map)
        updated_pairs = pairs | missing_pairs
        enhanced_site_dict[site] = updated_pairs
        new_intersection = updated_pairs & common_taxonomy_set
        precision = len(new_intersection) / len(updated_pairs) if updated_pairs else 0.0
        recall = len(new_intersection) / len(common_taxonomy_set) if common_taxonomy_set else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        similarity_results.append(
            {
                "sitedomain": site,
                "precision": precision,
                "recall": recall,
                "similarity (F1-score)": f1,
                "completeness": recall,
                "missing_pairs": len(missing_pairs),
            }
        )
    similarity_df = pd.DataFrame(similarity_results)

    sites = list(enhanced_site_dict.keys())
    all_pairs = sorted(set(pair for pairs in enhanced_site_dict.values() for pair in pairs))
    binary_vectors = np.array([[1 if pair in enhanced_site_dict[site] else 0 for pair in all_pairs] for site in sites], dtype=bool)

    jaccard_sim_matrix = 1 - pairwise_distances(binary_vectors, metric="jaccard")
    cosine_sim_matrix = cosine_similarity(csr_matrix(binary_vectors.astype(int)))
    linkage_matrix = sch.linkage(1 - jaccard_sim_matrix, method="ward")
    raw_clusters = fcluster(linkage_matrix, args.num_clusters, criterion="maxclust")
    clusters, _ = relabel_clusters_left_to_right(linkage_matrix, raw_clusters)

    imitation_scores = MinMaxScaler().fit_transform(np.mean(cosine_sim_matrix, axis=1)[:, None]).ravel()
    site_sets = [enhanced_site_dict[site] for site in sites]
    innovation_raw = np.zeros(len(sites))
    for i in range(len(sites)):
        innovation_raw[i] = sum(len(site_sets[i] - site_sets[j]) for j in range(len(sites)) if i != j) / max(len(sites) - 1, 1)
    innovation_scores = MinMaxScaler().fit_transform(innovation_raw[:, None]).ravel()

    analysis_df = pd.DataFrame(
        {
            "sitedomain": sites,
            "imitation_score": imitation_scores,
            "innovation_score": innovation_scores,
            "cluster": clusters,
        }
    ).merge(similarity_df[["sitedomain", "similarity (F1-score)"]], on="sitedomain", how="left")

    cluster_palette = {1: SCI_BLUE, 2: SCI_TEAL, 3: SCI_ORANGE}
    dendrogram_palette = [cluster_palette[1], cluster_palette[2], cluster_palette[3]]
    hatch_map = {1: "///", 2: "...", 3: "xx"}
    above_threshold = darken(SCI_BLUE, 0.35)
    color_threshold = linkage_matrix[-(args.num_clusters - 1), 2] - 1e-12 if args.num_clusters > 1 else 0

    # Figure 2
    fig = plt.figure(figsize=(FIG_W_MM * MM_TO_INCH, 150 * MM_TO_INCH), dpi=OUT_DPI)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.17, 1.0], hspace=0.26, wspace=0.18)
    ax0 = fig.add_subplot(gs[0, :])

    sch.set_link_color_palette(dendrogram_palette)
    try:
        sch.dendrogram(linkage_matrix, ax=ax0, no_labels=True, color_threshold=color_threshold, above_threshold_color=above_threshold)
    finally:
        sch.set_link_color_palette(None)

    ax0.set_title("Hierarchical clustering of sites based on similarity", pad=4)
    ax0.set_ylabel("Ward linkage distance")
    ax0.set_xlabel("")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.tick_params(axis="x", length=0, labelbottom=False)
    ax0.tick_params(axis="y", length=3, width=0.6)
    ax0.grid(False)
    ax0.text(-0.08, 1.02, "(a)", transform=ax0.transAxes, fontsize=9, fontweight="bold")
    add_cluster_strip(ax0, linkage_matrix, clusters, cluster_palette, hatch_map)

    boxprops = dict(linewidth=0.85, edgecolor=darken(SCI_BLUE, 0.28))
    medianprops = dict(color=darken(SCI_BLUE, 0.38), linewidth=1.2)
    whiskerprops = dict(linewidth=0.8, color=darken(SCI_BLUE, 0.24))
    capprops = dict(linewidth=0.8, color=darken(SCI_BLUE, 0.24))
    flierprops = dict(
        marker="D", markersize=3.1, markerfacecolor=darken(SCI_BLUE, 0.18), markeredgecolor=darken(SCI_BLUE, 0.18), alpha=1
    )

    order = ["1", "2", "3"]
    palette_list = [cluster_palette[1], cluster_palette[2], cluster_palette[3]]
    hatch_list = [hatch_map[1], hatch_map[2], hatch_map[3]]

    ax1 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        data=analysis_df,
        x=analysis_df["cluster"].astype(str),
        y="imitation_score",
        order=order,
        palette=palette_list,
        ax=ax1,
        width=0.78,
        linewidth=0.8,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
        saturation=0.95,
    )
    apply_box_hatches(ax1, hatch_list)
    ax1.set_title("Imitation score across clusters", pad=4)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Imitation score")
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Cluster 1", "Cluster 2", "Cluster 3"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(length=3, width=0.6)
    ax1.text(-0.13, 1.02, "(b)", transform=ax1.transAxes, fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(gs[1, 1])
    sns.boxplot(
        data=analysis_df,
        x=analysis_df["cluster"].astype(str),
        y="innovation_score",
        order=order,
        palette=palette_list,
        ax=ax2,
        width=0.78,
        linewidth=0.8,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
        saturation=0.95,
    )
    apply_box_hatches(ax2, hatch_list)
    ax2.set_title("Innovation score across clusters", pad=4)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Innovation score")
    ax2.set_ylim(0, 1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Cluster 1", "Cluster 2", "Cluster 3"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(length=3, width=0.6)
    ax2.text(-0.13, 1.02, "(c)", transform=ax2.transAxes, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.965)
    save_all(fig, outdir / "Figure6")
    plt.close(fig)

    # Figure 3
    first_level_categories = taxonomy_df[taxonomy_df["hypernym"] == "Homepage"]["hyponym"].dropna().unique().tolist()
    coverage_matrix = pd.DataFrame(0, index=sites, columns=first_level_categories, dtype=float)
    for site_name, pairs in enhanced_site_dict.items():
        site_hyponyms = {pair[1] for pair in pairs if pair[0] == "Homepage"}
        for category in first_level_categories:
            coverage_matrix.loc[site_name, category] = 1.0 if category in site_hyponyms else 0.0

    ordered_sites = [sites[i] for i in sch.leaves_list(linkage_matrix)]
    coverage_matrix_ordered = coverage_matrix.loc[ordered_sites].copy()
    coverage_matrix_ordered["cluster"] = [clusters[sites.index(site_name)] for site_name in ordered_sites]
    cluster_coverage = coverage_matrix_ordered.groupby("cluster").mean().reindex([1, 2, 3])

    annot = cluster_coverage.map(format_val) if hasattr(pd.DataFrame, "map") else cluster_coverage.applymap(format_val)

    fig2 = plt.figure(figsize=(165 * MM_TO_INCH, 112 * MM_TO_INCH), dpi=OUT_DPI)
    ax = fig2.add_subplot(111)
    hm = sns.heatmap(
        cluster_coverage,
        cmap=SCI_SEQ_CMAP,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        cbar_kws={"fraction": 0.05, "pad": 0.06, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        ax=ax,
    )
    # ax.set_title("Comparison of first-level category coverage across clusters", pad=5, fontsize=10)
    ax.set_xlabel("First-level categories")
    ax.set_ylabel("Cluster ID")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in cluster_coverage.index], rotation=0)
    ax.tick_params(length=3, width=0.6)

    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            val = 0
        text.set_color("white" if val >= 0.52 else darken(SCI_BLUE, 0.35))
        text.set_fontsize(7.8)

    cbar = hm.collections[0].colorbar
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8.0, length=2.5, width=0.6)
    cbar.set_label("Coverage", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig2.subplots_adjust(left=0.09, right=0.93, bottom=0.30, top=0.90)
    save_all(fig2, outdir / "Figure7")
    plt.close(fig2)


if __name__ == "__main__":
    main()
