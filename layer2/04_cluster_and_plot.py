from __future__ import annotations


import argparse
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def find_missing_parents(pairs: set[tuple[str, str]], hypernym_map: dict[str, set[str]]) -> set[tuple[str, str]]:
    missing: set[tuple[str, str]] = set()
    visited: set[str] = set()
    for _, hyponym in pairs:
        current = hyponym
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-hierarchy", required=True, help="hierdata_refine1.xlsx")
    parser.add_argument("--taxonomy", required=True, help="test.xlsx")
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--output-dir", default="websiteanalysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_df = pd.read_excel(args.site_hierarchy)
    taxonomy_df = pd.read_excel(args.taxonomy)
    taxonomy_df = taxonomy_df[["hypernym", "hyponym"]].dropna().drop_duplicates().reset_index(drop=True)

    hypernym_map: dict[str, set[str]] = defaultdict(set)
    for h, y in zip(taxonomy_df["hypernym"], taxonomy_df["hyponym"]):
        hypernym_map[str(y)].add(str(h))

    common_taxonomy_set = set(zip(taxonomy_df["hypernym"].astype(str), taxonomy_df["hyponym"].astype(str)))
    site_dict = site_df.groupby("sitedomain").apply(
        lambda x: set(zip(x["hypernym"].astype(str), x["hyponym"].astype(str)))
    ).to_dict()

    similarity_results = []
    missing_pairs_dict = {}
    enhanced_site_dict = {}

    for site, pairs in site_dict.items():
        missing_pairs = find_missing_parents(pairs, hypernym_map)
        updated_pairs = pairs | missing_pairs
        enhanced_site_dict[site] = updated_pairs
        new_intersection = updated_pairs & common_taxonomy_set
        precision = len(new_intersection) / len(updated_pairs) if updated_pairs else 0.0
        recall = len(new_intersection) / len(common_taxonomy_set) if common_taxonomy_set else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        similarity_results.append({
            "sitedomain": site,
            "precision": precision,
            "recall": recall,
            "similarity (F1-score)": f1,
            "completeness": recall,
            "missing_pairs": len(missing_pairs),
        })
        missing_pairs_dict[site] = list(missing_pairs)

    similarity_df = pd.DataFrame(similarity_results)
    similarity_df.to_excel(output_dir / "site_similarity_metrics.xlsx", index=False)

    all_pairs = sorted(set(pair for pairs in enhanced_site_dict.values() for pair in pairs))
    sites = list(enhanced_site_dict.keys())

    binary_vectors = np.array([
        [1 if pair in enhanced_site_dict[site] else 0 for pair in all_pairs]
        for site in sites
    ])
    jaccard_sim_matrix = 1 - pairwise_distances(binary_vectors, metric="jaccard")
    jaccard_sim_df = pd.DataFrame(jaccard_sim_matrix, index=sites, columns=sites)

    site_matrix = csr_matrix(binary_vectors)
    cosine_sim_matrix = cosine_similarity(site_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=sites, columns=sites)

    linkage_matrix = sch.linkage(1 - jaccard_sim_matrix, method="ward")
    clusters = fcluster(linkage_matrix, args.num_clusters, criterion="maxclust")

    cluster_df = pd.DataFrame({"sitedomain": sites, "cluster": clusters})
    cluster_avg_similarity = cluster_df.groupby("cluster").apply(
        lambda x: cosine_sim_df.loc[x["sitedomain"], x["sitedomain"]].values.mean()
    ).reset_index(name="avg_similarity")
    sorted_clusters = cluster_df.merge(cluster_avg_similarity, on="cluster").sort_values(
        by="avg_similarity", ascending=False
    )
    sorted_clusters.to_csv(output_dir / "sorted_clusters.csv", index=False)

    # Distinctive pairs by cluster
    cluster_pairs = defaultdict(list)
    for site, cluster in zip(sorted_clusters["sitedomain"], sorted_clusters["cluster"]):
        cluster_pairs[cluster].extend(enhanced_site_dict[site])
    cluster_pair_freq = {c: Counter(pairs) for c, pairs in cluster_pairs.items()}
    cluster_pair_df = pd.DataFrame([
        {"cluster": c, "hypernym": pair[0], "hyponym": pair[1], "count": count}
        for c, pair_counts in cluster_pair_freq.items() for pair, count in pair_counts.items()
    ])
    global_pair_freq = Counter(chain.from_iterable(enhanced_site_dict.values()))
    global_pair_df = pd.DataFrame([
        {"hypernym": pair[0], "hyponym": pair[1], "global_count": count}
        for pair, count in global_pair_freq.items()
    ])
    cluster_pair_df = cluster_pair_df.merge(global_pair_df, on=["hypernym", "hyponym"], how="left").fillna(0)
    cluster_sizes = cluster_pair_df.groupby("cluster")["count"].sum().to_dict()
    cluster_pair_df["relative_frequency"] = cluster_pair_df.apply(
        lambda row: row["count"] / max(cluster_sizes[row["cluster"]], 1), axis=1
    )
    total_global_count = sum(global_pair_freq.values())
    cluster_pair_df["distinctiveness"] = np.where(
        cluster_pair_df["global_count"] > 0,
        cluster_pair_df["relative_frequency"] / (cluster_pair_df["global_count"] / total_global_count),
        0,
    )
    most_distinctive_pairs = cluster_pair_df.groupby("cluster").apply(
        lambda x: x.nlargest(5, "distinctiveness")
    ).reset_index(drop=True)
    least_distinctive_pairs = cluster_pair_df.groupby("cluster").apply(
        lambda x: x.nsmallest(5, "distinctiveness")
    ).reset_index(drop=True)
    most_distinctive_pairs.to_excel(output_dir / "most_distinctive_pairs.xlsx", index=False)
    least_distinctive_pairs.to_excel(output_dir / "least_distinctive_pairs.xlsx", index=False)

    # Fig. 2: dendrogram + imitation/innovation boxplots
    imitation_scores = np.mean(cosine_sim_matrix, axis=1)
    site_sets = [enhanced_site_dict[site] for site in sites]
    innovation_scores = np.zeros(len(sites))
    for i in range(len(sites)):
        unique_count = sum(len(site_sets[i] - site_sets[j]) for j in range(len(sites)) if i != j)
        innovation_scores[i] = unique_count / max(len(sites) - 1, 1)

    scaler = MinMaxScaler()
    imitation_scores = scaler.fit_transform(imitation_scores[:, None]).ravel()
    innovation_scores = scaler.fit_transform(innovation_scores[:, None]).ravel()

    analysis_df = pd.DataFrame({
        "sitedomain": sites,
        "imitation_score": imitation_scores,
        "innovation_score": innovation_scores,
        "cluster": clusters,
    })
    analysis_df = analysis_df.merge(similarity_df[["sitedomain", "similarity (F1-score)"]], on="sitedomain", how="left")
    analysis_df.to_excel(output_dir / "cluster_site_scores.xlsx", index=False)

    cluster_palette = {1: "orange", 2: "green", 3: "red"}
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    ax0 = fig.add_subplot(gs[0, :])
    sch.dendrogram(linkage_matrix, ax=ax0, no_labels=True, color_threshold=None)
    ax0.set_title("Hierarchical Clustering of Sites Based on Similarity")
    ax0.set_ylabel("Dissimilarity (1 - Jaccard Similarity)")
    ax0.set_xlabel("")

    ax1 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=analysis_df["cluster"].astype(str), y=analysis_df["imitation_score"], palette=cluster_palette, ax=ax1)
    ax1.set_title("Imitation Score  Across Clusters")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Imitation Score")
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[1, 1])
    sns.boxplot(x=analysis_df["cluster"].astype(str), y=analysis_df["innovation_score"], palette=cluster_palette, ax=ax2)
    ax2.set_title("Innovation Score  Across Clusters")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Innovation Score")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "Figure2_cluster_imitation_innovation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Fig. 3: first-level category coverage heatmap
    first_level_categories = taxonomy_df[taxonomy_df["hypernym"] == "Homepage"]["hyponym"].dropna().unique()
    coverage_matrix = pd.DataFrame(0, index=sites, columns=first_level_categories)
    for site, pairs in enhanced_site_dict.items():
        site_hyponyms = {pair[1] for pair in pairs if pair[0] == "Homepage"}
        for category in first_level_categories:
            if category in site_hyponyms:
                coverage_matrix.loc[site, category] = 1

    ordered_sites = [sites[i] for i in sch.leaves_list(linkage_matrix)]
    coverage_matrix_ordered = coverage_matrix.loc[ordered_sites].copy()
    coverage_matrix_ordered["cluster"] = [clusters[sites.index(site)] for site in ordered_sites]
    cluster_coverage = coverage_matrix_ordered.groupby("cluster").mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_coverage, cmap="Blues", annot=True, linewidths=0.5, linecolor="gray", cbar=True)
    plt.title("Comparison of First-Level Category Coverage Across Clusters")
    plt.xlabel("First-Level Categories")
    plt.ylabel("Cluster ID")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure3_first_level_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {output_dir / 'Figure2_cluster_imitation_innovation.png'} [Fig. 2]")
    print(f"saved: {output_dir / 'Figure3_first_level_heatmap.png'} [Fig. 3]")


if __name__ == "__main__":
    main()
