from __future__ import annotations


import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


def load_mapping(mapping_csv: str) -> dict[str, str]:
    df = pd.read_csv(mapping_csv)
    if set(["k", "v"]).issubset(df.columns):
        return dict(zip(df["k"].astype(str), df["v"].astype(str)))
    if df.shape[1] >= 2:
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    raise ValueError("mapping csv must have columns ['k','v'] or at least two columns")


def to_markdown_taxonomy(taxonomy_df: pd.DataFrame) -> str:
    children = defaultdict(list)
    for h, y in taxonomy_df[["hypernym", "hyponym"]].drop_duplicates().itertuples(index=False):
        children[h].append(y)

    lines: list[str] = ["# Unified Taxonomy", ""]

    def emit(node: str, level: int = 2) -> None:
        if node not in children:
            return
        if level == 2:
            lines.append(f"## {node}")
        for child in sorted(set(children[node])):
            if level >= 3:
                lines.append("- " + child)
            else:
                lines.append(f"- {child}")
            emit(child, level + 1)
        if level == 2:
            lines.append("")

    if "Homepage" in children:
        emit("Homepage", 2)
    else:
        for root in sorted(set(taxonomy_df["hypernym"]) - set(taxonomy_df["hyponym"])):
            emit(root, 2)
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-refine", required=True)
    parser.add_argument("--menu-hier", required=True)
    parser.add_argument("--mapping", required=True, help="tag_normalize.csv")
    parser.add_argument("--taxonomy-template", default=None, help="existing final taxonomy file, e.g. test.xlsx")
    parser.add_argument("--output-dir", default="websiteanalysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)

    df_url = pd.read_excel(args.url_refine)
    df_menu = pd.read_excel(args.menu_hier)

    df_url["Name_clean_Norm"] = df_url["Name_clean"].fillna("").astype(str).map(lambda x: mapping.get(x, ""))
    df_url["Parent_clean_Norm"] = df_url["Parent_clean"].fillna("").astype(str).map(lambda x: mapping.get(x, ""))

    df_menu["Response_loc_hypernym_Norm"] = df_menu["Response_loc_hypernym"].fillna("").astype(str).map(lambda x: mapping.get(x, ""))
    df_menu["Response_hyponym_Norm"] = df_menu["Response_hyponym"].fillna("").astype(str).map(lambda x: mapping.get(x, ""))

    df_url.to_excel(output_dir / "dfurlhier113Refiner_normalize.xlsx", index=False)
    df_menu.to_excel(output_dir / "dfmenu113hier_normalize.xlsx", index=False)

    a = df_url[["domain", "Parent_clean_Norm", "Name_clean_Norm"]].copy()
    a.columns = ["sitedomain", "hypernym", "hyponym"]
    b = df_menu[["sitedomain", "Response_loc_hypernym_Norm", "Response_hyponym_Norm"]].copy()
    b.columns = ["sitedomain", "hypernym", "hyponym"]

    hierdata_refine1 = pd.concat([a, b], ignore_index=True)
    hierdata_refine1 = hierdata_refine1.replace("", pd.NA).dropna(subset=["hypernym", "hyponym"])
    hierdata_refine1 = hierdata_refine1.drop_duplicates().reset_index(drop=True)
    hierdata_refine1.to_excel(output_dir / "hierdata_refine1.xlsx", index=False)

    # Final unified taxonomy
    if args.taxonomy_template:
        taxonomy_df = pd.read_excel(args.taxonomy_template)
        taxonomy_df = taxonomy_df[["hypernym", "hyponym"]].dropna().drop_duplicates().reset_index(drop=True)
    else:
        # Fallback: derive a draft taxonomy from normalized hierarchy only.
        # This is not identical to the manually finalized taxonomy, but it is usable.
        taxonomy_df = hierdata_refine1[["hypernym", "hyponym"]].drop_duplicates().reset_index(drop=True)

    taxonomy_df.to_excel(output_dir / "test.xlsx", index=False)
    md = to_markdown_taxonomy(taxonomy_df)
    (output_dir / "unified_taxonomy.md").write_text(md, encoding="utf-8")

    print(f"saved: {output_dir / 'dfurlhier113Refiner_normalize.xlsx'}")
    print(f"saved: {output_dir / 'dfmenu113hier_normalize.xlsx'}")
    print(f"saved: {output_dir / 'hierdata_refine1.xlsx'}")
    print(f"saved: {output_dir / 'test.xlsx'}")
    print(f"saved: {output_dir / 'unified_taxonomy.md'}")
    print("NOTE: taxonomy generation here assumes you already reviewed/merged terms manually.")


if __name__ == "__main__":
    main()
