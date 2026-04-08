from __future__ import annotations


import argparse
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd

EXCLUDE_EXTENSIONS = {
    ".js", ".css", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot",
}


def split_url_segments(url: str) -> list[str]:
    parsed = urlparse(url)
    path = unquote(parsed.path or "").strip("/")
    if not path:
        return []
    segments = [seg.strip() for seg in path.split("/") if seg.strip()]
    cleaned = []
    for seg in segments:
        seg = re.sub(r"(\?.*|&.*|#.*)", "", seg)
        lower = seg.lower()
        if any(lower.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
            continue
        cleaned.append(seg)
    return cleaned


def build_url_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in df.groupby("sitedomain"):
        for url in group["loc"].dropna().unique():
            segments = split_url_segments(url)
            if not segments:
                continue
            parent = "Homepage"
            for seg in segments:
                rows.append({
                    "domain": domain,
                    "URL": url,
                    "Parent": parent,
                    "Name": seg,
                })
                parent = seg
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def derive_menu_hierarchy(df_menu: pd.DataFrame) -> pd.DataFrame:
    df = df_menu.copy()
    df["relative_path"] = df["loc"].astype(str).str.replace(r"https?://[^/]+", "", regex=True).str.rstrip("/")
    df = df.sort_values(["site_name", "relative_path"])
    rows = []
    for site_name, group in df.groupby("site_name"):
        paths = group["relative_path"].tolist()
        labels = group["Response"].fillna("Home").tolist()
        for i, path_i in enumerate(paths):
            for j, path_j in enumerate(paths):
                if i == j:
                    continue
                if path_j.startswith(path_i) and path_j != path_i:
                    rows.append({
                        "site_name": site_name,
                        "loc_hypernym": path_i,
                        "Response_loc_hypernym": labels[i] or "Home",
                        "loc_loc_hyponym": path_j,
                        "Response_hyponym": labels[j],
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfalldom", required=True)
    parser.add_argument("--site-list", required=True, help="120list.xlsx or similar")
    parser.add_argument("--menu-category", required=True, help="dfattrib.xlsx from title categorization")
    parser.add_argument("--url-tags", required=True, help="url_tags_refine.csv")
    parser.add_argument("--output-dir", default="websiteanalysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_feather(args.dfalldom)
    dfurlhier = build_url_hierarchy(df)
    dfurlhier.to_feather(output_dir / "dfurlhier.feather")

    site_list = pd.read_excel(args.site_list)
    site_names = df[["site_name", "sitedomain"]].drop_duplicates()
    site_list = site_list.merge(site_names, left_on="website_menu", right_on="site_name", how="left")
    sitedomains = site_list["sitedomain"].dropna().unique()

    df113 = df[df["sitedomain"].isin(sitedomains)].copy()
    df113.reset_index(drop=True).to_feather(output_dir / "df113dom.feather")

    dfurlhier113 = dfurlhier[dfurlhier["domain"].isin(sitedomains)].copy()
    dfurlhier113.to_feather(output_dir / "dfurlhier113.feather")

    # URL-hierarchy refinement using externally reviewed url tags
    url_tags_df = pd.read_csv(args.url_tags)
    tag_col = url_tags_df.columns[0]
    url_tags = set(url_tags_df[tag_col].dropna().astype(str))

    dfurlhier113["ifwords"] = dfurlhier113[["Name", "Parent"]].apply(
        lambda x: (str(x.iloc[0]) in url_tags) or (str(x.iloc[1]) in url_tags), axis=1
    )
    refined = dfurlhier113[dfurlhier113["ifwords"]].copy()
    refined["Name_clean"] = refined["Name"].astype(str).map(lambda x: re.sub(r"(\?.*|&.*|#.*)", "", x))
    refined["Parent_clean"] = refined["Parent"].astype(str).map(lambda x: re.sub(r"(\?.*|&.*|#.*)", "", x))
    refined = refined[refined["Name_clean"].ne("")]
    refined.to_excel(output_dir / "dfurlhier113Refine.xlsx", index=False)

    # Menu-based hierarchy using title categories
    df113menu = df113[~df113["title_origin"].isna()].copy()
    dfmenu = pd.read_excel(args.menu_category, index_col=0)
    merge_cols = [c for c in ["site_name", "title_origin", "loc"] if c in dfmenu.columns]
    df113menu = df113menu.merge(dfmenu, on=merge_cols, how="left")
    if "Response" not in df113menu.columns:
        raise ValueError("menu category file must contain a Response column")

    menu_hier = derive_menu_hierarchy(df113menu)
    menu_hier = menu_hier.merge(df113[["site_name", "sitedomain"]].drop_duplicates(), on="site_name", how="left")
    menu_hier.to_excel(output_dir / "dfmenu113hier.xlsx", index=False)

    # Combined raw hierarchy
    a = refined[["domain", "Parent_clean", "Name_clean"]].copy()
    a.columns = ["sitedomain", "hypernym", "hyponym"]
    b = menu_hier[["sitedomain", "Response_loc_hypernym", "Response_hyponym"]].copy()
    b.columns = ["sitedomain", "hypernym", "hyponym"]
    hierdata = pd.concat([a, b], ignore_index=True).drop_duplicates()
    hierdata.to_excel(output_dir / "hierdata.xlsx", index=False)

    print(f"saved: {output_dir / 'dfurlhier.feather'}")
    print(f"saved: {output_dir / 'dfurlhier113Refine.xlsx'}")
    print(f"saved: {output_dir / 'dfmenu113hier.xlsx'}")
    print(f"saved: {output_dir / 'hierdata.xlsx'}")


if __name__ == "__main__":
    main()
