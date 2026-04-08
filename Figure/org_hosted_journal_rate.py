# -*- coding: utf-8 -*-
# Figure 4 (Recommended 1): Dumbbell plots + Wilson 95% CI
# - 4A: UN M49 Developed vs Developing hosted-journal rate
# - 4B: Engineering vs Non-engineering hosted-journal rate
#
# UPDATE (as requested):
#   - Use a country-alias dictionary to map GSA country values to UNSD canonical names
#     before normalizing and joining to the UNSD May 2022 historical classification.
#   - Keep the rest of your pipeline unchanged:
#       1) Filter out academy_type_annotation == "Transnational Academy" FIRST
#       2) Engineering = discipline contains "Engineering"
#
# Inputs:
#   gsa_zdb_koeRef_mapping.csv  (koeRef, Corporate Body, acad_id)
#   global_science_academies_final.xlsx  (acad_id, country, discipline, academy_type_annotation)
#   historical-classification-of-developed-and-developing-regions(Distinction as of May 2022).csv
#
# Output:
#   org_hosted_journal_rate/Figure4A_UNM49_dumbbell.png/svg
#   org_hosted_journal_rate/Figure4B_engineering_dumbbell.png/svg
#   org_hosted_journal_rate/Figure4_tables.xlsx

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).parent
FILE_MAPPING = BASE / "gsa_zdb_koeRef_mapping.csv"
FILE_GSA = BASE / "global_science_academies_final.xlsx"
FILE_UN_M49 = BASE / "historical-classification-of-developed-and-developing-regions(Distinction as of May 2022).csv"

OUT_DIR = BASE / "org_hosted_journal_rate"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Country alias dictionary (GSA -> UNSD "Country or Area")
# -----------------------------
COUNTRY_ALIAS = {
    # demonyms / adjectives -> country names
    "australian": "Australia",
    "german": "Germany",
    "romanian": "Romania",
    "tunisian": "Tunisia",
    "nicaraguan": "Nicaragua",

    # common short names -> UNSD canonical
    "palestine": "State of Palestine",
    "bolivia": "Bolivia (Plurinational State of)",
    "north korea": "Democratic People's Republic of Korea",
    "south korea": "Republic of Korea",
    "czech republic": "Czechia",
    "moldova": "Republic of Moldova",
    "russia": "Russian Federation",
    "vatican": "Holy See",
    "vatican city": "Holy See",
    "república dominicana": "Dominican Republic",
    "republica dominicana": "Dominican Republic",
    "tanzania": "United Republic of Tanzania",
    "iran": "Iran (Islamic Republic of)",
    "venezuela": "Venezuela (Bolivarian Republic of)",
    "vietnam": "Viet Nam",
    "united states": "United States of America",
    "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "türkiye": "Türkiye",
    "kosovo": "Kosovo",  # Will need manual handling in UNSD data
    "republic of the congo": "Republic of the Congo",
    # Leave these unmapped on purpose unless you decide a policy:
    # "taiwan": ???,
    # "tatarstan": ???,
}

def canonicalize_country_for_unsd(country_raw: str) -> str:
    """
    Apply alias mapping to convert GSA 'country' values into UNSD canonical
    'Country or Area' names, then return the canonical string.
    """
    s = norm_str(country_raw)
    key = s.strip().lower()
    
    # Handle multi-country entries - take the first country
    if ';' in s:
        s = s.split(';')[0].strip()
        key = s.strip().lower()
    
    return COUNTRY_ALIAS.get(key, s)


# -----------------------------
# Helpers
# -----------------------------
def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def is_missing_like(x) -> bool:
    s = norm_str(x).lower()
    return s in {"", "na", "n/a", "none", "-", "nan"}

def normalize_country_name(name: str) -> str:
    """
    Normalize for robust joining:
      - lower
      - remove punctuation (but keep unicode letters)
      - collapse whitespace
    """
    s = norm_str(name).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # Support unicode letters
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_engineering_from_discipline(discipline: str) -> int:
    """
    Engineering = 1 if discipline contains substring 'Engineering' (case-insensitive).
    """
    s = norm_str(discipline).lower()
    return int("engineering" in s)

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    Returns (lo, hi). If n=0 returns (nan, nan).
    """
    if n <= 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # 95% two-sided
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# -----------------------------
# Load & build academy-level table
# -----------------------------
def load_hosted_proxy_mapping() -> pd.DataFrame:
    """
    Load koeRef mapping from CSV and determine hosted journal status.
    Returns DataFrame with acad_id and hosted_journal flag.
    """
    df = pd.read_csv(FILE_MAPPING)
    
    # Check required columns
    required = ["koeRef", "Corporate Body", "acad_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV must include columns: {missing}")
    
    # Clean data
    df = df.copy()
    df["acad_id"] = df["acad_id"].astype(str).str.strip()
    df["koeRef"] = df["koeRef"].astype(str).str.strip()
    
    # Determine hosted journal status based on koeRef presence
    # If koeRef is not missing-like, assume hosted journal
    df["hosted_journal"] = df["koeRef"].apply(lambda x: 0 if is_missing_like(x) else 1).astype(int)
    
    # One row per academy (max is safe)
    df = df.groupby("acad_id", as_index=False)["hosted_journal"].max()
    
    print(f"[INFO] Loaded {len(df)} academy mappings from koeRef CSV")
    print(f"[INFO] Hosted journals: {df['hosted_journal'].sum()} / {len(df)} ({df['hosted_journal'].mean():.1%})")
    
    return df


def load_gsa_filter_transnational_and_engineering_flag() -> pd.DataFrame:
    """
    GSA filtering:
      - Drop blank acad_id rows
      - Filter OUT specific transnational academies
      - Engineering flag from discipline contains "Engineering"
      - Apply country alias mapping BEFORE normalization for UNSD join
    """
    df = pd.read_excel(FILE_GSA)

    required = ["acad_id", "country", "discipline", "acad_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GSA file missing required columns: {missing}")

    df = df.copy()
    df["acad_id"] = df["acad_id"].astype(str).str.strip()
    df = df[df["acad_id"].notna() & (df["acad_id"] != "")]

    # ---- Filter OUT specific transnational academies ----
    transnational_academies = [
        "World Academy of Sciences",
        "ASEAN Academy of Engineering and Technology",
        "World Academy of Art and Science", 
        "Academia de Ciências da América Latina",
        "Academia Europaea",
        "African Academy of Sciences",
        "Islamic World Academy of Sciences",
        "Académie des sciences, des arts, des cultures d'Afrique et des diasporas africaines"
    ]
    
    before = len(df)
    df["acad_name_clean"] = df["acad_name"].astype(str).str.strip()
    
    # Filter out any academy containing these transnational names
    mask = df["acad_name_clean"].str.contains("|".join(transnational_academies), case=False, na=False)
    df = df[~mask].copy()
    
    after = len(df)
    print(f"[INFO] Filtered {before - after} transnational academies, {after} remaining.")
    
    if before - after > 0:
        print(f"[INFO] Removed academies:")
        removed = df[mask]["acad_name_clean"].unique()
        for academy in removed:
            print(f"  - {academy}")

    # engineering flag
    df["is_engineering"] = df["discipline"].apply(is_engineering_from_discipline).astype(int)

    # country alias -> canonical, then normalize for join
    df["country_canonical"] = df["country"].apply(canonicalize_country_for_unsd)
    df["country_norm"] = df["country_canonical"].apply(normalize_country_name)

    return df[
        ["acad_id", "country", "country_canonical", "country_norm",
         "discipline", "acad_name_clean", "is_engineering"]
    ]


def load_un_m49() -> pd.DataFrame:
    """
    Reads UNSD 'historical classification of developed and developing regions'
    (Distinction as of May 2022) and returns:
      country_norm, dev_group in {Developed, Developing}
    """
    df = pd.read_csv(FILE_UN_M49, encoding="latin1", header=0)

    df = df.rename(columns={
        "Country or Area": "country",
        "Developed / Developing regions": "dev_group",
    })
    if "country" not in df.columns or "dev_group" not in df.columns:
        raise ValueError("UN M49 file columns not recognized. Please verify header row.")

    df = df.copy()
    df["country_norm"] = df["country"].apply(normalize_country_name)
    df["dev_group_raw"] = df["dev_group"].astype(str).str.strip()

    def map_dev(x: str) -> str:
        if "Developed" in x:
            return "Developed"
        if "Developing" in x:
            return "Developing"
        return "Unknown"

    df["dev_group"] = df["dev_group_raw"].apply(map_dev)
    df = df[df["dev_group"].isin(["Developed", "Developing"])].copy()
    
    # Add missing countries manually
    additional_countries = [
        {"country": "Kosovo", "dev_group": "Developing"},
        {"country": "Republic of the Congo", "dev_group": "Developing"},
        {"country": "Türkiye", "dev_group": "Developing"},  # Add both versions
        {"country": "Turkey", "dev_group": "Developing"},
    ]
    
    for country_data in additional_countries:
        country_name = country_data["country"]
        dev_group = country_data["dev_group"]
        country_norm = normalize_country_name(country_name)
        
        # Only add if not already present
        if not df[df["country_norm"] == country_norm].any().any():
            new_row = pd.DataFrame({
                "country": [country_name],
                "dev_group": [dev_group],
                "country_norm": [country_norm],
                "dev_group_raw": [dev_group]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"[INFO] Added missing country: {country_name} -> {dev_group}")
    
    df = df.drop_duplicates(subset=["country_norm"], keep="first")
    return df[["country_norm", "dev_group"]]


def build_academy_table() -> pd.DataFrame:
    host = load_hosted_proxy_mapping()
    gsa = load_gsa_filter_transnational_and_engineering_flag()
    m49 = load_un_m49()

    df = gsa.merge(host, on="acad_id", how="left")
    df["hosted_journal"] = df["hosted_journal"].fillna(0).astype(int)

    df = df.merge(m49, on="country_norm", how="left")
    df["dev_group"] = df["dev_group"].fillna("Unknown")
    return df


# -----------------------------
# Summaries for dumbbell
# -----------------------------
def summarize_rate_ci(df: pd.DataFrame, group_col: str, group_order: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_col)["hosted_journal"]
        .agg(n="count", k="sum")
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    out = out.set_index("group").reindex(group_order).reset_index()
    out["k"] = out["k"].fillna(0).astype(int)
    out["n"] = out["n"].fillna(0).astype(int)
    out["rate"] = out.apply(lambda r: (r["k"] / r["n"]) if r["n"] > 0 else np.nan, axis=1)

    cis = out.apply(lambda r: wilson_ci(int(r["k"]), int(r["n"])), axis=1, result_type="expand")
    out["ci_lo"] = cis[0]
    out["ci_hi"] = cis[1]
    return out


# -----------------------------
# Dumbbell plot (two groups) - clean layout
# -----------------------------
def plot_dumbbell_two_groups(
    left_label: str,
    right_label: str,
    left_row: pd.Series,
    right_row: pd.Series,
    title: str,
    subtitle: str,
    out_prefix: str,
):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.subplots_adjust(top=0.78)

    xl = float(left_row["rate"])
    xr = float(right_row["rate"])
    ll, lh = float(left_row["ci_lo"]), float(left_row["ci_hi"])
    rl, rh = float(right_row["ci_lo"]), float(right_row["ci_hi"])

    ax.hlines(y=0, xmin=min(xl, xr), xmax=max(xl, xr), linewidth=2)

    ax.hlines(y=-0.15, xmin=ll, xmax=lh, linewidth=4)
    ax.hlines(y=+0.15, xmin=rl, xmax=rh, linewidth=4)

    ax.scatter([xl], [0], s=160, zorder=3)
    ax.scatter([xr], [0], s=160, zorder=3)

    ax.text(xl, -0.36, f"{left_label}\n{xl:.1%} (n={int(left_row['n'])})\n{int(left_row['k'])} hosted",
            ha="center", va="top", fontsize=11)
    ax.text(xr, -0.36, f"{right_label}\n{xr:.1%} (n={int(right_row['n'])})\n{int(right_row['k'])} hosted",
            ha="center", va="top", fontsize=11)

    ax.set_ylim(-0.65, 0.40)
    ax.set_yticks([])
    ax.set_xlabel("Hosted-journal rate (Wilson 95% CI)")

    fig.suptitle(title, x=0.01, ha="left", y=0.98)
    fig.text(0.01, 0.90, subtitle, ha="left", va="top", fontsize=10)

    xmin = max(0.0, min(ll, rl, xl, xr) - 0.05)
    xmax = min(1.0, max(lh, rh, xl, xr) + 0.05)
    ax.set_xlim(xmin, xmax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_prefix}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{out_prefix}.svg")
    plt.close(fig)


# -----------------------------
# Create combined Figure 4 (A+B) with shared x-axis
# -----------------------------
def plot_figure4_combined(df: pd.DataFrame, dev_sum: pd.DataFrame, eng_sum: pd.DataFrame):
    # pick rows
    dev_developed = dev_sum[dev_sum["group"].eq("Developed")].iloc[0]
    dev_developing = dev_sum[dev_sum["group"].eq("Developing")].iloc[0]

    eng_non = eng_sum[eng_sum["group"].eq("Non-engineering")].iloc[0]
    eng_eng = eng_sum[eng_sum["group"].eq("Engineering")].iloc[0]

    # shared x-limits (make panels comparable)
    # Use the min/max across BOTH panels' CIs + padding
    vals = [
        float(dev_developed["ci_lo"]), float(dev_developed["ci_hi"]),
        float(dev_developing["ci_lo"]), float(dev_developing["ci_hi"]),
        float(eng_non["ci_lo"]), float(eng_non["ci_hi"]),
        float(eng_eng["ci_lo"]), float(eng_eng["ci_hi"]),
    ]
    xmin = max(0.0, min(vals) - 0.05)
    xmax = min(1.0, max(vals) + 0.05)
    xlim = (xmin, xmax)

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 6.2), sharex=True
    )

    # Panel A
    plot_dumbbell_two_groups_on_ax(
        ax=axes[0],
        left_label="Developing",
        right_label="Developed",
        left_row=dev_developing,
        right_row=dev_developed,
        panel_title="ORI fragmentation (hosting): Developed vs Developing",
        panel_letter="A",
        subtitle=None,          # keep inside-figure text minimal (journal style)
        xlim=xlim,
        show_xlabel=False,
    )

    # Panel B
    plot_dumbbell_two_groups_on_ax(
        ax=axes[1],
        left_label="Engineering",
        right_label="Non-engineering",
        left_row=eng_eng,
        right_row=eng_non,
        panel_title="ORI fragmentation (hosting): Engineering vs Non-engineering",
        panel_letter="B",
        subtitle=None,
        xlim=xlim,
        show_xlabel=True,
    )

    # spacing like top journals
    fig.subplots_adjust(top=0.93, hspace=0.25)

    fig.savefig(OUT_DIR / "Figure8.png", dpi=300)
    fig.savefig(OUT_DIR / "Figure8.svg")
    plt.close(fig)

def main():
    df = build_academy_table()

    print("[INFO] Total academies after filtering Transnational Academy:", len(df))

    # Quick diagnostic: which country strings still fail after alias mapping?
    unknown_countries = (
        df.loc[df["dev_group"] == "Unknown", "country"]
        .astype(str).value_counts()
    )
    if len(unknown_countries) > 0:
        print("[WARN] Countries still mapped to dev_group == 'Unknown' (top 30):")
        print(unknown_countries.head(30))

    # --------- 4A: UN M49 Developed vs Developing
    dev_order = ["Developed", "Developing", "Unknown"]
    dev_sum = summarize_rate_ci(df, "dev_group", dev_order)

    left = dev_sum[dev_sum["group"].eq("Developed")].iloc[0]
    right = dev_sum[dev_sum["group"].eq("Developing")].iloc[0]

    plot_dumbbell_two_groups(
        left_label="Developed",
        right_label="Developing",
        left_row=left,
        right_row=right,
        title="Journal hosting fragmentation: Developed vs Developing",
        subtitle="Classification follows UN M49 (UNSD historical distinction as of May 2022). Hosting is inferred from koeRef presence in ZDB.",
        out_prefix="Figure8A_UNM49_dumbbell",
    )

    # --------- 4B: Engineering vs Non-engineering
    df["eng_group"] = df["is_engineering"].map({1: "Engineering", 0: "Non-engineering"})
    eng_order = ["Non-engineering", "Engineering"]
    eng_sum = summarize_rate_ci(df, "eng_group", eng_order)

    left2 = eng_sum[eng_sum["group"].eq("Non-engineering")].iloc[0]
    right2 = eng_sum[eng_sum["group"].eq("Engineering")].iloc[0]

    plot_dumbbell_two_groups(
        left_label="Non-engineering",
        right_label="Engineering",
        left_row=left2,
        right_row=right2,
        title="Journal hosting fragmentation: Engineering vs Non-engineering",
        subtitle="Engineering is defined as discipline values containing 'Engineering'.",
        out_prefix="Figure8B_engineering_dumbbell",
    )
    plot_figure4_combined(df, dev_sum, eng_sum)
    
    # --------- Export tables
    out_xlsx = OUT_DIR / "Figure8s.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="academy_level", index=False)
        dev_sum.to_excel(w, sheet_name="by_un_m49_dev", index=False)
        eng_sum.to_excel(w, sheet_name="by_engineering", index=False)

    print("[INFO] Done. Outputs in:", OUT_DIR.resolve())

# -----------------------------
# Dumbbell plot (two groups) drawn on a provided Axes
# -----------------------------
def plot_dumbbell_two_groups_on_ax(
    ax: plt.Axes,
    left_label: str,
    right_label: str,
    left_row: pd.Series,
    right_row: pd.Series,
    panel_title: str,
    panel_letter: str,
    subtitle: str | None = None,
    xlim: tuple[float, float] | None = None,
    show_xlabel: bool = True,
):
    xl = float(left_row["rate"])
    xr = float(right_row["rate"])
    ll, lh = float(left_row["ci_lo"]), float(left_row["ci_hi"])
    rl, rh = float(right_row["ci_lo"]), float(right_row["ci_hi"])

    # connecting line
    ax.hlines(y=0, xmin=min(xl, xr), xmax=max(xl, xr), linewidth=2)

    # CI whiskers
    ax.hlines(y=-0.15, xmin=ll, xmax=lh, linewidth=4)
    ax.hlines(y=+0.15, xmin=rl, xmax=rh, linewidth=4)

    # points
    ax.scatter([xl], [0], s=160, zorder=3)
    ax.scatter([xr], [0], s=160, zorder=3)

    # group labels under points
    ax.text(xl, -0.36, f"{left_label}\n{xl:.1%} (n={int(left_row['n'])})\n{int(left_row['k'])} hosted",
            ha="center", va="top", fontsize=11)
    ax.text(xr, -0.36, f"{right_label}\n{xr:.1%} (n={int(right_row['n'])})\n{int(right_row['k'])} hosted",
            ha="center", va="top", fontsize=11)

    # axes formatting
    ax.set_ylim(-0.65, 0.40)
    ax.set_yticks([])

    if show_xlabel:
        ax.set_xlabel("Hosted-journal rate (Wilson 95% CI)")
    else:
        ax.set_xlabel("")

    # panel letter + short title (journal style)
    ax.text(0.0, 1.05, f"{panel_letter}. {panel_title}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=12)

    # optional short subtitle (keep minimal; usually move to caption)
    if subtitle:
        ax.text(0.0, 0.98, subtitle, transform=ax.transAxes,
                ha="left", va="top", fontsize=10)

    if xlim is None:
        xmin = max(0.0, min(ll, rl, xl, xr) - 0.05)
        xmax = min(1.0, max(lh, rh, xl, xr) + 0.05)
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(*xlim)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)





if __name__ == "__main__":
    main()
    
