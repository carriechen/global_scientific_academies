#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicate Table 2: Kaplan–Meier survival summaries for academy-linked journal title records.

Inputs expected in the working directory:
  - zdb_journal_results_periodic_appended_clean.xlsx
  - gsa_zdb_koeRef_mapping_update.xlsx
  - global_science_academies_final.xlsx

Outputs:
  - table2_replicated_survival.csv
  - table2_replicated_survival.xlsx

Assumptions:
  - Unit of analysis: unique ZDB-ID journal title record.
  - Duration: inclusive publication span, end_year - start_year + 1.
  - Open-ended records are right-censored at CENSOR_YEAR = 2025.
  - Journal records are joined to academies through Source_koeRef -> koeRef.
  - Engineering grouping comes from the academy discipline field.
  - Development grouping comes from country/region classification in the academy table
    when available; otherwise a small manual fallback dictionary can be edited below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


CENSOR_YEAR = 2025
TAU = 100
BOOTSTRAP_N = 2000
RANDOM_SEED = 42


# -----------------------------
# Helpers
# -----------------------------

def norm_colname(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    normalized = {norm_colname(c): c for c in df.columns}
    for cand in candidates:
        key = norm_colname(cand)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of these columns found: {list(candidates)}\nAvailable: {list(df.columns)}")


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_published_span(value: str, censor_year: int = CENSOR_YEAR) -> Tuple[float, float, bool]:
    """
    Parse a ZDB Published field into start, end, censored.
    Examples:
      "1900-"       -> (1900, 2025, True)
      "1900-1950"   -> (1900, 1950, False)
      "1900 - 1950" -> (1900, 1950, False)

    Returns (nan, nan, False) when no usable year exists.
    """
    s = clean_str(value)
    if not s:
        return np.nan, np.nan, False

    # Treat malformed 0000- as unparseable.
    if re.search(r"\b0000\s*-\s*$", s):
        return np.nan, np.nan, False

    years = [int(y) for y in re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", s)]
    if not years:
        return np.nan, np.nan, False

    start = min(years)

    # Open-ended record if the string ends with a dash after a year.
    censored = bool(re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\s*-\s*$", s))
    if censored:
        end = censor_year
    else:
        end = max(years)

    if end < start:
        return np.nan, np.nan, False

    return float(start), float(end), censored


def km_curve(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """
    Kaplan–Meier survival curve.

    durations: positive durations
    events: 1 for observed event, 0 for right-censored
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    event_times = np.sort(np.unique(durations[events == 1]))
    surv = 1.0
    rows = []

    for t in event_times:
        at_risk = np.sum(durations >= t)
        d_t = np.sum((durations == t) & (events == 1))
        if at_risk <= 0:
            continue
        surv *= (1.0 - d_t / at_risk)
        rows.append({"time": t, "survival": surv, "n_at_risk": at_risk, "events": d_t})

    return pd.DataFrame(rows)


def survival_at(km: pd.DataFrame, t: float) -> float:
    if km.empty:
        return np.nan
    sub = km[km["time"] <= t]
    if sub.empty:
        return 1.0
    return float(sub.iloc[-1]["survival"])


def median_survival(km: pd.DataFrame) -> float:
    if km.empty:
        return np.nan
    sub = km[km["survival"] <= 0.5]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["time"])


def rmst(km: pd.DataFrame, tau: float = TAU) -> float:
    """
    Restricted mean survival time up to tau using KM step function.

    S(t)=1 before first event.
    """
    if tau <= 0:
        return 0.0

    times = [0.0]
    survs = [1.0]

    if not km.empty:
        for _, row in km.iterrows():
            t = float(row["time"])
            if t > tau:
                break
            times.append(t)
            survs.append(float(row["survival"]))

    times.append(tau)

    area = 0.0
    # survival over [times[i], times[i+1]) is survs[i]
    for i in range(len(times) - 1):
        area += (times[i + 1] - times[i]) * survs[i]
    return float(area)


def group_summary(df: pd.DataFrame, group_col: str, group_value: str) -> Dict[str, float]:
    sub = df[df[group_col] == group_value].copy()
    durations = sub["duration"].to_numpy(float)
    events = sub["event"].to_numpy(int)

    km = km_curve(durations, events)
    return {
        "Group": group_value,
        "N": int(len(sub)),
        "Events": int(events.sum()),
        "Censored": int((1 - events).sum()),
        "Median": round(median_survival(km), 1) if not np.isnan(median_survival(km)) else np.nan,
        "S25": round(survival_at(km, 25), 2),
        "S50": round(survival_at(km, 50), 2),
        "S100": round(survival_at(km, 100), 2),
        "RMST100": round(rmst(km, TAU), 1),
    }


def logrank_pvalue(df: pd.DataFrame, group_col: str, group_a: str, group_b: str) -> float:
    """
    Two-sample log-rank test implemented directly.
    """
    dfa = df[df[group_col] == group_a]
    dfb = df[df[group_col] == group_b]

    times = np.sort(np.unique(df.loc[df["event"] == 1, "duration"].to_numpy(float)))
    Oa = 0.0
    Ea = 0.0
    Va = 0.0

    for t in times:
        n_a = np.sum(dfa["duration"].to_numpy(float) >= t)
        n_b = np.sum(dfb["duration"].to_numpy(float) >= t)
        n = n_a + n_b

        d_a = np.sum((dfa["duration"].to_numpy(float) == t) & (dfa["event"].to_numpy(int) == 1))
        d_b = np.sum((dfb["duration"].to_numpy(float) == t) & (dfb["event"].to_numpy(int) == 1))
        d = d_a + d_b

        if n <= 1 or d == 0:
            continue

        Oa += d_a
        Ea += d * (n_a / n)
        Va += (n_a * n_b * d * (n - d)) / (n**2 * (n - 1))

    if Va <= 0:
        return np.nan

    chi2 = (Oa - Ea) ** 2 / Va

    # Survival chi-square with 1 df p-value = erfc(sqrt(chi2/2))
    import math
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(p)


def bootstrap_rmst_diff(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap RMST difference group_a - group_b.
    """
    rng = np.random.default_rng(seed)
    a = df[df[group_col] == group_a].reset_index(drop=True)
    b = df[df[group_col] == group_b].reset_index(drop=True)

    def rmst_for(sub: pd.DataFrame) -> float:
        km = km_curve(sub["duration"].to_numpy(float), sub["event"].to_numpy(int))
        return rmst(km, TAU)

    observed = rmst_for(a) - rmst_for(b)

    boot = []
    for _ in range(n_boot):
        aa = a.iloc[rng.integers(0, len(a), len(a))]
        bb = b.iloc[rng.integers(0, len(b), len(b))]
        boot.append(rmst_for(aa) - rmst_for(bb))

    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Two-sided bootstrap p-value for H0: diff=0 by sign proportion.
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = min(float(p), 1.0)

    return float(observed), float(lo), float(hi), p


def p_fmt(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# -----------------------------
# Data preparation
# -----------------------------

def prepare_survival_data(
    journal_path: Path,
    mapping_path: Path,
    academy_path: Path,
) -> pd.DataFrame:
    journal = pd.read_excel(journal_path)
    mapping = pd.read_excel(mapping_path)
    academy = pd.read_excel(academy_path)

    # Column discovery.
    journal_zdb_col = find_col(journal, ["ZDB-ID", "ZDB ID", "zdb_id"])
    journal_pub_col = find_col(journal, ["Published", "published"])
    source_koeref_col = find_col(journal, ["Source_koeRef", "Source koeRef", "koeRef"])

    map_koeref_col = find_col(mapping, ["koeRef", "koeref"])
    map_acad_col = find_col(mapping, ["acad_id", "academy_id"])

    acad_id_col = find_col(academy, ["acad_id", "academy_id"])
    acad_name_col = find_col(academy, ["acad_name_en", "Academy name", "name"])

    # Try likely grouping columns.
    # Edit these candidates if your metadata uses different names.
    discipline_col = find_col(academy, ["discipline", "Discipline", "academy_type", "scope"])
    country_group_col = None
    for candidates in [
        ["development_context", "development status", "development_group"],
        ["country_development", "country income group", "dev_context"],
        ["Global_North_South", "global_north_south", "development"],
    ]:
        try:
            country_group_col = find_col(academy, candidates)
            break
        except KeyError:
            pass

    # If no development grouping exists, use a minimal fallback from a country field.
    country_col = None
    if country_group_col is None:
        for candidates in [["country", "Country", "country_region", "Country/Region"]]:
            try:
                country_col = find_col(academy, candidates)
                break
            except KeyError:
                pass

    # Clean mapping and join.
    mapping2 = mapping[[map_koeref_col, map_acad_col]].dropna().drop_duplicates()
    mapping2 = mapping2.rename(columns={map_koeref_col: "Source_koeRef", map_acad_col: "acad_id"})

    journal2 = journal.copy()
    journal2 = journal2.rename(columns={
        journal_zdb_col: "ZDB-ID",
        journal_pub_col: "Published",
        source_koeref_col: "Source_koeRef",
    })

    # Unique ZDB-ID title record.
    journal2 = journal2.drop_duplicates(subset=["ZDB-ID"]).copy()

    span = journal2["Published"].apply(parse_published_span)
    journal2["start_year"] = span.apply(lambda x: x[0])
    journal2["end_year"] = span.apply(lambda x: x[1])
    journal2["censored"] = span.apply(lambda x: x[2])

    journal2 = journal2.dropna(subset=["start_year", "end_year"]).copy()
    journal2["duration"] = (journal2["end_year"] - journal2["start_year"] + 1).astype(float)
    journal2["event"] = (~journal2["censored"]).astype(int)

    joined = journal2.merge(mapping2, on="Source_koeRef", how="left")

    academy_cols = [acad_id_col, acad_name_col, discipline_col]
    if country_group_col:
        academy_cols.append(country_group_col)
    elif country_col:
        academy_cols.append(country_col)

    academy2 = academy[academy_cols].copy()
    rename = {
        acad_id_col: "acad_id",
        acad_name_col: "acad_name_en",
        discipline_col: "discipline",
    }
    if country_group_col:
        rename[country_group_col] = "development_context"
    elif country_col:
        rename[country_col] = "country"

    academy2 = academy2.rename(columns=rename).drop_duplicates(subset=["acad_id"])
    joined = joined.merge(academy2, on="acad_id", how="left")

    # Engineering vs non-engineering.
    joined["engineering_group"] = np.where(
        joined["discipline"].astype(str).str.contains("engineer", case=False, na=False),
        "Engineering",
        "Non-engineering",
    )

    # Development grouping.
    if "development_context" in joined.columns:
        dev = joined["development_context"].astype(str).str.strip()
        joined["development_group"] = np.where(
            dev.str.contains("developing|south", case=False, na=False),
            "Global South / Developing",
            "Global North / Developed",
        )
    else:
        # Conservative editable fallback. Add countries here if needed.
        developing_countries = {
            "Türkiye", "Turkey", "China", "India", "Brazil", "Mexico", "South Africa",
            "Argentina", "Chile", "Colombia", "Egypt", "Indonesia", "Malaysia", "Thailand",
            "Philippines", "Vietnam", "Kazakhstan", "Kenya", "Nigeria", "Ghana", "Pakistan",
            "Bangladesh", "Morocco", "Tunisia", "Iran", "Iraq", "Jordan", "Lebanon",
            "Republic of the Congo", "Congo", "Kosovo"
        }
        country = joined.get("country", pd.Series([""] * len(joined))).astype(str).str.strip()
        joined["development_group"] = np.where(
            country.isin(developing_countries),
            "Global South / Developing",
            "Global North / Developed",
        )

    # Keep only rows with an academy link and valid survival data.
    joined = joined.dropna(subset=["acad_id", "duration", "event"]).copy()
    joined = joined[joined["duration"] > 0].copy()

    return joined


# -----------------------------
# Table construction
# -----------------------------

def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Panel A
    rows.append({"Panel": "Panel A. Journal title records linked to engineering vs. non-engineering academies"})
    rows.append(group_summary(df, "engineering_group", "Engineering"))
    rows.append(group_summary(df, "engineering_group", "Non-engineering"))

    p_lr = logrank_pvalue(df, "engineering_group", "Engineering", "Non-engineering")
    rows.append({"Group": "Log-rank p", "N": p_fmt(p_lr)})

    diff, lo, hi, p_boot = bootstrap_rmst_diff(df, "engineering_group", "Engineering", "Non-engineering")
    rows.append({
        "Group": "RMST diff, Engineering − Non-engineering",
        "N": f"{diff:.1f} [{lo:.1f}, {hi:.1f}]; boot p = {p_fmt(p_boot)}",
    })

    # Panel B
    rows.append({"Panel": "Panel B. Journal title records linked to Global North vs. Global South academies"})
    rows.append(group_summary(df, "development_group", "Global North / Developed"))
    rows.append(group_summary(df, "development_group", "Global South / Developing"))

    p_lr = logrank_pvalue(df, "development_group", "Global North / Developed", "Global South / Developing")
    rows.append({"Group": "Log-rank p", "N": p_fmt(p_lr)})

    diff, lo, hi, p_boot = bootstrap_rmst_diff(df, "development_group", "Global North / Developed", "Global South / Developing")
    rows.append({
        "Group": "RMST diff, Developed − Developing",
        "N": f"{diff:.1f} [{lo:.1f}, {hi:.1f}]; boot p = {p_fmt(p_boot)}",
    })

    columns = ["Panel", "Group", "N", "Events", "Censored", "Median", "S25", "S50", "S100", "RMST100"]
    out = pd.DataFrame(rows)
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    return out[columns]


def main() -> None:
    journal_path = Path("zdb_journal_results_periodic_appended_clean.xlsx")
    mapping_path = Path("gsa_zdb_koeRef_mapping_update.xlsx")
    academy_path = Path("global_science_academies_final.xlsx")

    data = prepare_survival_data(journal_path, mapping_path, academy_path)
    table2 = build_table2(data)

    table2.to_csv("table2_replicated_survival.csv", index=False, encoding="utf-8-sig")
    table2.to_excel("table2_replicated_survival.xlsx", index=False)

    print(table2.to_string(index=False))
    print()
    print(f"Rows used: {len(data)}")
    print("Saved table2_replicated_survival.csv")
    print("Saved table2_replicated_survival.xlsx")


if __name__ == "__main__":
    main()
