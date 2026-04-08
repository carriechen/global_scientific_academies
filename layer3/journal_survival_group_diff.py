# -*- coding: utf-8 -*-


from __future__ import annotations

import re
import math
import bisect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Column-robust accessors (avoid KeyError when column names differ)
# -----------------------------
def _pick_first_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Missing required column for {name}. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )

def get_duration_event_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return duration (years) and event(1=event/death, 0=censored) arrays.
    Supports multiple possible column names.
    """
    dur_col = _pick_first_col(df, ["duration", "duration_years", "duration_year", "lifespan_years"], "duration")
    evt_col = _pick_first_col(df, ["event", "event_observed", "event_flag", "ended"], "event")
    durations = df[dur_col].to_numpy(dtype=float)
    events = df[evt_col].to_numpy(dtype=int)
    return durations, events


# -----------------------------
# Paths (EDIT if needed)
# -----------------------------
BASE = Path(__file__).parent

def _resolve_first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    # return the first candidate as default (for clearer error messages later)
    return candidates[0]

FILE_GSA = _resolve_first_existing(
    BASE / "global_science_academies V1.xlsx",
)

FILE_UN_M49 = _resolve_first_existing(
    BASE / "historical-classification-of-developed-and-developing-regions(Distinction as of May 2022).csv",
)

# ZDB journal results
FILE_ZDB_JOURNALS = _resolve_first_existing(
    BASE / "zdb_result/zdb_journal_results.xlsx",
    BASE / "zdb_journal_results.xlsx",
)

# Preferred: explicit ZDB-ID <-> acad_id map (CSV). Fallback: build from corporate_body_mapping.xlsx.
FILE_ZDB_ACAD_MAP = _resolve_first_existing(
    BASE / "zdb_result/zdb_academies_readable.csv",
    BASE / "zdb_academies_readable.csv",
)
FILE_CORP_MAP = _resolve_first_existing(
    BASE / "corporate_body_mapping.xlsx",
)

OUT_DIR = BASE / "zdb_result/survival_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# Country alias dictionary (same as Fig4)
# -----------------------------
COUNTRY_ALIAS = {
    "australian": "Australia",
    "german": "Germany",
    "romanian": "Romania",
    "tunisian": "Tunisia",
    "nicaraguan": "Nicaragua",
    "palestine": "State of Palestine",
    "bolivia": "Bolivia (Plurinational State of)",
    "north korea": "Democratic People's Republic of Korea",
    "south korea": "Republic of Korea",
    "czech republic": "Czechia",
    "moldova": "Republic of Moldova",
    "russia": "Russian Federation",
    "vatican": "Holy See",
    "república dominicana": "Dominican Republic",
    "republica dominicana": "Dominican Republic",
    "tanzania": "United Republic of Tanzania",
    "iran": "Iran (Islamic Republic of)",
    "venezuela": "Venezuela (Bolivarian Republic of)",
    "vietnam": "Viet Nam",
    "united states": "United States of America",
    "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
}

def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def canonicalize_country_for_unsd(country_raw: str) -> str:
    s = norm_str(country_raw)
    key = s.strip().lower()
    return COUNTRY_ALIAS.get(key, s)

def normalize_country_name(name: str) -> str:
    s = norm_str(name).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_engineering_from_discipline(discipline: str) -> int:
    return int("engineering" in norm_str(discipline).lower())


# -----------------------------
# Load academy attributes (same logic as Fig4)
# -----------------------------
def load_gsa_filter_transnational_and_engineering_flag() -> pd.DataFrame:
    df = pd.read_excel(FILE_GSA)

    required = ["acad_id", "country", "discipline", "academy_type_annotation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GSA file missing required columns: {missing}")

    df = df.copy()
    df["acad_id"] = df["acad_id"].astype(str).str.strip()
    df = df[df["acad_id"].notna() & (df["acad_id"] != "")]

    # filter out transnational FIRST
    df["academy_type_annotation"] = df["academy_type_annotation"].astype(str).str.strip()
    df = df[df["academy_type_annotation"] != "Transnational Academy"].copy()

    df["is_engineering"] = df["discipline"].apply(is_engineering_from_discipline).astype(int)

    df["country_canonical"] = df["country"].apply(canonicalize_country_for_unsd)
    df["country_norm"] = df["country_canonical"].apply(normalize_country_name)

    return df[["acad_id", "country_norm", "is_engineering"]]

def load_un_m49() -> pd.DataFrame:
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
    df = df.drop_duplicates(subset=["country_norm"], keep="first")
    return df[["country_norm", "dev_group"]]

def build_academy_attributes() -> pd.DataFrame:
    gsa = load_gsa_filter_transnational_and_engineering_flag()
    m49 = load_un_m49()
    df = gsa.merge(m49, on="country_norm", how="left")
    df["dev_group"] = df["dev_group"].fillna("Unknown")
    df["eng_group"] = df["is_engineering"].map({1: "Engineering", 0: "Non-engineering"})
    return df[["acad_id", "eng_group", "dev_group"]]


# -----------------------------
# Lineage building from zdb_journal_results.xlsx
# -----------------------------
YEAR_PAT = re.compile(r"(1[5-9]\d{2}|20\d{2}|21\d{2})")

def parse_years(published: str) -> Tuple[float, float, bool]:
    s = "" if pd.isna(published) else str(published)
    m = list(YEAR_PAT.finditer(s))
    if not m:
        return (np.nan, np.nan, False)
    sy = int(m[0].group())
    ey = int(m[-1].group())
    cens = bool(re.search(r"(\d{4}|\[\d{4}\]|\d{4}\?)\s*-\s*$", s))
    return (sy, ey, cens)

def norm_title(t: str) -> str:
    if pd.isna(t):
        return ""
    t = str(t).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def extract_related_titles(rel: str) -> List[str]:
    if pd.isna(rel):
        return []
    parts = [p.strip() for p in str(rel).strip().split("/")]
    out, seen = [], set()
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2 or p2 in {"-", "—"}:
            continue
        key = norm_title(p2)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out

class DSU:
    def __init__(self, items: List[str]):
        self.p = {x: x for x in items}
        self.r = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

def build_lineages(CENSOR_YEAR: int = 2025) -> pd.DataFrame:
    z = pd.read_excel(FILE_ZDB_JOURNALS)

    # ---- keep only Periodical (Journal) ----
    pub_col = "Type of publication"
    if pub_col not in z.columns:
        raise ValueError(f"zdb_journal_results.xlsx missing column: {pub_col}")
    z[pub_col] = z[pub_col].astype(str).str.strip()
    before_pub = len(z)
    z = z[z[pub_col].eq("Periodical (Journal)")].copy()
    after_pub = len(z)
    print(f"[INFO] Filtered Type of publication == 'Periodical (Journal)': {before_pub-after_pub} removed, {after_pub} remaining.")

    need = ["ZDB-ID", "Title", "Published", "Former/later titles"]
    miss = [c for c in need if c not in z.columns]
    if miss:
        raise ValueError(f"zdb_journal_results.xlsx missing columns: {miss}")

    z = z.copy()
    z["ZDB-ID"] = z["ZDB-ID"].astype(str).str.strip()
    z["title_norm"] = z["Title"].apply(norm_title)

    sy_ey_c = z["Published"].apply(parse_years)
    z["start_year"] = sy_ey_c.apply(lambda x: x[0])
    z["end_year_parsed"] = sy_ey_c.apply(lambda x: x[1])
    z["is_censored_hint"] = sy_ey_c.apply(lambda x: x[2])

    # title -> ids
    title_to_ids: Dict[str, set] = {}
    for zid, tn in zip(z["ZDB-ID"], z["title_norm"]):
        if tn:
            title_to_ids.setdefault(tn, set()).add(zid)

    all_ids = z["ZDB-ID"].tolist()
    dsu = DSU(all_ids)

    # union by former/later titles (exact normalized title match)
    for _, row in z.iterrows():
        cur = str(row["ZDB-ID"])
        for tnorm in extract_related_titles(row.get("Former/later titles", np.nan)):
            if tnorm in title_to_ids:
                for oid in title_to_ids[tnorm]:
                    dsu.union(cur, oid)

    # components
    comp: Dict[str, List[str]] = {}
    for zid in all_ids:
        comp.setdefault(dsu.find(zid), []).append(zid)

    z_by_id = z.set_index("ZDB-ID").to_dict(orient="index")

    rows = []
    for root, members in comp.items():
        starts, ends = [], []
        cens = False

        for mid in members:
            r = z_by_id.get(mid, {})
            sy = r.get("start_year", np.nan)
            ey = r.get("end_year_parsed", np.nan)
            if not pd.isna(sy):
                starts.append(int(sy))
            if not pd.isna(ey):
                ends.append(int(ey))
            if bool(r.get("is_censored_hint", False)):
                cens = True

        if not starts:
            continue

        start = min(starts)
        end = max(ends) if ends else np.nan

        if cens or pd.isna(end):
            end_final, event = CENSOR_YEAR, 0
        else:
            end_final, event = int(end), 1

        duration = end_final - start + 1
        if duration < 1:
            continue

        rows.append({
            "lineage_id": root,
            "n_titles": len(members),
            "start_year": start,
            "end_year": end_final,
            "duration_years": int(duration),
            "event_observed": int(event),
            "member_zdb_ids": members,  # for mapping to academies
        })

    return pd.DataFrame(rows)


# -----------------------------
# Map lineages -> groups (by academy IDs hosting member titles)
# -----------------------------
def load_zdb_to_academy_map() -> pd.DataFrame:
    """
    Return mapping table with columns: acad_id, ZDB-ID.

    Priority:
      1) If zdb_academies_readable.csv exists (FILE_ZDB_ACAD_MAP), load it (expects columns Academy_ID, ZDB-ID).
      2) Else, if corporate_body_mapping.xlsx exists (FILE_CORP_MAP), build mapping via:
           zdb_journal_results.xlsx: (ZDB-ID, Source_koeRef)
           corporate_body_mapping.xlsx sheet 'cid_gsa': (koeRef -> acad_id)
         and merge on koeRef.

    This keeps the pipeline reproducible across different project directory layouts.
    """
    # (1) direct CSV mapping
    if FILE_ZDB_ACAD_MAP.exists():
        m = pd.read_csv(FILE_ZDB_ACAD_MAP, low_memory=False)
        need = ["Academy_ID", "ZDB-ID"]
        miss = [c for c in need if c not in m.columns]
        if miss:
            raise ValueError(f"zdb_academies_readable.csv missing columns: {miss}")
        m = m.copy()
        m["acad_id"] = m["Academy_ID"].astype(str).str.strip()
        m["ZDB-ID"] = m["ZDB-ID"].astype(str).str.strip()
        return m[["acad_id", "ZDB-ID"]].dropna().drop_duplicates()

    # (2) build mapping from corporate_body_mapping.xlsx
    if not FILE_CORP_MAP.exists():
        raise FileNotFoundError(
            "No academy mapping file found. Provide either "
            f"{FILE_ZDB_ACAD_MAP} or {FILE_CORP_MAP}."
        )

    # Read journal -> koeRef
    z = pd.read_excel(FILE_ZDB_JOURNALS)
    need_z = ["ZDB-ID", "Source_koeRef"]
    miss_z = [c for c in need_z if c not in z.columns]
    if miss_z:
        raise ValueError(f"zdb_journal_results.xlsx missing columns for mapping: {miss_z}")
    z = z[["ZDB-ID", "Source_koeRef"]].copy()
    z["ZDB-ID"] = z["ZDB-ID"].astype(str).str.strip()
    z["koeRef"] = z["Source_koeRef"].astype(str).str.strip()

    # Read koeRef -> acad_id
    cid = pd.read_excel(FILE_CORP_MAP, sheet_name="cid_gsa")
    # normalize columns that sometimes become 'Unnamed: 0'
    cols = [str(c).strip() for c in cid.columns]
    cid.columns = cols
    # find best column names
    koe_col = None
    for c in cid.columns:
        if c.lower() == "koeref":
            koe_col = c
            break
    acad_col = None
    for c in cid.columns:
        if c.lower() == "acad_id":
            acad_col = c
            break
    if koe_col is None or acad_col is None:
        raise ValueError(
            "corporate_body_mapping.xlsx sheet 'cid_gsa' must contain columns "
            "'koeRef' and 'acad_id'. Found: " + ", ".join(cid.columns)
        )

    cid = cid[[koe_col, acad_col]].copy()
    cid.rename(columns={koe_col: "koeRef", acad_col: "acad_id"}, inplace=True)
    cid["koeRef"] = cid["koeRef"].astype(str).str.strip()
    cid["acad_id"] = cid["acad_id"].astype(str).str.strip()

    m = z.merge(cid, how="left", left_on="koeRef", right_on="koeRef")
    m = m.dropna(subset=["acad_id"])
    m = m[["acad_id", "ZDB-ID"]].drop_duplicates()
    return m

def assign_lineage_group(
    lineages: pd.DataFrame,
    zdb2acad: pd.DataFrame,
    acad_attr: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    group_col in {"eng_group", "dev_group"}
    Strategy:
      - get academies for each lineage via member ZDB-IDs
      - map academy->group
      - if multiple groups present: Mixed
    """
    zdb_to_acads = zdb2acad.groupby("ZDB-ID")["acad_id"].apply(lambda x: list(set(x))).to_dict()
    acad_to_group = acad_attr.set_index("acad_id")[group_col].to_dict()

    out = lineages.copy()
    g_list = []
    for members in out["member_zdb_ids"]:
        acad_ids = []
        for zid in members:
            acad_ids.extend(zdb_to_acads.get(str(zid), []))
        acad_ids = list(set(acad_ids))

        groups = [acad_to_group.get(a, "Unknown") for a in acad_ids]
        groups = [g for g in groups if g not in {"Unknown", ""}]

        if len(set(groups)) == 0:
            g_list.append("Unknown")
        elif len(set(groups)) == 1:
            g_list.append(groups[0])
        else:
            g_list.append("Mixed")

    out[group_col] = g_list
    return out


# -----------------------------
# KM + log-rank (2 groups)
# -----------------------------
def km_curve(durations: np.ndarray, events: np.ndarray):
    """
    Returns: times, S, ci_lo, ci_hi, censor_times, censor_y
    CI uses log(-log) transform + Greenwood.
    """
    durations = durations.astype(int)
    events = events.astype(int)

    event_times = np.sort(np.unique(durations[events == 1]))
    times, S, se = [], [], []
    surv, var_cum = 1.0, 0.0

    for t in event_times:
        n_i = int((durations >= t).sum())
        d_i = int(((durations == t) & (events == 1)).sum())
        if n_i == 0:
            continue
        surv *= (1 - d_i / n_i)
        if n_i - d_i > 0:
            var_cum += d_i / (n_i * (n_i - d_i))
        times.append(int(t))
        S.append(float(surv))
        se.append(float(surv * math.sqrt(var_cum)))

    # CI (log(-log))
    z = 1.96
    S_arr = np.array(S)
    se_arr = np.array(se)
    eps = 1e-12
    S_clip = np.clip(S_arr, eps, 1 - eps)
    loglog = np.log(-np.log(S_clip))

    var_S = se_arr**2
    den = (S_clip**2) * (np.log(S_clip)**2)
    se_loglog = np.sqrt(np.clip(var_S / np.clip(den, eps, None), 0, None))

    ci_lo = np.exp(-np.exp(loglog + z * se_loglog))
    ci_hi = np.exp(-np.exp(loglog - z * se_loglog))

    # Censor ticks
    def surv_at(t):
        idx = bisect.bisect_right(times, t) - 1
        return 1.0 if idx < 0 else S[idx]

    censor_times = np.sort(durations[events == 0])
    censor_y = np.array([surv_at(int(t)) for t in censor_times])

    return times, S, ci_lo, ci_hi, censor_times, censor_y

def logrank_two_groups(dur_a, evt_a, dur_b, evt_b) -> float:
    """
    Mantel-Haenszel log-rank test (two-sided p-value).
    """
    # unique event times among both
    all_event_times = np.sort(np.unique(np.concatenate([dur_a[evt_a==1], dur_b[evt_b==1]])))

    O_minus_E = 0.0
    V = 0.0

    for t in all_event_times:
        # risk sets
        n1 = int((dur_a >= t).sum())
        n2 = int((dur_b >= t).sum())
        n = n1 + n2
        if n == 0:
            continue

        d1 = int(((dur_a == t) & (evt_a == 1)).sum())
        d2 = int(((dur_b == t) & (evt_b == 1)).sum())
        d = d1 + d2
        if d == 0:
            continue

        # expected in group1
        E1 = d * (n1 / n)

        # variance (hypergeometric)
        if n > 1:
            V1 = (n1 * n2 * d * (n - d)) / (n**2 * (n - 1))
        else:
            V1 = 0.0

        O_minus_E += (d1 - E1)
        V += V1

    if V <= 0:
        return np.nan

    z = O_minus_E / math.sqrt(V)
    # two-sided p using normal approx
    # p = 2*(1 - Phi(|z|))
    from math import erf, sqrt
    def phi(x):  # CDF standard normal
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    p = 2.0 * (1.0 - phi(abs(z)))
    return p

import numpy as np
import pandas as pd
import math
import bisect

# -----------------------------
# (i) Stratified log-rank by start_year cohorts
# -----------------------------
def make_start_year_strata(df: pd.DataFrame, col="start_year", q=5) -> pd.Series:
    """
    Create cohort strata using quantile bins of start_year.
    q=5 => quintiles (recommended: stable strata sizes).
    Returns a categorical Series like: Q1..Q5
    """
    x = df[col].astype(float)
    # pd.qcut can fail with many duplicates; allow drop duplicates
    strata = pd.qcut(x, q=q, duplicates="drop")
    # label for readability
    codes = strata.cat.codes
    k = int(codes.max() + 1)
    labels = [f"Q{i+1}" for i in range(k)]
    return pd.Categorical.from_codes(codes, categories=labels, ordered=True)

def stratified_logrank_two_groups(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    strata_col: str,
) -> float:
    """
    Mantel-Haenszel stratified log-rank test (two-sided p).
    Sum (O-E) and V across strata.
    """
    def phi(x):  # standard normal CDF
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    O_minus_E_total = 0.0
    V_total = 0.0

    for s, dfg in df.groupby(strata_col):
        a = dfg[dfg[group_col] == group_a]
        b = dfg[dfg[group_col] == group_b]
        if len(a) == 0 or len(b) == 0:
            continue

        dur_a = a["duration_years"].to_numpy().astype(int)
        evt_a = a["event_observed"].to_numpy().astype(int)
        dur_b = b["duration_years"].to_numpy().astype(int)
        evt_b = b["event_observed"].to_numpy().astype(int)

        # event times in this stratum
        all_event_times = np.sort(np.unique(np.concatenate([dur_a[evt_a==1], dur_b[evt_b==1]])))
        if all_event_times.size == 0:
            continue

        O_minus_E = 0.0
        V = 0.0

        for t in all_event_times:
            n1 = int((dur_a >= t).sum())
            n2 = int((dur_b >= t).sum())
            n = n1 + n2
            if n <= 1:
                continue

            d1 = int(((dur_a == t) & (evt_a == 1)).sum())
            d2 = int(((dur_b == t) & (evt_b == 1)).sum())
            d = d1 + d2
            if d == 0:
                continue

            E1 = d * (n1 / n)
            V1 = (n1 * n2 * d * (n - d)) / (n**2 * (n - 1))

            O_minus_E += (d1 - E1)
            V += V1

        O_minus_E_total += O_minus_E
        V_total += V

    if V_total <= 0:
        return np.nan

    z = O_minus_E_total / math.sqrt(V_total)
    p = 2.0 * (1.0 - phi(abs(z)))
    return p


# -----------------------------
# (ii) RMST (Restricted Mean Survival Time) up to tau
# -----------------------------
def km_survival_step(durations: np.ndarray, events: np.ndarray):
    """
    Build KM step function points:
    returns (times, S) where S is post-step survival at each event time.
    """
    durations = durations.astype(int)
    events = events.astype(int)

    event_times = np.sort(np.unique(durations[events == 1]))
    times, S = [], []
    surv = 1.0

    for t in event_times:
        n_i = int((durations >= t).sum())
        d_i = int(((durations == t) & (events == 1)).sum())
        if n_i == 0:
            continue
        surv *= (1 - d_i / n_i)
        times.append(int(t))
        S.append(float(surv))

    return times, S


def km_step_value_at(times: List[int], S: List[float], x: int) -> float:
    """Evaluate right-continuous KM step function at time x (years)."""
    if times is None or S is None or len(times) == 0:
        return 1.0
    # Find last index with times[idx] <= x
    idx = int(np.searchsorted(np.asarray(times, dtype=float), float(x), side="right") - 1)
    if idx < 0:
        return 1.0  # convention consistent with rmst_from_km()
    idx = min(idx, len(S) - 1)
    s = float(S[idx])
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s

def km_S_at(durations: np.ndarray, events: np.ndarray, at_times: Tuple[int, int, int] = (25, 50, 100)) -> Dict[str, float]:
    """Compute KM survival probabilities S(t) at selected integer times."""
    t_step, s_step = km_survival_step(durations.astype(int), events.astype(int))
    out = {}
    for t in at_times:
        out[f"S({int(t)})"] = km_step_value_at(t_step, s_step, int(t))
    return out

def rmst_from_km(durations: np.ndarray, events: np.ndarray, tau: int = 100) -> float:
    """
    RMST(tau) = integral_0^tau S(t) dt using KM step function.
    Convention: S(t)=1 for t<first event time.
    """
    times, S = km_survival_step(durations, events)
    if tau <= 0:
        return 0.0
    if len(times) == 0:
        return float(tau)  # no events observed -> survival ~ 1 across [0,tau]

    # integrate piecewise on intervals [0, t1), [t1, t2), ... , [tk, tau]
    rmst = 0.0
    prev_t = 0
    prev_s = 1.0

    for t, s in zip(times, S):
        if t >= tau:
            rmst += (tau - prev_t) * prev_s
            return rmst
        rmst += (t - prev_t) * prev_s
        prev_t = t
        prev_s = s

    # tail to tau
    rmst += (tau - prev_t) * prev_s
    return float(rmst)

def bootstrap_rmst_diff(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    tau: int = 100,
    B: int = 800,
    seed: int = 42,
) -> dict:
    """
    Bootstrap RMST difference (A - B) with percentile 95% CI.
    """
    rng = np.random.default_rng(seed)

    a = df[df[group_col] == group_a].copy()
    b = df[df[group_col] == group_b].copy()

    dur_a = a["duration_years"].to_numpy().astype(int)
    evt_a = a["event_observed"].to_numpy().astype(int)
    dur_b = b["duration_years"].to_numpy().astype(int)
    evt_b = b["event_observed"].to_numpy().astype(int)

    rmst_a = rmst_from_km(dur_a, evt_a, tau=tau)
    rmst_b = rmst_from_km(dur_b, evt_b, tau=tau)
    diff = rmst_a - rmst_b

    # bootstrap
    diffs = []
    na, nb = len(a), len(b)
    for _ in range(B):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)

        diffs.append(
            rmst_from_km(dur_a[ia], evt_a[ia], tau=tau) -
            rmst_from_km(dur_b[ib], evt_b[ib], tau=tau)
        )

    diffs = np.array(diffs, dtype=float)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # optional: bootstrap-based two-sided p (sign test style)
    p_boot = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    return {
        "tau": tau,
        "rmst_a": rmst_a,
        "rmst_b": rmst_b,
        "rmst_diff": diff,
        "rmst_diff_ci_lo": float(lo),
        "rmst_diff_ci_hi": float(hi),
        "rmst_diff_p_boot": float(p_boot),
        "B": B,
    }


# -----------------------------
# One-shot summary table for paper/appendix
# -----------------------------
def summarize_two_group_survival_with_rmst(
    df: pd.DataFrame,
    comparison: str,
    group_col: str,
    group_a: str,
    group_b: str,
    tau: int = 100,
    strata_q: int = 5,
) -> pd.Series:
    # basic counts
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]

    # median (KM)
    def median_km(durations, events):
        times, S = km_survival_step(durations, events)
        for t, s in zip(times, S):
            if s <= 0.5:
                return float(t)
        return np.nan

    med_a = median_km(a["duration_years"].to_numpy(), a["event_observed"].to_numpy())
    med_b = median_km(b["duration_years"].to_numpy(), b["event_observed"].to_numpy())

    # KM point estimates at key horizons (years)
    Sa = km_S_at(a["duration_years"].to_numpy(), a["event_observed"].to_numpy(), at_times=(25, 50, 100))
    Sb = km_S_at(b["duration_years"].to_numpy(), b["event_observed"].to_numpy(), at_times=(25, 50, 100))

    # unstratified log-rank p: reuse your existing function if already defined;
    # otherwise keep the p you already printed, or plug in your logrank function here.
    # Here we compute quickly using stratified function with one stratum as fallback:
    df_tmp = df.copy()
    df_tmp["_stratum"] = "All"
    p_lr = stratified_logrank_two_groups(df_tmp, group_col, group_a, group_b, "_stratum")

    # stratified by cohort
    df2 = df.copy()
    df2["cohort_stratum"] = make_start_year_strata(df2, col="start_year", q=strata_q)
    p_strat = stratified_logrank_two_groups(df2, group_col, group_a, group_b, "cohort_stratum")

    # RMST + bootstrap CI
    rmst = bootstrap_rmst_diff(df, group_col, group_a, group_b, tau=tau)

    return pd.Series({
        "Comparison": comparison,
        "Group_A": group_a,
        "Group_B": group_b,
        "N_A": int(len(a)),
        "Events_A": int(a["event_observed"].sum()),
        "Censored_A": int((a["event_observed"] == 0).sum()),
        "Median_A_years": med_a,
        "S25_A": Sa["S(25)"],
        "S50_A": Sa["S(50)"],
        "S100_A": Sa["S(100)"],
        "N_B": int(len(b)),
        "Events_B": int(b["event_observed"].sum()),
        "Censored_B": int((b["event_observed"] == 0).sum()),
        "Median_B_years": med_b,
        "S25_B": Sb["S(25)"],
        "S50_B": Sb["S(50)"],
        "S100_B": Sb["S(100)"],
        "Logrank_p": p_lr,
        f"StratifiedLogrank_p_(start_year_q{strata_q})": p_strat,
        f"RMST_A_(tau={tau})": rmst["rmst_a"],
        f"RMST_B_(tau={tau})": rmst["rmst_b"],
        f"RMST_Diff_A-B_(tau={tau})": rmst["rmst_diff"],
        f"RMST_Diff_CI95_Lo": rmst["rmst_diff_ci_lo"],
        f"RMST_Diff_CI95_Hi": rmst["rmst_diff_ci_hi"],
        f"RMST_Diff_p_boot": rmst["rmst_diff_p_boot"],
        "RMST_boot_B": rmst["B"],
    })


def export_paper_table(df_out: pd.DataFrame, out_csv: str, out_tex: str):
    df_out.to_csv(out_csv, index=False)

    df_tex = df_out.copy()

    def to_float_or_nan(x):
        # 把 str/object 尽量转成 float；失败就 NaN
        try:
            if pd.isna(x):
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    for c in df_tex.columns:
        cl = c.lower()

        # p-value 列：安全格式化
        if "p" in cl:
            vals = df_tex[c].apply(to_float_or_nan)
            df_tex[c] = vals.apply(lambda v: f"{v:.3g}" if pd.notna(v) else "")

        # RMST 数值列：安全格式化（但别动 RMST 的 p 列）
        if ("rmst" in cl) and ("p" not in cl):
            vals = df_tex[c].apply(to_float_or_nan)
            df_tex[c] = vals.apply(lambda v: f"{v:.1f}" if pd.notna(v) else "")

        # Median 列：安全格式化
        if "median" in cl:
            vals = df_tex[c].apply(to_float_or_nan)
            df_tex[c] = vals.apply(lambda v: f"{v:.0f}" if pd.notna(v) else "")

    latex = df_tex.to_latex(index=False, escape=True, longtable=False, bold_rows=False)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex)



def export_table4_main_authorea(df_out: pd.DataFrame, out_tex: str, tau: int = 100):
    """
    Export main-text Table 4 in an Authorea-friendly LaTeX form:
    - No threeparttable/tablenotes (often hidden in Authorea HTML preview)
    - Notes are placed in a minipage below the tabular
    - Uses S25/S50/S100 columns computed in summarize_two_group_survival_with_rmst()
    """
    def fmt_prob(x):
        try:
            if pd.isna(x):
                return "--"
            x = float(x)
            x = min(max(x, 0.0), 1.0)
            return f"{x:.2f}"
        except Exception:
            return "--"

    def fmt_p(x):
        try:
            if pd.isna(x):
                return "--"
            x = float(x)
            if x < 0.001:
                return r"$<0.001$"
            return f"{x:.3f}"
        except Exception:
            return "--"

    def fmt_num(x, nd=1):
        try:
            if pd.isna(x):
                return "--"
            return f"{float(x):.{nd}f}"
        except Exception:
            return "--"

    # pick the two intended comparisons
    eng = df_out[df_out["Comparison"].str.contains("Engineering", case=False, na=False)].iloc[0]
    ns  = df_out[df_out["Comparison"].str.contains("Global North", case=False, na=False)].iloc[0]

    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Survival summary of hosted journals.}",
        r"\label{tab:survival_summary}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Group & $N$ & Events & Censored & Median & $S(25)$ & $S(50)$ & $S(100)$ & RMST$(100)$ \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\textbf{Panel A. Engineering vs.\ non-engineering}}\\",
        f"Engineering & {int(eng['N_A'])} & {int(eng['Events_A'])} & {int(eng['Censored_A'])} & {fmt_num(eng['Median_A_years'],0)} & {fmt_prob(eng['S25_A'])} & {fmt_prob(eng['S50_A'])} & {fmt_prob(eng['S100_A'])} & {fmt_num(eng[f'RMST_A_(tau={tau})'],1)} \\",
        f"Non-engineering & {int(eng['N_B'])} & {int(eng['Events_B'])} & {int(eng['Censored_B'])} & {fmt_num(eng['Median_B_years'],0)} & {fmt_prob(eng['S25_B'])} & {fmt_prob(eng['S50_B'])} & {fmt_prob(eng['S100_B'])} & {fmt_num(eng[f'RMST_B_(tau={tau})'],1)} \\",
        r"\addlinespace[2pt]",
        f"\multicolumn{{8}}{{l}}{{Log-rank $p$}} & {fmt_p(eng['Logrank_p'])} \\",
        f"\multicolumn{{8}}{{l}}{{Stratified log-rank $p^{{\mathrm{{a}}}}$}} & {fmt_p(eng[f'StratifiedLogrank_p_(start_year_q5)'])} \\",
        f"\multicolumn{{8}}{{l}}{{RMST diff (Engineering--Non-engineering)}} & {fmt_num(eng[f'RMST_Diff_A-B_(tau={tau})'],1)} [{fmt_num(eng['RMST_Diff_CI95_Lo'],1)}, {fmt_num(eng['RMST_Diff_CI95_Hi'],1)}]; boot $p$={fmt_p(eng['RMST_Diff_p_boot'])} \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\textbf{Panel B. Global North vs.\ Global South}}\\",
        f"Global North (Developed) & {int(ns['N_A'])} & {int(ns['Events_A'])} & {int(ns['Censored_A'])} & {fmt_num(ns['Median_A_years'],0)} & {fmt_prob(ns['S25_A'])} & {fmt_prob(ns['S50_A'])} & {fmt_prob(ns['S100_A'])} & {fmt_num(ns[f'RMST_A_(tau={tau})'],1)} \\",
        f"Global South (Developing) & {int(ns['N_B'])} & {int(ns['Events_B'])} & {int(ns['Censored_B'])} & {fmt_num(ns['Median_B_years'],0)} & {fmt_prob(ns['S25_B'])} & {fmt_prob(ns['S50_B'])} & {fmt_prob(ns['S100_B'])} & {fmt_num(ns[f'RMST_B_(tau={tau})'],1)} \\",
        r"\addlinespace[2pt]",
        f"\multicolumn{{8}}{{l}}{{Log-rank $p$}} & {fmt_p(ns['Logrank_p'])} \\",
        f"\multicolumn{{8}}{{l}}{{Stratified log-rank $p^{{\mathrm{{a}}}}$}} & {fmt_p(ns[f'StratifiedLogrank_p_(start_year_q5)'])} \\",
        f"\multicolumn{{8}}{{l}}{{RMST diff (Global North--Global South)}} & {fmt_num(ns[f'RMST_Diff_A-B_(tau={tau})'],1)} [{fmt_num(ns['RMST_Diff_CI95_Lo'],1)}, {fmt_num(ns['RMST_Diff_CI95_Hi'],1)}]; boot $p$={fmt_p(ns['RMST_Diff_p_boot'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize",
        r"\textit{Notes:} Median and RMST are reported in years. $S(t)$ denotes the Kaplan--Meier survival probability at year $t$. RMST$(100)$ is the restricted mean survival time truncated at $\tau=100$ years. $^{\mathrm{a}}$ Stratified by journal start-year quintiles (as in the stratified log-rank test).",
        r"\end{minipage}",
        r"\end{table}",
    ]

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def median_survival(times, S) -> float:
    for t, s in zip(times, S):
        if s <= 0.5:
            return float(t)
    return np.nan


# -----------------------------
# Plotting (no legend; caption carries "95% CI")
# -----------------------------
def _step_survival_at(t_query: float, times: list[float], S: list[float]) -> float:
    """
    Return KM step survival S(t_query) given step points (times, S) produced by km_curve().
    times: increasing event times (t>0), S: survival values at each time (post-step).
    """
    if len(times) == 0:
        return 1.0
    if t_query < times[0]:
        return 1.0
    # rightmost index with times[i] <= t_query
    i = bisect.bisect_right(times, t_query) - 1
    i = max(0, min(i, len(S) - 1))
    return float(S[i])


def _median_from_km(times: np.ndarray, S: np.ndarray) -> float:
    """Return KM median time (first t with S(t) <= 0.5). np.nan if never crosses 0.5."""
    times = np.asarray(times, dtype=float)
    S = np.asarray(S, dtype=float)
    if times.size == 0 or S.size == 0:
        return np.nan
    idx = np.where(S <= 0.5)[0]
    if idx.size == 0:
        return np.nan
    return float(times[int(idx[0])])


def _format_median(m: float) -> str:
    return "NR" if (m is None or np.isnan(m)) else f"{m:.0f}"


def _style_km_axes(ax: plt.Axes):
    """Top-journal grayscale style (print-safe)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.4, length=5)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.set_ylim(0, 1.0)


def _nice_xmax(durations: np.ndarray, pad: int = 10) -> int:
    if durations.size == 0:
        return 100
    mx = int(np.nanmax(durations))
    mx = max(0, mx + pad)
    # round up to a "nice" tick end
    for end in [100, 120, 150, 180, 200, 250, 300]:
        if mx <= end:
            return end
    return int(math.ceil(mx / 50.0) * 50)


def draw_km_overall(ax: plt.Axes, durations: np.ndarray, events: np.ndarray):
    """Draw overall KM (Fig 5a) onto an axis."""
    times, S, lo, hi, c_t, c_y = km_curve(durations, events)

    # main curve + CI (subtle)
    ax.fill_between(times, lo, hi, step="post", alpha=0.10, color="0.6", linewidth=0)
    ax.step(times, S, where="post", color="0.0", linewidth=2.8)

    # censor ticks
    if len(c_t) > 0:
        ax.plot(c_t, c_y, linestyle="None", marker="|", color="0.0",
                markersize=9, markeredgewidth=1.2)

    # median reference (horizontal + vertical at median)
    med = _median_from_km(times, S)
    ax.axhline(0.5, color="0.25", linestyle=(0, (6, 2)), linewidth=1.6)
    if not np.isnan(med):
        ax.axvline(med, color="0.25", linestyle=(0, (6, 2)), linewidth=1.6)
        ax.text(med + 1.2, 0.53, f"Median = {int(round(med))} years", fontsize=14,
                ha="left", va="center", color="0.0")

    # axes
    _style_km_axes(ax)
    ax.set_ylabel("Survival probability", fontsize=16)
    ax.set_xlabel("Years", fontsize=16)
    xmax = _nice_xmax(durations)
    ax.set_xlim(0, xmax)


def draw_km_two_groups(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    tau: int = 100,
):
    """
    Draw 2-group KM (Fig 5b/5c) with:
    - grayscale print-safe lines (solid vs long-dash + sparse marker)
    - subtle CI bands
    - direct labels near curves
    - stats box: log-rank p, RMST(τ) Δ, Median(A/B)
    - median reference lines: y=0.5; verticals at each group's median (matching line style)
    """
    # split
    dfa = df[df[group_col] == group_a].copy()
    dfb = df[df[group_col] == group_b].copy()

    dur_a, evt_a = get_duration_event_arrays(dfa)
    dur_b, evt_b = get_duration_event_arrays(dfb)

    # curves
    ta, Sa, la, ha, cta, cya = km_curve(dur_a, evt_a)
    tb, Sb, lb, hb, ctb, cyb = km_curve(dur_b, evt_b)

    # CI (subtle)
    ax.fill_between(ta, la, ha, step="post", alpha=0.10, color="0.75", linewidth=0)
    ax.fill_between(tb, lb, hb, step="post", alpha=0.10, color="0.90", linewidth=0)

    # line styles (highly distinguishable in grayscale)
    lw = 2.8
    dash = (0, (12, 6))   # long dash
    ax.step(ta, Sa, where="post", color="0.0", linewidth=lw, linestyle="-")
    ax.step(tb, Sb, where="post", color="0.0", linewidth=lw, linestyle=dash)

    # optional sparse markers for the dashed line to further separate
    if len(tb) > 0:
        step = max(1, len(tb) // 12)
        ax.plot(tb[::step], Sb[::step], linestyle="None", marker="o",
                markersize=3.8, markeredgewidth=0.0, color="0.0", alpha=0.9)

    # censor ticks
    if len(cta) > 0:
        ax.plot(cta, cya, linestyle="None", marker="|", color="0.0",
                markersize=8, markeredgewidth=1.2)
    if len(ctb) > 0:
        ax.plot(ctb, cyb, linestyle="None", marker="|", color="0.0",
                markersize=8, markeredgewidth=1.2)

    # medians + reference line
    ax.axhline(0.5, color="0.25", linestyle=(0, (6, 2)), linewidth=1.4)
    med_a = _median_from_km(ta, Sa)
    med_b = _median_from_km(tb, Sb)
    if not np.isnan(med_a):
        ax.axvline(med_a, color="0.25", linestyle="-", linewidth=1.4)
    if not np.isnan(med_b):
        ax.axvline(med_b, color="0.25", linestyle=dash, linewidth=1.4)

    # statistics
    p_lr = logrank_two_groups(dur_a, evt_a, dur_b, evt_b)
    lr = {'p': p_lr}
    rmst_a = rmst_from_km(dur_a, evt_a, tau=tau)
    rmst_b = rmst_from_km(dur_b, evt_b, tau=tau)
    rmst_diff = rmst_a - rmst_b

    stats_txt = (
        f"Log-rank p = {lr['p']:.3g}\n"
        f"RMST(τ={tau}) Δ = {rmst_diff:+.1f} years\n"
        f"Median: {_format_median(med_a)} vs {_format_median(med_b)} years"
    )
    ax.text(0.98, 0.88, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color="0.0")

    # direct labels near end of curves
    def _label_end(times, S, label, dy=0.0):
        if len(times) == 0:
            return
        x = times[-1]
        y = S[-1] + dy
        ax.text(x + 2, y, label, fontsize=16, ha="left", va="center", color="0.0")

    _label_end(ta, Sa, group_a, dy=0.08)
    _label_end(tb, Sb, group_b, dy=-0.08)

    # axes formatting
    _style_km_axes(ax)
    ax.set_xlabel("Years", fontsize=16)
    xmax = _nice_xmax(np.concatenate([dur_a, dur_b]))
    ax.set_xlim(0, xmax)

    return {
        "logrank_p": lr["p"],
        "rmst_a": rmst_a,
        "rmst_b": rmst_b,
        "rmst_diff_a_minus_b": rmst_diff,
        "median_a": med_a,
        "median_b": med_b,
        "n_a": int(len(dur_a)),
        "n_b": int(len(dur_b)),
    }


def plot_figure5a_overall(lineages: pd.DataFrame, out_prefix: str = "Figure5a_lineage_KM_clean"):
    durations, events = get_duration_event_arrays(lineages)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    draw_km_overall(ax, durations, events)
    ax.set_title("")  # caption carries title

    out_png = OUT_DIR / f"{out_prefix}.png"
    out_pdf = OUT_DIR / f"{out_prefix}.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_km_two_groups(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    title: str,
    out_prefix: str,
):
    # single-panel wrapper (keeps previous API)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    stats = draw_km_two_groups(ax, df, group_col, group_a, group_b, tau=100)
    ax.set_ylabel("Survival probability", fontsize=16)
    ax.set_title("")  # caption carries title
    fig.tight_layout()

    out_png = OUT_DIR / f"{out_prefix}.png"
    out_pdf = OUT_DIR / f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    return stats


def plot_figure5_combined(
    lineages: pd.DataFrame,
    base_eng: pd.DataFrame,
    base_dev: pd.DataFrame,
    out_prefix: str = "Figure5_abc_combined",
):
    """Three panels (a/b/c) combined into a single figure."""
    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    # a: overall
    durations, events = get_duration_event_arrays(lineages)
    draw_km_overall(ax1, durations, events)
    ax1.set_xlabel("Years", fontsize=16)
    ax1.set_ylabel("Survival probability", fontsize=16)

    # b: engineering
    draw_km_two_groups(ax2, base_eng, "eng_group", "Engineering", "Non-engineering", tau=100)
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False)

    # c: north-south
    draw_km_two_groups(ax3, base_dev, "dev_group", "Developing", "Developed", tau=100)
    ax3.set_ylabel("")
    ax3.tick_params(labelleft=False)

    # panel labels
    for ax, lab in zip([ax1, ax2, ax3], ["a", "b", "c"]):
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=20, fontweight="bold", color="0.0")

    fig.tight_layout()

    out_png = OUT_DIR / f"{out_prefix}.png"
    out_pdf = OUT_DIR / f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    acad_attr = build_academy_attributes()
    zdb2acad = load_zdb_to_academy_map()
    lineages = build_lineages(CENSOR_YEAR=2025)

    # Assign groups
    lineages = assign_lineage_group(lineages, zdb2acad, acad_attr, "eng_group")
    lineages = assign_lineage_group(lineages, zdb2acad, acad_attr, "dev_group")

    # Drop mixed/unknown for clean two-group tests
    # 只保留干净两组（与你当前图一致）
    base_eng = lineages[lineages["eng_group"].isin(["Engineering", "Non-engineering"])].copy()
    base_dev = lineages[lineages["dev_group"].isin(["Developed", "Developing"])].copy()

    # Figure 5a: overall lineage survival
    plot_figure5a_overall(lineages, out_prefix="Figure5a_lineage_KM_clean")


    rows = []
    rows.append(summarize_two_group_survival_with_rmst(
        df=base_eng,
        comparison="Engineering vs Non-engineering",
        group_col="eng_group",
        group_a="Engineering",
        group_b="Non-engineering",
        tau=100,
        strata_q=5,   # start_year quintiles
    ))

    rows.append(summarize_two_group_survival_with_rmst(
        df=base_dev,
        comparison="Global North vs Global South (Developed vs Developing)",
        group_col="dev_group",
        group_a="Developed",
        group_b="Developing",
        tau=100,
        strata_q=5,
    ))

    table = pd.DataFrame(rows)

    export_paper_table(
        table,
        out_csv=str(OUT_DIR / "Figure5_survival_stratlogrank_rmst_table.csv"),
        out_tex=str(OUT_DIR / "Figure5_survival_stratlogrank_rmst_table.tex"),
    )

    # Main-text Table 4 (Authorea-friendly LaTeX; includes S(25)/S(50)/S(100))
    export_table4_main_authorea(
        table,
        out_tex=str(OUT_DIR / "Table4_survival_summary_main.tex"),
        tau=100,
    )

    print(table)

    summary = []

    # 5b Engineering vs Non-engineering
    summary.append(plot_km_two_groups(
        df=base_eng,
        group_col="eng_group",
        group_a="Engineering",
        group_b="Non-engineering",
        title="Journal lineage survival: Engineering vs Non-engineering (Kaplan–Meier)",
        out_prefix="Figure5b_KM_engineering_vs_nonengineering",
    ))

    # 5c North vs South (Developed vs Developing)
    summary.append(plot_km_two_groups(
        df=base_dev,
        group_col="dev_group",
        group_a="Developed",
        group_b="Developing",
        title="Journal lineage survival: Global North vs Global South (Kaplan–Meier)",
        out_prefix="Figure5c_KM_north_vs_south",
    ))

    # Combined Figure 5a/5b/5c
    plot_figure5_combined(lineages, base_eng, base_dev, out_prefix="Figure5_abc_combined")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUT_DIR / "Figure5_group_survival_summary.csv", index=False)

    print("[INFO] Done. Outputs in:", OUT_DIR.resolve())
    print(sum_df)


if __name__ == "__main__":
    main()