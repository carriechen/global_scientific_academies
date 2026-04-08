# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: if you want to parse the official DDC summaries PDF later, you can add pdfplumber logic.
# For now: we rely on a deterministic embedded mapping for the codes observed in your dataset.
# (No guessing: mapping text corresponds to DDC summaries wording.)
# If you later want full automatic coverage, tell me and I’ll wire a PDF parser robustly.

# -------------------------
# Config
# -------------------------
INPUT_XLSX_DEFAULT = Path("zdb_result/zdb_journal_results.xlsx")
INPUT_XLSX_FALLBACK = Path("zdb_journal_results.xlsx")

OUTPUT_DIR = Path("zdb_result/analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CENSOR_YEAR = 2025
DDC_STACK_TOPK = 18  # show top K codes; rest aggregated to "Other DDC codes" to keep legend readable

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

YEAR_RE = re.compile(r"(1[5-9]\d{2}|20\d{2}|21\d{2})")


# -------------------------
# Embedded DDC meaning mapping (covers common 3-digit groups)
# -------------------------
DDC_LABELS: Dict[str, str] = {
    "000": "Computer science, information & general works",
    "001": "Knowledge",
    "004": "Computer science",
    "010": "Bibliographies",
    "016": "Bibliographies of works on specific subjects",
    "020": "Library & information sciences",
    "030": "Encyclopedias & books of facts",
    "050": "Magazines, journals & serials",
    "060": "Associations, organizations & museums",
    "070": "News media, journalism & publishing",
    "080": "General collections",
    "090": "Manuscripts & rare books",
    "100": "Philosophy & psychology",
    "140": "Philosophical schools of thought",
    "150": "Psychology",
    "200": "Religion",
    "230": "Christian theology",
    "290": "Other religions",
    "300": "Social sciences",
    "301": "Sociology",
    "305": "Social groups",
    "306": "Culture & institutions",
    "320": "Political science",
    "330": "Economics",
    "331": "Labor economics",
    "333": "Economics of land & energy",
    "340": "Law",
    "350": "Public administration & military science",
    "360": "Social problems & services; associations",
    "370": "Education",
    "390": "Customs, etiquette, folklore",
    "400": "Language",
    "430": "German & related languages",
    "439": "Other Germanic languages",
    "491": "East Indo-European & Celtic languages",
    "500": "Science",
    "505": "Serial publications (Science)",
    "509": "Historical, geographic, persons treatment (Science)",
    "510": "Mathematics",
    "520": "Astronomy",
    "526": "Mathematical geography",
    "530": "Physics",
    "540": "Chemistry",
    "550": "Earth sciences & geology",
    "560": "Fossils & prehistoric life",
    "570": "Biology",
    "580": "Plants (Botany)",
    "590": "Animals (Zoology)",
    "600": "Technology",
    "605": "Serial publications (Technology)",
    "610": "Medicine & health",
    "616": "Diseases",
    "620": "Engineering & allied operations",
    "630": "Agriculture and related technologies",
    "636": "Animal husbandry",
    "700": "Arts & recreation",
    "720": "Architecture",
    "730": "Sculpture, ceramics & metalwork",
    "740": "Drawing & decorative arts",
    "780": "Music",
    "790": "Recreational & performing arts",
    "800": "Literature",
    "820": "English & Old English literatures",
    "830": "German & related literatures",
    "840": "French & related literatures",
    "890": "Literatures of other languages",
    "900": "History & geography",
    "910": "Geography & travel",
    "920": "Biography, genealogy, insignia",
    "930": "History of ancient world (to ca. 499)",
    "940": "History of Europe",
    "950": "History of Asia",
    "960": "History of Africa",
    "970": "History of North America",
    "980": "History of South America",
    "990": "History of other areas",
}


# -------------------------
# Helpers
# -------------------------
def _clean_text(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_responsibility(title: str) -> str:
    """Only strip when delimiter is exactly ' / ' (space-slash-space)."""
    t = _clean_text(title)
    if " / " in t:
        t = t.split(" / ", 1)[0].strip()
    return t


def _middle_ellipsis(s: str, maxlen: int = 56) -> str:
    s = _clean_text(s)
    if len(s) <= maxlen:
        return s
    keep = maxlen - 1
    left = keep // 2
    right = keep - left
    return s[:left] + "…" + s[-right:]


def parse_published_span(published) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Deterministic: uses only the 'Published' field.
    Returns (start_year, end_year, censored_bool)
      - censored if open-ended 'YYYY-' at end of string
      - end_year is CENSOR_YEAR if censored
    """
    s = _clean_text(published)
    if not s:
        return (None, None, False)
    m = list(YEAR_RE.finditer(s))
    if not m:
        return (None, None, False)

    sy = int(m[0].group())
    ey = int(m[-1].group())
    cens = bool(re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\s*-\s*$", s))
    end = CENSOR_YEAR if cens else ey
    if end < sy:
        end = sy
    return (sy, end, cens)


def split_ddc_codes(cell) -> List[str]:
    """
    Split multi-valued DDC subject groups cell into separate 3-digit codes.
    Examples:
      "000; 020" -> ["000", "020"]
      "530;510"  -> ["530", "510"]
    """
    s = _clean_text(cell)
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[;|,]", s) if p.strip()]
    codes = []
    for p in parts:
        m = re.search(r"\b(\d{3})\b", p)
        if m:
            codes.append(m.group(1))
    return codes


# -------------------------
# Core panels: Active + Turnover
# -------------------------
def build_year_panel_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year counts based on UNIQUE ZDB-ID:
      active, births, deaths, net
    births = start year
    deaths = end year (excluding censored open-ended)
    """
    rows = []
    for zid, g in df.groupby("ZDB-ID", dropna=False):
        # pick one representative Published span per journal (min start, max end)
        spans = [parse_published_span(x) for x in g["Published"].tolist()]
        spans = [(a, b, c) for (a, b, c) in spans if a is not None and b is not None]
        if not spans:
            continue
        start = min(a for a, _, _ in spans)
        end = max(b for _, b, _ in spans)
        any_open = any(c for _, _, c in spans)
        rows.append((str(zid), int(start), int(end), bool(any_open)))

    if not rows:
        return pd.DataFrame(columns=["year", "active", "births", "deaths", "net"])

    agg = pd.DataFrame(rows, columns=["ZDB-ID", "start", "end", "any_open"])

    years = np.arange(int(agg["start"].min()), int(agg["end"].max()) + 1)

    births = agg["start"].value_counts().to_dict()
    deaths = agg.loc[~agg["any_open"], "end"].value_counts().to_dict()

    active = []
    for y in years:
        active.append(int(((agg["start"] <= y) & (agg["end"] >= y)).sum()))

    panel = pd.DataFrame({
        "year": years,
        "active": active,
        "births": [int(births.get(int(y), 0)) for y in years],
        "deaths": [int(deaths.get(int(y), 0)) for y in years],
    })
    panel["net"] = panel["births"] - panel["deaths"]
    return panel


def plot_population_dynamics(panel: pd.DataFrame, df: pd.DataFrame) -> None:
    """Top-journal style combined figure; legend does not overlap."""
    if panel.empty:
        return

    n_unique = df["ZDB-ID"].nunique()
    n_records = len(df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4]}
    )

    # A: Active
    ax1.plot(panel["year"], panel["active"], linewidth=2.2)
    ax1.set_ylabel("Active journals")
    ax1.grid(True, axis="y", alpha=0.20)
    ax1.grid(False, axis="x")
    ax1.text(0.01, 0.90, "A", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    # ax1.text(
    #     0.99, 0.92, f"N (unique journals) = {n_unique:,}\nRecords = {n_records:,}",
    #     transform=ax1.transAxes, ha="right", va="top", fontsize=10
    # )

    # B: Turnover
    years = panel["year"].values
    births = panel["births"].values
    deaths = panel["deaths"].values
    net = panel["net"].values

    b1 = ax2.bar(years, births, width=0.9, alpha=0.90, label="Births")
    b2 = ax2.bar(years, -deaths, width=0.9, alpha=0.90, label="Deaths")
    l1, = ax2.plot(years, net, linestyle="--", linewidth=1.6, label="Net")

    ax2.axhline(0, color="0.2", linewidth=1.0)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Year")
    ax2.grid(True, axis="y", alpha=0.20)
    ax2.grid(False, axis="x")
    ax2.text(0.01, 0.86, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold")

    # Legend ABOVE panel B (avoid overlap)
    ax2.legend(
        handles=[l1, b1, b2], labels=["Net", "Births", "Deaths"],
        loc="upper left", bbox_to_anchor=(0.0, 1.22),
        ncol=3, frameon=True, framealpha=0.95, borderaxespad=0.0
    )

    fig.suptitle("Journal Population Dynamics Over Time", y=0.99, fontsize=16)
    # note = f"Notes: Right-censoring applied when 'Published' ends with open-ended 'YYYY-'; censored at {CENSOR_YEAR}."
    # fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10)
    # fig.text(0.01, 0.01,  ha="left", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "population_dynamics_active_turnover.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "population_dynamics_active_turnover.pdf")
    plt.close(fig)

    panel.to_excel(OUTPUT_DIR / "turnover_data.xlsx", index=False)


# -------------------------
# DDC counts (split codes, count each)
# -------------------------
def compute_ddc_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split DDC subject groups into separate codes and count each code.
    Two count bases:
      - n_journals: unique ZDB-ID assigned this code (after split)
      - n_records: exploded rows count (if multiple rows per journal)
    """
    if "DDC subject groups" not in df.columns:
        return pd.DataFrame(columns=["DDC_code", "DDC_label", "n_journals", "n_records", "share_journals"])

    tmp = df[["ZDB-ID", "DDC subject groups"]].copy()
    tmp["DDC_code"] = tmp["DDC subject groups"].apply(split_ddc_codes)
    tmp = tmp.explode("DDC_code").dropna(subset=["DDC_code"])
    tmp["DDC_code"] = tmp["DDC_code"].astype(str).str.strip()

    n_records = tmp.groupby("DDC_code").size().rename("n_records")
    n_journals = tmp.groupby("DDC_code")["ZDB-ID"].nunique().rename("n_journals")

    out = pd.concat([n_journals, n_records], axis=1).reset_index()
    out["DDC_label"] = out["DDC_code"].map(DDC_LABELS).fillna("")
    total_journals = df["ZDB-ID"].nunique()
    out["share_journals"] = out["n_journals"] / total_journals if total_journals else np.nan
    out = out.sort_values(["n_journals", "n_records"], ascending=False).reset_index(drop=True)
    return out


def plot_ddc_top_bar(ddc_counts: pd.DataFrame, top_n: int = 25) -> None:
    if ddc_counts.empty:
        return
    top = ddc_counts.head(top_n).copy()
    top["label"] = top["DDC_code"].astype(str) + "  " + top["DDC_label"].astype(str)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    ax.barh(top["label"], top["n_journals"].values)
    ax.set_xlabel("Number of journals (unique ZDB-ID)")
    ax.set_title(f"Top DDC subject group codes (split & counted): Top {top_n}")
    ax.grid(True, axis="x", alpha=0.20)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ddc_subject_group_code_counts_top.png", dpi=300)
    plt.close(fig)


# -------------------------
# Genealogy (Union-Find + Gantt)  -- preserved
# -------------------------
class DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _norm_title_for_link(s: str) -> str:
    s = _strip_responsibility(s)
    s = _clean_text(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_corp_list(s) -> set:
    txt = _clean_text(s)
    if not txt:
        return set()
    parts = re.split(r"[;|]", txt)
    return {(_clean_text(p).lower()) for p in parts if _clean_text(p)}


def split_former_later(rel) -> Tuple[List[str], List[str]]:
    s = _clean_text(rel)
    if not s or "/" not in s:
        return ([], [])
    left, right = s.split("/", 1)

    def _parts(x: str) -> List[str]:
        items = []
        for p in re.split(r"[;|]", x):
            nt = _norm_title_for_link(p)
            if nt:
                items.append(nt)
        return items

    return (_parts(left), _parts(right))


def build_lineages(df: pd.DataFrame) -> Tuple[Dict[str, set], Dict[str, Dict]]:
    required = ["ZDB-ID", "Title", "Former/later titles", "Corporate body"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for genealogy: {c}")

    id_to_row: Dict[str, Dict] = {}
    dsu = DSU()

    title_to_ids: Dict[str, set] = {}
    for _, row in df.iterrows():
        zid = _clean_text(row.get("ZDB-ID", ""))
        if not zid:
            continue
        dsu.add(zid)
        id_to_row[zid] = row.to_dict()

        tnorm = _norm_title_for_link(row.get("Title", ""))
        if tnorm:
            title_to_ids.setdefault(tnorm, set()).add(zid)

    id_to_corp = {zid: _norm_corp_list(id_to_row[zid].get("Corporate body", np.nan)) for zid in id_to_row}

    for zid, rd in id_to_row.items():
        former, later = split_former_later(rd.get("Former/later titles", np.nan))
        related = [t for t in (former + later) if t]
        if not related:
            continue

        corp_a = id_to_corp.get(zid, set())
        for tnorm in related:
            cand = title_to_ids.get(tnorm, set())
            if not cand:
                continue

            if len(cand) == 1:
                other = next(iter(cand))
                if other != zid:
                    dsu.union(zid, other)
            else:
                for other in cand:
                    if other == zid:
                        continue
                    corp_b = id_to_corp.get(other, set())
                    if corp_a and corp_b and (corp_a & corp_b):
                        dsu.union(zid, other)

    components: Dict[str, set] = {}
    for zid in id_to_row:
        root = dsu.find(zid)
        components.setdefault(root, set()).add(zid)

    return components, id_to_row


def analyze_genealogy(df: pd.DataFrame) -> None:
    components, id_to_row = build_lineages(df)
    comp_list = list(components.values())
    if not comp_list:
        return

    family_sizes = [len(c) for c in comp_list]
    family_data = []
    for i, members in enumerate(comp_list, start=1):
        titles = []
        for mid in sorted(members):
            titles.append(_strip_responsibility(id_to_row.get(mid, {}).get("Title", "")))
        family_data.append({
            "Family_ID": i,
            "Size": len(members),
            "ZDB_IDs": " | ".join(sorted(members)),
            "Titles": " | ".join([t for t in titles if t]),
        })
    pd.DataFrame(family_data).sort_values("Size", ascending=False).to_excel(
        OUTPUT_DIR / "genealogy_families.xlsx", index=False
    )

    largest_members = max(comp_list, key=len)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [0.70, 2.30]}
    )

    complex_families = [s for s in family_sizes if s > 1]
    if complex_families:
        ax1.hist(complex_families, bins=20, edgecolor="black", alpha=0.7)
    ax1.set_title("Distribution of Family Sizes (Excl. Singletons)")
    ax1.set_xlabel("Journals in Family")
    ax1.set_ylabel("Count of Families")

    tasks = []
    for mid in largest_members:
        row = id_to_row.get(mid, {})
        sy, end, cens = parse_published_span(row.get("Published", np.nan))
        if sy is None or end is None:
            continue
        title_main = _strip_responsibility(row.get("Title", mid))
        tasks.append({
            "title": _middle_ellipsis(title_main, maxlen=56) or str(mid),
            "start": int(sy),
            "end": int(end),
            "censored": bool(cens),
        })

    if tasks:
        tasks.sort(key=lambda t: (t["start"], -(t["end"] - t["start"])))
        y = np.arange(len(tasks))
        for i, t in enumerate(tasks):
            ax2.barh(
                y=i,
                width=(t["end"] - t["start"] + 1),
                left=t["start"],
                height=0.72,
                alpha=0.90,
                hatch="///" if t["censored"] else None,
            )
        ax2.set_yticks(y)
        ax2.set_yticklabels([t["title"] for t in tasks], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlim(min(t["start"] for t in tasks) - 1, max(t["end"] for t in tasks) + 1)
        ax2.set_xlabel("Year")
        ax2.set_title(f"Largest Family Gantt (size={len(largest_members)})")
        ax2.grid(axis="x", alpha=0.25)
        ax2.axvline(CENSOR_YEAR, alpha=0.25)
    else:
        ax2.set_title(f"Largest Family Gantt (size={len(largest_members)})")
        ax2.text(0.5, 0.5, "No parsed years from 'Published'.",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.55)
    fig.savefig(OUTPUT_DIR / "3_genealogy_gantt.png", dpi=300)
    plt.close(fig)


# -------------------------
# DDC stacked area by SPLIT codes (LAST output)
# -------------------------
def build_active_by_ddc_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Active-by-year per DDC code (split codes).
    Counting base: unique journal-code assignment.
    Note: Multi-label -> summing across codes double-counts journals (intended).
    """
    # Build unique journal spans
    agg = []
    for zid, g in df.groupby("ZDB-ID", dropna=False):
        spans = [parse_published_span(x) for x in g["Published"].tolist()]
        spans = [(a, b, c) for (a, b, c) in spans if a is not None and b is not None]
        if not spans:
            continue
        start = min(a for a, _, _ in spans)
        end = max(b for _, b, _ in spans)
        agg.append((str(zid), int(start), int(end)))
    if not agg:
        return pd.DataFrame()

    span_df = pd.DataFrame(agg, columns=["ZDB-ID", "start", "end"])

    # Build unique journal-code assignments (split + de-duplicate per journal)
    if "DDC subject groups" not in df.columns:
        return pd.DataFrame()

    code_df = df[["ZDB-ID", "DDC subject groups"]].copy()
    code_df["DDC_code"] = code_df["DDC subject groups"].apply(split_ddc_codes)
    code_df = code_df.explode("DDC_code").dropna(subset=["DDC_code"])
    code_df["DDC_code"] = code_df["DDC_code"].astype(str).str.strip()
    code_df = code_df.drop_duplicates(subset=["ZDB-ID", "DDC_code"])

    pairs = code_df.merge(span_df, on="ZDB-ID", how="inner")
    if pairs.empty:
        return pd.DataFrame()

    min_y = int(pairs["start"].min())
    max_y = int(pairs["end"].max())
    years = np.arange(min_y, max_y + 1)

    # Event-sweep per code
    code_events: Dict[str, Dict[int, int]] = {}
    for _, r in pairs.iterrows():
        code = r["DDC_code"]
        sy = int(r["start"])
        en = int(r["end"])
        ev = code_events.setdefault(code, {})
        ev[sy] = ev.get(sy, 0) + 1
        ev[en + 1] = ev.get(en + 1, 0) - 1

    out = pd.DataFrame(index=years)
    for code, ev in code_events.items():
        cur = 0
        vals = []
        for y in years:
            cur += ev.get(int(y), 0)
            vals.append(cur)
        out[code] = vals

    return out


def plot_ddc_stacked_area_last(df: pd.DataFrame) -> None:
    active_by_code = build_active_by_ddc_code(df)
    if active_by_code.empty:
        return

    # Rank codes by total active-years sum (multi-count)
    totals = active_by_code.sum(axis=0).sort_values(ascending=False)

    # Top-K + "Other"
    top_codes = totals.index[:DDC_STACK_TOPK].tolist()
    rest_codes = totals.index[DDC_STACK_TOPK:].tolist()

    plot_df = active_by_code[top_codes].copy()
    if rest_codes:
        plot_df["Other"] = active_by_code[rest_codes].sum(axis=1)

    # Labels: code + meaning
    labels = []
    for c in plot_df.columns:
        if c == "Other":
            labels.append("Other DDC codes")
        else:
            meaning = DDC_LABELS.get(str(c), "")
            labels.append(f"{c} {meaning}".strip())

    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.stackplot(
        plot_df.index.values,
        [plot_df[c].values for c in plot_df.columns],
        labels=labels,
        alpha=0.95,
    )
    # ax.set_title("Evolution of Active Journals by DDC Subject Groups (Split Codes, Stacked Area)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Active Journals (multi-assigned; may exceed total)")

    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "1_timeline_area_chart_ddc_codes.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "1_timeline_area_chart_ddc_codes.pdf")
    plt.close(fig)


# -------------------------
# I/O
# -------------------------
def load_data() -> pd.DataFrame:
    path = INPUT_XLSX_DEFAULT if INPUT_XLSX_DEFAULT.exists() else INPUT_XLSX_FALLBACK
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_XLSX_DEFAULT} or {INPUT_XLSX_FALLBACK}")

    df = pd.read_excel(path)
    print(f"Loaded {len(df)} records from {path}")

    if "Type of publication" not in df.columns:
        raise ValueError("Missing required column: 'Type of publication'")

    before = len(df)
    df = df[df["Type of publication"].astype(str).str.strip() == "Periodical (Journal)"].copy()
    print(f"Filtered Type of publication == 'Periodical (Journal)': {before} -> {len(df)}")

    for c in ["ZDB-ID", "Title", "Published"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["ZDB-ID"] = df["ZDB-ID"].astype(str).map(_clean_text)
    return df


def main() -> None:
    df = load_data()

    # 1) Active + turnover combined (keep)
    panel = build_year_panel_unique(df)
    plot_population_dynamics(panel, df)

    # 2) DDC split counts table + top bar (keep)
    ddc_counts = compute_ddc_counts(df)
    ddc_counts.to_excel(OUTPUT_DIR / "ddc_subject_group_code_counts.xlsx", index=False)
    plot_ddc_top_bar(ddc_counts, top_n=25)

    # 3) Genealogy (keep)
    #    Requires these columns; if absent, just skip cleanly.
    need_cols = {"Former/later titles", "Corporate body"}
    if need_cols.issubset(df.columns):
        analyze_genealogy(df)
    else:
        print("Skip genealogy: missing columns:", sorted(list(need_cols - set(df.columns))))

    # 4) (LAST) DDC stacked area by split codes + meaning labels
    plot_ddc_stacked_area_last(df)

    print("Done. Outputs in:", str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
