from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import journal_analysis as base
from scientometrics_plot_utils import (
    apply_style,
    DOUBLE_COL_MM,
    PANEL_FONT,
    LEGEND_FONT,
    TICK_FONT,
    format_axes,
    mm_to_in,
    save_figure_multi,
    EXTENDED_CATEGORICAL,
    SCI_BLUE,
    SCI_ORANGE,
    SCI_RED,
    SCI_TEAL,
    SCI_GREEN,
    SCI_SEQ_CMAP,
    lighten,
    darken,
)

apply_style()

base.OUTPUT_DIR = base.Path(".")
# base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AREA_COLORS = EXTENDED_CATEGORICAL
GEN_EDGE = darken(SCI_BLUE, 0.28)
GEN_GRID = "#E6ECF3"
GEN_CENSOR = lighten(SCI_ORANGE, 0.12)
GEN_HATCH_EDGE = darken(SCI_ORANGE, 0.18)



def plot_population_dynamics(panel: pd.DataFrame, df: pd.DataFrame) -> None:
    if panel.empty:
        return
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(112)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")

    ax1.plot(panel["year"], panel["active"], color=SCI_BLUE, linewidth=1.6)
    ax1.set_ylabel("Active journals")
    format_axes(ax1, xgrid=False, ygrid=True)
    ax1.text(0.01, 0.96, "(a)", transform=ax1.transAxes, fontsize=PANEL_FONT, fontweight="bold", va="top")

    years = panel["year"].values
    births = panel["births"].values
    deaths = panel["deaths"].values
    net = panel["net"].values
    b1 = ax2.bar(years, births, width=0.85, color=lighten(SCI_BLUE, 0.18), edgecolor=darken(SCI_BLUE, 0.16), linewidth=0.35, label="Births")
    b2 = ax2.bar(years, -deaths, width=0.85, color=lighten(SCI_RED, 0.10), edgecolor=darken(SCI_RED, 0.16), linewidth=0.35, label="Deaths")
    l1, = ax2.plot(years, net, color=SCI_TEAL, linestyle=(0, (5, 2)), linewidth=1.25, label="Net")
    ax2.axhline(0, color=darken(SCI_BLUE, 0.25), linewidth=0.8)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Year")
    format_axes(ax2, xgrid=False, ygrid=True)
    ax2.text(0.01, 0.96, "(b)", transform=ax2.transAxes, fontsize=PANEL_FONT, fontweight="bold", va="top")
    ax2.legend(handles=[l1, b1, b2], labels=["Net", "Births", "Deaths"], loc="upper left", ncol=3, frameon=False)

    save_figure_multi(fig, base.OUTPUT_DIR / "Figure9")
    plt.close(fig)
    panel.to_excel(base.OUTPUT_DIR / "Figure9.xlsx", index=False)



def plot_ddc_top_bar(ddc_counts: pd.DataFrame, top_n: int = 25) -> None:
    if ddc_counts.empty:
        return
    top = ddc_counts.head(top_n).copy()
    top["label"] = top["DDC_code"].astype(str) + "  " + top["DDC_label"].astype(str)
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(140)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bar_colors = [AREA_COLORS[i % len(AREA_COLORS)] for i in range(len(top))]
    ax.barh(top["label"], top["n_journals"].values, color=bar_colors, edgecolor=darken(SCI_BLUE, 0.24), linewidth=0.3)
    ax.set_xlabel("Number of journals (unique ZDB-ID)")
    format_axes(ax, xgrid=True, ygrid=False)
    save_figure_multi(fig, base.OUTPUT_DIR / "Figure10")
    plt.close(fig)



def analyze_genealogy(df: pd.DataFrame) -> None:
    components, id_to_row = base.build_lineages(df)
    comp_list = list(components.values())
    if not comp_list:
        return

    family_data = []
    for i, members in enumerate(comp_list, start=1):
        titles = []
        for mid in sorted(members):
            titles.append(base._strip_responsibility(id_to_row.get(mid, {}).get("Title", "")))
        family_data.append({
            "Family_ID": i,
            "Size": len(members),
            "ZDB_IDs": " | ".join(sorted(members)),
            "Titles": " | ".join([t for t in titles if t]),
        })
    pd.DataFrame(family_data).sort_values("Size", ascending=False).to_excel(base.OUTPUT_DIR / "genealogy_families.xlsx", index=False)

    largest_members = max(comp_list, key=len)
    tasks = []
    for mid in largest_members:
        row = id_to_row.get(mid, {})
        sy, end, cens = base.parse_published_span(row.get("Published", np.nan))
        if sy is None or end is None:
            continue
        title_main = base._strip_responsibility(row.get("Title", mid))
        tasks.append({
            "title": base._middle_ellipsis(title_main, maxlen=56) or str(mid),
            "start": int(sy),
            "end": int(end),
            "censored": bool(cens),
        })

    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(110)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if tasks:
        tasks.sort(key=lambda t: (t["start"], -(t["end"] - t["start"]), t["title"]))
        y = np.arange(len(tasks))
        colors = [AREA_COLORS[i % len(AREA_COLORS)] for i in range(len(tasks))]

        for i, (t, color) in enumerate(zip(tasks, colors)):
            width = t["end"] - t["start"] + 1
            ax.barh(y=i, width=width, left=t["start"], height=0.72, color=color, edgecolor=GEN_EDGE, linewidth=0.30, zorder=3)
            if t["censored"]:
                ax.barh(y=i, width=width, left=t["start"], height=0.72, color="none", edgecolor=GEN_HATCH_EDGE, linewidth=0.0, hatch="//", zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels([t["title"] for t in tasks], fontsize=TICK_FONT - 1.0)
        ax.invert_yaxis()
        ax.set_xlim(min(t["start"] for t in tasks) - 1, max(t["end"] for t in tasks) + 1)
        ax.set_xlabel("Year")
        ax.set_title(f"Largest family gantt (size={len(largest_members)})", pad=4)
        ax.grid(axis="x", color=GEN_GRID, linewidth=0.45)
        ax.grid(False, axis="y")
        ax.axvline(base.CENSOR_YEAR, color=GEN_CENSOR, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
        ax.tick_params(axis="x", width=0.8, length=3.2)
        ax.tick_params(axis="y", length=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(
            handles=[
                Patch(facecolor=SCI_BLUE, edgecolor=GEN_EDGE, linewidth=0.30, label="Journal span"),
                Patch(facecolor="white", edgecolor=GEN_HATCH_EDGE, hatch="//", linewidth=0.30, label=f"Right-censored at {base.CENSOR_YEAR}"),
            ],
            loc="upper right",
            frameon=False,
            borderaxespad=0.2,
            handlelength=1.4,
            handleheight=0.7,
            prop={"size": LEGEND_FONT},
        )
    else:
        ax.set_title(f"Largest family gantt (size={len(largest_members)})", pad=4)
        ax.text(0.5, 0.5, "No parsed years from 'Published'.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    fig.subplots_adjust(left=0.25, right=0.985, top=0.92, bottom=0.10)
    save_figure_multi(fig, base.OUTPUT_DIR / "Figure11")
    plt.close(fig)



def plot_ddc_stacked_area_last(df: pd.DataFrame) -> None:
    active_by_code = base.build_active_by_ddc_code(df)
    if active_by_code.empty:
        return
    totals = active_by_code.sum(axis=0).sort_values(ascending=False)
    top_codes = totals.index[: base.DDC_STACK_TOPK].tolist()
    rest_codes = totals.index[base.DDC_STACK_TOPK :].tolist()
    plot_df = active_by_code[top_codes].copy()
    if rest_codes:
        plot_df["Other"] = active_by_code[rest_codes].sum(axis=1)
    labels = []
    for c in plot_df.columns:
        if c == "Other":
            labels.append("Other DDC codes")
        else:
            meaning = base.DDC_LABELS.get(str(c), "")
            labels.append(f"{c} {meaning}".strip())
    colors = AREA_COLORS[: len(plot_df.columns)]
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(96)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.stackplot(plot_df.index.values, [plot_df[c].values for c in plot_df.columns], labels=labels, colors=colors, linewidth=0.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of active journals")
    format_axes(ax, xgrid=False, ygrid=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    save_figure_multi(fig, base.OUTPUT_DIR / "Figure10")
    plt.close(fig)


base.plot_population_dynamics = plot_population_dynamics
# base.plot_ddc_top_bar = plot_ddc_top_bar
base.analyze_genealogy = analyze_genealogy
base.plot_ddc_stacked_area_last = plot_ddc_stacked_area_last

if __name__ == "__main__":
    base.main()
