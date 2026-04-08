from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scientometrics_plot_utils import (
    apply_style,
    DOUBLE_COL_MM,
    BITMAP_DPI,
    SCI_SEQ_CMAP,
    mm_to_in,
    save_figure_multi,
)

# Scientometrics / Springer settings
FIG_W_MM = DOUBLE_COL_MM
FIG_H_MM = 108

apply_style()

path = Path("global_science_academies_final.xlsx")
df = pd.read_excel(path)


def century_label(year):
    if pd.isna(year):
        return np.nan
    y = int(year)
    if 1401 <= y <= 1500:
        return "15th"
    if 1501 <= y <= 1600:
        return "16th"
    if 1601 <= y <= 1700:
        return "17th"
    if 1701 <= y <= 1800:
        return "18th"
    if 1801 <= y <= 1900:
        return "19th"
    if 1901 <= y <= 2000:
        return "20th"
    if 2001 <= y <= 2100:
        return "21st"
    return np.nan


plot_df = df.copy()
plot_df["founding_date"] = pd.to_numeric(plot_df["founding_date"], errors="coerce")
plot_df["continent"] = plot_df["continent"].astype(str).str.strip()
plot_df["century"] = plot_df["founding_date"].apply(century_label)
plot_df = plot_df.dropna(subset=["continent", "century"])

continent_order = ["Europe", "Asia", "Africa", "South America", "North America", "Oceania"]
century_order = ["15th", "16th", "17th", "18th", "19th", "20th", "21st"]

heatmap = (
    pd.crosstab(plot_df["continent"], plot_df["century"])
    .reindex(index=continent_order, columns=century_order, fill_value=0)
)

fig, ax = plt.subplots(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=BITMAP_DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(
    heatmap.values,
    cmap=SCI_SEQ_CMAP,
    aspect="auto",
    vmin=0,
    vmax=float(heatmap.values.max()) if heatmap.values.size else 1.0,
)

ax.set_xticks(np.arange(len(century_order)))
ax.set_xticklabels(century_order)
ax.set_yticks(np.arange(len(continent_order)))
ax.set_yticklabels(continent_order)
ax.tick_params(axis="both", which="major", length=0)

ax.set_xticks(np.arange(-0.5, len(century_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(continent_order), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        val = int(heatmap.iat[i, j])
        rgba = SCI_SEQ_CMAP(im.norm(val))
        r, g, b, _ = rgba
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = "white" if luminance < 0.50 else "black"
        ax.text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=8.1)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlabel("Founding century")
ax.set_ylabel("Continent")

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.set_label("Count", fontsize=8.5)
cbar.ax.tick_params(labelsize=8, length=2)
cbar.outline.set_visible(False)

fig.subplots_adjust(left=0.15, right=0.93, bottom=0.18, top=0.97)
save_figure_multi(fig, Path("Figure3"))
plt.close(fig)

print("Created Figure3_scientometrics in PNG/PDF/TIFF/EPS")
