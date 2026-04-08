from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from scientometrics_plot_utils import (
    apply_style,
    DOUBLE_COL_MM,
    BITMAP_DPI,
    SCI_BLUE,
    SCI_ORANGE,
    SCI_TEAL,
    lighten,
    darken,
    mm_to_in,
    save_figure_multi,
)

FIG_W_MM = DOUBLE_COL_MM
FIG_H_MM = 125

apply_style()


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"
    return f"= {p_value:.3f}"


df = pd.read_csv("Figure4.csv")
plot_df = df[["log10_url_count", "median_depth"]].copy()
plot_df = plot_df.apply(pd.to_numeric, errors="coerce").dropna()

x = plot_df["log10_url_count"].to_numpy()
y = plot_df["median_depth"].to_numpy()

r, p = stats.pearsonr(x, y)
slope, intercept, r_val, _, _ = stats.linregress(x, y)
r2 = r_val ** 2
n = len(plot_df)

point_color = lighten(SCI_BLUE, 0.10)
line_color = SCI_ORANGE
hist_color = lighten(SCI_BLUE, 0.28)
hist_y_color = lighten(SCI_TEAL, 0.12)
grid_color = "#E6ECF3"

fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=BITMAP_DPI)
gs = GridSpec(2, 2, figure=fig, width_ratios=[2.45, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.34)

ax_scatter = fig.add_subplot(gs[:, 0])
ax_hist_x = fig.add_subplot(gs[0, 1])
ax_hist_y = fig.add_subplot(gs[1, 1])

ax_scatter.scatter(
    x,
    y,
    s=18,
    facecolor=point_color,
    edgecolor=darken(SCI_BLUE, 0.18),
    linewidth=0.4,
    alpha=0.9,
    label="Observations",
)

xline = np.linspace(x.min(), x.max(), 200)
ax_scatter.plot(xline, slope * xline + intercept, color=line_color, linewidth=1.2, label="OLS fit")

ax_scatter.set_xlabel("Website URL count (log10)")
ax_scatter.set_ylabel("Median URL depth")
ax_scatter.grid(True, color=grid_color, linewidth=0.5)
ax_scatter.spines["top"].set_visible(False)
ax_scatter.spines["right"].set_visible(False)

xpad = 0.05 * (x.max() - x.min() if x.max() > x.min() else 1)
ypad = 0.08 * (y.max() - y.min() if y.max() > y.min() else 1)
ax_scatter.set_xlim(max(0, x.min() - xpad), x.max() + xpad)
ax_scatter.set_ylim(max(0, y.min() - ypad), y.max() + ypad)

stats_text = f"r = {r:.2f}\np {format_p_value(p)}\n$R^2$ = {r2:.3f}\nn = {n}"
ax_scatter.text(
    0.03,
    0.97,
    stats_text,
    transform=ax_scatter.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=lighten(SCI_BLUE, 0.45), linewidth=0.5),
)
ax_scatter.legend(loc="lower right", frameon=False, handlelength=2.4)
ax_scatter.text(-0.10, 1.02, "(a)", transform=ax_scatter.transAxes, fontsize=9, fontweight="bold")

ax_hist_x.hist(x, bins=10, color=hist_color, edgecolor=darken(SCI_BLUE, 0.16), linewidth=0.45)
ax_hist_x.set_xlabel("Website URL count (log10)")
ax_hist_x.set_ylabel("Frequency")
ax_hist_x.grid(True, axis="y", color=grid_color, linewidth=0.5)
ax_hist_x.spines["top"].set_visible(False)
ax_hist_x.spines["right"].set_visible(False)
ax_hist_x.text(-0.15, 1.02, "(b)", transform=ax_hist_x.transAxes, fontsize=9, fontweight="bold")

bins_y = np.arange(np.floor(y.min()) - 0.5, np.ceil(y.max()) + 1.5, 1)
ax_hist_y.hist(y, bins=bins_y, color=hist_y_color, edgecolor=darken(SCI_TEAL, 0.18), linewidth=0.45)
ax_hist_y.set_xlabel("Median URL depth")
ax_hist_y.set_ylabel("Frequency")
ax_hist_y.grid(True, axis="y", color=grid_color, linewidth=0.5)
ax_hist_y.spines["top"].set_visible(False)
ax_hist_y.spines["right"].set_visible(False)
ax_hist_y.text(-0.15, 1.02, "(c)", transform=ax_hist_y.transAxes, fontsize=9, fontweight="bold")

fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.95)
save_figure_multi(fig, Path("Figure4"))
plt.close(fig)

print("Created Figure4 in PNG/PDF/TIFF/EPS")
