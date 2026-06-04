from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import org_hosted_journal_rate as base
from scientometrics_plot_utils import (
    apply_style,
    DOUBLE_COL_MM,
    AXIS_LABEL,
    PANEL_FONT,
    format_axes,
    mm_to_in,
    percent_axis,
    save_figure_multi,
)

apply_style()

base.OUT_DIR = base.BASE / "org_hosted_journal_rate_scientometrics"
base.OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_FACE = "#4D4D4D"
RIGHT_FACE = "white"
EDGE = "black"
LINE = "#6A6A6A"
CI = "#8A8A8A"


def _draw_dumbbell_on_ax(
    ax: plt.Axes,
    left_label: str,
    right_label: str,
    left_row: pd.Series,
    right_row: pd.Series,
    panel_letter: str,
    xlim: tuple[float, float] | None,
    show_xlabel: bool,
) -> None:
    xl = float(left_row["rate"])
    xr = float(right_row["rate"])
    ll, lh = float(left_row["ci_lo"]), float(left_row["ci_hi"])
    rl, rh = float(right_row["ci_lo"]), float(right_row["ci_hi"])

    ax.hlines(y=0, xmin=min(xl, xr), xmax=max(xl, xr), color=LINE, linewidth=1.6)
    ax.hlines(y=-0.13, xmin=ll, xmax=lh, color=CI, linewidth=2.6)
    ax.hlines(y=+0.13, xmin=rl, xmax=rh, color=CI, linewidth=2.6)

    ax.scatter([xl], [0], s=44, facecolor=LEFT_FACE, edgecolor=EDGE, linewidth=0.8, zorder=3)
    ax.scatter([xr], [0], s=44, facecolor=RIGHT_FACE, edgecolor=EDGE, linewidth=0.8, zorder=3)

    ax.text(xl, -0.31, f"{left_label}\n{xl:.1%} (n={int(left_row['n'])})\n{int(left_row['k'])} hosted",
            ha="center", va="top", fontsize=7.5)
    ax.text(xr, -0.31, f"{right_label}\n{xr:.1%} (n={int(right_row['n'])})\n{int(right_row['k'])} hosted",
            ha="center", va="top", fontsize=7.5)

    ax.text(0.0, 1.02, panel_letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PANEL_FONT, fontweight="bold")

    ax.set_ylim(-0.56, 0.30)
    ax.set_yticks([])
    if show_xlabel:
        ax.set_xlabel("Hosted-journal rate")
        percent_axis(ax)
    else:
        ax.set_xlabel("")
        percent_axis(ax)

    if xlim is None:
        xmin = max(0.0, min(ll, rl, xl, xr) - 0.05)
        xmax = min(1.0, max(lh, rh, xl, xr) + 0.05)
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(*xlim)

    format_axes(ax, xgrid=False, ygrid=False)
    ax.spines["left"].set_visible(False)


def plot_dumbbell_two_groups(left_label, right_label, left_row, right_row, title, subtitle, out_prefix):
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(55)))
    _draw_dumbbell_on_ax(ax, left_label, right_label, left_row, right_row, panel_letter="", xlim=None, show_xlabel=True)
    save_figure_multi(fig, base.OUT_DIR / out_prefix)
    plt.close(fig)


def plot_figure4_combined(df: pd.DataFrame, dev_sum: pd.DataFrame, eng_sum: pd.DataFrame):
    dev_developed = dev_sum[dev_sum["group"].eq("Developed")].iloc[0]
    dev_developing = dev_sum[dev_sum["group"].eq("Developing")].iloc[0]
    eng_non = eng_sum[eng_sum["group"].eq("Non-engineering")].iloc[0]
    eng_eng = eng_sum[eng_sum["group"].eq("Engineering")].iloc[0]

    vals = [
        float(dev_developed["ci_lo"]), float(dev_developed["ci_hi"]),
        float(dev_developing["ci_lo"]), float(dev_developing["ci_hi"]),
        float(eng_non["ci_lo"]), float(eng_non["ci_hi"]),
        float(eng_eng["ci_lo"]), float(eng_eng["ci_hi"]),
    ]
    xlim = (max(0.0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))

    fig, axes = plt.subplots(2, 1, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(105)), sharex=True)
    _draw_dumbbell_on_ax(
        axes[0], "Developing", "Developed", dev_developing, dev_developed,
        panel_letter="(a)", xlim=xlim, show_xlabel=False
    )
    _draw_dumbbell_on_ax(
        axes[1], "Engineering", "Non-engineering", eng_eng, eng_non,
        panel_letter="(b)", xlim=xlim, show_xlabel=True
    )
    fig.subplots_adjust(hspace=0.30)
    save_figure_multi(fig, base.OUT_DIR / "Figure8")
    plt.close(fig)


base.plot_dumbbell_two_groups = plot_dumbbell_two_groups
base.plot_figure4_combined = plot_figure4_combined


if __name__ == "__main__":
    base.main()
