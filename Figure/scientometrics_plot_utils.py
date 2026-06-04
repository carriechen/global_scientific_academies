from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

DOUBLE_COL_MM = 174
SINGLE_COL_MM = 84
MAX_HEIGHT_MM = 234

BASE_FONT = 8.5
AXIS_LABEL = 9.0
TITLE_FONT = 10.0
PANEL_FONT = 9.5
LEGEND_FONT = 7.8
TICK_FONT = 8.0
LINE_WIDTH = 1.2
MARKER_EDGE = 0.8
BITMAP_DPI = 600

# Unified low-saturation palette for a Nature/Springer-like manuscript style.
# This is a stylistic choice rather than a journal-mandated palette: muted categorical
# colors, restrained contrast, and print-safe luminance separation.
NATURE_MUTED10 = [
    "#4C78A8",  # muted blue
    "#DD8452",  # muted orange
    "#55A868",  # muted green
    "#8172B2",  # muted purple
    "#64B5CD",  # muted cyan/teal
    "#C44E52",  # muted red
    "#937860",  # warm brown
    "#DA8BC3",  # muted pink
    "#8C8C8C",  # medium gray
    "#CCB974",  # muted gold/olive
]

SCI_BLUE = NATURE_MUTED10[0]
SCI_ORANGE = NATURE_MUTED10[1]
SCI_GREEN = NATURE_MUTED10[2]
SCI_PURPLE = NATURE_MUTED10[3]
SCI_TEAL = NATURE_MUTED10[4]
SCI_RED = NATURE_MUTED10[5]
SCI_BROWN = NATURE_MUTED10[6]
SCI_PINK = NATURE_MUTED10[7]
SCI_GRAY = NATURE_MUTED10[8]
SCI_GOLD = NATURE_MUTED10[9]


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb)


def lighten(hex_color: str, frac: float = 0.25) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac))


def darken(hex_color: str, frac: float = 0.12) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r * (1 - frac), g * (1 - frac), b * (1 - frac)))


# Extended categorical palette with restrained light/dark variants for larger legends.
EXTENDED_CATEGORICAL = (
    NATURE_MUTED10
    + [lighten(c, 0.18) for c in NATURE_MUTED10]
    + [darken(c, 0.08) for c in NATURE_MUTED10]
)

PRIMARY3 = [SCI_BLUE, SCI_TEAL, SCI_ORANGE]
PRIMARY4 = [SCI_BLUE, SCI_TEAL, SCI_ORANGE, SCI_PURPLE]
PRIMARY6 = [SCI_BLUE, SCI_ORANGE, SCI_TEAL, SCI_GREEN, SCI_PURPLE, SCI_GOLD]

# Sequential maps stay colored, but with a softer ramp and higher low-end lightness.
SCI_SEQ_CMAP = LinearSegmentedColormap.from_list(
    "scientometrics_blue_seq",
    ["#FAFCFE", "#EAF1F7", "#D3E0EE", "#B4CADE", "#89AACB", SCI_BLUE, darken(SCI_BLUE, 0.12)],
)

SCI_TEAL_SEQ_CMAP = LinearSegmentedColormap.from_list(
    "scientometrics_teal_seq",
    ["#FBFDFD", "#EBF6F7", "#D6ECEF", "#B5DCE3", "#8FC7D4", SCI_TEAL, darken(SCI_TEAL, 0.12)],
)

SCI_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "scientometrics_diverging",
    [darken(SCI_BLUE, 0.08), "#F7F5F2", darken(SCI_ORANGE, 0.06)],
)

def mm_to_in(mm: float) -> float:
    return mm / 25.4



def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": BASE_FONT,
        "axes.labelsize": AXIS_LABEL,
        "axes.titlesize": TITLE_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": BITMAP_DPI,
        "figure.dpi": 150,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "hatch.linewidth": 0.55,
    })


apply_style()



def format_axes(ax: plt.Axes, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=0.8, length=3.5)
    if ygrid:
        ax.grid(True, axis="y", color="#D7DFEA", linewidth=0.5)
    else:
        ax.grid(False, axis="y")
    if xgrid:
        ax.grid(True, axis="x", color="#E6ECF3", linewidth=0.5)
    else:
        ax.grid(False, axis="x")



def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=PANEL_FONT,
            fontweight="bold", ha="left", va="bottom")



def save_figure_multi(fig: plt.Figure, out_base: Path, tight: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "facecolor": "white",
        "bbox_inches": "tight" if tight else None,
        "pad_inches": 0.02,
    }
    fig.savefig(out_base.with_suffix(".png"), dpi=BITMAP_DPI, **common)
    fig.savefig(out_base.with_suffix(".pdf"), **common)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=BITMAP_DPI, pil_kwargs={"compression": "tiff_lzw"}, **common)
    fig.savefig(out_base.with_suffix(".eps"), **common)



def percent_axis(ax: plt.Axes, decimals: int = 0) -> None:
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))
