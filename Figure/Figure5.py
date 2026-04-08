from __future__ import annotations

"""Draw the original circular taxonomy tree and export multiple formats.

This version keeps the original circular layout, but replaces the ad hoc named
colors with a unified journal-style categorical palette shared with the other
Scientometrics figure scripts.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from scientometrics_plot_utils import (
    apply_style,
    SCI_BLUE,
    SCI_ORANGE,
    SCI_GREEN,
    SCI_PURPLE,
    SCI_TEAL,
    SCI_RED,
    SCI_GOLD,
    SCI_BROWN,
    darken,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ete3 import NodeStyle, TextFace, Tree, TreeStyle  # type: ignore

apply_style()

# Unified, publication-grade categorical palette.
COLOR_MAP = {
    "Governance": SCI_BLUE,
    "Strategic Plan": SCI_BLUE,
    "Transparency": darken(SCI_BLUE, 0.08),
    "Information": SCI_ORANGE,
    "Knowledge Resources": SCI_GREEN,
    "Scientific Work": SCI_GREEN,
    "Publication": darken(SCI_GREEN, 0.06),
    "Organizational Role": SCI_PURPLE,
    "Membership": SCI_PURPLE,
    "Leadership": darken(SCI_PURPLE, 0.06),
    "Organizational Structure": SCI_BROWN,
    "Public Outreach": SCI_TEAL,
    "Science Advice": SCI_TEAL,
    "Scientific Cooperation": SCI_RED,
    "International Cooperation": SCI_RED,
    "Science Communication": SCI_GOLD,
    "Event": SCI_GOLD,
    "News": darken(SCI_GOLD, 0.10),
    "Social Media": darken(SCI_GOLD, 0.10),
    "Supporting Science": darken(SCI_BLUE, 0.20),
    "Scientific Culture": darken(SCI_BLUE, 0.20),
}


def pick_color(name: str, parent_name: str | None = None) -> str:
    if name in COLOR_MAP:
        return COLOR_MAP[name]
    if parent_name and parent_name in COLOR_MAP:
        return COLOR_MAP[parent_name]
    return "#333333"



def run_gs(pdf_path: Path, output_path: Path, device: str, dpi: int = 600) -> None:
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError("Ghostscript (gs) was not found on PATH.")

    cmd = [gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", f"-sDEVICE={device}"]
    if device.startswith("tiff") or device.startswith("png"):
        cmd.extend([f"-r{dpi}"])
    if device.startswith("tiff"):
        cmd.append("-sCompression=lzw")
    cmd.extend([f"-sOutputFile={output_path}", str(pdf_path)])
    subprocess.run(cmd, check=True)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default='Figure5.xlsx', help="Path to test.xlsx")
    parser.add_argument(
        "--output-stem",
        default="Figure5",
        help="Output stem without extension, e.g. outputs/circular_taxonomy_tree",
    )
    parser.add_argument("--width", type=int, default=1000, help="ETE render width")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for TIFF/PNG exports")
    args = parser.parse_args()

    hierdata = pd.read_excel(args.taxonomy)
    hierdata = hierdata[["hypernym", "hyponym"]].dropna().drop_duplicates()

    tree = Tree(name="Homepage")
    nodes = {"Homepage": tree}

    unresolved = hierdata.copy()
    progress = True
    while progress and not unresolved.empty:
        progress = False
        keep_rows = []
        for _, row in unresolved.iterrows():
            hypernym = str(row["hypernym"])
            hyponym = str(row["hyponym"])
            if hypernym in nodes:
                if hyponym not in nodes:
                    nodes[hyponym] = nodes[hypernym].add_child(name=hyponym)
                progress = True
            else:
                keep_rows.append(row)
        unresolved = pd.DataFrame(keep_rows)

    ts = TreeStyle()
    ts.mode = "c"
    ts.show_leaf_name = False
    ts.scale = 80
    ts.show_scale = False

    for node in tree.traverse():
        node_style = NodeStyle()
        color = pick_color(node.name, node.up.name if node.up else None)
        node_style["fgcolor"] = color
        node_style["size"] = 5
        node.set_style(node_style)
        face = TextFace(node.name, fsize=20, fgcolor=color)
        node.add_face(face, column=0)

    out_stem = Path(args.output_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out_stem.with_suffix(".pdf")
    eps_path = out_stem.with_suffix(".eps")
    tiff_path = out_stem.with_suffix(".tiff")
    png_path = out_stem.with_suffix(".png")

    tree.convert_to_ultrametric()
    tree.render(str(pdf_path), w=args.width, tree_style=ts)

    run_gs(pdf_path, eps_path, device="eps2write")
    run_gs(pdf_path, tiff_path, device="tiff24nc", dpi=args.dpi)
    run_gs(pdf_path, png_path, device="pngalpha", dpi=args.dpi)

    print("saved:")
    print(pdf_path)
    print(eps_path)
    print(tiff_path)
    print(png_path)


if __name__ == "__main__":
    main()
