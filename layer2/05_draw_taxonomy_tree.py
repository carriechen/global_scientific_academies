from __future__ import annotations


import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ete3 import Tree, TreeStyle, NodeStyle, TextFace  # type: ignore

COLOR_MAP = {
    "Governance": "darkblue",
    "Strategic Plan": "darkblue",
    "Transparency": "darkblue",
    "Information": "orange",
    "Knowledge Resources": "green",
    "Scientific Work": "green",
    "Publication": "green",
    "Organizational Role": "purple",
    "Membership": "purple",
    "Leadership": "purple",
    "Organizational Structure": "brown",
    "Public Outreach": "pink",
    "Science Advice": "pink",
    "Scientific Cooperation": "gray",
    "International Cooperation": "gray",
    "Science Communication": "black",
    "Event": "black",
    "News": "black",
    "Social Media": "black",
    "Supporting Science": "indigo",
    "Scientific Culture": "indigo",
}


def pick_color(name: str, parent_name: str | None = None) -> str:
    if name in COLOR_MAP:
        return COLOR_MAP[name]
    if parent_name and parent_name in COLOR_MAP:
        return COLOR_MAP[parent_name]
    return "black"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", required=True, help="test.xlsx")
    parser.add_argument("--output", default="circular_taxonomy_tree.pdf")
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

    for node in tree.traverse():
        node_style = NodeStyle()
        color = pick_color(node.name, node.up.name if node.up else None)
        node_style["fgcolor"] = color
        node_style["size"] = 5
        node.set_style(node_style)
        face = TextFace(node.name, fsize=20, fgcolor=color)
        node.add_face(face, column=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.convert_to_ultrametric()
    tree.render(str(out), w=1000, tree_style=ts)
    print(f"saved: {out} [Fig. 1]")


if __name__ == "__main__":
    main()
