#!/usr/bin/env python3
"""Extract the academies-by-country table from Academies.html into Excel."""

from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

try:
    from openpyxl import Workbook
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "openpyxl is required to write .xlsx files. Install it with "
        "`python3 -m pip install openpyxl`."
    ) from exc


OUTPUT_COLUMNS = [
    "Geographical Area",
    "Founded year",
    "Society Name",
    "url link",
    "Acronym",
]


ROW_RE = re.compile(r"<tr\b.*?>.*?</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b.*?>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
ANCHOR_RE = re.compile(r"<a\b[^>]*href=\"(.*?)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)


def clean_text(value: str) -> str:
    text = TAG_RE.sub(" ", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_table_html(html: str) -> str:
    anchor_match = re.search(r"<a\s+name=\"ByCountry\">", html, re.IGNORECASE)
    if not anchor_match:
        raise ValueError("Could not find the ByCountry anchor in the HTML file.")

    table_start = re.search(r"<table\b", html[anchor_match.start():], re.IGNORECASE)
    if not table_start:
        raise ValueError("Could not find the table after the ByCountry anchor.")
    start = anchor_match.start() + table_start.start()

    table_end = re.search(r"</table>", html[start:], re.IGNORECASE)
    if not table_end:
        raise ValueError("Could not find the end of the By Country table.")
    end = start + table_end.end()

    return html[start:end]


def extract_society_name_and_url(cell_html: str, base_url: str) -> tuple[str, str]:
    anchor_match = ANCHOR_RE.search(cell_html)
    if not anchor_match:
        return clean_text(cell_html), ""

    href, anchor_html = anchor_match.groups()
    return clean_text(anchor_html), urljoin(base_url, href.strip())


def extract_rows(html_path: Path) -> list[dict[str, str]]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    table_html = extract_table_html(html)
    base_url = html_path.as_uri()

    records: list[dict[str, str]] = []
    current_area = ""

    for row_html in ROW_RE.findall(table_html):
        cells = CELL_RE.findall(row_html)
        if not cells:
            continue

        if re.search(r"colspan\s*=\s*['\"]?3", row_html, re.IGNORECASE):
            area = clean_text(cells[0])
            if area and area != "Academies and Royal Societies of Broad Scope: By Country":
                current_area = area
            continue

        if len(cells) != 3:
            continue

        founded_year = clean_text(cells[0])
        if founded_year.lower() == "founded":
            continue

        society_name, href = extract_society_name_and_url(cells[1], base_url)
        acronym = clean_text(cells[2])

        if not society_name:
            continue

        records.append(
            {
                "Geographical Area": current_area,
                "Founded year": founded_year,
                "Society Name": society_name,
                "url link": href,
                "Acronym": acronym,
            }
        )

    return records


def write_xlsx(rows: list[dict[str, str]], output_path: Path) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Academies"
    worksheet.append(OUTPUT_COLUMNS)

    for row in rows:
        worksheet.append([row[column] for column in OUTPUT_COLUMNS])

    for column_letter, width in {
        "A": 24,
        "B": 14,
        "C": 65,
        "D": 55,
        "E": 18,
    }.items():
        worksheet.column_dimensions[column_letter].width = width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 'Academies and Royal Societies of Broad Scope: By Country' to Excel."
    )
    parser.add_argument(
        "--input",
        help="Path to Academies.html",
    )
    parser.add_argument(
        "--output",
        help="Path to the output .xlsx file",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = extract_rows(input_path)
    write_xlsx(rows, output_path)
    print(f"Extracted {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
