from pathlib import Path
from uuid import uuid4

import pandas as pd


INPUT_FILE = Path("global_science_academies V1.xlsx")
OUTPUT_FILE = INPUT_FILE.with_name(f"{INPUT_FILE.stem}_filled_ids.xlsx")
ID_PREFIX = "gsa"


def generate_unique_id(existing_ids: set[str]) -> str:
    while True:
        new_id = f"{ID_PREFIX}{uuid4().hex[:24]}"
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id


def main() -> None:
    df = pd.read_excel(INPUT_FILE)
    first_col = df.columns[0]

    existing_ids = {
        value.strip()
        for value in df[first_col].dropna().astype(str)
        if value.strip()
    }

    blank_mask = df[first_col].isna() | df[first_col].astype(str).str.strip().eq("")

    df.loc[blank_mask, first_col] = [
        generate_unique_id(existing_ids)
        for _ in range(int(blank_mask.sum()))
    ]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Filled {int(blank_mask.sum())} empty '{first_col}' cells.")
    print(f"Saved updated file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
