"""Load the drug vocabulary from the project Excel source."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_drug_dictionary_from_excel(
    excel_path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "medicine",
    sheet_name: str | int = 0,
) -> dict[str, int]:
    """Load a normalized token-to-label dictionary from the Excel database."""

    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Drug database not found: {path}")

    frame = pd.read_excel(path, sheet_name=sheet_name)
    missing_columns = {text_column, label_column}.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Drug database is missing required columns: {missing}")

    dictionary: dict[str, int] = {}
    for _, row in frame[[text_column, label_column]].dropna(subset=[text_column]).iterrows():
        token = str(row[text_column]).strip().casefold()
        if not token:
            continue
        try:
            dictionary[token] = int(row[label_column])
        except (TypeError, ValueError):
            dictionary[token] = 0

    if not dictionary:
        raise ValueError(f"Drug database is empty after parsing: {path}")

    return dictionary
