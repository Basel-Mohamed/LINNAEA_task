from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.drug_dictionary import load_drug_dictionary_from_excel


def test_load_drug_dictionary_from_excel_normalizes_tokens(tmp_path: Path):
    excel_path = tmp_path / "database.xlsx"
    pd.DataFrame(
        {
            "text": [" Amoxicillin ", "tablet", "PARACETAMOL"],
            "medicine": [1, 0, 1],
        }
    ).to_excel(excel_path, index=False)

    drug_dict = load_drug_dictionary_from_excel(excel_path)

    assert drug_dict == {
        "amoxicillin": 1,
        "tablet": 0,
        "paracetamol": 1,
    }


def test_load_drug_dictionary_from_excel_requires_expected_columns(tmp_path: Path):
    excel_path = tmp_path / "database.xlsx"
    pd.DataFrame({"token": ["abc"], "label": [1]}).to_excel(excel_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_drug_dictionary_from_excel(excel_path)
