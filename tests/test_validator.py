from __future__ import annotations

from src.ocr.extractor import SpatialToken
from src.ocr.validator import DosageRangeRule, DrugNameSanityRule, FrequencyPlausibilityRule, GarbageTextRule


def _token(text: str) -> SpatialToken:
    return SpatialToken(
        text=text,
        x_min=0,
        y_min=0,
        x_max=10,
        y_max=10,
        x_center=0.5,
        y_center=0.5,
        width_norm=0.1,
        height_norm=0.1,
        confidence=0.9,
        region_id=1,
    )


def test_dosage_range_rule_high_mutation():
    rule = DosageRangeRule()
    flag = rule.check(_token("10000mg"), drug_hint="omeprazole")
    assert flag is not None
    assert flag.action == "RESCAN"
    assert flag.severity == "HIGH"
    assert "outside" in flag.detail


def test_frequency_plausibility_rule():
    rule = FrequencyPlausibilityRule()
    flag = rule.check(_token("Every 1h"))
    assert flag is not None
    assert flag.action == "WARN"


def test_drug_sanity_rule_stamp_mutation():
    rule = DrugNameSanityRule()
    drug_dict = {"paracetamol": 1, "amoxicillin": 1}
    flag = rule.check(_token("12345"), drug_dict)
    assert flag is not None
    assert flag.severity == "HIGH"
    assert "numeric" in flag.detail.lower()


def test_garbage_rule_skips_arabic_dosage_tokens():
    rule = GarbageTextRule()
    assert rule.check(_token("\u0645\u0631\u0629 \u064a\u0648\u0645\u064a")) is None
