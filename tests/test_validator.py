"""
test_validator.py
=================
Proves the "Mass Spectrometer" (Task 2) catches mutations.
"""

import pytest
import sys
import os

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ocr.extractor import SpatialToken
from ocr.validator import DosageRangeRule, FrequencyPlausibilityRule, DrugNameSanityRule

def test_dosage_range_rule_high_mutation():
    """Test if an OCR mutation of 10mg -> 10000mg triggers the highest severity."""
    rule = DosageRangeRule()
    
    # Mock an OCR token where '10mg' was read as '10000mg' due to a blurry line
    token = SpatialToken(
        text="10000mg", x_min=0, y_min=0, x_max=10, y_max=10, 
        x_center=0.5, y_center=0.5, width_norm=0.1, height_norm=0.1, 
        confidence=0.9, region_id=1
    )
    
    flag = rule.check(token, drug_hint="omeprazole")
    
    assert flag is not None
    assert flag.action == "RESCAN"
    assert flag.severity == "HIGH"
    assert "outside" in flag.detail

def test_frequency_plausibility_rule():
    """Test if 'Every 1h' (24 times a day) triggers a warning."""
    rule = FrequencyPlausibilityRule()
    token = SpatialToken(
        text="Every 1h", x_min=0, y_min=0, x_max=10, y_max=10, 
        x_center=0.5, y_center=0.5, width_norm=0.1, height_norm=0.1, 
        confidence=0.9, region_id=1
    )
    
    flag = rule.check(token)
    assert flag is not None
    assert flag.action == "WARN"

def test_drug_sanity_rule_stamp_mutation():
    """Test if an overlapping stamp '12345' is misread as a drug."""
    rule = DrugNameSanityRule()
    drug_dict = {"paracetamol": 1, "amoxicillin": 1}
    
    token = SpatialToken(
        text="12345", x_min=0, y_min=0, x_max=10, y_max=10, 
        x_center=0.5, y_center=0.5, width_norm=0.1, height_norm=0.1, 
        confidence=0.9, region_id=1
    )
    
    flag = rule.check(token, drug_dict)
    assert flag is not None
    assert flag.severity == "HIGH"
    assert "numeric" in flag.detail.lower()