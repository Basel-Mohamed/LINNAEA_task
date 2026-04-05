from __future__ import annotations

import numpy as np

from src.ocr.extractor import LineMerger, PrescriptionExtractor, SpatialToken


def test_line_merger_merges_neighboring_boxes_into_one_line_crop():
    merger = LineMerger(y_tol=0.6, x_gap_tol=0.12)
    boxes = np.array(
        [
            [0.20, 0.20, 0.10, 0.05],
            [0.33, 0.205, 0.10, 0.05],
            [0.22, 0.60, 0.10, 0.05],
        ]
    )
    confs = np.array([0.9, 0.8, 0.95])

    merged = merger.merge(boxes, confs, img_w=1000, img_h=1000)

    assert len(merged) == 2
    assert merged[0]["x_min"] < 200
    assert merged[0]["x_max"] > 350


def test_easyocr_box_normalization_preserves_geometry():
    normalized = PrescriptionExtractor._normalize_box(
        x_min=100,
        x_max=300,
        y_min=50,
        y_max=150,
        img_w=1000,
        img_h=500,
    )

    assert normalized == [0.2, 0.2, 0.2, 0.2]


def test_freeform_box_is_converted_to_rectangular_bounds():
    x_min, x_max, y_min, y_max = PrescriptionExtractor._freeform_to_rect(
        [[10, 40], [110, 20], [120, 90], [20, 100]]
    )

    assert (x_min, x_max, y_min, y_max) == (10, 120, 20, 100)


def test_spatial_token_structure_exposes_coordinates_and_zone():
    token = SpatialToken(
        text="paracetamol 500mg",
        x_min=10,
        y_min=20,
        x_max=110,
        y_max=60,
        x_center=0.3,
        y_center=0.4,
        width_norm=0.2,
        height_norm=0.08,
        confidence=0.97,
        region_id=7,
        source_path="mock.png",
        zone="body",
    )

    payload = token.to_dict()

    assert payload["bbox"] == {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 60}
    assert payload["zone"] == "body"
    assert payload["x_center"] == 0.3
