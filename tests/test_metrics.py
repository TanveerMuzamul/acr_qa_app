import numpy as np
from pydicom.dataset import Dataset

from app.services.qa_metrics import build_reasoned_results, compute_metrics_for_slice



def test_compute_metrics_returns_expected_keys():
    ds = Dataset()
    ds.SliceThickness = 5.0
    ds.PixelSpacing = [1.0, 1.0]
    ds.Modality = "MR"
    ds.MagneticFieldStrength = 1.5
    img = np.full((64, 64), 100, dtype=np.uint16)

    raw = compute_metrics_for_slice(ds, img)

    assert raw["image_shape"] == "64 x 64"
    assert raw["slice_thickness_mm"] == 5.0
    assert raw["pixel_spacing"] == (1.0, 1.0)
    assert raw["modality"] == "MR"



def test_build_reasoned_results_marks_fail_when_uniformity_is_low():
    raw = {
        "modality": "MR",
        "field_strength_t": 1.5,
        "is_phantom_like": True,
        "pixel_spacing": (1.2, 1.2),
        "slice_thickness_mm": 6.2,
        "piu_percent": 60.0,
        "ghosting_ratio": 0.10,
        "lcd_spokes_total": 10,
        "lcd_spokes_per_slice": [2, 2, 3, 3],
        "series_kind": "T1",
    }

    results = build_reasoned_results(raw)

    assert results["overall_status"] == "FAIL"
    assert any(metric["status"] == "FAIL" for metric in results["metrics"])
