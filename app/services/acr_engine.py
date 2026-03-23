from __future__ import annotations
from typing import Any, Dict, List

from app.services.acr_legacy.dataset import scan, group_series, classify_all, build
from app.services.acr_legacy.algorithms.slice_position import need_slice_pos, slice_pos_measure
from app.services.acr_legacy.algorithms.slice_thickness import need_slice_thickness, slice_thickness_measure
from app.services.acr_legacy.algorithms.geometric_accuracy import need_geometric, evaluate_geometric
from app.services.acr_legacy.algorithms.ghosting import need_ghosting, ghosting_measure
from app.services.acr_legacy.algorithms.piu import need_piu, piu_measure
from app.services.acr_legacy.algorithms.high_contrast_resolution import need_resolution, resolution_measure
from app.services.acr_legacy.algorithms.low_contrast_detectability import need_lcd, lcd_series_measure


def _series_label(data: dict[str, Any], series_uid: str) -> str | None:
    for label in ["localizer", "acr_t1", "acr_t2", "site_t1", "site_t2"]:
        s = data.get(label)
        if s is not None and getattr(s, 'suid', None) == series_uid:
            return label
    return None


def compute_legacy_acr_metrics(job_dir: str, series_uid: str) -> List[Dict[str, Any]]:
    files = scan(job_dir)
    if not files:
        return []
    data = build(classify_all(group_series(files)))
    label = _series_label(data, series_uid)
    metrics: list[dict[str, Any]] = []

    def add(name: str, value: str, expected: str, status: str, reason: str = ""):
        metrics.append({"name": name, "value": value, "expected": expected, "status": status, "reason": reason})

    if label == 'acr_t1' and need_geometric(data.get('localizer'), data.get('acr_t1')):
        try:
            g = evaluate_geometric(data.get('localizer'), data.get('acr_t1'))
            vals = ", ".join([f"{c['name']}={c['measured_mm']:.2f} mm" for c in g.get('checks', [])])
            add('Geometric accuracy', vals or 'Measured', 'Within tolerance', g.get('status', 'UNAVAILABLE'), g.get('phantom_type', ''))
        except Exception as e:
            add('Geometric accuracy', 'N/A', 'Within tolerance', 'UNAVAILABLE', str(e))

    target_series = data.get(label) if label else None
    if target_series is None:
        return metrics

    if need_slice_thickness(target_series):
        try:
            m = slice_thickness_measure(target_series, 1)
            add('Slice thickness (legacy)', f"{m['thickness_mm']:.2f} mm", 'ACR tolerance', m.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('Slice thickness (legacy)', 'N/A', 'ACR tolerance', 'UNAVAILABLE', str(e))

    if need_resolution(target_series):
        try:
            m = resolution_measure(target_series, 1)
            add('High-contrast resolution (legacy)', f"RL {m.get('right_left_mm')} / TB {m.get('top_bottom_mm')}", '1.0 mm patterns', m.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('High-contrast resolution (legacy)', 'N/A', '1.0 mm patterns', 'UNAVAILABLE', str(e))

    if need_slice_pos(target_series):
        try:
            m1 = slice_pos_measure(target_series, 1)
            add('Slice position slice 1', f"{m1['slice_disp_mm']:+.2f} mm", 'ACR tolerance', m1.get('status','UNAVAILABLE'), '')
            if target_series.n >= 11:
                m11 = slice_pos_measure(target_series, 11)
                add('Slice position slice 11', f"{m11['slice_disp_mm']:+.2f} mm", 'ACR tolerance', m11.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('Slice position', 'N/A', 'ACR tolerance', 'UNAVAILABLE', str(e))

    if need_piu(target_series):
        try:
            m = piu_measure(target_series, 7)
            add('Image intensity uniformity (legacy)', f"{m['piu']:.2f}%", f">= {m['limits']['target']:.1f}%", m.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('Image intensity uniformity (legacy)', 'N/A', 'ACR tolerance', 'UNAVAILABLE', str(e))

    if label == 'acr_t1' and need_ghosting(target_series):
        try:
            m = ghosting_measure(target_series, 7)
            add('Percent signal ghosting (legacy)', f"{m['psg_percent']:.2f}%", '<= 3.0%', m.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('Percent signal ghosting (legacy)', 'N/A', '<= 3.0%', 'UNAVAILABLE', str(e))

    if need_lcd(target_series):
        try:
            m = lcd_series_measure(target_series)
            add('Low-contrast detectability (legacy)', f"total {m['total_seen']}", '>= 37 spokes', m.get('status','UNAVAILABLE'), '')
        except Exception as e:
            add('Low-contrast detectability (legacy)', 'N/A', '>= 37 spokes', 'UNAVAILABLE', str(e))

    return metrics
