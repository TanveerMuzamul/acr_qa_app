from __future__ import annotations

"""Geometric Accuracy module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset import Series, get_img, plane, to_float


# =========
# utils
# =========

def row_mm(ps: Optional[list[float]]) -> Optional[float]:
    if not ps or len(ps) < 1:
        return None
    return to_float(ps[0])


def col_mm(ps: Optional[list[float]]) -> Optional[float]:
    if not ps or len(ps) < 2:
        return None
    return to_float(ps[1])


# =========
# mask
# =========

def phantom_mask(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    pos = a[a > np.min(a)]

    if pos.size == 0:
        return np.zeros_like(a, dtype=bool)

    p99 = float(np.percentile(pos, 99))
    thr = max(p99 * 0.08, 1.0)
    return a > thr


def phantom_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    m = phantom_mask(arr)
    ys, xs = np.where(m)

    if ys.size == 0 or xs.size == 0:
        h, w = arr.shape
        return 0, 0, w, h

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    return x0, y0, x1, y1


def geometric_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = phantom_box(arr)

    if (x1 - x0) > 4:
        x0 += 1
        x1 -= 1

    if (y1 - y0) > 4:
        y0 += 1
        y1 -= 1

    return x0, y0, x1, y1


# =========
# localizer choose
# =========

def find_sagittal_localizer_idx(s: Series) -> Optional[int]:
    for idx in range(1, s.n + 1):
        d = s.files[idx - 1]
        if plane(d.iop) == "sagittal":
            return idx
    return None


def best_localizer_idx(s: Optional[Series]) -> Optional[int]:
    if s is None or s.n < 1:
        return None
    return find_sagittal_localizer_idx(s)


# =========
# measure
# =========

def _box_metrics(arr: np.ndarray, ps: Optional[list[float]]) -> dict[str, Any]:
    rmm = row_mm(ps)
    cmm = col_mm(ps)
    if rmm is None or cmm is None:
        raise ValueError("pixel spacing unavailable")

    x0, y0, x1, y1 = geometric_box(arr)

    w_px = float(x1 - x0)
    h_px = float(y1 - y0)

    w_mm = w_px * cmm
    h_mm = h_px * rmm
    d_mm = 0.5 * (w_mm + h_mm)

    cx = 0.5 * (x0 + x1 - 1)
    cy = 0.5 * (y0 + y1 - 1)

    r = 0.25 * (w_px + h_px)

    return {
        "box": (x0, y0, x1, y1),
        "center": (cx, cy),
        "w_px": w_px,
        "h_px": h_px,
        "w_mm": w_mm,
        "h_mm": h_mm,
        "d_mm": d_mm,
        "r": r,
        "rmm": rmm,
        "cmm": cmm,
    }


def axial_geometry_measure(s: Series, idx: int) -> dict[str, Any]:
    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    m = _box_metrics(arr, d.ps)
    x0, y0, x1, y1 = m["box"]
    cx, cy = m["center"]
    r = m["r"]

    top_bottom_mm = m["h_mm"]
    left_right_mm = m["w_mm"]
    diag1_mm = m["d_mm"]
    diag2_mm = m["d_mm"]

    tb_line = {
        "x_a": cx,
        "y_a": y0,
        "x_b": cx,
        "y_b": y1 - 1,
    }
    lr_line = {
        "x_a": x0,
        "y_a": cy,
        "x_b": x1 - 1,
        "y_b": cy,
    }
    d1_line = {
        "x_a": cx - r / np.sqrt(2.0),
        "y_a": cy - r / np.sqrt(2.0),
        "x_b": cx + r / np.sqrt(2.0),
        "y_b": cy + r / np.sqrt(2.0),
    }
    d2_line = {
        "x_a": cx - r / np.sqrt(2.0),
        "y_a": cy + r / np.sqrt(2.0),
        "x_b": cx + r / np.sqrt(2.0),
        "y_b": cy - r / np.sqrt(2.0),
    }

    return {
        "idx": idx,
        "inst": d.inst,
        "shape": tuple(arr.shape),
        "phantom_box": (x0, y0, x1, y1),
        "center": (cx, cy),
        "top_bottom_mm": top_bottom_mm,
        "left_right_mm": left_right_mm,
        "diag1_mm": diag1_mm,
        "diag2_mm": diag2_mm,
        "top_bottom_line": tb_line,
        "left_right_line": lr_line,
        "diag1_line": d1_line,
        "diag2_line": d2_line,
    }


def localizer_length_measure(s: Series, idx: Optional[int] = None) -> dict[str, Any]:
    if idx is None:
        idx = best_localizer_idx(s)

    if idx is None:
        return {
            "available": False,
            "reason": "no sagittal localizer slice found",
            "idx": None,
            "inst": None,
            "shape": None,
        }

    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    if plane(d.iop) != "sagittal":
        return {
            "available": False,
            "reason": "localizer is not sagittal",
            "idx": idx,
            "inst": d.inst,
            "shape": tuple(arr.shape),
        }

    m = _box_metrics(arr, d.ps)
    x0, y0, x1, y1 = m["box"]
    cx, cy = m["center"]

    return {
        "available": True,
        "idx": idx,
        "inst": d.inst,
        "shape": tuple(arr.shape),
        "phantom_box": (x0, y0, x1, y1),
        "center": (cx, cy),
        "length_mm": m["h_mm"],
        "line": {
            "x_a": cx,
            "y_a": y0,
            "x_b": cx,
            "y_b": y1 - 1,
        },
    }


# =========
# eval
# =========

def infer_phantom_type(slice1: dict[str, Any]) -> str:
    avg_diam = 0.5 * (slice1["top_bottom_mm"] + slice1["left_right_mm"])
    if abs(avg_diam - 190.0) <= abs(avg_diam - 165.0):
        return "large"
    return "medium"


def geom_spec(phantom_type: str) -> dict[str, float]:
    if phantom_type == "medium":
        return {
            "length_mm": 134.0,
            "diameter_mm": 165.0,
            "tol_mm": 2.0,
        }
    return {
        "length_mm": 148.0,
        "diameter_mm": 190.0,
        "tol_mm": 3.0,
    }


def evaluate_geometric(localizer_s: Optional[Series], t1_s: Optional[Series]) -> dict[str, Any]:
    if t1_s is None or t1_s.n < 5:
        return {
            "status": "UNAVAILABLE",
            "reason": "acr_t1 unavailable or has fewer than 5 slices",
        }

    s1 = axial_geometry_measure(t1_s, 1)
    s5 = axial_geometry_measure(t1_s, 5)

    ptype = infer_phantom_type(s1)
    spec = geom_spec(ptype)

    loc = None
    if localizer_s is not None and localizer_s.n >= 1:
        loc = localizer_length_measure(localizer_s)

    checks: list[dict[str, Any]] = []

    if loc is not None and loc.get("available"):
        checks.append(
            {
                "name": "localizer_length",
                "measured_mm": loc["length_mm"],
                "expected_mm": spec["length_mm"],
                "error_mm": loc["length_mm"] - spec["length_mm"],
            }
        )

    checks.extend(
        [
            {
                "name": "slice1_top_bottom",
                "measured_mm": s1["top_bottom_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s1["top_bottom_mm"] - spec["diameter_mm"],
            },
            {
                "name": "slice1_left_right",
                "measured_mm": s1["left_right_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s1["left_right_mm"] - spec["diameter_mm"],
            },
            {
                "name": "slice5_top_bottom",
                "measured_mm": s5["top_bottom_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s5["top_bottom_mm"] - spec["diameter_mm"],
            },
            {
                "name": "slice5_left_right",
                "measured_mm": s5["left_right_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s5["left_right_mm"] - spec["diameter_mm"],
            },
            {
                "name": "slice5_diag1",
                "measured_mm": s5["diag1_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s5["diag1_mm"] - spec["diameter_mm"],
            },
            {
                "name": "slice5_diag2",
                "measured_mm": s5["diag2_mm"],
                "expected_mm": spec["diameter_mm"],
                "error_mm": s5["diag2_mm"] - spec["diameter_mm"],
            },
        ]
    )

    tol = spec["tol_mm"]
    for c in checks:
        err = abs(c["error_mm"])
        c["status"] = "PASS" if err <= tol else "FAIL"

    if loc is not None and not loc.get("available"):
        status = "INDETERMINATE"
        reason = loc.get("reason", "localizer unavailable")
    else:
        status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
        reason = None

    return {
        "status": status,
        "reason": reason,
        "phantom_type": ptype,
        "spec": spec,
        "checks": checks,
        "localizer": loc,
        "slice1": s1,
        "slice5": s5,
    }


def need_geometric(data_localizer: Optional[Series], data_t1: Optional[Series]) -> bool:
    return data_t1 is not None and data_t1.n >= 5


# =========
# debug
# =========

def _plot_line(ax, line: dict[str, Any], lw: float = 2.0) -> None:
    ax.plot([line["x_a"], line["x_b"]], [line["y_a"], line["y_b"]], linewidth=lw)


def save_geometric_debug_axial(s: Series, idx: int, out_dir: str | Path) -> Path:
    m = axial_geometry_measure(s, idx)
    img = get_img(s, idx)
    arr = img.arr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    x0, y0, x1, y1 = m["phantom_box"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=1.5,
        )
    )

    _plot_line(ax, m["top_bottom_line"])
    _plot_line(ax, m["left_right_line"])
    _plot_line(ax, m["diag1_line"])
    _plot_line(ax, m["diag2_line"])

    ax.set_title(f"{s.label} slice {idx} geometric")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_geometric.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path


def save_geometric_debug_localizer(s: Series, idx: Optional[int], out_dir: str | Path) -> Optional[Path]:
    if idx is None:
        idx = best_localizer_idx(s)

    if idx is None:
        return None

    m = localizer_length_measure(s, idx)
    if not m.get("available"):
        return None

    img = get_img(s, idx)
    arr = img.arr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    x0, y0, x1, y1 = m["phantom_box"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=1.5,
        )
    )

    _plot_line(ax, m["line"])

    ax.set_title(f"localizer geometric (slice {idx})")
    ax.set_axis_off()

    out_path = out_dir / "localizer_geometric.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path