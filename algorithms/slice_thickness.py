# MRI quality analysis module
# Performs specific test and returns PASS/FAIL result
from __future__ import annotations

"""Slice Thickness module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset import Series, get_img, to_float


# =========
# utils
# =========

def smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode="same")


def norm01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1))
    hi = float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def col_mm(ps: Optional[list[float]]) -> Optional[float]:
    if not ps or len(ps) < 2:
        return None
    return to_float(ps[1])


# =========
# roi
# =========

def phantom_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    a = arr.astype(np.float32)
    pos = a[a > np.min(a)]
    if pos.size == 0:
        h, w = a.shape
        return 0, 0, w, h

    p99 = float(np.percentile(pos, 99))
    thr = max(p99 * 0.08, 1.0)
    mask = a > thr
    ys, xs = np.where(mask)

    if ys.size == 0 or xs.size == 0:
        h, w = a.shape
        return 0, 0, w, h

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    h, w = a.shape
    pad_x = max(2, int(0.02 * (x1 - x0)))
    pad_y = max(2, int(0.02 * (y1 - y0)))

    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    return x0, y0, x1, y1


def insert_roi(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    px0, py0, px1, py1 = phantom_box(arr)
    pw = px1 - px0
    ph = py1 - py0

    # 只取 slice-thickness wedge 所在的左侧区域
    rx0 = px0 + int(0.02 * pw)
    rx1 = px0 + int(0.38 * pw)

    # 竖向围绕黑色水平结构附近
    ry0 = py0 + int(0.30 * ph)
    ry1 = py0 + int(0.58 * ph)

    roi = arr[ry0:ry1, rx0:rx1].astype(np.float32)
    return roi, (rx0, ry0, rx1, ry1)


# =========
# ramps
# =========

def expand_from_peak(prof: np.ndarray, peak_idx: int, thr: float, x0: int, x1: int) -> tuple[int, int]:
    left = peak_idx
    right = peak_idx

    while left > x0 and prof[left - 1] >= thr:
        left -= 1
    while right < x1 - 1 and prof[right + 1] >= thr:
        right += 1

    return left, right


def ramp_profile(roi_n: np.ndarray, half: str) -> dict[str, Any]:
    h, w = roi_n.shape

    if half == "top":
        ya = 0
        yb = max(1, h // 2)
    else:
        ya = h // 2
        yb = h

    part = roi_n[ya:yb, :]

    # The left wedge is expected across most of the ROI width
    x0 = int(0.05 * w)
    x1 = int(0.95 * w)

    # Use a wider window to find the brightest row in this half
    row_sig = smooth1d(part[:, x0:x1].mean(axis=1), 9)
    y_rel = int(np.argmax(row_sig))
    y = ya + y_rel

    # Average multiple nearby rows for a more stable profile
    ra = max(0, y - 5)
    rb = min(h, y + 6)

    prof = roi_n[ra:rb, :].mean(axis=0)
    prof = smooth1d(prof, 15)

    work = prof[x0:x1]
    base = float(np.percentile(work, 20))
    peak = float(np.percentile(work, 99))

    # 再降低阈值
    thr = base + 0.08 * (peak - base)

    xm = int(np.argmax(work)) + x0
    xa, xb = expand_from_peak(prof, xm, thr, x0, x1)

    # 过短时再放宽一次
    if (xb - xa + 1) < 45:
        thr2 = base + 0.05 * (peak - base)
        xa, xb = expand_from_peak(prof, xm, thr2, x0, x1)
        thr = thr2

    length_px = float(xb - xa + 1)

    return {
        "y": int(y),
        "row_a": int(ra),
        "row_b": int(rb),
        "x_a": int(xa),
        "x_b": int(xb),
        "len_px": length_px,
        "prof": prof,
        "thr": thr,
    }


def detect_ramps(arr: np.ndarray) -> dict[str, Any]:
    roi, box = insert_roi(arr)
    roi_n = norm01(roi)

    top = ramp_profile(roi_n, "top")
    bottom = ramp_profile(roi_n, "bottom")

    x0, y0, x1, y1 = box

    return {
        "phantom_box": phantom_box(arr),
        "roi_box": box,
        "top_y": y0 + top["y"],
        "bottom_y": y0 + bottom["y"],
        "top_x_a": x0 + top["x_a"],
        "top_x_b": x0 + top["x_b"],
        "bottom_x_a": x0 + bottom["x_a"],
        "bottom_x_b": x0 + bottom["x_b"],
        "top_len_px": top["len_px"],
        "bottom_len_px": bottom["len_px"],
        "top_row_a": y0 + top["row_a"],
        "top_row_b": y0 + top["row_b"],
        "bottom_row_a": y0 + bottom["row_a"],
        "bottom_row_b": y0 + bottom["row_b"],
    }


# =========
# measure
# =========

def slice_thickness_measure(s: Series, idx: int = 1) -> dict[str, Any]:
    img = get_img(s, idx)
    d = img.dcm
    det = detect_ramps(img.arr)

    mm = col_mm(d.ps)
    if mm is None:
        raise ValueError("column pixel spacing unavailable")

    top_mm = float(det["top_len_px"] * mm)
    bottom_mm = float(det["bottom_len_px"] * mm)

    if top_mm + bottom_mm <= 0:
        raise ValueError("invalid ramp lengths")

    thick_mm = 0.2 * (top_mm * bottom_mm) / (top_mm + bottom_mm)

    if thick_mm < 4.0 or thick_mm > 6.0:
        status = "FAIL"
    elif thick_mm < 4.3 or thick_mm > 5.7:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "idx": idx,
        "inst": d.inst,
        "top_len_mm": top_mm,
        "bottom_len_mm": bottom_mm,
        "thickness_mm": thick_mm,
        "status": status,
        "phantom_box": det["phantom_box"],
        "roi_box": det["roi_box"],
        "top_y": det["top_y"],
        "bottom_y": det["bottom_y"],
        "top_x_a": det["top_x_a"],
        "top_x_b": det["top_x_b"],
        "bottom_x_a": det["bottom_x_a"],
        "bottom_x_b": det["bottom_x_b"],
        "top_row_a": det["top_row_a"],
        "top_row_b": det["top_row_b"],
        "bottom_row_a": det["bottom_row_a"],
        "bottom_row_b": det["bottom_row_b"],
        "ps": d.ps,
        "shape": tuple(img.arr.shape),
    }


def need_slice_thickness(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 1


# =========
# debug
# =========

def save_slice_thickness_debug(s: Series, idx: int, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    det = detect_ramps(arr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pbox = det["phantom_box"]
    rbox = det["roi_box"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    px0, py0, px1, py1 = pbox
    rx0, ry0, rx1, ry1 = rbox

    ax.add_patch(
        Rectangle(
            (px0, py0),
            px1 - px0,
            py1 - py0,
            fill=False,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Rectangle(
            (rx0, ry0),
            rx1 - rx0,
            ry1 - ry0,
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )
    )

    ax.hlines(det["top_y"], det["top_x_a"], det["top_x_b"], linewidth=2.0)
    ax.hlines(det["bottom_y"], det["bottom_x_a"], det["bottom_x_b"], linewidth=2.0)

    ax.add_patch(
        Rectangle(
            (det["top_x_a"], det["top_row_a"]),
            det["top_x_b"] - det["top_x_a"],
            det["top_row_b"] - det["top_row_a"],
            fill=False,
            linewidth=1.2,
            linestyle=":",
        )
    )

    ax.add_patch(
        Rectangle(
            (det["bottom_x_a"], det["bottom_row_a"]),
            det["bottom_x_b"] - det["bottom_x_a"],
            det["bottom_row_b"] - det["bottom_row_a"],
            fill=False,
            linewidth=1.2,
            linestyle=":",
        )
    )

    ax.set_title(f"{s.label} slice {idx} thickness")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_thickness.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path