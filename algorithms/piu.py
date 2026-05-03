# MRI quality analysis module
# Performs specific test and returns PASS/FAIL result
from __future__ import annotations

"""Piu module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pydicom

from dataset import Series, get_img, to_float


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


def px_area_mm2(ps: Optional[list[float]]) -> Optional[float]:
    r = row_mm(ps)
    c = col_mm(ps)
    if r is None or c is None:
        return None
    return r * c


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


def infer_phantom_type(arr: np.ndarray, ps: Optional[list[float]]) -> str:
    x0, y0, x1, y1 = phantom_box(arr)
    r = row_mm(ps)
    c = col_mm(ps)
    if r is None or c is None:
        return "large"

    h_mm = (y1 - y0) * r
    w_mm = (x1 - x0) * c
    d_mm = 0.5 * (h_mm + w_mm)

    if abs(d_mm - 190.0) <= abs(d_mm - 165.0):
        return "large"
    return "medium"


def field_strength_t(s: Series) -> Optional[float]:
    try:
        ds = pydicom.dcmread(str(s.first.path), stop_before_pixels=True, force=True)
        return to_float(getattr(ds, "MagneticFieldStrength", None))
    except Exception:
        return None


def circle_mask(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)


# =========
# roi
# =========

def large_roi_params(arr: np.ndarray, ps: Optional[list[float]]) -> dict[str, Any]:
    x0, y0, x1, y1 = phantom_box(arr)
    cx = 0.5 * (x0 + x1 - 1)
    cy = 0.5 * (y0 + y1 - 1)

    ptype = infer_phantom_type(arr, ps)

    # 再保守一些，避免吃到边缘亮带
    w = x1 - x0
    h = y1 - y0
    r_px = 0.30 * min(w, h)

    area_px_mm2 = px_area_mm2(ps)
    if area_px_mm2 is None:
        raise ValueError("pixel spacing unavailable")

    area_mm2 = float(np.pi * (r_px ** 2) * area_px_mm2)

    return {
        "phantom_type": ptype,
        "cx": float(cx),
        "cy": float(cy),
        "r_px": float(r_px),
        "area_mm2": area_mm2,
    }


def small_roi_radius_px(ps: Optional[list[float]]) -> float:
    pa = px_area_mm2(ps)
    if pa is None:
        raise ValueError("pixel spacing unavailable")
    return float(np.sqrt(100.0 / (np.pi * pa)))


def safe_search_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    r_large: float,
    r_small: float,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    # 只在更中心、更安全的区域搜索
    rr_outer = max(0.0, 0.72 * r_large - r_small)
    rr_inner = 0.08 * r_large

    outer_ok = dist2 <= (rr_outer ** 2)
    inner_ok = dist2 >= (rr_inner ** 2)

    # 再加矩形约束，避免跑到上方/边缘亮带
    x_ok = np.abs(xx - cx) <= 0.30 * r_large
    y_ok = np.abs(yy - cy) <= 0.30 * r_large

    return outer_ok & inner_ok & x_ok & y_ok


def circle_offsets(r: float) -> np.ndarray:
    ri = int(np.ceil(r))
    out = []
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            if dx * dx + dy * dy <= r * r:
                out.append((dy, dx))
    return np.asarray(out, dtype=np.int32)


def best_high_low(arr: np.ndarray, large: dict[str, Any], ps: Optional[list[float]]) -> dict[str, Any]:
    r_small = small_roi_radius_px(ps)
    offs = circle_offsets(r_small)

    centers_mask = safe_search_mask(arr.shape, large["cx"], large["cy"], large["r_px"], r_small)
    ys, xs = np.where(centers_mask)

    best_low = None
    best_high = None

    h, w = arr.shape

    for cy, cx in zip(ys, xs):
        yy = cy + offs[:, 0]
        xx = cx + offs[:, 1]

        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        if not np.all(ok):
            continue

        vals = arr[yy, xx]
        m = float(np.mean(vals))

        if best_low is None or m < best_low["mean"]:
            best_low = {"cx": float(cx), "cy": float(cy), "mean": m}
        if best_high is None or m > best_high["mean"]:
            best_high = {"cx": float(cx), "cy": float(cy), "mean": m}

    if best_low is None or best_high is None:
        raise ValueError("unable to place PIU small ROIs")

    return {
        "r_small_px": float(r_small),
        "low": best_low,
        "high": best_high,
    }


# =========
# measure
# =========

def piu_limits(phantom_type: str, b0_t: Optional[float]) -> dict[str, float]:
    is_3t = b0_t is not None and b0_t >= 3.0

    if phantom_type == "large":
        if is_3t:
            return {"target": 82.0, "fail": 80.0}
        return {"target": 87.5, "fail": 85.0}

    if is_3t:
        return {"target": 85.0, "fail": 85.0}
    return {"target": 90.0, "fail": 90.0}


def piu_measure(s: Series, idx: int = 7) -> dict[str, Any]:
    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    large = large_roi_params(arr, d.ps)
    small = best_high_low(arr, large, d.ps)

    high = float(small["high"]["mean"])
    low = float(small["low"]["mean"])

    piu = 100.0 * (1.0 - ((high - low) / (high + low)))

    b0 = field_strength_t(s)
    lim = piu_limits(large["phantom_type"], b0)

    if piu < lim["fail"]:
        status = "FAIL"
    elif piu < lim["target"]:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "idx": idx,
        "inst": d.inst,
        "phantom_type": large["phantom_type"],
        "b0_t": b0,
        "high": high,
        "low": low,
        "piu": piu,
        "status": status,
        "limits": lim,
        "large_roi": large,
        "small_roi": small,
        "shape": tuple(arr.shape),
        "ps": d.ps,
    }


def need_piu(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 7


# =========
# debug
# =========

def save_piu_debug(s: Series, idx: int, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    m = piu_measure(s, idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    lr = m["large_roi"]
    sr = m["small_roi"]

    ax.add_patch(
        Circle(
            (lr["cx"], lr["cy"]),
            lr["r_px"],
            fill=False,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Circle(
            (sr["low"]["cx"], sr["low"]["cy"]),
            sr["r_small_px"],
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )
    )

    ax.add_patch(
        Circle(
            (sr["high"]["cx"], sr["high"]["cy"]),
            sr["r_small_px"],
            fill=False,
            linewidth=1.5,
            linestyle=":",
        )
    )

    ax.set_title(f"{s.label} slice {idx} PIU")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_piu.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path