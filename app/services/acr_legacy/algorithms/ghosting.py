from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from app.services.acr_legacy.dataset import Series, get_img, to_float


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


def circle_mask(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)


def rect_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = shape
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    m = np.zeros(shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def mean_roi(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    if vals.size == 0:
        raise ValueError("empty ROI")
    return float(np.mean(vals))


# =========
# roi
# =========

def large_roi_params(arr: np.ndarray, ps: Optional[list[float]]) -> dict[str, Any]:
    x0, y0, x1, y1 = phantom_box(arr)
    cx = 0.5 * (x0 + x1 - 1)
    cy = 0.5 * (y0 + y1 - 1)

    area_px_mm2 = px_area_mm2(ps)
    if area_px_mm2 is None:
        raise ValueError("pixel spacing unavailable")

    ptype = infer_phantom_type(arr, ps)
    area_mm2 = 20000.0 if ptype == "large" else 16000.0

    r_px = np.sqrt(area_mm2 / (np.pi * area_px_mm2))

    w = x1 - x0
    h = y1 - y0
    r_max = 0.42 * min(w, h)
    r_px = min(r_px, r_max)

    return {
        "phantom_type": ptype,
        "cx": float(cx),
        "cy": float(cy),
        "r_px": float(r_px),
        "area_mm2": area_mm2,
    }


def background_roi_boxes(arr: np.ndarray, ps: Optional[list[float]]) -> dict[str, tuple[int, int, int, int]]:
    h, w = arr.shape
    x0, y0, x1, y1 = phantom_box(arr)

    pa = px_area_mm2(ps)
    if pa is None:
        raise ValueError("pixel spacing unavailable")

    target_area_px = 1000.0 / pa

    # top/bottom: horizontal 4:1
    bh = max(3, int(np.sqrt(target_area_px / 4.0)))
    bw = max(12, int(4.0 * bh))

    # left/right: vertical 4:1
    vw = max(3, int(np.sqrt(target_area_px / 4.0)))
    vh = max(12, int(4.0 * vw))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    top_gap = y0
    bottom_gap = h - y1
    left_gap = x0
    right_gap = w - x1

    bh = min(bh, max(3, int(0.8 * top_gap)), max(3, int(0.8 * bottom_gap)))
    bw = min(bw, max(12, int(0.6 * (x1 - x0))))

    vw = min(vw, max(3, int(0.8 * left_gap)), max(3, int(0.8 * right_gap)))
    vh = min(vh, max(12, int(0.6 * (y1 - y0))))

    top_cy = int(0.5 * y0)
    btm_cy = int(0.5 * (y1 + h))
    left_cx = int(0.5 * x0)
    right_cx = int(0.5 * (x1 + w))

    top = (
        int(cx - bw / 2),
        int(top_cy - bh / 2),
        int(cx + bw / 2),
        int(top_cy + bh / 2),
    )
    bottom = (
        int(cx - bw / 2),
        int(btm_cy - bh / 2),
        int(cx + bw / 2),
        int(btm_cy + bh / 2),
    )
    left = (
        int(left_cx - vw / 2),
        int(cy - vh / 2),
        int(left_cx + vw / 2),
        int(cy + vh / 2),
    )
    right = (
        int(right_cx - vw / 2),
        int(cy - vh / 2),
        int(right_cx + vw / 2),
        int(cy + vh / 2),
    )

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


# =========
# measure
# =========

def ghosting_measure(s: Series, idx: int = 7) -> dict[str, Any]:
    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    roi = large_roi_params(arr, d.ps)
    boxes = background_roi_boxes(arr, d.ps)

    large_m = circle_mask(arr.shape, roi["cx"], roi["cy"], roi["r_px"])
    top_m = rect_mask(arr.shape, *boxes["top"])
    btm_m = rect_mask(arr.shape, *boxes["bottom"])
    left_m = rect_mask(arr.shape, *boxes["left"])
    right_m = rect_mask(arr.shape, *boxes["right"])

    large_mean = mean_roi(arr, large_m)
    top_mean = mean_roi(arr, top_m)
    btm_mean = mean_roi(arr, btm_m)
    left_mean = mean_roi(arr, left_m)
    right_mean = mean_roi(arr, right_m)

    ratio = abs(((top_mean + btm_mean) - (left_mean + right_mean)) / (2.0 * large_mean))
    psg = ratio * 100.0

    status = "FAIL" if ratio > 0.030 else "PASS"

    return {
        "idx": idx,
        "inst": d.inst,
        "phantom_type": roi["phantom_type"],
        "large_mean": large_mean,
        "top_mean": top_mean,
        "bottom_mean": btm_mean,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "ghosting_ratio": ratio,
        "psg_percent": psg,
        "status": status,
        "large_roi": roi,
        "bg_boxes": boxes,
        "shape": tuple(arr.shape),
        "ps": d.ps,
    }


def need_ghosting(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 7


# =========
# debug
# =========

def save_ghosting_debug(s: Series, idx: int, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    m = ghosting_measure(s, idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    lr = m["large_roi"]
    boxes = m["bg_boxes"]

    ax.add_patch(
        Circle(
            (lr["cx"], lr["cy"]),
            lr["r_px"],
            fill=False,
            linewidth=1.5,
        )
    )

    for name in ["top", "bottom", "left", "right"]:
        x0, y0, x1, y1 = boxes[name]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=1.5,
                linestyle="--",
            )
        )

    ax.set_title(f"{s.label} slice {idx} ghosting")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_ghosting.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path