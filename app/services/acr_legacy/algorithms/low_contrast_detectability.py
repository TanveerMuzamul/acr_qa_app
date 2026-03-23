from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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


def norm01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1))
    hi = float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode="same")


def box_blur(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1

    tmp = np.empty_like(arr, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)

    for i in range(arr.shape[0]):
        tmp[i, :] = smooth1d(arr[i, :], k)

    for j in range(arr.shape[1]):
        out[:, j] = smooth1d(tmp[:, j], k)

    return out


# =========
# geometry
# =========

def lcd_center_radius(arr: np.ndarray) -> tuple[float, float, float]:
    x0, y0, x1, y1 = phantom_box(arr)
    cx = 0.5 * (x0 + x1 - 1)
    cy = 0.5 * (y0 + y1 - 1)
    r = 0.27 * min(x1 - x0, y1 - y0)
    return float(cx), float(cy), float(r)


def disk_response(cmap: np.ndarray, x: float, y: float, r: float) -> float:
    h, w = cmap.shape
    yy, xx = np.ogrid[:h, :w]
    m = ((xx - x) ** 2 + (yy - y) ** 2) <= (r ** 2)
    vals = cmap[m]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def spoke_response(
    cmap: np.ndarray,
    cx: float,
    cy: float,
    base_r: float,
    angle_deg: float,
    ptype: str,
) -> float:
    th = np.deg2rad(angle_deg)

    if ptype == "large":
        radii = [0.55, 0.75, 0.95]
    else:
        radii = [0.52, 0.72, 0.92]

    disk_r = max(1.2, 0.08 * base_r)

    vals = []
    for rr in radii:
        x = cx + rr * base_r * np.cos(th)
        y = cy + rr * base_r * np.sin(th)
        vals.append(disk_response(cmap, x, y, disk_r))

    return float(np.mean(vals))


def calibrate_start_angle(arr: np.ndarray, ptype: str) -> float:
    n = norm01(arr)
    local = np.abs(n - box_blur(n, 25))

    cx, cy, r = lcd_center_radius(arr)

    best_angle = -80.0
    best_score = -1e18

    for ang in np.linspace(-100.0, -60.0, 81):
        spoke = spoke_response(local, cx, cy, r, ang, ptype)
        gap = spoke_response(local, cx, cy, r, ang + 18.0, ptype)
        score = spoke - gap
        if score > best_score:
            best_score = score
            best_angle = float(ang)

    return best_angle


# =========
# measure
# =========

def lcd_slice_measure(s: Series, idx: int, start_angle: float) -> dict[str, Any]:
    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    ptype = infer_phantom_type(arr, d.ps)

    n = norm01(arr)
    local = np.abs(n - box_blur(n, 25))

    cx, cy, r = lcd_center_radius(arr)

    spoke_angles = [start_angle + 36.0 * i for i in range(10)]
    gap_angles = [a + 18.0 for a in spoke_angles]

    spoke_vals = [spoke_response(local, cx, cy, r, a, ptype) for a in spoke_angles]
    gap_vals = [spoke_response(local, cx, cy, r, a, ptype) for a in gap_angles]

    noise_mean = float(np.mean(gap_vals))
    noise_std = float(np.std(gap_vals))
    thr = noise_mean + 2.0 * noise_std

    score = 0
    for v in spoke_vals:
        if v > thr:
            score += 1
        else:
            break

    return {
        "idx": idx,
        "inst": d.inst,
        "phantom_type": ptype,
        "center": (cx, cy),
        "radius": r,
        "start_angle": start_angle,
        "spoke_angles": spoke_angles,
        "spoke_vals": spoke_vals,
        "gap_vals": gap_vals,
        "threshold": thr,
        "score": score,
    }


def lcd_series_measure(s: Series) -> dict[str, Any]:
    # ACR LCD uses slices 8 through 11
    img11 = get_img(s, 11)
    ptype = infer_phantom_type(img11.arr, img11.dcm.ps)
    start_angle = calibrate_start_angle(img11.arr, ptype)

    slices = []
    total = 0
    for idx in [8, 9, 10, 11]:
        m = lcd_slice_measure(s, idx, start_angle)
        slices.append(m)
        total += int(m["score"])

    # PDF threshold
    # passing total score is at least 37
    status = "PASS" if total >= 37 else "FAIL"

    return {
        "phantom_type": ptype,
        "start_angle": start_angle,
        "slice_scores": slices,
        "total_score": total,
        "status": status,
    }


def need_lcd(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 11


# =========
# debug
# =========

def save_lcd_debug(s: Series, idx: int, start_angle: float, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    ptype = infer_phantom_type(arr, img.dcm.ps)
    cx, cy, r = lcd_center_radius(arr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    spoke_angles = [start_angle + 36.0 * i for i in range(10)]

    for a in spoke_angles:
        th = np.deg2rad(a)
        for rr in ([0.55, 0.75, 0.95] if ptype == "large" else [0.52, 0.72, 0.92]):
            x = cx + rr * r * np.cos(th)
            y = cy + rr * r * np.sin(th)
            ax.add_patch(Circle((x, y), max(1.2, 0.08 * r), fill=False, linewidth=1.0))

    ax.plot([cx], [cy], marker="o", markersize=3)
    ax.set_title(f"{s.label} slice {idx} LCD")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_lcd.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path