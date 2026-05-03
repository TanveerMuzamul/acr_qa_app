from __future__ import annotations

"""Slice Position module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset import Series, get_img, row_mm


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


def top_roi(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = phantom_box(arr)
    w = x1 - x0
    h = y1 - y0

    # 更聚焦在顶部 slice-position wedge 的区域
    rx0 = x0 + int(0.24 * w)
    rx1 = x1 - int(0.24 * w)
    ry0 = y0 + int(0.02 * h)
    ry1 = y0 + max(24, int(0.30 * h))

    rx0 = max(0, rx0)
    ry0 = max(0, ry0)
    rx1 = min(arr.shape[1], rx1)
    ry1 = min(arr.shape[0], ry1)

    return arr[ry0:ry1, rx0:rx1], (rx0, ry0, rx1, ry1)


# =========
# bars
# =========

def local_max_idx(x: np.ndarray) -> list[int]:
    out: list[int] = []
    n = len(x)
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            out.append(i)
    return out


def pick_bar_pair(profile: np.ndarray) -> tuple[int, int, np.ndarray]:
    p = smooth1d(profile, 11)
    n = len(p)

    c0 = int(0.20 * n)
    c1 = int(0.80 * n)
    core = p[c0:c1]

    peak_thr = float(np.percentile(core, 75))
    peaks = [i for i in local_max_idx(core) if core[i] >= peak_thr]
    peaks = [i + c0 for i in peaks]

    if len(peaks) < 2:
        order = np.argsort(p)[::-1]
        peaks = []
        for i in order:
            if c0 <= i < c1:
                if not peaks or min(abs(i - j) for j in peaks) >= 4:
                    peaks.append(int(i))
                if len(peaks) >= 8:
                    break

    if len(peaks) < 2:
        order = np.argsort(p)[::-1]
        a = int(order[0])
        b = int(order[1])
        if a > b:
            a, b = b, a
        return a, b, p

    center = 0.5 * (n - 1)

    # Bar spacing is usually small, but it must not be too close
    target_dx = max(5.0, 0.08 * n)

    best_pair: Optional[tuple[int, int]] = None
    best_score = -1e18

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            a = peaks[i]
            b = peaks[j]
            dx = b - a

            if dx < 3 or dx > max(20, int(0.18 * n)):
                continue

            mid = 0.5 * (a + b)
            score = float(p[a] + p[b])

            # Prefer bar pairs closer to the ROI center
            score -= 1.5 * abs(mid - center) / max(1.0, n)

            # Prefer spacing that is close to the expected target
            score -= 1.0 * abs(dx - target_dx) / max(1.0, n)

            # Penalize pairs where the two peak heights differ too much
            score -= 0.7 * abs(float(p[a] - p[b]))

            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        peaks = sorted(peaks[:2])
        return peaks[0], peaks[1], p

    return best_pair[0], best_pair[1], p


def bar_bottom(roi_n: np.ndarray, x: int) -> tuple[int, np.ndarray]:
    h, w = roi_n.shape
    xa = max(0, x - 2)
    xb = min(w, x + 3)

    dark = 1.0 - roi_n[:, xa:xb].mean(axis=1)
    dark = smooth1d(dark, 7)

    search_a = 0
    search_b = min(h, max(20, int(0.72 * h)))
    work = dark[search_a:search_b]

    thr = max(
        float(np.percentile(work, 68)),
        float(work.mean() + 0.12 * work.std())
    )

    m = work > thr

    start = None
    for i in range(len(m)):
        if m[i]:
            start = i
            break

    if start is None:
        start = int(np.argmax(work))

    bottom = start
    gap = 0
    min_len = max(8, int(0.10 * h))

    for i in range(start, len(m)):
        if m[i]:
            bottom = i
            gap = 0
        else:
            gap += 1
            if i - start >= min_len and gap >= 4:
                break

    bottom += search_a
    return int(bottom), dark

def detect_bars(arr: np.ndarray) -> dict[str, Any]:
    roi, box = top_roi(arr)
    roi_n = norm01(roi)

    h, w = roi_n.shape

    yy = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None]
    dark = 1.0 - roi_n
    col_score = (dark * yy).mean(axis=0)

    xl, xr, prof = pick_bar_pair(col_score)
    yl, yl_prof = bar_bottom(roi_n, xl)
    yr, yr_prof = bar_bottom(roi_n, xr)


    if abs(yr - yl) > max(4, int(0.06 * h)):
        y_common = max(yl, yr)
        yl = y_common
        yr = y_common

    x0, y0, x1, y1 = box

    return {
        "phantom_box": phantom_box(arr),
        "roi_box": box,
        "left_x": int(x0 + xl),
        "right_x": int(x0 + xr),
        "left_bottom_y": int(y0 + yl),
        "right_bottom_y": int(y0 + yr),
        "col_score": prof,
        "left_row_score": yl_prof,
        "right_row_score": yr_prof,
    }


# =========
# measure
# =========

def slice_pos_measure(s: Series, idx: int) -> dict[str, Any]:
    img = get_img(s, idx)
    d = img.dcm
    det = detect_bars(img.arr)

    mm = row_mm(d)
    if mm is None:
        raise ValueError("row pixel spacing unavailable")

    diff_px = det["right_bottom_y"] - det["left_bottom_y"]
    diff_mm = float(diff_px * mm)
    disp_mm = diff_mm / 2.0

    if abs(diff_mm) > 7.0:
        status = "FAIL"
    elif abs(diff_mm) > 5.0:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "idx": idx,
        "inst": d.inst,
        "left_x": det["left_x"],
        "right_x": det["right_x"],
        "left_bottom_y": det["left_bottom_y"],
        "right_bottom_y": det["right_bottom_y"],
        "bar_diff_px": int(diff_px),
        "bar_diff_mm": diff_mm,
        "slice_disp_mm": disp_mm,
        "status": status,
        "lcd_fail": idx == 11 and abs(diff_mm) > 4.0,
        "ps": d.ps,
        "shape": tuple(img.arr.shape),
        "phantom_box": det["phantom_box"],
        "roi_box": det["roi_box"],
    }


def need_slice_pos(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 11


# =========
# debug
# =========

def save_slice_pos_debug(s: Series, idx: int, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    det = detect_bars(arr)

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

    lx = det["left_x"]
    rx = det["right_x"]
    ly = det["left_bottom_y"]
    ry = det["right_bottom_y"]

    ax.axvline(lx, linewidth=1.5)
    ax.axvline(rx, linewidth=1.5)
    ax.plot([lx], [ly], marker="o", markersize=6)
    ax.plot([rx], [ry], marker="o", markersize=6)

    ax.set_title(f"{s.label} slice {idx}")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path