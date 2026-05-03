from __future__ import annotations

"""High Contrast Resolution module for the MRI ACR QA application.

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
# roi
# =========

def resolution_insert_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    px0, py0, px1, py1 = phantom_box(arr)
    pw = px1 - px0
    ph = py1 - py0

    # 更聚焦在 slice 1 下部的高对比分辨率 insert
    sx0 = px0 + int(0.23 * pw)
    sx1 = px0 + int(0.77 * pw)
    sy0 = py0 + int(0.58 * ph)
    sy1 = py0 + int(0.83 * ph)

    return sx0, sy0, sx1, sy1


def pattern_sizes_mm(phantom_type: str) -> list[float]:
    if phantom_type == "medium":
        return [1.1, 1.0, 0.9, 0.8]
    return [1.1, 1.0, 0.9]


def pattern_windows(insert_shape: tuple[int, int], phantom_type: str) -> list[tuple[int, int, int, int]]:
    h, w = insert_shape

    if phantom_type == "medium":
        fracs = [
            (0.26, 0.39),
            (0.40, 0.53),
            (0.54, 0.67),
            (0.68, 0.81),
        ]
    else:
        fracs = [
            (0.31, 0.45),
            (0.46, 0.60),
            (0.61, 0.77),
        ]

    out = []
    y0 = int(0.18 * h)
    y1 = int(0.84 * h)

    for a, b in fracs:
        x0 = int(a * w)
        x1 = int(b * w)
        out.append((x0, y0, x1, y1))

    return out


# =========
# pattern score
# =========

def stripe_score(sub: np.ndarray, axis: str) -> float:
    n = norm01(sub)
    base = box_blur(n, 7)
    enh = n - base

    if axis == "vertical":
        prof = enh.mean(axis=0)
    else:
        prof = enh.mean(axis=1)

    prof = smooth1d(prof, 5)

    if prof.size < 5:
        return 0.0

    # 用峰谷起伏度作为简单解析度指标
    amp = float(np.percentile(prof, 90) - np.percentile(prof, 10))
    var = float(np.std(prof))

    return amp + 0.5 * var


def subarray_ok(sub: np.ndarray) -> dict[str, Any]:
    h, w = sub.shape
    if h < 6 or w < 6:
        return {"ok": False, "score_v": 0.0, "score_h": 0.0, "score": 0.0}

    upper = sub[: h // 2, :]
    lower = sub[h // 2 :, :]

    # 上半区主要看一组方向，下半区看另一组方向
    score_up = stripe_score(upper, "vertical")
    score_lo = stripe_score(lower, "horizontal")

    score = float(score_up + score_lo)

    ok = score >= 0.10

    return {
        "ok": ok,
        "score_v": float(score_up),
        "score_h": float(score_lo),
        "score": score,
    }


# =========
# measure
# =========

def resolution_measure(s: Series, idx: int = 1) -> dict[str, Any]:
    img = get_img(s, idx)
    arr = img.arr
    d = img.dcm

    ptype = infer_phantom_type(arr, d.ps)
    sizes = pattern_sizes_mm(ptype)

    ix0, iy0, ix1, iy1 = resolution_insert_box(arr)
    insert = arr[iy0:iy1, ix0:ix1]
    ih, iw = insert.shape

    wins = pattern_windows((ih, iw), ptype)

    groups = []
    rl_resolved = []
    tb_resolved = []

    for mm, (x0, y0, x1, y1) in zip(sizes, wins):
        sub = insert[y0:y1, x0:x1]
        score = subarray_ok(sub)

        # 同一个模式框里分别估计 RL / TB
        # 简化判定：如果整体条纹感足够强，就认为该组 resolved
        if score["score_v"] >= 0.05:
            rl_resolved.append(mm)
        if score["score_h"] >= 0.05:
            tb_resolved.append(mm)

        groups.append(
            {
                "mm": mm,
                "box": (x0, y0, x1, y1),
                "score": score,
            }
        )

    right_left_mm = min(rl_resolved) if rl_resolved else None
    top_bottom_mm = min(tb_resolved) if tb_resolved else None

    if right_left_mm is not None and top_bottom_mm is not None:
        if right_left_mm <= 1.0 and top_bottom_mm <= 1.0:
            status = "PASS"
        else:
            status = "FAIL"
    else:
        status = "UNDETERMINED"

    return {
        "idx": idx,
        "inst": d.inst,
        "phantom_type": ptype,
        "right_left_mm": right_left_mm,
        "top_bottom_mm": top_bottom_mm,
        "status": status,
        "insert_box": (ix0, iy0, ix1, iy1),
        "groups": groups,
        "pattern_sizes": sizes,
    }


def need_resolution(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 1


# =========
# debug
# =========

def save_resolution_debug(s: Series, idx: int, out_dir: str | Path) -> Path:
    img = get_img(s, idx)
    arr = img.arr
    m = resolution_measure(s, idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray")

    ix0, iy0, ix1, iy1 = m["insert_box"]
    ax.add_patch(
        Rectangle(
            (ix0, iy0),
            ix1 - ix0,
            iy1 - iy0,
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )
    )

    for g in m["groups"]:
        x0, y0, x1, y1 = g["box"]
        ax.add_patch(
            Rectangle(
                (ix0 + x0, iy0 + y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=1.2,
            )
        )

    ax.set_title(f"{s.label} slice {idx} resolution")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_resolution.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path