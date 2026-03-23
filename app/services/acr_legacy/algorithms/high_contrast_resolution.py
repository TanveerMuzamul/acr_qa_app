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
# components
# =========

def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or vis[y, x]:
                continue

            stack = [(y, x)]
            vis[y, x] = True
            pts = []

            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                    (cy - 1, cx - 1),
                    (cy - 1, cx + 1),
                    (cy + 1, cx - 1),
                    (cy + 1, cx + 1),
                ):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))

            comps.append(np.asarray(pts, dtype=np.int32))

    return comps


def cluster_1d(vals: list[float], tol: float) -> list[list[float]]:
    if not vals:
        return []

    s = sorted(vals)
    out = [[s[0]]]

    for v in s[1:]:
        if abs(v - np.mean(out[-1])) <= tol:
            out[-1].append(v)
        else:
            out.append([v])

    return out


# =========
# roi
# =========

def resolution_insert_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    px0, py0, px1, py1 = phantom_box(arr)
    pw = px1 - px0
    ph = py1 - py0

    sx0 = px0 + int(0.20 * pw)
    sx1 = px0 + int(0.80 * pw)
    sy0 = py0 + int(0.56 * ph)
    sy1 = py0 + int(0.84 * ph)

    return sx0, sy0, sx1, sy1


def pattern_sizes_mm(phantom_type: str) -> list[float]:
    if phantom_type == "medium":
        return [1.1, 1.0, 0.9, 0.8]
    return [1.1, 1.0, 0.9]


def pattern_windows(insert_shape: tuple[int, int], phantom_type: str) -> list[tuple[int, int, int, int]]:
    h, w = insert_shape

    if phantom_type == "medium":
        fracs = [
            (0.28, 0.40),
            (0.41, 0.53),
            (0.54, 0.66),
            (0.67, 0.80),
        ]
    else:
        fracs = [
            (0.34, 0.47),
            (0.48, 0.61),
            (0.62, 0.76),
        ]

    out = []
    y0 = int(0.18 * h)
    y1 = int(0.82 * h)

    for a, b in fracs:
        x0 = int(a * w)
        x1 = int(b * w)
        out.append((x0, y0, x1, y1))

    return out


# =========
# spots
# =========

def detect_spots(sub: np.ndarray) -> list[dict[str, Any]]:
    n = norm01(sub)
    base = box_blur(n, 9)
    enh = n - base

    h, w = enh.shape
    y0 = int(0.08 * h)
    y1 = int(0.92 * h)
    x0 = int(0.08 * w)
    x1 = int(0.92 * w)

    core = enh[y0:y1, x0:x1]
    if core.size == 0:
        return []

    thr = max(0.02, float(np.percentile(core, 98.7)))
    m = enh >= thr

    keep = np.zeros_like(m, dtype=bool)
    keep[y0:y1, x0:x1] = True
    m &= keep

    comps = connected_components(m)

    out = []
    for comp in comps:
        npx = comp.shape[0]
        if npx < 1 or npx > 28:
            continue

        ys = comp[:, 0]
        xs = comp[:, 1]

        bw = int(xs.max()) - int(xs.min()) + 1
        bh = int(ys.max()) - int(ys.min()) + 1

        if bw > 8 or bh > 8:
            continue

        out.append(
            {
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "x0": int(xs.min()),
                "x1": int(xs.max()) + 1,
                "y0": int(ys.min()),
                "y1": int(ys.max()) + 1,
                "n": int(npx),
            }
        )

    return out


def subarray_ok(spots: list[dict[str, Any]], h: int, w: int) -> dict[str, Any]:
    if not spots:
        return {
            "ok": False,
            "count": 0,
            "x_clusters": 0,
            "y_clusters": 0,
        }

    xs = [s["x"] for s in spots]
    ys = [s["y"] for s in spots]

    xg = cluster_1d(xs, tol=max(1.8, 0.12 * w))
    yg = cluster_1d(ys, tol=max(1.8, 0.12 * h))

    x_clusters = len(xg)
    y_clusters = len(yg)
    count = len(spots)

    ok = count >= 4 and x_clusters >= 2 and y_clusters >= 2

    return {
        "ok": ok,
        "count": count,
        "x_clusters": x_clusters,
        "y_clusters": y_clusters,
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
        h, w = sub.shape

        upper = sub[: h // 2, :]
        lower = sub[h // 2 :, :]

        up_spots = detect_spots(upper)
        lo_spots = detect_spots(lower)

        up_score = subarray_ok(up_spots, max(1, upper.shape[0]), max(1, upper.shape[1]))
        lo_score = subarray_ok(lo_spots, max(1, lower.shape[0]), max(1, lower.shape[1]))

        if up_score["ok"]:
            rl_resolved.append(mm)
        if lo_score["ok"]:
            tb_resolved.append(mm)

        groups.append(
            {
                "mm": mm,
                "box": (x0, y0, x1, y1),
                "upper_spots": up_spots,
                "lower_spots": lo_spots,
                "upper_score": up_score,
                "lower_score": lo_score,
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

        for sp in g["upper_spots"]:
            ax.add_patch(
                Circle(
                    (ix0 + x0 + sp["x"], iy0 + y0 + sp["y"]),
                    1.3,
                    fill=False,
                    linewidth=1.0,
                )
            )

        half_h = (y1 - y0) / 2.0
        for sp in g["lower_spots"]:
            ax.add_patch(
                Circle(
                    (ix0 + x0 + sp["x"], iy0 + y0 + half_h + sp["y"]),
                    1.3,
                    fill=False,
                    linewidth=1.0,
                    linestyle="--",
                )
            )

    ax.set_title(f"{s.label} slice {idx} resolution")
    ax.set_axis_off()

    out_path = out_dir / f"{s.label}_slice_{idx}_resolution.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

    return out_path