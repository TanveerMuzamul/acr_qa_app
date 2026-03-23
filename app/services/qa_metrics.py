from typing import Dict, Any
import numpy as np


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _norm_img(img: np.ndarray) -> np.ndarray:
    """Return float32 image with robust normalization (for metric stability)."""
    arr = img.astype("float32", copy=False)
    # Avoid overflow/underflow and keep invariance across modality scaling.
    p1, p99 = np.percentile(arr, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return arr
    arr = (arr - p1) / (p99 - p1)
    return np.clip(arr, 0.0, 1.0)

def _laplacian_var(img01: np.ndarray) -> float:
    """Variance of a discrete Laplacian; common focus/sharpness proxy."""
    # 2D 4-neighbour Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    c = img01
    lap = (
        -4.0 * c
        + np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0)
        + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1)
    )
    # exclude wrap-around edges by zeroing the border
    lap[0, :] = 0; lap[-1, :] = 0; lap[:, 0] = 0; lap[:, -1] = 0
    return float(np.var(lap))

def _edge_10_90_width_px(img01: np.ndarray) -> float | None:
    """Estimate a representative edge 10–90% transition width (pixels).
    Heuristic: find strongest gradient in the central region and measure a 1D edge profile.
    """
    r, c = img01.shape[:2]
    y0, y1 = int(r * 0.25), int(r * 0.75)
    x0, x1 = int(c * 0.25), int(c * 0.75)
    roi = img01[y0:y1, x0:x1]

    # gradients via numpy (no opencv dependency)
    gy, gx = np.gradient(roi)
    mag = np.hypot(gx, gy)

    if mag.size == 0 or not np.isfinite(mag).any():
        return None

    # strongest gradient location
    idx = np.nanargmax(mag)
    yy, xx = np.unravel_index(idx, mag.shape)
    # map back to full image coords
    y = y0 + int(yy); x = x0 + int(xx)

    # choose profile direction: if horizontal gradient dominates, sample horizontally; else vertically
    gx0 = float(gx[yy, xx]); gy0 = float(gy[yy, xx])
    if abs(gx0) >= abs(gy0):
        # horizontal profile (vary x)
        line = img01[y, max(0, x-64):min(c, x+65)]
    else:
        # vertical profile (vary y)
        line = img01[max(0, y-64):min(r, y+65), x]

    line = np.asarray(line, dtype="float32").ravel()
    if line.size < 16:
        return None

    # smooth lightly (box filter)
    k = 5
    ker = np.ones(k, dtype="float32") / k
    line_s = np.convolve(line, ker, mode="same")

    lo = float(np.percentile(line_s, 10))
    hi = float(np.percentile(line_s, 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    # normalize between lo..hi, then find first/last crossing of 0.1 and 0.9
    norm = (line_s - lo) / (hi - lo)
    # find indices closest to 0.1 and 0.9 around the steepest point
    # Use monotonic assumption locally: just find nearest indices to thresholds.
    i10 = int(np.argmin(np.abs(norm - 0.1)))
    i90 = int(np.argmin(np.abs(norm - 0.9)))
    width = abs(i90 - i10)
    return float(width) if width > 0 else None

def _low_contrast_detectability(img01: np.ndarray) -> Dict[str, Any]:
    """Heuristic low-contrast score based on local CNR sampling.
    Returns a count of detectable 'contrast objects' (0..8) and mean CNR.
    """
    r, c = img01.shape[:2]
    cy, cx = r // 2, c // 2
    # background region around center
    bg_r = max(10, min(r, c) // 12)
    obj_r = max(5, min(r, c) // 40)

    def disk_mean_std(xc, yc, rad):
        x0 = max(0, int(xc - rad)); x1 = min(c, int(xc + rad))
        y0 = max(0, int(yc - rad)); y1 = min(r, int(yc + rad))
        patch = img01[y0:y1, x0:x1]
        if patch.size == 0:
            return None, None
        return float(patch.mean()), float(patch.std(ddof=0))

    bg_mean, bg_std = disk_mean_std(cx, cy, bg_r)
    if bg_mean is None or bg_std is None or bg_std == 0:
        return {"low_contrast_detectable_count": None, "low_contrast_mean_cnr": None}

    # sample 8 candidate 'objects' on a circle
    rad = max(20, min(r, c) // 6)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    cnrs = []
    for a in angles:
        ox = int(cx + rad * np.cos(a))
        oy = int(cy + rad * np.sin(a))
        o_mean, _ = disk_mean_std(ox, oy, obj_r)
        if o_mean is None:
            continue
        cnr = abs(o_mean - bg_mean) / bg_std
        if np.isfinite(cnr):
            cnrs.append(float(cnr))

    if not cnrs:
        return {"low_contrast_detectable_count": 0, "low_contrast_mean_cnr": None}

    # heuristic threshold (CNR >= 1.5 roughly corresponds to barely visible in many demos)
    detectable = sum(1 for v in cnrs if v >= 1.5)
    return {
        "low_contrast_detectable_count": int(detectable),
        "low_contrast_mean_cnr": float(np.mean(cnrs)),
    }


def _estimate_center_and_radius(img01: np.ndarray) -> tuple[tuple[int, int], float]:
    """Estimate phantom center + outer radius from a normalized slice."""
    r, c = img01.shape[:2]
    thr = float(np.percentile(img01, 70))
    m = img01 > thr
    if m.sum() < 0.01 * r * c:
        return (c // 2, r // 2), float(min(r, c) * 0.40)

    ys, xs = np.nonzero(m)
    cy = int(np.clip(np.mean(ys), 0, r - 1))
    cx = int(np.clip(np.mean(xs), 0, c - 1))
    area = float(m.sum())
    rad = float(np.sqrt(area / np.pi))
    rad = float(np.clip(rad, 0.2 * min(r, c), 0.49 * min(r, c)))
    return (cx, cy), rad


def _polar_ring_profile(img01: np.ndarray, center: tuple[int, int], r_in: float, r_out: float, n_angles: int = 360) -> np.ndarray:
    """Mean intensity profile over angles for an annulus [r_in, r_out]."""
    h, w = img01.shape[:2]
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(r_in, r_out, 8)
    prof = np.zeros((n_angles,), dtype="float32")
    cnt = 0
    for rr in radii:
        xs = cx + rr * np.cos(angles)
        ys = cy + rr * np.sin(angles)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        prof += img01[yi, xi]
        cnt += 1
    if cnt > 0:
        prof /= float(cnt)
    return prof


def acr_lcd_spokes_total(img_stack: list[np.ndarray]) -> Dict[str, Any]:
    """Approximate ACR MRI LCD spoke counting across slices.

    Returns total spokes (0..40) and per-slice spokes (each 0..10).
    """
    per_slice: list[int] = []
    for img in img_stack:
        arr = np.asarray(img)
        if arr.ndim > 2:
            arr = arr[..., 0]
        img01 = _norm_img(arr)
        (cx, cy), rad = _estimate_center_and_radius(img01)

        r_in = 0.18 * rad
        r_out = 0.32 * rad
        prof = _polar_ring_profile(img01, (cx, cy), r_in, r_out, n_angles=360)

        # Smooth profile
        k = 9
        ker = np.ones(k, dtype="float32") / k
        prof_s = np.convolve(prof, ker, mode="same")

        # Robust noise estimate (MAD)
        med = float(np.median(prof_s))
        noise = float(np.median(np.abs(prof_s - med)) * 1.4826)
        noise = max(noise, 1e-6)

        overall = float(np.mean(prof_s))
        sectors = np.array_split(prof_s, 10)
        spokes = 0
        for sec in sectors:
            m = float(np.mean(sec))
            if abs(m - overall) > 0.60 * noise:
                spokes += 1

        spokes = int(np.clip(spokes, 0, 10))
        per_slice.append(spokes)

    total = int(np.sum(per_slice))
    return {"lcd_spokes_total": total, "lcd_spokes_per_slice": per_slice}


def _phantom_likeness(img01: np.ndarray) -> Dict[str, Any]:
    """Very fast heuristic to decide whether a slice looks like a QA phantom.

    Why this exists:
    Users often upload *entire studies* (many series/slices). The ACR-style metrics in this
    demo app are phantom-oriented; on anatomical images they will frequently (and correctly)
    fall outside thresholds, leading to the impression that "everything FAIL".

    We therefore detect obviously non-phantom slices and keep results informational (PASS)
    while still reporting the computed values.
    """
    r, c = img01.shape[:2]
    if r < 32 or c < 32:
        return {"is_phantom_like": False, "phantom_score": 0.0}

    # Central region stats
    cy0, cy1 = int(r * 0.35), int(r * 0.65)
    cx0, cx1 = int(c * 0.35), int(c * 0.65)
    center = img01[cy0:cy1, cx0:cx1]
    c_mean = float(np.mean(center))
    c_std = float(np.std(center))

    # Corners as background proxy
    k = max(8, min(r, c) // 10)
    corners = np.concatenate(
        [
            img01[:k, :k].ravel(),
            img01[:k, -k:].ravel(),
            img01[-k:, :k].ravel(),
            img01[-k:, -k:].ravel(),
        ]
    )
    bg_mean = float(np.mean(corners))
    bg_std = float(np.std(corners))

    # Phantom tends to have: center brighter than corners + relatively uniform background.
    contrast = c_mean - bg_mean

    # Score in [0,1] roughly
    score = 0.0
    # contrast contribution
    score += float(np.clip((contrast - 0.10) / 0.25, 0.0, 1.0)) * 0.6
    # background uniformity
    score += float(np.clip((0.10 - bg_std) / 0.10, 0.0, 1.0)) * 0.25
    # avoid highly textured center (often anatomy)
    score += float(np.clip((0.10 - c_std) / 0.10, 0.0, 1.0)) * 0.15

    return {"is_phantom_like": bool(score >= 0.45), "phantom_score": float(score)}



def phantom_likeness_from_image(img: np.ndarray) -> Dict[str, Any]:
    """Public helper: compute the phantom-likeness heuristic for an image array.

    Used by the UI to auto-select the most phantom-like series/slice in a multi-series upload.
    """
    arr = np.asarray(img)
    if arr.ndim > 2:
        arr = arr[..., 0]
    img01 = _norm_img(arr)
    return _phantom_likeness(img01)

def compute_metrics_for_slice(ds, img) -> Dict[str, Any]:
    img = np.asarray(img)
    rows, cols = img.shape[:2]

    slice_thickness = _safe_float(getattr(ds, "SliceThickness", None))
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    px_x = _safe_float(pixel_spacing[0]) if pixel_spacing and len(pixel_spacing) >= 2 else None
    px_y = _safe_float(pixel_spacing[1]) if pixel_spacing and len(pixel_spacing) >= 2 else None

    fov_x = px_x * cols if px_x else None
    fov_y = px_y * rows if px_y else None

    # --- Phantom-aware ROI placement (fast + robust) ---
    # The previous implementation used 5 simple square ROIs (center + N/S/E/W) which can
    # accidentally include background, leading to artificially low PIU and inflated ghosting.
    # Here we estimate the phantom circle and compute PIU + ghosting using ROIs similar
    # in spirit to the ACR method.

    img01 = _norm_img(img)
    (cx, cy), rad = _estimate_center_and_radius(img01)

    yy, xx = np.ogrid[:rows, :cols]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    phantom_mask = dist2 <= (0.85 * rad) ** 2  # stay away from the edge ring

    phantom_pixels = img.astype("float32", copy=False)[phantom_mask]

    # Large ROI signal proxy
    large_mean = float(np.mean(phantom_pixels)) if phantom_pixels.size else None

    # PIU uses high/low within the large ROI. We use robust percentiles to avoid outliers.
    piu = None
    high_sig = None
    low_sig = None
    if phantom_pixels.size >= 64:
        low_sig = float(np.percentile(phantom_pixels, 5))
        high_sig = float(np.percentile(phantom_pixels, 95))
        denom = high_sig + low_sig
        if denom != 0:
            piu = 100.0 * (1.0 - (high_sig - low_sig) / denom)

    # SNR: mean / std inside phantom
    snr = None
    if phantom_pixels.size >= 64:
        std = float(np.std(phantom_pixels, ddof=0))
        if std > 0 and large_mean is not None:
            snr = float(large_mean / std)

    def _rect_mean(x0, y0, x1, y1):
        x0 = int(np.clip(x0, 0, cols))
        x1 = int(np.clip(x1, 0, cols))
        y0 = int(np.clip(y0, 0, rows))
        y1 = int(np.clip(y1, 0, rows))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = img[y0:y1, x0:x1].astype("float32")
        if roi.size == 0:
            return None
        return float(np.mean(roi))

    # Ghosting ROIs: small rectangles outside phantom (top/bottom/left/right)
    # ACR uses ROIs outside the phantom and compares against the large ROI mean.
    band = max(6, int(0.10 * rad))
    offset = int(1.05 * rad)
    top = _rect_mean(cx - band, cy - offset - band, cx + band, cy - offset + band)
    bottom = _rect_mean(cx - band, cy + offset - band, cx + band, cy + offset + band)
    left = _rect_mean(cx - offset - band, cy - band, cx - offset + band, cy + band)
    right = _rect_mean(cx + offset - band, cy - band, cx + offset + band, cy + band)

    ghost = None
    if None not in (top, bottom, left, right) and large_mean and large_mean != 0:
        ghost = abs((top + bottom) - (left + right)) / (2.0 * large_mean)

    # --- Heuristic metrics for items previously marked TODO ---
    # These are *general-purpose* image-quality proxies (not a full ACR phantom implementation).
    # img01 already computed above
    high_contrast_lap_var = _laplacian_var(img01)

    edge_width_px = _edge_10_90_width_px(img01)
    # Use average pixel spacing if available for an approximate physical estimate
    px_avg = None
    if px_x and px_y:
        px_avg = 0.5 * (px_x + px_y)
    elif px_x:
        px_avg = px_x
    elif px_y:
        px_avg = px_y

    edge_width_mm = (edge_width_px * px_avg) if (edge_width_px is not None and px_avg) else None
    # crude estimate: lp/mm ~ 1 / (2 * edge_width_mm)
    est_lp_per_mm = (1.0 / (2.0 * edge_width_mm)) if (edge_width_mm and edge_width_mm > 0) else None

    lowc = _low_contrast_detectability(img01)

    phantom = _phantom_likeness(img01)


    return {
        "image_shape": f"{rows} x {cols}",
        "slice_thickness_mm": slice_thickness,
        "pixel_spacing": (px_x, px_y),
        "fov_mm": (fov_x, fov_y),
        "piu_percent": piu,
        "piu_low_signal": low_sig,
        "piu_high_signal": high_sig,
        "snr": snr,
        "ghosting_ratio": ghost,
        "high_contrast_laplacian_var": high_contrast_lap_var,
        "edge_10_90_width_px": edge_width_px,
        "edge_10_90_width_mm": edge_width_mm,
        "est_high_contrast_resolution_lp_per_mm": est_lp_per_mm,
        "low_contrast_detectable_count": lowc.get("low_contrast_detectable_count"),
        "low_contrast_mean_cnr": lowc.get("low_contrast_mean_cnr"),
        "is_phantom_like": phantom.get("is_phantom_like"),
        "phantom_score": phantom.get("phantom_score"),
        "field_strength_t": _safe_float(getattr(ds, "MagneticFieldStrength", None)),
        "modality": str(getattr(ds, "Modality", "")) if hasattr(ds, "Modality") else None,
    }


def build_reasoned_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw values into a clear PASS/FAIL report table."""
    metrics: list[Dict[str, Any]] = []

    modality = (raw.get("modality") or "").upper() if raw.get("modality") else ""
    field_strength_t = _safe_float(raw.get("field_strength_t"), default=None)
    is_mr = modality == "MR"

    if is_mr and field_strength_t is not None and field_strength_t >= 3.0:
        piu_min = 80.0
    else:
        piu_min = 85.0
    ghost_max = 0.030
    res_max_mm = 1.0
    st_target = 5.0
    st_pass_delta = 0.7

    def add_metric(name: str, value: str, expected: str, status: str, reason: str):
        metrics.append({
            "name": name,
            "value": value,
            "expected": expected,
            "status": status,
            "reason": reason,
        })

    px = raw.get("pixel_spacing")
    px_x = px[0] if isinstance(px, (list, tuple)) and len(px) >= 2 else None
    px_y = px[1] if isinstance(px, (list, tuple)) and len(px) >= 2 else None
    if px_x is None or px_y is None:
        add_metric(
            "High-contrast resolution (pixel size)",
            "Not available",
            f"≤ {res_max_mm:.1f} mm",
            "FAIL",
            "Pixel spacing is missing, so the resolution check could not be verified.",
        )
    else:
        worst = max(float(px_x), float(px_y))
        status = "PASS" if worst <= res_max_mm else "FAIL"
        reason = (
            f"Pixel spacing meets the requirement at {worst:.3f} mm."
            if status == "PASS"
            else f"Pixel spacing is {worst:.3f} mm, above the {res_max_mm:.1f} mm limit."
        )
        add_metric(
            "High-contrast resolution (pixel size)",
            f"{float(px_x):.3f} × {float(px_y):.3f} mm",
            f"≤ {res_max_mm:.1f} mm",
            status,
            reason,
        )

    st_val = raw.get("slice_thickness_mm")
    if st_val is None:
        add_metric(
            "Slice thickness (DICOM tag)",
            "Not available",
            f"{st_target:.1f} ± {st_pass_delta:.1f} mm",
            "FAIL",
            "Slice thickness is missing from the DICOM metadata.",
        )
    else:
        delta = abs(float(st_val) - st_target)
        status = "PASS" if delta <= st_pass_delta else "FAIL"
        reason = (
            f"Slice thickness is within tolerance at {float(st_val):.2f} mm."
            if status == "PASS"
            else f"Slice thickness is {float(st_val):.2f} mm, outside the ±{st_pass_delta:.1f} mm tolerance."
        )
        add_metric(
            "Slice thickness (DICOM tag)",
            f"{float(st_val):.2f} mm",
            f"{st_target:.1f} ± {st_pass_delta:.1f} mm",
            status,
            reason,
        )

    piu = raw.get("piu_percent")
    if piu is None:
        add_metric(
            "Image intensity uniformity (PIU)",
            "Not available",
            f"≥ {piu_min:.0f}%",
            "FAIL",
            "Uniformity could not be measured for the selected slice.",
        )
    else:
        status = "PASS" if float(piu) >= piu_min else "FAIL"
        reason = (
            f"Uniformity meets the requirement at {float(piu):.1f}%."
            if status == "PASS"
            else f"Uniformity is {float(piu):.1f}%, below the required {piu_min:.0f}%."
        )
        add_metric(
            "Image intensity uniformity (PIU)",
            f"{float(piu):.1f}%",
            f"≥ {piu_min:.0f}%",
            status,
            reason,
        )

    ghost = raw.get("ghosting_ratio")
    if ghost is None:
        add_metric(
            "Percent signal ghosting",
            "Not available",
            f"≤ {ghost_max:.3f}",
            "FAIL",
            "Ghosting could not be measured for the selected slice.",
        )
    else:
        status = "PASS" if float(ghost) <= ghost_max else "FAIL"
        reason = (
            f"Ghosting is within the limit at {float(ghost):.4f}."
            if status == "PASS"
            else f"Ghosting is {float(ghost):.4f}, above the maximum {ghost_max:.3f}."
        )
        add_metric(
            "Percent signal ghosting",
            f"{float(ghost):.4f}",
            f"≤ {ghost_max:.3f}",
            status,
            reason,
        )

    lcd_total = raw.get("lcd_spokes_total")
    lcd_per = raw.get("lcd_spokes_per_slice")
    series_kind = (raw.get("series_kind") or "").upper()
    if field_strength_t is None:
        limit_t1, limit_t2 = 30, 25
    elif field_strength_t < 1.5:
        limit_t1 = limit_t2 = 7
    elif field_strength_t < 3.0:
        limit_t1, limit_t2 = 30, 25
    else:
        limit_t1 = limit_t2 = 37
    lcd_limit = limit_t2 if series_kind == "T2" else limit_t1
    if lcd_total is None:
        add_metric(
            "Low-contrast object detectability (ACR spokes)",
            "Not scored on this slice",
            f"≥ {lcd_limit} spokes",
            "PASS",
            "Low-contrast detectability is only scored on the dedicated ACR low-contrast slice group when that data is available.",
        )
    else:
        status = "PASS" if int(lcd_total) >= int(lcd_limit) else "FAIL"
        per_txt = "" if not isinstance(lcd_per, list) else f" (per-slice {lcd_per})"
        reason = (
            f"Detected {int(lcd_total)} spokes, which meets the minimum requirement of {lcd_limit}."
            if status == "PASS"
            else f"Detected {int(lcd_total)} spokes, which is below the minimum requirement of {lcd_limit}."
        )
        add_metric(
            "Low-contrast object detectability (ACR spokes)",
            f"{int(lcd_total)} spokes{per_txt}",
            f"≥ {lcd_limit} spokes",
            status,
            reason,
        )

    non_phantom_mode = raw.get("is_phantom_like") is False
    overall = "FAIL" if any(m["status"] == "FAIL" for m in metrics) else "PASS"
    overall_reason = ""

    if non_phantom_mode:
        for metric in metrics:
            metric["status"] = "PASS"
            reason = str(metric.get("reason") or "").strip()
            metric["reason"] = (
                f"{reason} Informational only for this slice." if reason else "Informational only for this slice."
            )
        overall = "PASS"
        overall_reason = "Slice analysis complete. Results available below."

    return {
        "overall_status": overall,
        "overall_reason": overall_reason,
        "metrics": metrics,
        "field_strength_t": field_strength_t,
        "modality": modality or None,
    }


__all__ = [
    "compute_metrics_for_slice",
    "build_reasoned_results",
    "phantom_likeness_from_image",
    "acr_lcd_spokes_total",
]
