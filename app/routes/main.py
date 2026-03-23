"""Main application routes for dashboard, uploads, reports, and APIs."""

import json
import os
import io
import shutil
from pathlib import Path
from datetime import datetime, UTC

from flask import Blueprint, current_app, render_template, redirect, url_for, flash, request, send_file, jsonify
from flask_login import login_required, current_user

from app import db
from app.models import Job
from app.services.dicom_service import save_uploaded_inputs, load_series_slice
from app.services.storage_service import StorageService, StorageServiceError
from app.services.qa_metrics import (
    compute_metrics_for_slice,
    build_reasoned_results,
    phantom_likeness_from_image,
    acr_lcd_spokes_total,
)


main_bp = Blueprint("main", __name__)



def _series_public_id(series_obj: dict) -> str:
    return str(series_obj.get("series_key") or "")


def _find_series(index: dict, public_or_uid: str | None):
    series_list = list(index.get("series", []))
    if not series_list:
        return None
    if not public_or_uid:
        return series_list[0]
    for s in series_list:
        if public_or_uid == s.get("series_key"):
            return s
    return None


def _resolve_series_uid(index: dict, public_or_uid: str | None) -> str | None:
    s = _find_series(index, public_or_uid)
    return s.get("series_uid") if s else None


def _job_by_share_token(token: str):
    if not token:
        return None
    try:
        return Job.query.filter_by(share_token=token).first()
    except Exception:
        return None


def _pick_best_default(index: dict, job_dir: str) -> tuple[str, int]:
    """Choose the best initial series/slice by scoring candidate slices.

    The score favors slices with more PASS metrics and fewer FAIL metrics so the
    first view is representative instead of defaulting to an arbitrary middle slice.
    """
    series_list = list(index.get("series", []))
    if not series_list:
        return "", 0

    best_uid = series_list[0].get("series_uid", "")
    best_slice = 0
    best_score = -10**9
    cache_dir = os.path.join(job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    suggest_path = os.path.join(cache_dir, "suggestion.json")
    try:
        if os.path.exists(suggest_path):
            payload = json.loads(open(suggest_path, 'r', encoding='utf-8').read())
            series_pick = payload.get('series_id') or payload.get('series_uid')
            if series_pick:
                return series_pick, int(payload.get('slice_idx') or 0)
    except Exception:
        pass

    for s in sorted(series_list, key=lambda x: -int(x.get('num_slices', 0) or 0))[:10]:
        uid = s.get('series_uid')
        n = int(s.get('num_slices', 0) or 0)
        if not uid or n <= 0:
            continue
        candidates = sorted({0, max(0, n // 2), max(0, n - 1), max(0, n // 4), max(0, (3 * n) // 4)})
        for si in candidates:
            try:
                ds, img = load_series_slice(job_dir, uid, si)
                raw = compute_metrics_for_slice(ds, img)
                desc_u = (s.get('description') or '').upper()
                if 'T2' in desc_u:
                    raw['series_kind'] = 'T2'
                elif 'T1' in desc_u:
                    raw['series_kind'] = 'T1'
                else:
                    raw['series_kind'] = 'T1'
                results = build_reasoned_results(raw)
                passed = sum(1 for m in results.get('metrics', []) if m.get('status') == 'PASS')
                failed = sum(1 for m in results.get('metrics', []) if m.get('status') == 'FAIL')
                ph = float((phantom_likeness_from_image(img) or {}).get('phantom_score') or 0.0)
                score = passed * 100 - failed * 40 + ph
                if score > best_score:
                    best_score, best_uid, best_slice = score, uid, si
            except Exception:
                continue
    try:
        picked_series = _find_series(index, best_uid)
        Path(suggest_path).write_text(json.dumps({'series_id': _series_public_id(picked_series or {}), 'slice_idx': best_slice}), encoding='utf-8')
    except Exception:
        pass
    return best_uid, best_slice


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.get("/dashboard")
@login_required
def dashboard():
    jobs = Job.query.filter_by(user_id=current_user.id).order_by(Job.created_at.desc()).all()
    # Backfill share tokens for older jobs (best-effort).
    changed = False
    for j in jobs:
        if not getattr(j, "share_token", None):
            try:
                j.ensure_share_token()
                changed = True
            except Exception:
                pass
    if changed:
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
    return render_template("dashboard.html", jobs=jobs)




@main_bp.post("/jobs/<int:job_id>/delete")
@login_required
def delete_job(job_id: int):
    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    job_dir = job.job_dir
    try:
        db.session.delete(job)
        db.session.commit()
    except Exception:
        db.session.rollback()
        flash("Could not delete job.", "danger")
        return redirect(url_for("main.dashboard"))

    if job_dir and os.path.isdir(job_dir):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass

    flash(f"Job #{job_id} deleted.", "success")
    return redirect(url_for("main.dashboard"))


@main_bp.post("/jobs/clear")
@login_required
def clear_jobs():
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    count = len(jobs)
    job_dirs = [j.job_dir for j in jobs]
    try:
        for job in jobs:
            db.session.delete(job)
        db.session.commit()
    except Exception:
        db.session.rollback()
        flash("Could not clear jobs.", "danger")
        return redirect(url_for("main.dashboard"))

    for job_dir in job_dirs:
        if job_dir and os.path.isdir(job_dir):
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass

    flash(f"Cleared {count} job(s).", "success")
    return redirect(url_for("main.dashboard"))


@main_bp.post("/upload")
@login_required
def upload():
    """Accept ZIP files, direct DICOM files, and browser folder uploads."""
    uploaded_files = [f for f in request.files.getlist("input_files") if f and (f.filename or "").strip()]
    if not uploaded_files:
        flash("Please choose at least one ZIP file, DICOM file, or study folder.", "danger")
        return redirect(url_for("main.dashboard"))

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], f"job_{current_user.id}_{ts}")
    os.makedirs(job_dir, exist_ok=True)

    try:
        index = save_uploaded_inputs(uploaded_files, job_dir)
    except Exception as e:
        flash(f"Upload failed: {e}", "danger")
        return redirect(url_for("main.dashboard"))

    total_supported = int(index.get("dicom_files_detected", 0))
    if total_supported == 0:
        flash("No DICOM files were found. Upload a DICOM study, a ZIP that contains DICOM files, or a study folder.", "danger")
        return redirect(url_for("main.dashboard"))

    # Optional backup of the raw uploaded files to S3.
    try:
        storage = StorageService()
        raw_dir = os.path.join(job_dir, "raw_uploads")
        if storage.enabled and os.path.isdir(raw_dir):
            for name in os.listdir(raw_dir):
                local_file = os.path.join(raw_dir, name)
                storage.upload_file(local_file, object_name=f"{current_user.id}/{os.path.basename(job_dir)}/{name}")
    except StorageServiceError as e:
        flash(str(e), "danger")

    job = Job(user_id=current_user.id, job_dir=job_dir, summary_json=json.dumps(index))
    job.ensure_share_token()
    db.session.add(job)
    db.session.commit()

    try:
        cache_dir = os.path.join(job.job_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        series_list = list(index.get("series", []))
        if series_list:
            first = max(series_list, key=lambda s: int(s.get("num_slices", 0) or 0))
            warm_slice = max(0, int(first.get("num_slices", 1)) // 2)
            ds, img = load_series_slice(job.job_dir, first.get("series_uid"), warm_slice)
            raw = compute_metrics_for_slice(ds, img)
            desc_u = (first.get("description") or "").upper()
            raw["series_kind"] = "T2" if "T2" in desc_u else "T1"
            results = build_reasoned_results(raw)
            safe_uid = str(first.get("series_uid")).replace("/", "_")
            with open(os.path.join(cache_dir, f"metrics_{safe_uid}_{warm_slice}.json"), "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "results": results}, f)
        
    except Exception:
        pass

    return redirect(url_for("main.report", job_id=job.id))


@main_bp.get("/report/<int:job_id>")
@login_required
def report(job_id: int):
    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        flash("No supported series found in this upload.", "danger")
        return redirect(url_for("main.dashboard"))

    requested_series_uid = request.args.get("series")
    requested_slice = request.args.get("slice")

    if requested_series_uid:
        default_series_uid = requested_series_uid
    else:
        default_series_uid, suggested_slice = _pick_best_default(index, job.job_dir)
        if requested_slice is None:
            requested_slice = str(int(suggested_slice or 0))

    series_obj = _find_series(index, default_series_uid) or index["series"][0]
    max_idx = max(0, series_obj["num_slices"] - 1)
    slice_idx = int(requested_slice or (max_idx // 2))

    return render_template(
        "report.html",
        job=job,
        index=index,
        default_series_uid=_series_public_id(series_obj),
        default_slice_idx=max(0, min(slice_idx, max_idx)),
    )


@main_bp.get("/share/<token>")
def share_report(token: str):
    """Public, read-only report page (no login required)."""
    job = _job_by_share_token(token)
    if not job:
        return render_template("error.html", title="Not found", message="Invalid or expired share link."), 404

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        return render_template("error.html", title="No data", message="No DICOM series found for this job."), 404

    requested_series_uid = request.args.get("series")
    requested_slice = request.args.get("slice")

    default_series_uid = requested_series_uid
    if not default_series_uid:
        default_series_uid, suggested_slice = _pick_best_default(index, job.job_dir)
        if requested_slice is None:
            requested_slice = str(int(suggested_slice or 0))

    series_obj = _find_series(index, default_series_uid) or index["series"][0]
    max_idx = max(0, series_obj["num_slices"] - 1)
    slice_idx = int(requested_slice or (max_idx // 2))

    return render_template(
        "share_report.html",
        job=job,
        token=token,
        index=index,
        default_series_uid=_series_public_id(series_obj),
        default_slice_idx=max(0, min(slice_idx, max_idx)),
    )


@main_bp.get("/api/report/<int:job_id>")
@login_required
def api_report(job_id: int):
    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    index = json.loads(job.summary_json or "{}")
    series_uid = request.args.get("series")
    slice_idx = request.args.get("slice", "0")

    if not series_uid:
        return jsonify({"error": "series is required"}), 400
    if not slice_idx.isdigit():
        return jsonify({"error": "slice must be an integer"}), 400
    slice_idx = int(slice_idx)

    series_obj = _find_series(index, series_uid)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404
    series_uid = str(series_obj.get("series_uid"))

    slice_idx = max(0, min(slice_idx, series_obj["num_slices"] - 1))

    # Per-slice cache: avoids recomputing metrics on every UI refresh.
    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_uid = series_uid.replace("/", "_")
    metrics_cache_path = os.path.join(cache_dir, f"metrics_{safe_uid}_{slice_idx}.json")

    raw = None
    results = None
    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            raw = cached.get("raw")
            results = cached.get("results")
        except Exception:
            raw = None
            results = None

    if raw is None or results is None:
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        raw = compute_metrics_for_slice(ds, img)

        # Determine series kind (T1 vs T2) for ACR LCD thresholds.
        desc_u = (series_obj.get("description") or "").upper()
        if "T2" in desc_u:
            raw["series_kind"] = "T2"
        elif "T1" in desc_u:
            raw["series_kind"] = "T1"
        else:
            # Fall back to TR if available
            try:
                tr = float(getattr(ds, "RepetitionTime", 0) or 0)
            except Exception:
                tr = 0.0
            raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

        # Compute ACR LCD spoke count across slices 8–11 when possible (axial ACR series = 11 slices).
        # Cache per-series (not per-slice) to keep the UI snappy.
        lcd_cache_path = os.path.join(cache_dir, f"lcd_{safe_uid}.json")
        lcd_payload = None
        if os.path.exists(lcd_cache_path):
            try:
                with open(lcd_cache_path, "r", encoding="utf-8") as f:
                    lcd_payload = json.load(f)
            except Exception:
                lcd_payload = None

        if lcd_payload is None and int(series_obj.get("num_slices", 0)) >= 11:
            try:
                # 0-based indices for slices 8–11 -> 7..10
                imgs = []
                for si in range(7, 11):
                    _, im = load_series_slice(job.job_dir, series_uid, si)
                    imgs.append(im)
                lcd_payload = acr_lcd_spokes_total(imgs)
                with open(lcd_cache_path, "w", encoding="utf-8") as f:
                    json.dump(lcd_payload, f)
            except Exception:
                lcd_payload = None

        if isinstance(lcd_payload, dict):
            raw.update(lcd_payload)

        results = build_reasoned_results(raw)
        try:
            with open(metrics_cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "results": results}, f)
        except Exception:
            pass

    img_url = url_for("main.slice_png", job_id=job.id, series_id=_series_public_id(series_obj), slice_idx=slice_idx)

    return jsonify(
        {
            "dicom_files_detected": index.get("dicom_files_detected", 0),
            "series_found": len(index.get("series", [])),
            "series_id": _series_public_id(series_obj),
            "series_description": series_obj.get("description", ""),
            "num_slices_in_series": series_obj["num_slices"],
            "slice_idx": slice_idx,
            "image_shape": raw.get("image_shape"),
            "ignored_non_dicom_files": index.get("ignored_non_dicom_files", 0),
            "metrics": results.get("metrics", []),
            "overall_status": results.get("overall_status"),
            "overall_reason": results.get("overall_reason"),
            "image_url": img_url,
        }
    )


@main_bp.get("/api/share/report/<token>")
def api_share_report(token: str):
    """Public, read-only API for a shared job (no login)."""
    job = _job_by_share_token(token)
    if not job:
        return jsonify({"error": "not found"}), 404

    index = json.loads(job.summary_json or "{}")
    series_uid = request.args.get("series")
    slice_idx = request.args.get("slice", "0")

    if not series_uid:
        return jsonify({"error": "series is required"}), 400
    if not slice_idx.isdigit():
        return jsonify({"error": "slice must be an integer"}), 400
    slice_idx = int(slice_idx)

    series_obj = _find_series(index, series_uid)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404
    series_uid = str(series_obj.get("series_uid"))

    slice_idx = max(0, min(slice_idx, series_obj["num_slices"] - 1))

    # Reuse the same caching as the private API (on disk inside the job dir)
    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_uid = series_uid.replace("/", "_")
    metrics_cache_path = os.path.join(cache_dir, f"metrics_{safe_uid}_{slice_idx}.json")

    raw = None
    results = None
    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            raw = cached.get("raw")
            results = cached.get("results")
        except Exception:
            raw = None
            results = None

    if raw is None or results is None:
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        raw = compute_metrics_for_slice(ds, img)

        desc_u = (series_obj.get("description") or "").upper()
        if "T2" in desc_u:
            raw["series_kind"] = "T2"
        elif "T1" in desc_u:
            raw["series_kind"] = "T1"
        else:
            try:
                tr = float(getattr(ds, "RepetitionTime", 0) or 0)
            except Exception:
                tr = 0.0
            raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

        lcd_cache_path = os.path.join(cache_dir, f"lcd_{safe_uid}.json")
        lcd_payload = None
        if os.path.exists(lcd_cache_path):
            try:
                with open(lcd_cache_path, "r", encoding="utf-8") as f:
                    lcd_payload = json.load(f)
            except Exception:
                lcd_payload = None

        if lcd_payload is None and int(series_obj.get("num_slices", 0)) >= 11:
            try:
                imgs = []
                for si in range(7, 11):
                    _, im = load_series_slice(job.job_dir, series_uid, si)
                    imgs.append(im)
                lcd_payload = acr_lcd_spokes_total(imgs)
                with open(lcd_cache_path, "w", encoding="utf-8") as f:
                    json.dump(lcd_payload, f)
            except Exception:
                lcd_payload = None

        if isinstance(lcd_payload, dict):
            raw.update(lcd_payload)

        results = build_reasoned_results(raw)
        try:
            with open(metrics_cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "results": results}, f)
        except Exception:
            pass

    img_url = url_for("main.share_slice_png", token=token, series_id=_series_public_id(series_obj), slice_idx=slice_idx)

    return jsonify(
        {
            "dicom_files_detected": index.get("dicom_files_detected", 0),
            "series_found": len(index.get("series", [])),
            "series_id": _series_public_id(series_obj),
            "series_description": series_obj.get("description", ""),
            "num_slices_in_series": series_obj["num_slices"],
            "slice_idx": slice_idx,
            "image_shape": raw.get("image_shape"),
            "ignored_non_dicom_files": index.get("ignored_non_dicom_files", 0),
            "metrics": results.get("metrics", []),
            "overall_status": results.get("overall_status"),
            "overall_reason": results.get("overall_reason"),
            "image_url": img_url,
        }
    )




@main_bp.get("/api/suggest/<int:job_id>")
@login_required
def api_suggest(job_id: int):
    """Suggest the most phantom-like series + slice for a given job.

    Many users upload entire studies with multiple series. This helper picks a good default
    series/slice to start with, so the QA report is meaningful immediately.
    The result is cached on disk per-job.
    """
    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "suggestion.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass

    index = json.loads(job.summary_json or "{}")
    series_list = list(index.get("series", []))

    # Exclude the merged series unless it's the only option.
    non_all = [s for s in series_list if s.get("series_uid") != "__ALL__"]
    if non_all:
        series_list = non_all

    # To keep this fast, only evaluate the largest N series.
    series_list = sorted(series_list, key=lambda s: -int(s.get("num_slices", 0)))[:12]

    best = {
        "series_id": (_series_public_id(series_list[0]) if series_list else None),
        "slice_idx": 0,
        "phantom_score": 0.0,
        "is_phantom_like": False,
        "series_description": (series_list[0].get("description") if series_list else ""),
    }

    for s in series_list:
        uid = s.get("series_uid")
        n = int(s.get("num_slices", 0))
        if not uid or n <= 0:
            continue

        candidates = {max(0, min(n - 1, n // 2)), max(0, min(n - 1, n // 4)), max(0, min(n - 1, (3 * n) // 4))}
        for si in sorted(candidates):
            try:
                ds, img = load_series_slice(job.job_dir, uid, int(si))
                ph = phantom_likeness_from_image(img)
                score = float(ph.get("phantom_score") or 0.0)
            except Exception:
                continue

            if score > float(best.get("phantom_score") or 0.0):
                best = {
                    "series_id": _series_public_id(s),
                    "slice_idx": int(si),
                    "phantom_score": score,
                    "is_phantom_like": bool(ph.get("is_phantom_like")),
                    "series_description": s.get("description", ""),
                }

    # If we didn't find anything convincing, still return the biggest series mid-slice.
    if best.get("series_id") is None and series_list:
        s0 = series_list[0]
        n0 = int(s0.get("num_slices", 1))
        best = {
            "series_id": _series_public_id(s0),
            "slice_idx": max(0, n0 // 2),
            "phantom_score": 0.0,
            "is_phantom_like": False,
            "series_description": s0.get("description", ""),
        }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(best, f)
    except Exception:
        pass

    return jsonify(best)

@main_bp.get("/slice/<int:job_id>/<series_id>/<int:slice_idx>.png")
@login_required
def slice_png(job_id: int, series_id: str, slice_idx: int):
    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    index = json.loads(job.summary_json or "{}")
    series_obj = _find_series(index, series_id)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404
    series_uid = str(series_obj.get("series_uid"))

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    safe_id = _series_public_id(series_obj).replace("/", "_")
    png_path = os.path.join(cache_dir, f"slice_{safe_id}_{slice_idx}.png")

    if not os.path.exists(png_path):
        import numpy as np
        from PIL import Image, ImageDraw
        try:
            _, img = load_series_slice(job.job_dir, series_uid, slice_idx)
            arr = np.asarray(img).astype(np.float32)
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr8, mode="L")
        except Exception:
            im = Image.new("L", (640, 480), color=242)
            draw = ImageDraw.Draw(im)
            draw.text((180, 225), "Preview unavailable", fill=90)
        im.save(png_path, format="PNG", optimize=True)

    return send_file(png_path, mimetype="image/png")


@main_bp.get("/share/slice/<token>/<series_id>/<int:slice_idx>.png")
def share_slice_png(token: str, series_id: str, slice_idx: int):
    """Public, read-only slice PNG for shared jobs."""
    job = _job_by_share_token(token)
    if not job:
        return jsonify({"error": "not found"}), 404

    index = json.loads(job.summary_json or "{}")
    series_obj = _find_series(index, series_id)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404
    series_uid = str(series_obj.get("series_uid"))

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    safe_id = _series_public_id(series_obj).replace("/", "_")
    png_path = os.path.join(cache_dir, f"slice_{safe_id}_{slice_idx}.png")

    if not os.path.exists(png_path):
        import numpy as np
        from PIL import Image, ImageDraw
        try:
            _, img = load_series_slice(job.job_dir, series_uid, slice_idx)
            arr = np.asarray(img).astype(np.float32)
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr8, mode="L")
        except Exception:
            im = Image.new("L", (640, 480), color=242)
            draw = ImageDraw.Draw(im)
            draw.text((180, 225), "Preview unavailable", fill=90)
        im.save(png_path, format="PNG", optimize=True)

    return send_file(png_path, mimetype="image/png")


@main_bp.get("/report/<int:job_id>/download.json")
@login_required
def download_report_json(job_id: int):
    """Download the currently selected series/slice report as JSON."""

    job = db.get_or_404(Job, job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        flash("No supported series found in this upload.", "danger")
        return redirect(url_for("main.dashboard"))

    series_param = request.args.get("series_id") or request.args.get("series_uid") or _series_public_id(index["series"][0])
    slice_index = int(request.args.get("slice_index", "0"))
    series_obj = _find_series(index, series_param) or index["series"][0]
    series_uid = str(series_obj.get("series_uid"))

    ds, img = load_series_slice(job.job_dir, series_uid, slice_index)
    raw = compute_metrics_for_slice(ds, img)

    # Series kind inference (T1/T2)
    series_obj = _find_series(index, series_param) or {}
    desc_u = (series_obj.get("description") or "").upper()
    if "T2" in desc_u:
        raw["series_kind"] = "T2"
    elif "T1" in desc_u:
        raw["series_kind"] = "T1"
    else:
        try:
            tr = float(getattr(ds, "RepetitionTime", 0) or 0)
        except Exception:
            tr = 0.0
        raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

    # LCD spoke total (slices 8–11) if possible
    try:
        n = int(series_obj.get("num_slices", 0))
    except Exception:
        n = 0
    if n >= 11:
        try:
            imgs = []
            for si in range(7, 11):
                _, im = load_series_slice(job.job_dir, series_uid, si)
                imgs.append(im)
            raw.update(acr_lcd_spokes_total(imgs))
        except Exception:
            pass
    results = build_reasoned_results(raw)

    # Keep the payload stable and reusable: job + selection + metric values + pass/fail reasoning.
    payload = {
        "job": {
            "id": job.id,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            # NOTE: do not rely on optional DB columns (like "zip_name").
            # This project uses create_all() without migrations.
        },
        "selection": {"series_id": _series_public_id(series_obj), "series_description": series_obj.get("description"), "slice_index": slice_index},
        # Keep both raw + reasoned results to make the file useful in other tools.
        "raw": raw,
        "overall_status": results.get("overall_status"),
        "overall_reason": results.get("overall_reason"),
        "metrics_table": results.get("metrics"),
    }

    buf = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    filename = f"acr_qa_job_{job.id}_slice_{slice_index}.json"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/json")
