"""File ingestion helpers for MRI QA.

Supported inputs:
1. ZIP files containing DICOM files
2. Direct DICOM files
3. Folder uploads from the browser (relative paths are preserved)

Only real DICOM files are indexed for QA reporting. Other files are ignored.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.misc import is_dicom

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED_ZIP_EXTENSIONS = {".zip"}
EXCLUDE_SERIES_KEYWORDS = ("LOCALIZER", "SCOUT", "SURVEY", "T2*", "GRE", "MIP", "CALIB", "ASSET", "TRACE")


def _is_dicom(path: str) -> bool:
    """Return True when a path looks like a readable DICOM file."""
    if _is_normal_image(path):
        return False
    try:
        if is_dicom(path):
            return True
    except Exception:
        pass
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        return bool(getattr(ds, "SOPClassUID", None) or getattr(ds, "SeriesInstanceUID", None))
    except Exception:
        return False


def _is_normal_image(path: str) -> bool:
    """Return True for supported non-DICOM image formats."""
    return Path(path).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    """Safely extract ZIP files and block path traversal attacks."""
    os.makedirs(extract_dir, exist_ok=True)
    base = os.path.realpath(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = os.path.realpath(os.path.join(extract_dir, member.filename))
            if not target.startswith(base + os.sep) and target != base:
                raise ValueError(f"Unsafe ZIP entry detected: {member.filename}")
        archive.extractall(extract_dir)


def save_uploaded_inputs(uploaded_files, job_dir: str) -> Dict[str, Any]:
    """Save all uploaded files into the job folder and build an index.

    Parameters
    ----------
    uploaded_files:
        Iterable of Werkzeug FileStorage objects.
    job_dir:
        Job output directory.
    """
    raw_dir = Path(job_dir) / "raw_uploads"
    extracted_dir = Path(job_dir) / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for i, file_obj in enumerate(uploaded_files, start=1):
        incoming_name = str(file_obj.filename or f"upload_{i}").replace("\\", "/")
        rel_parts = [part for part in Path(incoming_name).parts if part not in {"", ".", ".."}]
        if not rel_parts:
            continue
        destination = raw_dir.joinpath(*rel_parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}_{i}{destination.suffix}")
        file_obj.save(destination)
        saved_files.append(str(destination))

    if not saved_files:
        raise ValueError("No files were uploaded.")

    all_candidate_files: List[str] = []
    zip_count = 0
    for path in saved_files:
        ext = Path(path).suffix.lower()
        if ext in ALLOWED_ZIP_EXTENSIONS:
            zip_count += 1
            target = extracted_dir / Path(path).stem
            target.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(path, str(target))
            for root, _, files in os.walk(target):
                for name in files:
                    all_candidate_files.append(str(Path(root) / name))
        else:
            rel_path = Path(path).relative_to(raw_dir)
            copied_path = extracted_dir / rel_path
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copied_path)
            all_candidate_files.append(str(copied_path))

    return index_uploaded_content(all_candidate_files, job_dir, zip_count=zip_count)


def _build_image_pseudo_dicom(image_path: str, series_uid: str, instance_number: int) -> FileDataset:
    """Wrap a normal image file in a minimal DICOM-like object.

    This keeps the rest of the application working with one consistent API.
    """
    image = Image.open(image_path).convert("L")
    pixel_array = np.asarray(image)

    dataset = Dataset()
    dataset.PatientName = "Image Upload"
    dataset.Modality = "OT"
    dataset.SeriesDescription = "Normal image upload"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.10.54321.1" if series_uid == "__NORMAL_IMAGES__" else series_uid
    dataset.InstanceNumber = instance_number
    dataset.Rows = int(pixel_array.shape[0])
    dataset.Columns = int(pixel_array.shape[1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsStored = 8
    dataset.BitsAllocated = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixel_array.tobytes()

    file_ds = FileDataset(image_path, {}, preamble=b"\0" * 128)
    for key, value in dataset.items():
        file_ds.add(value)
    return file_ds




def _make_series_key(position: int, uid: str) -> str:
    """Return a short stable series identifier safe for URLs and logs."""
    if uid == "__ALL__":
        return "all"
    if uid == "__NORMAL_IMAGES__":
        return "images"
    return f"series-{position:03d}"



def _looks_like_acr_qa_series(description: str, modality: str | None, num_slices: int) -> bool:
    desc = (description or "").upper()
    mod = (modality or "").upper()
    if mod and mod != "MR":
        return False
    if any(word in desc for word in EXCLUDE_SERIES_KEYWORDS):
        return False
    if "ACR" in desc:
        return True
    if "T1" in desc or "T2" in desc:
        return num_slices >= 7
    return num_slices >= 11


def index_uploaded_content(candidate_files: List[str], job_dir: str, zip_count: int = 0) -> Dict[str, Any]:
    """Create the series index used by the report pages."""
    dicom_files: List[str] = []
    normal_image_files: List[str] = []
    ignored_non_dicom_files: List[str] = []

    for path in candidate_files:
        if _is_dicom(path):
            dicom_files.append(path)
        elif _is_normal_image(path):
            normal_image_files.append(path)
        else:
            ignored_non_dicom_files.append(path)

    series_map: Dict[str, Dict[str, Any]] = {}
    file_meta: Dict[str, Dict[str, Any]] = {}
    detected_modality = None
    detected_field_strength_t = None

    needed_tags = [
        "SeriesInstanceUID",
        "SeriesDescription",
        "InstanceNumber",
        "ImagePositionPatient",
        "Modality",
        "MagneticFieldStrength",
    ]

    for path in dicom_files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=needed_tags)
        except Exception:
            continue

        if detected_modality is None:
            detected_modality = getattr(ds, "Modality", None)
        if detected_field_strength_t is None:
            try:
                value = getattr(ds, "MagneticFieldStrength", None)
                detected_field_strength_t = float(value) if value is not None else None
            except Exception:
                detected_field_strength_t = None

        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES"))
        desc = str(getattr(ds, "SeriesDescription", "DICOM series"))
        inst = getattr(ds, "InstanceNumber", None)
        ipp = getattr(ds, "ImagePositionPatient", None)

        ippz = None
        try:
            if ipp is not None and len(ipp) >= 3:
                ippz = float(ipp[2])
        except Exception:
            ippz = None

        file_meta[path] = {
            "uid": uid,
            "desc": desc,
            "inst": int(inst) if inst is not None else None,
            "ippz": ippz,
            "basename": os.path.basename(path),
            "kind": "dicom",
        }
        series_map.setdefault(uid, {"series_uid": uid, "description": desc, "files": [], "modality": getattr(ds, "Modality", None)})["files"].append(path)

    if normal_image_files:
        image_desc = "Image files"
        series_map["__NORMAL_IMAGES__"] = {
            "series_uid": "__NORMAL_IMAGES__",
            "description": image_desc,
            "files": list(normal_image_files),
            "modality": "OT",
        }
        for idx, path in enumerate(sorted(normal_image_files), start=1):
            file_meta[path] = {
                "uid": "__NORMAL_IMAGES__",
                "desc": image_desc,
                "inst": idx,
                "ippz": None,
                "basename": os.path.basename(path),
                "kind": "image",
            }

    def sort_key(path: str):
        meta = file_meta.get(path, {})
        if meta.get("inst") is not None:
            return (0, meta["inst"])
        if meta.get("ippz") is not None:
            return (1, meta["ippz"])
        return (2, meta.get("basename") or os.path.basename(path))

    series_list: List[Dict[str, Any]] = []
    ordered_entries = []
    for uid, entry in series_map.items():
        sorted_files = sorted(entry["files"], key=sort_key)
        ordered_entries.append((uid, entry, sorted_files))

    ordered_entries.sort(key=lambda item: (-len(item[2]), str(item[1].get("description", ""))))

    preliminary_series: List[Dict[str, Any]] = []
    for idx, (uid, entry, sorted_files) in enumerate(ordered_entries, start=1):
        preliminary_series.append(
            {
                "series_uid": uid,
                "series_key": _make_series_key(idx, uid),
                "description": entry.get("description", ""),
                "num_slices": len(sorted_files),
                "files": [os.path.relpath(path, job_dir) for path in sorted_files],
                "modality": entry.get("modality"),
            }
        )

    filtered_series = [
        s for s in preliminary_series
        if s.get("series_uid") == "__NORMAL_IMAGES__"
        or _looks_like_acr_qa_series(s.get("description", ""), s.get("modality"), int(s.get("num_slices", 0) or 0))
    ]
    if not filtered_series:
        filtered_series = preliminary_series

    for idx, item in enumerate(filtered_series, start=1):
        item["series_key"] = _make_series_key(idx, item["series_uid"])
        series_list.append(item)

    cache_path = Path(job_dir) / "cache_series_files.json"
    mapping_abs = {item["series_uid"]: [str(Path(job_dir) / rel) for rel in item["files"]] for item in series_list}
    cache_path.write_text(json.dumps(mapping_abs), encoding="utf-8")

    meta_path = Path(job_dir) / "cache" / "job_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "detected_modality": detected_modality,
                "detected_field_strength_t": detected_field_strength_t,
                "zip_files_uploaded": zip_count,
                "dicom_files_detected": len(dicom_files),
                "ignored_non_dicom_files": len(ignored_non_dicom_files),
            }
        ),
        encoding="utf-8",
    )

    return {
        "zip_files_uploaded": zip_count,
        "dicom_files_detected": len(dicom_files),
        "ignored_non_dicom_files": len(ignored_non_dicom_files),
        "series": series_list,
        "detected_modality": detected_modality,
        "detected_field_strength_t": detected_field_strength_t,
    }


def load_series_slice(job_dir: str, series_uid: str, slice_idx: int) -> Tuple[Any, Any]:
    """Load one slice from one series.

    Returns a tuple of (dataset_like_object, image_array).
    For normal image files, a pseudo DICOM dataset is created.
    """
    cache_path = Path(job_dir) / "cache_series_files.json"
    if not cache_path.exists():
        raise FileNotFoundError("Series cache file not found.")

    series_files: Dict[str, List[str]] = json.loads(cache_path.read_text(encoding="utf-8"))

    if series_uid == "__ALL__" and "__ALL__" not in series_files and series_files:
        merged: List[str] = []
        for uid, file_list in series_files.items():
            if uid == "__ALL__":
                continue
            merged.extend(file_list)
        series_files["__ALL__"] = merged

    files = series_files.get(series_uid, [])
    if not files:
        raise FileNotFoundError(f"Series not found: {series_uid}")

    slice_idx = max(0, min(slice_idx, len(files) - 1))
    selected_file = files[slice_idx]

    if _is_dicom(selected_file):
        ds = pydicom.dcmread(selected_file, force=True)
        return ds, ds.pixel_array
    if _is_normal_image(selected_file):
        ds = _build_image_pseudo_dicom(selected_file, series_uid, slice_idx + 1)
        return ds, np.asarray(Image.open(selected_file).convert("L"))

    raise ValueError(f"Unsupported file type: {selected_file}")
