from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Optional, Any
import math

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


# =========
# data
# =========

@dataclass
class Dcm:
    path: Path
    suid: str
    iuid: str

    desc: str = ""
    proto: str = ""
    seq: str = ""
    scan_seq: str = ""

    inst: Optional[int] = None
    echo_no: Optional[int] = None
    te: Optional[float] = None
    tr: Optional[float] = None
    etl: Optional[int] = None

    rows: Optional[int] = None
    cols: Optional[int] = None
    ps: Optional[list[float]] = None

    thick: Optional[float] = None
    space: Optional[float] = None

    iop: Optional[list[float]] = None
    ipp: Optional[list[float]] = None

    maker: str = ""
    mod: str = ""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Series:
    uid: str
    suid: str
    echo_no: Optional[int]
    te_key: Optional[int]

    files: list[Dcm]
    label: str = "unknown"
    score: float = 0.0

    @property
    def first(self) -> Dcm:
        return self.files[0]

    @property
    def n(self) -> int:
        return len(self.files)


@dataclass
class Img:
    dcm: Dcm
    idx: int
    arr: np.ndarray


# =========
# utils
# =========

def g(ds, key: str, default=None):
    return getattr(ds, key, default)


def to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def to_list(x) -> Optional[list]:
    if x is None:
        return None
    try:
        return list(x)
    except Exception:
        return None


def txt(*parts: str) -> str:
    return " ".join((p or "") for p in parts).strip().lower()


def near(x: Optional[float], y: float, tol: float) -> bool:
    return x is not None and abs(x - y) <= tol


# =========
# read
# =========

def read_one(path: Path) -> Optional[Dcm]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, Exception):
        return None

    suid = str(g(ds, "SeriesInstanceUID", "") or "")
    iuid = str(g(ds, "SOPInstanceUID", "") or "")
    mod = str(g(ds, "Modality", "") or "")

    if not suid or not iuid:
        return None

    return Dcm(
        path=path,
        suid=suid,
        iuid=iuid,
        desc=str(g(ds, "SeriesDescription", "") or ""),
        proto=str(g(ds, "ProtocolName", "") or ""),
        seq=str(g(ds, "SequenceName", "") or ""),
        scan_seq=str(g(ds, "ScanningSequence", "") or ""),
        inst=to_int(g(ds, "InstanceNumber")),
        echo_no=to_int(g(ds, "EchoNumbers")),
        te=to_float(g(ds, "EchoTime")),
        tr=to_float(g(ds, "RepetitionTime")),
        etl=to_int(g(ds, "EchoTrainLength")),
        rows=to_int(g(ds, "Rows")),
        cols=to_int(g(ds, "Columns")),
        ps=to_list(g(ds, "PixelSpacing")),
        thick=to_float(g(ds, "SliceThickness")),
        space=to_float(g(ds, "SpacingBetweenSlices")),
        iop=to_list(g(ds, "ImageOrientationPatient")),
        ipp=to_list(g(ds, "ImagePositionPatient")),
        maker=str(g(ds, "Manufacturer", "") or ""),
        mod=mod,
    )


def scan(root: str | Path) -> list[Dcm]:
    root = Path(root)
    out: list[Dcm] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        d = read_one(p)
        if d is None:
            continue
        if d.mod != "MR":
            continue
        out.append(d)

    return out


def read_ds(path: Path):
    return pydicom.dcmread(str(path), force=True)


# =========
# math
# =========

def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# =========
# spacing
# =========

def gap_mm(d: Dcm) -> Optional[float]:
    if d.space is None or d.thick is None:
        return None
    return d.space - d.thick


def row_mm(d: Dcm) -> Optional[float]:
    if not d.ps or len(d.ps) < 1:
        return None
    return to_float(d.ps[0])


# =========
# plane
# =========

def plane(iop: Optional[list[float]]) -> str:
    if not iop or len(iop) < 6:
        return "unknown"

    row = [float(x) for x in iop[:3]]
    col = [float(x) for x in iop[3:6]]
    nrm = cross(row, col)

    ax = abs(nrm[0])
    ay = abs(nrm[1])
    az = abs(nrm[2])

    if az >= ax and az >= ay:
        return "axial"
    if ax >= ay and ax >= az:
        return "sagittal"
    return "coronal"


# =========
# sort
# =========

def zpos(d: Dcm) -> Optional[float]:
    if not d.iop or not d.ipp or len(d.iop) < 6 or len(d.ipp) < 3:
        return None

    row = [float(x) for x in d.iop[:3]]
    col = [float(x) for x in d.iop[3:6]]
    nrm = cross(row, col)

    s = math.sqrt(sum(x * x for x in nrm))
    if s == 0:
        return None

    nrm = [x / s for x in nrm]
    return dot(nrm, [float(x) for x in d.ipp[:3]])


def sup_pos(d: Dcm) -> Optional[float]:
    if not d.ipp or len(d.ipp) < 3:
        return None
    return float(d.ipp[2])


def sort_files(files: list[Dcm]) -> list[Dcm]:
    if not files:
        return files

    p = plane(files[0].iop)

    if p == "axial":
        def key(d: Dcm):
            z = sup_pos(d)
            i = d.inst if d.inst is not None else 10**9
            return (z if z is not None else 10**18, i, str(d.path))
        return sorted(files, key=key)

    def key(d: Dcm):
        zp = zpos(d)
        i = d.inst if d.inst is not None else 10**9
        return (zp if zp is not None else 10**18, i, str(d.path))

    return sorted(files, key=key)


# =========
# group
# =========

def group_key(d: Dcm) -> tuple:
    te_key = round(d.te) if d.te is not None else None
    return (d.suid, d.echo_no, te_key)


def group_series(files: list[Dcm]) -> list[Series]:
    mp: dict[tuple, list[Dcm]] = defaultdict(list)

    for d in files:
        mp[group_key(d)].append(d)

    out: list[Series] = []
    for (suid, echo_no, te_key), fs in mp.items():
        fs = sort_files(fs)
        uid = f"{suid}|echo={echo_no}|te={te_key}"
        out.append(
            Series(
                uid=uid,
                suid=suid,
                echo_no=echo_no,
                te_key=te_key,
                files=fs,
            )
        )

    return out


# =========
# classify
# =========

def is_fse(t: str, etl: Optional[int]) -> bool:
    keys = ["fse", "tse", "fast spin", "turbo spin", "fast/turbo"]
    if any(k in t for k in keys):
        return True
    if etl is not None and etl > 1:
        return True
    return False


def classify(s: Series) -> Series:
    d = s.first
    t = txt(d.desc, d.proto, d.seq, d.scan_seq, str(d.path.parent.name))
    p = plane(d.iop)
    n = s.n

    sc = {
        "localizer": 0.0,
        "acr_t1": 0.0,
        "acr_t2": 0.0,
        "site_t1": 0.0,
        "site_t2": 0.0,
        "unknown": 0.0,
    }

    if "localizer" in t or "localiser" in t or "3-plane" in t or "3 plane" in t or "loc" in t:
        sc["localizer"] += 5.0

    if p == "sagittal":
        sc["localizer"] += 3.0

    if n in (1, 3):
        sc["localizer"] += 2.0

    if p == "axial":
        sc["acr_t1"] += 1.0
        sc["acr_t2"] += 1.0
        sc["site_t1"] += 1.0
        sc["site_t2"] += 1.0

    if 9 <= n <= 15:
        sc["acr_t1"] += 2.0
        sc["acr_t2"] += 2.0
        sc["site_t1"] += 1.0
        sc["site_t2"] += 1.0

    if "t1" in t:
        sc["acr_t1"] += 1.5
        sc["site_t1"] += 1.5

    if "t2" in t:
        sc["acr_t2"] += 1.5
        sc["site_t2"] += 1.5

    if "site" in t or "brain" in t or "routine" in t or "clinical" in t:
        sc["site_t1"] += 1.0
        sc["site_t2"] += 1.0

    if "acr" in t:
        sc["acr_t1"] += 1.0
        sc["acr_t2"] += 1.0

    if near(d.tr, 500, 80):
        sc["acr_t1"] += 3.0
    if near(d.te, 20, 8):
        sc["acr_t1"] += 2.0

    if near(d.tr, 2000, 300):
        sc["acr_t2"] += 3.0
    if near(d.te, 80, 10):
        sc["acr_t2"] += 3.0

    if near(d.te, 20, 8) and not near(d.tr, 500, 80):
        sc["acr_t2"] -= 2.0

    if is_fse(t, d.etl):
        sc["acr_t2"] -= 4.0

    if n >= 9:
        sc["localizer"] -= 3.0

    if near(d.thick, 5.0, 1.0):
        sc["acr_t1"] += 1.0
        sc["acr_t2"] += 1.0
        sc["site_t1"] += 1.0
        sc["site_t2"] += 1.0

    if d.rows == 256 and d.cols == 256:
        sc["acr_t1"] += 0.5
        sc["acr_t2"] += 0.5
        sc["localizer"] += 0.5

    label = max(sc, key=sc.get)
    score = sc[label]

    if score < 2.0:
        label = "unknown"

    s.label = label
    s.score = score
    return s


def classify_all(ss: list[Series]) -> list[Series]:
    return [classify(s) for s in ss]


# =========
# dataset
# =========

def best(ss: list[Series]) -> Optional[Series]:
    if not ss:
        return None
    return sorted(ss, key=lambda s: (s.score, s.n), reverse=True)[0]


def build(ss: list[Series]) -> dict[str, Any]:
    mp: dict[str, list[Series]] = defaultdict(list)

    for s in ss:
        mp[s.label].append(s)

    return {
        "localizer": best(mp["localizer"]),
        "acr_t1": best(mp["acr_t1"]),
        "acr_t2": best(mp["acr_t2"]),
        "site_t1": best(mp["site_t1"]),
        "site_t2": best(mp["site_t2"]),
        "unknown": mp["unknown"],
        "all": ss,
    }


# =========
# check
# =========

def check_series(s: Optional[Series], name: str) -> None:
    if s is None:
        print(f"{name:<10}: None")
        return

    d = s.first
    msg: list[str] = []

    if name == "localizer":
        if plane(d.iop) != "sagittal":
            msg.append("plane not sagittal")
        if not (near(d.thick, 10, 2) or near(d.thick, 20, 2) or d.thick is None):
            msg.append(f"thickness not ~10 or ~20 ({d.thick})")

    elif name == "acr_t1":
        gp = gap_mm(d)

        if not near(d.tr, 500, 80):
            msg.append(f"TR not ~500 ({d.tr})")
        if not near(d.te, 20, 8):
            msg.append(f"TE not ~20 ({d.te})")
        if s.n != 11:
            msg.append(f"slice count not 11 ({s.n})")
        if not near(d.thick, 5, 1):
            msg.append(f"thickness not ~5 ({d.thick})")
        if gp is None:
            msg.append("gap unavailable")
        elif not near(gp, 5, 1.5):
            msg.append(f"gap not ~5 ({gp})")

    elif name == "acr_t2":
        gp = gap_mm(d)

        if not near(d.tr, 2000, 300):
            msg.append(f"TR not ~2000 ({d.tr})")
        if not near(d.te, 80, 10):
            msg.append(f"TE not ~80 ({d.te})")
        if s.n != 11:
            msg.append(f"slice count not 11 ({s.n})")
        if not near(d.thick, 5, 1):
            msg.append(f"thickness not ~5 ({d.thick})")
        if gp is None:
            msg.append("gap unavailable")
        elif not near(gp, 5, 1.5):
            msg.append(f"gap not ~5 ({gp})")

    if msg:
        print(f"{name:<10}: WARNING -> " + "; ".join(msg))
    else:
        print(f"{name:<10}: OK")


# =========
# pixels
# =========

def load_px(d: Dcm) -> np.ndarray:
    ds = read_ds(d.path)
    arr = ds.pixel_array.astype(np.float32)

    slope = to_float(g(ds, "RescaleSlope", 1.0))
    inter = to_float(g(ds, "RescaleIntercept", 0.0))

    if slope is None:
        slope = 1.0
    if inter is None:
        inter = 0.0

    arr = arr * slope + inter

    photo = str(g(ds, "PhotometricInterpretation", "") or "")
    if photo.upper() == "MONOCHROME1":
        arr = arr.max() - arr

    return arr


# =========
# slice
# =========

def get_dcm(s: Series, idx: int) -> Dcm:
    if idx < 1 or idx > s.n:
        raise IndexError(f"slice index out of range: {idx}, valid 1..{s.n}")
    return s.files[idx - 1]


def get_img(s: Series, idx: int) -> Img:
    d = get_dcm(s, idx)
    arr = load_px(d)
    return Img(dcm=d, idx=idx, arr=arr)


def slice_info(s: Series, idx: int) -> dict[str, Any]:
    d = get_dcm(s, idx)
    img = get_img(s, idx)
    gp = gap_mm(d)

    return {
        "idx": idx,
        "inst": d.inst,
        "path": str(d.path),
        "shape": tuple(img.arr.shape),
        "min": float(np.min(img.arr)),
        "max": float(np.max(img.arr)),
        "mean": float(np.mean(img.arr)),
        "std": float(np.std(img.arr)),
        "tr": d.tr,
        "te": d.te,
        "thick": d.thick,
        "space": d.space,
        "gap": gp,
        "ps": d.ps,
        "plane": plane(d.iop),
    }


def need_slice_pair(s: Optional[Series]) -> bool:
    return s is not None and s.n >= 11


# =========
# print
# =========

def show_series(ss: list[Series]) -> None:
    print("\n=== series ===")
    for i, s in enumerate(sorted(ss, key=lambda x: (x.label, -x.score)), start=1):
        d = s.first
        print(
            f"[{i}] "
            f"label={s.label:<10} "
            f"score={s.score:>4.1f} "
            f"n={s.n:<3d} "
            f"plane={plane(d.iop):<9} "
            f"TR={d.tr!s:<8} "
            f"TE={d.te!s:<8} "
            f"echo={d.echo_no!s:<4} "
            f"desc={d.desc}"
        )


def show_data(data: dict[str, Any]) -> None:
    print("\n=== dataset ===")
    for k in ["localizer", "acr_t1", "acr_t2", "site_t1", "site_t2"]:
        s = data[k]
        if s is None:
            print(f"{k:<10}: None")
        else:
            d = s.first
            gp = gap_mm(d)
            print(
                f"{k:<10}: "
                f"n={s.n:<3d} "
                f"TR={d.tr!s:<8} "
                f"TE={d.te!s:<8} "
                f"thick={d.thick!s:<6} "
                f"space={d.space!s:<6} "
                f"gap={gp!s:<6} "
                f"desc={d.desc}"
            )
    print(f"unknown   : {len(data['unknown'])} series")


def show_check(data: dict[str, Any]) -> None:
    print("\n=== param check ===")
    check_series(data["localizer"], "localizer")
    check_series(data["acr_t1"], "acr_t1")
    check_series(data["acr_t2"], "acr_t2")


def print_slice_info(title: str, info: dict[str, Any]) -> None:
    print(
        f"{title:<12} "
        f"idx={info['idx']:<2d} "
        f"inst={str(info['inst']):<4} "
        f"shape={info['shape']} "
        f"min={info['min']:.2f} "
        f"max={info['max']:.2f} "
        f"mean={info['mean']:.2f} "
        f"std={info['std']:.2f} "
        f"ps={info['ps']}"
    )


def show_targets(data: dict[str, Any]) -> None:
    print("\n=== target slices ===")

    for name in ["acr_t1", "acr_t2"]:
        s = data[name]
        if not need_slice_pair(s):
            print(f"{name:<12} unavailable")
            continue

        info1 = slice_info(s, 1)
        info11 = slice_info(s, 11)

        print(f"\n{name}")
        print_slice_info("slice_1", info1)
        print_slice_info("slice_11", info11)


# =========
# main
# =========

def make_dataset(root: str | Path) -> dict[str, Any]:
    files = scan(root)
    print(f"\nMR files: {len(files)}")

    ss = group_series(files)
    print(f"series: {len(ss)}")

    ss = classify_all(ss)
    show_series(ss)

    data = build(ss)
    show_data(data)
    show_check(data)
    show_targets(data)

    return data


if __name__ == "__main__":
    root = r"./sample"
    data = make_dataset(root)