"""Governance check: are the 449 UNMAPPED_LOCAL_SOURCE originals near-duplicates of locked-47 images?

THE QUESTION
------------
`data/yolo_det_3class/all_labels/` holds 11600 YOLO labels for 580 original_ids x 20
augmentations, backed by `data/血管数据集/分类数据集/image/{original_id}_{k}.jpg`.
`artifacts/audits/vascular_dataset_governance_20260830/source_case_mapping.csv` mapped
those originals onto recovered archive cases by EXACT MD5: 96 development, 32 locked,
5 unlisted, and 449 left as UNMAPPED_LOCAL_SOURCE.

MD5 only catches byte-identical copies. A re-encoded / re-compressed / resized / slightly
cropped copy of a locked-47 image would land in the 449 pile and silently leak the locked
evaluation set into training. This script quantifies that risk by perceptual matching.

METHOD
------
Signature per image: decode with cv2.imdecode(np.fromfile(...)) -- plain cv2.imread returns
None on these non-ASCII Windows paths (confirmed, do not "fix" it back). Grayscale, resize
48x48, contrast-normalise (subtract mean, divide by std), flatten -> 2304-dim vector.
Matching is squared Euclidean distance to a gallery of locked-47 (or development-186) images.

WHY A LOWE RATIO TEST AND NOT AN ABSOLUTE DISTANCE THRESHOLD
------------------------------------------------------------
An absolute distance gate was tried on this data before and rejected every true match.
Nailfold capillaroscopy frames share a global appearance (dark field, bright loops, same
capture device), so absolute distances between a true re-encode and between two unrelated
cases overlap heavily. The ratio test asks the discriminative question instead: is the best
match distinctly better than the runner-up?

One adaptation is required here and it is not cosmetic. The textbook ratio d1/d2 over raw
nearest neighbours is DEGENERATE on this gallery: one archive case contributes up to 13
frames and 52 of the 596 locked images are byte-identical duplicates of another frame in the
same case, so for a true match the runner-up is usually another frame of the SAME case and
d1/d2 -> 1.0. The gate would then reject exactly the matches it is meant to find. We
therefore compute the case-aware ratio: d1 = nearest gallery image, d2 = nearest gallery
image belonging to a DIFFERENT exam_case_id than d1's case. Both ratios are reported; the
control in step 2 decides which one is usable on this data.

CONTROL (the part that makes the result interpretable)
-----------------------------------------------------
128 originals have MD5 ground truth (96 development + 32 locked). We re-run the identical
query on them and measure whether the method recovers the known case. If it cannot, the
check is INCONCLUSIVE and a "0 leaks" output means the instrument is broken, not that the
data is clean. We also report the ratio distribution for known-true vs known-unrelated
pairs (each known group queried against the gallery it does NOT belong to) so the 0.80 gate
is calibrated rather than assumed.

Two query variants are scored, because they answer different questions:
  * aug_rep  -- the representative augmented frame `{original_id}_{k_min}.jpg`, i.e. a real
                member of the training pool. Augmentations are geometric (empirically flips
                and 180-degree rotations), and a 48x48 signature is not invariant to them.
  * preaug   -- `分类数据集/扩充之前/images/{original_id}.jpg`, the pre-augmentation source
                each augmented frame is derived from. Same underlying photograph, no
                geometric transform, so this query is not blind to augmentation.
A dihedral-8 variant of the aug_rep query (min distance over the 8 flip/rotate views of the
query) is also scored as a sensitivity analysis.

LIMITATIONS
-----------
1. A global 48x48 signature is not invariant to rotation, large crops, or heavy zoom. Step 3
   measures this blind spot directly instead of hand-waving it.
2. The check is APPEARANCE-based. If a locked-47 patient also exists in the archives under a
   different case id, or as a different frame of the same eye/finger never filed under the
   locked case, appearance matching cannot see it. Absence of a match bounds, but does not
   eliminate, patient-level leakage.
3. `rep*.jpg` files are report screenshots, not capillary frames; they are kept in the
   gallery because a leaked copy of one would still be locked data.

SAFETY
------
Read-only. No training, no model inference, no image leaves this machine. Writes only to
artifacts/evidence/leakcheck_unmapped_20260920/.

Run:  PYTHONIOENCODING=utf-8 python scripts/leakcheck_unmapped_vs_locked.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAPPING_CSV = ROOT / "artifacts/audits/vascular_dataset_governance_20260830/source_case_mapping.csv"
MANIFEST_CSV = ROOT / "artifacts/manifest/locked_evaluation_v1.csv"
AUG_DIR = ROOT / "data/血管数据集/分类数据集/image"
PREAUG_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/images"
OUT_DIR = ROOT / "artifacts/evidence/leakcheck_unmapped_20260920"

SIG_SIDE = 48
SIG_DIM = SIG_SIDE * SIG_SIDE
RATIO_GATE = 0.80
N_AUG = 20
IMG_EXT = (".jpg", ".jpeg", ".png")

def decode(path: Path) -> np.ndarray | None:
    """cv2.imread returns None on these non-ASCII paths; imdecode+fromfile is required."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def signature(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SIG_SIDE, SIG_SIDE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((small - small.mean()) / (small.std() + 1e-8)).ravel()


def dihedral8(img: np.ndarray) -> list[np.ndarray]:
    out = []
    for rot in range(4):
        r = np.rot90(img, rot).copy()
        out.append(r)
        out.append(cv2.flip(r, 1))
    return out


def sig_of_path(path: Path) -> np.ndarray | None:
    img = decode(path)
    return None if img is None else signature(img)


def read_mapping() -> list[dict]:
    with open(MAPPING_CSV, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_locked_manifest() -> tuple[list[str], list[str]]:
    with open(MANIFEST_CSV, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    locked, dev = [], []
    for r in rows:
        (locked if r["development_fold"].strip() == "" else dev).append(r["exam_case_id"])
    return locked, dev


def case_images(case_id: str) -> list[Path]:
    d = ROOT / "data" / Path(case_id)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def build_gallery(case_ids: list[str]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    sigs, meta = [], []
    for cid in case_ids:
        for p in case_images(cid):
            s = sig_of_path(p)
            if s is None:
                print(f"  WARN undecodable gallery image {p}", file=sys.stderr)
                continue
            sigs.append(s)
            meta.append((cid, p.name))
    return np.asarray(sigs, dtype=np.float32), meta


def rep_aug_path(original_id: str) -> Path | None:
    for k in range(N_AUG):
        p = AUG_DIR / f"{original_id}_{k}.jpg"
        if p.exists():
            return p
    return None


def match(query_sigs: np.ndarray, gal: np.ndarray, gal_cases: list[str]) -> list[dict]:
    """Nearest gallery image plus the case-aware runner-up.

    d2_case is the nearest gallery image whose exam_case_id differs from the best match's
    case, which is the runner-up that matters when one case contributes many near-identical
    frames. d2_raw is the textbook second nearest neighbour, reported for comparison.
    """
    cases = np.asarray(gal_cases)
    gnorm = (gal**2).sum(1)
    out = []
    for q in query_sigs:
        d = gnorm - 2.0 * (gal @ q) + float((q**2).sum())
        order = np.argsort(d, kind="stable")
        i1 = int(order[0])
        d1 = float(max(d[i1], 0.0))
        d2_raw = float(max(d[order[1]], 0.0)) if len(order) > 1 else float("inf")
        other = order[cases[order] != cases[i1]]
        if other.size:
            j = int(other[0])
            d2_case, i2_case = float(max(d[j], 0.0)), j
        else:
            d2_case, i2_case = float("inf"), -1
        out.append(
            {
                "best_idx": i1,
                "d1": d1,
                "d2_raw": d2_raw,
                "d2_case": d2_case,
                "runnerup_case_idx": i2_case,
                "ratio_raw": d1 / d2_raw if d2_raw > 0 else 1.0,
                "ratio_case": d1 / d2_case if d2_case > 0 else 1.0,
            }
        )
    return out


def match_dihedral(imgs: list[np.ndarray], gal, gal_cases) -> list[dict]:
    """Query each of the 8 flip/rotate views, keep the view with the smallest d1."""
    res = []
    for img in imgs:
        best = None
        for view in dihedral8(img):
            r = match(np.asarray([signature(view)], dtype=np.float32), gal, gal_cases)[0]
            if best is None or r["d1"] < best["d1"]:
                best = r
        res.append(best)
    return res


def pct(xs: list[float]) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(a.max()), 4),
    }


def first_case(field: str) -> str:
    return field.split(";")[0].strip() if field.strip() else ""


def degrade(img: np.ndarray, kind: str) -> np.ndarray:
    """Synthesise the exact threat: a copy that survives md5 but not byte equality."""
    h, w = img.shape[:2]
    if kind.startswith("jpeg"):
        q = int(kind.split("q")[1])
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind.startswith("resize"):
        f = float(kind.split("_")[1])
        return cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    if kind.startswith("crop"):
        f = float(kind.split("_")[1])
        dy, dx = int(h * f), int(w * f)
        return img[dy : h - dy, dx : w - dx].copy()
    raise ValueError(kind)


DEGRADATIONS = ["jpeg_q75", "jpeg_q50", "resize_0.85", "resize_0.5", "crop_0.05", "crop_0.10"]


def fullres_mae(query_path: str, gal_case: str, gal_file: str) -> float | None:
    """Full-resolution mean abs difference, gallery resized onto the query grid."""
    a = decode(Path(query_path))
    b = decode(ROOT / "data" / Path(gal_case) / gal_file)
    if a is None or b is None:
        return None
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_mapping()
    locked_cases, dev_cases = read_locked_manifest()
    print(f"manifest: {len(locked_cases)} locked cases, {len(dev_cases)} development cases")

    print("building locked gallery ...")
    gal_L, meta_L = build_gallery(locked_cases)
    cases_L = [m[0] for m in meta_L]
    print("building development gallery ...")
    gal_D, meta_D = build_gallery(dev_cases)
    cases_D = [m[0] for m in meta_D]
    print(f"gallery sizes: locked={len(meta_L)} images, development={len(meta_D)} images")

    by_status = defaultdict(list)
    for r in rows:
        by_status[r["source_mapping_status"]].append(r)

    # ---- assemble query groups -------------------------------------------------
    groups = {
        "unmapped": by_status["UNMAPPED_LOCAL_SOURCE"],
        "known_locked": by_status["EXACT_RECOVERED_LOCKED"],
        "known_dev": by_status["EXACT_RECOVERED_DEVELOPMENT"],
        "unlisted": by_status["EXACT_RECOVERED_UNLISTED"],
    }

    queries = {}  # group -> list of dicts
    for gname, grows in groups.items():
        items = []
        for r in grows:
            oid = r["original_id"]
            aug = rep_aug_path(oid)
            pre = PREAUG_DIR / f"{oid}.jpg"
            items.append(
                {
                    "original_id": oid,
                    "status": r["source_mapping_status"],
                    "truth_locked": first_case(r["locked_cases"]),
                    "truth_dev": first_case(r["development_cases"]),
                    "aug_path": str(aug) if aug else "",
                    "aug_k": int(aug.stem.split("_")[1]) if aug else -1,
                    "preaug_path": str(pre) if pre.exists() else "",
                }
            )
        queries[gname] = items

    n_no_aug = sum(1 for g in queries.values() for it in g if not it["aug_path"])
    print(f"queries: " + ", ".join(f"{k}={len(v)}" for k, v in queries.items())
          + f" (originals with no augmented frame: {n_no_aug})")

    # ---- run all query variants against both galleries --------------------------
    galleries = {"locked": (gal_L, cases_L, meta_L), "development": (gal_D, cases_D, meta_D)}
    results = defaultdict(dict)  # (group, variant, gallery) -> list
    undecodable_queries: set[tuple[str, str]] = set()
    for gname, items in queries.items():
        for variant in ("aug_rep", "preaug"):
            key = "aug_path" if variant == "aug_rep" else "preaug_path"
            idxs, sigs, imgs = [], [], []
            for i, it in enumerate(items):
                if not it[key]:
                    continue
                img = decode(Path(it[key]))
                if img is None:
                    # 418.jpg / 423.jpg are truncated at 512 KB and cannot be decoded.
                    undecodable_queries.add((it["original_id"], variant))
                    continue
                idxs.append(i)
                imgs.append(img)
                sigs.append(signature(img))
            sigs = np.asarray(sigs, dtype=np.float32)
            for galname, (gal, gcases, _) in galleries.items():
                res = match(sigs, gal, gcases) if len(idxs) else []
                results[(gname, variant, galname)] = (idxs, res)
                if variant == "aug_rep":
                    d8 = match_dihedral(imgs, gal, gcases) if len(idxs) else []
                    results[(gname, "aug_rep_d8", galname)] = (idxs, d8)
            print(f"  matched {gname}/{variant}: {len(idxs)} queries")

    # ---- control: TPR on the 128 known pairs ------------------------------------
    control = {}
    ratio_dist = defaultdict(list)
    for gname, truth_key, galname in (
        ("known_locked", "truth_locked", "locked"),
        ("known_dev", "truth_dev", "development"),
    ):
        items = queries[gname]
        for variant in ("aug_rep", "aug_rep_d8", "preaug"):
            idxs, res = results[(gname, variant, galname)]
            meta = galleries[galname][2]
            hit_top1 = hit_gated_raw = hit_gated_case = 0
            gated_raw = gated_case = 0
            for i, r in zip(idxs, res):
                truth = items[i][truth_key]
                correct = meta[r["best_idx"]][0] == truth
                hit_top1 += correct
                if r["ratio_raw"] < RATIO_GATE:
                    gated_raw += 1
                    hit_gated_raw += correct
                if r["ratio_case"] < RATIO_GATE:
                    gated_case += 1
                    hit_gated_case += correct
                tag = "true" if correct else "wrongcase"
                ratio_dist[f"{gname}/{variant}/{galname}/{tag}/ratio_raw"].append(r["ratio_raw"])
                ratio_dist[f"{gname}/{variant}/{galname}/{tag}/ratio_case"].append(r["ratio_case"])
                ratio_dist[f"{gname}/{variant}/{galname}/{tag}/d1"].append(r["d1"])
            n = len(idxs)
            control[f"{gname}/{variant}"] = {
                "gallery": galname,
                "n_queries": n,
                "top1_correct_case": hit_top1,
                "tpr_top1": round(hit_top1 / n, 4) if n else None,
                "n_passing_ratio_raw_gate": gated_raw,
                "tpr_ratio_raw_gate": round(hit_gated_raw / n, 4) if n else None,
                "n_passing_ratio_case_gate": gated_case,
                "tpr_ratio_case_gate": round(hit_gated_case / n, 4) if n else None,
            }

    # known-unrelated pairs: query each known group against the gallery it does NOT belong to
    for gname, galname in (("known_locked", "development"), ("known_dev", "locked")):
        for variant in ("aug_rep", "aug_rep_d8", "preaug"):
            idxs, res = results[(gname, variant, galname)]
            for r in res:
                ratio_dist[f"{gname}/{variant}/{galname}/unrelated/ratio_raw"].append(r["ratio_raw"])
                ratio_dist[f"{gname}/{variant}/{galname}/unrelated/ratio_case"].append(r["ratio_case"])
                ratio_dist[f"{gname}/{variant}/{galname}/unrelated/d1"].append(r["d1"])

    # ---- control 2b: DEGRADATION control ----------------------------------------
    # The 128 known pairs are byte-identical, so they only prove the method finds EXACT
    # copies (d1 == 0.0). The actual threat is a re-encoded / resized / cropped copy. We
    # synthesise those from the 32 known-locked preaug images and re-run the query.
    known_locked = queries["known_locked"]
    degradation = {}
    for kind in DEGRADATIONS:
        n = hit = gate_raw = gate_case = 0
        d1s = []
        for it in known_locked:
            if not it["preaug_path"]:
                continue
            img = decode(Path(it["preaug_path"]))
            if img is None:
                continue
            q = degrade(img, kind)
            n += 1
            r = match(np.asarray([signature(q)], dtype=np.float32), gal_L, cases_L)[0]
            ok = meta_L[r["best_idx"]][0] == it["truth_locked"]
            hit += ok
            gate_raw += ok and r["ratio_raw"] < RATIO_GATE
            gate_case += ok and r["ratio_case"] < RATIO_GATE
            d1s.append(r["d1"])
        degradation[kind] = {
            "n": n,
            "top1_correct": hit,
            "tpr_top1": round(hit / n, 4) if n else None,
            "tpr_with_ratio_raw_gate": round(gate_raw / n, 4) if n else None,
            "tpr_with_ratio_case_gate": round(gate_case / n, 4) if n else None,
            "d1_distribution": pct(d1s),
        }
        print(f"  degradation {kind}: top1 {hit}/{n}, gated(raw) {gate_raw}/{n}")

    # ---- step 3: per-augmentation recovery for the 32 known-locked ---------------
    per_aug = {}
    for k in range(N_AUG):
        n = hit = hit_gate_case = hit_gate_raw = hit_d8 = 0
        for it in known_locked:
            p = AUG_DIR / f"{it['original_id']}_{k}.jpg"
            if not p.exists():
                continue
            img = decode(p)
            n += 1
            r = match(np.asarray([signature(img)], dtype=np.float32), gal_L, cases_L)[0]
            ok = meta_L[r["best_idx"]][0] == it["truth_locked"]
            hit += ok
            hit_gate_case += ok and r["ratio_case"] < RATIO_GATE
            hit_gate_raw += ok and r["ratio_raw"] < RATIO_GATE
            rd = match_dihedral([img], gal_L, cases_L)[0]
            hit_d8 += meta_L[rd["best_idx"]][0] == it["truth_locked"]
        per_aug[str(k)] = {
            "n": n,
            "top1_correct": hit,
            "rate_top1": round(hit / n, 4) if n else None,
            "top1_correct_and_ratio_case_gate": hit_gate_case,
            "top1_correct_and_ratio_raw_gate": hit_gate_raw,
            "top1_correct_dihedral8": hit_d8,
            "rate_dihedral8": round(hit_d8 / n, 4) if n else None,
        }
        print(f"  aug k={k}: top1 {hit}/{n}, d8 {hit_d8}/{n}")

    # ---- confirmation stage: full-resolution MAE with a calibrated null ----------
    # The 48x48 ratio gate is a screen, not proof. We confirm every top-1 match at full
    # resolution. The null is the 96 EXACT_RECOVERED_DEVELOPMENT originals queried against
    # the LOCKED gallery: those are known-unrelated, so their best-match MAE floor tells us
    # what "two different nailfold images" looks like.
    mae_cache: dict[tuple[str, str, str], float | None] = {}

    def mae_for(qpath: str, case: str, fname: str) -> float | None:
        k = (qpath, case, fname)
        if k not in mae_cache:
            mae_cache[k] = fullres_mae(qpath, case, fname)
        return mae_cache[k]

    null_maes = []
    idxs_n, res_n = results[("known_dev", "preaug", "locked")]
    for i, r in zip(idxs_n, res_n):
        cid, fn = meta_L[r["best_idx"]]
        m = mae_for(queries["known_dev"][i]["preaug_path"], cid, fn)
        if m is not None:
            null_maes.append(m)
    null_floor = float(min(null_maes)) if null_maes else float("nan")
    pos_maes = []
    idxs_p, res_p = results[("known_locked", "preaug", "locked")]
    for i, r in zip(idxs_p, res_p):
        cid, fn = meta_L[r["best_idx"]]
        m = mae_for(queries["known_locked"][i]["preaug_path"], cid, fn)
        if m is not None:
            pos_maes.append(m)
    mae_control = {
        "null_is": "96 EXACT_RECOVERED_DEVELOPMENT originals vs LOCKED gallery (known unrelated)",
        "null_mae": pct(null_maes),
        "known_locked_exact_copy_mae": pct(pos_maes),
        "null_floor_used_as_confirm_threshold": round(null_floor, 2),
    }
    print(f"  MAE null floor (known-unrelated best match) = {null_floor:.2f}")

    # ---- step 1: flag unmapped originals ----------------------------------------
    flags = {}
    for variant in ("aug_rep", "aug_rep_d8", "preaug"):
        idxs, res = results[("unmapped", variant, "locked")]
        items = queries["unmapped"]
        hits_raw, hits_case = [], []
        for i, r in zip(idxs, res):
            cid, fname = meta_L[r["best_idx"]]
            rec = {
                "original_id": items[i]["original_id"],
                "variant": variant,
                "matched_locked_case": cid,
                "matched_filename": fname,
                "d1": round(r["d1"], 2),
                "d2_raw": round(r["d2_raw"], 2),
                "d2_case": round(r["d2_case"], 2),
                "ratio_raw": round(r["ratio_raw"], 4),
                "ratio_case": round(r["ratio_case"], 4),
            }
            key = "preaug_path" if variant == "preaug" else "aug_path"
            m = mae_for(items[i][key], cid, fname)
            rec["fullres_mae"] = None if m is None else round(m, 2)
            rec["confirmed_below_null_floor"] = bool(m is not None and m < null_floor)
            if r["ratio_raw"] < RATIO_GATE:
                hits_raw.append(rec)
            if r["ratio_case"] < RATIO_GATE:
                hits_case.append(rec)
        confirmed = sorted(
            (h for h in hits_raw if h["confirmed_below_null_floor"]),
            key=lambda x: x["fullres_mae"],
        )
        flags[variant] = {
            "n_queried": len(idxs),
            "n_flagged_ratio_raw_gate": len(hits_raw),
            "n_flagged_ratio_case_gate": len(hits_case),
            "n_confirmed_fullres": len(confirmed),
            "confirmed_near_duplicates": confirmed,
            "flagged_ratio_case_gate": sorted(hits_case, key=lambda x: x["ratio_case"]),
            "flagged_ratio_raw_gate": sorted(hits_raw, key=lambda x: x["ratio_raw"]),
        }
        print(
            f"  unmapped/{variant}: flagged raw-gate={len(hits_raw)} "
            f"case-gate={len(hits_case)} CONFIRMED={len(confirmed)}"
        )

    # ---- per-query CSV ----------------------------------------------------------
    csv_path = OUT_DIR / "leakcheck_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "group", "original_id", "status", "query_variant", "query_path", "gallery",
            "truth_locked_case", "truth_dev_case", "best_case", "best_filename",
            "d1", "d2_raw", "d2_case", "ratio_raw", "ratio_case",
            "best_case_equals_truth", "passes_ratio_raw_gate", "passes_ratio_case_gate",
            "fullres_mae", "confirmed_near_duplicate",
        ])
        for (gname, variant, galname), (idxs, res) in sorted(results.items()):
            items = queries[gname]
            meta = galleries[galname][2]
            key = "preaug_path" if variant == "preaug" else "aug_path"
            for i, r in zip(idxs, res):
                it = items[i]
                bc, bf = meta[r["best_idx"]]
                truth = it["truth_locked"] if galname == "locked" else it["truth_dev"]
                w.writerow([
                    gname, it["original_id"], it["status"], variant, it[key], galname,
                    it["truth_locked"], it["truth_dev"], bc, bf,
                    round(r["d1"], 2), round(r["d2_raw"], 2), round(r["d2_case"], 2),
                    round(r["ratio_raw"], 4), round(r["ratio_case"], 4),
                    "" if not truth else int(bc == truth),
                    int(r["ratio_raw"] < RATIO_GATE), int(r["ratio_case"] < RATIO_GATE),
                    "" if (mv := mae_for(it[key], bc, bf)) is None else round(mv, 2),
                    int(mv is not None and mv < null_floor and r["ratio_raw"] < RATIO_GATE),
                ])

    payload = {
        "question": "Are any of the 449 UNMAPPED_LOCAL_SOURCE originals perceptual near-duplicates of locked-47 images?",
        "method": {
            "signature": "cv2.imdecode -> gray -> 48x48 INTER_AREA -> (x-mean)/std -> 2304-d",
            "distance": "squared Euclidean",
            "gate": f"Lowe ratio d1/d2 < {RATIO_GATE}",
            "ratio_raw": "d2 = 2nd nearest gallery image (textbook)",
            "ratio_case": "d2 = nearest gallery image from a different exam_case_id than the best match",
            "why_not_absolute_threshold": "an absolute distance gate was tried on this data before and rejected every true match; nailfold frames share global appearance so absolute distances do not separate",
        },
        "inventory": {
            "locked_cases": len(locked_cases),
            "development_cases": len(dev_cases),
            "n_locked_images": len(meta_L),
            "n_development_images": len(meta_D),
            "n_mapping_rows": len(rows),
            "status_counts": {k: len(v) for k, v in by_status.items()},
            "queries_per_group": {k: len(v) for k, v in queries.items()},
            "originals_without_augmented_frames": sorted(
                it["original_id"] for g in queries.values() for it in g if not it["aug_path"]
            ),
            "undecodable_query_images": sorted(f"{oid}/{v}" for oid, v in undecodable_queries),
        },
        "step1_unmapped_vs_locked": flags,
        "step2_control_known_pairs": control,
        "step2b_degradation_control": degradation,
        "step2c_fullres_mae_control": mae_control,
        "step2_ratio_distributions": {k: pct(v) for k, v in sorted(ratio_dist.items())},
        "step3_per_augmentation_recovery_known_locked": per_aug,
        "governance": {
            "n_locked_images_read": len(meta_L),
            "images_transmitted": 0,
            "models_run": 0,
            "training_runs": 0,
            "read_only_outside_output_dir": True,
            "output_dir": str(OUT_DIR),
        },
    }
    json_path = OUT_DIR / "leakcheck.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()


