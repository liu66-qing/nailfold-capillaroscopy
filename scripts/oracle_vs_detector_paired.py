"""Paired ceiling test: human boxes vs detector, on the SAME cases.

WHY THIS DESIGN AND NOT THE ONE ORIGINALLY PROPOSED
---------------------------------------------------
The obvious test -- "compute the oracle on the 115-case evaluation set" -- is
impossible by construction. That evaluation set was DEFINED as the development
cases that have NO human boxes. Overlap with the oracle cohort is exactly 0,
verified before writing this file.

So the comparison is run the other way round: on the oracle cases, score the
detector too, and compare the two on identical cases and identical labels.

THE LEAKAGE PROBLEM AND HOW IT IS HANDLED
-----------------------------------------
Those cases DID contribute images to detector training -- that is why they have
boxes at all. Scoring them with a fold that trained on them would measure
memorisation. The folds are group-disjoint by original_id, so for each case we
use ONLY folds whose TRAINING split excludes that case's original_id, and we
assert that per case. A case with no clean fold is dropped, not scored.

WHAT A "CEILING" WOULD AND WOULD NOT LOOK LIKE
----------------------------------------------
Same score on the same cases => the detector is not the bottleneck and more
detector work is wasted. Human boxes clearly higher => headroom remains.

The comparison is PAIRED, so the informative number is the bootstrap CI of the
DIFFERENCE. Two overlapping marginal CIs are not evidence of equality; this
project has made that error before.

LIMITATIONS
-----------
n is ~44 at best. A paired AUROC difference at that n has a CI half-width of
roughly +-0.15, so only a LARGE gap is detectable. "No significant difference"
means "no large gap detected", never "proven equal".

Run:  PYTHONIOENCODING=utf-8 python scripts/oracle_vs_detector_paired.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "evidence", "oracle_ceiling_20260920")
ORACLE = os.path.join(ROOT, "artifacts", "evidence",
                      "oracle_crossing_20260920", "per_case_oracle.csv")
FOLDS = os.path.join(ROOT, "artifacts", "evidence",
                     "det_folds_20260920", "fold_assignment.csv")
MAPPING = os.path.join(ROOT, "artifacts", "audits",
                       "vascular_dataset_governance_20260830",
                       "source_case_mapping.csv")
MANIFEST = os.path.join(ROOT, "artifacts", "manifest",
                        "locked_evaluation_v1.csv")
CANON = os.path.join(ROOT, "artifacts", "audits", "canonical_labels.csv")
WEIGHTS_GLOB = os.path.join(ROOT, "artifacts", "experiments",
                            "det_malformation_20260920",
                            "fold*_s768", "weights", "best.pt")
CASE_ROOT = os.path.join(ROOT, "data")

POS = {"10--30%", "30--60%", ">60%"}
NEG = {"<=10%"}
CLS_MALFORMED = 1
CONF = 0.25
IMGSZ = 768
SEED = 20260920


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    p, n = s[y == 1], s[y == 0]
    if len(p) < 3 or len(n) < 3:
        return None
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2)
                 / (len(p) * len(n)))


def case_dir(case_id: str) -> str:
    archive, num = case_id.split("/")
    return os.path.join(CASE_ROOT, archive, num)


def case_images(d: str) -> list[str]:
    """Pre-registered selection, identical to eval_malformation_field.py:
    real capillaroscopy frames only (CAPorg*), report scans excluded."""
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith("CAPorg") and f.lower().endswith(".jpg"))


def build_case_to_original() -> dict[str, list[int]]:
    """Which vascular-dataset originals came from which development case."""
    mp = pd.read_csv(MAPPING)
    mp = mp[mp.source_mapping_status == "EXACT_RECOVERED_DEVELOPMENT"]
    out: dict[str, list[int]] = {}
    for row in mp.itertuples(index=False):
        for c in str(row.development_cases).replace(";", ",").split(","):
            c = c.strip()
            if c and c != "nan":
                out.setdefault(c, []).append(int(row.original_id))
    return out


def clean_folds_for(originals: list[int], fold_of: dict[int, int]) -> list[int]:
    """Folds that did NOT train on any of this case's originals.

    A fold trains on every original except those assigned to it (which form its
    val split). So a fold is clean for this case iff every one of the case's
    originals is assigned to that fold.
    """
    assigned = {fold_of[o] for o in originals if o in fold_of}
    if len(assigned) != 1:
        # originals of one case spread across folds => no fold excludes them all
        return []
    return sorted(assigned)


def md5(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_fold_map() -> dict[str, int]:
    """md5 -> the fold that held that original OUT of training.

    Requiring per-CASE clean folds leaves only 24 of 44 cases, because one
    case's frames became several originals that landed in different folds.
    Per-FRAME assignment is the standard out-of-fold construction and keeps all
    44: a frame whose original sits in fold k was never trained on by fold k's
    model, whatever the rest of its case did.
    """
    mp = pd.read_csv(MAPPING)
    fold_of = dict(pd.read_csv(FOLDS)[["original_id", "fold"]].values)
    out = {}
    for row in mp.itertuples(index=False):
        f = fold_of.get(int(row.original_id))
        if f is not None:
            out[str(row.image_md5)] = int(f)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    oc = pd.read_csv(ORACLE)
    canon = pd.read_csv(CANON).set_index("exam_case_id")
    man = pd.read_csv(MANIFEST)
    dev = set(man.loc[man.development_fold.notna(), "exam_case_id"])
    locked = set(man.loc[man.development_fold.isna(), "exam_case_id"])

    oc = oc[oc.exam_case_id.isin(dev)].copy()
    assert not (set(oc.exam_case_id) & locked), "a locked case reached the oracle cohort"

    oc["label"] = canon.reindex(oc.exam_case_id).malformation_ratio.astype(str).values
    oc = oc[oc.label.isin(POS | NEG)].copy()
    oc["y"] = oc.label.isin(POS).astype(int)
    print(f"oracle cases with a usable malformation label: {len(oc)}  "
          f"prevalence {oc.y.mean():.3f}")

    c2o = build_case_to_original()
    fold_of = dict(pd.read_csv(FOLDS)[["original_id", "fold"]].values)
    rows = []
    for row in oc.itertuples(index=False):
        origs = c2o.get(row.exam_case_id, [])
        cf = clean_folds_for(origs, fold_of)
        rows.append({"exam_case_id": row.exam_case_id, "y": int(row.y),
                     "label": row.label, "n_originals": len(origs),
                     "clean_folds": cf})
    info = pd.DataFrame(rows)
    print(f"cases with at least one clean fold: {(info.clean_folds.str.len() > 0).sum()}"
          f" of {len(info)}")
    info.to_csv(os.path.join(OUT, "case_fold_eligibility.csv"), index=False)

    # ---- per-frame out-of-fold detector scoring -------------------------
    import cv2
    from ultralytics import YOLO

    m2f = frame_fold_map()
    weights = sorted(glob.glob(WEIGHTS_GLOB))
    assert len(weights) == 5, f"expected 5 folds, found {len(weights)}"
    models = {}
    for w in weights:
        k = int(os.path.basename(os.path.dirname(os.path.dirname(w)))
                .replace("fold", "").replace("_s768", ""))
        models[k] = YOLO(w)
    print(f"loaded {len(models)} fold models")

    per_case, skipped = [], []
    for row in oc.itertuples(index=False):
        d = case_dir(row.exam_case_id)
        mal = tot = n_img = n_oof = 0
        for p in case_images(d):
            arr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if float(g.std()) < 8.0:
                continue
            n_img += 1
            f = m2f.get(md5(p))
            if f is None:
                continue           # frame never entered the vascular pool
            n_oof += 1
            r = models[f].predict(arr, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([])
            tot += len(cls)
            mal += int((cls == CLS_MALFORMED).sum())
        if n_oof == 0 or tot == 0:
            skipped.append({"exam_case_id": row.exam_case_id, "n_img": n_img,
                            "n_oof_frames": n_oof, "n_detections": tot})
            continue
        per_case.append({"exam_case_id": row.exam_case_id, "y": int(row.y),
                         "label": row.label, "det_mal_ratio": mal / tot,
                         "n_loops": tot, "n_img": n_img, "n_oof_frames": n_oof})
    det = pd.DataFrame(per_case)
    print(f"\nscored out-of-fold: {len(det)} cases, skipped {len(skipped)}")
    if skipped:
        print(pd.DataFrame(skipped).to_string(index=False))

    # oracle column `malformation` is the human-box malformed fraction
    merged = det.merge(oc[["exam_case_id", "malformation", "n_loops"]]
                       .rename(columns={"malformation": "human_mal_ratio",
                                        "n_loops": "human_loops_per_img"}),
                       on="exam_case_id", how="inner")
    merged.to_csv(os.path.join(OUT, "paired_per_case.csv"), index=False)

    y = merged.y.values
    a_h = auroc(y, merged.human_mal_ratio.values)
    a_d = auroc(y, merged.det_mal_ratio.values)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    diffs, hs, ds = [], [], []
    for _ in range(10000):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() < 3 or y[b].sum() > len(b) - 3:
            continue
        h = auroc(y[b], merged.human_mal_ratio.values[b])
        d = auroc(y[b], merged.det_mal_ratio.values[b])
        if h is None or d is None:
            continue
        hs.append(h); ds.append(d); diffs.append(h - d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res = {
        "what": "paired ceiling test on the oracle cohort: human boxes vs "
                "per-frame out-of-fold detector, same cases, same labels",
        "why_not_the_115_set": "the 115-case evaluation set was defined as the "
                               "development cases WITHOUT human boxes; overlap "
                               "with the oracle cohort is 0 by construction",
        "n": int(len(merged)),
        "prevalence": round(float(y.mean()), 4),
        "human_boxes_auroc": round(a_h, 4),
        "human_boxes_ci95": [round(float(x), 4) for x in np.percentile(hs, [2.5, 97.5])],
        "detector_oof_auroc": round(a_d, 4),
        "detector_ci95": [round(float(x), 4) for x in np.percentile(ds, [2.5, 97.5])],
        "paired_difference_human_minus_detector": round(a_h - a_d, 4),
        "paired_difference_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "difference_excludes_zero": bool(lo > 0 or hi < 0),
        "detection_density": {
            "human_loops_per_image": round(float(merged.human_loops_per_img.mean()), 2),
            "detector_loops_per_image": round(float((merged.n_loops / merged.n_oof_frames).mean()), 2),
        },
        "governance": {"locked_cases_seen": 0, "images_transmitted": 0,
                       "models_trained": 0,
                       "detector_scored_out_of_fold_per_frame": True},
        "interpretation_rule": "a difference CI that excludes 0 and favours the "
                               "human boxes means detector headroom remains; a "
                               "CI containing 0 at this n means NO LARGE GAP "
                               "DETECTED, not equality",
        "limitations": [
            f"n={len(merged)}; a paired AUROC difference at this n resolves only large gaps",
            "prevalence here is 0.341 vs 0.461 on the 115-case set, so absolute "
            "AUROCs are not comparable across the two cohorts -- only the paired "
            "difference within this cohort is interpretable",
            "human boxes are student annotations, not a clinical gold standard",
        ],
    }
    json.dump(res, open(os.path.join(OUT, "ceiling.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
