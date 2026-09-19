#!/usr/bin/env python
"""Instance geometry for locked-47, byte-identical in method to the development set.

Why a separate script
---------------------
`extract_instance_geometry.py` hard-asserts that no locked case reaches the image
index -- correctly, because it feeds development features. This script is the
locked-side twin. It imports the geometry functions from that file rather than
reimplementing them, so the feature definitions cannot drift between the training
features and the test features.

Fold assignment: the one decision that matters
----------------------------------------------
Development features are produced by ONE model per image (the fold that held that
case out). If locked features were produced by averaging or by picking among 5
models, locked feature vectors would have different noise characteristics than the
vectors the classifier was fitted on -- a silent distribution shift that would
show up as a performance drop and be misread as failure to generalise.

So locked also gets exactly one model per image. The assignment comes from
`artifacts/manifest/stratified_folds_v3.csv`, which already assigns a fold to all
233 cases including the 47 locked ones. That file predates this work and is
label-blind, so the assignment is not a choice made now with any knowledge of the
outcome. All 5 checkpoints are equally valid for locked cases:
`artifacts/seg_data_governance_approval.json` records
gate `no_locked_overlap` = PASS, "Intersection = 0. Zero locked case data in
segmentation training set", so no locked case was trained on by any fold.

This is deliberately NOT the "median-count fold selection per image" rule used by
`features/seg_vessel_v2/locked_test_inference.json`: that rule picks a model per
image based on how many vessels it found, which makes locked features
systematically median-ish while development features are single-model. Matching
the development distribution matters more than reducing locked variance.

Settings are copied from features/seg_instance_c05/meta.json and must not change:
conf=0.05, imgsz=1024, no max_det override (Ultralytics default 300, as in the
development run).

What this script does NOT do
---------------------------
It reads no labels and computes no metric. It writes features only, into its own
new directory. Nothing existing is modified.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_instance_geometry import (  # noqa: E402
    image_row, pairwise_topology, polygon_features, sha256,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/features/dinov2_locked/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--folds", default="artifacts/manifest/stratified_folds_v3.csv")
    ap.add_argument("--data-root", default="/root/autodl-tmp/nailfold/data")
    ap.add_argument("--weights-dir",
                    default="/root/autodl-tmp/nailfold/artifacts/models")
    ap.add_argument("--out",
                    default="/root/autodl-tmp/nailfold/artifacts/features/seg_instance_c05_locked")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man["evaluation_role"].astype(str)
                         .str.contains("locked", case=False), "exam_case_id"])
    dev = set(man.loc[man.evaluation_role == "development", "exam_case_id"])
    assert len(locked) == 47, f"expected 47 locked cases, manifest has {len(locked)}"

    # locked-47 carries no development_fold by construction; assert that, because
    # if one did it would mean the role and fold columns disagree
    devfold = man.dropna(subset=["development_fold"])
    assert not (set(devfold["exam_case_id"]) & locked), \
        "a locked case carries a development_fold -- refusing to run"

    fl = pd.read_csv(args.folds, encoding="utf-8-sig")
    fl["exam_case_id"] = fl["exam_case_id"].astype(str)
    fmap = fl.drop_duplicates("exam_case_id").set_index("exam_case_id")["fold"].astype(int)
    missing = locked - set(fmap.index)
    assert not missing, f"locked cases absent from folds_v3: {sorted(missing)[:5]}"

    idx = pd.read_csv(args.index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    idx = idx[idx.exam_case_id.isin(locked)].reset_index(drop=True)
    assert not (set(idx.exam_case_id) & dev), "a development case reached the locked index"
    assert idx.exam_case_id.nunique() == 47, \
        f"index covers {idx.exam_case_id.nunique()} locked cases, expected 47"
    idx["fold"] = idx["exam_case_id"].map(fmap).astype(int)
    print(f"locked images: {len(idx)} / cases: {idx.exam_case_id.nunique()}", flush=True)
    print("fold assignment (from folds_v3, label-blind):", flush=True)
    print(idx.drop_duplicates("exam_case_id").fold.value_counts().sort_index().to_string(),
          flush=True)

    from ultralytics import YOLO

    wsha, rows, counts, failures = {}, [], [], []
    for k, grp in idx.groupby("fold", sort=True):
        wp = os.path.join(args.weights_dir, f"seg_vessel_fold{k}", "weights", "best.pt")
        wsha[f"fold{k}"] = {"path": wp, "sha256": sha256(wp),
                            "bytes": os.path.getsize(wp)}
        model = YOLO(wp)
        print(f"[fold {k}] {len(grp)} locked images  weights={wp}", flush=True)
        for n, (_, r) in enumerate(grp.iterrows(), 1):
            p = os.path.join(args.data_root, str(r["image_path"]).replace("\\", "/"))
            if not os.path.exists(p):
                failures.append({"image_path": r["image_path"], "why": "missing"})
                continue
            try:
                res = model.predict(p, conf=args.conf, imgsz=args.imgsz,
                                    verbose=False, device=args.device)[0]
            except Exception as e:
                failures.append({"image_path": r["image_path"], "why": repr(e)[:200]})
                continue
            h, w = res.orig_shape
            inst = []
            if res.masks is not None and res.masks.xy is not None:
                for poly in res.masks.xy:
                    f = polygon_features(np.asarray(poly))
                    if f is not None:
                        inst.append(f)
            row = image_row(inst, pairwise_topology(inst, w, h), w, h)
            row["exam_case_id"] = r["exam_case_id"]
            row["image_path"] = r["image_path"]
            row["fold"] = int(k)
            row["conf_mean"] = float(res.boxes.conf.mean()) if len(res.boxes) else np.nan
            rows.append(row)
            counts.append(len(inst))
            if n % 50 == 0:
                print(f"  [fold {k}] {n}/{len(grp)}", flush=True)

    img = pd.DataFrame(rows)
    img.to_csv(os.path.join(args.out, "image_geometry.csv"), index=False)

    num = img.select_dtypes(include=[np.number]).columns.drop(["fold"], errors="ignore")
    g = img.groupby("exam_case_id")
    case = g[list(num)].mean()
    case = case.join(g[list(num)].std().add_suffix("_bcase"))
    case = case.join(g[list(num)].max().add_suffix("_bmax"))
    case["n_images"] = g.size()
    case.to_csv(os.path.join(args.out, "case_geometry.csv"))

    meta = {
        "purpose": "locked-47 instance geometry, method-identical to "
                   "features/seg_instance_c05 so the test vectors match the "
                   "vectors the classifier was fitted on",
        "evaluation_role": "locked_test",
        "locked_test_examined": True,
        "n_cases": int(case.shape[0]),
        "n_images": int(img.shape[0]),
        "development_cases_seen": 0,
        "fold_rule": "single model per image; fold from stratified_folds_v3.csv "
                     "(label-blind, predates this work). NOT median-count "
                     "selection, NOT a 5-fold average -- both would give locked "
                     "features different noise than development features.",
        "leakage_basis": "artifacts/seg_data_governance_approval.json gate "
                         "no_locked_overlap = PASS, intersection 0: no locked "
                         "case is in any fold's segmentation training set",
        "weights": wsha,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "instances_per_image_mean": float(np.mean(counts)) if counts else 0.0,
        "instances_per_image_median": float(np.median(counts)) if counts else 0.0,
        "instances_total": int(np.sum(counts)),
        "n_failures": len(failures),
        "failures": failures[:50],
        "limitations": [
            "pixels and scale-free ratios only; no micron claim, "
            "calibration_factors.json deliberately unused",
            "crossing is a geometric proxy, not a clinical crossing definition",
            "malformation proxies use the same a priori thresholds as the "
            "development run (circularity<0.30, solidity<0.80, elongation>4.0)",
            "features only; this file makes no accuracy claim about any field",
        ],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"instances/image mean={meta['instances_per_image_mean']:.2f} "
          f"median={meta['instances_per_image_median']:.1f} "
          f"failures={len(failures)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
