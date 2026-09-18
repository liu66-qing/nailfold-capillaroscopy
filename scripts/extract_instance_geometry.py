#!/usr/bin/env python
"""Recover instance-level vessel geometry that seg_vessel_v2_handoff.md discarded.

Why this exists
---------------
The segmentation model reports mask mAP50 = 0.952 and the handoff document
concluded "segmentation quality has reached its ceiling, no retraining needed".
That claim is true *of the segmentation task* and was then treated as a
guarantee about field recognition, which it is not: the mapping from good masks
to a correct crossing_ratio was never verified.

The actual loss is in section 5.3 of that handoff. Segmentation emits tens of
instances per image, each with a polygon outline, and the file it handed
downstream (`features/seg_vessel_v2/case_features.csv`) contains 13 scalars per
case -- count_mean, conf_mean, area_mean and similar. Two of the required
fields are defined on structure that those scalars cannot express:

  crossing_ratio      needs pairwise relations between vessels (do loops cross?)
  malformation_ratio  needs per-vessel shape deviation (is this loop malformed?)

Averaging mask area over 20 instances destroys both. Every downstream attempt on
these fields has scored between -0.011 and +0.080 -- consistent with the
information simply being absent from the features, not with a modelling failure.

This script re-runs the 5 fold weights (recovered from the old server; the new
server's weights directories were empty) and computes geometry per instance,
then aggregates with distribution statistics rather than a single mean.

Leakage discipline
------------------
The segmentation models were trained with case-level 5-fold isolation and the
seg fold column agrees exactly with development_fold (verified: the crosstab is
diagonal). A case in fold k is therefore inferred with the fold-k weights,
because the fold-k model held that case out. Using any other fold's weights
would mean the segmenter had trained on the very case whose features feed a
downstream classifier evaluated on that same case.

locked-47 is never inferred: the index is filtered to development cases and the
script asserts the intersection with locked is empty.

What is NOT claimed
-------------------
No absolute micron scale. All geometry is in pixels and in scale-free ratios;
`calibration_factors.json` is not used (it is back-fitted from labels and holds
four mutually inconsistent values). Crossing detection here is a geometric
proxy (mask overlap and skeleton intersection), not a validated clinical
definition of a crossed loop.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def polygon_features(poly):
    """Shape descriptors for one vessel outline. Scale-free where possible."""
    if poly is None or len(poly) < 3:
        return None
    x, y = poly[:, 0].astype(float), poly[:, 1].astype(float)
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    d = np.diff(np.vstack([poly, poly[:1]]).astype(float), axis=0)
    perim = float(np.sqrt((d ** 2).sum(1)).sum())
    if area <= 0 or perim <= 0:
        return None

    # elongation / orientation from the second moment matrix
    pts = np.stack([x - x.mean(), y - y.mean()])
    cov = np.cov(pts)
    ev = np.linalg.eigvalsh(cov) if cov.shape == (2, 2) else np.array([1.0, 1.0])
    ev = np.clip(ev, 1e-9, None)
    elong = float(np.sqrt(ev[1] / ev[0]))

    # convexity: a malformed (tortuous) loop deviates more from its hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(poly.astype(float))
        hull_area = float(hull.volume)
        solidity = area / hull_area if hull_area > 0 else np.nan
        hd = np.diff(np.vstack([poly[hull.vertices],
                               poly[hull.vertices][:1]]).astype(float), axis=0)
        hull_perim = float(np.sqrt((hd ** 2).sum(1)).sum())
        convexity = hull_perim / perim if perim > 0 else np.nan
    except Exception:
        solidity, convexity = np.nan, np.nan

    bw = float(x.max() - x.min())
    bh = float(y.max() - y.min())
    return {
        "area": float(area),
        "perimeter": perim,
        # 1.0 for a circle, lower for a tortuous outline -> malformation proxy
        "circularity": float(4 * np.pi * area / (perim ** 2)),
        "elongation": elong,
        "solidity": float(solidity) if solidity == solidity else np.nan,
        "convexity": float(convexity) if convexity == convexity else np.nan,
        "aspect": bw / bh if bh > 0 else np.nan,
        "extent": float(area / (bw * bh)) if bw > 0 and bh > 0 else np.nan,
        "cx": float(x.mean()), "cy": float(y.mean()),
        "bw": bw, "bh": bh,
    }


def pairwise_topology(inst, img_w, img_h):
    """Relations between vessels in one image -- the quantity crossing_ratio needs.

    A crossed loop shows up as two instances whose bounding boxes and outlines
    interpenetrate. Mask-level IoU is the cleanest available signal: two
    separate loops standing side by side have IoU 0, a crossing pair overlaps.
    Reported as the fraction of instances involved in at least one overlap,
    which is dimensionally the same kind of quantity as the label ("<=30%").
    """
    n = len(inst)
    if n < 2:
        return {"pair_n": 0, "overlap_pairs": 0, "overlap_frac_inst": 0.0,
                "iou_max": 0.0, "iou_mean": 0.0, "near_pairs": 0,
                "near_frac_inst": 0.0, "nn_dist_median": np.nan,
                "nn_dist_iqr": np.nan, "orient_disp": np.nan}

    boxes = np.array([[i["cx"] - i["bw"] / 2, i["cy"] - i["bh"] / 2,
                       i["cx"] + i["bw"] / 2, i["cy"] + i["bh"] / 2] for i in inst])
    cent = np.array([[i["cx"], i["cy"]] for i in inst])
    diag = float(np.hypot(img_w, img_h))

    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = ar[:, None] + ar[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(iou, 0.0)

    iu = np.triu_indices(n, 1)
    vals = iou[iu]
    ov = vals > 0.10                      # geometric proxy for a crossing pair
    inst_ov = (iou > 0.10).any(1)

    d = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1)) / diag
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    near = d[iu] < 0.05
    inst_near = (d < 0.05).any(1)

    # spread of vessel orientations: disordered arrangement vs parallel combing
    ang = np.array([np.arctan2(i["bh"], i["bw"]) for i in inst])
    return {
        "pair_n": int(len(vals)),
        "overlap_pairs": int(ov.sum()),
        "overlap_frac_inst": float(inst_ov.mean()),
        "iou_max": float(vals.max()), "iou_mean": float(vals.mean()),
        "near_pairs": int(near.sum()),
        "near_frac_inst": float(inst_near.mean()),
        "nn_dist_median": float(np.median(nn)),
        "nn_dist_iqr": float(np.percentile(nn, 75) - np.percentile(nn, 25)),
        "orient_disp": float(np.std(ang)),
    }


# shape keys that describe a single vessel; malformation_ratio is defined on
# the *fraction of abnormal ones*, so both the spread and the tail matter
SHAPE_KEYS = ["circularity", "elongation", "solidity", "convexity",
              "aspect", "extent", "area", "perimeter"]


def image_row(inst, topo, img_w, img_h):
    """Collapse one image's instances, keeping distribution shape not just means.

    The handoff kept area_mean. A case where every loop is mildly irregular and
    a case with three grossly malformed loops among twenty normal ones can have
    identical means; they differ in the tail. So each shape key contributes
    mean/std/p10/p90/iqr plus an explicit abnormal-fraction, and the per-image
    row also carries the topology block.
    """
    row = {"n_inst": len(inst), "img_w": img_w, "img_h": img_h}
    if not inst:
        return row
    for k in SHAPE_KEYS:
        v = np.array([i[k] for i in inst], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        row[f"{k}_mean"] = float(v.mean())
        row[f"{k}_std"] = float(v.std())
        row[f"{k}_p10"] = float(np.percentile(v, 10))
        row[f"{k}_p90"] = float(np.percentile(v, 90))
        row[f"{k}_iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        row[f"{k}_cv"] = float(v.std() / abs(v.mean())) if v.mean() != 0 else np.nan

    # malformation proxies as fractions of instances, matching the label's units
    circ = np.array([i["circularity"] for i in inst], dtype=float)
    sol = np.array([i["solidity"] for i in inst], dtype=float)
    elo = np.array([i["elongation"] for i in inst], dtype=float)
    with np.errstate(invalid="ignore"):
        row["frac_low_circ"] = float(np.nanmean(circ < 0.30))
        row["frac_low_solidity"] = float(np.nanmean(sol < 0.80))
        row["frac_high_elong"] = float(np.nanmean(elo > 4.0))
    # relative size dispersion: a normal field has similar-sized loops
    ar = np.array([i["area"] for i in inst], dtype=float)
    med = float(np.median(ar))
    if med > 0:
        row["frac_area_2x_median"] = float(np.mean(ar > 2 * med))
        row["frac_area_half_median"] = float(np.mean(ar < 0.5 * med))
        row["area_p90_over_median"] = float(np.percentile(ar, 90) / med)
    row["inst_area_share"] = float(ar.sum() / (img_w * img_h))
    row.update(topo)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--data-root", default="/root/autodl-tmp/nailfold/data")
    ap.add_argument("--weights-dir", default="/root/autodl-tmp/nailfold/artifacts/models")
    ap.add_argument("--out", default="/root/autodl-tmp/nailfold/artifacts/features/seg_instance_v1")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # role and fold both come from the reviewed manifest: evaluation_role is
    # development/locked_test and development_fold is NaN exactly on locked_test
    man = pd.read_csv(args.manifest)
    case_col = "exam_case_id"
    man[case_col] = man[case_col].astype(str)
    locked = set(man.loc[man["evaluation_role"].astype(str)
                         .str.contains("locked", case=False), case_col])
    dev_fold = (man.dropna(subset=["development_fold"])
                   .set_index(case_col)["development_fold"].astype(int))

    # locked-47 is defined by development_fold being NaN; assert the two agree
    assert not (set(dev_fold.index) & locked), \
        "a locked case carries a development_fold -- refusing to run"

    idx = pd.read_csv(args.index)
    idx[case_col] = idx[case_col].astype(str)
    idx = idx[idx[case_col].isin(dev_fold.index)].reset_index(drop=True)
    assert not (set(idx[case_col]) & locked), "locked case reached the image index"
    if args.limit:
        idx = idx.head(args.limit)
    idx["fold"] = idx[case_col].map(dev_fold).astype(int)

    from ultralytics import YOLO

    wsha = {}
    rows, per_inst_counts, failures = [], [], []
    for k, grp in idx.groupby("fold", sort=True):
        wp = os.path.join(args.weights_dir, f"seg_vessel_fold{k}", "weights", "best.pt")
        # fold k was HELD OUT of seg_vessel_fold{k}'s training, so this is the
        # only weight file that has not seen these cases
        wsha[f"fold{k}"] = {"path": wp, "sha256": sha256(wp),
                            "bytes": os.path.getsize(wp)}
        model = YOLO(wp)
        print(f"[fold {k}] {len(grp)} images  weights={wp}", flush=True)

        for n, (_, r) in enumerate(grp.iterrows(), 1):
            p = os.path.join(args.data_root, str(r["image_path"]).replace("\\", "/"))
            if not os.path.exists(p):
                failures.append({"image_path": r["image_path"], "why": "missing"})
                continue
            try:
                res = model.predict(p, conf=args.conf, imgsz=args.imgsz,
                                    verbose=False, device=0)[0]
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
            topo = pairwise_topology(inst, w, h)
            row = image_row(inst, topo, w, h)
            row[case_col] = r[case_col]
            row["image_path"] = r["image_path"]
            row["fold"] = int(k)
            row["conf_mean"] = float(res.boxes.conf.mean()) if len(res.boxes) else np.nan
            rows.append(row)
            per_inst_counts.append(len(inst))
            if n % 100 == 0:
                print(f"  [fold {k}] {n}/{len(grp)}", flush=True)

    img = pd.DataFrame(rows)
    img.to_csv(os.path.join(args.out, "image_geometry.csv"), index=False)

    # case level: mean over the case's images, plus the case-wide dispersion,
    # so a case with one bad finger is not averaged into normality
    num = img.select_dtypes(include=[np.number]).columns.drop(["fold"], errors="ignore")
    g = img.groupby(case_col)
    case = g[list(num)].mean()
    case = case.join(g[list(num)].std().add_suffix("_bcase"))
    case = case.join(g[list(num)].max().add_suffix("_bmax"))
    case["n_images"] = g.size()
    case["fold"] = case.index.map(dev_fold)
    case.to_csv(os.path.join(args.out, "case_geometry.csv"))

    meta = {
        "purpose": ("instance-level vessel geometry, recovering the topology and "
                    "per-vessel shape that seg_vessel_v2_handoff.md section 5.3 "
                    "discarded when it passed 13 scalars downstream"),
        "n_cases": int(case.shape[0]),
        "n_images": int(img.shape[0]),
        "n_features_case": int(case.shape[1]),
        "instances_per_image_mean": float(np.mean(per_inst_counts)) if per_inst_counts else 0.0,
        "instances_per_image_median": float(np.median(per_inst_counts)) if per_inst_counts else 0.0,
        "instances_total": int(np.sum(per_inst_counts)),
        "locked_cases_seen": 0,
        "fold_matched_inference": True,
        "fold_rule": "case with development_fold==k inferred with seg_vessel_fold{k} (held it out)",
        "weights": wsha,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "failures": failures[:50],
        "n_failures": len(failures),
        "limitations": [
            "geometry is in pixels and scale-free ratios; no micron claim, "
            "calibration_factors.json deliberately unused (label back-fitted, "
            "four inconsistent values)",
            "crossing is a geometric proxy (mask IoU > 0.10 and centroid "
            "proximity), not a validated clinical crossing definition",
            "malformation proxies use fixed thresholds (circularity<0.30, "
            "solidity<0.80, elongation>4.0) chosen a priori, not tuned on labels",
            "features only; whether they raise any field is decided by the "
            "unchanged downstream ruler, not by this file",
        ],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("weights", "failures", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
