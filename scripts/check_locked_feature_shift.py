#!/usr/bin/env python
"""Feature-only sanity check before the one-shot locked evaluation.

A classifier fitted on development vectors can only be tested on locked vectors if
the two were produced the same way. If locked features sit in a different part of
feature space -- different extractor, different preprocessing, different fold rule
-- the classifier will score badly and the result would be misread as "the model
does not generalise" when the real cause is a broken feature pipeline.

This reads NO labels. It compares feature distributions only, so running it costs
nothing from the one-shot locked evaluation budget: no accuracy, no field, no
model fit against any label.

What would be disqualifying
---------------------------
- geometry columns differing in name or order
- a geometry column whose locked median sits outside the development p1-p99 range
- DINOv2 norms differing by more than a few percent (indicates a different
  transform or a different checkpoint)
"""
import argparse
import json

import numpy as np
import pandas as pd


def load(geom, dino, index, ids):
    G = pd.read_csv(geom)
    G["exam_case_id"] = G["exam_case_id"].astype(str)
    G["image_path"] = G["image_path"].astype(str)
    G = G[G.exam_case_id.isin(ids)]
    idx = pd.read_csv(index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    F = np.load(dino).astype(np.float32)
    assert len(idx) == len(F), f"index {len(idx)} != features {len(F)}"
    D = pd.DataFrame(F)
    D["image_path"] = idx["image_path"].astype(str)
    D["exam_case_id"] = idx["exam_case_id"]
    D = D[D.exam_case_id.isin(ids)]
    m = G.merge(D, on=["exam_case_id", "image_path"], how="inner", suffixes=("", "_d"))
    gcols = [c for c in G.columns
             if c not in ("exam_case_id", "image_path", "fold")
             and pd.api.types.is_numeric_dtype(G[c])]
    return m, gcols, F.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-geom",
                    default="artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--dev-dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--dev-index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--locked-geom",
                    default="artifacts/features/seg_instance_c05_locked/image_geometry.csv")
    ap.add_argument("--locked-dino",
                    default="artifacts/features/dinov2_locked/features.npy")
    ap.add_argument("--locked-index",
                    default="artifacts/features/dinov2_locked/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", default="artifacts/features/seg_instance_c05_locked/shift_check.json")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    dev_ids = set(man.loc[man.evaluation_role == "development", "exam_case_id"])
    lk_ids = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])

    Md, gd, dimd = load(args.dev_geom, args.dev_dino, args.dev_index, dev_ids)
    Ml, gl, diml = load(args.locked_geom, args.locked_dino, args.locked_index, lk_ids)

    out = {"dev": {"images": len(Md), "cases": int(Md.exam_case_id.nunique())},
           "locked": {"images": len(Ml), "cases": int(Ml.exam_case_id.nunique())},
           "geom_cols_identical": gd == gl,
           "dino_dims_match": dimd == diml,
           "reads_labels": False}

    assert gd == gl, ("geometry columns differ -- the two extractors are not the "
                      f"same:\n dev only: {set(gd) - set(gl)}\n locked only: {set(gl) - set(gd)}")
    assert dimd == diml, f"DINOv2 dims differ: dev {dimd} vs locked {diml}"

    # DINOv2: vector norms are the cheapest signal that the transform matches
    dcols = list(range(dimd))
    nd = np.linalg.norm(Md[dcols].to_numpy(dtype=np.float32), axis=1)
    nl = np.linalg.norm(Ml[dcols].to_numpy(dtype=np.float32), axis=1)
    out["dino_norm"] = {
        "dev_mean": round(float(nd.mean()), 3), "dev_std": round(float(nd.std()), 3),
        "locked_mean": round(float(nl.mean()), 3), "locked_std": round(float(nl.std()), 3),
        "ratio_locked_over_dev": round(float(nl.mean() / nd.mean()), 4)}

    # geometry: flag any column whose locked median falls outside dev p1-p99
    flags = []
    for c in gd:
        a = Md[c].to_numpy(dtype=np.float64)
        b = Ml[c].to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        lo, hi = np.percentile(a, [1, 99])
        med = float(np.median(b))
        if med < lo or med > hi:
            flags.append({"col": c, "locked_median": round(med, 4),
                          "dev_p1": round(float(lo), 4), "dev_p99": round(float(hi), 4)})
    out["geom_cols_out_of_dev_range"] = flags
    out["n_geom_cols"] = len(gd)

    # headline comparables
    for c in ("n_inst", "frac_low_circ", "circularity_mean", "conf_mean"):
        if c in Md.columns and c in Ml.columns:
            out.setdefault("headline", {})[c] = {
                "dev_mean": round(float(Md[c].mean()), 4),
                "locked_mean": round(float(Ml[c].mean()), 4)}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nverdict:", "USABLE" if not flags else
          f"{len(flags)} geometry columns shifted -- inspect before evaluating")


if __name__ == "__main__":
    main()
