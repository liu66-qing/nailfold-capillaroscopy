#!/usr/bin/env python
"""Phase 3 decisive test: does miss density rise with malformation level?

The current segmenter's features show an INVERTED gradient:
  detected instances per case by malformation level 0/1/2/3 = 17.8/17.4/12.7/14.4
  frac_low_circ vs malformation rho = -0.265 (p=0.0009)   <- wrong sign
  circularity_mean vs malformation rho = +0.305

If the miss hypothesis is correct, the density of RECOVERED (missed) vessels
must correlate POSITIVELY with malformation level. That is the falsifiable
prediction this script tests, before any retraining.

locked-47: development cases only (development_fold non-NaN), asserted.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ordinal malformation level, matching the label vocabulary in the manifest
MALF_LEVEL = {"<=10%": 0, "[<10%]": 0, "10--30%": 1, "30--60%": 2,
              ">60%": 3, "60--80%": 3, ">80%": 3}
CROSS_LEVEL = {"<=30%": 0, "[<30%]": 0, "10--30%": 0, "30--60%": 1,
               "60--80%": 2, ">80%": 3}
COUNT_LEVEL = {">=7": 0, "5--6": 1, "3--4": 2, "1--2": 3, "<1": 4}


def ordinal(series, table):
    """Map label text to an ordinal level. Unparseable text -> NaN, never a class."""
    v = series.astype(str).str.strip()
    return v.map(table).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss-case", required=True)
    ap.add_argument("--baseline-image",
                    default="/root/autodl-tmp/nailfold/artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--manifest",
                    default="/root/autodl-tmp/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="sam")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])
    dev = man[man.evaluation_role == "development"].set_index("exam_case_id")

    M = pd.read_csv(args.miss_case)
    M["exam_case_id"] = M["exam_case_id"].astype(str)
    M = M.set_index("exam_case_id")
    assert not (set(M.index) & locked), "locked case in miss features"

    # baseline: the failing segmenter's own per-case instance count, recomputed
    # here so the inverted gradient is reproduced from the same source file
    B = pd.read_csv(args.baseline_image)
    B["exam_case_id"] = B["exam_case_id"].astype(str)
    assert not (set(B.exam_case_id) & locked), "locked case in baseline geometry"
    Bc = B.groupby("exam_case_id").agg(
        n_inst_c05=("n_inst", "mean"),
        frac_low_circ_c05=("frac_low_circ", "mean"),
        circularity_mean_c05=("circularity_mean", "mean"))

    df = Bc.join(M, how="inner")
    out = {"config": {"tag": args.tag, "n_cases": int(len(df)),
                      "locked_cases_seen": 0,
                      "miss_case_file": args.miss_case},
           "levels": {}, "correlations": {}, "gradients": {}}

    for name, col, table in [("malformation", "malformation_ratio", MALF_LEVEL),
                             ("crossing", "crossing_ratio", CROSS_LEVEL),
                             ("count", "capillary_count", COUNT_LEVEL)]:
        lvl = ordinal(dev[col], table).reindex(df.index)
        ok = lvl.notna()
        out["levels"][name] = {"n_labelled": int(ok.sum()),
                               "distribution": lvl[ok].value_counts().sort_index()
                                                  .astype(int).to_dict()}
        corrs, grads = {}, {}
        test_cols = ["n_inst_c05", "frac_low_circ_c05", "circularity_mean_c05",
                     "n_miss", "miss_density", "miss_share", "n_yolo",
                     "miss_area_share", "miss_frac_low_circ",
                     "miss_circularity_mean", "miss_area_mean"]
        for c in test_cols:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            m = ok & v.notna()
            if m.sum() < 30:
                continue
            rho, p = spearmanr(v[m], lvl[m])
            corrs[c] = {"rho": round(float(rho), 4), "p": float("%.3g" % p),
                        "n": int(m.sum())}
            grads[c] = {str(int(k)): round(float(x), 3) for k, x in
                        v[m].groupby(lvl[m]).mean().items()}
        out["correlations"][name] = corrs
        out["gradients"][name] = grads

    mc = out["correlations"]["malformation"]
    key = mc.get("miss_density")
    out["verdict"] = {
        "hypothesis": ("miss density correlates POSITIVELY with malformation "
                       "level (baseline frac_low_circ was -0.265)"),
        "miss_density_vs_malformation": key,
        "supported": bool(key and key["rho"] > 0 and key["p"] < 0.05),
    }
    out["limitations"] = [
        "development set only (186 cases); NOT product capability",
        "AI-proposed misses are not verified vessels; no human confirmation",
        "malformation level is an ordinal recode of interval label text",
        "pixel-scale geometry; calibration_factors.json unused",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print("=== malformation ===")
    for c, d in out["correlations"]["malformation"].items():
        print(f"  {c:26s} rho={d['rho']:+.4f} p={d['p']:<10} n={d['n']}"
              f"  grad={out['gradients']['malformation'].get(c)}")
    print("VERDICT supported =", out["verdict"]["supported"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
