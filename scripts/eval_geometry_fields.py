#!/usr/bin/env python
"""Do the recovered instance-level features move the three structure fields?

The three fields tested here are the ones whose definitions were destroyed by
seg_vessel_v2_handoff.md section 5.3:

  crossing_ratio      defined on vessel-vessel relations; the handoff passed
                      down conf_mean and area_mean, which contain no relations
  malformation_ratio  defined on per-vessel shape; area_mean over 16 instances
                      cannot express "what fraction of loops is malformed"
  capillary_count     the one field the 13 scalars did retain (count_mean), so
                      it acts as a positive control: if the pipeline is sound
                      this field should behave the same as before

Ruler -- identical to the delivered configuration, no adjustment:
  * binary label, accuracy minus ONE fixed majority answer
  * the majority answer is recomputed inside every bootstrap resample
  * 2000 case-level bootstraps, verdict by CI lower bound > 0
  * 5-fold by development_fold only (never the `split` column)
  * fixed equal-weight ensemble over PRESPECIFIED feature blocks, so no block
    is chosen after seeing results; choosing one would be the selection bias
    already refuted this project
  * locked-47 never loaded

The feature blocks are fixed before running and are defined by what part of the
vessel field they describe, not by how well they score.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

N_BOOT = 2000
SEED = 20260918
C_FIXED = 0.03

# Binarisation of the ordinal interval labels. These strings are the raw report
# values; the cut is placed at the lowest interval boundary (the clinically
# normal band) rather than at a threshold chosen to balance classes.
FIELDS = {
    "crossing_ratio": {"normal": ["<=30%", "[<30%]"]},
    "malformation_ratio": {"normal": ["<=10%", "[<10%]"]},
    "capillary_count": {"normal": [">=7"]},
}

# Prespecified feature blocks. Named by anatomy, fixed before any scoring.
BLOCKS = {
    # vessel-vessel relations: the quantity crossing_ratio is defined on
    "topology": ["overlap_frac_inst", "overlap_pairs", "iou_max", "iou_mean",
                 "near_pairs", "near_frac_inst", "nn_dist_median",
                 "nn_dist_iqr", "orient_disp", "pair_n"],
    # per-vessel shape spread and tails: what malformation_ratio is defined on
    "shape_spread": ["circularity_std", "circularity_p10", "circularity_iqr",
                     "circularity_cv", "solidity_std", "solidity_p10",
                     "solidity_iqr", "convexity_std", "convexity_p10",
                     "elongation_std", "elongation_p90", "elongation_iqr",
                     "aspect_std", "extent_std", "extent_p10"],
    # explicit abnormal fractions, in the same units as the label
    "abnormal_frac": ["frac_low_circ", "frac_low_solidity", "frac_high_elong",
                      "frac_area_2x_median", "frac_area_half_median",
                      "area_p90_over_median"],
    # density and central tendency: the block the handoff already had
    "density": ["n_inst", "inst_area_share", "area_mean", "area_std",
                "area_p90", "perimeter_mean", "circularity_mean",
                "solidity_mean", "elongation_mean", "conf_mean", "n_images"],
}


def binarise(s, normal):
    """1 = abnormal. Unparseable/unit/bracket-only text becomes NaN, never a class."""
    v = s.astype(str).str.strip()
    known_abnormal = ["30--60%", "60--80%", ">80%", "10--30%", ">60%",
                      "5--6", "3--4", "1--2", "<1"]
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[v.isin(normal)] = 0.0
    out[v.isin(known_abnormal)] = 1.0
    return out


def resolve(cols, names):
    """A block may reference a stat that does not exist; take what is there."""
    got = []
    for n in names:
        if n in cols:
            got.append(n)
        # case_geometry also carries between-image dispersion (_bcase) and the
        # per-case worst image (_bmax); both are legitimate for "one bad finger"
        for suf in ("_bcase", "_bmax"):
            if n + suf in cols:
                got.append(n + suf)
    return got


def oof_prob(X, y, folds):
    """Out-of-fold probability for one feature block. Case-level throughout."""
    p = pd.Series(np.nan, index=X.index, dtype=float)
    for k in sorted(folds.unique()):
        tr, te = folds != k, folds == k
        if y[tr].nunique() < 2:
            continue
        mdl = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C_FIXED, max_iter=4000, solver="liblinear"),
        )
        mdl.fit(X[tr], y[tr])
        p.loc[X.index[te]] = mdl.predict_proba(X[te])[:, 1]
    return p


def score(y, pred, rng):
    """Accuracy minus ONE fixed majority answer, baseline redrawn per resample."""
    ok = pred.notna() & y.notna()
    yt, yp = y[ok].to_numpy(), pred[ok].to_numpy()
    n = len(yt)
    if n < 20:
        return None
    acc = float((yp == yt).mean())
    maj = float(np.bincount(yt.astype(int)).argmax())
    base = float((yt == maj).mean())
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        ys, ps = yt[i], yp[i]
        m = float(np.bincount(ys.astype(int)).argmax())
        d[b] = (ps == ys).mean() - (ys == m).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4), "majority_baseline": round(base, 4),
            "delta": round(acc - base, 4),
            "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "usable": bool(lo > 0), "abnormal_share": round(float(yt.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="artifacts/features/seg_instance_v1/case_geometry.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", default="artifacts/features/seg_instance_v1/geometry_fields.json")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man["evaluation_role"] == "locked_test", "exam_case_id"])
    dev = man[man["evaluation_role"] == "development"].set_index("exam_case_id")

    G = pd.read_csv(args.geom, index_col=0)
    G.index = G.index.astype(str)
    assert not (set(G.index) & locked), "locked case present in geometry features"
    folds = G["fold"].astype(int)
    cols = set(G.columns)

    blocks = {b: resolve(cols, names) for b, names in BLOCKS.items()}
    rng = np.random.default_rng(SEED)
    out = {"config": {"C": C_FIXED, "n_boot": N_BOOT, "seed": SEED,
                      "locked_cases_seen": 0, "n_cases": int(G.shape[0]),
                      "blocks": {b: len(v) for b, v in blocks.items()},
                      "ensemble": "fixed equal weights over all 4 blocks, "
                                  "no block selected after seeing results"},
           "fields": {}}

    for field, cfg in FIELDS.items():
        if field not in dev.columns:
            out["fields"][field] = {"status": "NO_LABEL_COLUMN"}
            continue
        y = binarise(dev[field], cfg["normal"]).reindex(G.index)
        keep = y.notna()
        yk, fk = y[keep], folds[keep]

        probs, members = {}, {}
        for b, names in blocks.items():
            if len(names) < 3:
                continue
            p = oof_prob(G.loc[keep, names], yk, fk)
            probs[b] = p
            lab = p.where(p.isna(), (p > 0.5).astype(float))
            members[b] = score(yk, lab, np.random.default_rng(SEED))

        ens = pd.concat(probs.values(), axis=1).mean(axis=1)
        ens_lab = ens.where(ens.isna(), (ens > 0.5).astype(float))
        res = score(yk, ens_lab, np.random.default_rng(SEED))
        out["fields"][field] = {
            "labelled_cases": int(keep.sum()),
            "label_raw_top": {str(k): int(v) for k, v in
                              dev[field].astype(str).value_counts().head(6).items()},
            "ensemble": res,
            "members": members,
            "members_passing_alone": sum(1 for m in members.values()
                                         if m and m["usable"]),
            "note": ("only 'ensemble' is a deliverable number; a member that "
                     "passes alone was found by looking at all members, which "
                     "is the refuted selection bias"),
        }
        print(field, "ens", res["delta"] if res else None,
              res["delta_ci95"] if res else None, flush=True)

    out["limitations"] = [
        "development set only (n<=186); NOT product capability and NOT a "
        "locked-47 result",
        "labels are ordinal interval strings binarised at the normal-band "
        "boundary; a finer ordinal target was not attempted here",
        "crossing/malformation proxies are geometric (mask IoU, circularity, "
        "solidity) with a-priori thresholds, not validated clinical definitions",
        "no micron claim; calibration_factors.json unused",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
