#!/usr/bin/env python
"""Image-level training with case-level evaluation.

Why this is mechanistically different from everything already closed
-------------------------------------------------------------------
Every result so far (frozen-feature probes, LoRA, fullfov, spatial pooling,
per-field selection) trained on ONE vector per case: the 9.2 images of a case
were averaged into a single row, giving n=186 training rows. The diagnosed
bottleneck is variance, not capability -- a ~37-case inner fold could not even
select a config stably.

This script trains on 1708 image rows instead. The label is the case label
broadcast to its images (multiple-instance learning with a shared label), and
prediction is aggregated back to the case before scoring. That gives the
estimator ~9x more rows to fit while the evaluation unit stays the case, so
the ruler does not change.

This is NOT the closed frozen-feature MIL path: that one pooled over *frames of
a video* under a case-level probe. Here the unit is a still image of a
different finger/field, images are training rows rather than pooled inputs,
and the aggregation happens on predictions, not on features.

Protocol (identical to eval_spatial_fields.py / fixed_config_control.py, so
numbers are comparable):
  - case-level folds from development_fold (NaN = locked-47, excluded)
  - labels read verbatim from threshold_tuned_20260916/*__binary_oof.csv
  - baseline = single fixed majority answer, max(prevalence, 1-prevalence),
    recomputed inside every bootstrap resample
  - case-level bootstrap 2000x, verdict by CI lower bound of delta_vs_constant
  - NO per-fold selection, NO threshold tuning, NO class_weight (all three were
    measured to be net-negative)

Grouping discipline: images are split by CASE, never by image, so no case has
images on both sides of a fold boundary.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
# Fixed, prespecified: the config that won the fixed-config control.
C_FIXED = 0.03
DIM_FIXED = 64
N_BOOT = 2000
SEED = 20260917

# How to turn image predictions back into one case prediction.
# Prespecified before seeing results: "mean" (average probability).
AGGS = ["mean", "max", "topk2_mean", "vote"]
PRESPECIFIED_AGG = "mean"


def load_labels(oof_dir):
    out = {}
    for fn in sorted(os.listdir(oof_dir)):
        if not fn.endswith("__binary_oof.csv"):
            continue
        field = fn.split("__")[0]
        d = pd.read_csv(os.path.join(oof_dir, fn)).drop_duplicates("case_id")
        out[field] = d.set_index("case_id").truth.astype(float)
    return out


def load_features(feat_dir, pooling):
    idx = pd.read_csv(os.path.join(feat_dir, "index.csv"))
    X = np.load(os.path.join(feat_dir, "features_%s.npy" % pooling))
    assert len(idx) == X.shape[0], (len(idx), X.shape)
    return idx, X


def agg_case(prob, case_ids, how):
    """Aggregate image probabilities to one value per case."""
    s = pd.Series(prob, index=case_ids)
    g = s.groupby(level=0)
    if how == "mean":
        return g.mean()
    if how == "max":
        return g.max()
    if how == "topk2_mean":
        return g.apply(lambda v: np.sort(v.values)[-2:].mean())
    if how == "vote":
        return g.apply(lambda v: float((v > 0.5).mean()))
    raise ValueError(how)


def run_field(idx, X, y_case, folds, aggs):
    """Train on image rows, predict per image, aggregate to case, one OOF pass.

    Returns {agg: Series of case-level predicted label}.
    """
    cases = y_case.index.to_numpy()
    img_case = idx.exam_case_id.to_numpy()
    keep = np.isin(img_case, cases)
    Xi, ci = X[keep], img_case[keep]
    yi = y_case.reindex(ci).to_numpy()          # case label broadcast to images
    fi = folds.reindex(ci).to_numpy()           # case fold broadcast to images

    prob = {a: pd.Series(np.nan, index=cases, dtype=float) for a in aggs}
    for te in sorted(set(folds.dropna())):
        tr = fi != te
        ts = fi == te
        if tr.sum() == 0 or ts.sum() == 0:
            continue
        if len(set(yi[tr])) < 2:
            # degenerate training split: fall back to the train majority
            p = np.full(ts.sum(), float(yi[tr].mean()))
        else:
            dim = min(DIM_FIXED, Xi[tr].shape[0], Xi[tr].shape[1])
            pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=dim, random_state=SEED),
                LogisticRegression(C=C_FIXED, max_iter=4000),
            )
            pipe.fit(Xi[tr], yi[tr])
            p = pipe.predict_proba(Xi[ts])[:, 1]
        for a in aggs:
            s = agg_case(p, ci[ts], a)
            prob[a].loc[s.index] = s.to_numpy()
    return {a: v for a, v in prob.items()}


def score(y, pred, rng):
    """delta = accuracy - single fixed majority baseline, both bootstrapped."""
    y = y.to_numpy(dtype=float)
    pred = pred.to_numpy(dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(pred)
    y, pred = y[ok], pred[ok]
    n = len(y)
    acc = float((pred == y).mean())
    base = float(max(y.mean(), 1 - y.mean()))
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ys, ps = y[s], pred[s]
        d[b] = (ps == ys).mean() - max(ys.mean(), 1 - ys.mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": acc, "baseline_constant": base,
            "delta_vs_constant": acc - base,
            "delta_ci95": [float(lo), float(hi)],
            "prevalence": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True,
                    help="e.g. artifacts/features_spatial/native")
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-minority", type=float, default=0.08,
                    help="fields below this minority share are not modelled")
    a = ap.parse_args()

    roles = pd.read_csv(a.roles)
    folds = roles.set_index("exam_case_id").development_fold
    locked = set(folds[folds.isna()].index)
    folds = folds.dropna()
    assert len(folds) == 186, len(folds)

    labels = load_labels(a.oof_dir)
    rng = np.random.default_rng(SEED)
    out = {"config": {"C": C_FIXED, "pca_dim": DIM_FIXED, "n_boot": N_BOOT,
                      "seed": SEED, "prespecified_agg": PRESPECIFIED_AGG,
                      "feat_dir": a.feat_dir, "selection": "none",
                      "class_weight": None, "threshold": 0.5},
           "locked_cases_seen": 0, "fields": {}}

    for pooling in POOLINGS:
        idx, X = load_features(a.feat_dir, pooling)
        assert not (set(idx.exam_case_id) & locked), "locked case in features"
        for field, y_all in labels.items():
            y = y_all.reindex(folds.index).dropna()
            minority = float(min(y.mean(), 1 - y.mean()))
            if minority <= a.min_minority:
                continue  # delivered as a fixed majority answer by design
            preds = run_field(idx, X, y, folds, AGGS)
            for ag, pr in preds.items():
                lab = (pr > 0.5).astype(float)
                lab[pr.isna()] = np.nan
                key = "%s:%s:%s" % (field, pooling, ag)
                r = score(y, lab.reindex(y.index), rng)
                r.update({"field": field, "pooling": pooling, "agg": ag,
                          "minority": minority,
                          "prespecified": ag == PRESPECIFIED_AGG})
                out["fields"][key] = r
                print("%-46s d=%+.3f CI[%+.3f,%+.3f] n=%d%s"
                      % (key, r["delta_vs_constant"], r["delta_ci95"][0],
                         r["delta_ci95"][1], r["n"],
                         "  <-prespecified" if r["prespecified"] else ""),
                      flush=True)

    out["limitations"] = [
        "development set only (186 cases); not product capability",
        "case labels broadcast to images: an image may not show the lesion the "
        "case label records, so image-level labels are noisy by construction",
        "frozen DINOv2 features, no encoder training",
        "image counts per case vary 4-23, so cases contribute unequally to the fit",
        "aggregation choice reported for all 4 variants; only the prespecified "
        "one ('mean') may be read as a result, the rest are post-hoc",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()


