#!/usr/bin/env python
"""Fixed ensemble averaging instead of config selection.

The diagnosis from the fixed-config control was specific: per-fold selection
lost to a single fixed config on all 10 modelled fields because a ~37-case
inner validation fold cannot rank candidates stably. That is a variance
problem. Selection is the wrong response to variance -- averaging is.

So this script does not choose. It fits a prespecified, fixed set of views
(the 5 DINOv2 poolings, plus geometry and roi_quality when present), averages
their predicted probabilities with equal weights, and thresholds at 0.5. No
inner fold, no ranking, no tuning, nothing to overfit: the ensemble membership
is decided before any result is seen and is identical for every field.

Two controls make this interpretable:
  1. every member is scored alone under the same protocol, so a gain can be
     attributed to averaging rather than to one lucky member;
  2. `best_single_posthoc` records the best member for reference and is
     explicitly NOT a deliverable number (picking it is the bias we removed).

Same ruler as everything else: accuracy minus a single fixed majority answer,
case-level folds from development_fold, baseline recomputed inside each of
2000 case-level bootstrap resamples, verdict by CI lower bound > 0.
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

C_FIXED = 0.03
DIM_FIXED = 64
N_BOOT = 2000
SEED = 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]


def load_labels(oof_dir):
    out = {}
    for fn in sorted(os.listdir(oof_dir)):
        if fn.endswith("__binary_oof.csv"):
            d = pd.read_csv(os.path.join(oof_dir, fn)).drop_duplicates("case_id")
            out[fn.split("__")[0]] = d.set_index("case_id").truth.astype(float)
    return out


def case_matrix(feat_dir, pooling, cases):
    """Case-level matrix: mean over the images of each case."""
    idx = pd.read_csv(os.path.join(feat_dir, "index.csv"))
    X = np.load(os.path.join(feat_dir, "features_%s.npy" % pooling))
    df = pd.DataFrame(X, index=idx.exam_case_id.to_numpy())
    df = df.groupby(level=0).mean()
    return df.reindex(cases)


def extra_matrix(path, cases):
    """Optional tabular feature file keyed by exam_case_id."""
    if not path or not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    key = "exam_case_id" if "exam_case_id" in d.columns else d.columns[0]
    d = d.drop_duplicates(key).set_index(key)
    d = d.select_dtypes(include=[np.number])
    if d.shape[1] == 0:
        return None
    return d.groupby(level=0).mean().reindex(cases)


def oof_prob(M, y, folds):
    """One out-of-fold probability per case for a single view. No selection."""
    cases = y.index
    p = pd.Series(np.nan, index=cases, dtype=float)
    Xa = M.reindex(cases).to_numpy(dtype=float)
    Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    ya = y.to_numpy(dtype=float)
    fa = folds.reindex(cases).to_numpy(dtype=float)
    for te in sorted(set(fa[~np.isnan(fa)])):
        tr, ts = fa != te, fa == te
        if tr.sum() == 0 or ts.sum() == 0 or len(set(ya[tr])) < 2:
            p.iloc[np.where(ts)[0]] = float(ya[tr].mean()) if tr.sum() else 0.5
            continue
        dim = min(DIM_FIXED, int(tr.sum()) - 1, Xa.shape[1])
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=max(dim, 2), random_state=SEED),
            LogisticRegression(C=C_FIXED, max_iter=4000),
        )
        pipe.fit(Xa[tr], ya[tr])
        p.iloc[np.where(ts)[0]] = pipe.predict_proba(Xa[ts])[:, 1]
    return p


def score(y, pred, rng):
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
            "delta_ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--geom", default="")
    ap.add_argument("--roi", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-minority", type=float, default=0.08)
    a = ap.parse_args()

    roles = pd.read_csv(a.roles)
    folds = roles.set_index("exam_case_id").development_fold
    locked = set(folds[folds.isna()].index)
    folds = folds.dropna()
    assert len(folds) == 186, len(folds)
    cases = folds.index

    views = {}
    for p in POOLINGS:
        views["dinov2_" + p] = case_matrix(a.feat_dir, p, cases)
    for name, path in (("geometry", a.geom), ("roi_quality", a.roi)):
        m = extra_matrix(path, cases)
        if m is not None:
            views[name] = m
            print("added tabular view %s: %d cols" % (name, m.shape[1]))
    for name, m in views.items():
        assert not (set(m.index.dropna()) & locked), "locked case in " + name
    print("ensemble members (fixed, prespecified): %s\n" % ", ".join(sorted(views)))

    labels = load_labels(a.oof_dir)
    rng = np.random.default_rng(SEED)
    out = {"config": {"C": C_FIXED, "pca_dim": DIM_FIXED, "n_boot": N_BOOT,
                      "seed": SEED, "members": sorted(views),
                      "weights": "equal", "selection": "none",
                      "threshold": 0.5, "class_weight": None},
           "locked_cases_seen": 0, "fields": {}}

    for field, y_all in sorted(labels.items()):
        y = y_all.reindex(cases).dropna()
        minority = float(min(y.mean(), 1 - y.mean()))
        if minority <= a.min_minority:
            continue
        probs, members = {}, {}
        for name, M in views.items():
            pr = oof_prob(M, y, folds)
            probs[name] = pr
            members[name] = score(y, (pr > 0.5).astype(float), rng)
        avg = pd.DataFrame(probs).mean(axis=1)
        ens = score(y, (avg > 0.5).astype(float), rng)
        best = max(members.items(), key=lambda kv: kv[1]["delta_vs_constant"])
        out["fields"][field] = {
            "minority": minority, "ensemble": ens, "members": members,
            "best_single_posthoc": {"name": best[0], **best[1]},
            "note": ("best_single_posthoc is NOT a deliverable number: picking "
                     "the best member after seeing all members is exactly the "
                     "selection bias removed earlier. Only 'ensemble' is."),
        }
        print("%-28s ens %+.3f CI[%+.3f,%+.3f] | mean-member %+.3f | "
              "best-single(post-hoc) %+.3f (%s)"
              % (field, ens["delta_vs_constant"], ens["delta_ci95"][0],
                 ens["delta_ci95"][1],
                 float(np.mean([v["delta_vs_constant"] for v in members.values()])),
                 best[1]["delta_vs_constant"], best[0]), flush=True)

    out["limitations"] = [
        "development set only (186 cases); not product capability",
        "equal weights are prespecified, not fitted; learning weights would "
        "reintroduce the selection variance this design avoids",
        "members share one frozen encoder, so their errors are correlated and "
        "averaging gains less than it would for independent models",
        "geometry/roi views come from label-derived pipelines; no micron-accuracy "
        "claim is made and calibration_factors.json is not used as evidence",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()


