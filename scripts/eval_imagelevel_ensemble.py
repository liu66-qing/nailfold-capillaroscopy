#!/usr/bin/env python
"""Image-level training combined with fixed pooling ensemble.

Two independent mechanisms were measured separately and both help:

  1. Image-level training (imagelevel_control.py). Fitting on 1708 image rows
     instead of 186 case-mean rows beat the case-level arm in the same code
     path on most poolings (capillary_count 5/5, rbc_aggregation 5/5,
     microthrombus 3/5), with a label-shuffle null of -0.005..-0.055, so the
     gain is not case-identity leakage. But only 2/5 poolings crossed the bar
     on the borderline fields, so reading the best pooling would be the
     selection bias already refuted.

  2. Fixed equal-weight ensembling (eval_ensemble_fields.py). Averaging the 5
     poolings scored above their own mean on every field, which is the correct
     response to a variance problem -- unlike selection, which lost to a single
     fixed config on all 10 fields.

This script applies both at once and chooses nothing: train on image rows for
each of the 5 prespecified poolings, average the 5 predicted probabilities with
equal weights, aggregate to the case by mean probability, threshold at 0.5.
Ensemble membership, weights, aggregation and estimator are all fixed before
any result is seen, and are identical for every field.

Reported alongside, as controls rather than as candidate deliverables:
  - the case-level ensemble (same members, case-mean rows) to attribute the
    gain to image-level training;
  - each member alone, so a gain cannot be a single lucky pooling;
  - a label-shuffle null over the full stacked pipeline.

Ruler unchanged: accuracy minus a single fixed majority answer, case-level
folds from development_fold, baseline recomputed inside every one of 2000
case-level bootstrap resamples, verdict by CI lower bound > 0. No class_weight,
no threshold tuning, no inner validation fold.
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

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
N_SHUFFLE = 20


def load_labels(oof_dir):
    out = {}
    for fn in sorted(os.listdir(oof_dir)):
        if fn.endswith("__binary_oof.csv"):
            d = pd.read_csv(os.path.join(oof_dir, fn)).drop_duplicates("case_id")
            out[fn.split("__")[0]] = d.set_index("case_id").truth.astype(float)
    return out


def fit_predict(Xtr, ytr, Xts):
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(np.mean(ytr)))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def oof_prob(X, img_case, y, folds, level):
    """Out-of-fold case probability for one pooling. level: 'image' or 'case'."""
    cases = y.index.to_numpy()
    if level == "case":
        df = pd.DataFrame(X, index=img_case).groupby(level=0).mean()
        Xa = np.nan_to_num(df.reindex(cases).to_numpy(dtype=float))
        rows_case, ya = cases, y.to_numpy(dtype=float)
    else:
        keep = np.isin(img_case, cases)
        Xa, rows_case = X[keep], img_case[keep]
        ya = y.reindex(rows_case).to_numpy(dtype=float)
    fa = folds.reindex(rows_case).to_numpy(dtype=float)
    p = pd.Series(np.nan, index=cases, dtype=float)
    for te in sorted(set(fa[~np.isnan(fa)])):
        tr, ts = fa != te, fa == te
        if tr.sum() == 0 or ts.sum() == 0:
            continue
        pr = fit_predict(Xa[tr], ya[tr], Xa[ts])
        s = pd.Series(pr, index=rows_case[ts]).groupby(level=0).mean()
        p.loc[s.index] = s.to_numpy()
    return p


def to_label(prob):
    lab = (prob > 0.5).astype(float)
    lab[prob.isna()] = np.nan
    return lab


def score(y, pred, rng):
    ya = y.to_numpy(dtype=float)
    pa = pred.reindex(y.index).to_numpy(dtype=float)
    ok = ~np.isnan(ya) & ~np.isnan(pa)
    ya, pa = ya[ok], pa[ok]
    n = len(ya)
    acc = float((pa == ya).mean())
    base = float(max(ya.mean(), 1 - ya.mean()))
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ys, ps = ya[s], pa[s]
        d[b] = (ps == ys).mean() - max(ys.mean(), 1 - ys.mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": acc, "baseline_constant": base,
            "delta_vs_constant": acc - base,
            "delta_ci95": [float(lo), float(hi)], "passes": bool(lo > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-minority", type=float, default=0.08)
    a = ap.parse_args()

    roles = pd.read_csv(a.roles)
    folds = roles.set_index("exam_case_id").development_fold
    locked = set(folds[folds.isna()].index)
    folds = folds.dropna()
    assert len(folds) == 186, len(folds)

    idx = pd.read_csv(os.path.join(a.feat_dir, "index.csv"))
    img_case = idx.exam_case_id.to_numpy()
    assert not (set(img_case) & locked), "locked case in features"
    feats = {p: np.load(os.path.join(a.feat_dir, "features_%s.npy" % p))
             for p in POOLINGS}

    labels = load_labels(a.oof_dir)
    rng = np.random.default_rng(SEED)
    out = {"config": {"C": C_FIXED, "pca_dim": DIM_FIXED, "n_boot": N_BOOT,
                      "seed": SEED, "members": POOLINGS, "weights": "equal",
                      "level": "image", "agg": "mean probability",
                      "selection": "none", "threshold": 0.5,
                      "class_weight": None, "feat_dir": a.feat_dir},
           "locked_cases_seen": 0, "fields": {}}

    for field, y_all in sorted(labels.items()):
        y = y_all.reindex(folds.index).dropna()
        minority = float(min(y.mean(), 1 - y.mean()))
        if minority <= a.min_minority:
            continue
        pi = {p: oof_prob(feats[p], img_case, y, folds, "image") for p in POOLINGS}
        pc = {p: oof_prob(feats[p], img_case, y, folds, "case") for p in POOLINGS}
        ens_img = score(y, to_label(pd.DataFrame(pi).mean(axis=1)), rng)
        ens_case = score(y, to_label(pd.DataFrame(pc).mean(axis=1)), rng)
        members = {p: score(y, to_label(pi[p]), rng) for p in POOLINGS}

        sh, srng = [], np.random.default_rng(SEED + 1)
        for _ in range(N_SHUFFLE):
            yp = pd.Series(srng.permutation(y.to_numpy()), index=y.index)
            ps = {p: oof_prob(feats[p], img_case, yp, folds, "image")
                  for p in POOLINGS}
            lab = to_label(pd.DataFrame(ps).mean(axis=1))
            ok = ~lab.isna()
            sh.append(float((lab[ok] == yp[ok]).mean())
                      - float(max(yp.mean(), 1 - yp.mean())))

        out["fields"][field] = {
            "minority": minority,
            "ensemble_image_level": ens_img,
            "ensemble_case_level": ens_case,
            "gain_from_image_level": (ens_img["delta_vs_constant"]
                                      - ens_case["delta_vs_constant"]),
            "members_image_level": members,
            "n_members_passing_alone": sum(v["passes"] for v in members.values()),
            "shuffled_null_mean": float(np.mean(sh)),
            "shuffled_null_p95": float(np.percentile(sh, 95)),
        }
        print("%-28s ens-img %+.3f CI[%+.3f,%+.3f] %-4s | ens-case %+.3f | "
              "gain %+.3f | members alone %d/5 | null %+.3f (p95 %+.3f)"
              % (field, ens_img["delta_vs_constant"], ens_img["delta_ci95"][0],
                 ens_img["delta_ci95"][1], "PASS" if ens_img["passes"] else "",
                 ens_case["delta_vs_constant"],
                 out["fields"][field]["gain_from_image_level"],
                 out["fields"][field]["n_members_passing_alone"],
                 out["fields"][field]["shuffled_null_mean"],
                 out["fields"][field]["shuffled_null_p95"]), flush=True)

    out["limitations"] = [
        "development set only (186 cases); not product capability",
        "case labels broadcast to images are noisy by construction: an image "
        "need not show the lesion its case label records",
        "the 5 members share one frozen encoder, so their errors are correlated "
        "and averaging gains less than for independent models",
        "images per case vary 4-23, so cases contribute unequally to the fit",
        "shuffled null uses 20 permutations: enough to detect gross leakage, "
        "not enough for a tight null interval",
        "equal weights are prespecified, not fitted; fitting them would "
        "reintroduce the selection variance this design exists to avoid",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

