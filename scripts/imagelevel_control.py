#!/usr/bin/env python
"""Controls for the image-level result before it is allowed to be a result.

eval_imagelevel_fields.py reported 5 poolings x 4 aggregations per field. Two
fields (malformation_ratio, microthrombus) cross the bar in some cells. Reading
the best cell would be exactly the selection bias refuted in the fixed-config
control, so nothing here is believed until three controls pass.

Control A -- pooling robustness. A real image-level effect should show up in
most poolings, not one. Reports how many of the 5 poolings have CI lower bound
> 0 for the prespecified 'mean' aggregation. A field passing in 1/5 is a
pooling pick; a field passing in >=4/5 is a property of the data.

Control B -- case-level rerun, same code path. The only difference from the
image-level arm is whether rows are images or case-means. Same estimator, same
folds, same bootstrap stream. This isolates "more training rows" from every
other change and is the actual attribution claim.

Control C -- label-shuffle null. Case labels are permuted within the fold
structure and the whole image-level pipeline is rerun. Broadcasting one case
label to 9.2 correlated images could inflate apparent performance through
leakage of case identity rather than lesion signal; if the shuffled arm scores
above 0, the protocol is broken and the gain is an artifact. Expected: ~0.

Nothing here tunes anything. Estimator is fixed (logreg C=0.03, PCA 64,
unweighted, threshold 0.5), aggregation is fixed to the prespecified 'mean'.
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


def oof_image_level(X, img_case, y_case, folds):
    """Train on image rows; average predicted probability per case."""
    cases = y_case.index.to_numpy()
    keep = np.isin(img_case, cases)
    Xi, ci = X[keep], img_case[keep]
    yi = y_case.reindex(ci).to_numpy(dtype=float)
    fi = folds.reindex(ci).to_numpy(dtype=float)
    p = pd.Series(np.nan, index=cases, dtype=float)
    for te in sorted(set(fi[~np.isnan(fi)])):
        tr, ts = fi != te, fi == te
        if tr.sum() == 0 or ts.sum() == 0:
            continue
        pr = fit_predict(Xi[tr], yi[tr], Xi[ts])
        s = pd.Series(pr, index=ci[ts]).groupby(level=0).mean()
        p.loc[s.index] = s.to_numpy()
    return p


def oof_case_level(X, img_case, y_case, folds):
    """Identical code path, but each case is one averaged row."""
    df = pd.DataFrame(X, index=img_case).groupby(level=0).mean()
    cases = y_case.index
    Xa = np.nan_to_num(df.reindex(cases).to_numpy(dtype=float))
    ya = y_case.to_numpy(dtype=float)
    fa = folds.reindex(cases).to_numpy(dtype=float)
    p = pd.Series(np.nan, index=cases, dtype=float)
    for te in sorted(set(fa[~np.isnan(fa)])):
        tr, ts = fa != te, fa == te
        if tr.sum() == 0 or ts.sum() == 0:
            continue
        p.iloc[np.where(ts)[0]] = fit_predict(Xa[tr], ya[tr], Xa[ts])
    return p


def to_label(prob):
    """Threshold 0.5, preserving NaN (a case with no prediction stays missing)."""
    lab = (prob > 0.5).astype(float)
    lab[prob.isna()] = np.nan
    return lab


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
            "delta_ci95": [float(lo), float(hi)],
            "passes": bool(lo > 0)}


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
                      "seed": SEED, "agg": "mean", "n_shuffle": N_SHUFFLE,
                      "selection": "none", "class_weight": None},
           "locked_cases_seen": 0, "fields": {}}

    for field, y_all in sorted(labels.items()):
        y = y_all.reindex(folds.index).dropna()
        minority = float(min(y.mean(), 1 - y.mean()))
        if minority <= a.min_minority:
            continue
        rec = {"minority": minority, "per_pooling": {}}
        for p in POOLINGS:
            X = feats[p]
            img = score(y, to_label(oof_image_level(X, img_case, y, folds)), rng)
            cas = score(y, to_label(oof_case_level(X, img_case, y, folds)), rng)
            rec["per_pooling"][p] = {"image_level": img, "case_level": cas,
                                     "gain_from_image_level":
                                     img["delta_vs_constant"] - cas["delta_vs_constant"]}
        n_pass = sum(v["image_level"]["passes"] for v in rec["per_pooling"].values())
        gains = [v["gain_from_image_level"] for v in rec["per_pooling"].values()]
        rec["control_A_poolings_passing"] = "%d/5" % n_pass
        rec["control_B_median_gain_over_case_level"] = float(np.median(gains))
        rec["control_B_poolings_where_image_wins"] = int(sum(g > 0 for g in gains))

        # Control C: permute case labels, keep the fold structure.
        X = feats["mean"]
        sh = []
        srng = np.random.default_rng(SEED + 1)
        for _ in range(N_SHUFFLE):
            yp = pd.Series(srng.permutation(y.to_numpy()), index=y.index)
            pr = oof_image_level(X, img_case, yp, folds)
            lab = (pr > 0.5).astype(float)
            ok = ~pr.isna()
            acc = float((lab[ok] == yp[ok]).mean())
            sh.append(acc - float(max(yp.mean(), 1 - yp.mean())))
        rec["control_C_shuffled_delta_mean"] = float(np.mean(sh))
        rec["control_C_shuffled_delta_p95"] = float(np.percentile(sh, 95))

        verdict = ("ROBUST" if n_pass >= 4 and rec["control_C_shuffled_delta_p95"] < 0.03
                   else "POOLING_DEPENDENT" if n_pass >= 1 else "FAILS")
        rec["verdict"] = verdict
        out["fields"][field] = rec
        print("%-28s A:%s  B:median %+.3f (%d/5 image wins)  C:shuffled %+.3f "
              "(p95 %+.3f)  -> %s"
              % (field, rec["control_A_poolings_passing"],
                 rec["control_B_median_gain_over_case_level"],
                 rec["control_B_poolings_where_image_wins"],
                 rec["control_C_shuffled_delta_mean"],
                 rec["control_C_shuffled_delta_p95"], verdict), flush=True)

    out["limitations"] = [
        "development set only (186 cases); not product capability",
        "case labels broadcast to images are noisy by construction: an image "
        "need not show the lesion its case label records",
        "control C uses 20 permutations, enough to detect gross leakage but not "
        "to place a tight null CI",
        "images per case vary 4-23, so cases contribute unequally to the fit",
        "frozen DINOv2 features; no encoder training and no micron-scale claim",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

